import os
import re

MEM0_GRAPH_DEBUG = os.getenv("MEM0_GRAPH_DEBUG", "0").lower() in {"1", "true", "yes", "on"}

def _graph_debug_print(*args, **kwargs):
    if MEM0_GRAPH_DEBUG:
        print(*args, **kwargs)

import logging

from mem0.memory.utils import format_entities

try:
    from langchain_neo4j import Neo4jGraph
    # import Neo4jGraph
except ImportError:
    raise ImportError("langchain_neo4j is not installed. Please install it using pip install langchain-neo4j")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is not installed. Please install it using pip install rank-bm25")

from mem0.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)
from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT, get_delete_messages
from mem0.utils.factory import EmbedderFactory, LlmFactory
from sentence_transformers import SentenceTransformer
logger = logging.getLogger(__name__)

def _normalize_graph_name(value, fallback="null"):
    text = str(value or "").strip().lower()
    text = text.replace(" ", "_").replace(":", "_").replace(",", "_")
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback

def _sanitize_relationship_type(value, fallback="related_to"):
    text = _normalize_graph_name(value, fallback=fallback)
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"rel_{text}"
    return text

def _sanitize_entity_type(value):
    return _sanitize_relationship_type(value, fallback="entity")

def _coerce_entity_item(item):
    if not isinstance(item, dict):
        return None
    entity = item.get("entity") or item.get("name") or item.get("element")
    entity_type = item.get("entity_type") or item.get("type") or "entity"
    if not entity:
        return None
    return entity, entity_type

class LocalEmbedder:
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.embedding_dims = 768
        self.config = type("Config", (), {"embedding_dims": 768})()
    def embed(self, texts, *args, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        if texts is None:
            return []
        return self.model.encode(texts)

class MemoryGraph:
    def __init__(self, config):
        self.config = config
        self.graph = Neo4jGraph(
            self.config.graph_store.config.url,
            self.config.graph_store.config.username,
            self.config.graph_store.config.password,
            self.config.graph_store.config.database,
            refresh_schema=False,
            driver_config={"notifications_min_severity": "OFF"},
        )
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider, self.config.embedder.config, self.config.vector_store.config
        )
        self.embedding_model = LocalEmbedder()
        self.node_label = ":`__Entity__`" if self.config.graph_store.config.base_label else ""

        if self.config.graph_store.config.base_label:
            # Safely add user_id index
            try:
                self.graph.query(f"CREATE INDEX entity_single IF NOT EXISTS FOR (n {self.node_label}) ON (n.user_id)")
            except Exception:
                pass
            try:  # Safely try to add composite index (Enterprise only)
                self.graph.query(
                    f"CREATE INDEX entity_composite IF NOT EXISTS FOR (n {self.node_label}) ON (n.name, n.user_id)"
                )
            except Exception:
                pass

        self.llm_provider = "openai_structured"
        if self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store.llm:
            self.llm_provider = self.config.graph_store.llm.provider

        self.llm = LlmFactory.create(self.llm_provider, self.config.llm.config)
        self.user_id = None
        self.threshold = 0.7

    def reset(self):
        """Clear all nodes and relationships in the Neo4j graph."""
        self.graph.query("MATCH (n) DETACH DELETE n")

    def clear(self):
        """Clear all nodes and relationships in the graph store."""
        self.driver.execute_query("MATCH (n) DETACH DELETE n")

    def add(self, data, filters):
        """
        Adds data to the graph.

        Args:
            data (str): The data to add to the graph.
            filters (dict): A dictionary containing filters to be applied during the addition.
        """
        _graph_debug_print("memorygraph: here", flush=True) 
        entity_type_map = self._retrieve_nodes_from_data(data, filters)
        to_be_added = self._establish_nodes_relations_from_data(data, filters, entity_type_map)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)
        to_be_deleted = self._get_delete_entities_from_search_output(search_output, data, filters)

        # TODO: Batch queries with APOC plugin
        # TODO: Add more filter support
        deleted_entities = self._delete_entities(to_be_deleted, filters)
        added_entities = self._add_entities(to_be_added, filters, entity_type_map)

        return {"deleted_entities": deleted_entities, "added_entities": added_entities}




    def search(self, query, filters, limit=100):
        """
        Search for memories and related graph data.

        Args:
            query (str): Query to search for.
            filters (dict): A dictionary containing filters to be applied during the search.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.

        Returns:
            dict: A dictionary containing:
                - "contexts": List of search results from the base data store.
                - "entities": List of related graph data based on the query.
        """
        entity_type_map = self._retrieve_nodes_from_data(query, filters)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)

        if not search_output:
            return []

        search_outputs_sequence = [
            [item["source"], item["relationship"], item["destination"]] for item in search_output
        ]
        bm25 = BM25Okapi(search_outputs_sequence)

        tokenized_query = query.split(" ")
        reranked_results = bm25.get_top_n(tokenized_query, search_outputs_sequence, n=5)

        search_results = []
        for item in reranked_results:
            search_results.append({"source": item[0], "relationship": item[1], "destination": item[2]})

        logger.info(f"Returned {len(search_results)} search results")

        return search_results

    def delete_all(self, filters):
        if filters.get("agent_id"):
            cypher = f"""
            MATCH (n {self.node_label} {{user_id: $user_id, agent_id: $agent_id}})
            DETACH DELETE n
            """
            params = {"user_id": filters["user_id"], "agent_id": filters["agent_id"]}
        else:
            cypher = f"""
            MATCH (n {self.node_label} {{user_id: $user_id}})
            DETACH DELETE n
            """
            params = {"user_id": filters["user_id"]}
        self.graph.query(cypher, params=params)


    def get_all(self, filters, limit=100):
        """
        Retrieves all nodes and relationships from the graph database based on optional filtering criteria.
         Args:
            filters (dict): A dictionary containing filters to be applied during the retrieval.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.
        Returns:
            list: A list of dictionaries, each containing:
                - 'contexts': The base data store response for each memory.
                - 'entities': A list of strings representing the nodes and relationships
        """
        agent_filter = ""
        params = {"user_id": filters["user_id"], "limit": limit}
        if filters.get("agent_id"):
            agent_filter = "AND n.agent_id = $agent_id AND m.agent_id = $agent_id"
            params["agent_id"] = filters["agent_id"]

        query = f"""
        MATCH (n {self.node_label} {{user_id: $user_id}})-[r]->(m {self.node_label} {{user_id: $user_id}})
        WHERE 1=1 {agent_filter}
        RETURN n.name AS source, type(r) AS relationship, m.name AS target
        LIMIT $limit
        """
        results = self.graph.query(query, params=params)

        final_results = []
        for result in results:
            final_results.append(
                {
                    "source": result["source"],
                    "relationship": result["relationship"],
                    "target": result["target"],
                }
            )

        logger.info(f"Retrieved {len(final_results)} relationships")

        return final_results


    def _retrieve_nodes_from_data(self, data, filters):
        _graph_debug_print("_retrieve_nodes_from_data data: ", data, flush=True)
        """Extracts all the entities mentioned in the query."""
        _tools = [EXTRACT_ENTITIES_TOOL]
        # if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            # _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        # _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a smart assistant who understands entities and their types in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use {filters['user_id']} as the source entity. Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question.",
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
        )
        _graph_debug_print("search_results nodes_from_data", search_results, flush=True)
        entity_type_map = {}

        try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "extract_entities":
                    continue
                for item in tool_call["arguments"]["entities"]:
                    coerced = _coerce_entity_item(item)
                    if coerced is None:
                        logger.warning(f"Skipping malformed entity item: {item}")
                        continue
                    entity, entity_type = coerced
                    entity_type_map[entity] = entity_type
        except Exception as e:
            logger.exception(
                f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            )

        entity_type_map = {
            _normalize_graph_name(k): _sanitize_entity_type(v)
            for k, v in entity_type_map.items()
        }
        _graph_debug_print("entity_type_map: ", entity_type_map)
        logger.debug(f"Entity type map: {entity_type_map}\n search_results={search_results}")
        return entity_type_map

    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        """Establish relations among the extracted nodes."""

    # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"

        if self.config.graph_store.custom_prompt:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            # Add the custom prompt line if configured
            system_content = system_content.replace(
                "CUSTOM_PROMPT", f"4. {self.config.graph_store.custom_prompt}"
            )
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": data},
            ]
        else:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"List of entities: {list(entity_type_map.keys())}. \n\nText: {data}"},
            ]

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [RELATIONS_STRUCT_TOOL]

        # extracted_entities = self.llm.generate_response(
        #     messages=messages,
        #     tools=_tools,
        # )

        # entities = []
        # if extracted_entities.get("tool_calls"):
        #     entities = extracted_entities["tool_calls"][0].get("arguments", {}).get("entities", [])

        # entities = self._remove_spaces_from_entities(entities)
        # logger.debug(f"Extracted entities: {entities}")
        # return entities
 # 确保函数内或文件开头有这个导入
        import json
        extracted_entities = self.llm.generate_response(
            messages=messages,
            tools=_tools,
        )

        entities = []
        if extracted_entities.get("tool_calls"):
            # 先拿到 arguments 字典
            args = extracted_entities["tool_calls"][0].get("arguments", {})
            entities = args.get("entities", [])
            
            # 【关键修复】：如果 entities 是字符串，手动转回列表
            if isinstance(entities, str):
                try:
                    entities = json.loads(entities)
                    logger.info("检测到字符串格式的 entities，已成功解析为列表")
                except Exception as e:
                    logger.error(f"解析 entities 字符串失败: {e}")
                    entities = []

        # 增加一层保护，确保是列表才处理
        if isinstance(entities, list):
            entities = self._remove_spaces_from_entities(entities)
        else:
            entities = []
            
        logger.debug(f"Extracted entities: {entities}")
        return entities

    def _search_graph_db(self, node_list, filters, limit=100):
        """Search similar nodes among and their respective incoming and outgoing relations."""
        result_relations = []
        agent_filter = ""
        if filters.get("agent_id"):
            agent_filter = "AND n.agent_id = $agent_id AND m.agent_id = $agent_id"

        for node in node_list:
            n_embedding = self.embedding_model.embed(node)
            # 修复向量格式 - 添加这部分代码
            import math
            import numpy as np
            
            # 处理n_embedding
            if isinstance(n_embedding, np.ndarray):
                n_embedding = n_embedding.tolist()
            if isinstance(n_embedding, list) and len(n_embedding) > 0:
                if isinstance(n_embedding[0], list):
                    n_embedding = n_embedding[0]  # 展开嵌套列表
            # 确保所有值都是有限数值，使用numpy方法处理
            n_embedding = [float(x) for x in n_embedding if not (np.isnan(x) or np.isinf(x))]
            
            # 检查向量是否为空
            if not n_embedding or len(n_embedding) == 0:
                logger.warning(f"Empty embedding for node: {node}")
                continue
            cypher_query = f"""
            MATCH (n {self.node_label})
            WHERE n.embedding IS NOT NULL AND n.user_id = $user_id
            {agent_filter}
            WITH n, round(2 * vector.similarity.cosine(n.embedding, $n_embedding) - 1, 4) AS similarity // denormalize for backward compatibility
            WHERE similarity >= $threshold
            CALL {{
                MATCH (n)-[r]->(m)
                WHERE m.user_id = $user_id {agent_filter.replace("n.", "m.")} 
                RETURN n.name AS source, elementId(n) AS source_id, type(r) AS relationship, elementId(r) AS relation_id, m.name AS destination, elementId(m) AS destination_id
                UNION
                MATCH (m)-[r]->(n)
                WHERE m.user_id = $user_id {agent_filter.replace("n.", "m.")}
                RETURN m.name AS source, elementId(m) AS source_id, type(r) AS relationship, elementId(r) AS relation_id, n.name AS destination, elementId(n) AS destination_id
            }}
            WITH distinct source, source_id, relationship, relation_id, destination, destination_id, similarity
            RETURN source, source_id, relationship, relation_id, destination, destination_id, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """

            params = {
                "n_embedding": n_embedding,
                "threshold": self.threshold,
                "user_id": filters["user_id"],
                "limit": limit,
            }
            if filters.get("agent_id"):
                params["agent_id"] = filters["agent_id"]

            ans = self.graph.query(cypher_query, params=params)
            result_relations.extend(ans)

        return result_relations

    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        """Get the entities to be deleted from the search output."""
        search_output_string = format_entities(search_output)

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"

        system_prompt, user_prompt = get_delete_messages(search_output_string, data, user_identity)

        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [
                DELETE_MEMORY_STRUCT_TOOL_GRAPH,
            ]

        memory_updates = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
        )

        to_be_deleted = []
        for item in memory_updates.get("tool_calls", []):
            if item.get("name") == "delete_graph_memory":
                to_be_deleted.append(item.get("arguments"))
        # Clean entities formatting
        to_be_deleted = self._remove_spaces_from_entities(to_be_deleted)
        logger.debug(f"Deleted relationships: {to_be_deleted}")
        return to_be_deleted

    def _delete_entities(self, to_be_deleted, filters):
        """Delete the entities from the graph."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        results = []
        
        for item in to_be_deleted:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # Build the agent filter for the query
            agent_filter = ""
            params = {
                "source_name": source,
                "dest_name": destination,
                "user_id": user_id,
            }
            
            if agent_id:
                agent_filter = "AND n.agent_id = $agent_id AND m.agent_id = $agent_id"
                params["agent_id"] = agent_id

            # Delete the specific relationship between nodes
            cypher = f"""
            MATCH (n {self.node_label} {{name: $source_name, user_id: $user_id}})
            -[r:{relationship}]->
            (m {self.node_label} {{name: $dest_name, user_id: $user_id}})
            WHERE 1=1 {agent_filter}
            DELETE r
            RETURN 
                n.name AS source,
                m.name AS target,
                type(r) AS relationship
            """
            
            result = self.graph.query(cypher, params=params)
            results.append(result)
        
        return results

    def _add_entities(self, to_be_added, filters, entity_type_map):
        """Add the new entities to the graph. Merge the nodes if they already exist."""

    
        # user_id = filters["user_id"]
        # agent_id = filters.get("agent_id", None)
        # results = []
        # for item in to_be_added:
        #     # entities
        #     source = item["source"]
        #     destination = item["destination"]
        #     relationship = item["relationship"]

        #     # types
        #     source_type = entity_type_map.get(source, "__User__")
        #     source_label = self.node_label if self.node_label else f":`{source_type}`"
        #     source_extra_set = f", source:`{source_type}`" if self.node_label else ""
        #     destination_type = entity_type_map.get(destination, "__User__")
        #     destination_label = self.node_label if self.node_label else f":`{destination_type}`"
        #     destination_extra_set = f", destination:`{destination_type}`" if self.node_label else ""

        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        results = []

        # 过滤掉无效的关系
        valid_entities = []
        for item in to_be_added:
            source = item.get("source")
            destination = item.get("destination")
            relationship = item.get("relationship")

            # 检查必要字段是否存在且不为None
            if source is None or destination is None or relationship is None:
                logger.warning(f"Skipping invalid entity with None values: {item}")
                continue

            # 检查字段是否为空字符串
            if not source.strip() or not destination.strip() or not relationship.strip():
                logger.warning(f"Skipping entity with empty values: {item}")
                continue

            valid_entities.append(item)

        # 如果没有有效实体，直接返回
        if not valid_entities:
            logger.info("No valid entities to process")
            return {"deleted_entities": [], "added_entities": []}

        # 继续处理有效实体
        for item in valid_entities:
            # entities
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # types
            source_type = entity_type_map.get(source, "__User__")
            source_label = self.node_label if self.node_label else f":`{source_type}`"
            source_extra_set = f", source:`{source_type}`" if self.node_label else ""
            destination_type = entity_type_map.get(destination, "__User__")
            destination_label = self.node_label if self.node_label else f":`{destination_type}`"
            destination_extra_set = f", destination:`{destination_type}`" if self.node_label else ""


            # embeddings
            source_embedding = self.embedding_model.embed(source)
            dest_embedding = self.embedding_model.embed(destination)
            # 修复向量格式 - 添加这部分代码
            import math
            import numpy as np

            # 处理source_embedding
            if isinstance(source_embedding, np.ndarray):
                source_embedding = source_embedding.tolist()
            if isinstance(source_embedding, list) and len(source_embedding) > 0:
                if isinstance(source_embedding[0], list):
                    source_embedding = source_embedding[0]  # 展开嵌套列表
            # 确保所有值都是有限数值，使用numpy方法处理
            source_embedding = [float(x) for x in source_embedding if not (np.isnan(x) or np.isinf(x))]

            # 处理dest_embedding
            if isinstance(dest_embedding, np.ndarray):
                dest_embedding = dest_embedding.tolist()
            if isinstance(dest_embedding, list) and len(dest_embedding) > 0:
                if isinstance(dest_embedding[0], list):
                    dest_embedding = dest_embedding[0]  # 展开嵌套列表
            # 确保所有值都是有限数值，使用numpy方法处理
            dest_embedding = [float(x) for x in dest_embedding if not (np.isnan(x) or np.isinf(x))]

            # search for the nodes with the closest embeddings
            source_node_search_result = self._search_source_node(source_embedding, filters, threshold=0.9)
            destination_node_search_result = self._search_destination_node(dest_embedding, filters, threshold=0.9)

            # TODO: Create a cypher query and common params for all the cases
            if not destination_node_search_result and source_node_search_result:
                # Build destination MERGE properties
                merge_props = ["name: $destination_name", "user_id: $user_id"]
                if agent_id:
                    merge_props.append("agent_id: $agent_id")
                merge_props_str = ", ".join(merge_props)

                cypher = f"""
                MATCH (source)
                WHERE elementId(source) = $source_id
                SET source.mentions = coalesce(source.mentions, 0) + 1
                WITH source
                MERGE (destination {destination_label} {{{merge_props_str}}})
                ON CREATE SET
                    destination.created = timestamp(),
                    destination.mentions = 1
                    {destination_extra_set}
                ON MATCH SET
                    destination.mentions = coalesce(destination.mentions, 0) + 1
                WITH source, destination
                CALL db.create.setNodeVectorProperty(destination, 'embedding', $destination_embedding)
                WITH source, destination
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created = timestamp(),
                    r.mentions = 1
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """
                
                params = {
                    "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                    "destination_name": destination,
                    "destination_embedding": dest_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id

            elif destination_node_search_result and not source_node_search_result:
                # Build source MERGE properties
                merge_props = ["name: $source_name", "user_id: $user_id"]
                if agent_id:
                    merge_props.append("agent_id: $agent_id")
                merge_props_str = ", ".join(merge_props)

                cypher = f"""
                MATCH (destination)
                WHERE elementId(destination) = $destination_id
                SET destination.mentions = coalesce(destination.mentions, 0) + 1
                WITH destination
                MERGE (source {source_label} {{{merge_props_str}}})
                ON CREATE SET
                    source.created = timestamp(),
                    source.mentions = 1
                    {source_extra_set}
                ON MATCH SET
                    source.mentions = coalesce(source.mentions, 0) + 1
                WITH source, destination
                CALL db.create.setNodeVectorProperty(source, 'embedding', $source_embedding)
                WITH source, destination
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created = timestamp(),
                    r.mentions = 1
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "destination_id": destination_node_search_result[0]["elementId(destination_candidate)"],
                    "source_name": source,
                    "source_embedding": source_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id

            elif source_node_search_result and destination_node_search_result:
                cypher = f"""
                MATCH (source)
                WHERE elementId(source) = $source_id
                SET source.mentions = coalesce(source.mentions, 0) + 1
                WITH source
                MATCH (destination)
                WHERE elementId(destination) = $destination_id
                SET destination.mentions = coalesce(destination.mentions, 0) + 1
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created_at = timestamp(),
                    r.updated_at = timestamp(),
                    r.mentions = 1
                ON MATCH SET r.mentions = coalesce(r.mentions, 0) + 1
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                    "destination_id": destination_node_search_result[0]["elementId(destination_candidate)"],
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id

            else:
                # Build dynamic MERGE props for both source and destination
                source_props = ["name: $source_name", "user_id: $user_id"]
                dest_props = ["name: $dest_name", "user_id: $user_id"]
                if agent_id:
                    source_props.append("agent_id: $agent_id")
                    dest_props.append("agent_id: $agent_id")
                source_props_str = ", ".join(source_props)
                dest_props_str = ", ".join(dest_props)

                cypher = f"""
                MERGE (source {source_label} {{{source_props_str}}})
                ON CREATE SET source.created = timestamp(),
                            source.mentions = 1
                            {source_extra_set}
                ON MATCH SET source.mentions = coalesce(source.mentions, 0) + 1
                WITH source
                CALL db.create.setNodeVectorProperty(source, 'embedding', $source_embedding)
                WITH source
                MERGE (destination {destination_label} {{{dest_props_str}}})
                ON CREATE SET destination.created = timestamp(),
                            destination.mentions = 1
                            {destination_extra_set}
                ON MATCH SET destination.mentions = coalesce(destination.mentions, 0) + 1
                WITH source, destination
                CALL db.create.setNodeVectorProperty(destination, 'embedding', $dest_embedding)
                WITH source, destination
                MERGE (source)-[rel:{relationship}]->(destination)
                ON CREATE SET rel.created = timestamp(), rel.mentions = 1
                ON MATCH SET rel.mentions = coalesce(rel.mentions, 0) + 1
                RETURN source.name AS source, type(rel) AS relationship, destination.name AS target
                """

                params = {
                    "source_name": source,
                    "dest_name": destination,
                    "source_embedding": source_embedding,
                    "dest_embedding": dest_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
            result = self.graph.query(cypher, params=params)
            results.append(result)
        return results

    # def _remove_spaces_from_entities(self, entity_list):
    #     for item in entity_list:
    #         # item["source"] = item["source"].lower().replace(" ", "_")
    #         # item["relationship"] = item["relationship"].lower().replace(" ", "_")
    #         # item["destination"] = item["destination"].lower().replace(" ", "_")
    #         if item.get("source") is not None:
    #             item["source"] = item["source"].lower().replace(" ", "_")

    #         if item.get("relationship") is not None:
    #             item["relationship"] = item["relationship"].lower().replace(" ", "_")

    #         if item.get("destination") is not None:
    #             item["destination"] = item["destination"].lower().replace(" ", "_")
    #     return entity_list

    # def _remove_spaces_from_entities(self, entity_list):
    #     for item in entity_list:
    #         # 将 None 转换为 "null" 字符串，然后正常处理
    #         if item.get("source") is not None:
    #             item["source"] = item["source"].lower().replace(" ", "_").replace("-", "_")
    #         else:
    #             item["source"] = "null"

    #         if item.get("relationship") is not None:
    #             item["relationship"] = item["relationship"].lower().replace(" ", "_").replace("-", "_")
    #         else:
    #             item["relationship"] = "null"

    #         if item.get("destination") is not None:
    #             item["destination"] = item["destination"].lower().replace(" ", "_").replace("-", "_")
    #         else:
    #             item["destination"] = "null"

    #     return entity_list
    def _remove_spaces_from_entities(self, entity_list):
        for item in entity_list:
            for key in ["source", "relationship", "destination"]:
                value = item.get(key)
                if value is not None:
                    if key == "relationship":
                        value = _sanitize_relationship_type(value)
                    else:
                        value = _normalize_graph_name(value)
                    item[key] = value
                else:
                    item[key] = "null"
        return entity_list
    # def _search_source_node(self, source_embedding, filters, threshold=0.9):
    #     agent_filter = ""
    #     if filters.get("agent_id"):
    #         agent_filter = "AND source_candidate.agent_id = $agent_id"

    #     cypher = f"""
    #         MATCH (source_candidate {self.node_label})
    #         WHERE source_candidate.embedding IS NOT NULL 
    #         AND source_candidate.user_id = $user_id
    #         {agent_filter}

    #         WITH source_candidate,
    #         round(2 * vector.similarity.cosine(source_candidate.embedding, $source_embedding) - 1, 4) AS source_similarity // denormalize for backward compatibility
    #         WHERE source_similarity >= $threshold

    #         WITH source_candidate, source_similarity
    #         ORDER BY source_similarity DESC
    #         LIMIT 1

    #         RETURN elementId(source_candidate)
    #         """

    #     params = {
    #         "source_embedding": source_embedding,
    #         "user_id": filters["user_id"],
    #         "threshold": threshold,
    #     }
    #     if filters.get("agent_id"):
    #         params["agent_id"] = filters["agent_id"]

    #     result = self.graph.query(cypher, params=params)
    #     return result
    def _search_source_node(self, source_embedding, filters, threshold=0.9):
    # 修复向量格式 - 添加这部分代码
        import math
        import numpy as np

        # 处理source_embedding
        if isinstance(source_embedding, np.ndarray):
            source_embedding = source_embedding.tolist()
        if isinstance(source_embedding, list) and len(source_embedding) > 0:
            if isinstance(source_embedding[0], list):
                source_embedding = source_embedding[0]  # 展开嵌套列表
        # 确保所有值都是有限数值，使用numpy方法处理
        source_embedding = [float(x) for x in source_embedding if not (np.isnan(x) or np.isinf(x))]

        # 检查向量是否为空
        if not source_embedding or len(source_embedding) == 0:
            logger.warning(f"Empty source embedding")
            return []

        agent_filter = ""
        if filters.get("agent_id"):
            agent_filter = "AND source_candidate.agent_id = $agent_id"

        cypher = f"""
            MATCH (source_candidate {self.node_label})
            WHERE source_candidate.embedding IS NOT NULL 
            AND source_candidate.user_id = $user_id
            {agent_filter}

            WITH source_candidate,
            round(2 * vector.similarity.cosine(source_candidate.embedding, $source_embedding) - 1, 4) AS source_similarity // denormalize for backward compatibility
            WHERE source_similarity >= $threshold

            WITH source_candidate, source_similarity
            ORDER BY source_similarity DESC
            LIMIT 1

            RETURN elementId(source_candidate)
            """

        params = {
            "source_embedding": source_embedding,
            "user_id": filters["user_id"],
            "threshold": threshold,
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]

        try:
            result = self.graph.query(cypher, params=params)
            return result
        except Exception as e:
            logger.error(f"Neo4j query failed in _search_source_node: {e}")
            return []


        # def _search_destination_node(self, destination_embedding, filters, threshold=0.9):
        #     agent_filter = ""
        #     if filters.get("agent_id"):
        #         agent_filter = "AND destination_candidate.agent_id = $agent_id"

        #     cypher = f"""
        #         MATCH (destination_candidate {self.node_label})
        #         WHERE destination_candidate.embedding IS NOT NULL 
        #         AND destination_candidate.user_id = $user_id
        #         {agent_filter}

        #         WITH destination_candidate,
        #         round(2 * vector.similarity.cosine(destination_candidate.embedding, $destination_embedding) - 1, 4) AS destination_similarity // denormalize for backward compatibility

        #         WHERE destination_similarity >= $threshold

        #         WITH destination_candidate, destination_similarity
        #         ORDER BY destination_similarity DESC
        #         LIMIT 1

        #         RETURN elementId(destination_candidate)
        #         """

        #     params = {
        #         "destination_embedding": destination_embedding,
        #         "user_id": filters["user_id"],
        #         "threshold": threshold,
        #     }
        #     if filters.get("agent_id"):
        #         params["agent_id"] = filters["agent_id"]

        #     result = self.graph.query(cypher, params=params)
        #     return result
    def _search_destination_node(self, destination_embedding, filters, threshold=0.9):
        # 修复向量格式 - 添加这部分代码
        import math
        import numpy as np
        
        # 处理destination_embedding
        if isinstance(destination_embedding, np.ndarray):
            destination_embedding = destination_embedding.tolist()
        if isinstance(destination_embedding, list) and len(destination_embedding) > 0:
            if isinstance(destination_embedding[0], list):
                destination_embedding = destination_embedding[0]  # 展开嵌套列表
        # 确保所有值都是有限数值，使用numpy方法处理
        destination_embedding = [float(x) for x in destination_embedding if not (np.isnan(x) or np.isinf(x))]
        
        # 检查向量是否为空
        if not destination_embedding or len(destination_embedding) == 0:
            logger.warning(f"Empty destination embedding")
            return []
        
        agent_filter = ""
        if filters.get("agent_id"):
            agent_filter = "AND destination_candidate.agent_id = $agent_id"
    
        cypher = f"""
            MATCH (destination_candidate {self.node_label})
            WHERE destination_candidate.embedding IS NOT NULL 
            AND destination_candidate.user_id = $user_id
            {agent_filter}
    
            WITH destination_candidate,
            round(2 * vector.similarity.cosine(destination_candidate.embedding, $destination_embedding) - 1, 4) AS destination_similarity // denormalize for backward compatibility
    
            WHERE destination_similarity >= $threshold
    
            WITH destination_candidate, destination_similarity
            ORDER BY destination_similarity DESC
            LIMIT 1
    
            RETURN elementId(destination_candidate)
            """
    
        params = {
            "destination_embedding": destination_embedding,
            "user_id": filters["user_id"],
            "threshold": threshold,
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
    
        try:
            result = self.graph.query(cypher, params=params)
            return result
        except Exception as e:
            logger.error(f"Neo4j query failed in _search_destination_node: {e}")
            return []
    
