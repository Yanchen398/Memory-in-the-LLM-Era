"""NetworkX-backed Graphiti-style mid-term memory for SOTA ablations."""

import json
import math
import os
import re
import threading
from collections import Counter, deque
from datetime import datetime, timezone

import networkx as nx
import numpy as np
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from graphiti_core.prompts.dedupe_nodes import NodeResolutions
from graphiti_core.prompts.extract_edges import ExtractedEdges
from graphiti_core.prompts.extract_nodes import ExtractedEntities

from .utils import ensure_directory_exists, generate_id, get_embedding


_ENTITY_TYPES = [
    {
        "entity_type_id": 0,
        "entity_type_name": "Entity",
        "entity_type_description": (
            "Default entity classification. Use this entity type if the entity "
            "is not one of the other listed types."
        ),
    }
]


def _normalise_name(value):
    value = re.sub(r"\s+", " ", str(value).lower()).strip()
    return value


def _fuzzy_name(value):
    value = re.sub(r"[^a-z0-9' ]", " ", _normalise_name(value))
    return re.sub(r"\s+", " ", value).strip()


def _shingles(value):
    compact = _fuzzy_name(value).replace(" ", "")
    if len(compact) < 3:
        return {compact} if compact else set()
    return {compact[i:i + 3] for i in range(len(compact) - 2)}


def _jaccard(left, right):
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _cosine(left, right):
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _tokens(text):
    return re.findall(r"[\w']+", str(text).lower(), flags=re.UNICODE)


def _bm25_scores(query, documents):
    """Small dependency-free BM25 implementation used in Graphiti-style hybrid search."""
    if not documents:
        return []
    tokenised = [_tokens(document) for document in documents]
    query_tokens = _tokens(query)
    average_length = sum(len(item) for item in tokenised) / max(1, len(tokenised))
    document_frequency = Counter()
    for document in tokenised:
        document_frequency.update(set(document))

    scores = []
    k1, b = 1.5, 0.75
    for document in tokenised:
        frequencies = Counter(document)
        score = 0.0
        for token in query_tokens:
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            df = document_frequency.get(token, 0)
            idf = math.log(1 + (len(tokenised) - df + 0.5) / (df + 0.5))
            norm = frequency + k1 * (
                1 - b + b * len(document) / max(average_length, 1.0)
            )
            score += idf * frequency * (k1 + 1) / norm
        scores.append(score)
    return scores


def _rank(scores, positive_only=False):
    ordered = sorted(
        enumerate(scores),
        key=lambda item: (-item[1], item[0]),
    )
    if positive_only:
        ordered = [item for item in ordered if item[1] > 0]
    return {index: rank + 1 for rank, (index, _) in enumerate(ordered)}


class GraphMemory:
    """Persisted MultiDiGraph using Graphiti prompts and hybrid retrieval."""

    def __init__(
        self,
        file_path,
        client,
        llm_model,
        context_window=3,
        candidate_count=10,
        dedupe_candidate_count=20,
        fuzzy_threshold=0.90,
        search_hops=1,
        use_llm_dedup=True,
        use_edge_dedup=True,
    ):
        self.file_path = file_path
        self.client = client
        self.llm_model = llm_model
        self.context_window = max(0, int(context_window))
        self.candidate_count = max(1, int(candidate_count))
        self.dedupe_candidate_count = max(1, int(dedupe_candidate_count))
        self.fuzzy_threshold = float(fuzzy_threshold)
        self.search_hops = max(0, int(search_hops))
        self.use_llm_dedup = bool(use_llm_dedup)
        self.use_edge_dedup = bool(use_edge_dedup)
        self.graph = nx.MultiDiGraph()
        self.episodes = []
        self._lock = threading.RLock()
        self._load()

    def _load(self):
        if not self.file_path or not os.path.exists(self.file_path):
            return
        try:
            with open(self.file_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.graph = nx.node_link_graph(
                payload.get("graph", {}),
                directed=True,
                multigraph=True,
                edges="links",
            )
            self.episodes = payload.get("episodes", [])
            print(
                f"GraphMemory: loaded {self.graph.number_of_nodes()} entities and "
                f"{self.graph.number_of_edges()} facts from {self.file_path}"
            )
        except (OSError, ValueError, nx.NetworkXError) as exc:
            raise RuntimeError(f"Could not load graph memory {self.file_path}: {exc}") from exc

    def save(self):
        ensure_directory_exists(self.file_path)
        payload = {
            "version": 1,
            "episodes": self.episodes,
            "graph": nx.node_link_data(self.graph, edges="links"),
        }
        temporary_path = f"{self.file_path}.tmp"
        with self._lock:
            with open(temporary_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(temporary_path, self.file_path)

    @staticmethod
    def _prompt_messages(messages):
        return [
            {"role": message.role, "content": message.content}
            for message in messages
        ]

    def _structured_completion(self, messages, response_model, prompt_name):
        schema = response_model.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": prompt_name.replace(".", "_"),
                "schema": schema,
            },
        }
        raw = self.client.chat_completion(
            model=self.llm_model,
            messages=self._prompt_messages(messages),
            temperature=0.0,
            max_tokens=(
                1024
                if prompt_name == "dedupe_edges.resolve_edge"
                else 3000
            ),
            response_format=response_format,
            stage=f"graph:{prompt_name}",
        )
        if not raw or raw.startswith("Error:"):
            raise RuntimeError(f"LLM failed for Graphiti prompt {prompt_name}: {raw}")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                raise RuntimeError(
                    f"Non-JSON response for Graphiti prompt {prompt_name}: {raw[:200]}"
                )
            payload = json.loads(match.group(0))
        return response_model.model_validate(payload)

    @staticmethod
    def _episode_content(content, unit_type):
        if unit_type != "message":
            return content
        lines = str(content).splitlines()
        if lines and lines[0].startswith("Conversation Timestamp:"):
            lines = lines[1:]
        return "\n".join(lines).strip()

    def _candidate_ids(self, name, embedding):
        candidates = []
        name_shingles = _shingles(name)
        for node_id, attributes in self.graph.nodes(data=True):
            exact = _normalise_name(attributes.get("name", "")) == _normalise_name(name)
            fuzzy = _jaccard(name_shingles, _shingles(attributes.get("name", "")))
            similarity = _cosine(embedding, attributes.get("name_embedding", []))
            candidates.append((exact, fuzzy, similarity, node_id))
        candidates.sort(key=lambda item: (-int(item[0]), -item[1], -item[2], item[3]))
        return [item[3] for item in candidates[:self.candidate_count]]

    def _deterministic_resolution(self, name):
        exact_matches = [
            node_id
            for node_id, attributes in self.graph.nodes(data=True)
            if _normalise_name(attributes.get("name", "")) == _normalise_name(name)
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]

        name_shingles = _shingles(name)
        if len(_fuzzy_name(name)) < 6 or not name_shingles:
            return None
        fuzzy_matches = []
        for node_id, attributes in self.graph.nodes(data=True):
            score = _jaccard(name_shingles, _shingles(attributes.get("name", "")))
            if score >= self.fuzzy_threshold:
                fuzzy_matches.append((score, node_id))
        fuzzy_matches.sort(key=lambda item: (-item[0], item[1]))
        return fuzzy_matches[0][1] if len(fuzzy_matches) == 1 else None

    def _resolve_entities(self, entities, content, previous_episodes):
        if not entities:
            return []

        name_embeddings = get_embedding([item["name"] for item in entities])
        resolved = [None] * len(entities)
        unresolved = []
        candidate_ids = []

        for index, entity in enumerate(entities):
            entity["name_embedding"] = name_embeddings[index].tolist()
            node_id = self._deterministic_resolution(entity["name"])
            if node_id is not None:
                resolved[index] = (node_id, self.graph.nodes[node_id].get("name", entity["name"]))
                continue
            unresolved.append(index)
            for candidate_id in self._candidate_ids(
                entity["name"], entity["name_embedding"]
            ):
                if candidate_id not in candidate_ids:
                    candidate_ids.append(candidate_id)

        # Graphiti resolves against a bounded search result set rather than
        # serialising the full graph into the LLM prompt. Keep the globally
        # most similar candidates so prompt size cannot grow with the graph.
        if len(candidate_ids) > self.dedupe_candidate_count:
            candidate_ids.sort(
                key=lambda node_id: (
                    -max(
                        _cosine(
                            entities[index]["name_embedding"],
                            self.graph.nodes[node_id].get("name_embedding", []),
                        )
                        for index in unresolved
                    ),
                    node_id,
                )
            )
            candidate_ids = candidate_ids[:self.dedupe_candidate_count]

        if unresolved and candidate_ids and self.use_llm_dedup:
            extracted_context = [
                {
                    "id": relative_id,
                    "name": entities[entity_index]["name"],
                    "entity_type": ["Entity"],
                    "entity_type_description": "Default Entity Type",
                }
                for relative_id, entity_index in enumerate(unresolved)
            ]
            existing_context = [
                {
                    "idx": index,
                    "name": self.graph.nodes[node_id].get("name", ""),
                    "entity_types": self.graph.nodes[node_id].get(
                        "entity_types", ["Entity"]
                    ),
                    "summary": self.graph.nodes[node_id].get("summary", ""),
                    "aliases": self.graph.nodes[node_id].get("aliases", []),
                }
                for index, node_id in enumerate(candidate_ids)
            ]
            context = {
                "extracted_nodes": extracted_context,
                "existing_nodes": existing_context,
                "episode_content": content,
                "previous_episodes": previous_episodes,
            }
            response = self._structured_completion(
                prompt_library.dedupe_nodes.nodes(context),
                NodeResolutions,
                "dedupe_nodes.nodes",
            )
            for resolution in response.entity_resolutions:
                if not 0 <= resolution.id < len(unresolved):
                    continue
                entity_index = unresolved[resolution.id]
                if 0 <= resolution.duplicate_idx < len(candidate_ids):
                    node_id = candidate_ids[resolution.duplicate_idx]
                    resolved[entity_index] = (
                        node_id,
                        resolution.name or self.graph.nodes[node_id].get("name", ""),
                    )
                elif resolution.name:
                    entities[entity_index]["name"] = resolution.name

        output = []
        for index, entity in enumerate(entities):
            if resolved[index] is None:
                node_id = generate_id("entity")
                self.graph.add_node(
                    node_id,
                    name=entity["name"],
                    entity_types=["Entity"],
                    summary="",
                    aliases=[entity["name"]],
                    name_embedding=entity["name_embedding"],
                    episode_ids=[],
                    mention_count=0,
                )
            else:
                node_id, best_name = resolved[index]
                attributes = self.graph.nodes[node_id]
                aliases = list(attributes.get("aliases", []))
                if entity["name"] not in aliases:
                    aliases.append(entity["name"])
                attributes["aliases"] = aliases
                if best_name and len(best_name) > len(attributes.get("name", "")):
                    attributes["name"] = best_name
                    attributes["name_embedding"] = get_embedding(best_name)[0].tolist()
            output.append(
                {
                    "id": entity["id"],
                    "node_id": node_id,
                    "name": self.graph.nodes[node_id].get("name", entity["name"]),
                }
            )
        return output

    def _touch_entities(self, resolved_entities, episode_id, content):
        excerpt = re.sub(r"\s+", " ", content).strip()[:800]
        for entity in resolved_entities:
            attributes = self.graph.nodes[entity["node_id"]]
            episode_ids = list(attributes.get("episode_ids", []))
            if episode_id not in episode_ids:
                episode_ids.append(episode_id)
            attributes["episode_ids"] = episode_ids
            attributes["mention_count"] = int(attributes.get("mention_count", 0)) + 1
            old_summary = attributes.get("summary", "")
            if excerpt and excerpt not in old_summary:
                attributes["summary"] = (old_summary + " " + excerpt).strip()[-1600:]

    def _resolve_edge(self, edge, source_id, target_id, timestamp):
        related = []
        for left, right, key, attributes in self.graph.edges(keys=True, data=True):
            if {left, right} == {source_id, target_id}:
                related.append((left, right, key, attributes))

        normalised_fact = _normalise_name(edge.fact)
        for _, _, _, attributes in related:
            if _normalise_name(attributes.get("fact", "")) == normalised_fact:
                return attributes, True

        # Exact matching above still covers every stored edge. For fuzzy LLM
        # resolution, bound the candidates by relevance instead of allowing
        # the prompt to grow with the number of stored facts.
        if len(related) > self.dedupe_candidate_count:
            scores = _bm25_scores(
                edge.fact,
                [item[3].get("fact", "") for item in related],
            )
            ranked = sorted(
                range(len(related)),
                key=lambda index: (-scores[index], -index),
            )
            related = [
                related[index]
                for index in ranked[:self.dedupe_candidate_count]
            ]

        if not related or not self.use_edge_dedup:
            return None, False

        existing_edges = [
            {
                "idx": index,
                "fact": attributes.get("fact", ""),
                "relation_type": attributes.get("relation_type", "DEFAULT"),
                "valid_at": attributes.get("valid_at"),
                "invalid_at": attributes.get("invalid_at"),
            }
            for index, (_, _, _, attributes) in enumerate(related)
        ]
        new_edge = {
            "fact": edge.fact,
            "relation_type": edge.relation_type,
            "valid_at": edge.valid_at,
            "invalid_at": edge.invalid_at,
        }
        context = {
            "edge_types": [],
            "existing_edges": existing_edges,
            "edge_invalidation_candidates": existing_edges,
            "new_edge": new_edge,
        }
        response = self._structured_completion(
            prompt_library.dedupe_edges.resolve_edge(context),
            EdgeDuplicate,
            "dedupe_edges.resolve_edge",
        )
        for index in response.contradicted_facts:
            if 0 <= index < len(related):
                related[index][3]["invalid_at"] = timestamp
        if response.duplicate_facts:
            index = response.duplicate_facts[0]
            if 0 <= index < len(related):
                return related[index][3], True
        return None, False

    def add_memory(self, content, timestamp=None, unit_type="message"):
        """Extract, resolve, and persist one episodic memory unit."""
        timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        episode_content = self._episode_content(content, unit_type)
        previous_episodes = [
            item["content"] for item in self.episodes[-self.context_window:]
        ] if self.context_window else []

        node_context = {
            "entity_types": _ENTITY_TYPES,
            "previous_episodes": previous_episodes,
            "episode_content": episode_content,
            "custom_prompt": "",
            "source_description": "Conversation memory",
        }
        node_prompt = (
            prompt_library.extract_nodes.extract_message(node_context)
            if unit_type == "message"
            else prompt_library.extract_nodes.extract_text(node_context)
        )
        extracted = self._structured_completion(
            node_prompt,
            ExtractedEntities,
            (
                "extract_nodes.extract_message"
                if unit_type == "message"
                else "extract_nodes.extract_text"
            ),
        )
        entities = [
            {"id": index, "name": item.name, "entity_type_id": item.entity_type_id}
            for index, item in enumerate(extracted.extracted_entities)
            if item.name.strip()
        ]
        resolved = self._resolve_entities(
            entities, episode_content, previous_episodes
        )
        episode_id = generate_id("episode")
        self._touch_entities(resolved, episode_id, episode_content)

        edge_context = {
            "edge_types": [],
            "previous_episodes": previous_episodes,
            "episode_content": episode_content,
            "nodes": [
                {"id": item["id"], "name": item["name"]}
                for item in resolved
            ],
            "reference_time": str(timestamp).replace(" ", "T"),
            "custom_prompt": "",
        }
        extracted_edges = self._structured_completion(
            prompt_library.extract_edges.edge(edge_context),
            ExtractedEdges,
            "extract_edges.edge",
        )
        entity_by_id = {item["id"]: item for item in resolved}
        stored_edge_ids = []
        for edge in extracted_edges.edges:
            source = entity_by_id.get(edge.source_entity_id)
            target = entity_by_id.get(edge.target_entity_id)
            if not source or not target or source["node_id"] == target["node_id"]:
                continue
            duplicate, is_duplicate = self._resolve_edge(
                edge, source["node_id"], target["node_id"], timestamp
            )
            if is_duplicate:
                episode_ids = list(duplicate.get("episode_ids", []))
                if episode_id not in episode_ids:
                    episode_ids.append(episode_id)
                duplicate["episode_ids"] = episode_ids
                continue
            edge_id = generate_id("fact")
            fact_embedding = get_embedding(edge.fact)[0].tolist()
            self.graph.add_edge(
                source["node_id"],
                target["node_id"],
                key=edge_id,
                edge_id=edge_id,
                relation_type=edge.relation_type,
                fact=edge.fact,
                fact_embedding=fact_embedding,
                valid_at=edge.valid_at,
                invalid_at=edge.invalid_at,
                episode_ids=[episode_id],
            )
            stored_edge_ids.append(edge_id)

        self.episodes.append(
            {
                "episode_id": episode_id,
                "content": episode_content,
                "timestamp": timestamp,
                "unit_type": unit_type,
                "entity_ids": [item["node_id"] for item in resolved],
                "edge_ids": stored_edge_ids,
            }
        )
        self.save()
        return episode_id

    def _edge_documents(self):
        records = []
        for source, target, key, attributes in self.graph.edges(keys=True, data=True):
            source_name = self.graph.nodes[source].get("name", source)
            target_name = self.graph.nodes[target].get("name", target)
            relation = attributes.get("relation_type", "RELATED_TO")
            fact = attributes.get("fact", "")
            time_text = ""
            if attributes.get("valid_at"):
                time_text += f" valid_at={attributes['valid_at']}"
            if attributes.get("invalid_at"):
                time_text += f" invalid_at={attributes['invalid_at']}"
            text = f"{source_name} --{relation}--> {target_name}: {fact}{time_text}"
            records.append(
                {
                    "key": key,
                    "source": source,
                    "target": target,
                    "text": text,
                    "embedding": attributes.get("fact_embedding", []),
                    "episode_ids": list(attributes.get("episode_ids", [])),
                }
            )
        return records

    def retrieve(self, query, top_k=10, include_episode_ids=False):
        """Graphiti-like hybrid edge search: BM25 + cosine + BFS with RRF."""
        records = self._edge_documents()
        if not records:
            node_records = [
                {
                    "graph": (
                        f"{attributes.get('name', node_id)}: "
                        f"{attributes.get('summary', '')}"
                    )
                }
                for node_id, attributes in self.graph.nodes(data=True)
            ]
            return node_records[:top_k]

        query_embedding = get_embedding(query)[0].tolist()
        cosine_scores = [
            _cosine(query_embedding, record["embedding"])
            for record in records
        ]
        bm25_scores = _bm25_scores(query, [record["text"] for record in records])
        cosine_ranks = _rank(cosine_scores)
        bm25_ranks = _rank(bm25_scores, positive_only=True)

        node_scores = []
        node_ids = []
        for node_id, attributes in self.graph.nodes(data=True):
            node_ids.append(node_id)
            node_scores.append(
                _cosine(query_embedding, attributes.get("name_embedding", []))
            )
        seed_ids = [
            node_ids[index]
            for index in sorted(
                range(len(node_scores)),
                key=lambda idx: (-node_scores[idx], node_ids[idx]),
            )[:3]
        ]

        bfs_distance = {}
        undirected = self.graph.to_undirected()
        for seed_id in seed_ids:
            if seed_id not in undirected:
                continue
            lengths = nx.single_source_shortest_path_length(
                undirected, seed_id, cutoff=self.search_hops
            )
            for node_id, distance in lengths.items():
                bfs_distance[node_id] = min(
                    distance, bfs_distance.get(node_id, distance)
                )
        bfs_scores = []
        for record in records:
            distances = [
                bfs_distance[node_id]
                for node_id in (record["source"], record["target"])
                if node_id in bfs_distance
            ]
            bfs_scores.append(1.0 / (1 + min(distances)) if distances else 0.0)
        bfs_ranks = _rank(bfs_scores, positive_only=True)

        fused = []
        for index, record in enumerate(records):
            score = 0.0
            for ranks in (cosine_ranks, bm25_ranks, bfs_ranks):
                if index in ranks:
                    score += 1.0 / (60 + ranks[index])
            fused.append((score, record["key"], record["text"]))
        fused.sort(key=lambda item: (-item[0], item[1]))
        record_by_key = {record["key"]: record for record in records}
        results = []
        for _, edge_id, text in fused[:int(top_k)]:
            result = {"graph": text}
            if include_episode_ids:
                result["edge_id"] = edge_id
                result["episode_ids"] = list(
                    record_by_key[edge_id].get("episode_ids", [])
                )
            results.append(result)
        return results

    def retrieve_with_original_text(self, query, episode_memory, top_k=10):
        """Return Graph top-k facts augmented by their exact source pages.

        The graph ranking is unchanged. Source text is resolved only through
        each selected edge's persisted episode_ids; no second retrieval or
        indexing step is performed.
        """
        ranked = self.retrieve(
            query,
            top_k=top_k,
            include_episode_ids=True,
        )
        graph_episodes = {
            item.get("episode_id"): item
            for item in self.episodes
            if item.get("episode_id")
        }
        output = []
        for rank, item in enumerate(ranked, start=1):
            original_texts = []
            seen = set()
            for episode_id in item.get("episode_ids", []):
                memory = episode_memory.get(episode_id, {})
                for page in memory.get("pages", {}).values():
                    content = str(page.get("content", "")).strip()
                    if content and content not in seen:
                        seen.add(content)
                        original_texts.append(content)
                if not original_texts:
                    fallback = str(
                        graph_episodes.get(episode_id, {}).get("content", "")
                    ).strip()
                    if fallback and fallback not in seen:
                        seen.add(fallback)
                        original_texts.append(fallback)

            original_block = (
                "\n\n".join(original_texts)
                if original_texts
                else "No persisted source text was found."
            )
            combined = (
                f"[Graph fact rank {rank}]\n{item['graph']}\n"
                f"[Related original conversation]\n{original_block}"
            )
            output.append({"graph": combined})
        return output
