import asyncio
import numpy as np
import concurrent
import gc
import hashlib
import traceback
import json
import logging
import os
import uuid
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional

import pytz
from pydantic import ValidationError

from mem0.configs.base import MemoryConfig, MemoryItem
from mem0.configs.enums import MemoryType
from mem0.configs.prompts import (
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    get_update_memory_messages,
)
from mem0.memory.base import MemoryBase
from mem0.memory.setup import mem0_dir, setup_config
from mem0.memory.storage import SQLiteManager
from mem0.memory.telemetry import capture_event
from mem0.memory.utils import (
    get_fact_retrieval_messages,
    parse_messages,
    parse_vision_messages,
    process_telemetry_filters,
    remove_code_blocks,
)
from mem0.utils.factory import EmbedderFactory, LlmFactory, VectorStoreFactory
from sentence_transformers import SentenceTransformer

# class LocalEmbedder:
#     def __init__(self, model_name='all-mpnet-base-v2'):
#         self.model = SentenceTransformer(model_name)
#         self.embedding_dims = 768
#         self.config = type("Config", (), {"embedding_dims": 768})()
#     def embed(self, texts, *args, **kwargs):
#         if isinstance(texts, str):
#             texts = [texts]
#         return self.model.encode(texts)
class LocalEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        device = os.getenv("MEM0_EMBEDDING_DEVICE", "cpu")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dims = 384
        self.config = type("Config", (), {"embedding_dims": 384})()

    def embed(self, texts, *args, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts)
def _build_filters_and_metadata(
    *,  # Enforce keyword-only arguments
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    actor_id: Optional[str] = None,  # For query-time filtering
    input_metadata: Optional[Dict[str, Any]] = None,
    input_filters: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Constructs metadata for storage and filters for querying based on session and actor identifiers.

    This helper supports multiple session identifiers (`user_id`, `agent_id`, and/or `run_id`)
    for flexible session scoping and optionally narrows queries to a specific `actor_id`. It returns two dicts:

    1. `base_metadata_template`: Used as a template for metadata when storing new memories.
       It includes all provided session identifier(s) and any `input_metadata`.
    2. `effective_query_filters`: Used for querying existing memories. It includes all
       provided session identifier(s), any `input_filters`, and a resolved actor
       identifier for targeted filtering if specified by any actor-related inputs.

    Actor filtering precedence: explicit `actor_id` arg → `filters["actor_id"]`
    This resolved actor ID is used for querying but is not added to `base_metadata_template`,
    as the actor for storage is typically derived from message content at a later stage.

    Args:
        user_id (Optional[str]): User identifier, for session scoping.
        agent_id (Optional[str]): Agent identifier, for session scoping.
        run_id (Optional[str]): Run identifier, for session scoping.
        actor_id (Optional[str]): Explicit actor identifier, used as a potential source for
            actor-specific filtering. See actor resolution precedence in the main description.
        input_metadata (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session identifiers for the storage metadata template. Defaults to an empty dict.
        input_filters (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session and actor identifiers for query filters. Defaults to an empty dict.

    Returns:
        tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - base_metadata_template (Dict[str, Any]): Metadata template for storing memories,
              scoped to the provided session(s).
            - effective_query_filters (Dict[str, Any]): Filters for querying memories,
              scoped to the provided session(s) and potentially a resolved actor.
    """

    base_metadata_template = deepcopy(input_metadata) if input_metadata else {}
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    # ---------- add all provided session ids ----------
    session_ids_provided = []

    if user_id:
        base_metadata_template["user_id"] = user_id
        effective_query_filters["user_id"] = user_id
        session_ids_provided.append("user_id")

    if agent_id:
        base_metadata_template["agent_id"] = agent_id
        effective_query_filters["agent_id"] = agent_id
        session_ids_provided.append("agent_id")

    if run_id:
        base_metadata_template["run_id"] = run_id
        effective_query_filters["run_id"] = run_id
        session_ids_provided.append("run_id")

    if not session_ids_provided:
        raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be provided.")

    # ---------- optional actor filter ----------
    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    if resolved_actor_id:
        effective_query_filters["actor_id"] = resolved_actor_id

    return base_metadata_template, effective_query_filters


setup_config()
logger = logging.getLogger(__name__)


def _normalize_memory_actions(payload):
    """Normalize valid-but-off-schema memory-update responses."""
    if isinstance(payload, dict):
        actions = payload.get("memory", [])
    elif isinstance(payload, list):
        actions = payload
    elif isinstance(payload, str):
        actions = [payload]
    else:
        logging.warning("Skipping memory-update response with unsupported type: %s", type(payload).__name__)
        return []

    if isinstance(actions, (dict, str)):
        actions = [actions]
    elif not isinstance(actions, list):
        logging.warning("Skipping `memory` field with unsupported type: %s", type(actions).__name__)
        return []

    normalized_actions = []
    for idx, action in enumerate(actions):
        if isinstance(action, dict):
            normalized_actions.append(action)
            continue

        if isinstance(action, str):
            action_text = action.strip()
            if action_text:
                logging.warning("Treating string memory-update entry as an ADD action.")
                normalized_actions.append(
                    {
                        "id": str(idx),
                        "text": action_text,
                        "event": "ADD",
                    }
                )
            continue

        logging.warning("Skipping memory-update entry with unsupported type: %s", type(action).__name__)

    return normalized_actions


MEM0_DEBUG_SEARCH = os.getenv("MEM0_DEBUG_SEARCH", "0").lower() in {"1", "true", "yes", "on"}

def _debug_print(*args, **kwargs):
    if MEM0_DEBUG_SEARCH:
        print(*args, **kwargs)



class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.custom_fact_extraction_prompt = self.config.custom_fact_extraction_prompt
        self.custom_update_memory_prompt = self.config.custom_update_memory_prompt

        self.embedding_model = LocalEmbedder()

        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )

        llm_config = dict(getattr(config.llm, "config", {}) or {})
        qwen_base_url = (
            os.getenv("QWEN35_BASE_URL")
            or llm_config.get("openai_base_url")
            or llm_config.get("vllm_base_url")
            or os.getenv("OPENAI_BASE_URL")
            or "http://127.0.0.1:8000/v1"
        )
        qwen_model = (
            llm_config.get("model")
            or os.getenv("QWEN35_MODEL")
            or "Qwen/Qwen3.5-9B"
        )
        config.llm.provider = "openai"
        config.llm.config = {
            **llm_config,
            "api_key": llm_config.get("api_key") or os.getenv("OPENAI_API_KEY") or "fake_key",
            "openai_base_url": qwen_base_url,
            "model": qwen_model,
        }

        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        # self.api_version = self.config.version
        self.api_version = "v1.1"

        
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)



        self.enable_graph = False
        # self.enable_graph = True

        if hasattr(self.config, 'graph_store') and self.config.graph_store and self.config.graph_store.config:
            try:
                if self.config.graph_store.provider == "memgraph":
                    from mem0.memory.memgraph_memory import MemoryGraph
                else:
                    from mem0.memory.graph_memory import MemoryGraph

                self.graph = MemoryGraph(self.config)
                self.enable_graph = True
                print("Graph store initialized successfully")
            except Exception as e:
                print(f"Failed to initialize graph store: {e}")
                self.graph = None
                self.enable_graph = False
        else:
            print("Graph store not configured, disabling graph functionality")
            self.graph = None
            self.enable_graph = False
    
        self.config.vector_store.config.collection_name = "mem0migrations"
        if self.config.vector_store.provider in ["faiss", "qdrant"]:
            provider_path = f"migrations_{self.config.vector_store.provider}"
            self.config.vector_store.config.path = os.path.join(mem0_dir, provider_path)
            os.makedirs(self.config.vector_store.config.path, exist_ok=True)
        self._telemetry_vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        capture_event("mem0.init", self, {"sync_type": "sync"})

    def close(self):
        self.executor.shutdown(wait=True)

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "graph_store" in config_dict:
            if "vector_store" not in config_dict and "embedder" in config_dict:
                config_dict["vector_store"] = {}
                config_dict["vector_store"]["config"] = {}
                config_dict["vector_store"]["config"]["embedding_model_dims"] = config_dict["embedder"]["config"][
                    "embedding_dims"
                ]
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    def add_old(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """
        Create a new memory.

        Adds new memories scoped to a single session id (e.g. `user_id`, `agent_id`, or `run_id`). One of those ids is required.

        Args:
            messages (str or List[Dict[str, str]]): The message content or list of messages
                (e.g., `[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]`)
                to be processed and stored.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): If True (default), an LLM is used to extract key facts from
                'messages' and decide whether to add, update, or delete related memories.
                If False, 'messages' are added as raw memories directly.
            memory_type (str, optional): Specifies the type of memory. Currently, only
                `MemoryType.PROCEDURAL.value` ("procedural_memory") is explicitly handled for
                creating procedural memories (typically requires 'agent_id'). Otherwise, memories
                are treated as general conversational/factual memories.memory_type (str, optional): Type of memory to create. Defaults to None. By default, it creates the short term memories and long term (semantic and episodic) memories. Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.


        Returns:
            dict: A dictionary containing the result of the memory addition operation, typically
                  including a list of memory items affected (added, updated) under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "event": "ADD"}]}`
        """

        _debug_print("used here=============================================")
        _debug_print(" [add] called")
        _debug_print(f" messages={messages}")
        _debug_print(f" user_id={user_id}, agent_id={agent_id}, run_id={run_id}")
        _debug_print(f" metadata={metadata}, infer={infer}, memory_type={memory_type}, prompt={prompt}")
        _debug_print(f" [add] processed_metadata={processed_metadata}")
        _debug_print(f" [add] effective_filters={effective_filters}")
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise ValueError(
                f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise ValueError("messages must be str, dict, or list[dict]")

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = self._create_procedural_memory(messages, metadata=processed_metadata, prompt=prompt)
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(self._add_to_vector_store, messages, processed_metadata, effective_filters, infer)
            future2 = executor.submit(self._add_to_graph, messages, effective_filters)

            concurrent.futures.wait([future1, future2])

            vector_store_result = future1.result()
            graph_result = future2.result()

        if self.api_version == "v1.0":
            warnings.warn(
                "The current add API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return vector_store_result

        if self.enable_graph:
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }

        return {"results": vector_store_result}
    def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """
        Create a new memory.

        Adds new memories scoped to a single session id (e.g. `user_id`, `agent_id`, or `run_id`). One of those ids is required.

        Args:
            messages (str or List[Dict[str, str]]): The message content or list of messages
                (e.g., `[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]`)
                to be processed and stored.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): If True (default), an LLM is used to extract key facts from
                'messages' and decide whether to add, update, or delete related memories.
                If False, 'messages' are added as raw memories directly.
            memory_type (str, optional): Specifies the type of memory. Currently, only
                `MemoryType.PROCEDURAL.value` ("procedural_memory") is explicitly handled for
                creating procedural memories (typically requires 'agent_id'). Otherwise, memories
                are treated as general conversational/factual memories.memory_type (str, optional): Type of memory to create. Defaults to None. By default, it creates the short term memories and long term (semantic and episodic) memories. Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.


        Returns:
            dict: A dictionary containing the result of the memory addition operation, typically
                  including a list of memory items affected (added, updated) under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "event": "ADD"}]}`
        """

        _debug_print("=== add() called ===")
        _debug_print(f"Input messages type: {type(messages)}, value: {messages}")
        _debug_print(f"user_id: {user_id}, agent_id: {agent_id}, run_id: {run_id}")
        _debug_print(f"metadata: {metadata}, infer: {infer}, memory_type: {memory_type}, prompt: {prompt}")

        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )
        _debug_print(f"Processed metadata: {processed_metadata}")
        _debug_print(f"Effective filters: {effective_filters}")

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            print(f"Error: Invalid memory_type '{memory_type}'")
            raise ValueError(
                f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories."
            )

        if isinstance(messages, str):
            _debug_print("messages is str, converting to list of dict")
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            _debug_print("messages is dict, converting to list with one dict")
            messages = [messages]

        elif not isinstance(messages, list):
            print("Error: messages is not str, dict or list")
            raise ValueError("messages must be str, dict, or list[dict]")

        _debug_print(f"Normalized messages: {messages}")

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            _debug_print("Creating procedural memory")
            results = self._create_procedural_memory(messages, metadata=processed_metadata, prompt=prompt)
            _debug_print(f"Procedural memory creation results: {results}")
            return results

        if self.config.llm.config.get("enable_vision"):
            _debug_print("Vision enabled, parsing vision messages with vision details")
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            _debug_print("Vision not enabled, parsing vision messages without vision details")
            messages = parse_vision_messages(messages)

        _debug_print(f"Messages after vision parsing: {messages}")
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        if True:
            _debug_print("Starting concurrent futures for vector store and graph")
            # future1 = self.executor.submit(self._add_to_vector_store, messages, processed_metadata, effective_filters, infer)
            # future2 = self.executor.submit(self._add_to_graph, messages, effective_filters)
            
            # future1 = executor.submit(self._add_to_vector_store, messages, processed_metadata, effective_filters, infer)
            # future2 = executor.submit(self._add_to_graph, messages, effective_filters)

            # concurrent.futures.wait([future1, future2])
            # print("Concurrent futures completed")

            # 容错处理
            # current_result = future1.result(timeout=600)
            # current_result = self._add_to_vector_store(messages, processed_metadata, effective_filters, infer)
            # print("current_result: ", current_result)
            try:
                # vector_store_result = future1.result(timeout=600)
                vector_store_result = self._add_to_vector_store(messages, processed_metadata, effective_filters, infer)
                _debug_print("vector_store_result1: ", vector_store_result)
            except Exception as e:
                print(f"[Vector Store Error] {type(e)} {e}")
                print(traceback.format_exc())
                print(f"[Vector Store Error] {e}")
                vector_store_result = None

            # try:
            #     graph_result = future2.result(timeout=30)
            # except Exception as e:
            #     print(f"[Graph Store Error] {e}")
            #     graph_result = None
            if self.enable_graph:
                try:
                    graph_result = self._add_to_graph(messages, effective_filters)
                    _debug_print("graph_result1: ", graph_result)
                except Exception as e:
                    print(f"[Graph Store Error] {e}")
                    graph_result = None


            _debug_print(f"Vector store result: {vector_store_result}")
            if self.enable_graph:
                _debug_print(f"Graph result: {graph_result}")
        # # if self.api_version == "v1.0":
        # #     print("API version is v1.0, warning about deprecated output format")
        # #     warnings.warn(
        # #         "The current add API output format is deprecated. "
        # #         "To use the latest format, set `api_version='v1.1'`. "
        # #         "The current format will be removed in mem0ai 1.1.0 and later versions.",
        # #         category=DeprecationWarning,
        # #         stacklevel=2,
        # #     )
        # #     print("Returning vector store result only")
        # #     return vector_store_result

        # self.enable_graph = True
        if self.enable_graph:
        # if True:
            _debug_print("Graph enabled, returning both results and relations")
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }
        else:
            _debug_print("Graph not enabled, returning both results and relations")
            return {
                "results": vector_store_result,
                "relations": None,
            }

        _debug_print("Returning only results")
        # return {"results": vector_store_result}
    def add_graph(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """
        Create a new memory.

        Adds new memories scoped to a single session id (e.g. `user_id`, `agent_id`, or `run_id`). One of those ids is required.

        Args:
            messages (str or List[Dict[str, str]]): The message content or list of messages
                (e.g., `[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]`)
                to be processed and stored.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): If True (default), an LLM is used to extract key facts from
                'messages' and decide whether to add, update, or delete related memories.
                If False, 'messages' are added as raw memories directly.
            memory_type (str, optional): Specifies the type of memory. Currently, only
                `MemoryType.PROCEDURAL.value` ("procedural_memory") is explicitly handled for
                creating procedural memories (typically requires 'agent_id'). Otherwise, memories
                are treated as general conversational/factual memories.memory_type (str, optional): Type of memory to create. Defaults to None. By default, it creates the short term memories and long term (semantic and episodic) memories. Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.


        Returns:
            dict: A dictionary containing the result of the memory addition operation, typically
                  including a list of memory items affected (added, updated) under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "event": "ADD"}]}`
        """

        _debug_print("=== add() called ===")
        _debug_print(f"Input messages type: {type(messages)}, value: {messages}")
        _debug_print(f"user_id: {user_id}, agent_id: {agent_id}, run_id: {run_id}")
        _debug_print(f"metadata: {metadata}, infer: {infer}, memory_type: {memory_type}, prompt: {prompt}")

        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )
        _debug_print(f"Processed metadata: {processed_metadata}")
        _debug_print(f"Effective filters: {effective_filters}")

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            print(f"Error: Invalid memory_type '{memory_type}'")
            raise ValueError(
                f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories."
            )

        if isinstance(messages, str):
            _debug_print("messages is str, converting to list of dict")
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            _debug_print("messages is dict, converting to list with one dict")
            messages = [messages]

        elif not isinstance(messages, list):
            print("Error: messages is not str, dict or list")
            raise ValueError("messages must be str, dict, or list[dict]")

        _debug_print(f"Normalized messages: {messages}")

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            _debug_print("Creating procedural memory")
            results = self._create_procedural_memory(messages, metadata=processed_metadata, prompt=prompt)
            _debug_print(f"Procedural memory creation results: {results}")
            return results

        if self.config.llm.config.get("enable_vision"):
            _debug_print("Vision enabled, parsing vision messages with vision details")
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            _debug_print("Vision not enabled, parsing vision messages without vision details")
            messages = parse_vision_messages(messages)

        _debug_print(f"Messages after vision parsing: {messages}")


        # print("Start adding to graph")
        # graph_result = self._add_to_graph(messages, effective_filters)
        # print(f"Graph result: {graph_result}")



        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     print("Starting concurrent futures for vector store and graph")
        #     future1 = executor.submit(self._add_to_vector_store, messages, processed_metadata, effective_filters, infer)
        #     future2 = executor.submit(self._add_to_graph, messages, effective_filters)

        #     concurrent.futures.wait([future1, future2])
        #     # concurrent.futures.wait([future2])
        #     print("Concurrent futures completed")

        #     vector_store_result = future1.result()
        #     graph_result = future2.result()

        #     print(f"Vector store result: {vector_store_result}")
        #     print(f"Graph result: {graph_result}")

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        if True: 
            _debug_print("Starting concurrent futures for vector store and graph")
            future1 = self.executor.submit(self._add_to_vector_store, messages, processed_metadata, effective_filters, infer)
            future2 = self.executor.submit(self._add_to_graph, messages, effective_filters)

            concurrent.futures.wait([future1, future2])
            _debug_print("Concurrent futures completed")

            # 容错处理
            try:
                # vector_store_result = future1.result(timeout=20)
                vector_store_result = future1.result
            except Exception as e:
                print(f"[Vector Store Error] {e}")
                vector_store_result = None

            try:
                graph_result = future2.result(timeout=30)
            except Exception as e:
                print(f"[Graph Store Error] {e}")
                graph_result = None

            _debug_print(f"Vector store result: {vector_store_result}")
            print(f"Graph result: {graph_result}")





        # # if self.api_version == "v1.0":
        # #     print("API version is v1.0, warning about deprecated output format")
        # #     warnings.warn(
        # #         "The current add API output format is deprecated. "
        # #         "To use the latest format, set `api_version='v1.1'`. "
        # #         "The current format will be removed in mem0ai 1.1.0 and later versions.",
        # #         category=DeprecationWarning,
        # #         stacklevel=2,
        # #     )
        # #     print("Returning vector store result only")
        # #     return vector_store_result

        self.enable_graph = True
        if self.enable_graph:
            _debug_print("Graph enabled, returning both results and relations")
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }

        _debug_print("Returning only results")
        # return {"results": vector_store_result}


    def _add_to_vector_store(self, messages, metadata, filters, infer):
        if not infer:
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format: {message_dict}")
                    continue

                if message_dict["role"] == "system":
                    continue

                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                msg_content = message_dict["content"]
                msg_embeddings = self.embedding_model.embed(msg_content, "add")
                mem_id = self._create_memory(msg_content, msg_embeddings, per_msg_meta)


                _debug_print("memories_before_add_vector_store: ", message_dict)

                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        parsed_messages = parse_messages(messages)

        self.config.custom_fact_extraction_prompt = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories from the speaker's perspective, clearly identifying who each fact belongs to"

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations

Return the memories strictly following the json format of the example: {{"facts" : ["Had a meeting with John On 1 December", "Discussed the new project on 12 Octorber","Favourite movies are Inception and Interstellar", "Has camped in forest and mountains"]}}
Note that do not return content in the example.
"""

        # self.config.custom_fact_extraction_prompt = """
# Generate personal memories that follow these guidelines:

# 1. Each memory should be self-contained with complete context, including:
#    - The person's name, do not use "user" while creating memories
#    - Personal details (career aspirations, hobbies, life circumstances)
#    - Emotional states and reactions
#    - Ongoing journeys or future plans
#    - Specific dates when events occurred

# 2. Include meaningful personal narratives focusing on:
#    - Identity and self-acceptance journeys
#    - Family planning and parenting
#    - Creative outlets and hobbies
#    - Mental health and self-care activities
#    - Career aspirations and education goals
#    - Important life events and milestones

# 3. Make each memory rich with specific details rather than general statements
#    - Include timeframes (exact dates when possible)
#    - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
#    - Include emotional context and personal growth elements

# 4. Extract memories only from user messages, not incorporating assistant responses

# 5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations

# Return the memories strictly following the json format of the example: {{"facts" : ["Had a meeting with John On 1 December", "Discussed the new project on 12 Octorber","Favourite movies are Inception and Interstellar", "Has camped in forest and mountains"]}}
# Note that do not return content in the example.
# """


        if self.config.custom_fact_extraction_prompt:
            system_prompt = self.config.custom_fact_extraction_prompt
            user_prompt = f"Input:\n{parsed_messages}"
        else:
            system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages)


        _debug_print("system_prompt: ", system_prompt)
        _debug_print("user_prompt: ", user_prompt)

        try: 
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            print("problem what666: ")
            print(traceback.format_exc())
            logging.error(f"Error in fact extraction response: {e}")
            response = ""

        _debug_print("what's the response: ", response)
        try:
            response = remove_code_blocks(response)
            new_retrieved_facts = json.loads(response)["facts"]
        except Exception as e:
            logging.error(f"Error in new_retrieved_facts: {e}")
            new_retrieved_facts = []
        
        if not new_retrieved_facts:
            logger.debug("No new facts retrieved from input. Skipping memory update LLM call.")

        retrieved_old_memory = []
        new_message_embeddings = {}
        for new_mem in new_retrieved_facts:
            _debug_print("new_memL: ", new_mem)
            # temp_result = f"On {new_mem.get('date', '')} {new_mem.get('event', '')}"
            # new_mem = temp_result
            # new_mem = f"On {new_mem.get('date', '')} {new_mem.get('event', '')}"
            # messages_embeddings = self.embedding_model.embed(new_mem, "add")
            messages_embeddings = self.embedding_model.embed(new_mem, "add")
            new_message_embeddings[new_mem] = messages_embeddings
            existing_memories = self.vector_store.search(
                query=new_mem,
                vectors=messages_embeddings,
                limit=5,
                filters=filters,
            )
            for mem in existing_memories:
                retrieved_old_memory.append({"id": mem.id, "text": mem.payload["data"]})

        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())
        logging.info(f"Total existing memories: {len(retrieved_old_memory)}")

        # mapping UUIDs with integers for handling UUID hallucinations
        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)

        if new_retrieved_facts:
            function_calling_prompt = get_update_memory_messages(
                retrieved_old_memory, new_retrieved_facts, self.config.custom_update_memory_prompt
            )


            _debug_print("function_calling_prompt_fortest: ", function_calling_prompt)
            # response: str = self.llm.generate_response(
            #         messages=[{"role": "user", "content": function_calling_prompt}],
            #         response_format={"type": "json_object"},
            #     )

            def fallback_add_memories(facts, metadata=None):
                mems = []
                for idx, fact in enumerate(facts):
                    mems.append({
                        "id": str(idx),
                        "text": fact,
                        "event": "ADD",
                        "timestamp": metadata.get("timestamp") if metadata else "",
                        "speaker": metadata.get("speaker") if metadata else ""
                    })
                return {"memory": mems}

            try:
                response: str = self.llm.generate_response(
                    messages=[{"role": "user", "content": function_calling_prompt}],
                    response_format={"type": "json_object"},
                )
                try:
                    # response = remove_code_blocks(response)
                    if isinstance(response, dict):
                        new_memories_with_actions = response
                    else:
                        response = remove_code_blocks(response)
                        new_memories_with_actions = json.loads(response)
                        # new_memories_with_actions = json.loads(response)
                except Exception as e:
                    logging.warning(f"Invalid JSON memory-update response; using fallback ADD memories: {type(e).__name__}: {e}")
                    new_memories_with_actions = fallback_add_memories(new_retrieved_facts, metadata)
                    # new_memories_with_actions = json.loads(response)
            except Exception as e:
                logging.warning(f"Memory update response generation failed; using fallback ADD memories: {type(e).__name__}: {e}")
                # fallback: facts????ADD
                new_memories_with_actions = fallback_add_memories(new_retrieved_facts, metadata)
            # print("temp_response: ", temp_response)
            # try:
            #     response: str = self.llm.generate_response(
            #         messages=[{"role": "user", "content": function_calling_prompt}],
            #         response_format={"type": "json_object"},
            #     )
            # except Exception as e:
            #     print("problem what: ")
            #     print(traceback.format_exc())
            #     logging.error(f"Error in new memory actions response: {e}")
            #     response = ""


        else:
            new_memories_with_actions = {}

        returned_memories = []
        # try:
        for resp in _normalize_memory_actions(new_memories_with_actions):
            logging.info(resp)
            # try:
            if True:
                action_text = resp.get("text")
                if not action_text:
                    logging.info("Skipping memory entry because of empty `text` field.")
                    continue

                event_type = resp.get("event")

                mem_id_str = resp.get("id")
                if event_type in {"UPDATE", "DELETE"}:
                    if mem_id_str not in temp_uuid_mapping:
                        logging.warning(f"Memory ID '{mem_id_str}' not found in temp_uuid_mapping, skipping.")
                        continue
                
                # memory_id = self._create_memory(
                #     data=action_text,
                #     existing_embeddings=new_message_embeddings,
                #     metadata=deepcopy(metadata),
                # )
                # print("memory_id: ", memory_id)
                # returned_memories.append({"id": memory_id, "memory": action_text, "event": event_type})
                   
                
                if event_type == "ADD":
                    memory_id = self._create_memory(
                        data=action_text,
                        existing_embeddings=new_message_embeddings,
                        metadata=deepcopy(metadata),
                    )
                    _debug_print("memory_id: ", memory_id)
                    returned_memories.append({"id": memory_id, "memory": action_text, "event": event_type})
                elif event_type == "UPDATE":
                    memory_id = self._create_memory(
                        data=action_text,
                        existing_embeddings=new_message_embeddings,
                        metadata=deepcopy(metadata),
                    )
                    _debug_print("memory_id: ", memory_id)
                    returned_memories.append({"id": memory_id, "memory": action_text, "event": event_type})
                    # self._update_memory(
                    #     memory_id=temp_uuid_mapping[mem_id_str],
                    #     data=action_text,
                    #     existing_embeddings=new_message_embeddings,
                    #     metadata=deepcopy(metadata),
                    # )
                    # returned_memories.append(
                    #     {
                    #         "id": temp_uuid_mapping[mem_id_str],
                    #         "memory": action_text,
                    #         "event": event_type,
                    #         "previous_memory": resp.get("old_memory"),
                    #     }
                    # )
                # elif event_type == "DELETE":
                #     self._delete_memory(temp_uuid_mapping[mem_id_str])
                #     returned_memories.append(
                #         {
                #             "id": temp_uuid_mapping[mem_id_str],
                #             "memory": None,
                #             "event": event_type,
                #         }
                #     )
                #     # continue
                elif event_type == "NONE":
                    logging.info("NOOP for Memory.")
            # except Exception as e:F
                # logging.error(f"Error processing memory action: {resp}, Error: {e}")
        # except Exception as e:
        #     logging.error(f"Error iterating new_memories_with_actions: {e}")

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"},
        )
        _debug_print("returned_memories: ", returned_memories)
    

        return returned_memories




    # def _add_to_graph(self, messages, filters):
    #     print("Enter _add_to_graph", flush=True)
    #     print("self.enable_graph: ", self.enable_graph, flush=True)
    #     added_entities = []
    #     print("self.enable_graph: ", self.enable_graph)
    #     if self.enable_graph:
    #         if filters.get("user_id") is None:
    #             filters["user_id"] = "user"
            
    #         data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
    #         print("Adding to graph with data: ", data)
    #         print("Before graph.add", flush=True)
    #         added_entities = self.graph.add(data, filters)
    #         print("After graph.add", flush=True)
    #         print("Added entities to graph: ", added_entities)
    #     return added_entities
    def _add_to_graph(self, messages, filters):
        _debug_print("Enter _add_to_graph", flush=True)
        _debug_print("self.enable_graph: ", self.enable_graph, flush=True)
        added_entities = []

        if self.enable_graph and self.graph is not None:
            if filters.get("user_id") is None:
                filters["user_id"] = "user"

            data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
            _debug_print("Adding to graph with data: ", data)
            _debug_print("Before graph.add", flush=True)
            try:
                added_entities = self.graph.add(data, filters)
                _debug_print("After graph.add", flush=True)
                _debug_print("Added entities to graph: ", added_entities)
            except Exception as e:
                print(f"Error adding to graph: {e}")
        else:
            _debug_print("Graph is disabled or not initialized, skipping graph operations")
        
        return added_entities

    def get(self, memory_id):
        """
        Retrieve a memory by ID.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "sync"})
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory:
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload["data"],
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        return result_item

    def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ):
        """
        List all memories.

        Args:
            user_id (str, optional): user id
            agent_id (str, optional): agent id
            run_id (str, optional): run id
            filters (dict, optional): Additional custom key-value filters to apply to the search.
                These are merged with the ID-based scoping filters. For example,
                `filters={"actor_id": "some_user"}`.
            limit (int, optional): The maximum number of memories to return. Defaults to 100.

        Returns:
            dict: A dictionary containing a list of memories under the "results" key,
                  and potentially "relations" if graph store is enabled. For API v1.0,
                  it might return a direct list (see deprecation warning).
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`
        """

        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"}
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._get_all_from_vector_store, effective_filters, limit)
            future_graph_entities = (
                executor.submit(self.graph.get_all, effective_filters, limit) if self.enable_graph else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            all_memories_result = future_memories.result()
            graph_entities_result = future_graph_entities.result() if future_graph_entities else None

        if self.enable_graph:
            return {"results": all_memories_result, "relations": graph_entities_result}

        if self.api_version == "v1.0":
            warnings.warn(
                "The current get_all API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'` (which returns a dict with a 'results' key). "
                "The current format (direct list for v1.0) will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return all_memories_result
        else:
            return {"results": all_memories_result}

    def _get_all_from_vector_store(self, filters, limit):
        memories_result = self.vector_store.list(filters=filters, limit=limit)
        actual_memories = (
            memories_result[0]
            if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0
            else memories_result
        )

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload["data"],
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
    ):
        """
        Searches for memories based on a query
        Args:
            query (str): Query to search for.
            user_id (str, optional): ID of the user to search for. Defaults to None.
            agent_id (str, optional): ID of the agent to search for. Defaults to None.
            run_id (str, optional): ID of the run to search for. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to 100.
            filters (dict, optional): Filters to apply to the search. Defaults to None..
            threshold (float, optional): Minimum score for a memory to be included in the results. Defaults to None.

        Returns:
            dict: A dictionary containing the search results, typically under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`
        """
        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )
        _debug_print("effective_filters in searchL ", effective_filters)
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")

        _debug_print("here1 ")
        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "sync",
                "threshold": threshold,
            },
        )
        _debug_print("here2")
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        # if True:


            # print("here3")
        #     # future_memories = executor.submit(self._search_vector_store, query, effective_filters, limit, threshold)
        #     future_memories = self.executor.submit(self._search_vector_store, query, effective_filters, limit, threshold)



        #     # future_graph_entities = (
        #     #     executor.submit(self.graph.search, query, effective_filters, limit) if self.enable_graph else None
        #     # )
        #     if self.enable_graph and self.graph is not None:
        #         future_graph_entities = self.executor.submit(self.graph.search, query, effective_filters, limit)
        #     else:
        #         future_graph_entities = None
        #     # concurrent.futures.wait(
        #     #     [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
        #     # )
        #     if future_graph_entities:
        #         concurrent.futures.wait([future_memories, future_graph_entities])
        #         graph_entities = future_graph_entities.result()
        #     else:
        #         concurrent.futures.wait([future_memories])
        #         graph_entities = None
        #     # concurrent.futures.wait(
        #         # [future_memories]
        #     # )
        #     print("future memories", future_memories)
        #     original_memories = future_memories.result()
        #     print("original_memories", original_memories)
            
        #     graph_entities = future_graph_entities.result() if future_graph_entities else None

        
        # # return {"results": original_memories}
    
        # if self.enable_graph:
        #     return {"results": original_memories, "relations": graph_entities}

        # return {"results": original_memories}

        # if self.api_version == "v1.0":
        #     warnings.warn(
        #         "The current search API output format is deprecated. "
        #         "To use the latest format, set `api_version='v1.1'`. "
        #         "The current format will be removed in mem0ai 1.1.0 and later versions.",
        #         category=DeprecationWarning,
        #         stacklevel=2,
        #     )
        #     return {"results": original_memories}
        # else:
        #     return {"results": original_memories}
        # ------------------- 去掉 futures，直接同步调用 -------------------
        original_memories = self._search_vector_store(query, effective_filters, limit, threshold)
        _debug_print("original_memories", original_memories)

        if self.enable_graph and self.graph is not None:
            graph_entities = self.graph.search(query, effective_filters, limit)
            _debug_print("searched graph_entities", graph_entities)
        else:
            graph_entities = None

        # ---------------------------------------------------------------

        if self.enable_graph:
            return {"results": original_memories, "relations": graph_entities}

        return {"results": original_memories}

        if self.api_version == "v1.0":
            warnings.warn(
                "The current search API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return {"results": original_memories}
        else:
            return {"results": original_memories}

    def _search_vector_store(self, query, filters, limit, threshold: Optional[float] = None):
        embeddings = self.embedding_model.embed(query, "search")
        _debug_print("==== DEBUG BEFORE SEARCH ====")
        _debug_print("Query:", query)
        _debug_print("Embeddings shape:", np.array(embeddings).shape)
        # print("First 5 dims of embedding:", embeddings[:5])
        _debug_print("Search filters:", filters)
        _debug_print("Filter type:", type(filters))
        # print("Embeddings:", embeddings[:5])
        # print("Vector store count:", self.vector_store.count())
        _debug_print("Vector store count:", self.vector_store.index.ntotal)
        _debug_print("sellf.vector+store:",  dir(self.vector_store))
        _debug_print("Vector index size:", self.vector_store.index.ntotal)
        _debug_print("Vector store type:", type(self.vector_store))
        _debug_print("\n=== Documents in docstore ===")
        # for i, doc in enumerate(self.vector_store.docstore.values()):
        #     try:
        #         print(f"[{i}] Metadata keys:", list(doc.metadata.keys()))
        #         print(f"[{i}] Metadata:", doc.metadata)
        #     except Exception as e:
        #         print(f"[{i}] Error reading metadata: {e}")
        if MEM0_DEBUG_SEARCH:
            with open("docstore_dump1.txt", "w", encoding="utf-8") as f:
                for i, doc in enumerate(self.vector_store.docstore.values()):
                    f.write(f"=== Document {i} ===\n")
                    f.write(json.dumps(doc, indent=2, ensure_ascii=False, default=str))
                    f.write("\n\n")
            with open("docstore_dump.json", "w", encoding="utf-8") as f:
                json.dump(self.vector_store.docstore, f, indent=2, ensure_ascii=False, default=str)

        _debug_print("================================\n")
        for i, doc in enumerate(self.vector_store.docstore.values()):
            if i<5:
                _debug_print(f"[{i}] Metadata:", doc.get("metadata"))
                _debug_print(f"[{i}] Page content:", doc.get("page_content", "")[:100], "...")
        _debug_print("Search filters:", filters)
        _debug_print("Threshold:", threshold)
        _debug_print("==============================")

        # print(f"embeddings type: {type(embeddings)}, value: {embeddings}")
        memories = self.vector_store.search(query=query, vectors=embeddings, limit=limit, filters=filters)

        _debug_print("memories", memories)

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        original_memories = []
        for mem in memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload["data"],
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
                score=mem.score,
            ).model_dump()

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            if threshold is None or mem.score >= threshold:
                original_memories.append(memory_item_dict)

        return original_memories

    def update(self, memory_id, data):
        """
        Update a memory by ID.

        Args:
            memory_id (str): ID of the memory to update.
            data (dict): Data to update the memory with.

        Returns:
            dict: Updated memory.
        """
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "sync"})

        existing_embeddings = {data: self.embedding_model.embed(data, "update")}

        self._update_memory(memory_id, data, existing_embeddings)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "sync"})
        self._delete_memory(memory_id)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"})
        memories = self.vector_store.list(filters=filters)[0]
        for memory in memories:
            self._delete_memory(memory.id)

        logger.info(f"Deleted {len(memories)} memories")

        if self.enable_graph:
            self.graph.delete_all(filters)

        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id):
        """
        Get the history of changes for a memory by ID.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "sync"})
        return self.db.get_history(memory_id)

    def _create_memory(self, data, existing_embeddings, metadata=None):
        logging.debug(f"Creating memory with {data=}")
        # if data in existing_embeddings:
        #     embeddings = existing_embeddings[data]
        # else:
        #     embeddings = self.embedding_model.embed(data, memory_action="add")
        
        if isinstance(existing_embeddings, dict):
            embeddings = existing_embeddings.get(data)
            if embeddings is None:
                embeddings = self.embedding_model.embed(data, memory_action="add")
        else:
            embeddings = existing_embeddings 
        if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2 and embeddings.shape[0] == 1:
            embeddings = embeddings[0]
        memory_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata["data"] = data
        metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[metadata],
        )
        _debug_print(f"[DEBUG] Inserted new memory with ID: {memory_id}")
        _debug_print(f"[DEBUG] Embedding shape: {embeddings.shape}")
        _debug_print(f"[DEBUG] Metadata: {metadata}")
        _debug_print(f"[DEBUG] Total vectors in store now: {self.vector_store.index.ntotal}")
        # results = self.vector_store.search(
        #     query = "nouse",
        #     vectors=[embeddings],
        #     limit=10,
        #     filters=None,
        # )
        # print("Search results:", results)
        # print("self.vector_store.index.ntotal", self.vector_store.index.ntotal)
        # 








        self.db.add_history(
            memory_id,
            None,
            data,
            "ADD",
            created_at=metadata.get("created_at"),
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
        )
        capture_event("mem0._create_memory", self, {"memory_id": memory_id, "sync_type": "sync"})
        return memory_id

    def _create_procedural_memory(self, messages, metadata=None, prompt=None):
        """
        Create a procedural memory

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]

        try:
            procedural_memory = self.llm.generate_response(messages=parsed_messages)
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata["memory_type"] = MemoryType.PROCEDURAL.value
        embeddings = self.embedding_model.embed(procedural_memory, memory_action="add")
        memory_id = self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "sync"})

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = deepcopy(metadata) if metadata is not None else {}

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        if "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]
        if "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, "update")
        if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2 and embeddings.shape[0] == 1:
            embeddings = embeddings[0]
        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        capture_event("mem0._update_memory", self, {"memory_id": memory_id, "sync_type": "sync"})
        return memory_id

    def _delete_memory(self, memory_id):
        logging.info(f"Deleting memory with {memory_id=}")
        existing_memory = self.vector_store.get(vector_id=memory_id)
        prev_value = existing_memory.payload["data"]
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(
            memory_id,
            prev_value,
            None,
            "DELETE",
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )
        capture_event("mem0._delete_memory", self, {"memory_id": memory_id, "sync_type": "sync"})
        return memory_id

    def reset(self):
        """
        Reset the memory store by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")
        if hasattr(self.vector_store, "docstore"):
            self.vector_store.docstore.clear()
        if hasattr(self.vector_store, "index_to_id"):
            self.vector_store.index_to_id.clear()
        if hasattr(self.db, "connection") and self.db.connection:
            self.db.connection.execute("DROP TABLE IF EXISTS history")
            self.db.connection.close()


        if self.enable_graph and self.graph is not None:
            try:
                self.graph.reset()
                print("Graph store reset (Neo4j cleared).")
            except Exception as e:
                print(f"Graph reset failed: {e}")

        self.db = SQLiteManager(self.config.history_db_path)

        if hasattr(self.vector_store, "reset"):
            self.vector_store = VectorStoreFactory.reset(self.vector_store)
        else:
            logger.warning("Vector store does not support reset. Skipping.")
            if hasattr(self.vector_store, "client"):
                try:
                    self.vector_store.client.delete_collection(collection_name=self.vector_store.collection_name)
                    logger.info(f"Deleted Qdrant collection {self.vector_store.collection_name}")
                except Exception as e:
                    logger.warning(f"Failed to delete Qdrant collection: {e}")
            self.vector_store.delete_col()
            self.vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, self.config.vector_store.config
            )
        capture_event("mem0.reset", self, {"sync_type": "sync"})

    def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")


class AsyncMemory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version

        self.enable_graph = False

        if self.config.graph_store.config:
            from mem0.memory.graph_memory import MemoryGraph

            self.graph = MemoryGraph(self.config)
            self.enable_graph = True
        else:
            self.graph = None

        capture_event("mem0.init", self, {"sync_type": "async"})

    @classmethod
    async def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "graph_store" in config_dict:
            if "vector_store" not in config_dict and "embedder" in config_dict:
                config_dict["vector_store"] = {}
                config_dict["vector_store"]["config"] = {}
                config_dict["vector_store"]["config"]["embedding_model_dims"] = config_dict["embedder"]["config"][
                    "embedding_dims"
                ]
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    async def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
        llm=None,
    ):
        """
        Create a new memory asynchronously.

        Args:
            messages (str or List[Dict[str, str]]): Messages to store in the memory.
            user_id (str, optional): ID of the user creating the memory.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): Whether to infer the memories. Defaults to True.
            memory_type (str, optional): Type of memory to create. Defaults to None.
                                         Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.
            llm (BaseChatModel, optional): LLM class to use for generating procedural memories. Defaults to None. Useful when user is using LangChain ChatModel.
        Returns:
            dict: A dictionary containing the result of the memory addition operation.
        """

        _debug_print("used here2=============================================")
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_metadata=metadata
        )

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise ValueError(
                f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise ValueError("messages must be str, dict, or list[dict]")

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = await self._create_procedural_memory(
                messages, metadata=processed_metadata, prompt=prompt, llm=llm
            )
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        vector_store_task = asyncio.create_task(
            self._add_to_vector_store(messages, processed_metadata, effective_filters, infer)
        )
        graph_task = asyncio.create_task(self._add_to_graph(messages, effective_filters))

        vector_store_result, graph_result = await asyncio.gather(vector_store_task, graph_task)

        if self.api_version == "v1.0":
            warnings.warn(
                "The current add API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return vector_store_result

        if self.enable_graph:
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }

        return {"results": vector_store_result}

    async def _add_to_vector_store(
        self,
        messages: list,
        metadata: dict,
        effective_filters: dict,
        infer: bool,
    ):
        if not infer:
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format (async): {message_dict}")
                    continue

                if message_dict["role"] == "system":
                    continue

                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                msg_content = message_dict["content"]
                msg_embeddings = await asyncio.to_thread(self.embedding_model.embed, msg_content, "add")
                mem_id = await self._create_memory(msg_content, msg_embeddings, per_msg_meta)

                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        parsed_messages = parse_messages(messages)
        if self.config.custom_fact_extraction_prompt:
            system_prompt = self.config.custom_fact_extraction_prompt
            user_prompt = f"Input:\n{parsed_messages}"
        else:
            system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages)

        response = await asyncio.to_thread(
            self.llm.generate_response,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
        )
        try:
            response = remove_code_blocks(response)
            new_retrieved_facts = json.loads(response)["facts"]
        except Exception as e:
            logging.error(f"Error in new_retrieved_facts: {e}")
            new_retrieved_facts = []
        
        if not new_retrieved_facts:
            logger.debug("No new facts retrieved from input. Skipping memory update LLM call.")

        retrieved_old_memory = []
        new_message_embeddings = {}

        async def process_fact_for_search(new_mem_content):
            embeddings = await asyncio.to_thread(self.embedding_model.embed, new_mem_content, "add")
            new_message_embeddings[new_mem_content] = embeddings
            existing_mems = await asyncio.to_thread(
                self.vector_store.search,
                query=new_mem_content,
                vectors=embeddings,
                limit=5,
                filters=effective_filters,  # 'filters' is query_filters_for_inference
            )
            return [{"id": mem.id, "text": mem.payload["data"]} for mem in existing_mems]

        search_tasks = [process_fact_for_search(fact) for fact in new_retrieved_facts]
        search_results_list = await asyncio.gather(*search_tasks)
        for result_group in search_results_list:
            retrieved_old_memory.extend(result_group)

        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())
        logging.info(f"Total existing memories: {len(retrieved_old_memory)}")
        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)

        if new_retrieved_facts:
            function_calling_prompt = get_update_memory_messages(
                retrieved_old_memory, new_retrieved_facts, self.config.custom_update_memory_prompt
            )
            try:
                response = await asyncio.to_thread(
                    self.llm.generate_response,
                    messages=[{"role": "user", "content": function_calling_prompt}],
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                logging.error(f"Error in new memory actions response: {e}")
                response = ""
            try:
                response = remove_code_blocks(response)
                new_memories_with_actions = json.loads(response)
            except Exception as e:
                logging.error(f"Invalid JSON response: {e}")
                new_memories_with_actions = {
                    "memory": [
                        {
                            "id": str(idx),
                            "text": fact,
                            "event": "ADD",
                            "timestamp": metadata.get("timestamp", ""),
                            "speaker": metadata.get("speaker", ""),
                        }
                        for idx, fact in enumerate(new_retrieved_facts)
                    ]
                }

        returned_memories = []
        try:
            memory_tasks = []
            for resp in _normalize_memory_actions(new_memories_with_actions):
                logging.info(resp)
                try:
                    action_text = resp.get("text")
                    if not action_text:
                        continue
                    event_type = resp.get("event")

                    if event_type == "ADD":
                        task = asyncio.create_task(
                            self._create_memory(
                                data=action_text,
                                existing_embeddings=new_message_embeddings,
                                metadata=deepcopy(metadata),
                            )
                        )
                        memory_tasks.append((task, resp, "ADD", None))
                    elif event_type == "UPDATE":
                        task = asyncio.create_task(
                            self._update_memory(
                                memory_id=temp_uuid_mapping[resp["id"]],
                                data=action_text,
                                existing_embeddings=new_message_embeddings,
                                metadata=deepcopy(metadata),
                            )
                        )
                        memory_tasks.append((task, resp, "UPDATE", temp_uuid_mapping[resp["id"]]))
                    elif event_type == "DELETE":
                        task = asyncio.create_task(self._delete_memory(memory_id=temp_uuid_mapping[resp.get("id")]))
                        memory_tasks.append((task, resp, "DELETE", temp_uuid_mapping[resp.get("id")]))
                    elif event_type == "NONE":
                        logging.info("NOOP for Memory (async).")
                except Exception as e:
                    logging.error(f"Error processing memory action (async): {resp}, Error: {e}")

            for task, resp, event_type, mem_id in memory_tasks:
                try:
                    result_id = await task
                    if event_type == "ADD":
                        returned_memories.append({"id": result_id, "memory": resp.get("text"), "event": event_type})
                    elif event_type == "UPDATE":
                        returned_memories.append(
                            {
                                "id": mem_id,
                                "memory": resp.get("text"),
                                "event": event_type,
                                "previous_memory": resp.get("old_memory"),
                            }
                        )
                    elif event_type == "DELETE":
                        returned_memories.append({"id": mem_id, "memory": resp.get("text"), "event": event_type})
                except Exception as e:
                    logging.error(f"Error awaiting memory task (async): {e}")
        except Exception as e:
            logging.error(f"Error in memory processing loop (async): {e}")

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"},
        )
        return returned_memories

    async def _add_to_graph(self, messages, filters):
        added_entities = []
        if self.enable_graph:
            if filters.get("user_id") is None:
                filters["user_id"] = "user"

            data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
            added_entities = await asyncio.to_thread(self.graph.add, data, filters)

        return added_entities

    async def get(self, memory_id):
        """
        Retrieve a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "async"})
        memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        if not memory:
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload["data"],
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        return result_item

    async def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ):
        """
        List all memories.

         Args:
             user_id (str, optional): user id
             agent_id (str, optional): agent id
             run_id (str, optional): run id
             filters (dict, optional): Additional custom key-value filters to apply to the search.
                 These are merged with the ID-based scoping filters. For example,
                 `filters={"actor_id": "some_user"}`.
             limit (int, optional): The maximum number of memories to return. Defaults to 100.

         Returns:
             dict: A dictionary containing a list of memories under the "results" key,
                   and potentially "relations" if graph store is enabled. For API v1.0,
                   it might return a direct list (see deprecation warning).
                   Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`
        """

        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "When 'conversation_id' is not provided (classic mode), "
                "at least one of 'user_id', 'agent_id', or 'run_id' must be specified for get_all."
            )

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"}
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._get_all_from_vector_store, effective_filters, limit)
            future_graph_entities = (
                executor.submit(self.graph.get_all, effective_filters, limit) if self.enable_graph else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            all_memories_result = future_memories.result()
            graph_entities_result = future_graph_entities.result() if future_graph_entities else None

        if self.enable_graph:
            return {"results": all_memories_result, "relations": graph_entities_result}

        if self.api_version == "v1.0":
            warnings.warn(
                "The current get_all API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'` (which returns a dict with a 'results' key). "
                "The current format (direct list for v1.0) will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return all_memories_result
        else:
            return {"results": all_memories_result}

    async def _get_all_from_vector_store(self, filters, limit):
        memories_result = await asyncio.to_thread(self.vector_store.list, filters=filters, limit=limit)
        actual_memories = (
            memories_result[0]
            if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0
            else memories_result
        )

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload["data"],
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    async def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
    ):
        """
        Searches for memories based on a query
        Args:
            query (str): Query to search for.
            user_id (str, optional): ID of the user to search for. Defaults to None.
            agent_id (str, optional): ID of the agent to search for. Defaults to None.
            run_id (str, optional): ID of the run to search for. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to 100.
            filters (dict, optional): Filters to apply to the search. Defaults to None.
            threshold (float, optional): Minimum score for a memory to be included in the results. Defaults to None.

        Returns:
            dict: A dictionary containing the search results, typically under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`
        """

        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("at least one of 'user_id', 'agent_id', or 'run_id' must be specified ")

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "async",
                "threshold": threshold,
            },
        )

        vector_store_task = asyncio.create_task(self._search_vector_store(query, effective_filters, limit, threshold))

        graph_task = None
        if self.enable_graph:
            if hasattr(self.graph.search, "__await__"):  # Check if graph search is async
                graph_task = asyncio.create_task(self.graph.search(query, effective_filters, limit))
            else:
                graph_task = asyncio.create_task(asyncio.to_thread(self.graph.search, query, effective_filters, limit))

        if graph_task:
            original_memories, graph_entities = await asyncio.gather(vector_store_task, graph_task)
        else:
            original_memories = await vector_store_task
            graph_entities = None

        if self.enable_graph:
            return {"results": original_memories, "relations": graph_entities}

        if self.api_version == "v1.0":
            warnings.warn(
                "The current search API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return {"results": original_memories}
        else:
            return {"results": original_memories}

    async def _search_vector_store(self, query, filters, limit, threshold: Optional[float] = None):
        embeddings = await asyncio.to_thread(self.embedding_model.embed, query, "search")
        memories = await asyncio.to_thread(
            self.vector_store.search, query=query, vectors=embeddings, limit=limit, filters=filters
        )

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        original_memories = []
        for mem in memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload["data"],
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
                score=mem.score,
            ).model_dump()

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            if threshold is None or mem.score >= threshold:
                original_memories.append(memory_item_dict)

        return original_memories

    async def update(self, memory_id, data):
        """
        Update a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to update.
            data (dict): Data to update the memory with.

        Returns:
            dict: Updated memory.
        """
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "async"})

        embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")
        existing_embeddings = {data: embeddings}

        await self._update_memory(memory_id, data, existing_embeddings)
        return {"message": "Memory updated successfully!"}

    async def delete(self, memory_id):
        """
        Delete a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "async"})
        await self._delete_memory(memory_id)
        return {"message": "Memory deleted successfully!"}

    async def delete_all(self, user_id=None, agent_id=None, run_id=None):
        """
        Delete all memories asynchronously.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"})
        memories = await asyncio.to_thread(self.vector_store.list, filters=filters)

        delete_tasks = []
        for memory in memories[0]:
            delete_tasks.append(self._delete_memory(memory.id))

        await asyncio.gather(*delete_tasks)

        logger.info(f"Deleted {len(memories[0])} memories")

        if self.enable_graph:
            await asyncio.to_thread(self.graph.delete_all, filters)

        return {"message": "Memories deleted successfully!"}

    async def history(self, memory_id):
        """
        Get the history of changes for a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "async"})
        return await asyncio.to_thread(self.db.get_history, memory_id)

    async def _create_memory(self, data, existing_embeddings, metadata=None):
        logging.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, memory_action="add")

        memory_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata["data"] = data
        metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        await asyncio.to_thread(
            self.vector_store.insert,
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[metadata],
        )

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            None,
            data,
            "ADD",
            created_at=metadata.get("created_at"),
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
        )

        capture_event("mem0._create_memory", self, {"memory_id": memory_id, "sync_type": "async"})
        return memory_id

    async def _create_procedural_memory(self, messages, metadata=None, llm=None, prompt=None):
        """
        Create a procedural memory asynchronously

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            llm (llm, optional): LLM to use for the procedural memory creation. Defaults to None.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        try:
            from langchain_core.messages.utils import (
                convert_to_messages,  # type: ignore
            )
        except Exception:
            logger.error(
                "Import error while loading langchain-core. Please install 'langchain-core' to use procedural memory."
            )
            raise

        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {"role": "user", "content": "Create procedural memory of the above conversation."},
        ]

        try:
            if llm is not None:
                parsed_messages = convert_to_messages(parsed_messages)
                response = await asyncio.to_thread(llm.invoke, input=parsed_messages)
                procedural_memory = response.content
            else:
                procedural_memory = await asyncio.to_thread(self.llm.generate_response, messages=parsed_messages)
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata["memory_type"] = MemoryType.PROCEDURAL.value
        embeddings = await asyncio.to_thread(self.embedding_model.embed, procedural_memory, memory_action="add")
        memory_id = await self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "async"})

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    async def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = deepcopy(metadata) if metadata is not None else {}

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        if "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]

        if "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")

        await asyncio.to_thread(
            self.vector_store.update,
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        capture_event("mem0._update_memory", self, {"memory_id": memory_id, "sync_type": "async"})
        return memory_id

    async def _delete_memory(self, memory_id):
        logging.info(f"Deleting memory with {memory_id=}")
        existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        prev_value = existing_memory.payload["data"]

        await asyncio.to_thread(self.vector_store.delete, vector_id=memory_id)
        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            None,
            "DELETE",
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )

        capture_event("mem0._delete_memory", self, {"memory_id": memory_id, "sync_type": "async"})
        return memory_id

    async def reset(self):
        """
        Reset the memory store asynchronously by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")
        await asyncio.to_thread(self.vector_store.delete_col)

        gc.collect()

        if hasattr(self.vector_store, "client") and hasattr(self.vector_store.client, "close"):
            await asyncio.to_thread(self.vector_store.client.close)

        if hasattr(self.db, "connection") and self.db.connection:
            await asyncio.to_thread(lambda: self.db.connection.execute("DROP TABLE IF EXISTS history"))
            await asyncio.to_thread(self.db.connection.close)

        self.db = SQLiteManager(self.config.history_db_path)

        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        capture_event("mem0.reset", self, {"sync_type": "async"})

    async def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")
