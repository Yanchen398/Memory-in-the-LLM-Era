import asyncio
import copy
import hashlib
import inspect
import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.openai import OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
)

from .main import (
    TruncatingOpenAIEmbedder,
    build_answer_context,
    build_qwen_openai_client,
    extract_retrieved_facts,
)

try:
    from ..interface import (
        PreservingCloseState,
        format_memory_prompt,
        normalize_chunk,
        resolve_context_char_budget,
        resolve_state_root,
        resolve_top_k,
        safe_user_id,
        select_endpoint,
    )
except ImportError:
    from Method_memoryarena.interface import (
        PreservingCloseState,
        format_memory_prompt,
        normalize_chunk,
        resolve_context_char_budget,
        resolve_state_root,
        resolve_top_k,
        safe_user_id,
        select_endpoint,
    )


_ASYNC_LOOP = asyncio.new_event_loop()
_LOOP_READY = threading.Event()
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_READY = False
_OPERATION_SEMAPHORE = threading.BoundedSemaphore(
    max(1, int(os.getenv("ZEP_MAX_CONCURRENT_OPERATIONS", "8")))
)


def _run_loop() -> None:
    asyncio.set_event_loop(_ASYNC_LOOP)
    _LOOP_READY.set()
    _ASYNC_LOOP.run_forever()


_LOOP_THREAD = threading.Thread(
    target=_run_loop,
    name="memoryarena-zep-async-loop",
    daemon=True,
)
_LOOP_THREAD.start()
_LOOP_READY.wait()


def _run(coroutine, timeout: Optional[float] = None):
    wait_seconds = timeout or float(os.getenv("ZEP_OPERATION_TIMEOUT_SECONDS", "1800"))
    future = asyncio.run_coroutine_threadsafe(coroutine, _ASYNC_LOOP)
    try:
        return future.result(timeout=wait_seconds)
    except BaseException:
        future.cancel()
        raise


def _select_base_url(user_id: str) -> str:
    return select_endpoint(
        user_id,
        kind="llm",
        method_list_env="ZEP_MEMORYARENA_LLM_BASE_URLS",
        method_single_env="ZEP_LLM_BASE_URL",
    )


async def _create_graphiti(base_url: str, user_id: str) -> Graphiti:
    api_key = os.getenv("ZEP_LLM_API_KEY") or os.getenv(
        "MEMORYARENA_LLM_API_KEY", "EMPTY"
    )
    model = os.getenv("ZEP_LLM_MODEL") or os.getenv(
        "MEMORYARENA_LLM_MODEL", "Qwen3.5-9B"
    )
    max_output_tokens = int(os.getenv("ZEP_LLM_MAX_COMPLETION_TOKENS", "4096"))
    config = LLMConfig(
        api_key=api_key,
        model=model,
        small_model=model,
        base_url=base_url,
        max_tokens=max_output_tokens,
    )
    llm_client = OpenAIGenericClient(
        config=config,
        client=build_qwen_openai_client(api_key, base_url),
        max_tokens=max_output_tokens,
    )
    llm_client.MAX_RETRIES = int(os.getenv("ZEP_LLM_JSON_MAX_RETRIES", "5"))
    return Graphiti(
        os.getenv("ZEP_NEO4J_URI", "bolt://localhost:7687"),
        os.getenv("ZEP_NEO4J_USER", "neo4j"),
        os.getenv("ZEP_NEO4J_PASSWORD", "neo4jneo4j"),
        llm_client=llm_client,
        embedder=TruncatingOpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                embedding_model=os.getenv("ZEP_EMBEDDING_MODEL")
                or os.getenv(
                    "MEMORYARENA_EMBEDDING_MODEL",
                    "/path/to/local/all-MiniLM-L6-v2",
                ),
                api_key=os.getenv("ZEP_EMBEDDING_API_KEY")
                or os.getenv("MEMORYARENA_EMBEDDING_API_KEY", "EMPTY"),
                base_url=select_endpoint(
                    user_id,
                    kind="embedding",
                    method_list_env="ZEP_EMBEDDING_BASE_URLS",
                    method_single_env="ZEP_EMBEDDING_BASE_URL",
                ),
                embedding_dim=int(os.getenv("ZEP_EMBEDDING_DIM", "384")),
            )
        ),
        cross_encoder=OpenAIRerankerClient(
            config=config,
            client=build_qwen_openai_client(api_key, base_url),
        ),
        max_coroutines=int(os.getenv("ZEP_GRAPHITI_MAX_COROUTINES", "4")),
    )


def _ensure_schema(graphiti: Graphiti) -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return
        _run(graphiti.build_indices_and_constraints())
        _SCHEMA_READY = True


class ZepMemorySystem:
    """MemoryArena adapter for the local Graphiti-based Zep implementation."""

    def __init__(self, user_id: Optional[str] = None, top_k: int = 10):
        self.user_id = str(user_id or f"memoryarena_{time.time_ns()}")
        self.safe_user_id = safe_user_id(self.user_id)
        self.top_k = resolve_top_k(top_k, "ZEP_RETRIEVAL_TOP_K")
        self.context_char_budget = resolve_context_char_budget(
            "ZEP_CONTEXT_CHAR_BUDGET"
        )
        self.base_url = _select_base_url(self.user_id)
        run_namespace = safe_user_id(
            os.getenv("ZEP_RUN_NAMESPACE")
            or os.getenv("MEMORYARENA_RUN_ID", "default")
        )
        digest = hashlib.sha256(self.user_id.encode("utf-8")).hexdigest()[:24]
        self.group_id = f"memoryarena_zep_{run_namespace}_{digest}"
        self._lock = threading.RLock()
        self._close_state = PreservingCloseState()
        self._episode_count = 0
        self._added_hashes = set()

        storage_root = resolve_state_root(
            "zep", "ZEP_MEMORYARENA_DATA_ROOT"
        )
        self.storage_dir = storage_root / self.safe_user_id
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.storage_dir / "added_chunks.json"
        recovered = self._load_journal()

        with _OPERATION_SEMAPHORE:
            self.graphiti = _run(_create_graphiti(self.base_url, self.user_id))
            _ensure_schema(self.graphiti)

        self.search_config = copy.deepcopy(
            COMBINED_HYBRID_SEARCH_CROSS_ENCODER
        )
        self.search_config.limit = self.top_k
        if not recovered:
            self._save_journal()
        print(
            f"Zep initialized user_id={self.user_id} group_id={self.group_id} "
            f"recovered={recovered} "
            f"top_k={self.top_k} llm={self.base_url}",
            flush=True,
        )

    def _load_journal(self) -> bool:
        if not self.journal_path.exists():
            return False
        try:
            payload = json.loads(self.journal_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"Cannot recover Zep journal {self.journal_path}: {exc}"
            ) from exc
        if payload.get("user_id") != self.user_id:
            raise RuntimeError("Zep journal user_id does not match requested user")
        if payload.get("group_id") != self.group_id:
            raise RuntimeError("Zep journal group_id does not match requested namespace")
        hashes = payload.get("hashes", [])
        episode_count = payload.get("episode_count", 0)
        if not isinstance(hashes, list) or not all(
            isinstance(item, str) for item in hashes
        ):
            raise RuntimeError("Zep journal hashes must be a list of strings")
        if isinstance(episode_count, bool) or not isinstance(episode_count, int):
            raise RuntimeError("Zep journal episode_count must be an integer")
        if episode_count < 0:
            raise RuntimeError("Zep journal episode_count cannot be negative")
        self._added_hashes = set(hashes)
        self._episode_count = episode_count
        return True

    def _save_journal(self) -> None:
        payload = {
            "user_id": self.user_id,
            "group_id": self.group_id,
            "hashes": sorted(self._added_hashes),
            "episode_count": self._episode_count,
            "llm_base_url": self.base_url,
        }
        temporary = self.journal_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.journal_path)

    async def _add_episode(self, text: str, episode_number: int) -> None:
        await self.graphiti.add_episode(
            name=f"memoryarena_{episode_number:05d}",
            episode_body=text,
            source_description="MemoryArena interaction or task experience",
            reference_time=datetime.now(timezone.utc),
            source=EpisodeType.message,
            group_id=self.group_id,
            update_communities=False,
        )

    def add_chunk(self, chunk: str):
        text = normalize_chunk(chunk)
        if text is None:
            return {"stored": False, "skipped": True, "characters": 0}

        chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock:
            self._close_state.ensure_open(f"Zep memory {self.user_id}")
            if chunk_hash in self._added_hashes:
                return {
                    "stored": False,
                    "duplicate": True,
                    "characters": len(text),
                }

            episode_number = self._episode_count + 1
            with _OPERATION_SEMAPHORE:
                _run(self._add_episode(text, episode_number))
            self._added_hashes.add(chunk_hash)
            self._episode_count = episode_number
            self._save_journal()
            return {
                "stored": True,
                "characters": len(text),
                "episode_count": self._episode_count,
                "group_id": self.group_id,
            }

    def _format_context(self, prompt: str):
        if self._episode_count == 0:
            return ""

        with _OPERATION_SEMAPHORE:
            results = _run(
                self.graphiti.search_(
                    str(prompt),
                    config=self.search_config,
                    group_ids=[self.group_id],
                )
            )
        return build_answer_context(extract_retrieved_facts(results))

    def wrap_user_prompt(self, prompt: str) -> str:
        with self._lock:
            self._close_state.ensure_open(f"Zep memory {self.user_id}")
            context = self._format_context(str(prompt))
        return format_memory_prompt(
            prompt,
            [context] if context else [],
            char_budget=self.context_char_budget,
        )

    async def _close_async(self) -> None:
        result = self.graphiti.close()
        if inspect.isawaitable(result):
            await result

    def _release_resources(self):
        with self._lock:
            self._save_journal()
            with _OPERATION_SEMAPHORE:
                _run(self._close_async())
        return ["zep.graphiti"]

    def close(self, completed: bool = False):
        receipt = self._close_state.close(completed, self._release_resources)
        return {
            **receipt,
            "group_id": self.group_id,
            "group_deleted": False,
            "episode_count": self._episode_count,
        }


def create_backend(*, user_id=None, top_k=10):
    """Create Zep through the common MemoryArena factory."""

    return ZepMemorySystem(user_id=user_id, top_k=top_k)
