import hashlib
import json
import os
import threading
import time
from typing import Optional

from .config import MemoryConfig
from .memory import MemGASMemory

try:
    from ..interface import (
        PreservingCloseState,
        close_resource,
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
        close_resource,
        format_memory_prompt,
        normalize_chunk,
        resolve_context_char_budget,
        resolve_state_root,
        resolve_top_k,
        safe_user_id,
        select_endpoint,
    )


class MemGASMemorySystem:
    """MemoryArena adapter that keeps MemGAS add and retrieval unchanged."""

    def __init__(self, user_id: Optional[str] = None, top_k: int = 10):
        self.user_id = str(user_id or f"memoryarena_{time.time_ns()}")
        self.top_k = resolve_top_k(top_k, "MEMGAS_RETRIEVAL_TOP_K")
        self.context_char_budget = resolve_context_char_budget(
            "MEMGAS_CONTEXT_CHAR_BUDGET"
        )
        self._lock = threading.RLock()
        self._close_state = PreservingCloseState()

        storage_root = resolve_state_root(
            "memgas", "MEMGAS_MEMORYARENA_DATA_ROOT"
        )
        self.storage_dir = storage_root / safe_user_id(self.user_id, limit=160)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.storage_dir / "added_chunks.json"
        self._added_hashes = self._load_added_hashes()

        config = MemoryConfig(
            storage_dir=str(self.storage_dir),
            embedder="openai",
            device="cpu",
            batch_size=int(os.getenv("MEMGAS_EMBEDDING_BATCH_SIZE", "64")),
            embedder_api_key=os.getenv("MEMGAS_EMBEDDING_API_KEY")
            or os.getenv("MEMORYARENA_EMBEDDING_API_KEY", "EMPTY"),
            embedder_base_url=select_endpoint(
                self.user_id,
                kind="embedding",
                method_list_env="MEMGAS_EMBEDDING_BASE_URLS",
                method_single_env="MEMGAS_EMBEDDING_BASE_URL",
            ),
            embedder_model=os.getenv("MEMGAS_EMBEDDING_MODEL")
            or os.getenv(
                "MEMORYARENA_EMBEDDING_MODEL",
                "/path/to/local/all-MiniLM-L6-v2",
            ),
            embedder_max_tokens=int(os.getenv("MEMGAS_EMBEDDING_MAX_TOKENS", "256")),
            llm_model=os.getenv("MEMGAS_LLM_MODEL")
            or os.getenv("MEMORYARENA_LLM_MODEL", "Qwen3.5-9B"),
            llm_provider="vllm",
            llm_api_key=os.getenv("MEMGAS_LLM_API_KEY")
            or os.getenv("MEMORYARENA_LLM_API_KEY", "EMPTY"),
            llm_base_url=select_endpoint(
                self.user_id,
                kind="llm",
                method_list_env="MEMGAS_LLM_BASE_URLS",
                method_single_env="MEMGAS_LLM_BASE_URL",
            ),
            llm_max_tokens=int(os.getenv("MEMGAS_LLM_MAX_TOKENS", "500")),
            llm_temperature=float(os.getenv("MEMGAS_LLM_TEMPERATURE", "0")),
            llm_max_retries=int(os.getenv("MEMGAS_LLM_MAX_RETRIES", "5")),
            llm_retry_wait_sec=float(os.getenv("MEMGAS_LLM_RETRY_WAIT_SEC", "2")),
            llm_context_window=int(os.getenv("MEMGAS_LLM_CONTEXT_WINDOW", "16384")),
            llm_prompt_token_buffer=int(os.getenv("MEMGAS_PROMPT_TOKEN_BUFFER", "128")),
            default_mode="memgas",
            auto_save=True,
        )
        self.memory = MemGASMemory(config)

    def _load_added_hashes(self):
        if not self.journal_path.exists():
            return set()
        try:
            payload = json.loads(self.journal_path.read_text(encoding="utf-8"))
            return set(payload.get("hashes", []))
        except (OSError, ValueError, TypeError):
            return set()

    def _save_added_hashes(self):
        tmp_path = self.journal_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps({"hashes": sorted(self._added_hashes)}, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, self.journal_path)

    def add_chunk(self, chunk: str):
        text = normalize_chunk(chunk)
        if text is None:
            return {"stored": False, "skipped": True, "characters": 0}
        chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock:
            self._close_state.ensure_open(f"MemGAS memory {self.user_id}")
            if chunk_hash in self._added_hashes:
                return {"stored": False, "duplicate": True, "characters": len(text)}
            memory_id = self.memory.add(
                session=[text],
                conversation_id=self.user_id,
                metadata={"source": "memoryarena", "task_id": self.user_id},
            )
            self._added_hashes.add(chunk_hash)
            self._save_added_hashes()
        return {"stored": True, "characters": len(text), "memory_id": memory_id}

    def wrap_user_prompt(self, prompt: str) -> str:
        with self._lock:
            self._close_state.ensure_open(f"MemGAS memory {self.user_id}")
            hits = self.memory.retrieve(
                str(prompt),
                topk=self.top_k,
                conversation_id=self.user_id,
                mode="memgas",
            )

        entries = []
        for hit in hits:
            session_text = "\n".join(hit.get("session") or [])
            summary = str(hit.get("summary") or "")
            keywords = "; ".join(hit.get("keywords") or [])
            item = (
                f"Session Content:\n{session_text}\n"
                f"Session Summary:\n{summary}\n"
                f"Session Keywords:\n{keywords}"
            )
            entries.append(item)
        return format_memory_prompt(
            prompt,
            entries,
            char_budget=self.context_char_budget,
        )

    def _release_resources(self):
        self.memory.save()
        seen = set()
        resources = close_resource(self.memory, "memgas", seen=seen)
        if resources:
            return resources
        resources.extend(
            close_resource(
                getattr(self.memory.embedder, "client", None),
                "memgas.embedder.client",
                seen=seen,
            )
        )
        resources.extend(
            close_resource(
                getattr(self.memory.llm, "client", None),
                "memgas.llm.client",
                seen=seen,
            )
        )
        return resources

    def close(self, completed: bool = False):
        return self._close_state.close(completed, self._release_resources)


def create_backend(*, user_id=None, top_k=10):
    """Create MemGAS through the common MemoryArena factory."""

    return MemGASMemorySystem(user_id=user_id, top_k=top_k)
