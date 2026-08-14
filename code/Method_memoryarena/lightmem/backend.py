import hashlib
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

_LOCAL_SOURCE = Path(__file__).resolve().parent / "src"
if str(_LOCAL_SOURCE) not in sys.path:
    sys.path.insert(0, str(_LOCAL_SOURCE))

from .main import build_lightmem_config

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


_LIGHTMEM_INIT_LOCK = threading.RLock()


class LightMemMemorySystem:
    """MemoryArena adapter that preserves the native LightMem pipeline."""

    def __init__(self, user_id: Optional[str] = None, top_k: int = 10):
        from lightmem.memory.lightmem import LightMemory

        self.user_id = str(user_id or f"memoryarena_{time.time_ns()}")
        self.top_k = resolve_top_k(top_k, "LIGHTMEM_RETRIEVAL_TOP_K")
        self.context_char_budget = resolve_context_char_budget(
            "LIGHTMEM_CONTEXT_CHAR_BUDGET"
        )
        safe_id = safe_user_id(self.user_id)
        storage_root = resolve_state_root(
            "lightmem", "LIGHTMEM_MEMORYARENA_DATA_ROOT"
        )
        self.storage_dir = storage_root / safe_id
        self.log_dir = self.storage_dir / "logs"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._journal_path = self.storage_dir / "added_chunks.json"
        self._added_hashes = self._load_added_hashes()
        self._lock = threading.RLock()
        self._close_state = PreservingCloseState()

        config = build_lightmem_config(
            collection_name=f"memoryarena_{safe_id}",
            storage_dir=str(self.storage_dir),
            log_dir=str(self.log_dir),
            llm_model=os.getenv("LIGHTMEM_LLM_MODEL")
            or os.getenv("MEMORYARENA_LLM_MODEL", "Qwen3.5-9B"),
            llm_api_key=os.getenv("LIGHTMEM_LLM_API_KEY")
            or os.getenv("MEMORYARENA_LLM_API_KEY", "EMPTY"),
            llm_base_url=select_endpoint(
                self.user_id,
                kind="llm",
                method_list_env="LIGHTMEM_LLM_BASE_URLS",
                method_single_env="LIGHTMEM_LLM_BASE_URL",
            ),
            llm_provider="openai",
            embedding_model_name=os.getenv("LIGHTMEM_EMBEDDING_MODEL")
            or os.getenv(
                "MEMORYARENA_EMBEDDING_MODEL",
                "/path/to/local/all-MiniLM-L6-v2",
            ),
            embedding_dim=384,
            embedding_device=os.getenv("LIGHTMEM_EMBEDDING_DEVICE", "cpu"),
            pre_compress=True,
            llmlingua_model_path=os.getenv(
                "LIGHTMEM_LLMLINGUA_MODEL",
                "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            ),
            compression_rate=float(os.getenv("LIGHTMEM_COMPRESSION_RATE", "0.6")),
            topic_segment=True,
            metadata_generate=True,
            text_summary=True,
            extraction_mode="flat",
        )
        with _LIGHTMEM_INIT_LOCK:
            self.memory = LightMemory.from_config(config)

    def _load_added_hashes(self):
        if not self._journal_path.exists():
            return set()
        try:
            data = json.loads(self._journal_path.read_text(encoding="utf-8"))
            return set(data.get("hashes", []))
        except (OSError, ValueError, TypeError):
            return set()

    def _save_added_hashes(self):
        tmp_path = self._journal_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps({"hashes": sorted(self._added_hashes)}, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, self._journal_path)

    def add_chunk(self, chunk: str):
        text = normalize_chunk(chunk)
        if text is None:
            return {"stored": False, "skipped": True, "characters": 0}
        chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock:
            self._close_state.ensure_open(f"LightMem memory {self.user_id}")
            if chunk_hash in self._added_hashes:
                return {"stored": False, "duplicate": True, "characters": len(text)}
            now = datetime.now().strftime("%Y/%m/%d (%a) %H:%M:%S")
            result = self.memory.add_memory(
                messages=[
                    {
                        "role": "user",
                        "content": text,
                        "speaker_id": "memoryarena_task",
                        "speaker_name": "MemoryArena",
                        "time_stamp": now,
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "speaker_id": "memoryarena_task",
                        "speaker_name": "MemoryArena",
                        "time_stamp": now,
                    },
                ],
                force_segment=True,
                force_extract=True,
            )
            self._added_hashes.add(chunk_hash)
            self._save_added_hashes()
        return {
            "stored": True,
            "characters": len(text),
            "api_call_nums": int((result or {}).get("api_call_nums", 0)),
        }

    def wrap_user_prompt(self, prompt: str) -> str:
        with self._lock:
            self._close_state.ensure_open(f"LightMem memory {self.user_id}")
            entries = self.memory.retrieve(str(prompt), limit=self.top_k)
        return format_memory_prompt(
            prompt,
            entries,
            char_budget=self.context_char_budget,
        )

    def _release_resources(self):
        seen = set()
        resources = close_resource(self.memory, "lightmem", seen=seen)
        if resources:
            return resources
        for component_name in (
            "manager",
            "text_embedder",
            "embedding_retriever",
            "summary_retriever",
            "context_retriever",
        ):
            component = getattr(self.memory, component_name, None)
            component_resources = close_resource(
                component,
                f"lightmem.{component_name}",
                seen=seen,
            )
            resources.extend(component_resources)
            if not component_resources:
                resources.extend(
                    close_resource(
                        getattr(component, "client", None),
                        f"lightmem.{component_name}.client",
                        seen=seen,
                    )
                )
        return resources

    def close(self, completed: bool = False):
        return self._close_state.close(completed, self._release_resources)


def create_backend(*, user_id=None, top_k=10):
    """Create LightMem through the common MemoryArena factory."""

    return LightMemMemorySystem(user_id=user_id, top_k=top_k)
