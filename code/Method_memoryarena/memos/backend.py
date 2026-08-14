import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

_LOCAL_SOURCE = Path(__file__).resolve().parent / "src"
if str(_LOCAL_SOURCE) not in sys.path:
    sys.path.insert(0, str(_LOCAL_SOURCE))

from memos.configs.mem_cube import GeneralMemCubeConfig
from memos.configs.mem_os import MOSConfig
from memos.mem_cube.general import GeneralMemCube
from memos.mem_os.main import MOS

from .configuration import (
    build_mem_cube_config,
    build_mos_config,
    build_runtime_config,
)

try:
    from ..interface import (
        PreservingCloseState,
        close_resource,
        format_memory_prompt,
        normalize_chunk,
        resolve_context_char_budget,
        resolve_state_root,
        resolve_top_k,
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
        select_endpoint,
    )


_INITIALIZE_LOCK = threading.RLock()


def _safe_id(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]
    return f"{clean[:96]}_{digest}"


class MemosMemorySystem:
    """MemoryArena adapter around the native MemOS tree-memory implementation."""

    def __init__(self, user_id: Optional[str] = None, top_k: int = 10):
        self.user_id = str(user_id or f"memoryarena_{time.time_ns()}")
        self.safe_user_id = _safe_id(self.user_id)
        self.top_k = resolve_top_k(top_k, "MEMOS_RETRIEVAL_TOP_K")
        self.context_char_budget = resolve_context_char_budget(
            "MEMOS_CONTEXT_CHAR_BUDGET"
        )
        self._lock = threading.RLock()
        self._close_state = PreservingCloseState()

        run_name = os.getenv("MEMOS_MEMORYARENA_RUN_ID") or os.getenv(
            "MEMORYARENA_RUN_ID", "default"
        )
        storage_root = resolve_state_root(
            "memos", "MEMOS_MEMORYARENA_DATA_ROOT"
        )
        self.storage_dir = storage_root / self.safe_user_id
        self.journal_path = self.storage_dir / "added_chunks.json"
        identity = f"{run_name}:{self.user_id}"
        identity_hash = hashlib.sha1(identity.encode("utf-8")).hexdigest()
        self.cube_id = f"memoscube{identity_hash}"
        self.db_name = f"memosma{identity_hash}"

        base_url = select_endpoint(
            self.user_id,
            kind="llm",
            method_list_env="MEMOS_LLM_BASE_URLS",
            method_single_env="MEMOS_LLM_BASE_URL",
        )
        runtime = build_runtime_config({
            "version": re.sub(r"[^A-Za-z0-9]", "", run_name)[:32] or "memoryarena",
            "result_dir": str(storage_root),
            "storage_dir": str(storage_root),
            "top_k": self.top_k,
            "retrieve_top_ks": [self.top_k],
            "ingestion_top_k": self.top_k,
            "llm_model": os.getenv("MEMOS_LLM_MODEL")
            or os.getenv("MEMORYARENA_LLM_MODEL", "Qwen3.5-9B"),
            "llm_api_key": os.getenv("MEMOS_LLM_API_KEY")
            or os.getenv("MEMORYARENA_LLM_API_KEY", "EMPTY"),
            "llm_base_url": base_url,
            "embedding_model_name": os.getenv("MEMOS_EMBEDDING_MODEL")
            or os.getenv(
                "MEMORYARENA_EMBEDDING_MODEL",
                "/path/to/local/all-MiniLM-L6-v2",
            ),
            "enable_textual_memory": True,
            "enable_activation_memory": False,
            "enable_parametric_memory": False,
            "graph_db_uri": os.getenv("MEMOS_GRAPH_DB_URI", "bolt://127.0.0.1:9687"),
            "graph_db_user": os.getenv("MEMOS_GRAPH_DB_USER", "neo4j"),
            "graph_db_password": os.getenv("MEMOS_GRAPH_DB_PASSWORD", "neo4jneo4j"),
            "graph_db_auto_create": True,
            "track_tokens": False,
        })
        mos_data = build_mos_config(runtime, top_k=self.top_k)
        cube_data = build_mem_cube_config(runtime, self.user_id)
        cube_data["cube_id"] = self.cube_id
        cube_data["text_mem"]["config"]["graph_db"]["config"]["db_name"] = self.db_name

        with _INITIALIZE_LOCK:
            memos_dir = Path(os.getenv(
                "MEMOS_DIR",
                str(storage_root / "system"),
            ))
            memos_dir.mkdir(parents=True, exist_ok=True)
            self.mos = MOS(MOSConfig(**mos_data))
            if not any(user["user_id"] == self.user_id for user in self.mos.list_users()):
                self.mos.create_user(user_id=self.user_id)

            config_path = self.storage_dir / "config.json"
            if not config_path.exists():
                self.storage_dir.mkdir(parents=True, exist_ok=True)
                mem_cube = GeneralMemCube(GeneralMemCubeConfig(**cube_data))
                mem_cube.dump(str(self.storage_dir))
                mem_cube.text_mem.graph_store.driver.close()
            self.mos.register_mem_cube(
                mem_cube_name_or_path=str(self.storage_dir),
                mem_cube_id=self.cube_id,
                user_id=self.user_id,
            )
        self._added_hashes = self._load_added_hashes()

    def _load_added_hashes(self):
        if not self.journal_path.exists():
            return set()
        try:
            data = json.loads(self.journal_path.read_text(encoding="utf-8"))
            return set(data.get("hashes", []))
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
            self._close_state.ensure_open(f"MemOS memory {self.user_id}")
            if chunk_hash in self._added_hashes:
                return {"stored": False, "duplicate": True, "characters": len(text)}
            self.mos.add(
                memory_content=text,
                mem_cube_id=self.cube_id,
                user_id=self.user_id,
            )
            self._added_hashes.add(chunk_hash)
            self._save_added_hashes()
        return {"stored": True, "characters": len(text)}

    def _retrieve(self, prompt: str):
        result = self.mos.search(
            query=str(prompt),
            user_id=self.user_id,
            install_cube_ids=[self.cube_id],
        )
        contexts = []
        for cube_result in result.get("text_mem", []):
            for memory in cube_result.get("memories", []):
                text = getattr(memory, "memory", None)
                if text:
                    contexts.append(str(text))
        return contexts[: self.top_k]

    def wrap_user_prompt(self, prompt: str) -> str:
        with self._lock:
            self._close_state.ensure_open(f"MemOS memory {self.user_id}")
            contexts = self._retrieve(prompt)
        return format_memory_prompt(
            prompt,
            contexts,
            char_budget=self.context_char_budget,
        )

    def _release_resources(self):
        cube = self.mos.mem_cubes.get(self.cube_id)
        text_mem = cube.text_mem if cube is not None else None
        seen = set()
        resources = []
        for label, resource in (
            ("memos.graph", getattr(getattr(text_mem, "graph_store", None), "driver", None)),
            ("memos.vector", getattr(getattr(text_mem, "vector_db", None), "client", None)),
            ("memos.extractor_llm", getattr(getattr(text_mem, "extractor_llm", None), "client", None)),
            ("memos.dispatcher_llm", getattr(getattr(text_mem, "dispatcher_llm", None), "client", None)),
            ("memos.embedder", getattr(getattr(text_mem, "embedder", None), "client", None)),
            ("memos.chat_llm", getattr(getattr(self.mos, "chat_llm", None), "client", None)),
        ):
            resources.extend(close_resource(resource, label, seen=seen))
        if cube is not None:
            self.mos.unregister_mem_cube(self.cube_id, user_id=self.user_id)
        return resources

    def close(self, completed: bool = False):
        return self._close_state.close(completed, self._release_resources)


def create_backend(*, user_id=None, top_k=10):
    """Create MemOS through the common MemoryArena factory."""

    return MemosMemorySystem(user_id=user_id, top_k=top_k)
