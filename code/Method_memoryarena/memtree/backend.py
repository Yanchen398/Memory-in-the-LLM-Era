import hashlib
import json
import os
import pickle
import threading
import time
from types import SimpleNamespace
from typing import Optional

# Avoid loading the LOCOMO/default singleton when importing memtree modules here.
os.environ.setdefault("MEMTREE_DISABLE_AUTO_GLOBALCONFIG", "1")

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

from . import config as mem_config
from . import structure as mem_structure
from . import utils as mem_utils
from .structure import MemTree

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


_GLOBAL_LOCK = threading.RLock()
_EMBEDDING_MODEL_CACHE = {}


def _embedding_model(model_name: str, device: str):
    aliases = {
        "minilm": "/path/to/local/all-MiniLM-L6-v2",
        "all-minilm-l6-v2": "/path/to/local/all-MiniLM-L6-v2",
        "all-MiniLM-L6-v2": "/path/to/local/all-MiniLM-L6-v2",
    }
    resolved = aliases.get(model_name, model_name)
    key = (resolved, device)
    with _GLOBAL_LOCK:
        model = _EMBEDDING_MODEL_CACHE.get(key)
        if model is None:
            model = SentenceTransformer(resolved, device=device)
            _EMBEDDING_MODEL_CACHE[key] = model
            print(f"Loaded MemoryArena memtree embedding model {resolved} on {device}", flush=True)
        return model


class MemTreeMemorySystem:
    """MemoryArena adapter for the native MemTree implementation.

    Each MemoryArena user_id owns an isolated Milvus-lite DB, collection, tree
    pickle, and duplicate journal. The original MemTree add/search/update logic
    is used through its module-level helpers; a global lock protects those
    helpers because the upstream implementation is driven by a singleton config.
    """

    def __init__(self, user_id: Optional[str] = None, top_k: int = 10):
        self.user_id = str(user_id or f"memoryarena_{time.time_ns()}")
        self.safe_user_id = safe_user_id(self.user_id)
        self.top_k = resolve_top_k(top_k, "MEMTREE_RETRIEVAL_TOP_K")
        self.context_char_budget = resolve_context_char_budget(
            "MEMTREE_CONTEXT_CHAR_BUDGET"
        )
        self._lock = threading.RLock()
        self._close_state = PreservingCloseState()

        storage_root = resolve_state_root(
            "memtree", "MEMTREE_MEMORYARENA_DATA_ROOT"
        )
        self.storage_dir = storage_root / self.safe_user_id
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.tree_path = self.storage_dir / "tree.pkl"
        self.journal_path = self.storage_dir / "added_chunks.json"
        self.db_path = self.storage_dir / "milvus_memtree.db"
        short_hash = hashlib.sha1(self.user_id.encode("utf-8")).hexdigest()[:16]
        self.collection_name = f"memtree_{short_hash}"

        embedding_model_name = os.getenv("MEMTREE_EMBEDDING_MODEL") or os.getenv(
            "MEMORYARENA_EMBEDDING_MODEL", "/path/to/local/all-MiniLM-L6-v2"
        )
        embedding_device = os.getenv("MEMTREE_EMBEDDING_DEVICE", "cpu")
        self.cfg = SimpleNamespace(
            dataset_name=f"memoryarena_{self.safe_user_id}",
            dimension=384,
            embedding_model_name=embedding_model_name,
            embedding_device=embedding_device,
            embedding_batch_size=int(os.getenv("MEMTREE_EMBEDDING_BATCH_SIZE", "256")),
            llm_base_url=select_endpoint(
                self.user_id,
                kind="llm",
                method_list_env="MEMTREE_LLM_BASE_URLS",
                method_single_env="MEMTREE_LLM_BASE_URL",
            ),
            llm_api_key=os.getenv("MEMTREE_LLM_API_KEY")
            or os.getenv("MEMORYARENA_LLM_API_KEY", "EMPTY"),
            llm_model=os.getenv("MEMTREE_LLM_MODEL")
            or os.getenv("MEMORYARENA_LLM_MODEL", "Qwen3.5-9B"),
            llm_parallel_nums=int(os.getenv("MEMTREE_LLM_PARALLEL_NUMS", "16")),
            answer_parallel_nums=int(os.getenv("MEMTREE_ANSWER_PARALLEL_NUMS", "16")),
            base_threshold=float(os.getenv("MEMTREE_BASE_THRESHOLD", "0.4")),
            rate=float(os.getenv("MEMTREE_RATE", "0.5")),
            max_depth=int(os.getenv("MEMTREE_MAX_DEPTH", "15")),
            top_k_retrieve=self.top_k,
            retrieve_top_ks=[self.top_k],
            collection_name=self.collection_name,
            db_name=str(self.db_path),
            save_path=str(self.tree_path),
            model=_embedding_model(embedding_model_name, embedding_device),
        )
        self.cfg.client = MilvusClient(str(self.db_path))
        mem_config.create_collections(self.cfg.client, self.collection_name, self.cfg.dimension)
        self._added_hashes = self._load_added_hashes()
        self.tree = self._load_tree()

    def _activate(self):
        mem_config.globalconfig = self.cfg
        mem_utils.globalconfig = self.cfg
        mem_structure.globalconfig = self.cfg

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
        tmp_path.write_text(json.dumps({"hashes": sorted(self._added_hashes)}, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.journal_path)

    def _load_tree(self):
        if not self.tree_path.exists():
            return MemTree("")
        try:
            with self.tree_path.open("rb") as f:
                tree = pickle.load(f)
            if not getattr(tree, "nodes", None):
                return MemTree("")
            return tree
        except Exception as exc:
            raise RuntimeError(f"Failed to load memtree for {self.user_id}: {exc}") from exc

    def _save_tree(self):
        tmp_path = self.tree_path.with_suffix(".tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(self.tree, f)
        os.replace(tmp_path, self.tree_path)

    def _root_node_id(self) -> int:
        for node_id, node in self.tree.nodes.items():
            if getattr(node, "pv", None) is None:
                return node_id
        return next(iter(self.tree.nodes.keys()))

    def add_chunk(self, chunk: str):
        text = normalize_chunk(chunk)
        if text is None:
            return {"stored": False, "skipped": True, "characters": 0}
        chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock, _GLOBAL_LOCK:
            self._close_state.ensure_open(f"MemTree memory {self.user_id}")
            if chunk_hash in self._added_hashes:
                return {"stored": False, "duplicate": True, "characters": len(text)}
            self._activate()
            self.tree.add_node(text, self._root_node_id())
            self._added_hashes.add(chunk_hash)
            self._save_added_hashes()
            self._save_tree()
            return {"stored": True, "characters": len(text), "nodes": len(self.tree.nodes)}

    def _retrieve_contexts(self, prompt: str):
        if len(self.tree.nodes) <= 1:
            return []
        query_embeddings = mem_utils.get_embedding([str(prompt)], self.cfg.embedding_batch_size)
        hits = mem_utils.search([query_embeddings[0]], top_k=self.top_k)
        contexts = []
        for item in hits[0] if hits else []:
            node = self.tree.nodes.get(item.get("id"))
            if node is not None and getattr(node, "cv", None):
                contexts.append(str(node.cv))
        return contexts

    def wrap_user_prompt(self, prompt: str) -> str:
        with self._lock, _GLOBAL_LOCK:
            self._close_state.ensure_open(f"MemTree memory {self.user_id}")
            self._activate()
            contexts = self._retrieve_contexts(prompt)
        return format_memory_prompt(
            prompt,
            contexts,
            char_budget=self.context_char_budget,
        )

    def _release_resources(self):
        with _GLOBAL_LOCK:
            self._activate()
            self._save_tree()
            self._save_added_hashes()
        return close_resource(self.cfg.client, "memtree.milvus")

    def close(self, completed: bool = False):
        return self._close_state.close(completed, self._release_resources)


def create_backend(*, user_id=None, top_k=10):
    """Create MemTree through the common MemoryArena factory."""

    return MemTreeMemorySystem(user_id=user_id, top_k=top_k)
