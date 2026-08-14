import json
import os
import re
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _safe_name(value: Optional[str]) -> str:
    text = str(value or "default")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:160] or "default"


def _index_root() -> Path:
    return Path(
        os.getenv(
            "MEMORYARENA_MEMORY_INDEX_ROOT",
            str(Path.home() / ".local" / "state" / "memoryarena"),
        )
    )


def _load_embedder(model_name: str, device: str) -> SentenceTransformer:
    key = (model_name, device)
    model = _MODEL_CACHE.get(key)
    if model is None:
        with _MODEL_CACHE_LOCK:
            model = _MODEL_CACHE.get(key)
            if model is None:
                model = SentenceTransformer(model_name, device=device)
                _MODEL_CACHE[key] = model
    return model


class MemoryBankMemorySystem:
    """MemoryBank retrieval memory for the MemoryArena server.

    The official runner only needs add_chunk() and wrap_user_prompt().  This
    class preserves every task-specific memory index on disk so formal runs can
    be audited or reused later.
    """

    def __init__(
        self,
        user_id: Optional[str] = None,
        embedding_model: Optional[str] = None,
        top_k: Optional[int] = None,
        chunk_chars: Optional[int] = None,
    ):
        self.user_id = user_id or "default"
        self.embedding_model = embedding_model or os.getenv(
            "MEMORYBANK_EMBEDDING_MODEL", "/path/to/local/all-MiniLM-L6-v2"
        )
        self.device = os.getenv("MEMORYBANK_EMBED_DEVICE", "cpu")
        self.top_k = int(top_k or os.getenv("MEMORYARENA_RETRIEVE_K", "5"))
        self.chunk_chars = int(chunk_chars or os.getenv("MEMORYBANK_ARENA_CHUNK_CHARS", "6000"))
        self.max_context_chars = int(os.getenv("MEMORYBANK_ARENA_MAX_CONTEXT_CHARS", "36000"))
        self.state_dir = _index_root() / "memorybank_indepth" / _safe_name(self.user_id)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = _load_embedder(self.embedding_model, self.device)
        self.docs: List[str] = []
        self.embeddings = None
        self._load_existing_state()

    def add_chunk(self, chunk: str):
        started = time.perf_counter()
        added = 0
        for piece in self._split_chunk(chunk):
            if not piece.strip():
                continue
            emb = self.embedder.encode([piece], convert_to_numpy=True, normalize_embeddings=True)
            self.docs.append(piece)
            if self.embeddings is None:
                self.embeddings = emb
            else:
                self.embeddings = np.vstack([self.embeddings, emb])
            added += 1
        self._persist()
        return {
            "stored_chunks": added,
            "total_chunks": len(self.docs),
            "index_dir": str(self.state_dir),
            "latency_seconds": round(time.perf_counter() - started, 6),
        }

    def wrap_user_prompt(self, prompt: str) -> str:
        started = time.perf_counter()
        results = self._retrieve(prompt)
        lines = ["<memory_context>"]
        if not results:
            lines.append("None")
        else:
            used_chars = 0
            for rank, text in enumerate(results, start=1):
                if used_chars >= self.max_context_chars:
                    break
                remaining = self.max_context_chars - used_chars
                clipped = text[:remaining]
                lines.append(f"<memory rank=\"{rank}\">{clipped}</memory>")
                used_chars += len(clipped)
        lines.append("</memory_context>")
        lines.append(f"User Prompt: {prompt}")
        self._write_last_retrieval(prompt, results, time.perf_counter() - started)
        return "\n".join(lines)

    def _split_chunk(self, chunk: str) -> List[str]:
        text = (chunk or "").strip()
        if not text:
            return []
        if len(text) <= self.chunk_chars:
            return [text]
        pieces = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_chars)
            if end < len(text):
                newline = text.rfind("\n", start, end)
                if newline > start + self.chunk_chars // 2:
                    end = newline
            pieces.append(text[start:end].strip())
            start = end
        return [piece for piece in pieces if piece]

    def _retrieve(self, query: str) -> List[str]:
        if not self.docs or self.embeddings is None:
            return []
        q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = self.embeddings @ q_emb
        order = scores.argsort()[::-1][: self.top_k]
        return [self.docs[int(i)] for i in order]

    def _load_existing_state(self) -> None:
        docs_path = self.state_dir / "memory_docs.json"
        emb_path = self.state_dir / "memory_doc_embeddings.npy"
        if docs_path.exists():
            with docs_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.docs = [str(item) for item in data]
        if emb_path.exists():
            self.embeddings = np.load(emb_path)

    def _persist(self) -> None:
        docs_path = self.state_dir / "memory_docs.json"
        emb_path = self.state_dir / "memory_doc_embeddings.npy"
        with docs_path.open("w", encoding="utf-8") as f:
            json.dump(self.docs, f, ensure_ascii=False, indent=2)
        if self.embeddings is not None:
            np.save(emb_path, self.embeddings)
        manifest = {
            "method": "memorybank_indepth",
            "user_id": self.user_id,
            "embedding_model": self.embedding_model,
            "embedding_device": self.device,
            "retrieve_k": self.top_k,
            "chunk_chars": self.chunk_chars,
            "memory_docs": str(docs_path),
            "memory_doc_embeddings": str(emb_path) if self.embeddings is not None else None,
            "total_chunks": len(self.docs),
        }
        with (self.state_dir / "index_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def close(self, completed: bool = False) -> dict:
        self._persist()
        return {
            "closed": True,
            "completed": bool(completed),
            "state_preserved": True,
            "resources_closed": [],
        }

    def _write_last_retrieval(self, prompt: str, results: List[str], latency_seconds: float) -> None:
        payload = {
            "retrieve_k": self.top_k,
            "latency_seconds": round(latency_seconds, 6),
            "query_preview": prompt[:1000],
            "result_count": len(results),
            "result_previews": [item[:1000] for item in results],
        }
        with (self.state_dir / "last_retrieval.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def create_backend(*, user_id=None, top_k=10):
    """Create MemoryBank through the common MemoryArena factory."""

    return MemoryBankMemorySystem(user_id=user_id, top_k=top_k)
