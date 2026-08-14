import json
import os
import re
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import List, Optional


if os.getenv("MEMORYARENA_AMEM_FORCE_CPU_EMBED", "1") != "0":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from .simple_qa import SimpleMemAgent, persist_amem_index


# AMEM and tqdm temporarily replace process-global stdout while adding memory.
# Serialize adapter operations so independent FastAPI requests cannot cross-talk.
_PROCESS_MEMORY_LOCK = threading.RLock()


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


class AMemMemorySystem:
    """A-MEM adapter for the MemoryArena server."""

    def __init__(
        self,
        user_id: Optional[str] = None,
        model: Optional[str] = None,
        backend: str = "openai",
        top_k: Optional[int] = None,
    ):
        self.user_id = user_id or "default"
        self.model = model or os.getenv("QWEN35_MODEL", "Qwen/Qwen3.5-9B")
        self.backend = backend
        self.top_k = int(top_k or os.getenv("MEMORYARENA_RETRIEVE_K", "5"))
        self.query_chars = int(os.getenv("AMEM_ARENA_QUERY_CHARS", "16000"))
        self.add_chunk_chars = int(os.getenv("AMEM_ARENA_ADD_CHUNK_CHARS", "12000"))
        self.max_context_chars = int(os.getenv("AMEM_ARENA_MAX_CONTEXT_CHARS", "36000"))
        self.state_dir = _index_root() / "amem_indepth" / _safe_name(self.user_id)
        self.index_dir = self.state_dir / "amem_index"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with _PROCESS_MEMORY_LOCK:
            self.agent = SimpleMemAgent(self.model, self.backend, self.top_k)
        self._add_count = 0

    def add_chunk(self, chunk: str):
        with _PROCESS_MEMORY_LOCK:
            return self._add_chunk_serialized(chunk)

    def _add_chunk_serialized(self, chunk: str):
        started = time.perf_counter()
        stored = 0
        for piece in self._split_chunk(chunk):
            if not piece.strip():
                continue
            with (self.state_dir / "amem_stdout.log").open("a", encoding="utf-8") as logf:
                with redirect_stdout(logf):
                    self.agent.add_memory(piece)
            stored += 1
        self._add_count += stored
        manifest = self._persist()
        return {
            "stored_chunks": stored,
            "total_add_calls": self._add_count,
            "memory_count": self._memory_count(),
            "index_dir": str(self.index_dir),
            "manifest": manifest,
            "latency_seconds": round(time.perf_counter() - started, 6),
        }

    def wrap_user_prompt(self, prompt: str) -> str:
        with _PROCESS_MEMORY_LOCK:
            return self._wrap_user_prompt_serialized(prompt)

    def _wrap_user_prompt_serialized(self, prompt: str) -> str:
        started = time.perf_counter()
        retrieval_query = self._build_retrieval_query(prompt)
        raw_context = self._retrieve(retrieval_query)
        context_lines = self._context_lines(raw_context)
        lines = ["<memory_context>"]
        if context_lines:
            lines.extend(context_lines)
        else:
            lines.append("None")
        lines.append("</memory_context>")
        lines.append(f"User Prompt: {prompt}")
        self._write_last_retrieval(prompt, retrieval_query, context_lines, time.perf_counter() - started)
        return "\n".join(lines)

    def _split_chunk(self, chunk: str) -> List[str]:
        text = (chunk or "").strip()
        if not text:
            return []
        if len(text) <= self.add_chunk_chars:
            return [text]
        pieces = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.add_chunk_chars)
            if end < len(text):
                newline = text.rfind("\n", start, end)
                if newline > start + self.add_chunk_chars // 2:
                    end = newline
            pieces.append(text[start:end].strip())
            start = end
        return [piece for piece in pieces if piece]

    def _build_retrieval_query(self, prompt: str) -> str:
        clipped = (prompt or "")[: self.query_chars]
        try:
            query = self.agent.generate_query_llm(clipped)
            if query and str(query).strip():
                return str(query).strip()
        except Exception as exc:
            self._write_warning("query_generation_error", exc)
        return clipped

    def _retrieve(self, retrieval_query: str) -> str:
        try:
            raw = self.agent.retrieve_memory(retrieval_query, k=self.top_k)
        except Exception as exc:
            self._write_warning("retrieval_error", exc)
            raw = ""
        return str(raw or "")

    def _context_lines(self, raw_context: str) -> List[str]:
        raw_context = (raw_context or "").strip()
        if not raw_context:
            return []
        clipped = raw_context[: self.max_context_chars]
        chunks = [line.strip() for line in clipped.splitlines() if line.strip()]
        if not chunks:
            chunks = [clipped]
        return [f"<memory rank=\"{idx}\">{chunk}</memory>" for idx, chunk in enumerate(chunks, start=1)]

    def _memory_count(self) -> int:
        try:
            return len(getattr(self.agent.memory_system, "memories", {}) or {})
        except Exception:
            return 0

    def _persist(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        try:
            paths = persist_amem_index(self.agent, str(self.index_dir))
        except Exception as exc:
            self._write_warning("persist_error", exc)
            paths = {"persist_error": str(exc)}
        manifest = {
            "method": "amem_indepth",
            "user_id": self.user_id,
            "model": self.model,
            "backend": self.backend,
            "retrieve_k": self.top_k,
            "memory_count": self._memory_count(),
            "index_dir": str(self.index_dir),
            "persist_paths": paths,
        }
        with (self.state_dir / "index_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return manifest

    def close(self, completed: bool = False) -> dict:
        with _PROCESS_MEMORY_LOCK:
            manifest = self._persist()
        return {
            "closed": True,
            "completed": bool(completed),
            "state_preserved": True,
            "manifest": manifest,
        }

    def _write_last_retrieval(
        self,
        prompt: str,
        retrieval_query: str,
        context_lines: List[str],
        latency_seconds: float,
    ) -> None:
        payload = {
            "retrieve_k": self.top_k,
            "latency_seconds": round(latency_seconds, 6),
            "query_preview": prompt[:1000],
            "retrieval_query": retrieval_query[:1000],
            "result_count": len(context_lines),
            "result_previews": [line[:1000] for line in context_lines],
        }
        with (self.state_dir / "last_retrieval.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _write_warning(self, name: str, exc: Exception) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "warning": name,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "time": time.time(),
        }
        with (self.state_dir / f"{name}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def create_backend(*, user_id=None, top_k=10):
    """Create A-MEM through the common MemoryArena factory."""

    return AMemMemorySystem(user_id=user_id, top_k=top_k)
