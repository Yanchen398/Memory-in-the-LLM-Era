from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


MEM0_SOURCE = Path(__file__).resolve().parents[1] / "mem0"
while str(MEM0_SOURCE) in sys.path:
    sys.path.remove(str(MEM0_SOURCE))
sys.path.insert(0, str(MEM0_SOURCE))

os.environ.setdefault("OPENAI_API_KEY", "fake_key")
os.environ.setdefault("MEM0_EMBEDDING_DEVICE", "cpu")
# This benchmark is local-only. Disable mem0's anonymous PostHog client before
# importing mem0 so no telemetry process can attempt an outbound connection.
os.environ["MEM0_TELEMETRY"] = "False"

from mem0 import Memory
from mem0.llms.openai import consume_context_failure_events

import mem0


MEM0_IMPORTED_FROM = Path(mem0.__file__).resolve()
if MEM0_SOURCE.resolve() not in MEM0_IMPORTED_FROM.parents:
    raise RuntimeError(
        f"Local mem0 source was not selected: imported {MEM0_IMPORTED_FROM}"
    )


def _safe_user_name(value: Optional[str]) -> str:
    raw = str(value or "default")
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[:120] or "default"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _state_root() -> Path:
    configured = os.getenv("MEMORYARENA_MEMORY_INDEX_ROOT")
    return Path(configured) if configured else (
        Path.home() / ".local" / "state" / "memoryarena"
    )


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str))


class Mem0MemorySystem:
    """Local Mem0 adapter for the MemoryArena server."""

    def __init__(
        self,
        user_id: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        top_k: Optional[int] = None,
    ):
        self.user_id = str(user_id or "default")
        self.model = model or os.getenv("QWEN35_MODEL", "Qwen/Qwen3.5-9B")
        self.base_url = base_url or os.getenv(
            "QWEN35_BASE_URL", "http://127.0.0.1:8001/v1"
        )
        if not self.base_url.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise RuntimeError(
                f"Local mem0 requires a loopback Qwen endpoint, got {self.base_url}"
            )

        self.top_k = int(top_k or os.getenv("MEMORYARENA_RETRIEVE_K", "10"))
        self.state_dir = (
            _state_root() / "mem0_indepth" / _safe_user_name(self.user_id)
        )
        self.index_dir = self.state_dir / "faiss"
        self.history_db = self.state_dir / "history.db"
        self.events_path = self.state_dir / "events.jsonl"
        self.manifest_path = self.state_dir / "index_manifest.json"
        self.last_retrieval_path = self.state_dir / "last_retrieval.json"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._add_calls = 0
        self._stored_add_calls = 0
        self._noop_add_calls = 0
        self._failed_add_calls = 0
        self._method_context_failure_count = 0
        self._wrap_calls = 0
        self._created_at = time.time()

        self.memory = Memory.from_config(
            {
                "vector_store": {
                    "provider": "faiss",
                    "config": {
                        "path": str(self.index_dir),
                        "embedding_model_dims": 384,
                    },
                },
                "llm": {
                    "provider": "vllm",
                    "config": {
                        "api_key": "fake_key",
                        "vllm_base_url": self.base_url,
                        "model": self.model,
                    },
                },
                "history_db_path": str(self.history_db),
            }
        )
        self._write_event(
            "initialize",
            {
                "local_only": True,
                "vector_store": "faiss",
                "embedder": "sentence-transformers/all-MiniLM-L6-v2",
                "llm_endpoint": self.base_url,
            },
        )
        self._write_manifest()

    def add_chunk(self, chunk: str) -> Dict[str, Any]:
        text = str(chunk or "")
        if not text.strip():
            return {
                "stored": False,
                "reason": "empty_chunk",
                "memory_count": self._memory_count(),
            }

        started = time.perf_counter()
        with self._lock:
            self._add_calls += 1
            normalized = " ".join(text.lower().split()).rstrip(".")
            if normalized == "initial result: empty":
                self._noop_add_calls += 1
                latency = time.perf_counter() - started
                self._write_event(
                    "add",
                    {
                        "call_index": self._add_calls,
                        "stored": False,
                        "reason": "official_empty_initial_result_sentinel",
                        "chunk_chars": len(text),
                        "chunk_sha256": hashlib.sha256(
                            text.encode("utf-8")
                        ).hexdigest(),
                        "memory_count": self._memory_count(),
                        "latency_seconds": round(latency, 6),
                    },
                )
                self._write_manifest()
                return {
                    "stored": False,
                    "reason": "official_empty_initial_result_sentinel",
                    "add_call": self._add_calls,
                    "memory_count": self._memory_count(),
                    "latency_seconds": round(latency, 6),
                }

            memory_count_before = self._memory_count()
            consume_context_failure_events()
            response = self.memory.add(
                messages=[{"role": "user", "content": text}],
                user_id=self.user_id,
                metadata={
                    "source": "memoryarena_official_memory_update",
                    "add_call": self._add_calls,
                },
                infer=True,
            )
            context_failures = consume_context_failure_events()
            if context_failures:
                memory_count = self._memory_count()
                if memory_count != memory_count_before:
                    raise RuntimeError(
                        "Local mem0 context failure left a partial memory write: "
                        f"before={memory_count_before} after={memory_count}"
                    )
                self._failed_add_calls += 1
                self._method_context_failure_count += len(context_failures)
                latency = time.perf_counter() - started
                response_safe = _json_safe(response)
                self._write_event(
                    "add",
                    {
                        "call_index": self._add_calls,
                        "stored": False,
                        "reason": "method_context_limit",
                        "chunk_chars": len(text),
                        "chunk_sha256": hashlib.sha256(
                            text.encode("utf-8")
                        ).hexdigest(),
                        "memory_count": memory_count,
                        "latency_seconds": round(latency, 6),
                        "context_failures": context_failures,
                        "response": response_safe,
                    },
                )
                self._write_manifest()
                return {
                    "stored": False,
                    "reason": "method_context_limit",
                    "add_call": self._add_calls,
                    "memory_count": memory_count,
                    "latency_seconds": round(latency, 6),
                    "context_failures": context_failures,
                }

            self._stored_add_calls += 1
            memory_count = self._memory_count()
            latency = time.perf_counter() - started
            response_safe = _json_safe(response)
            self._write_event(
                "add",
                {
                    "call_index": self._add_calls,
                    "stored": True,
                    "chunk_chars": len(text),
                    "chunk_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    "memory_count": memory_count,
                    "latency_seconds": round(latency, 6),
                    "response": response_safe,
                },
            )
            self._write_manifest()

        return {
            "stored": True,
            "add_call": self._add_calls,
            "memory_count": memory_count,
            "index_dir": str(self.index_dir),
            "latency_seconds": round(latency, 6),
            "mem0_response": response_safe,
        }

    def wrap_user_prompt(self, prompt: str) -> str:
        question = str(prompt or "")
        started = time.perf_counter()
        with self._lock:
            response = self.memory.search(
                query=question,
                user_id=self.user_id,
                limit=self.top_k,
            )
            results = self._results(response)
            self._wrap_calls += 1
            latency = time.perf_counter() - started
            context_lines = self._format_results(results)
            retrieval_payload = {
                "call_index": self._wrap_calls,
                "retrieve_k": self.top_k,
                "latency_seconds": round(latency, 6),
                "query_preview": question[:2000],
                "result_count": len(results),
                "results": [_json_safe(item) for item in results],
            }
            self.last_retrieval_path.write_text(
                json.dumps(retrieval_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._write_event("wrap_user_prompt", retrieval_payload)
            self._write_manifest()

        lines = ["<memory_context>"]
        lines.extend(context_lines or ["None"])
        lines.append("</memory_context>")
        lines.append(f"User Prompt: {question}")
        return "\n".join(lines)

    def close(self, completed: bool = False) -> dict:
        with self._lock:
            try:
                self.memory.close()
            finally:
                self._write_event("close", {"memory_count": self._memory_count()})
                self._write_manifest()
        return {
            "closed": True,
            "completed": bool(completed),
            "state_preserved": True,
            "resources_closed": ["mem0.memory"],
        }

    def _results(self, response: Any) -> List[Dict[str, Any]]:
        if isinstance(response, dict):
            raw = response.get("results") or []
        elif isinstance(response, list):
            raw = response
        else:
            raw = []
        return [item for item in raw if isinstance(item, dict)]

    def _format_results(self, results: List[Dict[str, Any]]) -> List[str]:
        lines = []
        for rank, item in enumerate(results, start=1):
            text = item.get("memory") or item.get("text") or item.get("data")
            if not text:
                continue
            attributes = [f'rank="{rank}"']
            if item.get("score") is not None:
                attributes.append(f'score="{item["score"]}"')
            if item.get("id"):
                attributes.append(f'id="{item["id"]}"')
            lines.append(f"<memory {' '.join(attributes)}>{text}</memory>")
        return lines

    def _memory_count(self) -> int:
        vector_store = getattr(self.memory, "vector_store", None)
        index = getattr(vector_store, "index", None)
        return int(getattr(index, "ntotal", 0) or 0)

    def _write_event(self, event: str, payload: Dict[str, Any]) -> None:
        record = {
            "event": event,
            "timestamp": time.time(),
            "user_id": self.user_id,
            **_json_safe(payload),
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_manifest(self) -> None:
        index_files = sorted(str(path) for path in self.index_dir.glob("*"))
        payload = {
            "method": "mem0_indepth",
            "implementation": "local_mem0_memory_from_config",
            "local_only": True,
            "cloud_client_used": False,
            "user_id": self.user_id,
            "model": self.model,
            "llm_endpoint": self.base_url,
            "vector_store": "faiss",
            "embedder": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dims": 384,
            "retrieve_k": self.top_k,
            "memory_count": self._memory_count(),
            "add_calls": self._add_calls,
            "stored_add_calls": self._stored_add_calls,
            "noop_add_calls": self._noop_add_calls,
            "failed_add_calls": self._failed_add_calls,
            "method_context_failure_count": self._method_context_failure_count,
            "wrap_calls": self._wrap_calls,
            "created_at": self._created_at,
            "updated_at": time.time(),
            "index_dir": str(self.index_dir),
            "index_files": index_files,
            "history_db": str(self.history_db),
            "events_path": str(self.events_path),
            "last_retrieval_path": str(self.last_retrieval_path),
        }
        self.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def create_backend(*, user_id=None, top_k=10):
    """Create Mem0 through the common MemoryArena factory."""

    return Mem0MemorySystem(user_id=user_id, top_k=top_k)
