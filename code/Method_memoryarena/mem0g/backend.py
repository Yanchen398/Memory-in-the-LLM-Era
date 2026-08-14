from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional


MEM0_SOURCE = Path(__file__).resolve().parents[1] / "mem0"
while str(MEM0_SOURCE) in sys.path:
    sys.path.remove(str(MEM0_SOURCE))
sys.path.insert(0, str(MEM0_SOURCE))

os.environ.setdefault("OPENAI_API_KEY", "fake_key")
os.environ.setdefault("MEM0_EMBEDDING_DEVICE", "cpu")
os.environ["MEM0_TELEMETRY"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from mem0 import Memory
from mem0.llms.openai import consume_context_failure_events

import mem0
import mem0.memory.graph_memory as graph_memory_module


MEM0_IMPORTED_FROM = Path(mem0.__file__).resolve()
if MEM0_SOURCE.resolve() not in MEM0_IMPORTED_FROM.parents:
    raise RuntimeError(
        f"Local mem0 source was not selected: imported {MEM0_IMPORTED_FROM}"
    )


_GRAPH_EMBEDDER_LOCK = threading.Lock()
_GRAPH_EMBEDDER_MODEL: Any = None


class _SharedGraphEmbedder:
    """Share one offline CPU graph embedder inside each lane process."""

    def __init__(self, model_name: Optional[str] = None):
        global _GRAPH_EMBEDDER_MODEL
        selected = model_name or os.getenv(
            "MEM0G_GRAPH_EMBEDDER",
            "sentence-transformers/all-mpnet-base-v2",
        )
        with _GRAPH_EMBEDDER_LOCK:
            if _GRAPH_EMBEDDER_MODEL is None:
                from sentence_transformers import SentenceTransformer

                _GRAPH_EMBEDDER_MODEL = SentenceTransformer(
                    selected,
                    device="cpu",
                    local_files_only=True,
                )
        self.model = _GRAPH_EMBEDDER_MODEL
        self.embedding_dims = int(self.model.get_sentence_embedding_dimension())
        self.config = type(
            "GraphEmbedderConfig",
            (),
            {"embedding_dims": self.embedding_dims},
        )()

    def embed(self, texts: Any, *args: Any, **kwargs: Any) -> Any:
        return self.model.encode(texts)


class _LocalGraphEmbedderFactory:
    @staticmethod
    def create(*args: Any, **kwargs: Any) -> _SharedGraphEmbedder:
        return _SharedGraphEmbedder()


# The modified local mem0 graph implementation constructs then replaces an
# unused provider embedder. Patch only this module in the lane process so both
# construction sites use one offline CPU model and never instantiate a cloud
# embedder.
graph_memory_module.LocalEmbedder = _SharedGraphEmbedder
graph_memory_module.EmbedderFactory = _LocalGraphEmbedderFactory


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


def _loopback_url(value: str) -> bool:
    return value.startswith(
        (
            "bolt://127.0.0.1:",
            "bolt://localhost:",
            "neo4j://127.0.0.1:",
            "neo4j://localhost:",
        )
    )


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str))


def _classify_graph_llm_exception(exc: Exception) -> Dict[str, str]:
    message = str(exc)
    lowered = message.lower()
    if "invalid json: eof while parsing" in lowered:
        kind = "tool_arguments_json_eof"
    elif (
        "input tokens and requested" in lowered
        or "context length is only" in lowered
    ):
        kind = "context_limit"
    else:
        kind = "unexpected_graph_llm_error"
    return {
        "kind": kind,
        "exception_type": type(exc).__name__,
        "error": message,
    }


class Mem0gMemorySystem:
    """Local graph-enabled Mem0 adapter for MemoryArena."""

    def __init__(
        self,
        user_id: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        top_k: Optional[int] = None,
    ):
        self.user_id = str(user_id or "default")
        self.model = model or os.getenv("QWEN35_MODEL", "Qwen3.5-9B")
        self.base_url = base_url or os.getenv(
            "QWEN35_BASE_URL", "http://127.0.0.1:8003/v1"
        )
        if not self.base_url.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise RuntimeError(
                f"Local Mem0g requires a loopback Qwen endpoint, got {self.base_url}"
            )

        self.graph_url = os.getenv("MEM0G_NEO4J_URL", "")
        if not _loopback_url(self.graph_url):
            raise RuntimeError(
                f"Local Mem0g requires a loopback Neo4j endpoint, got {self.graph_url!r}"
            )
        self.graph_database = os.getenv("MEM0G_NEO4J_DATABASE", "neo4j")
        self.graph_username = os.getenv("MEM0G_NEO4J_USERNAME", "neo4j")
        self.graph_password = os.getenv(
            "MEM0G_NEO4J_PASSWORD", "memoryarena-local-only"
        )
        self.top_k = int(top_k or os.getenv("MEMORYARENA_RETRIEVE_K", "10"))

        self.state_dir = (
            _state_root() / "mem0g_indepth" / _safe_user_name(self.user_id)
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
        self._graph_add_failures = 0
        self._partial_graph_add_calls = 0
        self._partial_graph_vector_noop_calls = 0
        self._graph_tool_output_failures = 0
        self._graph_llm_failure_events: List[Dict[str, str]] = []
        self._method_context_failure_count = 0
        self._graph_retrieval_context_failure_count = 0
        self._wrap_calls = 0
        self._graph_relations_returned = 0
        self._created_at = time.time()
        self._closed = False

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
                "graph_store": {
                    "provider": "neo4j",
                    "config": {
                        "url": self.graph_url,
                        "username": self.graph_username,
                        "password": self.graph_password,
                        "database": self.graph_database,
                        "base_label": True,
                    },
                },
            }
        )
        if not getattr(self.memory, "enable_graph", False):
            raise RuntimeError("Mem0g graph initialization was disabled")
        if getattr(self.memory, "graph", None) is None:
            raise RuntimeError("Mem0g graph object is missing")
        self._require_graph_tool_calls()

        graph_counts = self._graph_counts()
        if graph_counts["cross_user_relationship_count"]:
            raise RuntimeError(
                "Mem0g graph has cross-user relationships at initialization: "
                f"{graph_counts}"
            )
        self._write_event(
            "initialize",
            {
                "local_only": True,
                "vector_store": "faiss",
                "graph_store": "neo4j",
                "graph_url": self.graph_url,
                "graph_database": self.graph_database,
                "graph_counts": graph_counts,
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
                return self._record_noop_add(
                    text,
                    started,
                    "official_empty_initial_result_sentinel",
                )

            memory_count_before = self._memory_count()
            graph_before = self._graph_counts()
            messages = [{"role": "user", "content": text}]
            metadata = {
                "source": "memoryarena_official_memory_update",
                "add_call": self._add_calls,
                "user_id": self.user_id,
            }
            filters = {"user_id": self.user_id}

            consume_context_failure_events()
            vector_result = self.memory._add_to_vector_store(
                messages,
                metadata,
                filters,
                True,
            )
            vector_failures = consume_context_failure_events()
            if vector_failures:
                self._rollback_vector(vector_result, memory_count_before)
                return self._record_context_failure(
                    text=text,
                    started=started,
                    stage="vector",
                    failures=vector_failures,
                    graph_before=graph_before,
                    graph_after=self._graph_counts(),
                )

            self._consume_graph_llm_failure_events()
            graph_result = self.memory._add_to_graph(messages, filters)
            graph_failures = consume_context_failure_events()
            graph_llm_failures = self._consume_graph_llm_failure_events()
            graph_after = self._graph_counts()
            if graph_failures:
                if graph_after["cross_user_relationship_count"]:
                    raise RuntimeError(
                        "Mem0g graph context failure caused cross-user "
                        f"contamination: {graph_after}"
                    )
                return self._record_graph_context_failure(
                    text=text,
                    started=started,
                    failures=graph_failures,
                    vector_result=vector_result,
                    memory_count_before=memory_count_before,
                    graph_result=graph_result,
                    graph_before=graph_before,
                    graph_after=graph_after,
                )

            tool_output_failures = [
                failure
                for failure in graph_llm_failures
                if failure.get("kind") == "tool_arguments_json_eof"
            ]
            unexpected_graph_failures = [
                failure
                for failure in graph_llm_failures
                if failure.get("kind")
                not in {"context_limit", "tool_arguments_json_eof"}
            ]
            if unexpected_graph_failures:
                vector_outcome = self._vector_outcome(
                    vector_result,
                    memory_count_before,
                    self._memory_count(),
                )
                self._graph_add_failures += 1
                self._write_event(
                    "graph_add_failure",
                    {
                        "call_index": self._add_calls,
                        "reason": "unexpected_graph_llm_error",
                        **vector_outcome,
                        "failures": _json_safe(unexpected_graph_failures),
                        "memory_count_before": memory_count_before,
                        "memory_count_after": self._memory_count(),
                        "graph_counts_before": graph_before,
                        "graph_counts_after": graph_after,
                    },
                )
                self._write_manifest()
                raise RuntimeError(
                    "Mem0g graph LLM failed unexpectedly; "
                    "vector write retained for audit"
                )
            if tool_output_failures:
                if graph_after["cross_user_relationship_count"]:
                    raise RuntimeError(
                        "Mem0g graph tool-output failure caused cross-user "
                        f"contamination: {graph_after}"
                    )
                return self._record_graph_tool_output_failure(
                    text=text,
                    started=started,
                    failures=tool_output_failures,
                    vector_result=vector_result,
                    memory_count_before=memory_count_before,
                    graph_result=graph_result,
                    graph_before=graph_before,
                    graph_after=graph_after,
                )

            if not isinstance(graph_result, dict):
                vector_outcome = self._vector_outcome(
                    vector_result,
                    memory_count_before,
                    self._memory_count(),
                )
                self._graph_add_failures += 1
                self._write_event(
                    "graph_add_failure",
                    {
                        "call_index": self._add_calls,
                        "reason": "non_auditable_graph_result",
                        **vector_outcome,
                        "memory_count_before": memory_count_before,
                        "memory_count_after": self._memory_count(),
                        "graph_counts_before": graph_before,
                        "graph_counts_after": graph_after,
                        "graph_response": _json_safe(graph_result),
                    },
                )
                self._write_manifest()
                raise RuntimeError(
                    "Mem0g graph add did not return an auditable result: "
                    f"{_json_safe(graph_result)}; vector write retained for audit"
                )
            if graph_after["cross_user_relationship_count"]:
                raise RuntimeError(
                    f"Mem0g graph cross-user contamination detected: {graph_after}"
                )

            self._stored_add_calls += 1
            memory_count = self._memory_count()
            latency = time.perf_counter() - started
            response = {
                "results": _json_safe(vector_result),
                "relations": _json_safe(graph_result),
            }
            payload = {
                "call_index": self._add_calls,
                "stored": True,
                "chunk_chars": len(text),
                "chunk_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "memory_count": memory_count,
                "latency_seconds": round(latency, 6),
                "graph_counts_before": graph_before,
                "graph_counts_after": graph_after,
                "response": response,
            }
            self._write_event("add", payload)
            self._write_manifest()

        return {
            "stored": True,
            "add_call": self._add_calls,
            "memory_count": memory_count,
            "index_dir": str(self.index_dir),
            "latency_seconds": round(latency, 6),
            "mem0g_response": response,
            "graph_counts": graph_after,
        }

    def wrap_user_prompt(self, prompt: str) -> str:
        question = str(prompt or "")
        started = time.perf_counter()
        with self._lock:
            consume_context_failure_events()
            response = self.memory.search(
                query=question,
                user_id=self.user_id,
                limit=self.top_k,
            )
            context_failures = consume_context_failure_events()
            if context_failures:
                self._method_context_failure_count += len(context_failures)
                self._graph_retrieval_context_failure_count += len(context_failures)

            results = self._results(response)
            relations = self._relations(response)
            graph_counts = self._graph_counts()
            if graph_counts["cross_user_relationship_count"]:
                raise RuntimeError(
                    f"Mem0g graph cross-user contamination detected: {graph_counts}"
                )

            self._wrap_calls += 1
            self._graph_relations_returned += len(relations)
            latency = time.perf_counter() - started
            context_lines = self._format_results(results)
            context_lines.extend(self._format_relations(relations))
            retrieval_payload = {
                "call_index": self._wrap_calls,
                "retrieve_k": self.top_k,
                "latency_seconds": round(latency, 6),
                "query_preview": question[:2000],
                "vector_result_count": len(results),
                "graph_relation_count": len(relations),
                "results": [_json_safe(item) for item in results],
                "relations": [_json_safe(item) for item in relations],
                "graph_counts": graph_counts,
                "context_failures": _json_safe(context_failures),
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
            if self._closed:
                return {
                    "closed": True,
                    "completed": bool(completed),
                    "state_preserved": True,
                    "already_closed": True,
                }

            memory_count = self._memory_count()
            graph_counts = self._graph_counts()
            self._write_event(
                "close",
                {
                    "memory_count": memory_count,
                    "graph_counts": graph_counts,
                },
            )
            self._write_manifest(
                memory_count_override=memory_count,
                graph_counts_override=graph_counts,
            )
            self._closed = True

            graph_wrapper = getattr(
                getattr(self.memory, "graph", None),
                "graph",
                None,
            )
            driver = getattr(graph_wrapper, "_driver", None)
            if driver is not None and hasattr(driver, "close"):
                driver.close()
            self.memory.close()
        return {
            "closed": True,
            "completed": bool(completed),
            "state_preserved": True,
            "resources_closed": ["mem0g.graph", "mem0g.memory"],
        }

    def _record_noop_add(
        self,
        text: str,
        started: float,
        reason: str,
    ) -> Dict[str, Any]:
        latency = time.perf_counter() - started
        graph_counts = self._graph_counts()
        payload = {
            "call_index": self._add_calls,
            "stored": False,
            "reason": reason,
            "chunk_chars": len(text),
            "chunk_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "memory_count": self._memory_count(),
            "graph_counts": graph_counts,
            "latency_seconds": round(latency, 6),
        }
        self._write_event("add", payload)
        self._write_manifest()
        return payload

    def _require_graph_tool_calls(self) -> None:
        graph_llm = getattr(getattr(self.memory, "graph", None), "llm", None)
        generate_response = getattr(graph_llm, "generate_response", None)
        if not callable(generate_response):
            raise RuntimeError("Mem0g graph LLM is missing generate_response")

        def required_generate_response(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("tools"):
                kwargs["tool_choice"] = "required"
            try:
                return generate_response(*args, **kwargs)
            except Exception as exc:
                self._graph_llm_failure_events.append(
                    _classify_graph_llm_exception(exc)
                )
                raise

        graph_llm.generate_response = required_generate_response
        self._graph_tool_choice = "required"

    def _consume_graph_llm_failure_events(self) -> List[Dict[str, str]]:
        failures = list(self._graph_llm_failure_events)
        self._graph_llm_failure_events.clear()
        return failures

    def _record_context_failure(
        self,
        *,
        text: str,
        started: float,
        stage: str,
        failures: List[Dict[str, Any]],
        graph_before: Dict[str, int],
        graph_after: Dict[str, int],
    ) -> Dict[str, Any]:
        self._failed_add_calls += 1
        self._method_context_failure_count += len(failures)
        latency = time.perf_counter() - started
        payload = {
            "call_index": self._add_calls,
            "stored": False,
            "reason": "method_context_limit",
            "failure_stage": stage,
            "chunk_chars": len(text),
            "chunk_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "memory_count": self._memory_count(),
            "latency_seconds": round(latency, 6),
            "context_failures": _json_safe(failures),
            "graph_counts_before": graph_before,
            "graph_counts_after": graph_after,
        }
        self._write_event("add", payload)
        self._write_manifest()
        return payload

    def _record_graph_context_failure(
        self,
        *,
        text: str,
        started: float,
        failures: List[Dict[str, Any]],
        vector_result: Any,
        memory_count_before: int,
        graph_result: Any,
        graph_before: Dict[str, int],
        graph_after: Dict[str, int],
    ) -> Dict[str, Any]:
        memory_count = self._memory_count()
        vector_outcome = self._vector_outcome(
            vector_result,
            memory_count_before,
            memory_count,
        )
        vector_mutated = bool(vector_outcome["vector_mutated"])
        if vector_mutated:
            self._stored_add_calls += 1
        else:
            self._noop_add_calls += 1
            self._partial_graph_vector_noop_calls += 1
        self._graph_add_failures += 1
        self._partial_graph_add_calls += 1
        self._method_context_failure_count += len(failures)

        latency = time.perf_counter() - started
        response = {
            "results": _json_safe(vector_result),
            "relations": _json_safe(graph_result),
        }
        payload = {
            "call_index": self._add_calls,
            "stored": vector_mutated,
            **vector_outcome,
            "graph_stored": graph_after != graph_before,
            "reason": "graph_context_limit_vector_retained",
            "failure_stage": "graph",
            "chunk_chars": len(text),
            "chunk_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "memory_count_before": memory_count_before,
            "memory_count": memory_count,
            "latency_seconds": round(latency, 6),
            "context_failures": _json_safe(failures),
            "graph_counts_before": graph_before,
            "graph_counts_after": graph_after,
            "response": response,
        }
        self._write_event("add", payload)
        self._write_manifest()
        return {
            "stored": vector_mutated,
            "reason": "graph_context_limit_vector_retained",
            **vector_outcome,
            "add_call": self._add_calls,
            "memory_count": memory_count,
            "index_dir": str(self.index_dir),
            "latency_seconds": round(latency, 6),
            "context_failures": _json_safe(failures),
            "mem0g_response": response,
            "graph_counts": graph_after,
        }

    def _record_graph_tool_output_failure(
        self,
        *,
        text: str,
        started: float,
        failures: List[Dict[str, str]],
        vector_result: Any,
        memory_count_before: int,
        graph_result: Any,
        graph_before: Dict[str, int],
        graph_after: Dict[str, int],
    ) -> Dict[str, Any]:
        memory_count = self._memory_count()
        vector_outcome = self._vector_outcome(
            vector_result,
            memory_count_before,
            memory_count,
        )
        vector_mutated = bool(vector_outcome["vector_mutated"])
        if vector_mutated:
            self._stored_add_calls += 1
        else:
            self._noop_add_calls += 1
            self._partial_graph_vector_noop_calls += 1
        self._graph_add_failures += 1
        self._partial_graph_add_calls += 1
        self._graph_tool_output_failures += len(failures)

        latency = time.perf_counter() - started
        response = {
            "results": _json_safe(vector_result),
            "relations": _json_safe(graph_result),
        }
        payload = {
            "call_index": self._add_calls,
            "stored": vector_mutated,
            **vector_outcome,
            "graph_stored": graph_after != graph_before,
            "reason": "graph_tool_arguments_eof_vector_retained",
            "failure_stage": "graph_tool_output",
            "chunk_chars": len(text),
            "chunk_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "memory_count_before": memory_count_before,
            "memory_count": memory_count,
            "latency_seconds": round(latency, 6),
            "graph_llm_failures": _json_safe(failures),
            "graph_counts_before": graph_before,
            "graph_counts_after": graph_after,
            "response": response,
        }
        self._write_event("add", payload)
        self._write_manifest()
        return {
            "stored": vector_mutated,
            "reason": "graph_tool_arguments_eof_vector_retained",
            **vector_outcome,
            "add_call": self._add_calls,
            "memory_count": memory_count,
            "index_dir": str(self.index_dir),
            "latency_seconds": round(latency, 6),
            "graph_llm_failures": _json_safe(failures),
            "mem0g_response": response,
            "graph_counts": graph_after,
        }

    def _rollback_vector(self, result: Any, expected_count: int) -> None:
        rows = result if isinstance(result, list) else []
        for item in reversed(rows):
            if isinstance(item, dict) and item.get("id"):
                self.memory.delete(str(item["id"]))
        actual = self._memory_count()
        if actual != expected_count:
            raise RuntimeError(
                "Mem0g vector rollback failed: "
                f"expected={expected_count} actual={actual} result={_json_safe(result)}"
            )

    def _results(self, response: Any) -> List[Dict[str, Any]]:
        if isinstance(response, dict):
            raw = response.get("results") or []
        elif isinstance(response, list):
            raw = response
        else:
            raw = []
        return [item for item in raw if isinstance(item, dict)]

    def _relations(self, response: Any) -> List[Dict[str, Any]]:
        raw = response.get("relations") if isinstance(response, dict) else []
        return [item for item in (raw or []) if isinstance(item, dict)]

    def _format_results(self, results: List[Dict[str, Any]]) -> List[str]:
        lines = []
        for rank, item in enumerate(results, start=1):
            text = item.get("memory") or item.get("text") or item.get("data")
            if not text:
                continue
            attributes = [f'rank="{rank}"']
            if item.get("score") is not None:
                attributes.append(f'score="{escape(str(item["score"]))}"')
            if item.get("id"):
                attributes.append(f'id="{escape(str(item["id"]))}"')
            lines.append(
                f"<memory {' '.join(attributes)}>{escape(str(text))}</memory>"
            )
        return lines

    def _format_relations(self, relations: List[Dict[str, Any]]) -> List[str]:
        lines = []
        for rank, item in enumerate(relations, start=1):
            source = str(item.get("source") or "")
            relationship = str(item.get("relationship") or "")
            destination = str(item.get("destination") or item.get("target") or "")
            if not source or not relationship or not destination:
                continue
            relation_text = f"{source} {relationship} {destination}"
            lines.append(
                f'<graph_relation rank="{rank}">{escape(relation_text)}</graph_relation>'
            )
        return lines

    def _memory_count(self) -> int:
        vector_store = getattr(self.memory, "vector_store", None)
        index = getattr(vector_store, "index", None)
        return int(getattr(index, "ntotal", 0) or 0)

    def _vector_outcome(
        self,
        result: Any,
        memory_count_before: int,
        memory_count_after: int,
    ) -> Dict[str, Any]:
        if not isinstance(result, list):
            raise RuntimeError(
                "Mem0g vector phase returned a non-list result: "
                f"{_json_safe(result)}"
            )

        event_counts: Dict[str, int] = {}
        mutation_count = 0
        for item in result:
            if not isinstance(item, dict):
                raise RuntimeError(
                    "Mem0g vector phase returned a non-object row: "
                    f"{_json_safe(item)}"
                )
            event = str(item.get("event") or "").upper()
            event_counts[event or "UNSPECIFIED"] = (
                event_counts.get(event or "UNSPECIFIED", 0) + 1
            )
            if event in {"ADD", "UPDATE", "DELETE"}:
                mutation_count += 1

        vector_noop = mutation_count == 0
        memory_count_delta = memory_count_after - memory_count_before
        if vector_noop and memory_count_delta != 0:
            raise RuntimeError(
                "Mem0g vector no-op changed memory count: "
                f"before={memory_count_before} after={memory_count_after} "
                f"result={_json_safe(result)}"
            )

        return {
            "vector_phase_completed": True,
            "vector_mutated": not vector_noop,
            "vector_stored": not vector_noop,
            "vector_noop": vector_noop,
            "vector_mutation_count": mutation_count,
            "vector_event_counts": event_counts,
            "memory_count_delta": memory_count_delta,
        }

    def _graph_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        graph = getattr(getattr(self.memory, "graph", None), "graph", None)
        if graph is None:
            raise RuntimeError("Mem0g Neo4j graph handle is missing")
        rows = graph.query(query, params=params or {})
        return rows if isinstance(rows, list) else []

    def _graph_counts(self, best_effort: bool = False) -> Dict[str, int]:
        try:
            node_rows = self._graph_query(
                """
                MATCH (n)
                RETURN count(n) AS total_nodes,
                       count(CASE WHEN n.user_id = $user_id THEN 1 END) AS user_nodes,
                       count(DISTINCT n.user_id) AS distinct_users
                """,
                {"user_id": self.user_id},
            )
            relation_rows = self._graph_query(
                """
                MATCH (a)-[r]->(b)
                RETURN count(r) AS total_relationships,
                       count(CASE
                           WHEN a.user_id = $user_id AND b.user_id = $user_id
                           THEN 1
                       END) AS user_relationships,
                       count(CASE
                           WHEN coalesce(a.user_id, '') <> coalesce(b.user_id, '')
                           THEN 1
                       END) AS cross_user_relationships
                """,
                {"user_id": self.user_id},
            )
            nodes = node_rows[0] if node_rows else {}
            relations = relation_rows[0] if relation_rows else {}
            return {
                "total_node_count": int(nodes.get("total_nodes") or 0),
                "user_node_count": int(nodes.get("user_nodes") or 0),
                "distinct_user_count": int(nodes.get("distinct_users") or 0),
                "total_relationship_count": int(
                    relations.get("total_relationships") or 0
                ),
                "user_relationship_count": int(
                    relations.get("user_relationships") or 0
                ),
                "cross_user_relationship_count": int(
                    relations.get("cross_user_relationships") or 0
                ),
            }
        except Exception:
            if best_effort:
                return {
                    "total_node_count": -1,
                    "user_node_count": -1,
                    "distinct_user_count": -1,
                    "total_relationship_count": -1,
                    "user_relationship_count": -1,
                    "cross_user_relationship_count": -1,
                }
            raise

    def _write_event(self, event: str, payload: Dict[str, Any]) -> None:
        record = {
            "event": event,
            "timestamp": time.time(),
            "user_id": self.user_id,
            **_json_safe(payload),
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_manifest(
        self,
        best_effort: bool = False,
        memory_count_override: Optional[int] = None,
        graph_counts_override: Optional[Dict[str, int]] = None,
    ) -> None:
        index_files = sorted(str(path) for path in self.index_dir.glob("*"))
        payload = {
            "method": "mem0g_indepth",
            "implementation": "local_mem0_staged_vector_and_neo4j_graph_v5",
            "local_only": True,
            "cloud_client_used": False,
            "user_id": self.user_id,
            "model": self.model,
            "llm_endpoint": self.base_url,
            "vector_store": "faiss",
            "graph_store": "neo4j",
            "graph_tool_choice": self._graph_tool_choice,
            "graph_url": self.graph_url,
            "graph_database": self.graph_database,
            "graph_embedder": os.getenv(
                "MEM0G_GRAPH_EMBEDDER",
                "sentence-transformers/all-mpnet-base-v2",
            ),
            "vector_embedder": "sentence-transformers/all-MiniLM-L6-v2",
            "retrieve_k": self.top_k,
            "memory_count": (
                self._memory_count()
                if memory_count_override is None
                else memory_count_override
            ),
            "graph_counts": (
                self._graph_counts(best_effort=best_effort)
                if graph_counts_override is None
                else graph_counts_override
            ),
            "add_calls": self._add_calls,
            "stored_add_calls": self._stored_add_calls,
            "noop_add_calls": self._noop_add_calls,
            "failed_add_calls": self._failed_add_calls,
            "graph_add_failures": self._graph_add_failures,
            "partial_graph_add_calls": self._partial_graph_add_calls,
            "partial_graph_vector_noop_calls": (
                self._partial_graph_vector_noop_calls
            ),
            "graph_tool_output_failures": self._graph_tool_output_failures,
            "method_context_failure_count": self._method_context_failure_count,
            "graph_retrieval_context_failure_count": (
                self._graph_retrieval_context_failure_count
            ),
            "wrap_calls": self._wrap_calls,
            "graph_relations_returned": self._graph_relations_returned,
            "created_at": self._created_at,
            "updated_at": time.time(),
            "state_dir": str(self.state_dir),
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
    """Create Mem0g through the common MemoryArena factory."""

    return Mem0gMemorySystem(user_id=user_id, top_k=top_k)
