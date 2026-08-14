"""Dependency-free contract helpers shared by local MemoryArena adapters."""

from __future__ import annotations

import hashlib
import inspect
import os
import re
import threading
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol, runtime_checkable
from urllib.parse import urlparse


INITIAL_RESULT_SENTINEL = "Initial result: Empty"
DEFAULT_CONTEXT_CHAR_BUDGET = 8000


@runtime_checkable
class MemoryArenaBackend(Protocol):
    """Structural interface implemented by every local MemoryArena backend."""

    user_id: str
    top_k: int

    def add_chunk(self, chunk: str) -> dict:
        ...

    def wrap_user_prompt(self, prompt: str) -> str:
        ...

    def close(self, completed: bool = False) -> dict:
        ...


def safe_user_id(value: object, limit: int = 120) -> str:
    """Return a filesystem- and backend-safe stable identifier."""

    if limit <= 0:
        raise ValueError("safe user id limit must be positive")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))
    return cleaned[:limit] or "anonymous"


def normalize_chunk(chunk: object) -> Optional[str]:
    """Normalize a memory write and suppress the shared empty-result sentinel."""

    text = str(chunk or "").strip()
    if not text or text == INITIAL_RESULT_SENTINEL:
        return None
    return text


def _configured_value(*names: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    for name in names:
        if not name:
            continue
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip(), name
    return None, None


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer, got {value!r}") from exc
    if str(parsed) != str(value).strip() and not isinstance(value, int):
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    if parsed <= 0:
        raise ValueError(f"{label} must be positive, got {parsed}")
    return parsed


def resolve_top_k(top_k: object, method_env: Optional[str] = None) -> int:
    """Resolve and validate the common retrieval budget."""

    configured, source = _configured_value(method_env, "MEMORYARENA_RETRIEVAL_TOP_K")
    value = configured if configured is not None else top_k
    label = source or "top_k"
    return _positive_int(value, label)


def resolve_context_char_budget(method_env: Optional[str] = None) -> int:
    configured, source = _configured_value(
        method_env,
        "MEMORYARENA_CONTEXT_CHAR_BUDGET",
    )
    value = configured if configured is not None else DEFAULT_CONTEXT_CHAR_BUDGET
    return _positive_int(value, source or "context character budget")


def resolve_state_root(method: str, method_env: Optional[str] = None) -> Path:
    """Resolve a portable, method-isolated state directory without creating it."""

    method_name = safe_user_id(method, limit=80)
    method_value, _ = _configured_value(method_env)
    if method_value is not None:
        return Path(method_value).expanduser().absolute()

    shared_value, _ = _configured_value("MEMORYARENA_STATE_ROOT")
    if shared_value is not None:
        return (Path(shared_value).expanduser() / method_name).absolute()

    return (Path.home() / ".local" / "state" / "memoryarena" / method_name).absolute()


def _validate_endpoint(value: str, source: str) -> str:
    endpoint = value.strip().rstrip("/")
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"{source} must contain an explicit HTTP(S) endpoint, got {value!r}"
        )
    return endpoint


def select_endpoint(
    user_id: object,
    *,
    kind: str,
    method_list_env: Optional[str] = None,
    method_single_env: Optional[str] = None,
) -> str:
    """Select an explicitly configured endpoint with a shared local fallback."""

    normalized_kind = kind.strip().upper()
    if normalized_kind not in {"LLM", "EMBEDDING"}:
        raise ValueError(f"unsupported endpoint kind: {kind!r}")

    value, source = _configured_value(
        method_list_env,
        method_single_env,
        f"MEMORYARENA_{normalized_kind}_BASE_URLS",
        f"MEMORYARENA_{normalized_kind}_BASE_URL",
    )
    if value is None or source is None:
        expected = [
            name
            for name in (
                method_list_env,
                method_single_env,
                f"MEMORYARENA_{normalized_kind}_BASE_URLS",
                f"MEMORYARENA_{normalized_kind}_BASE_URL",
            )
            if name
        ]
        raise RuntimeError(
            f"{normalized_kind} endpoint is not configured; set one of: "
            + ", ".join(expected)
        )

    endpoints = [
        _validate_endpoint(item, source)
        for item in value.split(",")
        if item.strip()
    ]
    if not endpoints:
        raise RuntimeError(f"{source} contains no endpoints")
    digest = hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()
    return endpoints[int(digest[:8], 16) % len(endpoints)]


def format_memory_prompt(
    prompt: object,
    memories: Iterable[object],
    *,
    char_budget: object,
) -> str:
    """Build the exact shared MemoryArena memory-context prompt."""

    budget = _positive_int(char_budget, "context character budget")
    entries = [str(item).strip() for item in memories if str(item).strip()]
    lines = ["<memory_context>"]
    used = 0
    emitted = 0
    truncation = "... [truncated]"

    for entry in entries:
        remaining = budget - used
        if remaining <= 0:
            break
        item = entry
        if len(item) > remaining:
            if remaining > len(truncation):
                item = item[: remaining - len(truncation)] + truncation
            else:
                item = item[:remaining]
        emitted += 1
        lines.append(f'<memory rank="{emitted}">{item}</memory>')
        used += len(item)

    if not emitted:
        lines.append("None")
    lines.extend(["</memory_context>", f"User Prompt: {prompt}"])
    return "\n".join(lines)


def close_resource(
    resource: object,
    label: str,
    *,
    seen: Optional[set[int]] = None,
) -> list[str]:
    """Close one explicit resource if it exposes a synchronous close method."""

    if resource is None:
        return []
    identity = id(resource)
    if seen is not None and identity in seen:
        return []
    close = getattr(resource, "close", None)
    if not callable(close):
        return []
    result = close()
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError(f"{label}.close() is asynchronous and needs an adapter closer")
    if seen is not None:
        seen.add(identity)
    return [label]


class PreservingCloseState:
    """Make adapter close idempotent while preserving every persisted state artifact."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._closing = False
        self._receipt: Optional[dict] = None

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._receipt is not None

    def ensure_open(self, label: str) -> None:
        with self._lock:
            if self._closing:
                raise RuntimeError(f"{label} is closing")
            if self._receipt is not None:
                raise RuntimeError(f"{label} is already closed")

    def close(
        self,
        completed: bool,
        release: Callable[[], Optional[Iterable[str]]],
    ) -> dict:
        with self._condition:
            while self._closing:
                self._condition.wait()
            if self._receipt is not None:
                return {**self._receipt, "already_closed": True}
            self._closing = True

        try:
            resources = list(release() or [])
        except BaseException:
            with self._condition:
                self._closing = False
                self._condition.notify_all()
            raise

        with self._condition:
            self._receipt = {
                "closed": True,
                "completed": bool(completed),
                "state_preserved": True,
                "resources_closed": resources,
            }
            self._closing = False
            self._condition.notify_all()
            return dict(self._receipt)
