"""Lazy registry for the twelve unified MemoryArena baseline backends."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class BackendSpec:
    label: str
    key: str
    package: str


BACKENDS: tuple[BackendSpec, ...] = (
    BackendSpec("A-MEM", "amem", "amem"),
    BackendSpec("MemoryBank", "memorybank", "memorybank"),
    BackendSpec("MemGPT", "memgpt", "memgpt"),
    BackendSpec("Mem0", "mem0", "mem0"),
    BackendSpec("Mem0g", "mem0g", "mem0g"),
    BackendSpec("MemoChat", "memochat", "memochat"),
    BackendSpec("Zep", "zep", "zep"),
    BackendSpec("MemTree", "memtree", "memtree"),
    BackendSpec("MemoryOS", "memoryos", "memoryos"),
    BackendSpec("MemOS", "memos", "memos"),
    BackendSpec("MemGAS", "memgas", "memgas"),
    BackendSpec("LightMem", "lightmem", "lightmem"),
)

CANONICAL_BASELINES = {spec.label: spec.key for spec in BACKENDS}
BACKEND_KEYS = tuple(spec.key for spec in BACKENDS)
_PACKAGES = {spec.key: spec.package for spec in BACKENDS}

ALIASES = {
    "amem_indepth": "amem",
    "memorybank_indepth": "memorybank",
    "letta": "memgpt",
    "mem0_indepth": "mem0",
    "mem0g_indepth": "mem0g",
    "mem0-g": "mem0g",
}


def canonical_key(name: str) -> str:
    normalized = str(name).strip()
    key = ALIASES.get(normalized, normalized)
    if key not in _PACKAGES:
        raise KeyError(f"Unsupported MemoryArena baseline: {name}")
    return key


def load_factory(name: str) -> Callable[..., object]:
    key = canonical_key(name)
    module = importlib.import_module(
        f"{__package__}.{_PACKAGES[key]}.backend"
    )
    factory = getattr(module, "create_backend", None)
    if not callable(factory):
        raise TypeError(f"{module.__name__} does not export create_backend()")
    return factory


def create_backend(
    name: str,
    *,
    user_id: Optional[str] = None,
    top_k: int = 10,
) -> object:
    return load_factory(name)(user_id=user_id, top_k=top_k)
