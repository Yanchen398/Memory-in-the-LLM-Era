"""Unified local HTTP interface for the twelve MemoryArena baselines."""

from __future__ import annotations

import inspect
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from .prompt_contract import (
        canonicalize_memory_prompt,
        context_char_budget,
        extract_memory_entries,
    )
    from .registry import (
        ALIASES,
        BACKEND_KEYS,
        CANONICAL_BASELINES,
        canonical_key,
        create_backend,
    )
except ImportError:  # Support direct execution from the repository checkout.
    _CODE_ROOT = Path(__file__).resolve().parents[1]
    if str(_CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(_CODE_ROOT))
    from Method_memoryarena.prompt_contract import (
        canonicalize_memory_prompt,
        context_char_budget,
        extract_memory_entries,
    )
    from Method_memoryarena.registry import (
        ALIASES,
        BACKEND_KEYS,
        CANONICAL_BASELINES,
        canonical_key,
        create_backend,
    )


load_dotenv()


class InitializeRequest(BaseModel):
    user_id: str
    memory_system_name: str
    run_id: Optional[str] = None
    config_digest: Optional[str] = None


class AddRequest(BaseModel):
    user_id: str
    chunk: str
    memory_system_name: str
    op_id: Optional[str] = None
    seq: Optional[int] = None
    phase: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    user_id: str
    question: str
    memory_system_name: str


class CloseRequest(BaseModel):
    user_id: str
    memory_system_name: str
    completed: bool = False


def _retrieval_top_k() -> int:
    top_k = int(os.getenv("MEMORYARENA_RETRIEVAL_TOP_K", "10"))
    if top_k <= 0:
        raise ValueError("MEMORYARENA_RETRIEVAL_TOP_K must be positive")
    return top_k


CANONICAL_METHODS = BACKEND_KEYS
SUPPORTED_MEMORY_SYSTEMS = tuple(sorted(set(BACKEND_KEYS) | set(ALIASES)))

app = FastAPI(title="MemoryArena Memory Server")


@dataclass
class MemorySystemEntry:
    name: str
    system: object
    run_id: Optional[str] = None
    config_digest: Optional[str] = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    add_receipts: Dict[str, Dict[str, Any]] = field(default_factory=dict)


MEMORY_SYSTEMS: Dict[str, MemorySystemEntry] = {}
_INITIALIZE_LOCK = threading.RLock()


def _canonical_or_http(name: str) -> str:
    try:
        return canonical_key(name)
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported memory_system: {name}",
        ) from exc


def _get_memory(user_id: str, memory_system: str) -> MemorySystemEntry:
    entry = MEMORY_SYSTEMS.get(user_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="User not initialized")
    if entry.name != _canonical_or_http(memory_system):
        raise HTTPException(status_code=400, detail="Mismatched memory_system for user")
    return entry


def _close_memory_system(memory_system: object, completed: bool) -> object:
    close_method = getattr(memory_system, "close", None)
    if close_method is None:
        return {"closed": True, "completed": completed, "state_preserved": True}
    parameters = inspect.signature(close_method).parameters
    response = (
        close_method(completed=completed)
        if "completed" in parameters
        else close_method()
    )
    return response if response is not None else {
        "closed": True,
        "completed": completed,
        "state_preserved": True,
    }


@app.post("/memory/initialize")
def initialize(req: InitializeRequest):
    with _INITIALIZE_LOCK:
        name = _canonical_or_http(req.memory_system_name)
        previous = MEMORY_SYSTEMS.get(req.user_id)
        if previous is not None:
            compatible = (
                previous.name == name
                and (not req.run_id or not previous.run_id or previous.run_id == req.run_id)
                and (
                    not req.config_digest
                    or not previous.config_digest
                    or previous.config_digest == req.config_digest
                )
            )
            if compatible:
                return {
                    "status": "ok",
                    "user_id": req.user_id,
                    "memory_system_name": name,
                    "already_initialized": True,
                }
            raise HTTPException(
                status_code=409,
                detail="User is already initialized with a different memory configuration",
            )

        memory_system = create_backend(
            name,
            user_id=req.user_id,
            top_k=_retrieval_top_k(),
        )
        MEMORY_SYSTEMS[req.user_id] = MemorySystemEntry(
            name=name,
            system=memory_system,
            run_id=req.run_id,
            config_digest=req.config_digest,
        )
        return {
            "status": "ok",
            "user_id": req.user_id,
            "memory_system_name": name,
            "already_initialized": False,
        }


@app.post("/memory/add")
def add(req: AddRequest):
    entry = _get_memory(req.user_id, req.memory_system_name)
    with entry.lock:
        if req.op_id and req.op_id in entry.add_receipts:
            receipt = dict(entry.add_receipts[req.op_id])
            receipt["duplicate_request"] = True
            return receipt
        response = entry.system.add_chunk(req.chunk)
        outputs = {
            "status": "ok",
            "user_id": req.user_id,
            "op_id": req.op_id,
            "seq": req.seq,
            "phase": req.phase,
            "response": response,
            "duplicate_request": False,
        }
        if req.op_id:
            entry.add_receipts[req.op_id] = outputs
        return outputs


@app.post("/memory/wrap_user_prompt")
def wrap_user_prompt(req: QueryRequest):
    entry = _get_memory(req.user_id, req.memory_system_name)
    with entry.lock:
        started = time.perf_counter()
        backend_prompt = entry.system.wrap_user_prompt(req.question)
        latency_ms = (time.perf_counter() - started) * 1000.0
    retrieved = extract_memory_entries(backend_prompt)
    prompt = canonicalize_memory_prompt(
        req.question,
        backend_prompt,
        max_context_chars=context_char_budget(),
    )
    return {
        "status": "ok",
        "user_id": req.user_id,
        "prompt": prompt,
        "retrieved": retrieved,
        "retrieved_count": len(retrieved),
        "retrieval_latency_ms": round(latency_ms, 3),
    }


@app.post("/memory/close")
def close(req: CloseRequest):
    entry = _get_memory(req.user_id, req.memory_system_name)
    with entry.lock:
        response = _close_memory_system(entry.system, completed=req.completed)
    with _INITIALIZE_LOCK:
        if MEMORY_SYSTEMS.get(req.user_id) is entry:
            MEMORY_SYSTEMS.pop(req.user_id, None)
    return {"status": "ok", "user_id": req.user_id, "response": response}


@app.get("/memory/methods")
def methods():
    return {
        "canonical_methods": list(CANONICAL_METHODS),
        "canonical_baselines": dict(CANONICAL_BASELINES),
        "compatibility_aliases": dict(ALIASES),
        "proposed_methods": [],
        "supported_memory_systems": list(SUPPORTED_MEMORY_SYSTEMS),
        "retrieval_top_k": _retrieval_top_k(),
        "context_char_budget": context_char_budget(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
