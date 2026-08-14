"""MemoryArena adapter for the native MemoChat summary/retrieval pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .main import build_tokenizer, run_retrieval, run_summary, write_json_atomic

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


class MemoChatMemorySystem:
    """Persist MemoChat topics per sample and expose only retrieved evidence."""

    def __init__(self, user_id: Optional[str] = None, top_k: int = 10):
        self.user_id = str(user_id or f"memoryarena_{time.time_ns()}")
        self.top_k = resolve_top_k(top_k, "MEMOCHAT_RETRIEVAL_TOP_K")
        self.context_char_budget = resolve_context_char_budget(
            "MEMOCHAT_CONTEXT_CHAR_BUDGET"
        )
        self._lock = threading.RLock()
        self._close_state = PreservingCloseState()

        state_root = resolve_state_root(
            "memochat", "MEMOCHAT_MEMORYARENA_DATA_ROOT"
        )
        self.storage_dir = state_root / safe_user_id(self.user_id, limit=160)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.storage_dir / "state.json"

        prompt_path = Path(
            os.getenv(
                "MEMOCHAT_PROMPT_PATH",
                str(Path(__file__).resolve().with_name("prompts.json")),
            )
        ).expanduser()
        if not prompt_path.is_file():
            raise FileNotFoundError(f"MemoChat prompt file not found: {prompt_path}")
        self.prompts = json.loads(prompt_path.read_text(encoding="utf-8"))

        self.model = os.getenv("MEMOCHAT_LLM_MODEL") or os.getenv(
            "MEMORYARENA_LLM_MODEL", "Qwen3.5-9B"
        )
        self.client = OpenAI(
            api_key=os.getenv("MEMOCHAT_LLM_API_KEY")
            or os.getenv("MEMORYARENA_LLM_API_KEY", "EMPTY"),
            base_url=select_endpoint(
                self.user_id,
                kind="llm",
                method_list_env="MEMOCHAT_LLM_BASE_URLS",
                method_single_env="MEMOCHAT_LLM_BASE_URL",
            ),
        )
        self.encoding = build_tokenizer(self.model)
        self._summary_word_threshold = int(
            os.getenv("MEMOCHAT_SUMMARY_WORD_THRESHOLD", "1024")
        )
        self._summary_turn_threshold = int(
            os.getenv("MEMOCHAT_SUMMARY_TURN_THRESHOLD", "10")
        )
        if self._summary_word_threshold <= 0 or self._summary_turn_threshold <= 0:
            raise ValueError("MemoChat summary thresholds must be positive")

        self.history = {
            "Recent Dialogs": [],
            "Related Topics": [],
            "Related Summaries": [],
            "Related Dialogs": [],
            "User Input": "",
        }
        self.memo = {"NOTO": []}
        self.bot_thinking = {"retrieval": "", "summarization": ""}
        self._added_hashes: set[str] = set()
        self._load_state()

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        history = payload.get("history")
        memo = payload.get("memo")
        hashes = payload.get("added_hashes")
        if not isinstance(history, dict) or not isinstance(memo, dict):
            raise ValueError(f"invalid MemoChat state: {self.state_path}")
        if not isinstance(hashes, list) or not all(isinstance(item, str) for item in hashes):
            raise ValueError(f"invalid MemoChat add journal: {self.state_path}")
        self.history.update(history)
        self.memo = memo
        self.memo.setdefault("NOTO", [])
        self._added_hashes = set(hashes)

    def _save_state(self) -> None:
        write_json_atomic(
            str(self.state_path),
            {
                "schema_version": "memoryarena.memochat.v1",
                "user_id": self.user_id,
                "history": self.history,
                "memo": self.memo,
                "added_hashes": sorted(self._added_hashes),
            },
        )

    def _needs_summary(self) -> bool:
        recent = self.history["Recent Dialogs"]
        return (
            len(" ### ".join(recent).split()) > self._summary_word_threshold
            or len(recent) >= self._summary_turn_threshold
        )

    def _summarize_recent(self) -> None:
        self.history, self.memo, self.bot_thinking = run_summary(
            self.history,
            self.memo,
            self.bot_thinking,
            self.prompts,
            self.client,
            self.model,
            self.encoding,
        )

    def add_chunk(self, chunk: str):
        text = normalize_chunk(chunk)
        if text is None:
            return {"stored": False, "skipped": True, "characters": 0}
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock:
            self._close_state.ensure_open(f"MemoChat memory {self.user_id}")
            if digest in self._added_hashes:
                return {"stored": False, "duplicate": True, "characters": len(text)}
            self.history["Recent Dialogs"].append(text)
            if self._needs_summary():
                self._summarize_recent()
            self._added_hashes.add(digest)
            self._save_state()
        return {"stored": True, "characters": len(text)}

    def wrap_user_prompt(self, prompt: str) -> str:
        with self._lock:
            self._close_state.ensure_open(f"MemoChat memory {self.user_id}")
            memo = copy.deepcopy(self.memo)
            recent = list(self.history.get("Recent Dialogs") or [])
            if recent:
                memo["RECENT"] = [
                    {
                        "summary": "Recent unsummarized task conversation.",
                        "dialogs": recent,
                    }
                ]
            available = any(values for topic, values in memo.items() if topic != "NOTO")
            if not available:
                return format_memory_prompt(
                    prompt,
                    [],
                    char_budget=self.context_char_budget,
                )
            query_history = {
                "Recent Dialogs": [],
                "Related Topics": [],
                "Related Summaries": [],
                "Related Dialogs": [],
                "User Input": "user: " + str(prompt),
            }
            retrieval_trace = {"retrieval": "", "summarization": ""}
            query_history, _ = run_retrieval(
                query_history,
                memo,
                retrieval_trace,
                self.prompts,
                self.client,
                self.model,
                self.encoding,
                retrieve_top_k=self.top_k,
            )
            entries = [
                f"Topic: {topic}\nSummary: {summary}\nDialogs: {dialogs}"
                for topic, summary, dialogs in zip(
                    query_history["Related Topics"],
                    query_history["Related Summaries"],
                    query_history["Related Dialogs"],
                )
            ]
        return format_memory_prompt(
            prompt,
            entries,
            char_budget=self.context_char_budget,
        )

    def _release_resources(self):
        self._save_state()
        return close_resource(self.client, "memochat.llm_client", seen=set())

    def close(self, completed: bool = False):
        return self._close_state.close(completed, self._release_resources)


def create_backend(*, user_id=None, top_k=10):
    """Create MemoChat through the common MemoryArena factory."""

    return MemoChatMemorySystem(user_id=user_id, top_k=top_k)
