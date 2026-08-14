"""Shared MemoryArena prompt contract for every local memory method."""

import os
import re
from html import escape
from typing import Iterable, List


DEFAULT_CONTEXT_CHAR_BUDGET = 8000
_CONTEXT_RE = re.compile(
    r"<memory_context>(.*?)</memory_context>",
    flags=re.IGNORECASE | re.DOTALL,
)
_MEMORY_RE = re.compile(
    r"<memory(?:\s+[^>]*)?>(.*?)</memory>",
    flags=re.IGNORECASE | re.DOTALL,
)


def context_char_budget() -> int:
    value = int(
        os.getenv(
            "MEMORYARENA_CONTEXT_CHAR_BUDGET",
            str(DEFAULT_CONTEXT_CHAR_BUDGET),
        )
    )
    if value <= 0:
        raise ValueError("MEMORYARENA_CONTEXT_CHAR_BUDGET must be positive")
    return value


def extract_memory_entries(backend_prompt: str) -> List[str]:
    """Extract only retrieved evidence, excluding method-specific prompt suffixes."""
    text = str(backend_prompt or "")
    context_match = _CONTEXT_RE.search(text)
    if context_match:
        context = context_match.group(1).strip()
    else:
        context = re.split(
            r"(?:<memory_retrieval_latency_ms>|\n\s*User Prompt:|\n\s*User:)",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
    if not context or context.casefold() == "none":
        return []

    entries = [item.strip() for item in _MEMORY_RE.findall(context) if item.strip()]
    return entries or [context]


def _clip_entries(entries: Iterable[str], max_context_chars: int) -> List[str]:
    clipped: List[str] = []
    used = 0
    marker = "... [truncated]"
    for value in entries:
        remaining = max_context_chars - used
        if remaining <= 0:
            break
        item = str(value).strip()
        if len(item) > remaining:
            if remaining > len(marker):
                item = item[: remaining - len(marker)] + marker
            else:
                item = item[:remaining]
        if item:
            clipped.append(item)
            used += len(item)
    return clipped


def canonicalize_memory_prompt(
    question: str,
    backend_prompt: str,
    *,
    max_context_chars: int,
) -> str:
    """Return the identical evidence envelope consumed by every task agent."""
    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be positive")
    entries = _clip_entries(
        extract_memory_entries(backend_prompt),
        max_context_chars,
    )
    lines = ["<memory_context>"]
    if entries:
        lines.extend(
            f'<memory rank="{rank}">{escape(entry, quote=False)}</memory>'
            for rank, entry in enumerate(entries, start=1)
        )
    else:
        lines.append("None")
    lines.extend(("</memory_context>", f"User Prompt: {question}"))
    return "\n".join(lines)
