"""Input, path, and evaluation helpers for benchmark adapters."""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SESSION_KEY_RE = re.compile(r"^session_(\d+)$")
RAW_RESULT_FILENAME = "result_raw.json"

_PLACEHOLDER_RE = re.compile(
    r"(?:^|[\\/])your(?:[\\/]|$)|<[^>]*(?:path|file|dir)[^>]*>|"
    r"\b(?:change[_ -]?me|replace[_ -]?me|your[_ -]?path)\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


def session_number(key: object) -> int | None:
    """Return the numeric suffix of an exact ``session_N`` key."""

    match = SESSION_KEY_RE.fullmatch(str(key))
    return int(match.group(1)) if match else None


def natural_session_keys(conversation: Mapping[str, Any]) -> list[str]:
    """Return exact session keys in numeric order, independent of dict order."""

    numbered = []
    for key, value in conversation.items():
        number = session_number(key)
        if number is not None and isinstance(value, list):
            numbered.append((number, str(key)))
    numbered.sort(key=lambda item: (item[0], item[1]))
    return [key for _, key in numbered]


def format_turn_text(turn: Mapping[str, Any]) -> str:
    """Render a turn without discarding an image caption."""

    text = str(turn.get("text") or "").strip()
    caption = str(turn.get("blip_caption") or "").strip()
    if not caption:
        return text
    caption_text = f"[Image description: {caption}]"
    return f"{text} {caption_text}".strip()


def _normalized_turns_for_session(
    conversation: Mapping[str, Any], session_key: str
) -> dict[str, Any] | None:
    raw_turns = conversation.get(session_key)
    if session_number(session_key) is None or not isinstance(raw_turns, list):
        return None
    timestamp = conversation.get(f"{session_key}_date_time")
    turns = []
    for turn_index, raw_turn in enumerate(raw_turns):
        if not isinstance(raw_turn, Mapping):
            raise ValueError(f"{session_key}[{turn_index}] must be an object")
        turns.append(
            {
                "speaker": str(raw_turn.get("speaker") or "Unknown"),
                "text": format_turn_text(raw_turn),
                "caption": str(raw_turn.get("blip_caption") or ""),
                "dia_id": raw_turn.get("dia_id"),
                "timestamp": raw_turn.get("timestamp", timestamp),
                "turn_index": turn_index,
            }
        )
    return {
        "session_key": session_key,
        "session_number": session_number(session_key),
        "timestamp": timestamp,
        "turns": turns,
    }


def normalized_session_turns(
    conversation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return lossless, ordered session records for method-specific adapters."""

    sessions = []
    for key in natural_session_keys(conversation):
        sessions.append(_normalized_turns_for_session(conversation, key))
    return sessions


def pair_session_turns(
    conversation: Mapping[str, Any],
    session_key: str,
    speaker_a: str,
    speaker_b: str,
) -> list[dict[str, Any]]:
    """Build A/B-shaped exchanges while preserving every source turn once.

    A chronological ``speaker_a`` -> ``speaker_b`` pair remains paired.  Any
    other turn becomes a one-sided exchange instead of being dropped or
    overwriting an earlier turn.
    """

    normalized = _normalized_turns_for_session(conversation, session_key)
    if normalized is None:
        return []

    turns = normalized["turns"]
    exchanges = []
    index = 0
    while index < len(turns):
        current = turns[index]
        following = turns[index + 1] if index + 1 < len(turns) else None
        if (
            current["speaker"] == speaker_a
            and following is not None
            and following["speaker"] == speaker_b
        ):
            exchanges.append(
                {
                    "query": f"{speaker_a}: {current['text']}",
                    "response": f"{speaker_b}: {following['text']}",
                    "source_turn_indices": [current["turn_index"], following["turn_index"]],
                    "timestamp": normalized["timestamp"],
                }
            )
            index += 2
            continue

        line = f"{current['speaker']}: {current['text']}"
        exchanges.append(
            {
                "query": line if current["speaker"] != speaker_b else "",
                "response": line if current["speaker"] == speaker_b else "",
                "source_turn_indices": [current["turn_index"]],
                "timestamp": normalized["timestamp"],
            }
        )
        index += 1
    return exchanges


def raw_result_path(path: str | os.PathLike[str] | None) -> str:
    """Return a path that cannot mislabel unsimplified predictions."""

    value = os.fspath(path) if path is not None else RAW_RESULT_FILENAME
    if not value or value.endswith((os.sep, "/", "\\")):
        return os.path.join(value or ".", RAW_RESULT_FILENAME)
    if not os.path.splitext(value)[1]:
        return os.path.join(value, RAW_RESULT_FILENAME)
    if os.path.basename(value).lower() == "result_simplified.json":
        return os.path.join(os.path.dirname(value), RAW_RESULT_FILENAME)
    return value


def reject_placeholder(value: str, *, field: str = "path") -> None:
    """Reject template paths before a run can write to an unintended place."""

    if _PLACEHOLDER_RE.search(value):
        raise ValueError(f"{field} contains an unresolved placeholder: {value!r}")


def resolve_required_endpoint(
    value: str | None,
    *,
    env_var: str = "MEMORY_EMBEDDING_BASE_URL",
    field: str = "embedding_base_url",
) -> str:
    """Resolve an explicit service endpoint without retired-port fallbacks."""

    resolved = str(value or os.getenv(env_var, "")).strip()
    if not resolved:
        raise ValueError(
            f"{field} is required; provide it explicitly or set {env_var}"
        )
    reject_placeholder(resolved, field=field)
    if not _URL_RE.match(resolved):
        raise ValueError(f"{field} must be an absolute URL: {resolved!r}")
    return resolved.rstrip("/")


def resolve_config_path(
    value: str | os.PathLike[str] | None,
    *,
    repo_root: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
    field: str = "path",
) -> str | None:
    """Resolve a config path deterministically.

    Relative paths are repository-relative by default.  ``config://`` is an
    explicit option for files stored beside a config.
    URLs are returned unchanged.
    """

    if value is None:
        return None
    raw = os.path.expandvars(os.path.expanduser(os.fspath(value).strip()))
    if not raw:
        return raw
    reject_placeholder(raw, field=field)

    repo = Path(repo_root).resolve()
    config_dir = Path(config_path).resolve().parent if config_path else repo
    if raw.startswith("repo://"):
        candidate = repo / raw[len("repo://") :]
    elif raw.startswith("config://"):
        candidate = config_dir / raw[len("config://") :]
    elif _URL_RE.match(raw):
        return raw
    else:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = repo / raw
    return os.path.normpath(str(candidate.resolve(strict=False)))


def _qa_identity(sample_id: str, qa: Mapping[str, Any], qa_index: int) -> tuple[str, str]:
    explicit = qa.get("qa_idx", qa.get("question_id"))
    if explicit is not None:
        return sample_id, f"id:{explicit}"
    question = str(qa.get("question") or "").strip()
    if not question:
        raise ValueError(f"sample {sample_id!r} QA {qa_index} has no question or QA id")
    return sample_id, f"question:{question}"


def _sample_identifier(sample: Mapping[str, Any], sample_index: int, *, label: str) -> str:
    if "sample_id" not in sample or sample.get("sample_id") is None:
        raise ValueError(f"{label} sample {sample_index} has no sample_id")
    sample_id = str(sample["sample_id"]).strip()
    if not sample_id:
        raise ValueError(f"{label} sample {sample_index} has no sample_id")
    return sample_id


def _reference_identities(
    reference: Sequence[Mapping[str, Any]],
    categories: set[Any],
    known_categories: set[Any],
) -> set[tuple[str, str]]:
    identities = set()
    sample_ids = set()
    for sample_index, sample in enumerate(reference):
        if not isinstance(sample, Mapping):
            raise ValueError(f"reference sample {sample_index} is not an object")
        sample_id = _sample_identifier(sample, sample_index, label="reference")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate reference sample_id: {sample_id}")
        sample_ids.add(sample_id)
        qa_items = sample.get("qa")
        if not isinstance(qa_items, list) or not qa_items:
            raise ValueError(f"reference sample {sample_id!r} has no non-empty QA list")
        for qa_index, qa in enumerate(qa_items):
            if not isinstance(qa, Mapping):
                raise ValueError(f"reference sample {sample_id!r} QA {qa_index} is not an object")
            category = qa.get("category")
            if category not in known_categories:
                raise ValueError(
                    f"reference sample {sample_id!r} QA {qa_index} has unsupported "
                    f"category {category!r}"
                )
            if category in categories:
                identity = _qa_identity(sample_id, qa, qa_index)
                if identity in identities:
                    raise ValueError(f"duplicate reference QA identity: {identity}")
                identities.add(identity)
    return identities


def validate_evaluation_payload(
    responses: Any,
    *,
    categories: Iterable[Any],
    excluded_categories: Iterable[Any] = (),
    expected_samples: int | None = None,
    expected_qa: int | None = None,
    reference: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate result coverage before any metric silently drops rows."""

    if not isinstance(responses, list) or not responses:
        raise ValueError("evaluation input must be a non-empty JSON list")

    valid_categories = set(categories)
    excluded = set(excluded_categories)
    known_categories = valid_categories | excluded
    sample_ids = set()
    qa_identities = set()
    category_counts: Counter[Any] = Counter()
    excluded_counts: Counter[Any] = Counter()

    for sample_index, sample in enumerate(responses):
        if not isinstance(sample, Mapping):
            raise ValueError(f"sample {sample_index} is not an object")
        sample_id = _sample_identifier(sample, sample_index, label="result")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)

        qa_items = sample.get("qa")
        if not isinstance(qa_items, list) or not qa_items:
            raise ValueError(f"sample {sample_id!r} must contain a non-empty QA list")
        for qa_index, qa in enumerate(qa_items):
            if not isinstance(qa, Mapping):
                raise ValueError(f"sample {sample_id!r} QA {qa_index} is not an object")
            category = qa.get("category")
            if category not in known_categories:
                raise ValueError(
                    f"sample {sample_id!r} QA {qa_index} has unsupported category {category!r}"
                )
            identity = _qa_identity(sample_id, qa, qa_index)
            if identity in qa_identities:
                raise ValueError(f"duplicate QA identity: {identity}")
            qa_identities.add(identity)
            if "answer" not in qa:
                raise ValueError(f"sample {sample_id!r} QA {qa_index} has no reference answer")
            if "response" not in qa:
                raise ValueError(f"sample {sample_id!r} QA {qa_index} has no response field")
            if category in valid_categories:
                category_counts[category] += 1
            else:
                excluded_counts[category] += 1

    missing_categories = valid_categories - set(category_counts)
    if missing_categories:
        raise ValueError(f"evaluation input has no rows for categories: {sorted(missing_categories, key=str)}")
    if expected_samples is not None and len(sample_ids) != expected_samples:
        raise ValueError(f"expected {expected_samples} samples, found {len(sample_ids)}")
    evaluated_qa = sum(category_counts.values())
    if expected_qa is not None and evaluated_qa != expected_qa:
        raise ValueError(f"expected {expected_qa} evaluated QA rows, found {evaluated_qa}")

    if reference is not None:
        expected_identities = _reference_identities(
            reference,
            valid_categories,
            known_categories,
        )
        actual_identities = set()
        for sample_index, sample in enumerate(responses):
            sample_id = _sample_identifier(sample, sample_index, label="result")
            for qa_index, qa in enumerate(sample["qa"]):
                if qa.get("category") in valid_categories:
                    actual_identities.add(_qa_identity(sample_id, qa, qa_index))
        missing = expected_identities - actual_identities
        unexpected = actual_identities - expected_identities
        if missing or unexpected:
            raise ValueError(
                "result/reference QA coverage mismatch: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

    return {
        "sample_count": len(sample_ids),
        "evaluated_qa_count": evaluated_qa,
        "category_counts": dict(category_counts),
        "excluded_category_counts": dict(excluded_counts),
        "duplicate_samples": 0,
        "duplicate_qa": 0,
    }
