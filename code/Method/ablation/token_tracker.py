"""Thread-safe process-local token accounting for SOTA ablations."""

import json
import os
import threading
from copy import deepcopy


_LOCK = threading.Lock()
_USAGE = {}


def _empty_usage():
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "requests": 0,
        "failed_requests": 0,
        "stages": {},
        "endpoints": {},
    }


def reset_token_usage():
    global _USAGE
    with _LOCK:
        _USAGE = _empty_usage()


def _usage_values(usage):
    if usage is None:
        return 0, 0, 0
    if isinstance(usage, dict):
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", prompt + completion) or 0)
    else:
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", prompt + completion) or 0)
    return prompt, completion, total or prompt + completion


def _add_counts(node, prompt, completion, total, failed=False):
    node["prompt_tokens"] = int(node.get("prompt_tokens", 0)) + prompt
    node["completion_tokens"] = int(node.get("completion_tokens", 0)) + completion
    node["total_tokens"] = int(node.get("total_tokens", 0)) + total
    node["requests"] = int(node.get("requests", 0)) + (0 if failed else 1)
    node["failed_requests"] = int(node.get("failed_requests", 0)) + int(failed)


def record_usage(usage, stage="llm", endpoint="unknown", model=None):
    prompt, completion, total = _usage_values(usage)
    with _LOCK:
        if not _USAGE:
            reset_required = True
        else:
            reset_required = False
    if reset_required:
        reset_token_usage()
    with _LOCK:
        _add_counts(_USAGE, prompt, completion, total)
        stage_node = _USAGE["stages"].setdefault(stage, _empty_usage())
        endpoint_node = _USAGE["endpoints"].setdefault(endpoint, _empty_usage())
        _add_counts(stage_node, prompt, completion, total)
        _add_counts(endpoint_node, prompt, completion, total)
        if model:
            stage_node["model"] = model


def record_failure(stage="llm", endpoint="unknown"):
    with _LOCK:
        if not _USAGE:
            _USAGE.update(_empty_usage())
        _add_counts(_USAGE, 0, 0, 0, failed=True)
        stage_node = _USAGE["stages"].setdefault(stage, _empty_usage())
        endpoint_node = _USAGE["endpoints"].setdefault(endpoint, _empty_usage())
        _add_counts(stage_node, 0, 0, 0, failed=True)
        _add_counts(endpoint_node, 0, 0, 0, failed=True)


def get_token_usage():
    with _LOCK:
        if not _USAGE:
            return _empty_usage()
        return deepcopy(_USAGE)


def merge_token_usages(usages):
    merged = _empty_usage()
    for usage in usages:
        if not usage:
            continue
        merged["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        merged["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        merged["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        merged["requests"] += int(usage.get("requests", 0) or 0)
        merged["failed_requests"] += int(usage.get("failed_requests", 0) or 0)
        for group_name in ("stages", "endpoints"):
            for name, node in usage.get(group_name, {}).items():
                target = merged[group_name].setdefault(name, _empty_usage())
                target["prompt_tokens"] += int(node.get("prompt_tokens", 0) or 0)
                target["completion_tokens"] += int(node.get("completion_tokens", 0) or 0)
                target["total_tokens"] += int(node.get("total_tokens", 0) or 0)
                target["requests"] += int(node.get("requests", 0) or 0)
                target["failed_requests"] += int(node.get("failed_requests", 0) or 0)
                if node.get("model"):
                    target["model"] = node["model"]
    return merged


def save_token_usage(path, usage):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(usage, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


reset_token_usage()
