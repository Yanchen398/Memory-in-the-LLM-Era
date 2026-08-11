import contextlib
import importlib
import json
import os
import time
from typing import Dict


class TokenTracker:
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.root = self._create_stage_node("root")
        self.stack = [self.root]
        self._patched_target = None
        self._patched_name = None
        self._original_call = None

    @staticmethod
    def _create_stage_node(name: str) -> Dict:
        return {
            "name": name,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0.0,
            "sub_stages": {},
        }

    @contextlib.contextmanager
    def stage(self, name: str):
        parent = self.stack[-1]
        if name not in parent["sub_stages"]:
            parent["sub_stages"][name] = self._create_stage_node(name)
        node = parent["sub_stages"][name]
        start_time = time.time()
        node["start_time"] = start_time
        self.stack.append(node)
        try:
            yield
        finally:
            end_time = time.time()
            node["end_time"] = end_time
            node["duration_seconds"] += end_time - start_time
            self.stack.pop()

    def _extract_usage(self, response) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is not None:
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }
        if isinstance(response, dict):
            usage = response.get("usage", {}) or {}
            return {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def patch_openai(self):
        resource_module = importlib.import_module("openai.resources.chat.completions")
        target = resource_module.Completions
        function_name = "create"
        original = getattr(target, function_name)
        tracker = self

        def wrapped(*args, **kwargs):
            response = original(*args, **kwargs)
            usage = tracker._extract_usage(response)
            current = tracker.stack[-1]
            current["prompt_tokens"] += usage["prompt_tokens"]
            current["completion_tokens"] += usage["completion_tokens"]
            current["total_tokens"] += usage["total_tokens"]
            return response

        self._patched_target = target
        self._patched_name = function_name
        self._original_call = original
        setattr(target, function_name, wrapped)

    def restore(self):
        if self._patched_target is not None and self._patched_name and self._original_call is not None:
            setattr(self._patched_target, self._patched_name, self._original_call)
        self._patched_target = None
        self._patched_name = None
        self._original_call = None

    def save_to_json(self):
        def aggregate(node: Dict):
            for child in node["sub_stages"].values():
                aggregate(child)
            node["prompt_tokens"] += sum(child["prompt_tokens"] for child in node["sub_stages"].values())
            node["completion_tokens"] += sum(child["completion_tokens"] for child in node["sub_stages"].values())
            node["total_tokens"] += sum(child["total_tokens"] for child in node["sub_stages"].values())

        aggregate(self.root)
        parent = os.path.dirname(self.output_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as file_obj:
            json.dump(self.root, file_obj, ensure_ascii=False, indent=2)
