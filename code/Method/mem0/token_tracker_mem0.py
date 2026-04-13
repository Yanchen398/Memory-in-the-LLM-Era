import json
import time


class TokenTracker:
    def __init__(self, output_file=None):
        self.output_file = output_file
        self.reset()
        self._stage_stack = ["root"]

    def reset(self):
        self.stats = {
            "name": "root",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0,
            "sub_stages": {}
        }
        self._current_node = self.stats

    def add_usage(self, prompt_tokens, completion_tokens, total_tokens, extra=None):
        node = self._get_current_node()
        node["prompt_tokens"] += prompt_tokens
        node["completion_tokens"] += completion_tokens
        node["total_tokens"] += total_tokens
        if extra:
            node.setdefault("records", []).append({"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens, "extra": extra})

        if self.output_file:
            self.save()

    def stage(self, name):
        return _StageContext(self, name)

    def _get_current_node(self):
        node = self.stats
        for stage in self._stage_stack[1:]:
            node = node["sub_stages"].setdefault(stage, {
                "name": stage,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "start_time": None,
                "end_time": None,
                "duration_seconds": 0,
                "sub_stages": {}
            })
        return node

    def save(self):
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

    def summary(self):
        return self.stats

class _StageContext:
    def __init__(self, tracker, name):
        self.tracker = tracker
        self.name = name

    def __enter__(self):
        self.tracker._stage_stack.append(self.name)
        node = self.tracker._get_current_node()
        node["start_time"] = time.time()
        return node

    def __exit__(self, exc_type, exc_val, exc_tb):
        node = self.tracker._get_current_node()
        node["end_time"] = time.time()
        if node["start_time"]:
            node["duration_seconds"] = node["end_time"] - node["start_time"]
        self.tracker._stage_stack.pop()
        if self.tracker.output_file:
            self.tracker.save()
