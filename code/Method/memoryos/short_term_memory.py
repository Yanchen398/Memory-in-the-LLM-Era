import json
from collections import deque
from .utils import get_timestamp

class ShortTermMemory:
    def __init__(self, max_capacity=10, file_path="short_term.json"):
        self.max_capacity = max_capacity
        self.file_path = file_path
        self.memory = deque(maxlen=max_capacity)
        self.load()

    def add_qa_pair(self, qa_pair):
        qa_pair["timestamp"] = qa_pair.get("timestamp", get_timestamp())
        self.memory.append(qa_pair)
        print(f"Short-term memory: added QA pair for user input: {qa_pair.get('user_input','')[:30]}...")
        self.save()

    def get_all(self):
        return list(self.memory)

    def is_full(self):
        return len(self.memory) == self.max_capacity

    def pop_oldest(self):
        if self.memory:
            msg = self.memory.popleft()
            print("Short-term memory: evicted the oldest QA pair.")
            self.save()
            return msg
        return None

    def save(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(list(self.memory), f, ensure_ascii=False, indent=2)

    def load(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.memory = deque(data, maxlen=self.max_capacity)
            print("Short-term memory: loaded successfully.")
        except Exception:
            self.memory = deque(maxlen=self.max_capacity)
            print("Short-term memory: no historical data found.")
