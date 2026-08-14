from .utils import ensure_directory_exists, OpenAIClient
import json
from collections import defaultdict

class MidTermMemory:
    def __init__(self, file_path: str, client: OpenAIClient):
        self.file_path = file_path
        ensure_directory_exists(self.file_path)
        self.client = client
        self.segments = {} # {segment_id: segment_object}
        self.load()
    
    def load(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.segments = data
            print(f"MidTermMemory: Loaded from {self.file_path}. Segments: {len(self.segments)}.")
        except FileNotFoundError:
            print(f"MidTermMemory: No history file found at {self.file_path}. Initializing new memory.")
        except json.JSONDecodeError:
            print(f"MidTermMemory: Error decoding JSON from {self.file_path}. Initializing new memory.")
        except Exception as e:
            print(f"MidTermMemory: An unexpected error occurred during load from {self.file_path}: {e}. Initializing new memory.")     
        
    def save(self):
        data_to_save = self.segments
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Error saving MidTermMemory to {self.file_path}: {e}")
    
    def add_segment(self, segment_id: str, segment_object: dict):
        self.segments[segment_id] = segment_object
        # print(f"MidTermMemory: Added/Updated segment {segment_id}. Total segments: {len(self.segments)}.")
        self.save()