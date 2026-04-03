import json
import numpy as np
from .utils import get_timestamp, get_embedding, normalize_vector

class LongTermMemory:
    def __init__(self, file_path="long_term.json"):
        self.file_path = file_path
        self.user_profiles = {}
        self.knowledge_base = []
        self.assistant_knowledge = []
        self.load()

    def update_user_profile(self, user_id, new_data, merge=False):
        """
        Update the user profile.
        :param user_id: User ID.
        :param new_data: New profile data.
        :param merge: Whether to merge into existing data (True) or overwrite it (False).
        """
        if merge and user_id in self.user_profiles:
            current_data = self.user_profiles[user_id]["data"]
            if isinstance(current_data, str) and isinstance(new_data, str):
                # Preserve the original text profile and append the new content.
                updated_data = f"{current_data}\n\n--- Updated ---\n{new_data}"
            else:
                updated_data = new_data  # Overwrite non-string data directly.
        else:
            updated_data = new_data
        
        self.user_profiles[user_id] = {
            "data": updated_data,
            "last_updated": get_timestamp()
        }
        print("Long-term memory: updated the user profile.")
        self.save()
    def add_assistant_knowledge(self, knowledge_text):
        """
        Add assistant-related knowledge or traits.
        """
        if knowledge_text.strip() == "" or knowledge_text.strip() == "- None" or knowledge_text.strip() == "- None.":
            print("Long-term memory: assistant knowledge is empty and will not be saved.")
            return
        vec = get_embedding(knowledge_text)
        vec = normalize_vector(vec).tolist()
        entry = {
            "knowledge": knowledge_text,
            "timestamp": get_timestamp(),
            "knowledge_embedding": vec
        }
        self.assistant_knowledge.append(entry)
        print("Long-term memory: added assistant knowledge.")
        self.save()

    def get_assistant_knowledge(self):
        """
        Return all assistant knowledge.
        """
        return self.assistant_knowledge


    def get_raw_user_profile(self, user_id):
        """Return the raw user profile data."""
        return self.user_profiles.get(user_id, {}).get("data", "")
    
    def get_user_profile(self, user_id):
        return self.user_profiles.get(user_id, {})

    def add_knowledge(self, knowledge_text):
        if knowledge_text.strip() == "" or knowledge_text.strip() == "- None"or knowledge_text.strip() == "- None.":
            print("Long-term memory: private knowledge is empty and will not be saved.")
            return
        vec = get_embedding(knowledge_text)
        vec = normalize_vector(vec).tolist()
        entry = {
            "knowledge": knowledge_text,
            "timestamp": get_timestamp(),
            "knowledge_embedding": vec
        }
        self.knowledge_base.append(entry)
        print("Long-term memory: added private knowledge.")
        self.save()

    def get_knowledge(self):
        return self.knowledge_base

    def search_knowledge(self, query, threshold=0.1, top_k=10):
        if not self.knowledge_base:
            return []
        query_vec = get_embedding(query)
        query_vec = normalize_vector(query_vec)
        embeddings = []
        for entry in self.knowledge_base:
            embeddings.append(np.array(entry["knowledge_embedding"], dtype=np.float32))
        embeddings = np.array(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        from faiss import IndexFlatIP
        dim = embeddings.shape[1]
        index = IndexFlatIP(dim)
        index.add(embeddings)
        query_arr = np.array([query_vec], dtype=np.float32)
        distances, indices = index.search(query_arr, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if dist >= threshold:
                results.append(self.knowledge_base[idx])
        print(f"Long-term memory: retrieved {len(results)} matching knowledge items.")
        return results

    def save(self):
        data = {
            "user_profiles": self.user_profiles,
            "knowledge_base": self.knowledge_base,
            "assistant_knowledge": self.assistant_knowledge
        }
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("Long-term memory: saved successfully.")

    def load(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.user_profiles = data.get("user_profiles", {})
                self.knowledge_base = data.get("knowledge_base", [])
                self.assistant_knowledge = data.get("assistant_knowledge", [])  # Load assistant knowledge.
            print("Long-term memory: loaded successfully.")
        except Exception:
            self.user_profiles = {}
            self.knowledge_base = []
            print("Long-term memory: no historical data found.")
