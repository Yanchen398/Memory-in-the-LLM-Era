from ray import get
from .short_term import ShortTermMemory
from .mid_term import MidTermMemory
from .utils import OpenAIClient, generate_id, get_embedding, get_timestamp, insert
from .structure import MemTree
from scipy.spatial.distance import cosine
from .prompt import SEGMENT_SUMMARY_PROMPT

class Updater:
    def __init__(self, 
                 short_term_memory: ShortTermMemory, 
                 mid_term_memory: MidTermMemory,
                 client: OpenAIClient,
                 llm_model: str,
                 tree: MemTree = MemTree(""),
                 segment_threshold=0.5):
        self.short_term_memory = short_term_memory
        self.mid_term_memory = mid_term_memory
        self.client = client
        self.llm_model = llm_model
        self.tree = tree
        self.segment_threshold = segment_threshold
    
    def process_short_term_to_mid_term(self, mode='half'):
        evicted_qas = []
        if mode == 'half':
            while self.short_term_memory.is_half():
                qa = self.short_term_memory.pop_oldest()
                if qa and qa.get("speaker_a_input") and qa.get("speaker_b_input"):
                    evicted_qas.append(qa)
        if mode == 'all':
            while not self.short_term_memory.is_empty():
                qa = self.short_term_memory.pop_oldest()
                if qa and qa.get("speaker_a_input") and qa.get("speaker_b_input"):
                    evicted_qas.append(qa)
        
        if not evicted_qas:
            print("Updater: No QAs evicted from short-term memory.")
            return
        
        # print(f"Updater: Processing {len(evicted_qas)} QAs from short-term to mid-term tree.")

        all_evicted_pages = []
        all_evicted_pages_id = []
        all_page_thresholds = []

        for qa_pair in evicted_qas:
            current_page_obj = {
                "page_id": generate_id("page"),
                "speaker_a": qa_pair.get("speaker_a", ""),
                "speaker_b": qa_pair.get("speaker_b", ""),
                "speaker_a_input": qa_pair.get("speaker_a_input", ""),
                "speaker_b_input": qa_pair.get("speaker_b_input", ""),
                "timestamp": qa_pair.get("timestamp", get_timestamp()),
                "content": None,
                "embedding": None
            }
            current_page_obj["content"] = f"Conversation Timestamp: {current_page_obj.get('timestamp','')}\n{current_page_obj.get('speaker_a','')}: {current_page_obj.get('speaker_a_input','')}\n{current_page_obj.get('speaker_b','')}: {current_page_obj.get('speaker_b_input','')}\n"
            current_page_obj_id = id(current_page_obj)
            current_page_obj["embedding"] = get_embedding(current_page_obj["content"]).flatten().tolist()
            
            # Insert the raw dialogue into Milvus.
            insert([{"id": current_page_obj_id, "vector": current_page_obj["embedding"], "text": current_page_obj["content"], "type": "dialogue"}])

            # Compute the similarity to the previous page.
            if all_evicted_pages:
                cos_similarity = 1 - cosine(current_page_obj["embedding"], all_evicted_pages[-1]["embedding"])
                all_page_thresholds.append(cos_similarity)

            all_evicted_pages.append(current_page_obj)
            all_evicted_pages_id.append(current_page_obj_id)

        seg_id = [i+1 for i in range(len(all_page_thresholds)) if all_page_thresholds[i] < self.segment_threshold]
        divided_segments = [all_evicted_pages[i:j] for i, j in zip([0]+seg_id, seg_id+[len(all_evicted_pages)])]
        devided_pages_id = [all_evicted_pages_id[i:j] for i, j in zip([0]+seg_id, seg_id+[len(all_evicted_pages_id)])]

        # Insert segments into mid-term memory
        # print(f"Updater: Inserting {len(divided_segments)} segments into mid-term.")

        # Generate a summary for each segment and add it to the tree.
        for segment, pages_id in zip(divided_segments, devided_pages_id):
            segment_text = ""
            for page in segment:
                segment_text += page["content"] + "\n"
            speaker_a = segment[0]["speaker_a"]
            speaker_b = segment[0]["speaker_b"]
            prompt = SEGMENT_SUMMARY_PROMPT.format(speaker_a=speaker_a, speaker_b=speaker_b, segment_text=segment_text)
            segment_summary = self.client.chat_completion(
                model=self.llm_model, 
                messages=[{"role": "user", "content": prompt}]
            )
            seg_node_id = self.tree.add_node(segment_summary, id(self.tree.root))
            # print(f"Updater: Added segment node to mid-term tree with ID {seg_node_id}. Summary: {segment_summary[:30]}...")

            # Store the segment in mid-term memory.
            segment_obj = {
                "segment_id": generate_id("segment"),
                "summary": segment_summary,
                "pages": {}
            }
            for i in range(len(segment)):
                segment_obj["pages"][pages_id[i]] = segment[i]

            self.mid_term_memory.add_segment(seg_node_id, segment_obj)
            
