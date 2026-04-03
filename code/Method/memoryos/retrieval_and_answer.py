from collections import deque
from .utils import get_timestamp
import heapq
class RetrievalAndAnswer:
    def __init__(self, short_term_memory, mid_term_memory, long_term_memory, dynamic_updater, queue_capacity=25):
        self.short_term_memory = short_term_memory
        self.mid_term_memory = mid_term_memory
        self.long_term_memory = long_term_memory
        self.dynamic_updater = dynamic_updater
        self.queue_capacity = queue_capacity
        self.retrieval_queue = deque(maxlen=queue_capacity)

    def retrieve(self, user_query, segment_threshold=0.7, page_threshold=0.7, knowledge_threshold=0.7, client=None):
            print("Retrieval: searching mid-term memory...")
            matched = self.mid_term_memory.search_sessions_by_summary(user_query, client, segment_threshold, page_threshold)
            
            # Use a heap to keep the highest-scoring pages.
            top_pages_heap = []
            
            for item in matched:
                for page_info in item["matched_pages"]:  # Each page_info is [page, overall_score].
                    page, overall_score = page_info
                    # Use a min-heap to keep only the top queue_capacity items.
                    if len(top_pages_heap) < self.queue_capacity:
                        heapq.heappush(top_pages_heap, (overall_score, page))
                    else:
                        # Replace the smallest item if the current score is higher.
                        if overall_score > top_pages_heap[0][0]:
                            heapq.heappop(top_pages_heap)
                            heapq.heappush(top_pages_heap, (overall_score, page))
            
            # Rebuild the retrieval queue from high score to low score.
            self.retrieval_queue.clear()
            for score, page in sorted(top_pages_heap, key=lambda x: x[0], reverse=True):
                self.retrieval_queue.append(page)
            
            print(f"Retrieval: recalled {len(self.retrieval_queue)} QA pairs from mid-term memory into the queue.")
            long_term_info = self.long_term_memory.search_knowledge(user_query, threshold=knowledge_threshold)
            # print(long_term_info[0].keys())
            print(f"Retrieval: recalled {len(long_term_info)} knowledge items from long-term memory.")
            
            return {
                "retrieval_queue": list(self.retrieval_queue),
                "long_term_knowledge": long_term_info,
                "retrieved_at": get_timestamp()
            }
