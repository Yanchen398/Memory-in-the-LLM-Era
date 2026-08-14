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

    def retrieve(self, user_query, segment_threshold=0.7, page_threshold=0.7, knowledge_threshold=0.7, client=None, top_k=None, update_stats=True, use_llm_keywords=True):
            print("Retrieval: searching mid-term memory...")
            effective_top_k = int(top_k or self.queue_capacity)
            matched = self.mid_term_memory.search_sessions_by_summary(
                user_query,
                client,
                segment_threshold,
                page_threshold,
                top_k=effective_top_k,
                update_stats=update_stats,
                use_llm_keywords=use_llm_keywords,
            )
            
            # Use a heap to keep the highest-scoring pages.
            top_pages_heap = []
            
            heap_counter = 0
            for item in matched:
                for page_info in item["matched_pages"]:  # Each page_info is [page, overall_score].
                    page, overall_score = page_info
                    heap_counter += 1
                    heap_item = (overall_score, heap_counter, page)
                    # Use a min-heap to keep only the requested top-k items.
                    if len(top_pages_heap) < effective_top_k:
                        heapq.heappush(top_pages_heap, heap_item)
                    else:
                        # Replace the smallest item if the current score is higher.
                        if overall_score > top_pages_heap[0][0]:
                            heapq.heappop(top_pages_heap)
                            heapq.heappush(top_pages_heap, heap_item)
            
            # Rebuild the retrieval queue from high score to low score.
            self.retrieval_queue.clear()
            for score, _, page in sorted(top_pages_heap, key=lambda x: (x[0], x[1]), reverse=True):
                self.retrieval_queue.append(page)
            
            print(f"Retrieval: recalled {len(self.retrieval_queue)} QA pairs from mid-term memory into the queue.")
            long_term_info = self.long_term_memory.search_knowledge(user_query, threshold=knowledge_threshold, top_k=effective_top_k)
            # print(long_term_info[0].keys())
            print(f"Retrieval: recalled {len(long_term_info)} knowledge items from long-term memory.")
            
            return {
                "retrieval_queue": list(self.retrieval_queue),
                "long_term_knowledge": long_term_info,
                "retrieved_at": get_timestamp(),
                "top_k": effective_top_k
            }
