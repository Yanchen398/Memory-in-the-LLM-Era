"""Segment-granularity short-term to mid-term conversion for MemoryOS.

Only the conversion unit is changed: consecutive MemoryOS pages are first
grouped by semantic similarity, then each group is passed through MemoryOS's
existing multi-topic summarizer and mid-term insertion logic.
"""

import numpy as np

from Method.memoryos import utils as memoryos_utils
from Method.memoryos.dynamic_update import DynamicUpdate
from Method.memoryos.utils import generate_id, gpt_generate_multi_summary


class SegmentDynamicUpdate(DynamicUpdate):
    """MemoryOS updater that indexes semantic segments instead of one message."""

    def __init__(self, *args, segment_threshold=0.5, segment_max_messages=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.segment_threshold = float(segment_threshold)
        self.segment_max_messages = max(0, int(segment_max_messages or 0))
        if not -1.0 <= self.segment_threshold <= 1.0:
            raise ValueError("segment_threshold must be between -1 and 1")

    @staticmethod
    def _page_embedding_text(page):
        return (
            f"Conversation Timestamp: {page.get('timestamp', '')}\n"
            f"User: {page.get('user_input', '')}\n"
            f"Assistant: {page.get('agent_response', '')}\n"
        )

    @staticmethod
    def _cosine_similarity(left, right):
        left = np.asarray(left, dtype=np.float32).reshape(-1)
        right = np.asarray(right, dtype=np.float32).reshape(-1)
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator == 0.0:
            return 0.0
        return float(np.dot(left, right) / denominator)

    def _split_semantic_segments(self, pages):
        if not pages:
            return []
        embeddings = [
            memoryos_utils.get_embedding(self._page_embedding_text(page))
            for page in pages
        ]
        segments = []
        start = 0
        for index in range(1, len(pages)):
            similarity = self._cosine_similarity(
                embeddings[index - 1], embeddings[index]
            )
            reached_size_limit = (
                self.segment_max_messages > 0
                and index - start >= self.segment_max_messages
            )
            if similarity < self.segment_threshold or reached_size_limit:
                segments.append(pages[start:index])
                start = index
        segments.append(pages[start:])
        return segments

    def bulk_evict_and_update_mid_term(self, force=False):
        evicted = []
        if not force and not self.short_term_memory.is_full():
            return
        while self.short_term_memory.memory:
            message = self.short_term_memory.pop_oldest()
            if message and message.get("user_input") and message.get("agent_response"):
                evicted.append(message)

        if not evicted:
            return

        # Preserve MemoryOS page creation, continuity links, and rolling context.
        pages = []
        for qa in evicted:
            page = {
                "page_id": generate_id("page"),
                "user_input": qa.get("user_input", ""),
                "agent_response": qa.get("agent_response", ""),
                "timestamp": qa.get("timestamp"),
                "preloaded": False,
                "analyzed": False,
                "pre_page": None,
                "next_page": None,
                "meta_info": None,
            }

            if self.fast_index:
                pages.append(page)
                self.last_evicted_page = page
                continue

            is_continuous = self._is_conversation_continuing(
                self.last_evicted_page, page
            )
            if is_continuous and self.last_evicted_page:
                page["pre_page"] = self.last_evicted_page["page_id"]
                self.last_evicted_page["next_page"] = page["page_id"]
                last_meta = self.last_evicted_page.get("meta_info")
                new_meta_info = self._generate_meta_info(last_meta, page)
                page["meta_info"] = new_meta_info
                self._update_connected_pages(page["pre_page"], new_meta_info)
            else:
                page["meta_info"] = self._generate_meta_info(None, page)

            pages.append(page)
            self.last_evicted_page = page

        # This is the sole method-level ablation: segment before the original
        # MemoryOS topic extraction and storage path.
        segments = self._split_semantic_segments(pages)
        print(
            "MemoryOS segment conversion: "
            f"{len(pages)} pages -> {len(segments)} semantic segments "
            f"(threshold={self.segment_threshold})."
        )
        for segment_index, segment_pages in enumerate(segments, start=1):
            input_text = "\n".join(
                f"User: {page.get('user_input', '')}\n"
                for page in segment_pages
            )
            print(
                "Dynamic update: calling GPT to generate multi-topic summaries "
                f"for semantic segment {segment_index}/{len(segments)}..."
            )
            multi_summary = gpt_generate_multi_summary(input_text, self.client)
            for summary_dict in multi_summary.get("summaries", []):
                if not isinstance(summary_dict, dict):
                    print(
                        "Dynamic update: skipping malformed summary item: "
                        f"{summary_dict!r}"
                    )
                    continue
                sub_summary = summary_dict.get("content", "")
                sub_keywords = summary_dict.get("keywords", [])
                if self.fast_index:
                    for page in segment_pages:
                        page["meta_info"] = sub_summary
                        page["page_keywords"] = sub_keywords
                print(
                    "Dynamic update: processing subtopic "
                    f"[{summary_dict.get('theme', '')}] and inserting its "
                    "semantic segment into mid-term memory..."
                )
                self.mid_term_memory.insert_pages_into_session(
                    sub_summary,
                    sub_keywords,
                    segment_pages,
                    self.topic_similarity_threshold,
                )
