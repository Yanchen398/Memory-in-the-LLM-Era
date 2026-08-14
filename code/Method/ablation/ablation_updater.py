"""Configurable short-term to mid-term conversion used by SOTA ablations."""

from scipy.spatial.distance import cosine

from .prompt import SEGMENT_SUMMARY_PROMPT
from .utils import generate_id, get_embedding, get_timestamp, insert


class AblationUpdater:
    def __init__(
        self,
        short_term_memory,
        mid_term_memory,
        client,
        llm_model,
        memory_structure,
        memory_granularity="segment",
        mid_term_structure="tree",
        segment_threshold=0.5,
    ):
        if memory_granularity not in {"segment", "message"}:
            raise ValueError(
                "memory_granularity must be either 'segment' or 'message'"
            )
        if mid_term_structure not in {"tree", "graph"}:
            raise ValueError("mid_term_structure must be either 'tree' or 'graph'")
        self.short_term_memory = short_term_memory
        self.mid_term_memory = mid_term_memory
        self.client = client
        self.llm_model = llm_model
        self.memory_structure = memory_structure
        self.memory_granularity = memory_granularity
        self.mid_term_structure = mid_term_structure
        self.segment_threshold = float(segment_threshold)
        self.last_evicted_page = self._load_last_message_page()

    def _load_last_message_page(self):
        for memory in reversed(list(self.mid_term_memory.segments.values())):
            if memory.get("unit_type") != "message":
                continue
            pages = list(memory.get("pages", {}).values())
            if pages:
                return pages[-1]
        return None

    def _evict(self, mode):
        evicted = []
        if mode == "half":
            while self.short_term_memory.is_half():
                qa = self.short_term_memory.pop_oldest()
                if qa and qa.get("speaker_a_input") and qa.get("speaker_b_input"):
                    evicted.append(qa)
        elif mode == "all":
            while not self.short_term_memory.is_empty():
                qa = self.short_term_memory.pop_oldest()
                if qa and qa.get("speaker_a_input") and qa.get("speaker_b_input"):
                    evicted.append(qa)
        else:
            raise ValueError("mode must be 'half' or 'all'")
        return evicted

    @staticmethod
    def _page_content(page):
        return (
            f"Conversation Timestamp: {page.get('timestamp', '')}\n"
            f"{page.get('speaker_a', '')}: {page.get('speaker_a_input', '')}\n"
            f"{page.get('speaker_b', '')}: {page.get('speaker_b_input', '')}\n"
        )

    def _create_pages(self, evicted_qas):
        pages = []
        vector_ids = []
        for qa_pair in evicted_qas:
            page = {
                "page_id": generate_id("page"),
                "speaker_a": qa_pair.get("speaker_a", ""),
                "speaker_b": qa_pair.get("speaker_b", ""),
                "speaker_a_input": qa_pair.get("speaker_a_input", ""),
                "speaker_b_input": qa_pair.get("speaker_b_input", ""),
                "timestamp": qa_pair.get("timestamp", get_timestamp()),
                "pre_page": None,
                "next_page": None,
                "meta_info": None,
                "content": None,
                "embedding": None,
            }
            page["content"] = self._page_content(page)
            page["embedding"] = get_embedding(page["content"]).flatten().tolist()
            vector_id = id(page)
            insert(
                [
                    {
                        "id": vector_id,
                        "vector": page["embedding"],
                        "text": page["content"],
                        "type": "dialogue",
                    }
                ]
            )
            pages.append(page)
            vector_ids.append(vector_id)
        return pages, vector_ids

    def _segment_units(self, pages, vector_ids):
        thresholds = [
            1 - cosine(pages[index]["embedding"], pages[index - 1]["embedding"])
            for index in range(1, len(pages))
        ]
        cuts = [
            index + 1
            for index, similarity in enumerate(thresholds)
            if similarity < self.segment_threshold
        ]
        ranges = list(zip([0] + cuts, cuts + [len(pages)]))
        units = []
        for start, end in ranges:
            segment = pages[start:end]
            segment_text = "\n".join(page["content"] for page in segment)
            prompt = SEGMENT_SUMMARY_PROMPT.format(
                speaker_a=segment[0]["speaker_a"],
                speaker_b=segment[0]["speaker_b"],
                segment_text=segment_text,
            )
            summary = self.client.chat_completion(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                stage="segment_summary",
            )
            units.append(
                {
                    "unit_type": "segment",
                    "content": summary,
                    "timestamp": segment[-1]["timestamp"],
                    "pages": segment,
                    "vector_ids": vector_ids[start:end],
                    "meta_info": None,
                }
            )
        return units

    def _is_conversation_continuing(self, previous_page, current_page):
        if not previous_page:
            return False
        prompt = """Determine if these two conversation pages are continuous (true continuation without topic shift).
Return ONLY "true" or "false".

Previous Page:
User: {prev_user}
Assistant: {prev_agent}

Current Page:
User: {curr_user}
Assistant: {curr_agent}

Continuous?""".format(
            prev_user=previous_page.get("speaker_a_input", ""),
            prev_agent=previous_page.get("speaker_b_input", ""),
            curr_user=current_page.get("speaker_a_input", ""),
            curr_agent=current_page.get("speaker_b_input", ""),
        )
        response = self.client.chat_completion(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a conversation continuity detector. "
                        "Return ONLY 'true' or 'false'."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
            stage="message_continuity",
        )
        return response.strip().lower() == "true"

    def _generate_meta_info(self, last_page_meta, current_page):
        current_conversation = (
            f"{current_page.get('speaker_a', 'User')}: "
            f"{current_page.get('speaker_a_input', '')}\n"
            f"{current_page.get('speaker_b', 'Assistant')}: "
            f"{current_page.get('speaker_b_input', '')}"
        )
        prompt = """Update the conversation meta-summary by incorporating the new dialogue while maintaining continuity.

Guidelines:
1. Start from the previous meta-summary (if exists)
2. Add/update information based on the new dialogue
3. Keep it concise (1-2 sentences max)
4. Maintain context coherence

Previous Meta-summary: {last_meta}
New Dialogue:
{new_dialogue}

Updated Meta-summary:""".format(
            last_meta=last_page_meta if last_page_meta else "None",
            new_dialogue=current_conversation,
        )
        return self.client.chat_completion(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a conversation meta-summary updater. Your task is to:\n"
                        "1. Preserve relevant context from previous meta-summary\n"
                        "2. Integrate new information from current dialogue\n"
                        "3. Output ONLY the updated summary (no explanations)"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=100,
            stage="message_meta_summary",
        ).strip()

    def _message_units(self, pages, vector_ids):
        units = []
        for page, vector_id in zip(pages, vector_ids):
            is_continuous = self._is_conversation_continuing(
                self.last_evicted_page, page
            )
            if is_continuous and self.last_evicted_page:
                page["pre_page"] = self.last_evicted_page["page_id"]
                self.last_evicted_page["next_page"] = page["page_id"]
                previous_meta = self.last_evicted_page.get("meta_info")
            else:
                previous_meta = None
            page["meta_info"] = self._generate_meta_info(previous_meta, page)
            contextual_content = (
                page["content"].rstrip()
                + "\nConversation chain overview: "
                + page["meta_info"]
            )
            units.append(
                {
                    "unit_type": "message",
                    "content": contextual_content,
                    "timestamp": page["timestamp"],
                    "pages": [page],
                    "vector_ids": [vector_id],
                    "meta_info": page["meta_info"],
                }
            )
            self.last_evicted_page = page
        return units

    def _store_unit(self, unit):
        if self.mid_term_structure == "tree":
            structure_id = self.memory_structure.add_node(
                unit["content"], id(self.memory_structure.root)
            )
        else:
            structure_id = self.memory_structure.add_memory(
                unit["content"],
                timestamp=unit["timestamp"],
                unit_type=unit["unit_type"],
            )

        memory_object = {
            "memory_id": generate_id(unit["unit_type"]),
            "unit_type": unit["unit_type"],
            "summary": unit["content"],
            "meta_info": unit["meta_info"],
            "pages": {
                str(vector_id): page
                for vector_id, page in zip(unit["vector_ids"], unit["pages"])
            },
        }
        self.mid_term_memory.add_segment(str(structure_id), memory_object)

    def process_short_term_to_mid_term(self, mode="half"):
        evicted_qas = self._evict(mode)
        if not evicted_qas:
            print("AblationUpdater: no QAs evicted from short-term memory.")
            return
        pages, vector_ids = self._create_pages(evicted_qas)
        units = (
            self._segment_units(pages, vector_ids)
            if self.memory_granularity == "segment"
            else self._message_units(pages, vector_ids)
        )
        for unit in units:
            self._store_unit(unit)
