from .structure import MemTree
import os
from .utils import OpenAIClient, ensure_directory_exists, get_timestamp, retrieve
from .short_term import ShortTermMemory
from .mid_term import MidTermMemory
from .ablation_updater import AblationUpdater
from .graph_memory import GraphMemory
from . import prompt

class Memoryos:
    def __init__(self, user_id: str, 
                 openai_api_key: str, 
                 data_storage_path: str,
                 openai_base_url: str, 
                 llm_model: str,
                 short_term_capacity=10,
                 tree: MemTree = None,
                 segment_threshold=0.5,
                 memory_granularity="segment",
                 mid_term_structure="tree",
                 graph_path=None,
                 graph_options=None,
                 top_k_retrieve=10,
                 graph_only_retrieval=False,
                 graph_include_original_text=False,
                 dialogue_top_k=None,
                 other_memory_top_k=None,
                 include_recent_dialogue_in_response=True,
                 response_prompt_variant="legacy",
                 ):
        self.user_id = user_id
        self.data_storage_path = os.path.abspath(data_storage_path)
        self.llm_model = llm_model
        self.segment_threshold = segment_threshold
        self.memory_granularity = memory_granularity
        self.mid_term_structure = mid_term_structure
        self.graph_options = graph_options or {}
        self.top_k_retrieve = int(top_k_retrieve)
        self.graph_only_retrieval = bool(graph_only_retrieval)
        self.graph_include_original_text = bool(graph_include_original_text)
        self.dialogue_top_k = int(dialogue_top_k or self.top_k_retrieve)
        self.other_memory_top_k = int(
            other_memory_top_k or self.top_k_retrieve
        )
        self.include_recent_dialogue_in_response = bool(
            include_recent_dialogue_in_response
        )
        self.response_prompt_variant = str(response_prompt_variant)

        print(f"Initializing Memoryos for user '{self.user_id}'. Data path: {self.data_storage_path}")
        print(f"Using unified LLM model: {self.llm_model}")

        # Initialize OpenAI Client
        self.client = OpenAIClient(api_key=openai_api_key, base_url=openai_base_url)
        if self.mid_term_structure == "graph":
            if not graph_path:
                raise ValueError("graph_path is required when mid_term_structure='graph'")
            self.memory_structure = GraphMemory(
                file_path=graph_path,
                client=self.client,
                llm_model=self.llm_model,
                context_window=self.graph_options.get("context_window", 3),
                candidate_count=self.graph_options.get("candidate_count", 10),
                dedupe_candidate_count=self.graph_options.get(
                    "dedupe_candidate_count", 20
                ),
                fuzzy_threshold=self.graph_options.get("fuzzy_threshold", 0.90),
                search_hops=self.graph_options.get("search_hops", 1),
                use_llm_dedup=self.graph_options.get("use_llm_dedup", True),
                use_edge_dedup=self.graph_options.get("use_edge_dedup", True),
            )
        else:
            self.memory_structure = tree or MemTree("")
        self.tree = self.memory_structure

        # Define file paths for user-specific data
        self.user_data_dir = os.path.join(self.data_storage_path, self.user_id)
        user_short_term_path = os.path.join(self.user_data_dir, "short_term.json")
        user_mid_term_path = os.path.join(self.user_data_dir, "mid_term.json")
        # Ensure directories exist
        ensure_directory_exists(user_short_term_path) # ensure_directory_exists operates on the file path, creating parent dirs
        ensure_directory_exists(user_mid_term_path)

        # Initialize Memory Modules for User
        self.short_term_memory = ShortTermMemory(file_path=user_short_term_path, max_capacity=short_term_capacity)
        self.mid_term_memory = MidTermMemory(file_path=user_mid_term_path, client=self.client)

        # Initialize Orchestration Modules
        self.updater = AblationUpdater(
            short_term_memory=self.short_term_memory,
            mid_term_memory=self.mid_term_memory,
            client=self.client,
            llm_model=self.llm_model,
            memory_structure=self.memory_structure,
            memory_granularity=self.memory_granularity,
            mid_term_structure=self.mid_term_structure,
            segment_threshold=self.segment_threshold,
        )
    
    def _proccess_mid_term_to_long_term(self):
        pass    

    def add_memory(self, speaker_a: str, speaker_b: str, speaker_a_input: str, speaker_b_input: str, timestamp: str = None):
        """
        Adds a new QA pair (memory) to the system.
        meta_data is not used in the current refactoring but kept for future use.
        """
        if not timestamp:
            timestamp = get_timestamp()
        
        qa_pair = {
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "speaker_a_input": speaker_a_input,
            "speaker_b_input": speaker_b_input,
            "timestamp": timestamp
            # meta_data can be added here if it needs to be stored with the QA pair
        }
        self.short_term_memory.add_qa_pair(qa_pair)
        # print(f"Memoryos: Added QA to short-term. {speaker_a}: {speaker_a_input[:20]}... {speaker_b}: {speaker_b_input[:20]}...")

        if self.short_term_memory.is_full():
            # print("Memoryos: Short-term memory full. Processing to mid-term.")
            # self.updater.process_short_term_to_mid_term()
            self.updater.process_short_term_to_mid_term(mode='half')
        
        self._proccess_mid_term_to_long_term()

    def get_response(self, query: str, mode:str, speaker_a: str, speaker_b: str) -> str:

        if self.include_recent_dialogue_in_response:
            short_term_history = self.short_term_memory.get_all()
            history_text = "\n".join(
                [
                    f"Timestamp: {qa.get('timestamp', '')}\n"
                    f"{qa.get('speaker_a', '')}: "
                    f"{qa.get('speaker_a_input', '')}\n"
                    f"{qa.get('speaker_b', '')}: "
                    f"{qa.get('speaker_b_input', '')}"
                    for qa in short_term_history
                ]
            )
            history_text = history_text if history_text.strip() else "None"
        else:
            history_text = "None"

        if mode == "merge":        
            """
            First version: flattened retrieval over all dialogs and summaries.
            """
            # Retrieve relevant mid-term memories
            if self.mid_term_structure == "graph":
                contexts = self.memory_structure.retrieve(
                    query, top_k=self.top_k_retrieve
                )
            else:
                contexts = retrieve(query)
            contexts = [list(item.values())[0] for item in contexts]
            contexts_text = "\n".join(contexts)
            response_prompt = prompt.RESPONSE_PROMPT_MERGE_2.format(
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                history=history_text,
                retrieved=contexts_text,
                query=query,
            )
            response = self.client.chat_completion(model=self.llm_model, messages=[{"role": "system", "content": f"You are an expert in memory analysis. You can find and analyze relevant content within conversations and other information based on the questions provided.Your task is to answer questions between {speaker_a} and {speaker_b}."}, 
                                                                                   {"role": "user", "content": response_prompt}], stage="answer")
            return contexts, response
        
        if mode == "split":
            if self.mid_term_structure == "graph":
                if self.graph_include_original_text:
                    seg_contexts = (
                        self.memory_structure.retrieve_with_original_text(
                            query,
                            self.mid_term_memory.segments,
                            top_k=self.top_k_retrieve,
                        )
                    )
                else:
                    seg_contexts = self.memory_structure.retrieve(
                        query, top_k=self.top_k_retrieve
                    )
                dial_contexts = (
                    []
                    if self.graph_only_retrieval
                    else retrieve(query, mode="dial")
                )
            else:
                seg_contexts = retrieve(
                    query, mode="seg", top_k=self.other_memory_top_k
                )
                dial_contexts = retrieve(
                    query, mode="dial", top_k=self.dialogue_top_k
                )
            seg_contexts = [list(item.values())[0] for item in seg_contexts]
            dial_contexts = [list(item.values())[0] for item in dial_contexts]
            if self.response_prompt_variant == "tree_optimized_no_recent":
                seg_contexts_text = "\n\n".join(
                    f"[TREE-{index}]\n{context}"
                    for index, context in enumerate(seg_contexts, start=1)
                )
                dial_contexts_text = "\n\n".join(
                    f"[DIALOGUE-{index}]\n{context}"
                    for index, context in enumerate(dial_contexts, start=1)
                )
                response_prompt = (
                    prompt.RESPONSE_PROMPT_TREE_OPTIMIZED_NO_RECENT.format(
                        speaker_a=speaker_a,
                        speaker_b=speaker_b,
                        seg_retrieved=seg_contexts_text or "None",
                        dial_retrieved=dial_contexts_text or "None",
                        query=query,
                    )
                )
                system_prompt = (
                    "You are a grounded conversation-memory QA assistant. "
                    "Return only the final concise answer supported by evidence."
                )
                answer_temperature = 0.0
            else:
                seg_contexts_text = "\n".join(seg_contexts)
                dial_contexts_text = "\n".join(dial_contexts)
                response_prompt = prompt.RESPONSE_PROMPT_SPLIT.format(
                    speaker_a=speaker_a,
                    speaker_b=speaker_b,
                    history=history_text,
                    seg_retrieved=seg_contexts_text,
                    dial_retrieved=dial_contexts_text,
                    query=query,
                )
                system_prompt = (
                    "You are an expert in memory analysis. You can find and "
                    "analyze relevant content within conversations and other "
                    f"information based on the questions provided. Your task "
                    f"is to answer questions between {speaker_a} and {speaker_b}."
                )
                answer_temperature = 0.7
            response = self.client.chat_completion(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": response_prompt},
                ],
                temperature=answer_temperature,
                stage="answer",
            )
            return [seg_contexts, dial_contexts], response
