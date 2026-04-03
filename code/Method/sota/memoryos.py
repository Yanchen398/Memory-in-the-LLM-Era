from urllib import response
from .structure import MemTree
import os
from .utils import OpenAIClient, ensure_directory_exists, get_timestamp, retrieve
from .short_term import ShortTermMemory
from .mid_term import MidTermMemory
from .updater import Updater
from . import prompt

class Memoryos:
    def __init__(self, user_id: str, 
                 openai_api_key: str, 
                 data_storage_path: str,
                 openai_base_url: str, 
                 llm_model: str,
                 short_term_capacity=10,
                 tree: MemTree = MemTree(""),
                 segment_threshold=0.5
                 ):
        self.user_id = user_id
        self.data_storage_path = os.path.abspath(data_storage_path)
        self.llm_model = llm_model
        self.tree = tree
        self.segment_threshold = segment_threshold

        print(f"Initializing Memoryos for user '{self.user_id}'. Data path: {self.data_storage_path}")
        print(f"Using unified LLM model: {self.llm_model}")

        # Initialize OpenAI Client
        self.client = OpenAIClient(api_key=openai_api_key, base_url=openai_base_url)

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
        self.updater = Updater(short_term_memory=self.short_term_memory, 
                               mid_term_memory=self.mid_term_memory,
                               client=self.client,
                               llm_model=self.llm_model,
                               tree=self.tree,
                               segment_threshold=self.segment_threshold)
    
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

        # Get short-term history
        short_term_history = self.short_term_memory.get_all()
        history_text = "\n".join([f"Timestamp: {qa.get('timestamp','')}\n{qa.get('speaker_a','')}: {qa.get('speaker_a_input','')}\n{qa.get('speaker_b','')}: {qa.get('speaker_b_input','')}" for qa in short_term_history])
        history_text = history_text if history_text.strip() else "None"

        if mode == "merge":        
            """
            First version: flattened retrieval over all dialogs and summaries.
            """
            # Retrieve relevant mid-term memories
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
                                                                                   {"role": "user", "content": response_prompt}])
            return contexts, response
        
        if mode == "split":
            seg_contexts = retrieve(query, mode="seg")
            dial_contexts = retrieve(query, mode="dial")
            seg_contexts = [list(item.values())[0] for item in seg_contexts]
            dial_contexts = [list(item.values())[0] for item in dial_contexts]
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
            response = self.client.chat_completion(model=self.llm_model, messages=[{"role": "system", "content": f"You are an expert in memory analysis. You can find and analyze relevant content within conversations and other information based on the questions provided.Your task is to answer questions between {speaker_a} and {speaker_b}."}, 
                                                                                   {"role": "user", "content": response_prompt}])
            return [seg_contexts, dial_contexts], response
