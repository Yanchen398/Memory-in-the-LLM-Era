import json

from datetime import datetime, timedelta

from memos.configs.mem_scheduler import GeneralSchedulerConfig
from memos.llms.base import BaseLLM
from memos.log import get_logger
from memos.mem_cube.general import GeneralMemCube
from memos.mem_scheduler.base_scheduler import BaseScheduler
from memos.mem_scheduler.modules.monitor import SchedulerMonitor
from memos.mem_scheduler.modules.retriever import SchedulerRetriever
from memos.mem_scheduler.modules.schemas import (
    ANSWER_LABEL,
    DEFAULT_ACT_MEM_DUMP_PATH,
    DEFAULT_ACTIVATION_MEM_SIZE,
    NOT_INITIALIZED,
    QUERY_LABEL,
    ScheduleLogForWebItem,
    ScheduleMessageItem,
    TextMemory_SEARCH_METHOD,
    TreeTextMemory_SEARCH_METHOD,
)
from memos.memories.textual.tree import TextualMemoryItem, TreeTextMemory
from memos.templates.mem_scheduler_prompts import MEMORY_ASSEMBLY_TEMPLATE


logger = get_logger(__name__)


class GeneralScheduler(BaseScheduler):
    def __init__(self, config: GeneralSchedulerConfig):
        """Initialize the scheduler with the given configuration."""
        super().__init__(config)
        self.top_k = self.config.get("top_k", 10)
        self.top_n = self.config.get("top_n", 5)
        self.act_mem_update_interval = self.config.get("act_mem_update_interval", 300)
        self.context_window_size = self.config.get("context_window_size", 5)
        self.activation_mem_size = self.config.get(
            "activation_mem_size", DEFAULT_ACTIVATION_MEM_SIZE
        )
        self.act_mem_dump_path = self.config.get("act_mem_dump_path", DEFAULT_ACT_MEM_DUMP_PATH)
        self.search_method = TextMemory_SEARCH_METHOD
        self._last_activation_mem_update_time = 0.0
        self.query_list = []

        # register handlers
        handlers = {
            QUERY_LABEL: self._query_message_consume,
            ANSWER_LABEL: self._answer_message_consume,
        }
        self.dispatcher.register_handlers(handlers)

    def initialize_modules(self, chat_llm: BaseLLM):
        self.chat_llm = chat_llm
        self.monitor = SchedulerMonitor(
            chat_llm=self.chat_llm, activation_mem_size=self.activation_mem_size
        )
        self.retriever = SchedulerRetriever(chat_llm=self.chat_llm)
        logger.debug("GeneralScheduler has been initialized")

    def _answer_message_consume(self, messages: list[ScheduleMessageItem]) -> None:
        """
        Process and handle answer trigger messages from the queue.

        Args:
          messages: List of answer messages to process
        """
        # TODO: This handler is not ready yet
        logger.debug(f"Messages {messages} assigned to {ANSWER_LABEL} handler.")
        for msg in messages:
            if msg.label is not ANSWER_LABEL:
                logger.error(f"_answer_message_consume is not designed for {msg.label}")
                continue
            answer = msg.content
            self._current_user_id = msg.user_id
            self._current_mem_cube_id = msg.mem_cube_id
            self._current_mem_cube = msg.mem_cube

            # Get current activation memory items
            current_activation_mem = [
                item["memory"]
                for item in self.monitor.activation_memory_freq_list
                if item["memory"] is not None
            ]

            # Update memory frequencies based on the answer
            # TODO: not implemented
            self.monitor.activation_memory_freq_list = self.monitor.update_freq(
                answer=answer, activation_memory_freq_list=self.monitor.activation_memory_freq_list
            )

            # Check if it's time to update activation memory
            now = datetime.now()
            if (now - self._last_activation_mem_update_time) >= timedelta(
                seconds=self.act_mem_update_interval
            ):
                # TODO: not implemented
                self.update_activation_memory(current_activation_mem)
                self._last_activation_mem_update_time = now

            # recording messages
            log_message = self.create_autofilled_log_item(
                log_title="memos answer triggers scheduling...",
                label=ANSWER_LABEL,
                log_content="activation_memory has been updated",
            )
            self._submit_web_logs(messages=log_message)

    def _query_message_consume(self, messages: list[ScheduleMessageItem]) -> None:
        """
        Process and handle query trigger messages from the queue.

        Args:
            messages: List of query messages to process
        """
        logger.debug(f"Messages {messages} assigned to {QUERY_LABEL} handler.")
        for msg in messages:
            if msg.label is not QUERY_LABEL:
                logger.error(f"_query_message_consume is not designed for {msg.label}")
                continue
            # Process the query in a session turn
            self._current_user_id = msg.user_id
            self._current_mem_cube_id = msg.mem_cube_id
            self._current_mem_cube = msg.mem_cube
            self.process_session_turn(query=msg.content, top_k=self.top_k, top_n=self.top_n)

    def process_session_turn(
        self,
        query: str,
        top_k: int = 10,
        top_n: int = 5,
    ) -> None:
        """
        Process a dialog turn:
        - If q_list reaches window size, trigger retrieval;
        - Immediately switch to the new memory if retrieval is triggered.
        """
        q_list = [query]
        self.query_list.append(query)
        text_mem_base = self.mem_cube.text_mem
        if isinstance(text_mem_base, TreeTextMemory):
            working_memory: list[TextualMemoryItem] = text_mem_base.get_working_memory()
        else:
            logger.error("Not implemented!")
            return
        text_working_memory: list[str] = [w_m.memory for w_m in working_memory]
        intent_result = self.monitor.detect_intent(
            q_list=q_list, text_working_memory=text_working_memory
        )
        if intent_result["trigger_retrieval"]:
            missing_evidence = intent_result["missing_evidence"]
            num_evidence = len(missing_evidence)
            k_per_evidence = max(1, top_k // max(1, num_evidence))
            new_candidates = []
            for item in missing_evidence:
                logger.debug(f"missing_evidence: {item}")
                results = self.search(query=item, top_k=k_per_evidence, method=self.search_method)
                logger.debug(f"search results for {missing_evidence}: {results}")
                new_candidates.extend(results)

            # recording messages
            log_message = self.create_autofilled_log_item(
                log_title="user query triggers scheduling...",
                label=QUERY_LABEL,
                log_content=f"search new candidates for working memory: {len(new_candidates)}",
            )
            self._submit_web_logs(messages=log_message)
            new_order_working_memory = self.replace_working_memory(
                original_memory=working_memory, new_memory=new_candidates, top_k=top_k, top_n=top_n
            )
            self.update_activation_memory(new_order_working_memory)

    def create_autofilled_log_item(
        self, log_title: str, log_content: str, label: str
    ) -> ScheduleLogForWebItem:
        # TODO: create the log iterm with real stats
        text_mem_base: TreeTextMemory = self.mem_cube.text_mem
        current_memory_sizes = {
            "long_term_memory_size": NOT_INITIALIZED,
            "user_memory_size": NOT_INITIALIZED,
            "working_memory_size": NOT_INITIALIZED,
            "transformed_act_memory_size": NOT_INITIALIZED,
            "parameter_memory_size": NOT_INITIALIZED,
        }
        memory_capacities = {
            "long_term_memory_capacity": text_mem_base.memory_manager.memory_size["LongTermMemory"],
            "user_memory_capacity": text_mem_base.memory_manager.memory_size["UserMemory"],
            "working_memory_capacity": text_mem_base.memory_manager.memory_size["WorkingMemory"],
            "transformed_act_memory_capacity": NOT_INITIALIZED,
            "parameter_memory_capacity": NOT_INITIALIZED,
        }

        log_message = ScheduleLogForWebItem(
            user_id=self._current_user_id,
            mem_cube_id=self._current_mem_cube_id,
            label=label,
            log_title=log_title,
            log_content=log_content,
            current_memory_sizes=current_memory_sizes,
            memory_capacities=memory_capacities,
        )
        return log_message

    @property
    def mem_cube(self) -> GeneralMemCube:
        """The memory cube associated with this MemChat."""
        return self._current_mem_cube

    @mem_cube.setter
    def mem_cube(self, value: GeneralMemCube) -> None:
        """The memory cube associated with this MemChat."""
        self._current_mem_cube = value
        self.retriever.mem_cube = value

    def replace_working_memory(
        self,
        original_memory: list[TextualMemoryItem],
        new_memory: list[TextualMemoryItem],
        top_k: int = 10,
        top_n: int = 5,
    ) -> None | list[TextualMemoryItem]:
        new_order_memory = None
        text_mem_base = self.mem_cube.text_mem
        if isinstance(text_mem_base, TreeTextMemory):
            text_mem_base: TreeTextMemory = text_mem_base
            combined_text_memory = [new_m.memory for new_m in original_memory] + [
                new_m.memory for new_m in new_memory
            ]
            combined_memory = original_memory + new_memory
            memory_map = {mem_obj.memory: mem_obj for mem_obj in combined_memory}

            unique_memory = list(dict.fromkeys(combined_text_memory))
            prompt = self.build_prompt(
                "memory_reranking", query="", current_order=unique_memory, staging_buffer=[]
            )
            response = self.chat_llm.generate([{"role": "user", "content": prompt}])
            response = json.loads(response)
            new_order_text_memory = response.get("new_order", [])[: top_n + top_k]

            new_order_memory = []
            for text in new_order_text_memory:
                if text in memory_map:
                    new_order_memory.append(memory_map[text])
                else:
                    logger.warning(
                        f"Memory text not found in memory map. text: {text}; memory_map: {memory_map}"
                    )

            text_mem_base.replace_working_memory(new_order_memory[top_n:])
            new_order_memory = new_order_memory[:top_n]
            logger.info(
                f"The working memory has been replaced with {len(new_order_memory)} new memories."
            )
        else:
            logger.error("memory_base is not supported")

        return new_order_memory

    def search(self, query: str, top_k: int, method=TreeTextMemory_SEARCH_METHOD):
        text_mem_base = self.mem_cube.text_mem
        if isinstance(text_mem_base, TreeTextMemory) and method == TextMemory_SEARCH_METHOD:
            results_long_term = text_mem_base.search(
                query=query, top_k=top_k, memory_type="LongTermMemory"
            )
            results_user = text_mem_base.search(query=query, top_k=top_k, memory_type="UserMemory")
            results = results_long_term + results_user
        else:
            logger.error("Not implemented.")
            results = None
        return results

    def update_activation_memory(self, new_memory: list[str | TextualMemoryItem]) -> None:
        """
        Update activation memory by extracting KVCacheItems from new_memory (list of str),
        add them to a KVCacheMemory instance, and dump to disk.
        """
        # TODO: The function of update activation memory is waiting to test
        if len(new_memory) == 0:
            logger.error("update_activation_memory: new_memory is empty.")
            return
        if isinstance(new_memory[0], TextualMemoryItem):
            new_text_memory = [mem.memory for mem in new_memory]
        elif isinstance(new_memory[0], str):
            new_text_memory = new_memory
        else:
            logger.error("Not Implemented.")

        try:
            act_mem = self.mem_cube.act_mem

            text_memory = MEMORY_ASSEMBLY_TEMPLATE.format(
                memory_text="".join(
                    [
                        f"{i + 1}. {sentence.strip()}\n"
                        for i, sentence in enumerate(new_text_memory)
                        if sentence.strip()  # Skip empty strings
                    ]
                )
            )
            act_mem.delete_all()
            cache_item = act_mem.extract(text_memory)
            act_mem.add(cache_item)
            act_mem.dump(self.act_mem_dump_path)
        except Exception as e:
            logger.warning(f"MOS-based activation memory update failed: {e}")
