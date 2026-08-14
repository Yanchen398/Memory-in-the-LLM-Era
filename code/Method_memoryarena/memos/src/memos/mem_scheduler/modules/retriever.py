from memos.log import get_logger
from memos.mem_scheduler.modules.base import BaseSchedulerModule


logger = get_logger(__name__)


class SchedulerRetriever(BaseSchedulerModule):
    def __init__(self, chat_llm, context_window_size=5):
        """
        monitor: Object used to acquire monitoring information
        mem_cube: Object/interface for querying the underlying database
        context_window_size: Size of the context window for conversation history
        """
        super().__init__()

        self.monitors = {}
        self.context_window_size = context_window_size

        self._chat_llm = chat_llm
        self._current_mem_cube = None

    @property
    def memory_texts(self) -> list[str]:
        """The memory cube associated with this MemChat."""
        return self._memory_text_list

    @memory_texts.setter
    def memory_texts(self, value: list[str]) -> None:
        """The memory cube associated with this MemChat."""
        self._memory_text_list = value

    def fetch_context(self):
        """
        Extract the context window from the current conversation
        conversation_history: a list (in chronological order)
        """
        return self._memory_text_list[-self.context_window_size :]

    def retrieve(self, query: str, memory_texts: list[str], top_k: int = 5) -> list[str]:
        return None
