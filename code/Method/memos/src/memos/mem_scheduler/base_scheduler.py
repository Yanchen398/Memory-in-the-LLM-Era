import queue
import threading
import time

from abc import abstractmethod
from queue import Queue

from memos.configs.mem_scheduler import BaseSchedulerConfig
from memos.llms.base import BaseLLM
from memos.log import get_logger
from memos.mem_scheduler.modules.dispatcher import SchedulerDispatcher
from memos.mem_scheduler.modules.redis_service import RedisSchedulerModule
from memos.mem_scheduler.modules.schemas import (
    DEFAULT_CONSUME_INTERVAL_SECONDS,
    DEFAULT_THREAD__POOL_MAX_WORKERS,
    ScheduleLogForWebItem,
    ScheduleMessageItem,
)


logger = get_logger(__name__)


class BaseScheduler(RedisSchedulerModule):
    """Base class for all mem_scheduler."""

    def __init__(self, config: BaseSchedulerConfig):
        """Initialize the scheduler with the given configuration."""
        super().__init__()
        self.config = config
        self.max_workers = self.config.get(
            "thread_pool_max_workers", DEFAULT_THREAD__POOL_MAX_WORKERS
        )
        self.retriever = None
        self.monitor = None
        self.enable_parallel_dispatch = self.config.get("enable_parallel_dispatch", False)
        self.dispatcher = SchedulerDispatcher(
            max_workers=self.max_workers, enable_parallel_dispatch=self.enable_parallel_dispatch
        )

        # message queue
        self.memos_message_queue: Queue[ScheduleMessageItem] = Queue()
        self._web_log_message_queue: Queue[ScheduleLogForWebItem] = Queue()
        self._consumer_thread = None  # Reference to our consumer thread
        self._running = False
        self._consume_interval = self.config.get(
            "consume_interval_seconds", DEFAULT_CONSUME_INTERVAL_SECONDS
        )

        # others
        self._current_user_id: str | None = None

    @abstractmethod
    def initialize_modules(self, chat_llm: BaseLLM) -> None:
        """Initialize all necessary modules for the scheduler

        Args:
            chat_llm: The LLM instance to be used for chat interactions
        """

    def submit_messages(self, messages: ScheduleMessageItem | list[ScheduleMessageItem]):
        """Submit multiple messages to the message queue."""
        if isinstance(messages, ScheduleMessageItem):
            messages = [messages]  # transform single message to list

        for message in messages:
            self.memos_message_queue.put(message)
            logger.info(f"Submitted message: {message.label} - {message.content}")

    def _submit_web_logs(self, messages: ScheduleLogForWebItem | list[ScheduleLogForWebItem]):
        if isinstance(messages, ScheduleLogForWebItem):
            messages = [messages]  # transform single message to list

        for message in messages:
            self._web_log_message_queue.put(message)
            logger.info(
                f"Submitted Scheduling log for web: {message.log_title} - {message.log_content}"
            )
        logger.debug(f"{len(messages)} submitted. {self._web_log_message_queue.qsize()} in queue.")

    def get_web_log_messages(self) -> list[dict]:
        """
        Retrieves all web log messages from the queue and returns them as a list of JSON-serializable dictionaries.

        Returns:
            List[dict]: A list of dictionaries representing ScheduleLogForWebItem objects,
                       ready for JSON serialization. The list is ordered from oldest to newest.
        """
        messages = []

        # Process all items in the queue
        while not self._web_log_message_queue.empty():
            item = self._web_log_message_queue.get()
            # Convert the ScheduleLogForWebItem to a dictionary and ensure datetime is serialized
            item_dict = item.to_dict()
            messages.append(item_dict)
        return messages

    def _message_consumer(self) -> None:
        """
        Continuously checks the queue for messages and dispatches them.

        Runs in a dedicated thread to process messages at regular intervals.
        """
        while self._running:  # Use a running flag for graceful shutdown
            try:
                # Check if queue has messages (non-blocking)
                if not self.memos_message_queue.empty():
                    # Get all available messages at once
                    messages = []
                    while not self.memos_message_queue.empty():
                        try:
                            messages.append(self.memos_message_queue.get_nowait())
                        except queue.Empty:
                            break

                    if messages:
                        try:
                            self.dispatcher.dispatch(messages)
                        except Exception as e:
                            logger.error(f"Error dispatching messages: {e!s}")
                        finally:
                            # Mark all messages as processed
                            for _ in messages:
                                self.memos_message_queue.task_done()

                # Sleep briefly to prevent busy waiting
                time.sleep(self._consume_interval)  # Adjust interval as needed

            except Exception as e:
                logger.error(f"Unexpected error in message consumer: {e!s}")
                time.sleep(self._consume_interval)  # Prevent tight error loops

    def start(self) -> None:
        """
        Start the message consumer thread.

        Initializes and starts a daemon thread that will periodically
        check for and process messages from the queue.
        """
        if self._consumer_thread is not None and self._consumer_thread.is_alive():
            logger.warning("Consumer thread is already running")
            return

        self._running = True
        self._consumer_thread = threading.Thread(
            target=self._message_consumer,
            daemon=True,  # Allows program to exit even if thread is running
            name="MessageConsumerThread",
        )
        self._consumer_thread.start()
        logger.info("Message consumer thread started")

    def stop(self) -> None:
        """Stop the consumer thread and clean up resources."""
        if self._consumer_thread is None or not self._running:
            logger.warning("Consumer thread is not running")
            return
        self._running = False
        if self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=5.0)  # Wait up to 5 seconds
            if self._consumer_thread.is_alive():
                logger.warning("Consumer thread did not stop gracefully")
        logger.info("Message consumer thread stopped")
