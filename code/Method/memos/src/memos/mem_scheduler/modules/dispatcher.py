from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from memos.log import get_logger
from memos.mem_scheduler.modules.base import BaseSchedulerModule
from memos.mem_scheduler.modules.schemas import ScheduleMessageItem


logger = get_logger(__name__)


class SchedulerDispatcher(BaseSchedulerModule):
    """
    Thread pool-based message dispatcher that routes messages to dedicated handlers
    based on their labels.

    Features:
    - Dedicated thread pool per message label
    - Batch message processing
    - Graceful shutdown
    - Bulk handler registration
    """

    def __init__(self, max_workers=3, enable_parallel_dispatch=False):
        super().__init__()
        # Main dispatcher thread pool
        self.max_workers = max_workers
        # Only initialize thread pool if in parallel mode
        self.enable_parallel_dispatch = enable_parallel_dispatch
        if self.enable_parallel_dispatch:
            self.dispatcher_executor = ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix="dispatcher"
            )
        else:
            self.dispatcher_executor = None
        logger.info(f"enable_parallel_dispatch is set to {self.enable_parallel_dispatch}")
        # Registered message handlers
        self.handlers: dict[str, Callable] = {}
        # Dispatcher running state
        self._running = False

    def register_handler(self, label: str, handler: Callable[[list[ScheduleMessageItem]], None]):
        """
        Register a handler function for a specific message label.

        Args:
            label: Message label to handle
            handler: Callable that processes messages of this label
        """
        self.handlers[label] = handler

    def register_handlers(
        self, handlers: dict[str, Callable[[list[ScheduleMessageItem]], None]]
    ) -> None:
        """
        Bulk register multiple handlers from a dictionary.

        Args:
            handlers: Dictionary mapping labels to handler functions
                      Format: {label: handler_callable}
        """
        for label, handler in handlers.items():
            if not isinstance(label, str):
                logger.error(f"Invalid label type: {type(label)}. Expected str.")
                continue
            if not callable(handler):
                logger.error(f"Handler for label '{label}' is not callable.")
                continue
            self.register_handler(label=label, handler=handler)
        logger.info(f"Registered {len(handlers)} handlers in bulk")

    def _default_message_handler(self, messages: list[ScheduleMessageItem]) -> None:
        logger.debug(f"Using _default_message_handler to deal with messages: {messages}")

    def dispatch(self, msg_list: list[ScheduleMessageItem]):
        """
        Dispatch a list of messages to their respective handlers.

        Args:
            msg_list: List of ScheduleMessageItem objects to process
        """

        # Group messages by their labels
        label_groups = defaultdict(list)

        # Organize messages by label
        for message in msg_list:
            label_groups[message.label].append(message)

        # Process each label group
        for label, msgs in label_groups.items():
            if label not in self.handlers:
                logger.error(f"No handler registered for label: {label}")
                handler = self._default_message_handler
            else:
                handler = self.handlers[label]
            # dispatch to different handler
            logger.debug(f"Dispatch {len(msgs)} messages to {label} handler.")
            if self.enable_parallel_dispatch and self.dispatcher_executor is not None:
                self.dispatcher_executor.submit(handler, msgs)
            else:
                handler(msgs)  # Direct serial execution
