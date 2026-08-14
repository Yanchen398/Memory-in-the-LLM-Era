from abc import ABC, abstractmethod

from memos.configs.llm import BaseLLMConfig
from memos.types import MessageList


class BaseLLM(ABC):
    """Base class for all LLMs."""

    @abstractmethod
    def __init__(self, config: BaseLLMConfig):
        """Initialize the LLM with the given configuration."""

    @abstractmethod
    def generate(self, messages: MessageList, **kwargs) -> str:
        """Generate a response from the LLM."""
