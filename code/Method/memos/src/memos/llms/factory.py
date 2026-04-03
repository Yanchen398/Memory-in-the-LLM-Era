from typing import Any, ClassVar

from memos.configs.llm import LLMConfigFactory
from memos.llms.base import BaseLLM
from memos.llms.hf import HFLLM
from memos.llms.ollama import OllamaLLM
from memos.llms.openai import OpenAILLM


class LLMFactory(BaseLLM):
    """Factory class for creating LLM instances."""

    backend_to_class: ClassVar[dict[str, Any]] = {
        "openai": OpenAILLM,
        "ollama": OllamaLLM,
        "huggingface": HFLLM,
    }

    @classmethod
    def from_config(cls, config_factory: LLMConfigFactory) -> BaseLLM:
        backend = config_factory.backend
        if backend not in cls.backend_to_class:
            raise ValueError(f"Invalid backend: {backend}")
        llm_class = cls.backend_to_class[backend]
        return llm_class(config_factory.config)
