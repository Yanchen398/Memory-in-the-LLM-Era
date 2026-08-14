from abc import ABC, abstractmethod

from memos.configs.embedder import BaseEmbedderConfig


class BaseEmbedder(ABC):
    """Base class for all Embedding models."""

    @abstractmethod
    def __init__(self, config: BaseEmbedderConfig):
        """Initialize the embedding model with the given configuration."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for the given texts."""
