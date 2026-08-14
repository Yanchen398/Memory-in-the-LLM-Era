import uuid

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from transformers import DynamicCache


class ActivationMemoryItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory: Any
    metadata: dict = {}


class KVCacheItem(ActivationMemoryItem):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory: DynamicCache = Field(
        default_factory=DynamicCache,
        description="Dynamic cache for storing key-value pairs in the memory.",
    )
    metadata: dict = Field(
        default_factory=dict, description="Metadata associated with the KV cache item."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)  # To allow DynamicCache as a field type
