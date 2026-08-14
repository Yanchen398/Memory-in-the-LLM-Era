from typing import Any, ClassVar

from pydantic import ConfigDict, Field, field_validator, model_validator

from memos.configs.base import BaseConfig
from memos.mem_scheduler.modules.schemas import (
    DEFAULT_ACT_MEM_DUMP_PATH,
    DEFAULT_ACTIVATION_MEM_SIZE,
    DEFAULT_CONSUME_INTERVAL_SECONDS,
    DEFAULT_THREAD__POOL_MAX_WORKERS,
)


class BaseSchedulerConfig(BaseConfig):
    """Base configuration class for mem_scheduler."""

    top_k: int = Field(
        default=10, description="Number of top candidates to consider in initial retrieval"
    )
    top_n: int = Field(default=5, description="Number of final results to return after processing")
    enable_parallel_dispatch: bool = Field(
        default=True, description="Whether to enable parallel message processing using thread pool"
    )
    thread_pool_max_workers: int = Field(
        default=DEFAULT_THREAD__POOL_MAX_WORKERS,
        gt=1,
        lt=20,
        description=f"Maximum worker threads in pool (default: {DEFAULT_THREAD__POOL_MAX_WORKERS})",
    )
    consume_interval_seconds: int = Field(
        default=DEFAULT_CONSUME_INTERVAL_SECONDS,
        gt=0,
        le=60,
        description=f"Interval for consuming messages from queue in seconds (default: {DEFAULT_CONSUME_INTERVAL_SECONDS})",
    )


class GeneralSchedulerConfig(BaseSchedulerConfig):
    act_mem_update_interval: int | None = Field(
        default=300, description="Interval in seconds for updating activation memory"
    )
    context_window_size: int | None = Field(
        default=5, description="Size of the context window for conversation history"
    )
    activation_mem_size: int | None = Field(
        default=DEFAULT_ACTIVATION_MEM_SIZE,  # Assuming DEFAULT_ACTIVATION_MEM_SIZE is 1000
        description="Maximum size of the activation memory",
    )
    act_mem_dump_path: str | None = Field(
        default=DEFAULT_ACT_MEM_DUMP_PATH,  # Replace with DEFAULT_ACT_MEM_DUMP_PATH
        description="File path for dumping activation memory",
    )


class SchedulerConfigFactory(BaseConfig):
    """Factory class for creating scheduler configurations."""

    backend: str = Field(..., description="Backend for scheduler")
    config: dict[str, Any] = Field(..., description="Configuration for the scheduler backend")

    model_config = ConfigDict(extra="forbid", strict=True)
    backend_to_class: ClassVar[dict[str, Any]] = {
        "general_scheduler": GeneralSchedulerConfig,
    }

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, backend: str) -> str:
        """Validate the backend field."""
        if backend not in cls.backend_to_class:
            raise ValueError(f"Invalid backend: {backend}")
        return backend

    @model_validator(mode="after")
    def create_config(self) -> "SchedulerConfigFactory":
        config_class = self.backend_to_class[self.backend]
        self.config = config_class(**self.config)
        return self
