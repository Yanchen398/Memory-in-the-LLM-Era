from typing import Any, ClassVar

from pydantic import BaseModel, Field, field_validator, model_validator

from memos.configs.base import BaseConfig


class BaseGraphDBConfig(BaseConfig):
    """Base class for all graph database configurations."""

    uri: str
    user: str
    password: str


class Neo4jGraphDBConfig(BaseGraphDBConfig):
    """Neo4j-specific configuration."""

    db_name: str = Field(..., description="The name of the target Neo4j database")
    auto_create: bool = Field(
        default=False, description="Whether to create the DB if it doesn't exist"
    )
    embedding_dimension: int = Field(default=384, description="Dimension of vector embedding")


class GraphDBConfigFactory(BaseModel):
    backend: str = Field(..., description="Backend for graph database")
    config: dict[str, Any] = Field(..., description="Configuration for the graph database backend")

    backend_to_class: ClassVar[dict[str, Any]] = {
        "neo4j": Neo4jGraphDBConfig,
    }

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, backend: str) -> str:
        if backend not in cls.backend_to_class:
            raise ValueError(f"Unsupported graph db backend: {backend}")
        return backend

    @model_validator(mode="after")
    def instantiate_config(self):
        config_class = self.backend_to_class[self.backend]
        self.config = config_class(**self.config)
        return self
