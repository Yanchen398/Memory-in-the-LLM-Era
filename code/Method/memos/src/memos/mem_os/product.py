import json

from collections.abc import Generator
from typing import Literal

from memos.configs.mem_os import MOSConfig
from memos.mem_os.core import MOSCore
from memos.memories.activation.item import ActivationMemoryItem
from memos.memories.parametric.item import ParametricMemoryItem
from memos.memories.textual.item import TextualMemoryMetadata, TreeNodeTextualMemoryMetadata
from memos.types import MessageList


class MOSProduct(MOSCore):
    """
    The MOSProduct class inherits from MOSCore mainly for product usage.
    """

    def __init__(self, config: MOSConfig):
        super().__init__(config)

    def get_suggestion_query(self, user_id: str) -> list[str]:
        """Get suggestion query from LLM.
        Args:
            user_id (str, optional): Custom user ID.

        Returns:
            list[str]: The suggestion query list.
        """

    def chat(
        self,
        query: str,
        user_id: str,
        cube_id: str | None = None,
        history: MessageList | None = None,
    ) -> Generator[str, None, None]:
        """Chat with LLM SSE Type.
        Args:
            query (str): Query string.
            user_id (str, optional): Custom user ID.
            cube_id (str, optional): Custom cube ID for user.
            history (list[dict], optional): Chat history.

        Returns:
            Generator[str, None, None]: The response string generator.
        """
        memories_list = self.search(query)["act_mem"]
        content_list = []
        for memory in memories_list:
            content_list.append(memory.content)
        yield f"data: {json.dumps({'type': 'metadata', 'content': content_list})}\n\n"
        llm_response = super().chat(query, user_id)
        for chunk in llm_response:
            chunk_data: str = f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"
            yield chunk_data
        reference = [{"id": "1234"}]
        yield f"data: {json.dumps({'type': 'reference', 'content': reference})}\n\n"
        yield f"data: {json.dumps({'type': 'end'})}\n\n"

    def get_all(
        self,
        user_id: str,
        memory_type: Literal["text_mem", "act_mem", "param_mem"],
        cube_id: str | None = None,
    ) -> list[
        dict[
            str,
            str
            | list[
                TextualMemoryMetadata
                | TreeNodeTextualMemoryMetadata
                | ActivationMemoryItem
                | ParametricMemoryItem
            ],
        ]
    ]:
        """Get all memory items for a user.

        Args:
            user_id (str): The ID of the user.
            cube_id (str | None, optional): The ID of the cube. Defaults to None.
            memory_type (Literal["text_mem", "act_mem", "param_mem"]): The type of memory to get.

        Returns:
            list[TextualMemoryMetadata | TreeNodeTextualMemoryMetadata | ActivationMemoryItem | ParametricMemoryItem]: A list of memory items.
        """
        memory_list = super().get_all(user_id, cube_id)[memory_type]
        return memory_list
