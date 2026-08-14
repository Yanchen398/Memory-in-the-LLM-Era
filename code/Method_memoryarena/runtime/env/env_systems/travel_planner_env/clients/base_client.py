from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ToolCall:
    """Represents a tool call from the model"""
    id: str
    name: str
    arguments: dict


@dataclass
class ModelResponse:
    """Standardized response from any model"""
    content: Optional[str]  # Text content (may be None if only tool calls)
    tool_calls: Optional[List[ToolCall]]  # Tool calls (may be None if only text)
    raw_response: Any  # Original response object for debugging


class BaseModelClient(ABC):
    """
    Abstract base class for model clients.
    Implement this class to support different model providers.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
    
    @abstractmethod
    def chat_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> ModelResponse:
        pass
    
    @abstractmethod
    def format_tool_result(self, tool_call_id: str, result: str, name: str = None) -> Dict:
        pass
    
    @abstractmethod
    def format_assistant_tool_calls(self, tool_calls: List[ToolCall]) -> Dict:
        pass
    
    def get_usage_stats(self) -> Dict:
        """Get token usage statistics"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost
        }
    
    def reset_usage_stats(self):
        """Reset token usage statistics"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

