"""
Anthropic Client - Implementation of BaseModelClient for Anthropic Claude API
Supports Claude's tool use API
"""

import os
import json
from typing import List, Dict, Optional

from .base_client import BaseModelClient, ModelResponse, ToolCall

# Import Anthropic
try:
    import anthropic
except ImportError:
    raise ImportError("Please install anthropic: pip install anthropic")


class AnthropicClient(BaseModelClient):
    def __init__(self, model_name: str = "claude-sonnet-4-20250514", api_key: str = None, base_url: str = None):
        super().__init__(model_name)
        
        # Configure API key
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        # Configure base URL (for proxies)
        base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        
        if base_url:
            self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        else:
            self.client = anthropic.Anthropic(api_key=api_key)
    
    def _convert_tools_to_anthropic_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI-style tools to Anthropic format."""
        if not tools:
            return []
        
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                })
        
        return anthropic_tools
    
    def _convert_messages_to_anthropic_format(self, messages: List[Dict]) -> tuple:
        """
        Convert OpenAI-style messages to Anthropic format.
        Returns (system_prompt, messages_list)
        """
        system_prompt = None
        anthropic_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if role == "system":
                system_prompt = content
            elif role == "user":
                if content:  # Skip if content is None or empty
                    anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                # Check if this is a tool call message
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    # Convert to Anthropic tool_use format
                    content_blocks = []
                    if content:
                        content_blocks.append({"type": "text", "text": content})
                    for tc in tool_calls:
                        func = tc.get("function", tc)
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", func.get("id", "")),
                            "name": func.get("name", ""),
                            "input": json.loads(func.get("arguments", "{}")) if isinstance(func.get("arguments"), str) else func.get("arguments", {})
                        })
                    anthropic_messages.append({"role": "assistant", "content": content_blocks})
                elif content:
                    anthropic_messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                # Tool result - add as user message with tool_result
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": content
                    }]
                })
        
        return system_prompt, anthropic_messages
    
    def chat_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 8192
    ) -> ModelResponse:
        """
        Send messages to Claude with tools enabled.
        """
        try:
            anthropic_tools = self._convert_tools_to_anthropic_format(tools)
            
            # Convert messages
            system_prompt, anthropic_messages = self._convert_messages_to_anthropic_format(messages)
            
            # Build request kwargs
            kwargs = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "messages": anthropic_messages
            }
            
            if system_prompt:
                kwargs["system"] = system_prompt
            
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools
            
            if temperature > 0:
                kwargs["temperature"] = temperature
            
            response = self.client.messages.create(**kwargs)
            
            if hasattr(response, 'usage'):
                self.total_input_tokens += response.usage.input_tokens
                self.total_output_tokens += response.usage.output_tokens
            
            # Parse response
            content = None
            tool_calls = None
            
            text_parts = []
            func_calls = []
            
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    func_calls.append(ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {}
                    ))
            
            if text_parts:
                content = "\n".join(text_parts)
            if func_calls:
                tool_calls = func_calls
            
            # Convert raw response to dict for serialization (full trace)
            try:
                raw_response_dict = response.to_dict()
            except Exception:
                try:
                    raw_response_dict = response.model_dump()
                except Exception:
                    raw_response_dict = {"error": "Could not serialize response"}
            
            return ModelResponse(
                content=content,
                tool_calls=tool_calls,
                raw_response=raw_response_dict
            )
            
        except Exception as e:
            print(f"Anthropic API error: {e}")
            raise
    
    def format_tool_result(self, tool_call_id: str, result: str, name: str = None) -> Dict:
        """
        Format tool result as a message for Claude.
        Note: Claude doesn't need the name parameter, but we accept it for API consistency.
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        }
    
    def format_assistant_tool_calls(self, tool_calls: List[ToolCall]) -> Dict:
        """
        Format assistant's tool calls as a message.
        """
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                    }
                }
                for tc in tool_calls
            ]
        }
    
    def get_usage_stats(self) -> Dict:
        input_cost_per_1m = 3.0
        output_cost_per_1m = 15.0
        
        total_cost = (
            self.total_input_tokens * input_cost_per_1m / 1_000_000 +
            self.total_output_tokens * output_cost_per_1m / 1_000_000
        )
        
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_cost': total_cost
        }


# For testing
if __name__ == "__main__":
    client = AnthropicClient(model_name="claude-sonnet-4-20250514")
    
    messages = [
        {"role": "system", "content": "You are a travel planning assistant."},
        {"role": "user", "content": "Search for flights from New York to Los Angeles on 2024-03-15"}
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "FlightSearch",
                "description": "Search for flights",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string"},
                        "destination": {"type": "string"},
                        "date": {"type": "string"}
                    },
                    "required": ["origin", "destination", "date"]
                }
            }
        }
    ]
    
    response = client.chat_with_tools(messages, tools)
    print(f"Content: {response.content}")
    print(f"Tool calls: {response.tool_calls}")
    print(f"Usage: {client.get_usage_stats()}")
