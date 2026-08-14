"""
OpenAI Client - Implementation of BaseModelClient for OpenAI API
Supports OpenAI's Tools API (function calling)
"""

import os
import json
import re
import time
from typing import Any, List, Dict, Optional
from openai import BadRequestError, OpenAI

from .base_client import BaseModelClient, ModelResponse, ToolCall
from ..cost_tracker import CostTracker


QWEN35_CHAT_EXTRA_BODY = {
    "top_k": 20,
    "chat_template_kwargs": {"enable_thinking": False},
}
CONTEXT_LIMIT_RE = re.compile(
    r"You passed (\d+) input tokens and requested (\d+) output tokens.*?"
    r"context length is only (\d+) tokens",
    re.IGNORECASE,
)
CONTEXT_LIMIT_TRANSFORMERS_RE = re.compile(
    r"maximum context length is (\d+) tokens.*?"
    r"requested (\d+) tokens \((\d+) in the messages, "
    r"(\d+) in the completion\)",
    re.IGNORECASE,
)
CONTEXT_RETRY_SAFETY_TOKENS = 64
CONTEXT_RETRY_MAX_ATTEMPTS = 9
CONTEXT_RETRY_MIN_TOKENS = 1
CONTEXT_RETRY_MIN_REDUCTION = 1024


def _is_qwen_model(model_name: str) -> bool:
    return "qwen" in (model_name or "").lower()


def _context_limited_max_tokens(error: Exception, requested: int) -> Optional[int]:
    error_text = str(error)
    match = CONTEXT_LIMIT_RE.search(error_text)
    if match:
        input_tokens, _, context_tokens = (
            int(value) for value in match.groups()
        )
    else:
        match = CONTEXT_LIMIT_TRANSFORMERS_RE.search(error_text)
        if not match:
            return None
        context_tokens, _, input_tokens, _ = (
            int(value) for value in match.groups()
        )
    raw_available = context_tokens - input_tokens
    if raw_available < CONTEXT_RETRY_MIN_TOKENS:
        return None
    if requested <= CONTEXT_RETRY_MIN_TOKENS:
        return None

    # Some vLLM releases report the prompt as exactly one token over the
    # request-dependent input limit. In that case, subtracting only the
    # reported overflow repeatedly uncovers one more prompt token and never
    # reaches the true prompt length. Always make a material reduction while
    # retaining the full prompt.
    safety_cap = max(
        CONTEXT_RETRY_MIN_TOKENS,
        raw_available - CONTEXT_RETRY_SAFETY_TOKENS,
    )
    reduction = max(CONTEXT_RETRY_MIN_REDUCTION, requested // 4)
    material_cap = max(
        CONTEXT_RETRY_MIN_TOKENS,
        requested - reduction,
    )
    retry_tokens = min(safety_cap, material_cap)
    if retry_tokens >= requested:
        return None
    return retry_tokens


class OpenAIClient(BaseModelClient):
    """
    OpenAI client with Tools API support.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str = None,
        base_url: str = None,
        max_tokens: int = 8192,
    ):
        super().__init__(model_name)
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_API_BASE"),
        )
        self.cost_tracker = CostTracker(model_name)
        self.max_tokens = max_tokens
    
    def chat_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Any] = None,
    ) -> ModelResponse:
        """
        Send messages to OpenAI with tools enabled.
        """
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        try:
            if _is_qwen_model(self.model_name):
                requested_tokens = max_tokens
                for attempt in range(CONTEXT_RETRY_MAX_ATTEMPTS):
                    try:
                        response = self._create_qwen_completion(
                            messages, tools, requested_tokens, tool_choice
                        )
                        break
                    except BadRequestError as error:
                        retry_max_tokens = _context_limited_max_tokens(
                            error, requested_tokens
                        )
                        if (
                            retry_max_tokens is None
                            or attempt + 1 >= CONTEXT_RETRY_MAX_ATTEMPTS
                        ):
                            raise
                        print(
                            "Context boundary retry: preserving the full "
                            "prompt and reducing max_tokens from "
                            f"{requested_tokens} to {retry_max_tokens} with "
                            "an adaptive reduction and a "
                            f"{CONTEXT_RETRY_SAFETY_TOKENS}-token safety "
                            f"margin (attempt {attempt + 2}/"
                            f"{CONTEXT_RETRY_MAX_ATTEMPTS})."
                        )
                        requested_tokens = retry_max_tokens
            elif self.model_name.startswith('gpt-5'):
                # gpt-5 系列: 用 max_completion_tokens, 不传 temperature
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools if tools else None,
                    tool_choice=tool_choice or ("auto" if tools else None),
                    max_completion_tokens=max_tokens
                )
            else:
                # gpt-4 / Claude / 其他: 用 temperature + max_tokens
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools if tools else None,
                    tool_choice=tool_choice or ("auto" if tools else None),
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            # Track usage
            if response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
                self.cost_tracker.add_usage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens
                )
            
            # Parse response
            message = response.choices[0].message
            
            # Extract tool calls if any
            tool_calls = None
            if message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                    )
                    for tc in message.tool_calls
                ]
            
            return ModelResponse(
                content=message.content,
                tool_calls=tool_calls,
                raw_response=response
            )
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise

    def _create_qwen_completion(
        self,
        messages: List[Dict],
        tools: List[Dict],
        max_tokens: int,
        tool_choice: Optional[Any] = None,
    ):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools if tools else None,
            tool_choice=tool_choice or ("auto" if tools else None),
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body=QWEN35_CHAT_EXTRA_BODY,
        )
    
    def format_tool_result(self, tool_call_id: str, result: str, name: str = None) -> Dict:
        """
        Format tool result as a message for OpenAI.
        Note: OpenAI doesn't need the name parameter, but we accept it for API consistency.
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        }
    
    def format_assistant_tool_calls(self, tool_calls: List[ToolCall]) -> Dict:
        """
        Format assistant's tool calls as a message.
        This is needed to maintain proper conversation history.
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
        """Get usage stats using CostTracker"""
        cost_info = self.cost_tracker.get_cost()
        return {
            'total_input_tokens': cost_info['input_tokens'],
            'total_output_tokens': cost_info['output_tokens'],
            'total_cost': cost_info['total_cost']
        }


# For testing
if __name__ == "__main__":
    client = OpenAIClient(model_name="gpt-4o-mini")
    
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
