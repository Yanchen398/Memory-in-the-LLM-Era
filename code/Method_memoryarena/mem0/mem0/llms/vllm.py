# import json
# import os

MEM0_DEBUG_LLM = os.getenv("MEM0_DEBUG_LLM", "0").lower() in {"1", "true", "yes", "on"}

def _llm_debug_print(*args, **kwargs):
    if MEM0_DEBUG_LLM:
        print(*args, **kwargs)
# from typing import Dict, List, Optional

# from mem0.configs.llms.base import BaseLlmConfig
# from mem0.llms.base import LLMBase
# from mem0.memory.utils import extract_json

# from openai import OpenAI
# class VllmLLM(LLMBase):
#     def __init__(self, config: Optional[BaseLlmConfig] = None):
#         super().__init__(config)

#         if not self.config.model:
#             self.config.model = "Qwen/Qwen2.5-32B-Instruct"

#         self.config.api_key = self.config.api_key or os.getenv("VLLM_API_KEY") or "vllm-api-key"
#         base_url = self.config.vllm_base_url or os.getenv("VLLM_BASE_URL")

#         self.client = OpenAI(base_url=base_url, api_key=self.config.api_key)

#     def _parse_response(self, response, tools):
#         """
#         Process the response based on whether tools are used or not.

#         Args:
#             response: The raw response from API.
#             tools: The list of tools provided in the request.

#         Returns:
#             str or dict: The processed response.
#         """
#         if tools:
#             processed_response = {
#                 "content": response.choices[0].message.content,
#                 "tool_calls": [],
#             }

#             if response.choices[0].message.tool_calls:
#                 for tool_call in response.choices[0].message.tool_calls:
#                     processed_response["tool_calls"].append({
#                         "name": tool_call.function.name,
#                         "arguments": json.loads(extract_json(tool_call.function.arguments)),
#                     })

#             return processed_response
#         else:
#             return response.choices[0].message.content

#     def generate_response(
#         self,
#         messages: List[Dict[str, str]],
#         response_format=None,
#         tools: Optional[List[Dict]] = None,
#         tool_choice: str = "auto",
#     ):
#         """
#         Generate a response based on the given messages using vLLM.

#         Args:
#             messages (list): List of message dicts containing 'role' and 'content'.
#             response_format (str or object, optional): Format of the response. Defaults to "text".
#             tools (list, optional): List of tools that the model can call. Defaults to None.
#             tool_choice (str, optional): Tool choice method. Defaults to "auto".

#         Returns:
#             str: The generated response.
#         """
#         params = {
#             "model": self.config.model,
#             "messages": messages,
#             "temperature": self.config.temperature,
#             "max_tokens": self.config.max_tokens,
#             "top_p": self.config.top_p,
#         }

#         if response_format:
#             params["response_format"] = response_format

#         if tools:
#             params["tools"] = tools
#             params["tool_choice"] = tool_choice

#         response = self.client.chat.completions.create(**params)
#         # return self._parse_response(response, tools)

#         # 主动统计 token usage（自动嵌入当前阶段）
#         usage = getattr(response, "usage", None)
#         if self.token_tracker and usage:
#             prompt_tokens = getattr(usage, "prompt_tokens", 0)
#             completion_tokens = getattr(usage, "completion_tokens", 0)
#             total_tokens = getattr(usage, "total_tokens", 0)
#             self.token_tracker.add_usage(prompt_tokens, completion_tokens, total_tokens)

#         return self._parse_response(response, tools)


#         # response = self.client.chat.completions.create(**params)
#         # content = self._parse_response(response, tools)
#         # # 提取 total_tokens
#         # total_tokens = None
#         # if hasattr(response, "usage") and response.usage is not None:
#         #     total_tokens = getattr(response.usage, "total_tokens", None)
#         # elif hasattr(response, "to_dict"):
#         #     # 某些openai兼容实现会有 to_dict
#         #     usage = response.to_dict().get("usage", {})
#         #     total_tokens = usage.get("total_tokens", None)

#         # return {
#         # "content": content,
#         # "total_tokens": total_tokens
#         # }


import json
import os

MEM0_DEBUG_LLM = os.getenv("MEM0_DEBUG_LLM", "0").lower() in {"1", "true", "yes", "on"}

def _llm_debug_print(*args, **kwargs):
    if MEM0_DEBUG_LLM:
        print(*args, **kwargs)
from typing import Dict, List, Optional

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase
from mem0.memory.utils import extract_json

from openai import OpenAI

QWEN35_MODEL = os.getenv("QWEN35_MODEL", "Qwen/Qwen3.5-9B")
QWEN35_MAX_TOKENS = int(os.getenv("QWEN35_MAX_TOKENS", "32768"))
QWEN35_CHAT_EXTRA_BODY = {
    "top_k": 20,
    "chat_template_kwargs": {"enable_thinking": False},
}

class VllmLLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None, token_tracker=None):
        super().__init__(config)
        if not self.config.model:
            self.config.model = QWEN35_MODEL
        self.config.api_key = self.config.api_key or os.getenv("VLLM_API_KEY") or "vllm-api-key"
        base_url = self.config.vllm_base_url or os.getenv("VLLM_BASE_URL")
        self.client = OpenAI(base_url=base_url, api_key=self.config.api_key)
        self.token_tracker = token_tracker  # 新增，外部传入

    def _parse_response(self, response, tools):
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append({
                        "name": tool_call.function.name,
                        "arguments": json.loads(extract_json(tool_call.function.arguments)),
                    })
            return processed_response
        else:
            return response.choices[0].message.content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        params = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": QWEN35_MAX_TOKENS,
            "temperature": 0.7,
            "top_p": 0.8,
            "presence_penalty": 1.5,
            "extra_body": QWEN35_CHAT_EXTRA_BODY,
        }
        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**params)
        usage = getattr(response, "usage", None)
        if self.token_tracker and usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)
            self.token_tracker.add_usage(prompt_tokens, completion_tokens, total_tokens)
        if os.getenv("MEM0_DEBUG_LLM") == "1":
            _llm_debug_print("usage in generate_response:", usage, flush=True)
        return self._parse_response(response, tools)
