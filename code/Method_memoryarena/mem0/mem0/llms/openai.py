import json
import os
import re
import threading

MEM0_DEBUG_LLM = os.getenv("MEM0_DEBUG_LLM", "0").lower() in {"1", "true", "yes", "on"}

def _llm_debug_print(*args, **kwargs):
    if MEM0_DEBUG_LLM:
        print(*args, **kwargs)
import warnings
from typing import Dict, List, Optional

from openai import OpenAI

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase
from mem0.memory.utils import extract_json

QWEN35_MAX_TOKENS = int(os.getenv("QWEN35_GRAPH_MAX_TOKENS", "2048"))
QWEN35_CHAT_EXTRA_BODY = {
    "top_k": 20,
    "chat_template_kwargs": {"enable_thinking": False},
}

_CONTEXT_FAILURE_EVENTS = []
_CONTEXT_FAILURE_LOCK = threading.Lock()


def _context_limit_details(exc):
    match = re.search(
        r"You passed (\d+) input tokens and requested (\d+) output tokens.*?"
        r"context length is only (\d+) tokens",
        str(exc),
    )
    if not match:
        return None
    input_tokens, requested_tokens, context_tokens = map(int, match.groups())
    return {
        "input_tokens": input_tokens,
        "requested_tokens": requested_tokens,
        "context_tokens": context_tokens,
        "error": str(exc),
    }


def _record_context_failure(exc):
    details = _context_limit_details(exc)
    if details is None:
        return False
    # mem0 may execute LLM calls in a worker thread while add_chunk consumes the
    # audit event in the request thread. The memory server is isolated to one
    # sequential task worker, so a locked process-local queue preserves the
    # event across that thread boundary without mixing formal workers.
    with _CONTEXT_FAILURE_LOCK:
        _CONTEXT_FAILURE_EVENTS.append(details)
    return True


def consume_context_failure_events():
    with _CONTEXT_FAILURE_LOCK:
        events = list(_CONTEXT_FAILURE_EVENTS)
        _CONTEXT_FAILURE_EVENTS.clear()
        return events


class OpenAILLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)
        self.token_tracker = None
        self._local_base = False
        self._context_budget_retry = False
        if not self.config.model:
            self.config.model = "gpt-4o-mini"
        # self.config.model  =  "/path/to/local/qwen2_5_lora_sft"
        if os.environ.get("OPENROUTER_API_KEY"):  # Use OpenRouter
            self.client = OpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url=self.config.openrouter_base_url
                or os.getenv("OPENROUTER_API_BASE")
                or "https://openrouter.ai/api/v1",
            )
        else:
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            base_url = (
                self.config.openai_base_url
                or os.getenv("OPENAI_API_BASE")
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.openai.com/v1"
            )
            if os.environ.get("OPENAI_API_BASE"):
                warnings.warn(
                    "The environment variable 'OPENAI_API_BASE' is deprecated and will be removed in the 0.1.80. "
                    "Please use 'OPENAI_BASE_URL' instead.",
                    DeprecationWarning,
                )

            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=1800.0)
            fallback_mode = os.getenv("MEM0_JSON_TOOL_FALLBACK", "auto").lower()
            local_base = any(marker in str(base_url) for marker in ("127.0.0.1", "localhost"))
            self._local_base = local_base
            retry_mode = os.getenv("MEM0_CONTEXT_BUDGET_RETRY", "auto").lower()
            self._context_budget_retry = retry_mode in {"1", "true", "yes", "on"} or (
                retry_mode == "auto" and local_base
            )
            self._json_tool_fallback = fallback_mode in {"1", "true", "yes", "on"} or (fallback_mode == "auto" and local_base)

    def _create_chat_completion_with_budget_retry(self, params):
        try:
            return self.client.chat.completions.create(**params)
        except Exception as exc:
            if not self._context_budget_retry:
                raise

            details = _context_limit_details(exc)
            if details is None:
                raise

            input_tokens = details["input_tokens"]
            requested_tokens = details["requested_tokens"]
            context_tokens = details["context_tokens"]
            safety_margin = max(0, int(os.getenv("MEM0_CONTEXT_SAFETY_MARGIN", "64")))
            retry_tokens = context_tokens - input_tokens - safety_margin
            if retry_tokens < 1 or retry_tokens >= requested_tokens:
                raise

            retry_params = dict(params)
            retry_params["max_tokens"] = retry_tokens
            print(
                "[mem0-context-budget-retry] "
                f"input_tokens={input_tokens} requested_tokens={requested_tokens} "
                f"context_tokens={context_tokens} safety_margin={safety_margin} "
                f"retry_max_tokens={retry_tokens}",
                flush=True,
            )
            return self.client.chat.completions.create(**retry_params)

    def set_token_tracker(self, tracker):
        self.token_tracker = tracker
    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(extract_json(tool_call.function.arguments)),
                        }
                    )
                    if os.getenv("MEM0_DEBUG_LLM") == "1":
                        _llm_debug_print(f"Tool call: {tool_call.function.name} with arguments: {tool_call.function.arguments}", flush=True)

            return processed_response
        else:
            return response.choices[0].message.content

    def _record_usage(self, usage):
        if self.token_tracker and usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)
            self.token_tracker.add_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
            if os.getenv("MEM0_DEBUG_LLM") == "1":
                print(f"[TRACKER] prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}", flush=True)

    def _normalize_json_tool_calls(self, content, tools):
        tool_defs = [tool.get("function", tool) for tool in (tools or [])]
        tool_name = tool_defs[0].get("name", "tool") if tool_defs else "tool"
        try:
            payload = json.loads(extract_json(content))
        except Exception:
            payload = {}

        raw_calls = payload.get("tool_calls") if isinstance(payload, dict) else None
        calls = []
        if isinstance(raw_calls, list):
            for item in raw_calls:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or tool_name
                args = item.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(extract_json(args))
                    except Exception:
                        args = {}
                if isinstance(args, dict):
                    calls.append({"name": name, "arguments": args})
        elif isinstance(payload, dict):
            if tool_name in payload:
                args = payload.get(tool_name)
            elif "arguments" in payload:
                args = payload.get("arguments")
            else:
                args = payload
            if isinstance(args, str):
                try:
                    args = json.loads(extract_json(args))
                except Exception:
                    args = {}
            if isinstance(args, list):
                if tool_name == "delete_graph_memory":
                    calls.extend({"name": tool_name, "arguments": item} for item in args if isinstance(item, dict))
                else:
                    calls.append({"name": tool_name, "arguments": {"entities": args}})
            elif isinstance(args, dict):
                if args:
                    calls.append({"name": tool_name, "arguments": args})

        return {"content": content, "tool_calls": calls}

    def _generate_json_tool_fallback(self, messages, tools):
        tool_defs = [tool.get("function", tool) for tool in (tools or [])]
        tool_names = [tool.get("name", "tool") for tool in tool_defs]
        instruction = (
            "You are replacing OpenAI tool calling for a local vLLM endpoint. "
            "Return only valid JSON, with no markdown or commentary. "
            "The JSON must have this shape: "
            "{\"tool_calls\":[{\"name\":<one allowed tool name>,\"arguments\":<object matching that tool schema>}]} . "
            "If no tool call is needed, return {\"tool_calls\":[]}. "
            f"Allowed tool definitions: {json.dumps(tool_defs, ensure_ascii=False)}. "
            f"Allowed tool names: {tool_names}."
        )
        system_parts = [instruction]
        non_system_messages = []
        for message in messages:
            if message.get("role") == "system":
                system_parts.append(str(message.get("content", "")))
            else:
                non_system_messages.append(message)
        fallback_messages = [{"role": "system", "content": "\n\n".join(system_parts)}] + non_system_messages
        params = {
            "model": self.config.model,
            "messages": fallback_messages,
            "max_tokens": QWEN35_MAX_TOKENS,
            "temperature": 0.1,
            "top_p": 0.8,
            "presence_penalty": 1.0,
            "extra_body": QWEN35_CHAT_EXTRA_BODY,
            "response_format": {"type": "json_object"},
        }
        try:
            response = self._create_chat_completion_with_budget_retry(params)
        except Exception:
            params.pop("response_format", None)
            response = self._create_chat_completion_with_budget_retry(params)
        usage = getattr(response, "usage", None)
        content = response.choices[0].message.content or "{}"
        parsed = self._normalize_json_tool_calls(content, tools)
        self._record_usage(usage)
        return parsed

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        """
        Generate a response based on the given messages using OpenAI.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str: The generated response.
        """
        if tools and getattr(self, "_json_tool_fallback", False):
            try:
                return self._generate_json_tool_fallback(messages, tools)
            except Exception as exc:
                _record_context_failure(exc)
                raise

        params = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": QWEN35_MAX_TOKENS,
            "temperature": 0.7,
            "top_p": 0.8,
            "presence_penalty": 1.5,
            "extra_body": QWEN35_CHAT_EXTRA_BODY,
        }

        # if os.getenv("OPENROUTER_API_KEY"):
        #     openrouter_params = {}
        #     if self.config.models:
        #         openrouter_params["models"] = self.config.models
        #         openrouter_params["route"] = self.config.route
        #         params.pop("model")

        #     if self.config.site_url and self.config.app_name:
        #         extra_headers = {
        #             "HTTP-Referer": self.config.site_url,
        #             "X-Title": self.config.app_name,
        #         }
        #         openrouter_params["extra_headers"] = extra_headers

        #     params.update(**openrouter_params)

        if response_format:
            params["response_format"] = response_format
        if tools:  # TODO: Remove tools if no issues found with new memory addition logic
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        try:
            response = self._create_chat_completion_with_budget_retry(params)
        except Exception as exc:
            _record_context_failure(exc)
            raise
        usage = getattr(response, "usage", None)

        if os.getenv("MEM0_DEBUG_LLM") == "1":
            _llm_debug_print("response here: ", response, flush=True)
            _llm_debug_print("Before _parse_response", flush=True)
        parsed = self._parse_response(response, tools)
        if os.getenv("MEM0_DEBUG_LLM") == "1":
            _llm_debug_print("After _parse_response", flush=True)
            _llm_debug_print("parsed: ", parsed, flush=True)

        self._record_usage(usage)

        return parsed
