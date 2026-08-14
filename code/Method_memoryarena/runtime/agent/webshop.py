from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base_agent import BaseAgent


DEFAULT_SYSTEM_PROMPT = (
    "You are a WebShop shopping agent. "
    "Reply with exactly one action in the format search[...] or click[...]. "
    "Do not add explanations, JSON, or multiple actions."
)

ACTION_PATTERN = re.compile(r"(search\[[^\]]*\]|click\[[^\]]*\])", re.IGNORECASE | re.DOTALL)
QWEN35_CHAT_EXTRA_BODY = {
    "top_k": 20,
    "chat_template_kwargs": {"enable_thinking": False},
}


class LLMFatalError(RuntimeError):
    def __init__(self, message: str, original_exception: Exception | None = None) -> None:
        super().__init__(message)
        self.original_exception = original_exception


class LLMRetryableError(RuntimeError):
    def __init__(self, message: str, original_exception: Exception | None = None) -> None:
        super().__init__(message)
        self.original_exception = original_exception


def _is_billing_error(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code.lower() in {"insufficient_quota", "billing_hard_limit_reached"}:
        return True
    message = str(exc).lower()
    if "insufficient_quota" in message:
        return True
    if "exceeded your current quota" in message:
        return True
    if "no active subscription" in message:
        return True
    if "payment required" in message:
        return True
    if "billing" in message and "rate limit" not in message:
        return True
    return False


def _is_qwen_model(model_name: str) -> bool:
    return "qwen" in (model_name or "").lower()


def _compact_qwen_webshop_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    compacted: List[Dict[str, Any]] = []
    removed_duplicates = 0
    for message in messages:
        copied = dict(message)
        if (
            compacted
            and (copied.get("role") or "").lower() == "user"
            and (compacted[-1].get("role") or "").lower() == "user"
            and copied.get("content") == compacted[-1].get("content")
        ):
            removed_duplicates += 1
            continue
        compacted.append(copied)

    marker = "Instruction: [SEP]"
    user_payloads = []
    for index, message in enumerate(compacted):
        if (message.get("role") or "").lower() != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        marker_index = content.find(marker)
        if marker_index < 0:
            continue
        user_payloads.append(
            (index, content[marker_index + len(marker):].lstrip())
        )

    trimmed_instructions = 0
    if len(user_payloads) >= 2:
        common_prefix = os.path.commonprefix(
            [user_payloads[0][1], user_payloads[1][1]]
        )
        separator_index = common_prefix.rfind("[SEP]")
        if separator_index >= 0 and separator_index >= 512:
            repeated_prefix = user_payloads[1][1][: separator_index + 5]
            for index, payload in user_payloads[1:]:
                if not payload.startswith(repeated_prefix):
                    continue
                observation = payload[len(repeated_prefix):].strip()
                compacted[index]["content"] = (
                    "Current WebShop observation: " + observation
                )
                trimmed_instructions += 1

    if removed_duplicates or trimmed_instructions:
        print(
            "WebShop message compaction: "
            f"removed_duplicates={removed_duplicates}, "
            f"trimmed_repeated_instructions={trimmed_instructions}."
        )
    return compacted


class WebShopAgent(BaseAgent):
    """OpenAI-compatible agent for WebShop action generation."""

    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        temperature: float = 0.0,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        super().__init__(model_name, temperature)
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.turns: List[Dict[str, Any]] = []
        self._last_raw_output: Optional[str] = None
        self._last_reasoning_content: Optional[str] = None

    def reset(self) -> None:
        self.turns = []
        self._last_raw_output = None
        self._last_reasoning_content = None

    def record_turn(
        self,
        *,
        turn_idx: int,
        prompt: str,
        action: str,
        observation: Dict[str, Any],
        reward: Any,
        done: bool,
        info: Dict[str, Any],
        raw_output: Optional[str] = None,
    ) -> None:
        self.turns.append(
            {
                "turn_idx": turn_idx,
                "prompt": prompt,
                "action": action,
                "reward": reward,
                "done": done,
                "info": info,
                "observation": observation,
                "raw_output": raw_output or self._last_raw_output,
            }
        )

    def _use_completion_tokens_param(self) -> bool:
        model_l = self.model_name.lower()
        return any(token in model_l for token in ("gpt-5", "gpt-4.1", "o4", "o3"))

    def _normalize_action(self, raw_text: str) -> str:
        text = (raw_text or "").strip()
        match = ACTION_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return text

    def _create_completion(self, messages: List[Dict[str, str]]) -> str:
        payload = list(messages)
        if not any((message.get("role") or "").lower() == "system" for message in payload):
            payload = [{"role": "system", "content": self.system_prompt}] + payload
        if _is_qwen_model(self.model_name):
            payload = _compact_qwen_webshop_messages(payload)

        max_retries_raw = os.getenv("OPENAI_MAX_RETRIES", "3")
        try:
            max_retries = max(1, int(max_retries_raw))
        except ValueError:
            max_retries = 3

        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": payload,
        }
        if _is_qwen_model(self.model_name):
            params.update(
                {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "presence_penalty": 1.5,
                    "extra_body": QWEN35_CHAT_EXTRA_BODY,
                }
            )
            if self.max_tokens is not None:
                # WebShop requires exactly one short action. Preserve the full
                # prompt and memory context while avoiding an unnecessary
                # multi-thousand-token output reservation at the 32K boundary.
                action_token_cap = max(
                    32,
                    int(os.getenv("MEMORYARENA_WEBSHOP_ACTION_MAX_TOKENS", "32")),
                )
                params["max_tokens"] = min(self.max_tokens, action_token_cap)
        else:
            params["temperature"] = self.temperature
            if self.max_tokens is not None:
                key = "max_completion_tokens" if self._use_completion_tokens_param() else "max_tokens"
                params[key] = self.max_tokens

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(**params)
                message = response.choices[0].message
                raw_text = message.content or ""
                self._last_raw_output = raw_text
                self._last_reasoning_content = getattr(message, "reasoning_content", None)
                return raw_text
            except Exception as exc:
                if _is_billing_error(exc):
                    raise LLMFatalError(
                        f"Fatal LLM error (billing/quota): {exc}",
                        original_exception=exc,
                    ) from exc
                if attempt >= max_retries:
                    raise LLMRetryableError(
                        f"LLM error after {max_retries} attempts: {exc}",
                        original_exception=exc,
                    ) from exc
                time.sleep(1 + attempt)
        raise LLMRetryableError("LLM generation failed without a response.")

    def act_with_messages(self, messages: List[Dict[str, str]]) -> str:
        raw_text = self._create_completion(messages)
        return self._normalize_action(raw_text)

    def act(self, prompt: str) -> str:
        return self.act_with_messages(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        )

    def build_memory_entry(
        self,
        task: str,
        action: str,
        observation: Dict[str, Any],
        reward: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        history_specified = "history" in kwargs
        history_value = kwargs.get("history")
        payload: Dict[str, Any] = {
            "task": task,
            "final_action": action,
            "reward": reward,
            "final_observation": observation,
        }
        for key in (
            "turn_idx",
            "prompt",
            "prompt_source",
            "done",
            "info",
            "history",
            "raw_output",
            "memory_mode",
        ):
            value = kwargs.get(key)
            if value is not None:
                payload[key] = value

        if history_specified and history_value is not None:
            payload["turns"] = history_value
        elif not history_specified:
            payload["turns"] = self.turns
        if "raw_output" not in payload and self._last_raw_output:
            payload["last_raw_output"] = self._last_raw_output
        return json.dumps(payload, ensure_ascii=False)
