from __future__ import annotations

import os
import re
import time
from typing import List, Optional, Tuple

from openai import OpenAI

SUMMARY_PROMPT = """
Below is an user-user dialogue memory. Please summarize the following dialogue as concisely as possible in a short paragraph, extracting the main themes and key information.

{session_text}

Your answer:
""".strip()

KEYWORD_PROMPT = """
Below is an user-user dialogue memory. Please extract the most relevant keywords, separated by semicolon.

{session_text}

Your answer:
""".strip()


class OpenAICompatibleLLM:
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str],
        model: str,
        max_tokens: int = 500,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_wait_sec: float = 2.0,
        context_window: int = 20000,
        prompt_token_buffer: int = 128,
        use_qwen_thinking_control: bool = True,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_wait_sec = retry_wait_sec
        self.context_window = int(os.getenv("MEMGAS_LLM_CONTEXT_WINDOW", context_window or 20000))
        self.prompt_token_buffer = int(os.getenv("MEMGAS_PROMPT_TOKEN_BUFFER", prompt_token_buffer or 128))
        self.use_qwen_thinking_control = bool(use_qwen_thinking_control)
        self._tokenizer = self._load_tokenizer(model)

    def _complete(self, prompt: str) -> str:
        prompt = self._truncate_prompt(prompt)
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                request = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if self.use_qwen_thinking_control:
                    request["extra_body"] = {
                        "chat_template_kwargs": {
                            "enable_thinking": False,
                        },
                    }
                response = self.client.chat.completions.create(**request)
                content = response.choices[0].message.content
                return (content or "").strip()
            except Exception as err:  # pragma: no cover - network/runtime dependent
                last_err = err
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait_sec)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")


    def _available_prompt_tokens(self) -> int:
        return max(256, self.context_window - self.max_tokens - self.prompt_token_buffer)

    @staticmethod
    def _load_tokenizer(model: str):
        try:
            from transformers import AutoTokenizer
        except Exception:
            return None

        candidates = [model]
        if model and not os.path.isabs(model):
            candidates.append(os.path.join("/path/to/local", model))

        for candidate in candidates:
            try:
                return AutoTokenizer.from_pretrained(
                    candidate,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except Exception:
                continue
        return None

    def _truncate_prompt(self, prompt: str) -> str:
        max_prompt_tokens = self._available_prompt_tokens()
        if max_prompt_tokens <= 0:
            return prompt

        if self._tokenizer is not None:
            token_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
            if len(token_ids) <= max_prompt_tokens:
                return prompt
            head_count = min(max(128, max_prompt_tokens // 4), max_prompt_tokens // 2)
            tail_count = max_prompt_tokens - head_count
            kept_ids = token_ids[:head_count] + token_ids[-tail_count:]
            return self._tokenizer.decode(kept_ids, skip_special_tokens=True)

        approx_tokens = max(1, len(prompt) // 4)
        if approx_tokens <= max_prompt_tokens:
            return prompt
        max_chars = max_prompt_tokens * 4
        head_chars = min(max(512, max_chars // 4), max_chars // 2)
        tail_chars = max_chars - head_chars
        return prompt[:head_chars] + "\n...[truncated for context window]...\n" + prompt[-tail_chars:]

    def summarize_and_keywords(self, session_text: str) -> Tuple[str, List[str]]:
        summary = self._complete(SUMMARY_PROMPT.format(session_text=session_text))
        raw_keywords = self._complete(KEYWORD_PROMPT.format(session_text=session_text))
        keywords = self._parse_keywords(raw_keywords)
        return summary, keywords

    @staticmethod
    def _parse_keywords(raw_keywords: str) -> List[str]:
        # Support semicolon/comma/newline outputs from different models.
        chunks = re.split(r"[;,\n，；]+", raw_keywords)
        keywords: List[str] = []
        seen = set()
        for chunk in chunks:
            token = chunk.strip()
            if not token:
                continue
            if token in seen:
                continue
            seen.add(token)
            keywords.append(token)
        return keywords
