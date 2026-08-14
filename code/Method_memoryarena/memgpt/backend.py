import hashlib
import os
import re
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import requests
from letta_client import Letta
from letta_client.types import EmbeddingConfig, LlmConfig


def _safe_name(value: Optional[str]) -> str:
    raw = str(value or "default")
    text = re.sub(r"[^A-Za-z0-9 _-]+", "_", raw)
    if text != raw:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        text = f"{text[:96]}_{digest}"
    return text[:108] or "default"


@lru_cache(maxsize=8)
def _load_embedding_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
        use_fast=True,
    )


class LettaMemorySystem:
    """Local Letta adapter for the official MemoryArena memory API."""

    def __init__(self, user_id: Optional[str] = None, top_k: int = 10):
        self.user_id = user_id or "default"
        self.base_url = os.getenv("LETTA_BASE_URL", "http://127.0.0.1:8283").rstrip("/")
        self.top_k = int(os.getenv("MEMORYARENA_RETRIEVE_K", str(top_k)))
        self.max_context_chars = int(
            os.getenv("LETTA_MEMORYARENA_MAX_CONTEXT_CHARS", "36000")
        )
        self.max_search_query_chars = int(
            os.getenv("LETTA_MEMORYARENA_SEARCH_QUERY_CHARS", "256")
        )
        self.direct_chunk_chars = int(
            os.getenv("LETTA_MEMORYARENA_DIRECT_CHUNK_CHARS", "320")
        )
        self.embedding_token_limit = int(
            os.getenv("LETTA_MEMORYARENA_EMBEDDING_TOKEN_LIMIT", "240")
        )
        self.max_steps = int(os.getenv("LETTA_MEMORYARENA_MAX_STEPS", "6"))
        self.agent_mode = os.getenv("LETTA_MEMORYARENA_AGENT_MODE", "0") == "1"
        self.client = Letta(base_url=self.base_url)

        llm_model = os.getenv("LETTA_LLM_MODEL", "Qwen/Qwen3.5-9B")
        llm_endpoint = os.getenv("LETTA_LLM_BASE_URL", "http://127.0.0.1:8007/v1")
        embedding_model = os.getenv(
            "LETTA_EMBEDDING_MODEL",
            "/path/to/local/all-MiniLM-L6-v2",
        )
        self.embedding_tokenizer_name = os.getenv(
            "LETTA_EMBEDDING_TOKENIZER",
            embedding_model,
        )
        embedding_endpoint = os.getenv("LETTA_EMBEDDING_BASE_URL", "").strip()
        if not embedding_endpoint:
            raise RuntimeError(
                "LETTA_EMBEDDING_BASE_URL is required for the local MemGPT backend"
            )

        llm_config = LlmConfig(
            model=llm_model,
            model_endpoint=llm_endpoint,
            model_endpoint_type="openai",
            context_window=int(os.getenv("LETTA_CONTEXT_WINDOW", "32768")),
            temperature=0.7,
            max_tokens=int(os.getenv("LETTA_MAX_TOKENS", "4096")),
        )
        embedding_config = EmbeddingConfig(
            embedding_model=embedding_model,
            embedding_endpoint=embedding_endpoint,
            embedding_endpoint_type="hugging-face",
            embedding_dim=int(os.getenv("LETTA_EMBEDDING_DIM", "384")),
            embedding_chunk_size=int(
                os.getenv("LETTA_EMBEDDING_CHUNK_SIZE", "200")
            ),
        )
        self.agent_state = self._create_agent(llm_config, embedding_config)

    @staticmethod
    def _model_payload(model: Any) -> Dict[str, Any]:
        if hasattr(model, "model_dump"):
            return model.model_dump(exclude_none=True)
        if hasattr(model, "dict"):
            return model.dict(exclude_none=True)
        raise TypeError(f"Unsupported Letta configuration type: {type(model)!r}")

    def _create_agent(
        self,
        llm_config: LlmConfig,
        embedding_config: EmbeddingConfig,
    ) -> SimpleNamespace:
        # letta-client 0.1.319 hard-codes agent creation to localhost:8283.
        # Use the same API payload against the configured isolated Letta service.
        response = requests.post(
            f"{self.base_url}/v1/agents",
            json={
                "name": f"memoryarena_{_safe_name(self.user_id)}",
                "llm_config": self._model_payload(llm_config),
                "embedding_config": self._model_payload(embedding_config),
                "memory_blocks": [
                    {"label": "human", "value": ""},
                    {
                        "label": "persona",
                        "value": (
                            "You are the memory component of a MemoryArena agent. "
                            "Store durable task evidence and retrieve only relevant evidence."
                        ),
                    },
                ],
                "tools": [],
                "include_base_tools": True,
            },
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        agent_id = payload.get("id") if isinstance(payload, dict) else None
        if not agent_id:
            raise RuntimeError("Letta agent creation response is missing an id")
        return SimpleNamespace(id=str(agent_id))

    def add_chunk(self, chunk: str) -> Dict[str, Any]:
        text = (chunk or "").strip()
        if not text:
            return {"stored": False, "reason": "empty_chunk"}

        if not self.agent_mode:
            passages = self._direct_archival_add(text)
            return {
                "stored": True,
                "agent_id": self.agent_state.id,
                "storage": "letta_archival_memory",
                "passage_count": len(passages),
            }

        prompt = (
            "Store the following task evidence in archival memory. "
            "Use archival_memory_insert and do not summarize away names, identifiers, "
            "constraints, actions, observations, or rewards.\n\n"
            f"{text}"
        )
        response = self.client.agents.messages.create(
            agent_id=self.agent_state.id,
            messages=[{"role": "user", "content": prompt}],
            max_steps=self.max_steps,
            include_return_message_types=[
                "assistant_message",
                "tool_call_message",
                "tool_return_message",
            ],
        )
        parsed = self._parse_messages(response)
        archival_success = any(
            item.get("type") == "tool_return"
            and item.get("name") == "archival_memory_insert"
            and item.get("status") != "error"
            for item in parsed
        )
        if not archival_success:
            self._direct_archival_add(text)
        return {
            "stored": True,
            "agent_id": self.agent_state.id,
            "agent_messages": parsed,
            "direct_archival_fallback": not archival_success,
        }

    def _lossless_embedding_chunks(self, text: str) -> List[str]:
        tokenizer = _load_embedding_tokenizer(self.embedding_tokenizer_name)

        def split_for_token_limit(chunk: str) -> List[str]:
            token_count = len(
                tokenizer.encode(
                    chunk,
                    add_special_tokens=True,
                    truncation=False,
                )
            )
            if token_count <= self.embedding_token_limit:
                return [chunk]
            if len(chunk) <= 1:
                raise RuntimeError(
                    "Unable to split Letta passage below the embedding token limit"
                )
            midpoint = len(chunk) // 2
            return split_for_token_limit(chunk[:midpoint]) + split_for_token_limit(
                chunk[midpoint:]
            )

        char_chunks = (
            [text]
            if len(text) <= self.direct_chunk_chars
            else [
                text[offset : offset + self.direct_chunk_chars]
                for offset in range(0, len(text), self.direct_chunk_chars)
            ]
        )
        chunks = [
            token_chunk
            for char_chunk in char_chunks
            for token_chunk in split_for_token_limit(char_chunk)
        ]
        if "".join(chunks) != text:
            raise RuntimeError("Lossless Letta chunking invariant failed")
        return chunks

    def _direct_archival_add(self, text: str) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/v1/agents/{self.agent_state.id}/archival-memory"
        # Letta 0.12 can ignore the configured embedding chunk size for long
        # passages. Preserve the existing fixed-width passage boundaries unless
        # the actual embedding tokenizer says a passage exceeds its token limit.
        chunks = self._lossless_embedding_chunks(text)
        passages: List[Dict[str, Any]] = []
        for chunk in chunks:
            chunk_response = requests.post(
                url,
                json={"text": chunk},
                timeout=300,
            )
            chunk_response.raise_for_status()
            payload = chunk_response.json()
            passages.extend(payload if isinstance(payload, list) else [payload])
        return passages

    def wrap_user_prompt(self, prompt: str) -> str:
        context = self._direct_archival_search(prompt)
        if context or not self.agent_mode:
            return self._wrapped_prompt(prompt, context)

        response = self.client.agents.messages.create(
            agent_id=self.agent_state.id,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Retrieve the most relevant evidence from archival and core memory "
                        "for the following MemoryArena prompt. Return the evidence as plain "
                        "text without answering the task itself.\n\n"
                        f"{prompt}"
                    ),
                }
            ],
            max_steps=self.max_steps,
            include_return_message_types=[
                "assistant_message",
                "tool_call_message",
                "tool_return_message",
            ],
        )
        parsed = self._parse_messages(response)
        context = [
            str(item["content"]).strip()
            for item in parsed
            if item.get("type") == "text" and str(item.get("content") or "").strip()
        ]
        if not context:
            context = self._direct_archival_search(prompt)

        return self._wrapped_prompt(prompt, context)

    @staticmethod
    def _wrapped_prompt(prompt: str, context: List[str]) -> str:
        lines = ["<memory_context>"]
        lines.extend(context or ["None"])
        lines.append("</memory_context>")
        lines.append(f"User Prompt: {prompt}")
        return "\n".join(lines)

    def _retrieval_query(self, query: str) -> str:
        text = str(query or "").strip()
        for marker in ("### PROBLEM:", "User Prompt:"):
            if marker in text:
                text = text.rsplit(marker, 1)[-1]
        text = " ".join(text.split())
        if len(text) > self.max_search_query_chars:
            text = text[-self.max_search_query_chars:]
        return text or "MemoryArena task"

    def _direct_archival_search(self, query: str) -> List[str]:
        search_query = self._retrieval_query(query)
        response = requests.get(
            f"{self.base_url}/v1/agents/{self.agent_state.id}/archival-memory",
            params={"search": search_query, "limit": self.top_k},
            timeout=300,
        )
        if not response.ok and len(search_query) > 128:
            response = requests.get(
                f"{self.base_url}/v1/agents/{self.agent_state.id}/archival-memory",
                params={"search": search_query[-128:], "limit": self.top_k},
                timeout=300,
            )
        response.raise_for_status()
        results = []
        used_chars = 0
        for passage in response.json():
            text = passage.get("text") or passage.get("content")
            if not text or used_chars >= self.max_context_chars:
                continue
            remaining = self.max_context_chars - used_chars
            clipped = str(text)[:remaining]
            if clipped:
                results.append(clipped)
                used_chars += len(clipped)
        return results

    @staticmethod
    def _parse_messages(response: Any) -> List[Dict[str, Any]]:
        parsed: List[Dict[str, Any]] = []
        for message in getattr(response, "messages", None) or []:
            message_type = getattr(message, "message_type", None)
            if message_type == "assistant_message":
                parsed.append(
                    {"type": "text", "content": getattr(message, "content", "")}
                )
            elif message_type == "tool_call_message":
                tool_call = getattr(message, "tool_call", None)
                if tool_call is not None:
                    parsed.append(
                        {
                            "type": "tool_call",
                            "name": getattr(tool_call, "name", None),
                            "arguments": getattr(tool_call, "arguments", None),
                        }
                    )
            elif message_type == "tool_return_message":
                parsed.append(
                    {
                        "type": "tool_return",
                        "name": getattr(message, "name", None),
                        "status": getattr(message, "status", None),
                        "content": getattr(message, "tool_return", None),
                    }
                )
        return parsed

    def close(self, completed: bool = False) -> dict:
        close = getattr(self.client, "close", None)
        resources = []
        if callable(close):
            close()
            resources.append("letta.client")
        return {
            "closed": True,
            "completed": bool(completed),
            "state_preserved": True,
            "resources_closed": resources,
            "agent_id": str(self.agent_state.id),
        }


def create_backend(*, user_id=None, top_k=10):
    """Create MemGPT/Letta through the common MemoryArena factory."""

    return LettaMemorySystem(user_id=user_id, top_k=top_k)
