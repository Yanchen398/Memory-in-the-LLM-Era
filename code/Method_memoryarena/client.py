import os
import hashlib
import json
import time
import uuid
from typing import Optional

import requests
from requests.exceptions import HTTPError, RequestException


def configuration_digest(payload: dict) -> str:
    """Return a stable digest for initialization ownership checks."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class MemoryClient:
    """Thin client for the unified MemoryArena memory API."""

    def __init__(
        self,
        user_id: str,
        memory_system_name: str = "mem0",
        base_url: str = "http://127.0.0.1:8000",
        session: Optional[requests.Session] = None,
        timeout: int = 300,
        run_id: Optional[str] = None,
        config_digest: Optional[str] = None,
    ):
        self.user_id = str(user_id)
        self.memory_system_name = memory_system_name
        self.base_url = base_url.rstrip("/")
        self.run_id = run_id
        self.config_digest = config_digest
        self._owns_session = session is None
        self.session = session or requests.Session()
        configured_timeout = os.getenv("MEMORYARENA_MEMORY_CALL_TIMEOUT_SECONDS", "").strip()
        self.timeout = float(configured_timeout) if configured_timeout else timeout
        self._initialize()

    def _initialize(self) -> None:
        payload = {
            "user_id": self.user_id,
            "memory_system_name": self.memory_system_name,
        }
        if self.run_id:
            payload["run_id"] = self.run_id
        if self.config_digest:
            payload["config_digest"] = self.config_digest
        self._post(
            "/memory/initialize",
            payload,
            allow_recover=False,
        )

    def wrap_user_prompt(self, question: str) -> str:
        """Request a prompt wrapped with memory context."""
        data = self._post(
            "/memory/wrap_user_prompt",
            {
                "user_id": self.user_id,
                "memory_system_name": self.memory_system_name,
                "question": question,
            },
        )
        return data["prompt"]

    def add(
        self,
        chunk: str,
        *,
        op_id: Optional[str] = None,
        seq: Optional[int] = None,
        phase: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Add a chunk to the user's memory."""
        payload = {
            "user_id": self.user_id,
            "memory_system_name": self.memory_system_name,
            "chunk": chunk,
            "op_id": op_id or uuid.uuid4().hex,
        }
        if seq is not None:
            payload["seq"] = int(seq)
        if phase is not None:
            payload["phase"] = str(phase)
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        return self._post(
            "/memory/add",
            payload,
        )

    def close(self, completed: bool = False) -> dict:
        """Release this sample's server-side memory instance."""
        try:
            return self._post(
                "/memory/close",
                {
                    "user_id": self.user_id,
                    "memory_system_name": self.memory_system_name,
                    "completed": bool(completed),
                },
                allow_recover=False,
            )
        finally:
            if self._owns_session:
                self.session.close()

    def _post(self, path: str, payload: dict, allow_recover: bool = True) -> dict:
        attempts = int(os.getenv("MEMORYARENA_MEMORY_CALL_RETRIES", "3"))
        delay = float(os.getenv("MEMORYARENA_MEMORY_CALL_RETRY_DELAY", "2"))
        if attempts <= 0:
            raise ValueError("MEMORYARENA_MEMORY_CALL_RETRIES must be positive")
        if delay < 0:
            raise ValueError("MEMORYARENA_MEMORY_CALL_RETRY_DELAY cannot be negative")
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}{path}", json=payload, timeout=self.timeout
                )
                recover_enabled = os.getenv(
                    "MEMORYARENA_ALLOW_REINITIALIZE_ON_404", "0"
                ).strip().lower() in {"1", "true", "yes", "on"}
                if (
                    allow_recover
                    and recover_enabled
                    and response.status_code == 404
                    and path != "/memory/initialize"
                ):
                    print(
                        f"Memory user {self.user_id} missing on server; reinitializing and retrying {path}",
                        flush=True,
                    )
                    self._initialize()
                    response = self.session.post(
                        f"{self.base_url}{path}", json=payload, timeout=self.timeout
                    )
                response.raise_for_status()
                return response.json()
            except RequestException as exc:
                last_exc = exc
                status = (
                    exc.response.status_code
                    if isinstance(exc, HTTPError) and exc.response is not None
                    else None
                )
                retryable_status = status in {408, 425, 429} or (
                    status is not None and status >= 500
                )
                if status is not None and not retryable_status:
                    raise
                if attempt >= attempts:
                    break
                print(
                    f"Memory request {path} failed on attempt {attempt}/{attempts}: {exc}; retrying in {delay}s",
                    flush=True,
                )
                time.sleep(delay)
        raise last_exc
