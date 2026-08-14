from __future__ import annotations

from typing import Any, Callable, Mapping


class BaseTask:
    env_client_cls: Callable[..., Any]
    env_name: str

    def __init__(self, client_args: Mapping[str, Any], n_clients: int = 1) -> None:
        if self.env_client_cls is None or self.env_name is None:
            raise NotImplementedError
        self.clients = [self.env_client_cls(**client_args) for _ in range(n_clients)]
        self.len = len(self.clients[0])
