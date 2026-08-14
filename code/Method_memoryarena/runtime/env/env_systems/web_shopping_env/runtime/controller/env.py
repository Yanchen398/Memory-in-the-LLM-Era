from __future__ import annotations

from abc import ABCMeta, abstractmethod

from .types import ActionFormat, StepOutput


class BaseEnvClient(metaclass=ABCMeta):
    def __init__(self, action_format: ActionFormat | str = "react") -> None:
        self.action_format = ActionFormat(action_format)

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def observe(self) -> str:
        pass

    @abstractmethod
    def step(self, action: str) -> StepOutput:
        pass

    @abstractmethod
    def reset(self, idx: int):
        pass
