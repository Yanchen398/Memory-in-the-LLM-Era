from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, TypedDict


ConversationMessage = TypedDict(
    "ConversationMessage",
    {"from": str, "loss": Optional[bool], "value": str},
)


class ActionFormat(Enum):
    REACT = "react"
    FUNCTION_CALLING = "function_calling"
    CODE_AS_ACTION = "code_as_action"


@dataclass
class StepOutput:
    state: str
    reward: float
    done: bool


@dataclass
class ActionWithTought:
    thought: str
    action: str


@dataclass
class ExperienceOutput:
    conversation: list[ConversationMessage]
    reward: float
    text: str
    seq_ids: Sequence[int]
    attention_mask: Sequence[int]
    action_mask: Sequence[int]
