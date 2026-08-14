from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from speaking_agent.campaign import Campaign
from speaking_agent.domain import ConversationState


class ConversationModelError(RuntimeError):
    """A recoverable failure while interpreting a conversation turn."""


@dataclass(frozen=True)
class ModelInterpretation:
    suggested_outcome: str | None = None
    field_updates: Mapping[str, Any] = field(default_factory=dict)
    answer: str | None = None
    acknowledgement: str | None = None
    callback_requested: bool | None = None
    human_transfer_requested: bool = False


class ConversationModel(Protocol):
    async def prepare(self) -> None: ...

    async def interpret(
        self,
        utterance: str,
        state: ConversationState,
        campaign: Campaign,
    ) -> ModelInterpretation: ...

    async def close(self) -> None: ...
