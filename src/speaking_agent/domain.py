from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ConversationStage(StrEnum):
    OPENING = "OPENING"
    DISCOVERY = "DISCOVERY"
    QUALIFICATION = "QUALIFICATION"
    CLOSING = "CLOSING"
    COMPLETED = "COMPLETED"


class SessionAction(StrEnum):
    CONTINUE = "CONTINUE"
    HANG_UP = "HANG_UP"
    TRANSFER = "TRANSFER"


@dataclass
class ConversationState:
    call_id: str
    session_id: str
    campaign_id: str
    stage: ConversationStage = ConversationStage.OPENING
    outcome: str = "UNKNOWN"
    fields: dict[str, Any] = field(default_factory=dict)
    recent_owner_utterances: list[str] = field(default_factory=list)
    recent_dialogue: list[dict[str, str]] = field(default_factory=list)
    skipped_fields: set[str] = field(default_factory=set)
    asked_fields: set[str] = field(default_factory=set)
    asked_field_counts: dict[str, int] = field(default_factory=dict)
    last_asked_field: str | None = None
    unclear_turns: int = 0
    model_failures: int = 0
    do_not_contact: bool = False
    callback_requested: bool = False
    human_transfer_requested: bool = False
    ended: bool = False


@dataclass(frozen=True)
class AgentReply:
    text: str
    action: SessionAction = SessionAction.CONTINUE
    question_field: str | None = None


@dataclass(frozen=True)
class LeadOutcome:
    outcome: str
    qualified: bool
    summary: str
    fields: dict[str, Any]
    callback_requested: bool
    human_followup_required: bool
