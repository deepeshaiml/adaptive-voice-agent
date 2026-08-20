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


@dataclass(frozen=True, slots=True)
class ConversationContext:
    recipient_name: str | None = None
    property_reference: str | None = None
    known_fields: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, maximum_length in (
            ("recipient_name", 120),
            ("property_reference", 200),
        ):
            value = getattr(self, name)
            if value is None:
                continue
            normalized = " ".join(value.split())
            if not normalized or len(normalized) > maximum_length:
                raise ValueError(
                    f"Conversation metadata {name} must contain 1-{maximum_length} characters"
                )
            object.__setattr__(self, name, normalized)
        if bool(self.recipient_name) != bool(self.property_reference):
            raise ValueError(
                "Conversation metadata recipient_name and property_reference "
                "must be provided together"
            )
        normalized_fields: dict[str, Any] = {}
        for name, value in self.known_fields.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Conversation metadata field names must be non-empty strings")
            if isinstance(value, str):
                value = " ".join(value.split())
                if not value or len(value) > 200:
                    raise ValueError(
                        f"Conversation metadata field {name!r} must contain 1-200 characters"
                    )
            elif not isinstance(value, (bool, int, float)):
                raise ValueError(
                    f"Conversation metadata field {name!r} has an unsupported value"
                )
            normalized_fields[name.strip()] = value
        object.__setattr__(self, "known_fields", normalized_fields)


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
    transcript: list[dict[str, str]] = field(default_factory=list)
    market_context: dict[str, Any] | None = None
    market_feedback_discussed: bool = False
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
    market_data: dict[str, Any] | None = None
    market_feedback_discussed: bool = False
    transcript: tuple[dict[str, str], ...] = ()
