from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Protocol


class AttemptReservationStatus(StrEnum):
    RESERVED = "RESERVED"
    SUPPRESSED = "SUPPRESSED"
    MAXIMUM_REACHED = "MAXIMUM_REACHED"
    TOO_SOON = "TOO_SOON"


class SuppressionKeyMismatchError(RuntimeError):
    pass


class RetentionMigrationRequiredError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CallRecord:
    call_id: str
    session_id: str
    campaign_id: str
    connection_result: str
    outcome: str
    qualified: bool
    summary: str
    fields: dict[str, Any] = field(default_factory=dict)
    callback_requested: bool = False
    human_followup_required: bool = False
    answer_kind: str | None = None
    phone_number_masked: str | None = None
    interruptions: int = 0
    disconnected: bool = False
    duration_seconds: float = 0.0
    latencies: dict[str, float] = field(default_factory=dict)
    completed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    error: str | None = None


class CallRepository(Protocol):
    async def prepare(self) -> None: ...

    async def save(self, record: CallRecord) -> None: ...

    async def get(self, call_id: str) -> CallRecord | None: ...

    async def list_recent(self, limit: int = 50) -> list[CallRecord]: ...

    async def suppress_contact(
        self,
        phone_fingerprint: str,
        source_call_id: str,
    ) -> None: ...

    async def is_contact_suppressed(self, phone_fingerprint: str) -> bool: ...

    async def ensure_suppression_key(self, key_identifier: str) -> None: ...

    async def purge_expired(self) -> int: ...

    async def reserve_contact_attempt(
        self,
        phone_fingerprint: str,
        call_id: str,
        attempted_at: str,
        count_since: str,
        interval_since: str,
        maximum_attempts: int,
    ) -> AttemptReservationStatus: ...

    async def close(self) -> None: ...
