from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timedelta, timezone

from speaking_agent.records import AttemptReservationStatus, CallRepository


class ContactSuppressedError(PermissionError):
    pass


class ContactAttemptLimitError(PermissionError):
    pass


def phone_fingerprint(phone_number: str, secret: str) -> str:
    if len(secret.encode("utf-8")) < 32:
        raise ValueError("SPEAKING_AGENT_SUPPRESSION_KEY must be at least 32 bytes")
    return hmac.new(
        secret.encode("utf-8"),
        phone_number.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def suppression_key_identifier(secret: str) -> str:
    if len(secret.encode("utf-8")) < 32:
        raise ValueError("SPEAKING_AGENT_SUPPRESSION_KEY must be at least 32 bytes")
    return hmac.new(
        secret.encode("utf-8"),
        b"speaking-agent-suppression-key-v1",
        hashlib.sha256,
    ).hexdigest()


class ContactSuppressionService:
    def __init__(self, repository: CallRepository, secret: str) -> None:
        self._repository = repository
        self._secret = secret

    async def prepare(self) -> None:
        await self._repository.ensure_suppression_key(
            suppression_key_identifier(self._secret)
        )

    async def ensure_allowed(self, phone_number: str) -> None:
        await self.prepare()
        fingerprint = phone_fingerprint(phone_number, self._secret)
        if await self._repository.is_contact_suppressed(fingerprint):
            raise ContactSuppressedError("Contact has an active do-not-contact record")

    async def suppress(self, phone_number: str, source_call_id: str) -> None:
        await self.prepare()
        await self._repository.suppress_contact(
            phone_fingerprint(phone_number, self._secret),
            source_call_id,
        )


class ContactAttemptPolicy:
    def __init__(
        self,
        repository: CallRepository,
        secret: str,
        *,
        maximum_attempts: int,
        window_hours: int,
        minimum_interval_minutes: int,
    ) -> None:
        self._repository = repository
        self._secret = secret
        self.maximum_attempts = maximum_attempts
        self.window_hours = window_hours
        self.minimum_interval_minutes = minimum_interval_minutes

    async def reserve(
        self,
        phone_number: str,
        call_id: str,
        *,
        now: datetime | None = None,
    ) -> None:
        current = now or datetime.now(timezone.utc)
        await self._repository.ensure_suppression_key(
            suppression_key_identifier(self._secret)
        )
        fingerprint = phone_fingerprint(phone_number, self._secret)
        status = await self._repository.reserve_contact_attempt(
            fingerprint,
            call_id,
            current.isoformat(),
            (current - timedelta(hours=self.window_hours)).isoformat(),
            (
                current - timedelta(minutes=self.minimum_interval_minutes)
            ).isoformat(),
            self.maximum_attempts,
        )
        if status == AttemptReservationStatus.SUPPRESSED:
            raise ContactSuppressedError("Contact has an active do-not-contact record")
        if status == AttemptReservationStatus.MAXIMUM_REACHED:
            raise ContactAttemptLimitError("Maximum call attempts reached")
        if status == AttemptReservationStatus.TOO_SOON:
            raise ContactAttemptLimitError("Minimum call interval has not elapsed")