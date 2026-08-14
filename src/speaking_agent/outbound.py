from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import StrEnum
import importlib
import re
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from speaking_agent.answering import AnswerKind


class DialStatus(StrEnum):
    CONNECTED = "CONNECTED"
    NO_ANSWER = "NO_ANSWER"
    BUSY = "BUSY"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    INVALID_NUMBER = "INVALID_NUMBER"


class OutboundCallResultKind(StrEnum):
    ANSWERED_HUMAN = "ANSWERED_HUMAN"
    VOICEMAIL = "VOICEMAIL"
    ANSWERED_IVR = "ANSWERED_IVR"
    MACHINE_UNAVAILABLE = "MACHINE_UNAVAILABLE"
    ANSWERED_UNCERTAIN = "ANSWERED_UNCERTAIN"
    DROPPED = "DROPPED"
    NO_ANSWER = "NO_ANSWER"
    BUSY = "BUSY"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    INVALID_NUMBER = "INVALID_NUMBER"


@dataclass(frozen=True, slots=True)
class OutboundDialRequest:
    phone_number: str
    room_name: str
    trunk_id: str
    participant_identity: str
    participant_name: str = "Controlled test callee"
    ringing_timeout_seconds: int = 30
    maximum_call_seconds: int = 600
    participant_attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OutboundDialResult:
    status: DialStatus
    participant_identity: str
    sip_status_code: int | None = None
    sip_status: str | None = None


def is_e164(phone_number: str) -> bool:
    return re.fullmatch(r"\+[1-9][0-9]{7,14}", phone_number) is not None


def mask_phone_number(phone_number: str) -> str:
    return f"***{phone_number[-4:]}" if len(phone_number) >= 4 else "****"


def allowed_test_numbers(value: str | None) -> set[str]:
    return {
        number.strip()
        for number in (value or "").split(",")
        if number.strip()
    }


def ensure_controlled_test_number(phone_number: str, allowlist: set[str]) -> None:
    if not is_e164(phone_number):
        raise ValueError("Phone number must use E.164 format")
    if phone_number not in allowlist:
        raise PermissionError(
            "Phone number is not listed in SPEAKING_AGENT_ALLOWED_TEST_NUMBERS"
        )


def ensure_permitted_call_time(
    behavior: dict[str, Any],
    *,
    now: datetime | None = None,
) -> None:
    if behavior["controlled_test_mode"]:
        return
    timezone_name = behavior["calling_timezone"]
    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"Unknown calling timezone: {timezone_name}") from error
    current = now.astimezone(timezone) if now else datetime.now(timezone)
    start = time.fromisoformat(behavior["permitted_call_start"])
    end = time.fromisoformat(behavior["permitted_call_end"])
    local_time = current.timetz().replace(tzinfo=None)
    if not start <= local_time < end:
        raise PermissionError("Current time is outside the configured calling window")


def final_outbound_result(
    dial_status: DialStatus,
    answer_kind: AnswerKind | None = None,
    *,
    disconnected: bool = False,
) -> OutboundCallResultKind:
    if dial_status == DialStatus.CONNECTED:
        if disconnected:
            return OutboundCallResultKind.DROPPED
        return {
            AnswerKind.HUMAN: OutboundCallResultKind.ANSWERED_HUMAN,
            AnswerKind.VOICEMAIL: OutboundCallResultKind.VOICEMAIL,
            AnswerKind.IVR: OutboundCallResultKind.ANSWERED_IVR,
            AnswerKind.MACHINE_UNAVAILABLE: OutboundCallResultKind.MACHINE_UNAVAILABLE,
            AnswerKind.UNCERTAIN: OutboundCallResultKind.ANSWERED_UNCERTAIN,
            None: OutboundCallResultKind.ANSWERED_UNCERTAIN,
        }[answer_kind]
    return OutboundCallResultKind(dial_status.value)


class LiveKitSipDialer:
    def __init__(
        self,
        sip_service: Any,
        *,
        request_factory: Any | None = None,
        duration_factory: Any | None = None,
    ) -> None:
        self._sip_service = sip_service
        self._request_factory = request_factory
        self._duration_factory = duration_factory

    async def prepare(self) -> None:
        if self._request_factory is None:
            api = importlib.import_module("livekit.api")
            self._request_factory = api.CreateSIPParticipantRequest
        if self._duration_factory is None:
            duration_module = importlib.import_module("google.protobuf.duration_pb2")
            self._duration_factory = duration_module.Duration

    async def dial(self, request: OutboundDialRequest) -> OutboundDialResult:
        if not is_e164(request.phone_number):
            return OutboundDialResult(
                status=DialStatus.INVALID_NUMBER,
                participant_identity=request.participant_identity,
            )
        if self._request_factory is None or self._duration_factory is None:
            await self.prepare()

        ringing_timeout = self._duration_factory(
            seconds=request.ringing_timeout_seconds
        )
        maximum_duration = self._duration_factory(
            seconds=request.maximum_call_seconds
        )
        livekit_request = self._request_factory(
            room_name=request.room_name,
            sip_trunk_id=request.trunk_id,
            sip_call_to=request.phone_number,
            participant_identity=request.participant_identity,
            participant_name=request.participant_name,
            participant_attributes=request.participant_attributes,
            hide_phone_number=True,
            wait_until_answered=True,
            play_dialtone=False,
            ringing_timeout=ringing_timeout,
            max_call_duration=maximum_duration,
        )
        try:
            await self._sip_service.create_sip_participant(livekit_request)
        except Exception as error:
            status_code = getattr(error, "sip_status_code", None)
            status_text = getattr(error, "sip_status", None)
            return OutboundDialResult(
                status=self._map_failure(status_code),
                participant_identity=request.participant_identity,
                sip_status_code=status_code,
                sip_status=status_text,
            )
        return OutboundDialResult(
            status=DialStatus.CONNECTED,
            participant_identity=request.participant_identity,
        )

    @staticmethod
    def _map_failure(sip_status_code: int | None) -> DialStatus:
        if sip_status_code == 486:
            return DialStatus.BUSY
        if sip_status_code == 603:
            return DialStatus.REJECTED
        if sip_status_code in {408, 480}:
            return DialStatus.NO_ANSWER
        if sip_status_code in {404, 484}:
            return DialStatus.INVALID_NUMBER
        return DialStatus.FAILED
