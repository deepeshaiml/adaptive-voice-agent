from types import SimpleNamespace
from datetime import datetime
import unittest
from zoneinfo import ZoneInfo

from speaking_agent.answering import AnswerKind
from speaking_agent.outbound import (
    DialStatus,
    LiveKitSipDialer,
    OutboundCallResultKind,
    OutboundDialRequest,
    ensure_controlled_test_number,
    ensure_permitted_call_time,
    final_outbound_result,
    mask_phone_number,
)


class FakeSipError(Exception):
    def __init__(self, code: int, status: str) -> None:
        self.sip_status_code = code
        self.sip_status = status


class FakeSipService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.request = None

    async def create_sip_participant(self, request):
        self.request = request
        if self.error is not None:
            raise self.error
        return SimpleNamespace(participant_identity=request.participant_identity)


def dialer(service: FakeSipService) -> LiveKitSipDialer:
    return LiveKitSipDialer(
        service,
        request_factory=SimpleNamespace,
        duration_factory=SimpleNamespace,
    )


def dial_request(phone_number: str = "+15105550123") -> OutboundDialRequest:
    return OutboundDialRequest(
        phone_number=phone_number,
        room_name="controlled-room",
        trunk_id="ST_test",
        participant_identity="callee-test",
    )


class OutboundDialerTests(unittest.IsolatedAsyncioTestCase):
    async def test_connected_dial_uses_bounded_waiting_request(self) -> None:
        service = FakeSipService()
        sip_dialer = dialer(service)

        result = await sip_dialer.dial(dial_request())

        self.assertEqual(result.status, DialStatus.CONNECTED)
        self.assertTrue(service.request.hide_phone_number)
        self.assertTrue(service.request.wait_until_answered)
        self.assertEqual(service.request.ringing_timeout.seconds, 30)
        self.assertEqual(service.request.max_call_duration.seconds, 600)

    async def test_maps_sip_failure_statuses(self) -> None:
        cases = (
            (486, DialStatus.BUSY),
            (603, DialStatus.REJECTED),
            (408, DialStatus.NO_ANSWER),
            (404, DialStatus.INVALID_NUMBER),
            (500, DialStatus.FAILED),
        )
        for status_code, expected in cases:
            with self.subTest(status_code=status_code):
                sip_dialer = dialer(
                    FakeSipService(FakeSipError(status_code, "test"))
                )
                result = await sip_dialer.dial(dial_request())
                self.assertEqual(result.status, expected)

    async def test_invalid_number_never_calls_sip_service(self) -> None:
        service = FakeSipService()
        result = await dialer(service).dial(dial_request("555-1234"))

        self.assertEqual(result.status, DialStatus.INVALID_NUMBER)
        self.assertIsNone(service.request)

    def test_controlled_number_allowlist_and_masking(self) -> None:
        ensure_controlled_test_number("+15105550123", {"+15105550123"})
        self.assertEqual(mask_phone_number("+15105550123"), "***0123")
        with self.assertRaises(PermissionError):
            ensure_controlled_test_number("+15105550124", {"+15105550123"})

    def test_final_result_distinguishes_human_and_voicemail(self) -> None:
        self.assertEqual(
            final_outbound_result(DialStatus.CONNECTED, AnswerKind.HUMAN),
            OutboundCallResultKind.ANSWERED_HUMAN,
        )
        self.assertEqual(
            final_outbound_result(DialStatus.CONNECTED, AnswerKind.VOICEMAIL),
            OutboundCallResultKind.VOICEMAIL,
        )
        self.assertEqual(
            final_outbound_result(DialStatus.CONNECTED, AnswerKind.IVR),
            OutboundCallResultKind.ANSWERED_IVR,
        )
        self.assertEqual(
            final_outbound_result(DialStatus.CONNECTED, AnswerKind.MACHINE_UNAVAILABLE),
            OutboundCallResultKind.MACHINE_UNAVAILABLE,
        )
        self.assertEqual(
            final_outbound_result(DialStatus.CONNECTED, AnswerKind.UNCERTAIN),
            OutboundCallResultKind.ANSWERED_UNCERTAIN,
        )
        self.assertEqual(
            final_outbound_result(
                DialStatus.CONNECTED,
                AnswerKind.HUMAN,
                disconnected=True,
            ),
            OutboundCallResultKind.DROPPED,
        )

    def test_non_test_calling_window_is_enforced(self) -> None:
        behavior = {
            "controlled_test_mode": False,
            "calling_timezone": "Asia/Dubai",
            "permitted_call_start": "09:00",
            "permitted_call_end": "17:00",
        }
        ensure_permitted_call_time(
            behavior,
            now=datetime(2026, 8, 13, 10, 0, tzinfo=ZoneInfo("Asia/Dubai")),
        )
        with self.assertRaises(PermissionError):
            ensure_permitted_call_time(
                behavior,
                now=datetime(2026, 8, 13, 20, 0, tzinfo=ZoneInfo("Asia/Dubai")),
            )


if __name__ == "__main__":
    unittest.main()