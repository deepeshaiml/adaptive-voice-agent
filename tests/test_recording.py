from types import SimpleNamespace
import unittest

from speaking_agent.answering import AnswerKind
from speaking_agent.domain import LeadOutcome
from speaking_agent.observability import LatencyTrace, TimingEventName
from speaking_agent.recording import completed_call_record, failed_call_record
from speaking_agent.voice_session import VoiceCallResult


class CallRecordingTests(unittest.TestCase):
    def test_completed_record_contains_sales_summary_and_opt_in_transcript(self) -> None:
        trace = LatencyTrace()
        trace.record(TimingEventName.SPEECH_END)
        trace.record(TimingEventName.PLAYBACK_START)
        state = SimpleNamespace(
            call_id="call-1",
            session_id="session-1",
            campaign_id="campaign-1",
        )
        session = SimpleNamespace(
            conversation=SimpleNamespace(
                state=state,
                campaign=SimpleNamespace(behavior={"transcript_enabled": True}),
                context=SimpleNamespace(recipient_name="Ahmed"),
            ),
            trace=trace,
            audio_recorder=None,
        )
        result = VoiceCallResult(
            lead=LeadOutcome(
                outcome="SELL",
                qualified=True,
                summary="Owner may sell.",
                fields={"intent": "SELL"},
                callback_requested=False,
                human_followup_required=True,
                transcript=(
                    {"role": "owner", "text": "I am selling now."},
                ),
            ),
            answer_kind=AnswerKind.HUMAN,
            interruptions=1,
            disconnected=False,
            cleanup_errors=("transport:RuntimeError",),
        )

        record = completed_call_record(
            session,
            result,
            connection_result="ANSWERED_HUMAN",
            duration_seconds=12.5,
            phone_number_masked="***0123",
        )

        self.assertEqual(record.fields, {"intent": "SELL"})
        self.assertIn("speech_end_to_playback", record.latencies)
        self.assertEqual(record.error, "transport:RuntimeError")
        self.assertEqual(record.transcript[0]["role"], "owner")
        self.assertIn("CALL SUMMARY", record.summary)
        self.assertEqual(record.priority, "PRIORITY_3_POTENTIAL")

    def test_failure_record_preserves_recognized_do_not_contact(self) -> None:
        state = SimpleNamespace(
            call_id="call-1",
            session_id="session-1",
            campaign_id="campaign-1",
            outcome="DO_NOT_CONTACT",
            fields={"intent": "DO_NOT_CONTACT"},
            callback_requested=False,
            transcript=[],
            market_context=None,
            market_feedback_discussed=False,
        )
        campaign = SimpleNamespace(
            qualified_outcomes=(),
            human_followup_outcomes=(),
            behavior={"transcript_enabled": False},
        )
        session = SimpleNamespace(
            conversation=SimpleNamespace(
                state=state,
                campaign=campaign,
                context=SimpleNamespace(recipient_name=None),
            ),
            trace=LatencyTrace(),
            interruptions=0,
            _answer_kind=AnswerKind.HUMAN,
            audio_recorder=None,
        )

        record = failed_call_record(
            session,
            RuntimeError("hangup failed"),
            connection_result="ANSWERED_HUMAN",
            duration_seconds=3.0,
            phone_number_masked="***0123",
        )

        self.assertEqual(record.outcome, "DO_NOT_CONTACT")
        self.assertEqual(record.fields, {"intent": "DO_NOT_CONTACT"})
        self.assertEqual(record.error, "RuntimeError")


if __name__ == "__main__":
    unittest.main()