import unittest

from speaking_agent.metrics import aggregate_call_metrics
from speaking_agent.records import CallRecord


class CallMetricsTests(unittest.TestCase):
    def test_aggregates_operational_outcomes_and_excludes_blocked_attempts(self) -> None:
        records = [
            CallRecord(
                call_id="sell",
                session_id="session-sell",
                campaign_id="campaign",
                connection_result="ANSWERED_HUMAN",
                outcome="SELL",
                qualified=True,
                summary="Sell lead.",
                duration_seconds=10,
                latencies={"speech_end_to_playback": 0.4},
            ),
            CallRecord(
                call_id="voicemail",
                session_id="session-voicemail",
                campaign_id="campaign",
                connection_result="VOICEMAIL",
                outcome="UNKNOWN",
                qualified=False,
                summary="Voicemail.",
                duration_seconds=6,
                latencies={"speech_end_to_playback": 0.6},
            ),
            CallRecord(
                call_id="blocked",
                session_id="session-blocked",
                campaign_id="campaign",
                connection_result="BLOCKED_DO_NOT_CONTACT",
                outcome="DO_NOT_CONTACT",
                qualified=False,
                summary="Suppressed.",
            ),
        ]

        metrics = aggregate_call_metrics(records)

        self.assertEqual(metrics.records, 3)
        self.assertEqual(metrics.calls_attempted, 2)
        self.assertEqual(metrics.answered, 2)
        self.assertEqual(metrics.answered_human, 1)
        self.assertEqual(metrics.voicemail, 1)
        self.assertEqual(metrics.qualified, 1)
        self.assertEqual(metrics.sell, 1)
        self.assertEqual(metrics.do_not_contact, 1)
        self.assertEqual(metrics.average_duration_seconds, 8)
        self.assertEqual(
            metrics.average_latencies_seconds["speech_end_to_playback"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()