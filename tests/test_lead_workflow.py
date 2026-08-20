from datetime import datetime, timezone
import unittest

from speaking_agent.lead_workflow import (
    LeadPriority,
    LeadWorkflowEvent,
    NotificationMode,
    analyze_sales_call,
    build_whatsapp_url,
)


class LeadWorkflowTests(unittest.TestCase):
    def test_hot_seller_creates_immediate_complete_summary(self) -> None:
        transcript = (
            {"role": "owner", "text": "I am selling now if you can get AED 4 million."},
            {"role": "owner", "text": "It is a 4 bedroom townhouse in Nice."},
            {"role": "owner", "text": "What are similar units worth?"},
        )
        analysis = analyze_sales_call(
            outcome="SELL",
            fields={
                "selling_intention": "selling now",
                "project": "DAMAC Lagoons",
                "cluster": "Nice",
                "bedrooms": "4",
                "property_type": "townhouse",
                "asking_price": "AED 4 million",
                "whatsapp_permission": True,
                "whatsapp_number_confirmed": True,
                "floor_plan_available": True,
                "follow_up_timing": "immediately",
            },
            transcript=transcript,
            owner_name="Ahmed",
            phone_number_masked="***0123",
            completed_at="2026-08-20T12:30:00+00:00",
            duration_seconds=125.4,
            market_data={
                "recent_actual_transactions": {"count": 9},
                "current_asking_listings": {"count": 12},
            },
            market_feedback_discussed=True,
        )

        self.assertEqual(analysis.priority, LeadPriority.HOT)
        self.assertEqual(analysis.notification_mode, NotificationMode.IMMEDIATE)
        self.assertTrue(analysis.create_follow_up_task)
        self.assertEqual(analysis.follow_up_at, "2026-08-20T12:30:00+00:00")
        self.assertIn("CALL SUMMARY", analysis.summary_text)
        self.assertIn("Recent Transactions Discussed: Yes", analysis.summary_text)
        self.assertIn("Owner's Main Comments", analysis.summary_text)
        self.assertIn("Yasir should WhatsApp", analysis.recommended_next_action)
        self.assertEqual(
            type(analysis).from_dict(
                analysis.as_dict(),
                summary_text=analysis.summary_text,
            ),
            analysis,
        )

    def test_open_to_offer_has_priority_two_and_whatsapp_action(self) -> None:
        analysis = analyze_sales_call(
            outcome="SELL",
            fields={
                "selling_intention": "open to selling at the right price",
                "cluster": "Malta",
                "whatsapp_permission": True,
                "whatsapp_number_confirmed": True,
            },
            completed_at="2026-08-20T12:30:00+00:00",
        )
        event = LeadWorkflowEvent(
            call_id="call-1",
            campaign_id="campaign-1",
            owner_name="Ahmed",
            phone_number="+971501234567",
            phone_number_masked="***4567",
            analysis=analysis,
            transcript=(),
        )

        payload = event.payload()

        self.assertEqual(analysis.priority, LeadPriority.OPEN_TO_OFFER)
        self.assertTrue(payload["notify_yasir"])
        self.assertIn("wa.me/971501234567", payload["open_whatsapp_url"])
        self.assertNotIn("+971501234567", analysis.summary_text)

    def test_whatsapp_action_is_withheld_without_explicit_permission(self) -> None:
        analysis = analyze_sales_call(
            outcome="SELL",
            fields={
                "selling_intention": "selling now",
                "whatsapp_permission": False,
                "whatsapp_number_confirmed": False,
            },
        )
        event = LeadWorkflowEvent(
            call_id="call-1",
            campaign_id="campaign-1",
            owner_name="Ahmed",
            phone_number="+971501234567",
            phone_number_masked="***4567",
            analysis=analysis,
            transcript=(),
        )

        payload = event.payload()

        self.assertIsNone(payload["open_whatsapp_url"])
        self.assertIn("without permission", analysis.recommended_next_action)

    def test_future_seller_creates_task_without_urgent_notification(self) -> None:
        analysis = analyze_sales_call(
            outcome="FUTURE",
            fields={
                "selling_intention": "selling later",
                "follow_up_timing": "6 months later",
            },
            completed_at=datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc).isoformat(),
        )

        self.assertEqual(analysis.priority, LeadPriority.FUTURE)
        self.assertEqual(analysis.notification_mode, NotificationMode.NONE)
        self.assertTrue(analysis.create_follow_up_task)
        self.assertEqual(analysis.follow_up_at, "2027-02-20T12:00:00+00:00")

    def test_whatsapp_link_requires_e164(self) -> None:
        with self.assertRaisesRegex(ValueError, "E.164"):
            build_whatsapp_url("0501234567")


if __name__ == "__main__":
    unittest.main()