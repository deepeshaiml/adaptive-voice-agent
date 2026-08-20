import unittest

from dataclasses import replace
from pathlib import Path

from speaking_agent.campaign import load_campaign
from speaking_agent.conversation import ConversationSession
from speaking_agent.market_data import (
    ComparableProperty,
    HttpMarketDataProvider,
    MarketDataError,
    MarketSnapshot,
)
from speaking_agent.mock_model import MockConversationModel
from speaking_agent.model import ModelInterpretation


CAMPAIGN_PATH = (
    Path(__file__).parents[1] / "campaigns" / "neoai_property_owner.json"
)


class MarketDataTests(unittest.TestCase):
    def test_snapshot_keeps_transactions_and_asking_prices_distinct(self) -> None:
        query = ComparableProperty(
            project="DAMAC Lagoons",
            cluster="Nice",
            bedrooms="4",
            property_type="townhouse",
        )
        snapshot = MarketSnapshot.from_dict(
            query,
            {
                "actual_transactions": {
                    "source": "Dubai Land Department",
                    "as_of": "2026-08-20",
                    "count": 9,
                    "low_aed": 3_550_000,
                    "high_aed": 3_950_000,
                    "median_aed": 3_750_000,
                },
                "current_listings": {
                    "source": "Approved brokerage feed",
                    "as_of": "2026-08-20",
                    "count": 12,
                    "low_aed": 3_850_000,
                    "high_aed": 4_200_000,
                    "median_aed": 4_000_000,
                },
                "confidence": "high",
            },
        )

        feedback = snapshot.spoken_feedback()
        context = snapshot.prompt_context()

        self.assertIn("actual registered transactions", feedback)
        self.assertIn("AED 3.75 million", feedback)
        self.assertIn("asking prices, not completed sales", feedback)
        self.assertEqual(context["recent_actual_transactions"]["count"], 9)
        self.assertEqual(context["current_asking_listings"]["count"], 12)

    def test_snapshot_rejects_invalid_price_evidence(self) -> None:
        query = ComparableProperty("DAMAC Lagoons", "Nice", "4", "townhouse")

        with self.assertRaises(MarketDataError):
            MarketSnapshot.from_dict(
                query,
                {
                    "actual_transactions": {
                        "source": "DLD",
                        "as_of": "2026-08-20",
                        "count": 1,
                        "low_aed": 4_000_000,
                        "high_aed": 3_000_000,
                    },
                    "confidence": "high",
                },
            )

    def test_remote_http_feed_requires_tls(self) -> None:
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            HttpMarketDataProvider("http://market.example.test/comparables")

    def test_demo_snapshot_preserves_warning(self) -> None:
        query = ComparableProperty("DAMAC Lagoons", "Nice", "4", "townhouse")
        snapshot = MarketSnapshot.from_dict(
            query,
            {
                "actual_transactions": {
                    "source": "DEMO ONLY",
                    "as_of": "2026-08-20",
                    "count": 1,
                    "low_aed": 1,
                    "high_aed": 1,
                },
                "confidence": "low",
                "demo": True,
                "warning": "FICTIONAL DEMO DATA.",
            },
        )

        self.assertTrue(snapshot.demo)
        self.assertEqual(snapshot.prompt_context()["warning"], "FICTIONAL DEMO DATA.")
        self.assertTrue(snapshot.spoken_feedback().startswith("FICTIONAL DEMO DATA."))


class ConversationMarketDataTests(unittest.IsolatedAsyncioTestCase):
    async def test_conversation_fetches_and_speaks_grounded_comparables(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        query_seen = None

        class Provider:
            async def get_comparables(self, query):
                nonlocal query_seen
                query_seen = query
                return MarketSnapshot.from_dict(
                    query,
                    {
                        "actual_transactions": {
                            "source": "Dubai Land Department",
                            "as_of": "2026-08-20",
                            "count": 9,
                            "low_aed": 3_550_000,
                            "high_aed": 3_950_000,
                            "median_aed": 3_750_000,
                        },
                        "current_listings": {
                            "source": "Approved brokerage feed",
                            "as_of": "2026-08-20",
                            "count": 12,
                            "low_aed": 3_850_000,
                            "high_aed": 4_200_000,
                        },
                        "confidence": "high",
                    },
                )

        session = ConversationSession(
            campaign,
            MockConversationModel(
                {
                    "I am selling now in DAMAC Lagoons, Nice. It is a 4 bedroom townhouse.": ModelInterpretation(
                        suggested_outcome="SELL",
                        field_updates={
                            "selling_intention": "selling now",
                            "project": "DAMAC Lagoons",
                            "cluster": "Nice",
                            "bedrooms": "4",
                            "property_type": "townhouse",
                        },
                    )
                }
            ),
            market_data_provider=Provider(),
        )
        session.start()

        reply = await session.receive(
            "I am selling now in DAMAC Lagoons, Nice. It is a 4 bedroom townhouse."
        )

        self.assertIsNotNone(query_seen)
        self.assertEqual(query_seen.cluster, "Nice")
        self.assertIn("actual registered transactions", reply.text)
        self.assertIn("asking prices, not completed sales", reply.text)
        self.assertTrue(session.state.market_feedback_discussed)

    async def test_non_controlled_campaign_refuses_demo_market_evidence(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        campaign = replace(
            campaign,
            behavior={**campaign.behavior, "controlled_test_mode": False},
        )

        class Provider:
            async def get_comparables(self, query):
                return MarketSnapshot.from_dict(
                    query,
                    {
                        "actual_transactions": {
                            "source": "DEMO ONLY",
                            "as_of": "2026-08-20",
                            "count": 1,
                            "low_aed": 1,
                            "high_aed": 1,
                        },
                        "confidence": "low",
                        "demo": True,
                        "warning": "FICTIONAL DEMO DATA.",
                    },
                )

        utterance = (
            "I am selling now in DAMAC Lagoons, Nice. It is a 4 bedroom townhouse."
        )
        session = ConversationSession(
            campaign,
            MockConversationModel(
                {
                    utterance: ModelInterpretation(
                        suggested_outcome="SELL",
                        field_updates={
                            "selling_intention": "selling now",
                            "project": "DAMAC Lagoons",
                            "cluster": "Nice",
                            "bedrooms": "4",
                            "property_type": "townhouse",
                        },
                    )
                }
            ),
            market_data_provider=Provider(),
        )
        session.start()

        reply = await session.receive(utterance)

        self.assertNotIn("FICTIONAL", reply.text)
        self.assertFalse(session.state.market_feedback_discussed)
        self.assertEqual(
            session.state.market_context["message"],
            "Demo market evidence is disabled outside controlled tests.",
        )


if __name__ == "__main__":
    unittest.main()