from pathlib import Path
from dataclasses import replace
import unittest

from speaking_agent.campaign import load_campaign
from speaking_agent.conversation import ConversationSession
from speaking_agent.domain import ConversationState
from speaking_agent.mock_model import MockConversationModel
from speaking_agent.policy import ConversationPolicy


CAMPAIGN_PATH = (
    Path(__file__).parents[1] / "campaigns" / "neoai_property_owner.json"
)


class DamacCampaignPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.campaign = load_campaign(CAMPAIGN_PATH)
        self.policy = ConversationPolicy(self.campaign)

    def test_maps_right_price_and_holding_language_to_canonical_intentions(self) -> None:
        state = ConversationState("call-1", "session-1", self.campaign.campaign_id)

        open_to_offer = self.policy.deterministic_field_updates(
            state,
            "I might sell if the offer is at the right price.",
        )
        holding = self.policy.deterministic_field_updates(
            state,
            "No plans to sell, I am holding it for the long term.",
        )

        self.assertEqual(
            open_to_offer["selling_intention"],
            "open to selling at the right price",
        )
        self.assertEqual(holding["selling_intention"], "holding long term")

    def test_extracts_asking_and_minimum_prices_from_pending_questions(self) -> None:
        state = ConversationState("call-1", "session-1", self.campaign.campaign_id)
        state.last_asked_field = "asking_price"
        asking = self.policy.deterministic_field_updates(state, "AED 4.1 million")
        state.last_asked_field = "minimum_price"
        minimum = self.policy.deterministic_field_updates(state, "I would take 4 million AED")

        self.assertEqual(asking["asking_price"], "AED 4.1 million")
        self.assertEqual(minimum["minimum_price"], "4 million AED")

    def test_declined_whatsapp_permission_skips_number_and_documents(self) -> None:
        state = ConversationState("call-1", "session-1", self.campaign.campaign_id)
        state.outcome = "SELL"
        ordered_fields = (
            *self.campaign.required_fields,
            *self.campaign.fields_by_outcome["SELL"],
        )
        for field_name in ordered_fields:
            if field_name == "whatsapp_permission":
                break
            state.fields[field_name] = (
                False if self.campaign.field_types[field_name] == "boolean" else "known"
            )
        state.fields["whatsapp_permission"] = False

        next_field = self.policy.next_missing_field(state)

        self.assertEqual(next_field, "follow_up_timing")


class FullTranscriptTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_transcript_is_independent_of_prompt_memory_window(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        campaign = replace(
            campaign,
            behavior={**campaign.behavior, "conversation_memory_turns": 1},
        )
        session = ConversationSession(campaign, MockConversationModel())
        session.start()
        await session.receive("I am selling now.")
        await session.receive("It is in DAMAC Lagoons.")
        session.abort()

        result = session.result()

        self.assertGreater(len(result.transcript), len(session.state.recent_dialogue))
        self.assertEqual(
            [turn["text"] for turn in result.transcript if turn["role"] == "owner"],
            ["I am selling now.", "It is in DAMAC Lagoons."],
        )


if __name__ == "__main__":
    unittest.main()