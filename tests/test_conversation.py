from pathlib import Path
from dataclasses import replace
import asyncio
import unittest

from speaking_agent.campaign import load_campaign
from speaking_agent.conversation import ConversationSession
from speaking_agent.domain import SessionAction
from speaking_agent.mock_model import MockConversationModel
from speaking_agent.model import ModelInterpretation


CAMPAIGN_PATH = Path(__file__).parents[1] / "campaigns" / "property_owner.json"


class ConversationScenarioTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.campaign = load_campaign(CAMPAIGN_PATH)

    async def test_volunteered_sell_details_skip_known_questions(self) -> None:
        session = ConversationSession(self.campaign, MockConversationModel())
        session.start()

        reply = await session.receive(
            "I already listed my apartment for sale and may sell in two months."
        )

        self.assertEqual(session.state.outcome, "SELL")
        self.assertEqual(session.state.fields["intent"], "SELL")
        self.assertEqual(session.state.fields["property_type"], "apartment")
        self.assertTrue(session.state.fields["currently_listed"])
        self.assertEqual(session.state.fields["selling_timeline"], "in two months")
        self.assertEqual(reply.text, self.campaign.questions["property_location"])

    async def test_one_word_both_selects_the_combined_qualified_branch(self) -> None:
        session = ConversationSession(self.campaign, MockConversationModel())
        session.start()

        reply = await session.receive("both")

        self.assertEqual(session.state.outcome, "SELL_OR_RENT")
        self.assertEqual(session.state.fields["intent"], "SELL_OR_RENT")
        self.assertEqual(reply.text, self.campaign.questions["property_location"])

    async def test_hard_stop_is_enforced_without_model_cooperation(self) -> None:
        session = ConversationSession(self.campaign, MockConversationModel())
        session.start()

        reply = await session.receive("Please do not call me again.")

        self.assertTrue(session.state.do_not_contact)
        self.assertTrue(session.state.ended)
        self.assertEqual(session.result().outcome, "DO_NOT_CONTACT")
        self.assertEqual(reply.action, SessionAction.HANG_UP)

    async def test_do_not_contact_takes_precedence_over_other_hard_stops(self) -> None:
        campaign = replace(
            self.campaign,
            hard_stop_phrases={
                "NOT_INTERESTED": ("not interested",),
                "DO_NOT_CONTACT": ("do not call",),
            },
        )
        session = ConversationSession(campaign, MockConversationModel())
        session.start()

        await session.receive("I am not interested, so do not call me again.")

        self.assertEqual(session.result().outcome, "DO_NOT_CONTACT")
        self.assertTrue(session.state.do_not_contact)

    async def test_do_not_contact_cannot_be_replaced_by_a_later_hard_stop(self) -> None:
        session = ConversationSession(self.campaign, MockConversationModel())
        session.start()
        await session.receive("Please do not call me again.")

        reply = await session.receive("This is also the wrong number.")

        self.assertEqual(session.result().outcome, "DO_NOT_CONTACT")
        self.assertTrue(session.state.do_not_contact)
        self.assertEqual(
            reply.text,
            self.campaign.closing_messages["DO_NOT_CONTACT"],
        )

    async def test_answers_question_then_returns_to_objective(self) -> None:
        session = ConversationSession(self.campaign, MockConversationModel())
        session.start()

        reply = await session.receive("How did you get my number?")

        self.assertIn("contact list provided for this test call", reply.text)
        self.assertTrue(
            reply.text.endswith(self.campaign.question_variants["intent"][0])
        )
        self.assertFalse(session.state.ended)

        second_reply = await session.receive("What is this about?")
        self.assertTrue(
            second_reply.text.endswith(self.campaign.question_variants["intent"][1])
        )

    async def test_acknowledges_an_answer_before_the_next_question(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "I might sell": ModelInterpretation(
                        suggested_outcome="SELL",
                        acknowledgement="That makes sense.",
                    )
                }
            ),
        )
        session.start()

        reply = await session.receive("I might sell")

        self.assertEqual(
            reply.text,
            f"That makes sense. {self.campaign.questions['property_location']}",
        )

    async def test_model_question_is_removed_before_policy_question(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "Dubai question": ModelInterpretation(
                        answer=(
                            "I know the main property areas in Dubai. "
                            "Would you like details?"
                        ),
                    )
                }
            ),
        )
        session.start()

        reply = await session.receive("Dubai question")

        self.assertEqual(reply.text.count("?"), 1)
        self.assertTrue(reply.text.startswith("I know the main property areas in Dubai."))
        self.assertTrue(
            reply.text.endswith(self.campaign.question_variants["intent"][0])
        )

    async def test_mid_qualification_question_is_not_stored_as_a_field(self) -> None:
        session = ConversationSession(self.campaign, MockConversationModel())
        session.start()
        await session.receive("I want to sell")

        reply = await session.receive("How did you get my number")

        self.assertNotIn("property_location", session.state.fields)
        self.assertIn("contact list provided for this test call", reply.text)
        self.assertTrue(
            reply.text.endswith(
                self.campaign.question_variants["property_location"][0]
            )
        )

    async def test_complete_sell_scenario_produces_structured_result(self) -> None:
        session = ConversationSession(self.campaign, MockConversationModel())
        session.start()

        replies = [
            await session.receive("I might sell in two months."),
            await session.receive("Dubai Marina"),
            await session.receive("It is an apartment"),
            await session.receive("AED 2 million"),
            await session.receive("No"),
        ]

        result = session.result()
        self.assertEqual(result.outcome, "SELL")
        self.assertTrue(result.qualified)
        self.assertTrue(result.human_followup_required)
        self.assertEqual(result.fields["property_location"], "Dubai Marina")
        self.assertEqual(result.fields["selling_timeline"], "in two months")
        self.assertFalse(result.fields["currently_listed"])
        self.assertEqual(replies[-1].action, SessionAction.HANG_UP)

    async def test_invalid_model_data_is_ignored_and_retries_are_bounded(self) -> None:
        malformed = ModelInterpretation(
            suggested_outcome="INVENTED",
            field_updates={"unconfigured_field": "value"},
        )
        session = ConversationSession(
            self.campaign,
            MockConversationModel({"unclear": malformed}),
        )
        session.start()

        first_reply = await session.receive("unclear")
        final_reply = await session.receive("unclear")

        self.assertEqual(first_reply.action, SessionAction.CONTINUE)
        self.assertEqual(final_reply.action, SessionAction.HANG_UP)
        self.assertEqual(session.result().outcome, "UNKNOWN")
        self.assertNotIn("unconfigured_field", session.state.fields)

    async def test_engine_uses_a_replacement_campaign_without_code_changes(self) -> None:
        campaign_data = {
            "campaign_id": "community-survey",
            "name": "Community survey",
            "objective": "Ask whether the resident wants to participate.",
            "introduction": "Hello, this is an automated community survey.",
            "opening": "Hello, this is an automated community survey. Participate?",
            "opening_field": "intent",
            "required_disclosures": ["automated community survey"],
            "desired_outcomes": [
                "PARTICIPATE",
                "DECLINE",
                "DO_NOT_CONTACT",
                "UNKNOWN",
            ],
            "required_fields": ["intent"],
            "field_types": {"intent": "string", "topic": "string"},
            "fields_by_outcome": {"PARTICIPATE": ["topic"]},
            "outcome_guidance": {
                "PARTICIPATE": "The resident agrees to participate.",
                "DECLINE": "The resident declines.",
                "DO_NOT_CONTACT": "The resident asks not to be contacted again.",
                "UNKNOWN": "No decision is clear.",
            },
            "outcome_field": "intent",
            "questions": {
                "intent": "Would you like to participate?",
                "topic": "Which community topic matters most to you?",
            },
            "terminal_outcomes": ["DECLINE", "DO_NOT_CONTACT"],
            "closing_messages": {
                "PARTICIPATE": "Thank you for participating.",
                "DECLINE": "Understood. Goodbye.",
                "DO_NOT_CONTACT": "We will not contact you again. Goodbye.",
                "UNKNOWN": "Thank you. Goodbye.",
            },
            "hard_stop_phrases": {
                "DO_NOT_CONTACT": ["do not contact me"],
            },
            "transfer_unavailable_message": "Transfer is unavailable. Goodbye.",
            "qualified_outcomes": ["PARTICIPATE"],
            "human_followup_outcomes": [],
            "behavior": {
                "ask_one_question_at_a_time": True,
                "avoid_repeating_known_information": True,
                "concise_responses": True,
                "max_unclear_retries": 2,
                "max_model_failures": 2,
                "model_error_message": "Please repeat that response.",
                "data_retention_days": 30,
                "campaign_enabled": True,
                "recording_enabled": False,
                "model_timeout_seconds": 30,
                "asr_timeout_seconds": 30,
                "tts_timeout_seconds": 30,
                "initial_answer_timeout_seconds": 10,
                "conversation_idle_timeout_seconds": 20,
                "cleanup_timeout_seconds": 2,
                "controlled_test_mode": True,
                "maximum_call_attempts": 3,
                "call_attempt_window_hours": 24,
                "minimum_call_interval_minutes": 15,
            },
        }
        campaign = type(self.campaign).from_dict(campaign_data)
        model = MockConversationModel(
            {
                "yes": ModelInterpretation(
                    suggested_outcome="PARTICIPATE",
                    field_updates={"intent": "PARTICIPATE"},
                )
            }
        )
        session = ConversationSession(campaign, model)
        session.start()

        reply = await session.receive("yes")
        final_reply = await session.receive("Public transport")

        self.assertEqual(reply.text, campaign.questions["topic"])
        self.assertEqual(final_reply.text, campaign.closing_messages["PARTICIPATE"])
        self.assertEqual(session.result().fields["topic"], "Public transport")

    async def test_model_timeout_is_bounded_by_application_policy(self) -> None:
        class SlowModel:
            async def interpret(self, utterance, state, campaign):
                await asyncio.sleep(1)
                return ModelInterpretation()

        campaign = replace(
            self.campaign,
            behavior={
                **self.campaign.behavior,
                "model_timeout_seconds": 0.001,
                "max_model_failures": 1,
            },
        )
        session = ConversationSession(campaign, SlowModel())
        session.start()

        reply = await session.receive("I might sell")

        self.assertTrue(session.state.ended)
        self.assertEqual(reply.action, SessionAction.HANG_UP)
        self.assertEqual(session.result().outcome, "UNKNOWN")

    async def test_model_cannot_overwrite_outcome_field_or_authorize_transfer(self) -> None:
        interpretation = ModelInterpretation(
            suggested_outcome="SELL",
            field_updates={
                "intent": "RENT",
                "currently_listed": "yes",
            },
            human_transfer_requested=True,
        )
        session = ConversationSession(
            self.campaign,
            MockConversationModel({"I want to sell": interpretation}),
        )
        session.start()

        await session.receive("I want to sell")

        self.assertEqual(session.state.outcome, "SELL")
        self.assertEqual(session.state.fields["intent"], "SELL")
        self.assertNotIn("currently_listed", session.state.fields)
        self.assertFalse(session.state.human_transfer_requested)

    async def test_generic_interest_does_not_invent_sell_or_rent_intent(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "Yeah, I am interested": ModelInterpretation(
                        suggested_outcome="SELL_OR_RENT",
                        field_updates={"intent": "SELL_OR_RENT"},
                        acknowledgement="Great, thanks.",
                    )
                }
            ),
        )
        session.start()

        reply = await session.receive("Yeah, I am interested")

        self.assertEqual(session.state.outcome, "UNKNOWN")
        self.assertNotIn("intent", session.state.fields)
        self.assertTrue(
            reply.text.endswith(self.campaign.question_variants["intent"][0])
        )

    async def test_referential_address_is_not_stored_as_a_location(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "both": ModelInterpretation(
                        suggested_outcome="SELL_OR_RENT",
                    ),
                    "You should know my address": ModelInterpretation(
                        suggested_outcome="SELL_OR_RENT",
                        field_updates={"property_location": "my address"},
                        acknowledgement="Thanks for sharing that.",
                    ),
                }
            ),
        )
        session.start()
        await session.receive("both")

        reply = await session.receive("You should know my address")

        self.assertNotIn("property_location", session.state.fields)
        self.assertTrue(
            reply.text.endswith(
                self.campaign.question_variants["property_location"][0]
            )
        )

    async def test_unclear_secondary_field_is_skipped_without_losing_lead(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "both": ModelInterpretation(
                        suggested_outcome="SELL_OR_RENT",
                    ),
                    "Dubai Marina": ModelInterpretation(
                        suggested_outcome="SELL_OR_RENT",
                        field_updates={"property_location": "Dubai Marina"},
                    ),
                    "something else": ModelInterpretation(
                        suggested_outcome="SELL_OR_RENT",
                        field_updates={"property_type": "something else"},
                        acknowledgement="I see.",
                    ),
                    "not sure": ModelInterpretation(
                        suggested_outcome="SELL_OR_RENT",
                        field_updates={"property_type": "not sure"},
                        acknowledgement="No problem.",
                    ),
                }
            ),
        )
        session.start()
        await session.receive("both")
        await session.receive("Dubai Marina")
        first_reply = await session.receive("something else")

        final_reply = await session.receive("not sure")

        self.assertIn("property_type", session.state.skipped_fields)
        self.assertNotIn("property_type", session.state.fields)
        self.assertTrue(first_reply.action == SessionAction.CONTINUE)
        self.assertEqual(final_reply.action, SessionAction.HANG_UP)
        self.assertEqual(session.result().outcome, "SELL_OR_RENT")
        self.assertTrue(session.result().qualified)

    async def test_property_type_allowlist_rejects_noise_but_accepts_flat(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "both": ModelInterpretation(
                        suggested_outcome="SELL_OR_RENT",
                    ),
                    "Dubai Marina": ModelInterpretation(
                        suggested_outcome="SELL_OR_RENT",
                        field_updates={"property_location": "Dubai Marina"},
                    ),
                    "leg": ModelInterpretation(
                        suggested_outcome="SELL_OR_RENT",
                        field_updates={"property_type": "leg"},
                    ),
                    "flat": ModelInterpretation(
                        suggested_outcome="SELL_OR_RENT",
                        field_updates={"property_type": "flat"},
                    ),
                }
            ),
        )
        session.start()
        await session.receive("both")
        await session.receive("Dubai Marina")

        clarification = await session.receive("leg")
        final_reply = await session.receive("flat")

        self.assertNotIn("leg", session.state.fields.values())
        self.assertEqual(clarification.action, SessionAction.CONTINUE)
        self.assertEqual(session.result().fields["property_type"], "flat")
        self.assertEqual(final_reply.action, SessionAction.HANG_UP)

    async def test_validated_human_transfer_outcome_authorizes_transfer(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "transfer me": ModelInterpretation(
                        suggested_outcome="HUMAN_TRANSFER",
                        human_transfer_requested=True,
                    )
                }
            ),
        )
        session.start()

        reply = await session.receive("transfer me")

        self.assertTrue(session.state.human_transfer_requested)
        self.assertEqual(reply.action, SessionAction.TRANSFER)

    async def test_explicit_callback_cancellation_clears_state(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "call me back": ModelInterpretation(
                        suggested_outcome="CALLBACK",
                        callback_requested=True,
                    ),
                    "actually sell": ModelInterpretation(
                        suggested_outcome="SELL",
                        callback_requested=False,
                    ),
                }
            ),
        )
        session.start()

        await session.receive("call me back")
        await session.receive("actually sell")

        self.assertEqual(session.state.outcome, "SELL")
        self.assertFalse(session.state.callback_requested)


if __name__ == "__main__":
    unittest.main()