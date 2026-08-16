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

    async def test_model_receives_bounded_prior_dialogue(self) -> None:
        observed_histories: list[list[str]] = []
        observed_dialogues: list[list[dict[str, str]]] = []

        class HistoryModel:
            async def interpret(self, utterance, state, campaign):
                del utterance, campaign
                observed_histories.append(list(state.recent_owner_utterances))
                observed_dialogues.append(list(state.recent_dialogue))
                return ModelInterpretation(answer="I remember that.")

        campaign = replace(
            self.campaign,
            behavior={**self.campaign.behavior, "conversation_memory_turns": 2},
        )
        session = ConversationSession(campaign, HistoryModel())
        session.start()

        await session.receive("First detail")
        await session.receive("Second detail")
        await session.receive("What did I say earlier?")

        self.assertEqual(observed_histories[0], [])
        self.assertEqual(observed_histories[1], ["First detail"])
        self.assertEqual(
            observed_histories[2],
            ["First detail", "Second detail"],
        )
        self.assertEqual(
            observed_dialogues[0],
            [
                {
                    "role": "agent",
                    "text": session.opening,
                    "delivery": "delivered",
                    "question_field": "intent",
                }
            ],
        )
        self.assertEqual(
            [turn["role"] for turn in observed_dialogues[2]],
            ["agent", "owner", "agent", "owner", "agent"],
        )
        self.assertEqual(observed_dialogues[2][1]["text"], "First detail")
        self.assertEqual(observed_dialogues[2][3]["text"], "Second detail")
        self.assertEqual(
            session.state.recent_owner_utterances,
            ["Second detail", "What did I say earlier?"],
        )
        self.assertLessEqual(len(session.state.recent_dialogue), 5)

    async def test_one_word_both_selects_the_combined_qualified_branch(self) -> None:
        session = ConversationSession(self.campaign, MockConversationModel())
        session.start()

        reply = await session.receive("both")

        self.assertEqual(session.state.outcome, "SELL_OR_RENT")
        self.assertEqual(session.state.fields["intent"], "SELL_OR_RENT")
        self.assertEqual(reply.text, self.campaign.questions["property_location"])

    async def test_explicit_sell_takes_precedence_over_future_timing(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "I might sell next year": ModelInterpretation(
                        suggested_outcome="FUTURE",
                        field_updates={"selling_timeline": "next year"},
                    )
                }
            ),
        )
        session.start()

        await session.receive("I might sell next year")

        self.assertEqual(session.state.outcome, "SELL")
        self.assertEqual(session.state.fields["intent"], "SELL")
        self.assertEqual(session.state.fields["selling_timeline"], "next year")

    async def test_hard_stop_is_enforced_without_model_cooperation(self) -> None:
        session = ConversationSession(self.campaign, MockConversationModel())
        session.start()

        reply = await session.receive("Please do not call me again.")

        self.assertTrue(session.state.do_not_contact)
        self.assertTrue(session.state.ended)
        self.assertEqual(session.result().outcome, "DO_NOT_CONTACT")
        self.assertEqual(reply.action, SessionAction.HANG_UP)

    async def test_unicode_apostrophe_do_not_contact_is_enforced(self) -> None:
        session = ConversationSession(self.campaign, MockConversationModel())
        session.start()

        await session.receive("Please don’t call me again")

        self.assertTrue(session.state.do_not_contact)
        self.assertEqual(session.result().outcome, "DO_NOT_CONTACT")

        for wording in (
            "Please do not contact me again",
            "Please dont call me again",
        ):
            with self.subTest(wording=wording):
                variant = ConversationSession(
                    self.campaign,
                    MockConversationModel(
                        {wording: ModelInterpretation(suggested_outcome="DO_NOT_CONTACT")}
                    ),
                )
                variant.start()
                await variant.receive(wording)
                self.assertTrue(variant.state.do_not_contact)

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

        transfer_session = ConversationSession(
            self.campaign,
            MockConversationModel(),
        )
        transfer_session.start()
        transfer_reply = await transfer_session.receive(
            "Do not call me again, connect me to someone"
        )

        self.assertEqual(transfer_reply.action, SessionAction.HANG_UP)
        self.assertTrue(transfer_session.state.do_not_contact)

        mixed_timing_session = ConversationSession(
            self.campaign,
            MockConversationModel(),
        )
        mixed_timing_session.start()
        mixed_timing_reply = await mixed_timing_session.receive(
            "Don't call me today or ever again, connect me to someone"
        )

        self.assertEqual(mixed_timing_reply.action, SessionAction.HANG_UP)
        self.assertTrue(mixed_timing_session.state.do_not_contact)

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

    async def test_temporary_do_not_call_request_is_not_persisted_as_dnc(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "Do not call now, call me tomorrow": ModelInterpretation(
                        suggested_outcome="CALLBACK",
                        field_updates={"callback_time": "tomorrow"},
                        callback_requested=True,
                    )
                }
            ),
        )
        session.start()

        await session.receive("Do not call now, call me tomorrow")

        self.assertFalse(session.state.do_not_contact)
        self.assertEqual(session.result().outcome, "CALLBACK")
        self.assertEqual(session.result().fields["callback_time"], "tomorrow")

        until_session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "Do not call me until tomorrow": ModelInterpretation(
                        suggested_outcome="DO_NOT_CONTACT"
                    )
                }
            ),
        )
        until_session.start()

        await until_session.receive("Do not call me until tomorrow")

        self.assertFalse(until_session.state.do_not_contact)
        self.assertEqual(until_session.state.outcome, "CALLBACK")

        compact_session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "not now, call tomorrow": ModelInterpretation(
                        suggested_outcome="DO_NOT_CONTACT"
                    )
                }
            ),
        )
        compact_session.start()

        await compact_session.receive("not now, call tomorrow")

        self.assertFalse(compact_session.state.do_not_contact)
        self.assertEqual(compact_session.state.outcome, "CALLBACK")
        self.assertTrue(compact_session.state.callback_requested)

        marketing_session = ConversationSession(
            self.campaign,
            MockConversationModel(),
        )
        marketing_session.start()

        await marketing_session.receive("Do not call me for marketing purposes")

        self.assertTrue(marketing_session.state.do_not_contact)
        self.assertEqual(marketing_session.result().outcome, "DO_NOT_CONTACT")

    async def test_model_only_do_not_contact_without_evidence_is_rejected(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "Yes, that sounds interesting": ModelInterpretation(
                        suggested_outcome="DO_NOT_CONTACT"
                    )
                }
            ),
        )
        session.start()

        await session.receive("Yes, that sounds interesting")

        self.assertFalse(session.state.do_not_contact)
        self.assertEqual(session.state.outcome, "UNKNOWN")

    async def test_lexical_collisions_do_not_create_property_intent(self) -> None:
        transfer_session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "Please connect me to someone": ModelInterpretation(
                        suggested_outcome="HUMAN_TRANSFER"
                    )
                }
            ),
        )
        transfer_session.start()
        transfer_reply = await transfer_session.receive("Please connect me to someone")

        neither_session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "neither": ModelInterpretation(
                        suggested_outcome="SELL_OR_RENT"
                    )
                }
            ),
        )
        neither_session.start()
        await neither_session.receive("neither")

        self.assertEqual(transfer_reply.action, SessionAction.TRANSFER)
        self.assertNotEqual(transfer_session.state.outcome, "RENT")
        self.assertEqual(neither_session.state.outcome, "UNKNOWN")

        negative_session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "negative": ModelInterpretation(
                        suggested_outcome="SELL_OR_RENT"
                    )
                }
            ),
        )
        negative_session.start()
        await negative_session.receive("I don't want to sell or rent")

        mixed_transfer_session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "mixed": ModelInterpretation(suggested_outcome="SELL")
                }
            ),
        )
        mixed_transfer_session.start()
        mixed_reply = await mixed_transfer_session.receive(
            "I want to sell, but connect me to a person"
        )

        self.assertEqual(negative_session.state.outcome, "UNKNOWN")
        self.assertEqual(mixed_reply.action, SessionAction.TRANSFER)
        self.assertEqual(mixed_transfer_session.state.outcome, "HUMAN_TRANSFER")

        rent_session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "rent": ModelInterpretation(suggested_outcome="UNKNOWN")
                }
            ),
        )
        rent_session.start()
        await rent_session.receive("I wouldn't sell, but I'd rent it")

        annoyed_transfer_session = ConversationSession(
            self.campaign,
            MockConversationModel(),
        )
        annoyed_transfer_session.start()
        annoyed_reply = await annoyed_transfer_session.receive(
            "I'm not interested, connect me to someone"
        )

        self.assertEqual(rent_session.state.outcome, "RENT")
        self.assertEqual(annoyed_reply.action, SessionAction.TRANSFER)

        pivot_session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {"pivot": ModelInterpretation(suggested_outcome="SELL")}
            ),
        )
        pivot_session.start()
        await pivot_session.receive("I don't want to rent and would rather sell")

        self.assertEqual(pivot_session.state.outcome, "SELL")

        punctuation_free_session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "pivot": ModelInterpretation(suggested_outcome="SELL")
                }
            ),
        )
        punctuation_free_session.start()
        await punctuation_free_session.receive("I am not renting I want to sell")

        self.assertEqual(punctuation_free_session.state.outcome, "SELL")

        reverse_pivot_session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "pivot": ModelInterpretation(suggested_outcome="RENT")
                }
            ),
        )
        reverse_pivot_session.start()
        await reverse_pivot_session.receive("I don't want to sell I'm renting it")

        self.assertEqual(reverse_pivot_session.state.outcome, "RENT")

    async def test_transfer_is_not_enabled_by_wording_when_campaign_omits_it(self) -> None:
        campaign = replace(
            self.campaign,
            desired_outcomes=tuple(
                outcome
                for outcome in self.campaign.desired_outcomes
                if outcome != "HUMAN_TRANSFER"
            ),
            terminal_outcomes=tuple(
                outcome
                for outcome in self.campaign.terminal_outcomes
                if outcome != "HUMAN_TRANSFER"
            ),
        )
        session = ConversationSession(
            campaign,
            MockConversationModel(
                {
                    "transfer": ModelInterpretation(
                        suggested_outcome="HUMAN_TRANSFER"
                    )
                }
            ),
        )
        session.start()

        reply = await session.receive("connect me to someone")

        self.assertEqual(reply.action, SessionAction.CONTINUE)
        self.assertFalse(session.state.human_transfer_requested)

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

    async def test_clean_faq_turn_bypasses_model_latency(self) -> None:
        class FailingModel:
            async def interpret(self, utterance, state, campaign):
                del utterance, state, campaign
                raise AssertionError("FAQ-only turn should not call the model")

        session = ConversationSession(self.campaign, FailingModel())
        session.start()

        reply = await session.receive("How did you get my number?")

        self.assertIn("contact list provided for this test call", reply.text)

    async def test_faq_mixed_with_property_intent_still_uses_model(self) -> None:
        calls: list[str] = []

        class MixedTurnModel:
            async def interpret(self, utterance, state, campaign):
                del state, campaign
                calls.append(utterance)
                return ModelInterpretation(suggested_outcome="SELL")

        session = ConversationSession(self.campaign, MixedTurnModel())
        session.start()

        await session.receive("How did you get my number? I want to sell.")

        self.assertEqual(calls, ["How did you get my number? I want to sell."])
        self.assertEqual(session.state.outcome, "SELL")

    async def test_transfer_worded_as_faq_bypasses_model_safely(self) -> None:
        calls: list[str] = []

        class TransferModel:
            async def interpret(self, utterance, state, campaign):
                del state, campaign
                calls.append(utterance)
                return ModelInterpretation(
                    suggested_outcome="HUMAN_TRANSFER",
                    human_transfer_requested=True,
                )

        session = ConversationSession(self.campaign, TransferModel())
        session.start()

        reply = await session.receive("Can I speak to a person?")

        self.assertEqual(calls, [])
        self.assertEqual(reply.action, SessionAction.TRANSFER)

    async def test_compound_faq_and_transfer_turn_prioritizes_transfer(self) -> None:
        calls: list[str] = []

        class TransferModel:
            async def interpret(self, utterance, state, campaign):
                del state, campaign
                calls.append(utterance)
                return ModelInterpretation(suggested_outcome="HUMAN_TRANSFER")

        session = ConversationSession(self.campaign, TransferModel())
        session.start()

        reply = await session.receive("Are you human? Connect me to someone")

        self.assertEqual(calls, [])
        self.assertEqual(reply.action, SessionAction.TRANSFER)

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

    async def test_repeated_acknowledgement_is_suppressed(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "sell": ModelInterpretation(
                        suggested_outcome="SELL",
                        acknowledgement="Got it.",
                    ),
                    "Dubai Marina": ModelInterpretation(
                        field_updates={"property_location": "Dubai Marina"},
                        acknowledgement="Got it.",
                    ),
                }
            ),
        )
        session.start()
        first_reply = await session.receive("sell")

        second_reply = await session.receive("Dubai Marina")

        self.assertTrue(first_reply.text.startswith("Got it."))
        self.assertFalse(second_reply.text.startswith("Got it."))

    async def test_model_can_rephrase_only_the_policy_selected_question(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "I might sell": ModelInterpretation(
                        suggested_outcome="SELL",
                        acknowledgement="That makes sense.",
                        next_question_field="property_location",
                        next_question="Which part of Dubai is the property in?",
                    )
                }
            ),
        )
        session.start()

        reply = await session.receive("I might sell")

        self.assertEqual(
            reply.text,
            "That makes sense. Which part of Dubai is the property in?",
        )

    async def test_unrelated_dynamic_question_uses_campaign_fallback(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "I might sell": ModelInterpretation(
                        suggested_outcome="SELL",
                        next_question_field="property_location",
                        next_question="Which area is your bank located in?",
                    )
                }
            ),
        )
        session.start()

        reply = await session.receive("I might sell")

        self.assertEqual(reply.text, self.campaign.questions["property_location"])

    async def test_invalid_dynamic_question_uses_campaign_fallback(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "I might sell": ModelInterpretation(
                        suggested_outcome="SELL",
                        next_question_field="property_type",
                        next_question="Where is it? What type is it?",
                    )
                }
            ),
        )
        session.start()

        reply = await session.receive("I might sell")

        self.assertEqual(reply.text, self.campaign.questions["property_location"])

    async def test_mislabeled_dynamic_question_uses_campaign_fallback(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "I might sell my apartment in Dubai Marina next year": ModelInterpretation(
                        suggested_outcome="SELL",
                        field_updates={
                            "property_location": "Dubai Marina",
                            "selling_timeline": "next year",
                        },
                        next_question_field="expected_price",
                        next_question="Is it an apartment or a villa?",
                    )
                }
            ),
        )
        session.start()

        reply = await session.receive(
            "I might sell my apartment in Dubai Marina next year"
        )

        self.assertEqual(reply.text, self.campaign.questions["expected_price"])

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

    async def test_unicode_human_identity_claim_is_filtered(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "human": ModelInterpretation(answer="I’m a human.")
                }
            ),
        )
        session.start()

        reply = await session.receive("human")

        self.assertNotIn("human", reply.text.casefold())

        actual_session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "actual": ModelInterpretation(answer="I’m an actual human.")
                }
            ),
        )
        actual_session.start()
        actual_reply = await actual_session.receive("actual")

        self.assertNotIn("actual human", actual_reply.text.casefold())

        for unsafe_answer in (
            "I'm actually a human caller.",
            "You're talking to a real person.",
        ):
            paraphrase_session = ConversationSession(
                self.campaign,
                MockConversationModel(
                    {"unsafe": ModelInterpretation(answer=unsafe_answer)}
                ),
            )
            paraphrase_session.start()
            paraphrase_reply = await paraphrase_session.receive("unsafe")
            self.assertNotIn("real person", paraphrase_reply.text.casefold())
            self.assertNotIn("human caller", paraphrase_reply.text.casefold())

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

    async def test_relational_phrase_is_not_stored_as_property_location(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "sell": ModelInterpretation(suggested_outcome="SELL"),
                    "name": ModelInterpretation(
                        field_updates={"property_location": "my wife's name"}
                    ),
                }
            ),
        )
        session.start()
        await session.receive("sell")

        await session.receive("My apartment is in my wife's name")

        self.assertNotIn("property_location", session.state.fields)

    async def test_configured_location_hint_is_extracted_deterministically(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {"sell": ModelInterpretation(suggested_outcome="SELL")}
            ),
        )
        session.start()
        await session.receive("sell")

        await session.receive("The property is in Dubai Marina")

        self.assertEqual(session.state.fields["property_location"], "Dubai Marina")

    async def test_negated_location_hint_is_not_extracted(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {"sell": ModelInterpretation(suggested_outcome="SELL")}
            ),
        )
        session.start()
        await session.receive("sell")

        await session.receive("The property is not in Dubai Marina")

        self.assertNotIn("property_location", session.state.fields)

    async def test_ungrounded_model_location_is_not_stored(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "sell": ModelInterpretation(suggested_outcome="SELL"),
                    "unknown": ModelInterpretation(
                        field_updates={"property_location": "Dubai Marina"}
                    ),
                }
            ),
        )
        session.start()
        await session.receive("sell")

        await session.receive("I would rather not say")

        self.assertNotIn("property_location", session.state.fields)

    async def test_ungrounded_model_price_and_boolean_are_not_stored(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "sell": ModelInterpretation(suggested_outcome="SELL"),
                    "refusal": ModelInterpretation(
                        field_updates={
                            "expected_price": "AED 2 million",
                            "currently_listed": False,
                        }
                    ),
                }
            ),
        )
        session.start()
        await session.receive("sell")

        await session.receive("I would rather not say")

        self.assertNotIn("expected_price", session.state.fields)
        self.assertNotIn("currently_listed", session.state.fields)

    async def test_model_boolean_requires_deterministic_context(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "sell": ModelInterpretation(suggested_outcome="SELL"),
                    "Dubai Marina": ModelInterpretation(
                        field_updates={"property_location": "Dubai Marina"}
                    ),
                    "apartment": ModelInterpretation(),
                    "next year": ModelInterpretation(
                        field_updates={"selling_timeline": "next year"}
                    ),
                    "two million": ModelInterpretation(
                        field_updates={"expected_price": "two million"}
                    ),
                    "no": ModelInterpretation(
                        field_updates={"currently_listed": True}
                    ),
                }
            ),
        )
        session.start()
        await session.receive("sell")
        await session.receive("Dubai Marina")
        await session.receive("apartment")
        await session.receive("next year")
        await session.receive("two million")

        await session.receive("no")

        self.assertFalse(session.result().fields["currently_listed"])

    async def test_negated_allowed_property_type_is_not_stored(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "sell": ModelInterpretation(suggested_outcome="SELL"),
                    "negative": ModelInterpretation(
                        field_updates={"property_type": "apartment"}
                    ),
                }
            ),
        )
        session.start()
        await session.receive("sell")

        await session.receive("It is not an apartment")

        self.assertNotIn("property_type", session.state.fields)

        correction_session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {"sell": ModelInterpretation(suggested_outcome="SELL")}
            ),
        )
        correction_session.start()
        await correction_session.receive("sell")

        await correction_session.receive("It is not an apartment it is a villa")

        self.assertEqual(correction_session.state.fields["property_type"], "villa")

    async def test_location_grounding_does_not_depend_on_hint_configuration(self) -> None:
        campaign = replace(self.campaign, field_extraction_hints={})
        session = ConversationSession(
            campaign,
            MockConversationModel(
                {
                    "sell": ModelInterpretation(suggested_outcome="SELL"),
                    "refusal": ModelInterpretation(
                        field_updates={"property_location": "I would rather not say"}
                    ),
                }
            ),
        )
        session.start()
        await session.receive("sell")

        await session.receive("I would rather not say")

        self.assertNotIn("property_location", session.state.fields)

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

    async def test_allowlisted_value_is_extracted_when_model_only_mentions_it(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "sell apartment": ModelInterpretation(
                        suggested_outcome="SELL",
                        acknowledgement="So you're selling an apartment.",
                    )
                }
            ),
        )
        session.start()

        await session.receive("I may sell my apartment")

        self.assertEqual(session.state.fields["property_type"], "apartment")

    async def test_explicit_property_in_area_extracts_location_without_timing(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "volunteered": ModelInterpretation(
                        suggested_outcome="SELL",
                        acknowledgement="That helps.",
                    )
                }
            ),
        )
        session.start()

        await session.receive(
            "I might sell my apartment in Dubai Marina next year."
        )

        self.assertEqual(session.state.fields["property_location"], "Dubai Marina")
        self.assertEqual(session.state.fields["property_type"], "apartment")
        self.assertEqual(session.state.fields["selling_timeline"], "next year")

    async def test_contextual_boolean_answer_is_extracted_without_model_update(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "sell": ModelInterpretation(suggested_outcome="SELL"),
                    "Dubai Marina": ModelInterpretation(
                        field_updates={"property_location": "Dubai Marina"}
                    ),
                    "apartment": ModelInterpretation(),
                    "two months": ModelInterpretation(
                        field_updates={"selling_timeline": "two months"}
                    ),
                    "two million": ModelInterpretation(
                        field_updates={"expected_price": "two million"}
                    ),
                    "no": ModelInterpretation(),
                }
            ),
        )
        session.start()
        await session.receive("sell")
        await session.receive("Dubai Marina")
        await session.receive("apartment")
        await session.receive("two months")
        await session.receive("two million")

        await session.receive("no")

        self.assertFalse(session.result().fields["currently_listed"])

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

    async def test_callback_keeps_an_established_property_outcome(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "sell": ModelInterpretation(suggested_outcome="SELL"),
                    "call me back": ModelInterpretation(
                        suggested_outcome="CALLBACK",
                        callback_requested=True,
                        field_updates={"callback_time": "tomorrow"},
                    ),
                }
            ),
        )
        session.start()
        await session.receive("sell")

        await session.receive("call me back tomorrow")

        self.assertEqual(session.state.outcome, "SELL")
        self.assertTrue(session.state.callback_requested)


if __name__ == "__main__":
    unittest.main()