from pathlib import Path
from dataclasses import replace
import asyncio
import unittest

from speaking_agent.campaign import load_campaign
from speaking_agent.conversation import ConversationSession
from speaking_agent.domain import ConversationContext, SessionAction
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

    async def test_verified_metadata_drives_personalized_preamble(self) -> None:
        context = ConversationContext(
            recipient_name="  Mr. Ahmed  ",
            property_reference="your apartment in Marina Gate, Dubai Marina",
            known_fields={
                "property_location": "Dubai Marina",
                "property_type": "apartment",
            },
        )
        session = ConversationSession(
            self.campaign,
            MockConversationModel(),
            context=context,
        )

        opening = session.start()
        property_prompt = await session.receive("Yes")
        intent_prompt = await session.receive("Yes, go ahead")
        qualification_reply = await session.receive(
            "I may sell if I get a good price."
        )

        self.assertIn("automated assistant", opening.text)
        self.assertIn("Sam", opening.text)
        self.assertIn("Mr. Ahmed", opening.text)
        self.assertNotIn("Marina Gate", opening.text)
        self.assertIn("Marina Gate", property_prompt.text)
        self.assertIn("good time", property_prompt.text.casefold())
        self.assertEqual(intent_prompt.question_field, "intent")
        self.assertIn("keep it brief", intent_prompt.text.casefold())
        self.assertEqual(session.state.outcome, "SELL")
        self.assertEqual(session.state.fields["property_location"], "Dubai Marina")
        self.assertEqual(session.state.fields["property_type"], "apartment")
        self.assertEqual(qualification_reply.question_field, "selling_timeline")

    async def test_timing_confirmation_answers_attached_contact_source_question(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(),
            context=ConversationContext(
                recipient_name="Mr. Ahmed",
                property_reference="your apartment in Marina Gate",
            ),
        )
        session.start()
        await session.receive("Yes")

        reply = await session.receive(
            "Yeah, it's a good time. But from where you got my number?"
        )

        self.assertIn("contact list provided for this test call", reply.text)
        self.assertEqual(reply.question_field, self.campaign.opening_field)

    async def test_negated_bad_time_continues_instead_of_callback(self) -> None:
        for utterance in (
            "I'm not busy, go ahead",
            "No, it isn't a bad time",
            "No need to call back, you can talk",
        ):
            with self.subTest(utterance=utterance):
                session = ConversationSession(
                    self.campaign,
                    MockConversationModel(),
                    context=ConversationContext(
                        recipient_name="Mr. Ahmed",
                        property_reference="your apartment in Marina Gate",
                    ),
                )
                session.start()
                await session.receive("Yes")

                reply = await session.receive(utterance)

                self.assertFalse(session.state.callback_requested)
                self.assertEqual(session.state.outcome, "UNKNOWN")
                self.assertEqual(reply.question_field, self.campaign.opening_field)

    async def test_personalized_preamble_handles_wrong_recipient_and_bad_time(self) -> None:
        context = ConversationContext(
            recipient_name="Mr. Ahmed",
            property_reference="your apartment in Marina Gate",
        )
        wrong_recipient = ConversationSession(
            self.campaign,
            MockConversationModel(),
            context=context,
        )
        wrong_recipient.start()

        wrong_reply = await wrong_recipient.receive("No, wrong person")

        self.assertEqual(wrong_recipient.result().outcome, "WRONG_NUMBER")
        self.assertEqual(wrong_reply.action, SessionAction.HANG_UP)

        negated_name = ConversationSession(
            self.campaign,
            MockConversationModel(),
            context=context,
        )
        negated_name.start()

        negated_reply = await negated_name.receive("No, I'm not Ahmed")

        self.assertEqual(negated_name.result().outcome, "WRONG_NUMBER")
        self.assertNotIn("Marina Gate", negated_reply.text)

        absent_recipient = ConversationSession(
            self.campaign,
            MockConversationModel(),
            context=context,
        )
        absent_recipient.start()

        absent_reply = await absent_recipient.receive("Ahmed isn't here")

        self.assertEqual(absent_recipient.result().outcome, "WRONG_NUMBER")
        self.assertNotIn("Marina Gate", absent_reply.text)

        for denial in (
            "Ahmed isn’t here",
            "This is Ali",
            "Yes, this is Ali",
            "This is Ali, speaking",
            "Ali speaking",
            "No, this is Ahmed",
            "This is Ahmed's assistant",
            "This is Ahmed's wife",
            "This is Ahmed from security",
            "Ahmed's assistant speaking",
        ):
            with self.subTest(denial=denial):
                different_recipient = ConversationSession(
                    self.campaign,
                    MockConversationModel(),
                    context=context,
                )
                different_recipient.start()

                denial_reply = await different_recipient.receive(denial)

                self.assertEqual(
                    different_recipient.result().outcome,
                    "WRONG_NUMBER",
                )
                self.assertNotIn("Marina Gate", denial_reply.text)

        for confirmation in (
            "Yes",
            "Speaking",
            "Yes, this is Ahmed",
            "This is Ahmed, speaking",
            "Ahmed speaking",
        ):
            with self.subTest(confirmation=confirmation):
                confirmed_recipient = ConversationSession(
                    self.campaign,
                    MockConversationModel(),
                    context=context,
                )
                confirmed_recipient.start()

                confirmation_reply = await confirmed_recipient.receive(
                    confirmation
                )

                self.assertFalse(confirmed_recipient.state.ended)
                self.assertIn("Marina Gate", confirmation_reply.text)

        busy_session = ConversationSession(
            self.campaign,
            MockConversationModel(),
            context=context,
        )
        busy_session.start()
        await busy_session.receive("Yes")

        callback_prompt = await busy_session.receive("No, I'm busy")
        callback_close = await busy_session.receive("Tomorrow afternoon")

        self.assertEqual(callback_prompt.question_field, "callback_time")
        self.assertEqual(busy_session.result().outcome, "CALLBACK")
        self.assertEqual(
            busy_session.result().fields["callback_time"],
            "Tomorrow afternoon",
        )
        self.assertEqual(callback_close.action, SessionAction.HANG_UP)

    async def test_personalized_confirmation_requires_delivered_prompt(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(),
            context=ConversationContext(
                recipient_name="Mr. Ahmed",
                property_reference="your apartment in Marina Gate",
            ),
            delivery_tracking=True,
        )
        opening = session.start()

        premature_reply = await session.receive("Yes?")

        self.assertEqual(premature_reply.text, opening.text)
        self.assertNotIn("Marina Gate", premature_reply.text)

        session.mark_agent_reply_delivery(premature_reply.text, "delivered")
        confirmed_reply = await session.receive("Yes")

        self.assertIn("Marina Gate", confirmed_reply.text)

    async def test_personalized_confirmation_matches_ordered_unicode_name(self) -> None:
        cases = (
            ("Mr. Ahmed Ali", "This is Ahmed Ali", True),
            ("Mr. Ahmed Ali", "This is Ali Ahmed", False),
            ("Mr. Mohammed Mohammed", "Mohammed Mohammed speaking", True),
            ("Mr. Mohammed Mohammed", "Mohammed speaking", False),
            ("Mr. José Álvarez", "This is José Álvarez", True),
            ("Mr. José Álvarez", "This is Jos Lvarez", False),
        )

        for recipient_name, utterance, should_confirm in cases:
            with self.subTest(
                recipient_name=recipient_name,
                utterance=utterance,
            ):
                session = ConversationSession(
                    self.campaign,
                    MockConversationModel(),
                    context=ConversationContext(
                        recipient_name=recipient_name,
                        property_reference="your apartment in Marina Gate",
                    ),
                )
                session.start()

                reply = await session.receive(utterance)

                self.assertEqual("Marina Gate" in reply.text, should_confirm)
                self.assertEqual(session.state.ended, not should_confirm)

    def test_conversation_context_rejects_invalid_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "recipient_name"):
            ConversationContext(recipient_name="   ")
        with self.assertRaisesRegex(ValueError, "property_reference"):
            ConversationContext(property_reference="x" * 201)
        with self.assertRaisesRegex(ValueError, "provided together"):
            ConversationContext(recipient_name="Mr. Ahmed")

        context = ConversationContext(known_fields={"unconfigured": "value"})
        with self.assertRaisesRegex(ValueError, "unconfigured"):
            ConversationSession(
                self.campaign,
                MockConversationModel(),
                context=context,
            )

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

        await session.receive("What is the first detail?")
        await session.receive("What is the second detail?")
        await session.receive("What did I say earlier?")

        self.assertEqual(observed_histories[0], [])
        self.assertEqual(observed_histories[1], ["What is the first detail?"])
        self.assertEqual(
            observed_histories[2],
            ["What is the first detail?", "What is the second detail?"],
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
        self.assertEqual(
            observed_dialogues[2][1]["text"],
            "What is the first detail?",
        )
        self.assertEqual(
            observed_dialogues[2][3]["text"],
            "What is the second detail?",
        )
        self.assertEqual(
            session.state.recent_owner_utterances,
            ["What is the second detail?", "What did I say earlier?"],
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

    async def test_bare_stop_ends_as_not_interested_not_wrong_number(self) -> None:
        class WrongNumberModel:
            async def interpret(self, utterance, state, campaign):
                del utterance, state, campaign
                return ModelInterpretation(suggested_outcome="WRONG_NUMBER")

        session = ConversationSession(self.campaign, WrongNumberModel())
        session.start()

        reply = await session.receive("Stop.")

        self.assertEqual(session.state.outcome, "NOT_INTERESTED")
        self.assertFalse(session.state.do_not_contact)
        self.assertEqual(reply.action, SessionAction.HANG_UP)

    async def test_model_cannot_invent_terminal_outcome_without_evidence(self) -> None:
        for suggested_outcome in ("WRONG_NUMBER", "NOT_INTERESTED"):
            with self.subTest(suggested_outcome=suggested_outcome):
                session = ConversationSession(
                    self.campaign,
                    MockConversationModel(
                        {
                            "Please repeat that": ModelInterpretation(
                                suggested_outcome=suggested_outcome
                            )
                        }
                    ),
                )
                session.start()

                reply = await session.receive("Please repeat that")

                self.assertEqual(session.state.outcome, "UNKNOWN")
                self.assertFalse(session.state.ended)
                self.assertEqual(reply.action, SessionAction.CONTINUE)

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

    async def test_contact_source_faq_accepts_standalone_asr_paraphrases(self) -> None:
        class FailingModel:
            async def interpret(self, utterance, state, campaign):
                del utterance, state, campaign
                raise AssertionError("Contact-source FAQ should bypass the model")

        for utterance in (
            "From where you got my name.",
            "First, tell me from from where you go.",
        ):
            with self.subTest(utterance=utterance):
                session = ConversationSession(self.campaign, FailingModel())
                session.start()

                reply = await session.receive(utterance)

                self.assertIn("contact list provided for this test call", reply.text)
                self.assertFalse(session.state.ended)

    async def test_repetition_complaint_then_identity_question_is_answered(self) -> None:
        class FailingModel:
            async def interpret(self, utterance, state, campaign):
                del utterance, state, campaign
                raise AssertionError("Configured conversation repair should bypass model")

        session = ConversationSession(self.campaign, FailingModel())
        session.start()

        complaint_reply = await session.receive("And you repeat again.")
        identity_reply = await session.receive("Who are you?")

        self.assertIn("sorry", complaint_reply.text.casefold())
        self.assertIsNone(complaint_reply.question_field)
        self.assertIn("acme property", identity_reply.text.casefold())
        self.assertIn("automated assistant", identity_reply.text.casefold())
        self.assertIsNone(identity_reply.question_field)
        self.assertNotIn("selling or renting", identity_reply.text.casefold())
        self.assertFalse(session.state.ended)

    async def test_buyer_valuation_and_whatsapp_questions_use_safe_campaign_answers(self) -> None:
        class FailingModel:
            async def interpret(self, utterance, state, campaign):
                del utterance, state, campaign
                raise AssertionError("Approved standalone FAQ should bypass the model")

        cases = (
            (
                "Do you actually have a buyer?",
                "don't have a confirmed buyer",
                self.campaign.opening_field,
            ),
            (
                "Do you actually have a buyer, or are you just trying to get my listing?",
                "don't have a confirmed buyer",
                self.campaign.opening_field,
            ),
            (
                "Okay, how much can you sell my apartment for?",
                "don't have a live valuation",
                None,
            ),
            (
                "I'm busy. Just WhatsApp me.",
                "can't send whatsapp",
                None,
            ),
        )

        for utterance, expected_answer, expected_question_field in cases:
            with self.subTest(utterance=utterance):
                session = ConversationSession(self.campaign, FailingModel())
                session.start()

                reply = await session.receive(utterance)

                self.assertIn(expected_answer, reply.text.casefold())
                self.assertEqual(reply.question_field, expected_question_field)
                self.assertFalse(session.state.ended)

    async def test_hesitation_fragments_preserve_question_and_retry_budget(self) -> None:
        session = ConversationSession(self.campaign, MockConversationModel())
        session.start()
        await session.receive("I want to sell")
        asked_field = session.state.last_asked_field
        unclear_turns = session.state.unclear_turns

        first_reply = await session.receive("Ah yeah, so I'm thinking ah.")
        second_reply = await session.receive("This is.")

        self.assertIsNone(first_reply.question_field)
        self.assertIsNone(second_reply.question_field)
        self.assertNotEqual(first_reply.text, second_reply.text)
        self.assertEqual(session.state.last_asked_field, asked_field)
        self.assertEqual(session.state.unclear_turns, unclear_turns)
        self.assertFalse(session.state.ended)

    async def test_unrelated_corrections_do_not_loop_pending_question(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "sell": ModelInterpretation(suggested_outcome="SELL"),
                    "The property is in Abu Dhabi": ModelInterpretation(
                        field_updates={"property_location": "Abu Dhabi"},
                        answer="Got it, the property is in Abu Dhabi.",
                    ),
                    "It is a villa": ModelInterpretation(
                        field_updates={"property_type": "villa"},
                        answer="Got it, it is a villa.",
                    ),
                }
            ),
            context=ConversationContext(
                known_fields={
                    "property_location": "Dubai Marina",
                    "property_type": "apartment",
                }
            ),
        )
        session.start()
        timeline_prompt = await session.receive("sell")
        self.assertEqual(timeline_prompt.question_field, "selling_timeline")

        first_retry = await session.receive("The property is in Abu Dhabi")
        next_question = await session.receive("It is a villa")

        self.assertEqual(first_retry.question_field, "selling_timeline")
        self.assertIn("selling_timeline", session.state.skipped_fields)
        self.assertEqual(next_question.question_field, "expected_price")
        self.assertEqual(session.state.fields["property_location"], "Abu Dhabi")
        self.assertEqual(session.state.fields["property_type"], "villa")

    async def test_unknown_model_suggestion_cannot_erase_established_intent(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "sell": ModelInterpretation(suggested_outcome="SELL"),
                    "The property is in Abu Dhabi": ModelInterpretation(
                        suggested_outcome="UNKNOWN",
                        field_updates={"property_location": "Abu Dhabi"},
                    ),
                    "It is a villa": ModelInterpretation(
                        suggested_outcome="UNKNOWN",
                        field_updates={"property_type": "villa"},
                    ),
                }
            ),
            context=ConversationContext(
                known_fields={
                    "property_location": "Dubai Marina",
                    "property_type": "apartment",
                }
            ),
        )
        session.start()
        await session.receive("sell")
        await session.receive("The property is in Abu Dhabi")

        reply = await session.receive("It is a villa")

        self.assertEqual(session.state.outcome, "SELL")
        self.assertEqual(session.state.fields["intent"], "SELL")
        self.assertEqual(session.state.fields["property_location"], "Abu Dhabi")
        self.assertEqual(session.state.fields["property_type"], "villa")
        self.assertEqual(reply.question_field, "expected_price")

    async def test_unknown_price_skips_price_and_does_not_ask_again(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "sell": ModelInterpretation(suggested_outcome="SELL"),
                    "Abu Dhabi": ModelInterpretation(
                        field_updates={"property_location": "Abu Dhabi"}
                    ),
                    "villa": ModelInterpretation(
                        field_updates={"property_type": "villa"}
                    ),
                    "soon": ModelInterpretation(
                        field_updates={"selling_timeline": "soon"}
                    ),
                }
            ),
        )
        session.start()
        await session.receive("sell")
        await session.receive("Abu Dhabi")
        await session.receive("villa")
        price_prompt = await session.receive("soon")
        self.assertEqual(price_prompt.question_field, "expected_price")

        valuation_reply = await session.receive(
            "I don't know the range. Can you tell me the range?"
        )

        self.assertIn("don't have a live valuation", valuation_reply.text.casefold())
        self.assertIn("expected_price", session.state.skipped_fields)
        self.assertIsNone(valuation_reply.question_field)

    async def test_common_unknown_price_question_skips_without_qwen_loop(self) -> None:
        class FailingModel:
            async def interpret(self, utterance, state, campaign):
                del utterance, state, campaign
                raise AssertionError("Known valuation FAQ should bypass Qwen")

        session = ConversationSession(
            self.campaign,
            FailingModel(),
        )
        session.start()
        session.policy.apply_outcome(session.state, "SELL")
        session.state.last_asked_field = "expected_price"

        reply = await session.receive(
            "I'm not sure what price to ask. What is my property worth?"
        )

        self.assertIn("don't have a live valuation", reply.text.casefold())
        self.assertIn("expected_price", session.state.skipped_fields)
        self.assertEqual(session.state.unclear_turns, 0)
        self.assertIsNone(reply.question_field)

    async def test_boolean_answer_with_fillers_is_grounded_once(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "sell": ModelInterpretation(suggested_outcome="SELL"),
                    "Abu Dhabi": ModelInterpretation(
                        field_updates={"property_location": "Abu Dhabi"}
                    ),
                    "villa": ModelInterpretation(
                        field_updates={"property_type": "villa"}
                    ),
                    "soon": ModelInterpretation(
                        field_updates={"selling_timeline": "soon"}
                    ),
                    "AED 4 million": ModelInterpretation(
                        field_updates={"expected_price": "AED 4 million"}
                    ),
                }
            ),
        )
        session.start()
        await session.receive("sell")
        await session.receive("Abu Dhabi")
        await session.receive("villa")
        await session.receive("soon")
        listing_prompt = await session.receive("AED 4 million")
        self.assertEqual(listing_prompt.question_field, "currently_listed")
        self.assertEqual(
            session.policy.deterministic_field_updates(session.state, "Ah yes."),
            {"currently_listed": True},
        )
        self.assertEqual(
            session.policy.deterministic_field_updates(session.state, "Yeah"),
            {"currently_listed": True},
        )
        self.assertFalse(session.policy.is_hesitation_fragment("Yeah"))

        reply = await session.receive("Ah no, not yet.")

        self.assertFalse(session.result().fields["currently_listed"])
        self.assertEqual(reply.action, SessionAction.HANG_UP)

    async def test_uncertain_no_boolean_does_not_persist_false(self) -> None:
        session = ConversationSession(self.campaign, MockConversationModel())
        session.state.last_asked_field = "currently_listed"

        for utterance in ("No idea", "No, I'm not sure"):
            with self.subTest(utterance=utterance):
                updates = session.policy.deterministic_field_updates(
                    session.state,
                    utterance,
                )
                self.assertNotIn("currently_listed", updates)

    async def test_ambiguous_sale_range_requires_full_aed_clarification(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "sell": ModelInterpretation(suggested_outcome="SELL"),
                    "Abu Dhabi": ModelInterpretation(
                        field_updates={"property_location": "Abu Dhabi"}
                    ),
                    "villa": ModelInterpretation(
                        field_updates={"property_type": "villa"}
                    ),
                    "soon": ModelInterpretation(
                        field_updates={"selling_timeline": "soon"}
                    ),
                    "four to five": ModelInterpretation(
                        field_updates={"expected_price": "four to five"}
                    ),
                    "AED four to five million": ModelInterpretation(
                        field_updates={
                            "expected_price": "AED four to five million"
                        }
                    ),
                }
            ),
        )
        session.start()
        await session.receive("sell")
        await session.receive("Abu Dhabi")
        await session.receive("villa")
        await session.receive("soon")

        clarification = await session.receive("four to five")

        self.assertNotIn("expected_price", session.state.fields)
        self.assertEqual(clarification.question_field, "expected_price")
        self.assertIn("full amount or range in AED", clarification.text)

        listing_prompt = await session.receive("AED four to five million")

        self.assertEqual(
            session.state.fields["expected_price"],
            "AED four to five million",
        )
        self.assertEqual(listing_prompt.question_field, "currently_listed")

    async def test_repetition_complaint_does_not_repeat_pending_question(self) -> None:
        session = ConversationSession(self.campaign, MockConversationModel())
        session.start()
        await session.receive("I want to sell")

        reply = await session.receive(
            "Why are you asking the same question again and again?"
        )

        self.assertIn("sorry", reply.text.casefold())
        self.assertIsNone(reply.question_field)

    async def test_repeated_acknowledgement_leadin_is_suppressed(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "sell": ModelInterpretation(
                        suggested_outcome="SELL",
                        acknowledgement="Got it, you're considering selling.",
                    ),
                    "Abu Dhabi": ModelInterpretation(
                        field_updates={"property_location": "Abu Dhabi"},
                        acknowledgement="Got it, the property is in Abu Dhabi.",
                    ),
                }
            ),
        )
        session.start()
        first_reply = await session.receive("sell")
        second_reply = await session.receive("Abu Dhabi")

        self.assertTrue(first_reply.text.casefold().startswith("got it"))
        self.assertFalse(second_reply.text.casefold().startswith("got it"))

    async def test_statement_model_answer_is_never_spoken_as_acknowledgement(self) -> None:
        hallucination = "Your villa is worth AED 9 million."
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "It is a villa": ModelInterpretation(
                        field_updates={"property_type": "villa"},
                        answer=hallucination,
                    )
                }
            ),
        )
        session.start()

        reply = await session.receive("It is a villa")

        self.assertNotIn(hallucination, reply.text)

    async def test_explicit_skip_resets_next_field_retry_budget(self) -> None:
        session = ConversationSession(
            self.campaign,
            MockConversationModel(
                {
                    "sell": ModelInterpretation(suggested_outcome="SELL"),
                    "Abu Dhabi": ModelInterpretation(
                        field_updates={"property_location": "Abu Dhabi"}
                    ),
                    "villa": ModelInterpretation(
                        field_updates={"property_type": "villa"}
                    ),
                }
            ),
        )
        session.start()
        await session.receive("sell")
        await session.receive("Abu Dhabi")
        await session.receive("villa")
        self.assertEqual(session.state.last_asked_field, "selling_timeline")

        next_prompt = await session.receive("I don't know")

        self.assertIn("selling_timeline", session.state.skipped_fields)
        self.assertEqual(session.state.unclear_turns, 0)
        self.assertEqual(next_prompt.question_field, "expected_price")

    async def test_realistic_refusal_dnc_and_rental_turns_remain_deterministic(self) -> None:
        refusal_session = ConversationSession(self.campaign, MockConversationModel())
        refusal_session.start()
        refusal_reply = await refusal_session.receive("No, not interested.")

        self.assertEqual(refusal_session.result().outcome, "NOT_INTERESTED")
        self.assertEqual(refusal_reply.action, SessionAction.HANG_UP)
        self.assertNotIn("currently rented", refusal_reply.text.casefold())

        dnc_session = ConversationSession(self.campaign, MockConversationModel())
        dnc_session.start()
        await dnc_session.receive("You agents call every day! Stop calling me!")

        self.assertEqual(dnc_session.result().outcome, "DO_NOT_CONTACT")
        self.assertTrue(dnc_session.state.do_not_contact)

        rental_session = ConversationSession(self.campaign, MockConversationModel())
        rental_session.start()
        rental_reply = await rental_session.receive(
            "I want to rent it. My tenant is leaving next month."
        )

        self.assertEqual(rental_session.state.outcome, "RENT")
        self.assertEqual(rental_session.state.fields["availability_date"], "next month")
        self.assertEqual(rental_reply.question_field, "property_location")

    async def test_model_capability_claims_are_filtered(self) -> None:
        for unsafe_answer in (
            "I have a confirmed buyer for your unit.",
            "I'll send you a WhatsApp now.",
            "I checked recent transactions for your building.",
        ):
            with self.subTest(unsafe_answer=unsafe_answer):
                session = ConversationSession(
                    self.campaign,
                    MockConversationModel(
                        {"claim": ModelInterpretation(answer=unsafe_answer)}
                    ),
                )
                session.start()

                reply = await session.receive("claim")

                self.assertNotIn(unsafe_answer.casefold(), reply.text.casefold())
                self.assertEqual(reply.question_field, self.campaign.opening_field)

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
                    "What areas do you know in Dubai?": ModelInterpretation(
                        answer=(
                            "I know the main property areas in Dubai. "
                            "Would you like details?"
                        ),
                    )
                }
            ),
        )
        session.start()

        reply = await session.receive("What areas do you know in Dubai?")

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