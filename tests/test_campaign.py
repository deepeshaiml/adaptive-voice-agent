from pathlib import Path
import unittest

from speaking_agent.campaign import load_campaign


CAMPAIGN_PATH = Path(__file__).parents[1] / "campaigns" / "property_owner.json"


class CampaignTests(unittest.TestCase):
    def test_loads_runtime_campaign_and_normalizes_field_groups(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)

        self.assertEqual(campaign.campaign_id, "property-owner-qualification")
        self.assertIn("UNKNOWN", campaign.desired_outcomes)
        self.assertIn("SELL_OR_RENT", campaign.qualified_outcomes)
        self.assertEqual(
            campaign.fields_by_outcome["SELL"][:2],
            ("property_location", "property_type"),
        )
        self.assertIn("intent", campaign.question_variants)
        self.assertGreaterEqual(len(campaign.opening_variants), 1)
        self.assertIn("flat", campaign.field_allowed_values["property_type"])
        self.assertIn(
            "Dubai Marina",
            campaign.field_extraction_hints["property_location"],
        )
        self.assertIn("Dubai", campaign.conversation_brief)
        self.assertIn(
            "where did you get my name",
            campaign.faq_aliases["how did you get my number"],
        )
        self.assertIn("who are you", campaign.faq_answer_only)
        self.assertIn(
            "{recipient_name}",
            campaign.personalized_preamble["recipient_confirmation"],
        )
        self.assertGreaterEqual(len(campaign.scenario_playbook), 1)
        self.assertEqual(campaign.voice_style["personality"].split(",")[0], "Warm")
        self.assertGreaterEqual(len(campaign.natural_conversation_rules), 1)
        self.assertEqual(campaign.conversation_flow[0]["stage"], "OPEN")
        self.assertGreaterEqual(len(campaign.field_collection_rules), 1)
        self.assertIn("acknowledgements", campaign.sample_phrases)
        self.assertIn(
            "allow_owner_to_interrupt",
            campaign.interruption_and_silence_handling,
        )
        self.assertIn("Jumeirah Village Circle", campaign.speech_recognition_context)

    def test_rejects_a_campaign_without_questions_for_configured_fields(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        data = {
            **campaign.__dict__,
            "desired_outcomes": list(campaign.desired_outcomes),
            "required_fields": ["intent", "missing_field"],
        }

        with self.assertRaisesRegex(ValueError, "missing_field"):
            type(campaign).from_dict(data)

    def test_rejects_references_to_undeclared_outcomes(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        data = {
            **campaign.__dict__,
            "desired_outcomes": list(campaign.desired_outcomes),
            "terminal_outcomes": [*campaign.terminal_outcomes, "UNDECLARED"],
        }

        with self.assertRaisesRegex(ValueError, "UNDECLARED"):
            type(campaign).from_dict(data)

    def test_requires_a_terminal_do_not_contact_hard_stop(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        without_dnc = {
            **campaign.__dict__,
            "desired_outcomes": [
                outcome
                for outcome in campaign.desired_outcomes
                if outcome != "DO_NOT_CONTACT"
            ],
            "terminal_outcomes": [
                outcome
                for outcome in campaign.terminal_outcomes
                if outcome != "DO_NOT_CONTACT"
            ],
            "closing_messages": {
                outcome: message
                for outcome, message in campaign.closing_messages.items()
                if outcome != "DO_NOT_CONTACT"
            },
            "hard_stop_phrases": {
                outcome: phrases
                for outcome, phrases in campaign.hard_stop_phrases.items()
                if outcome != "DO_NOT_CONTACT"
            },
        }

        with self.assertRaisesRegex(ValueError, "DO_NOT_CONTACT"):
            type(campaign).from_dict(without_dnc)

    def test_rejects_opening_without_required_disclosure_text(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        data = {
            **campaign.__dict__,
            "introduction": "Hello.",
            "opening": "Hello. Would you consider selling?",
        }

        with self.assertRaisesRegex(ValueError, "Acme Property"):
            type(campaign).from_dict(data)

    def test_rejects_missing_field_type(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        field_types = dict(campaign.field_types)
        field_types.pop("currently_listed")

        with self.assertRaisesRegex(ValueError, "currently_listed"):
            type(campaign).from_dict({**campaign.__dict__, "field_types": field_types})

    def test_rejects_invalid_field_allowed_values(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        invalid_configs = (
            {"unexpected": ["value"]},
            {"property_type": []},
            {"currently_listed": ["yes"]},
        )
        for field_allowed_values in invalid_configs:
            with self.subTest(field_allowed_values=field_allowed_values):
                with self.assertRaisesRegex(ValueError, "allowed values"):
                    type(campaign).from_dict(
                        {
                            **campaign.__dict__,
                            "field_allowed_values": field_allowed_values,
                        }
                    )

    def test_rejects_invalid_field_extraction_hints(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        invalid_configs = (
            {"unexpected": ["value"]},
            {"property_location": []},
            {"currently_listed": ["yes"]},
        )
        for field_extraction_hints in invalid_configs:
            with self.subTest(field_extraction_hints=field_extraction_hints):
                with self.assertRaisesRegex(ValueError, "extraction hints"):
                    type(campaign).from_dict(
                        {
                            **campaign.__dict__,
                            "field_extraction_hints": field_extraction_hints,
                        }
                    )

    def test_retention_must_cover_attempt_policy_horizons(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        behavior = {
            **campaign.behavior,
            "data_retention_days": 1,
            "call_attempt_window_hours": 48,
        }

        with self.assertRaisesRegex(ValueError, "retention"):
            type(campaign).from_dict({**campaign.__dict__, "behavior": behavior})

    def test_rejects_empty_disclosure_or_introduction(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        with self.assertRaisesRegex(ValueError, "introduction"):
            type(campaign).from_dict({**campaign.__dict__, "introduction": ""})
        with self.assertRaisesRegex(ValueError, "required_disclosures"):
            type(campaign).from_dict(
                {**campaign.__dict__, "required_disclosures": [""]}
            )
        with self.assertRaisesRegex(ValueError, "required_disclosures"):
            type(campaign).from_dict(
                {**campaign.__dict__, "required_disclosures": "Acme Property"}
            )

    def test_rejects_question_for_unconfigured_field(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        questions = {**campaign.questions, "unexpected": "Unexpected question?"}

        with self.assertRaisesRegex(ValueError, "unexpected"):
            type(campaign).from_dict({**campaign.__dict__, "questions": questions})

    def test_rejects_invalid_question_variants(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        invalid_variants = (
            {"unexpected": ["Question?"]},
            {"intent": []},
            {"intent": "Question?"},
        )
        for question_variants in invalid_variants:
            with self.subTest(question_variants=question_variants):
                with self.assertRaisesRegex(ValueError, "question variants"):
                    type(campaign).from_dict(
                        {
                            **campaign.__dict__,
                            "question_variants": question_variants,
                        }
                    )

    def test_rejects_invalid_adaptive_conversation_guidance(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        invalid_configs = (
            {"conversation_brief": ""},
            {"conversation_guidelines": [""]},
            {"scenario_playbook": "not-an-array"},
            {"scenario_playbook": [{"when": "Owner asks a question."}]},
            {
                "scenario_playbook": [
                    {"when": "Owner asks a question.", "strategy": ""}
                ]
            },
            {"voice_style": {"tone": ""}},
            {"natural_conversation_rules": [""]},
            {"field_collection_rules": [""]},
            {"sample_phrases": {"acknowledgements": [""]}},
            {"interruption_and_silence_handling": {"brief_pause_behavior": ""}},
            {"hard_stop_context_rules": [""]},
            {"conversation_flow": [{"stage": "OPEN"}]},
            {"opening_variants": ["Hello without disclosures."]},
            {"speech_recognition_context": ["not", "a", "string"]},
        )
        for invalid in invalid_configs:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    type(campaign).from_dict({**campaign.__dict__, **invalid})

    def test_requires_campaign_and_recording_switches(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        for key in ("campaign_enabled", "recording_enabled"):
            with self.subTest(key=key):
                behavior = {**campaign.behavior, key: "not-a-boolean"}
                with self.assertRaisesRegex(ValueError, key):
                    type(campaign).from_dict(
                        {**campaign.__dict__, "behavior": behavior}
                    )

    def test_rejects_malformed_retry_and_error_behavior(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        invalid_values = {
            "max_unclear_retries": "two",
            "max_model_failures": 0,
            "model_error_message": "",
            "conversation_memory_turns": 0,
        }
        for key, value in invalid_values.items():
            with self.subTest(key=key):
                behavior = {**campaign.behavior, key: value}
                with self.assertRaisesRegex(ValueError, key):
                    type(campaign).from_dict(
                        {**campaign.__dict__, "behavior": behavior}
                    )

    def test_rejects_empty_machine_and_prompt_policy_text(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        invalid_configs = (
            {"voicemail_message": ""},
            {"faq_answers": {"question": ""}},
            {"prohibited_statements": [""]},
            {"classification_rules": [""]},
            {"hard_stop_phrases": {"DO_NOT_CONTACT": [""]}},
        )
        for invalid in invalid_configs:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    type(campaign).from_dict({**campaign.__dict__, **invalid})

    def test_rejects_faq_answer_with_prohibited_statement(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)

        for unsafe_answer in (
            "I am a human",
            "I'm human.",
            "I’m a human.",
            "I’m an actual human.",
            "I'm actually a human caller.",
            "You’re speaking with a real person.",
            "You're talking to a real person.",
        ):
            with self.subTest(unsafe_answer=unsafe_answer):
                with self.assertRaisesRegex(ValueError, "prohibited statements"):
                    type(campaign).from_dict(
                        {
                            **campaign.__dict__,
                            "faq_answers": {"are you human": unsafe_answer},
                            "faq_aliases": {},
                            "faq_answer_only": (),
                        }
                    )

    def test_rejects_invalid_or_overlapping_faq_aliases(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        invalid_aliases = (
            ["not", "an", "object"],
            {"unknown question": ["alias"]},
            {"how did you get my number": []},
            {"how did you get my number": [""]},
            {
                "how did you get my number": ["shared alias"],
                "why are you calling me": ["shared alias"],
            },
        )

        for faq_aliases in invalid_aliases:
            with self.subTest(faq_aliases=faq_aliases):
                with self.assertRaisesRegex(ValueError, "FAQ aliases"):
                    type(campaign).from_dict(
                        {**campaign.__dict__, "faq_aliases": faq_aliases}
                    )

    def test_rejects_invalid_answer_only_faq_references(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        invalid_values = (
            "who are you",
            [""],
            ["unknown question"],
            ["who are you", "who are you"],
        )

        for faq_answer_only in invalid_values:
            with self.subTest(faq_answer_only=faq_answer_only):
                with self.assertRaisesRegex(ValueError, "faq_answer_only"):
                    type(campaign).from_dict(
                        {
                            **campaign.__dict__,
                            "faq_answer_only": faq_answer_only,
                        }
                    )

    def test_rejects_invalid_personalized_preamble(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        valid = campaign.personalized_preamble
        invalid_values = (
            {"recipient_confirmation": "Missing other templates?"},
            {**valid, "recipient_confirmation": "Am I speaking with {name}?"},
            {**valid, "property_timing": "Calling about {property_reference}."},
            {
                **valid,
                "recipient_confirmation": "Am I speaking with {recipient_name}?",
            },
        )

        for personalized_preamble in invalid_values:
            with self.subTest(personalized_preamble=personalized_preamble):
                with self.assertRaisesRegex(ValueError, "personalized"):
                    type(campaign).from_dict(
                        {
                            **campaign.__dict__,
                            "personalized_preamble": personalized_preamble,
                        }
                    )

    def test_rejects_prohibited_claim_in_every_directly_spoken_message(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        unsafe = "I'm a live human agent calling you."
        unsafe_introduction = f"{campaign.introduction} {unsafe}"
        cases = {
            "introduction": {
                "introduction": unsafe_introduction,
                "opening": f"{unsafe_introduction} {campaign.questions[campaign.opening_field]}",
            },
            "opening": {"opening": f"{campaign.opening} {unsafe}"},
            "opening_variant": {
                "opening_variants": [f"{campaign.opening} {unsafe}"]
            },
            "question": {
                "questions": {**campaign.questions, "intent": unsafe}
            },
            "question_variant": {
                "question_variants": {**campaign.question_variants, "intent": [unsafe]}
            },
            "closing": {
                "closing_messages": {**campaign.closing_messages, "UNKNOWN": unsafe}
            },
            "transfer": {"transfer_unavailable_message": unsafe},
            "voicemail": {
                "voicemail_message": f"{campaign.introduction} {unsafe}"
            },
            "faq": {
                "faq_answers": {"are you human": unsafe},
                "faq_aliases": {},
                "faq_answer_only": (),
            },
            "model_error": {
                "behavior": {**campaign.behavior, "model_error_message": unsafe}
            },
        }

        for surface, overrides in cases.items():
            with self.subTest(surface=surface):
                with self.assertRaisesRegex(ValueError, "prohibited statements"):
                    type(campaign).from_dict({**campaign.__dict__, **overrides})

    def test_voicemail_message_requires_disclosures(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)

        with self.assertRaisesRegex(ValueError, "voicemail_message"):
            type(campaign).from_dict(
                {
                    **campaign.__dict__,
                    "voicemail_message": "Please return our call. Thank you.",
                }
            )


if __name__ == "__main__":
    unittest.main()