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
        self.assertIn("flat", campaign.field_allowed_values["property_type"])
        self.assertIn("Dubai", campaign.conversation_brief)
        self.assertGreaterEqual(len(campaign.scenario_playbook), 1)

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