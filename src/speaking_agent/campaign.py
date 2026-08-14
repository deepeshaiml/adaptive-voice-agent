from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Campaign:
    campaign_id: str
    name: str
    objective: str
    introduction: str
    opening: str
    opening_field: str
    required_disclosures: tuple[str, ...]
    desired_outcomes: tuple[str, ...]
    required_fields: tuple[str, ...]
    fields_by_outcome: dict[str, tuple[str, ...]]
    outcome_guidance: dict[str, str]
    field_types: dict[str, str]
    field_allowed_values: dict[str, tuple[str, ...]]
    questions: dict[str, str]
    terminal_outcomes: tuple[str, ...]
    closing_messages: dict[str, str]
    transfer_unavailable_message: str
    question_variants: dict[str, tuple[str, ...]] = field(default_factory=dict)
    conversation_brief: str = ""
    conversation_guidelines: tuple[str, ...] = ()
    scenario_playbook: tuple[dict[str, str], ...] = ()
    voicemail_message: str | None = None
    outcome_field: str | None = None
    classification_rules: tuple[str, ...] = ()
    qualified_outcomes: tuple[str, ...] = ()
    human_followup_outcomes: tuple[str, ...] = ()
    hard_stop_phrases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    faq_answers: dict[str, str] = field(default_factory=dict)
    prohibited_statements: tuple[str, ...] = ()
    behavior: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Campaign:
        required_keys = {
            "campaign_id",
            "name",
            "objective",
            "introduction",
            "opening",
            "opening_field",
            "required_disclosures",
            "desired_outcomes",
            "required_fields",
            "field_types",
            "questions",
            "transfer_unavailable_message",
        }
        missing = required_keys - data.keys()
        if missing:
            raise ValueError(f"Campaign is missing required keys: {', '.join(sorted(missing))}")

        for key in (
            "campaign_id",
            "name",
            "objective",
            "introduction",
            "opening",
            "opening_field",
        ):
            if not isinstance(data[key], str) or not data[key].strip():
                raise ValueError(f"Campaign {key} must be a non-empty string")

        desired_outcomes = tuple(data["desired_outcomes"])
        if not desired_outcomes or any(
            not isinstance(outcome, str) or not outcome.strip()
            for outcome in desired_outcomes
        ):
            raise ValueError("Campaign desired_outcomes must contain non-empty strings")
        if len(set(desired_outcomes)) != len(desired_outcomes):
            raise ValueError("Campaign desired_outcomes must be unique")
        required_outcomes = {"DO_NOT_CONTACT", "UNKNOWN"}
        missing_required_outcomes = required_outcomes - set(desired_outcomes)
        if missing_required_outcomes:
            raise ValueError(
                "Campaign desired_outcomes must include: "
                + ", ".join(sorted(missing_required_outcomes))
            )

        required_fields = tuple(data["required_fields"])
        if not required_fields or any(
            not isinstance(name, str) or not name.strip()
            for name in required_fields
        ):
            raise ValueError("Campaign required_fields must contain non-empty strings")
        required_disclosures = tuple(data["required_disclosures"])
        if not required_disclosures or any(
            not isinstance(disclosure, str) or not disclosure.strip()
            for disclosure in required_disclosures
        ):
            raise ValueError(
                "Campaign required_disclosures must contain non-empty strings"
            )
        normalized_introduction = data["introduction"].casefold()
        missing_disclosures = [
            disclosure
            for disclosure in required_disclosures
            if disclosure.casefold() not in normalized_introduction
        ]
        if missing_disclosures:
            raise ValueError(
                "Campaign introduction is missing required disclosures: "
                + ", ".join(missing_disclosures)
            )
        if data["introduction"].casefold() not in data["opening"].casefold():
            raise ValueError("Campaign opening must include the introduction")
        fields_by_outcome = {
            outcome: tuple(fields)
            for outcome, fields in data.get("fields_by_outcome", {}).items()
        }
        outcome_guidance = dict(data.get("outcome_guidance", {}))
        terminal_outcomes = tuple(data.get("terminal_outcomes", ()))
        closing_messages = dict(data.get("closing_messages", {}))
        qualified_outcomes = tuple(data.get("qualified_outcomes", ()))
        human_followup_outcomes = tuple(data.get("human_followup_outcomes", ()))
        hard_stop_phrases = {
            outcome: tuple(phrases)
            for outcome, phrases in data.get("hard_stop_phrases", {}).items()
        }
        invalid_hard_stops = [
            outcome
            for outcome, phrases in hard_stop_phrases.items()
            if not phrases
            or any(not isinstance(phrase, str) or not phrase.strip() for phrase in phrases)
        ]
        if invalid_hard_stops:
            raise ValueError(
                "Campaign hard-stop phrases must be non-empty strings: "
                + ", ".join(sorted(invalid_hard_stops))
            )
        if "DO_NOT_CONTACT" not in hard_stop_phrases:
            raise ValueError(
                "Campaign hard_stop_phrases must include DO_NOT_CONTACT"
            )
        referenced_outcomes = {
            *fields_by_outcome,
            *outcome_guidance,
            *terminal_outcomes,
            *closing_messages,
            *qualified_outcomes,
            *human_followup_outcomes,
            *hard_stop_phrases,
        }
        undeclared_outcomes = referenced_outcomes - set(desired_outcomes)
        if undeclared_outcomes:
            raise ValueError(
                "Campaign references undeclared outcomes: "
                + ", ".join(sorted(undeclared_outcomes))
            )
        if "DO_NOT_CONTACT" not in terminal_outcomes:
            raise ValueError(
                "Campaign terminal_outcomes must include DO_NOT_CONTACT"
            )

        missing_closings = set(desired_outcomes) - closing_messages.keys()
        if missing_closings:
            raise ValueError(
                "Campaign has no closing message for outcomes: "
                + ", ".join(sorted(missing_closings))
            )
        invalid_closings = [
            outcome
            for outcome, message in closing_messages.items()
            if not isinstance(message, str) or not message.strip()
        ]
        if invalid_closings:
            raise ValueError(
                "Campaign closing messages must be non-empty strings: "
                + ", ".join(sorted(invalid_closings))
            )
        if (
            not isinstance(data["transfer_unavailable_message"], str)
            or not data["transfer_unavailable_message"].strip()
        ):
            raise ValueError(
                "Campaign transfer_unavailable_message must be a non-empty string"
            )

        configured_fields = set(required_fields)
        for fields in fields_by_outcome.values():
            configured_fields.update(fields)

        if data["opening_field"] not in configured_fields:
            raise ValueError("Campaign opening_field must be a configured field")
        outcome_field = data.get("outcome_field")
        if outcome_field is not None and outcome_field not in configured_fields:
            raise ValueError("Campaign outcome_field must be a configured field")

        field_types = dict(data["field_types"])
        supported_field_types = {"string", "boolean", "number"}
        missing_field_types = configured_fields - field_types.keys()
        if missing_field_types:
            raise ValueError(
                "Campaign has no type for fields: "
                + ", ".join(sorted(missing_field_types))
            )
        unknown_typed_fields = field_types.keys() - configured_fields
        if unknown_typed_fields:
            raise ValueError(
                "Campaign types unconfigured fields: "
                + ", ".join(sorted(unknown_typed_fields))
            )
        unsupported_types = set(field_types.values()) - supported_field_types
        if unsupported_types:
            raise ValueError(
                "Campaign uses unsupported field types: "
                + ", ".join(sorted(unsupported_types))
            )
        raw_field_allowed_values = data.get("field_allowed_values", {})
        if not isinstance(raw_field_allowed_values, dict):
            raise ValueError("Campaign field allowed values must be an object")
        unknown_value_fields = raw_field_allowed_values.keys() - configured_fields
        if unknown_value_fields:
            raise ValueError(
                "Campaign has allowed values for unconfigured fields: "
                + ", ".join(sorted(unknown_value_fields))
            )
        invalid_value_fields = [
            name
            for name, values in raw_field_allowed_values.items()
            if field_types.get(name) != "string"
            or not isinstance(values, (list, tuple))
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
        ]
        if invalid_value_fields:
            raise ValueError(
                "Campaign field allowed values must contain non-empty strings for "
                "string fields: "
                + ", ".join(sorted(invalid_value_fields))
            )
        field_allowed_values = {
            name: tuple(value.casefold().strip() for value in values)
            for name, values in raw_field_allowed_values.items()
        }

        missing_questions = configured_fields - data["questions"].keys()
        if missing_questions:
            raise ValueError(
                "Campaign has no question for fields: "
                + ", ".join(sorted(missing_questions))
            )
        extra_questions = data["questions"].keys() - configured_fields
        if extra_questions:
            raise ValueError(
                "Campaign has questions for unconfigured fields: "
                + ", ".join(sorted(extra_questions))
            )
        invalid_questions = [
            name
            for name, question in data["questions"].items()
            if not isinstance(question, str) or not question.strip()
        ]
        if invalid_questions:
            raise ValueError(
                "Campaign questions must be non-empty strings: "
                + ", ".join(sorted(invalid_questions))
            )
        raw_question_variants = data.get("question_variants", {})
        if not isinstance(raw_question_variants, dict):
            raise ValueError("Campaign question_variants must be an object")
        unknown_variant_fields = raw_question_variants.keys() - configured_fields
        if unknown_variant_fields:
            raise ValueError(
                "Campaign has question variants for unconfigured fields: "
                + ", ".join(sorted(unknown_variant_fields))
            )
        invalid_variant_fields = [
            name
            for name, variants in raw_question_variants.items()
            if not isinstance(variants, (list, tuple))
            or not variants
            or any(
                not isinstance(variant, str) or not variant.strip()
                for variant in variants
            )
        ]
        if invalid_variant_fields:
            raise ValueError(
                "Campaign question variants must contain non-empty strings: "
                + ", ".join(sorted(invalid_variant_fields))
            )
        question_variants = {
            name: tuple(variants)
            for name, variants in raw_question_variants.items()
        }

        behavior = dict(data.get("behavior", {}))
        for key in (
            "ask_one_question_at_a_time",
            "avoid_repeating_known_information",
            "concise_responses",
        ):
            if not isinstance(behavior.get(key), bool):
                raise ValueError(f"Campaign behavior.{key} must be a boolean")
        for key in ("max_unclear_retries", "max_model_failures"):
            if not isinstance(behavior.get(key), int) or behavior[key] < 1:
                raise ValueError(f"Campaign behavior.{key} must be a positive integer")
        if (
            not isinstance(behavior.get("model_error_message"), str)
            or not behavior["model_error_message"].strip()
        ):
            raise ValueError(
                "Campaign behavior.model_error_message must be a non-empty string"
            )
        retention_days = behavior.get("data_retention_days")
        if not isinstance(retention_days, int) or retention_days < 1:
            raise ValueError(
                "Campaign behavior.data_retention_days must be a positive integer"
            )
        controlled_test_mode = behavior.get("controlled_test_mode")
        if not isinstance(controlled_test_mode, bool):
            raise ValueError("Campaign behavior.controlled_test_mode must be a boolean")
        for key in ("campaign_enabled", "recording_enabled"):
            if not isinstance(behavior.get(key), bool):
                raise ValueError(f"Campaign behavior.{key} must be a boolean")
        for key in (
            "maximum_call_attempts",
            "call_attempt_window_hours",
            "minimum_call_interval_minutes",
        ):
            if not isinstance(behavior.get(key), int) or behavior[key] < 1:
                raise ValueError(f"Campaign behavior.{key} must be a positive integer")
        for key in (
            "model_timeout_seconds",
            "asr_timeout_seconds",
            "tts_timeout_seconds",
            "initial_answer_timeout_seconds",
            "conversation_idle_timeout_seconds",
            "cleanup_timeout_seconds",
        ):
            value = behavior.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"Campaign behavior.{key} must be positive")
        retention_minutes = retention_days * 24 * 60
        enforcement_minutes = max(
            behavior["call_attempt_window_hours"] * 60,
            behavior["minimum_call_interval_minutes"],
        )
        if retention_minutes < enforcement_minutes:
            raise ValueError(
                "Campaign data retention must cover all call-attempt policy horizons"
            )
        if not controlled_test_mode:
            for key in (
                "calling_timezone",
                "permitted_call_start",
                "permitted_call_end",
            ):
                if not isinstance(behavior.get(key), str) or not behavior[key]:
                    raise ValueError(
                        f"Campaign behavior.{key} is required outside controlled test mode"
                    )
            try:
                from datetime import time
                from zoneinfo import ZoneInfo

                ZoneInfo(behavior["calling_timezone"])
                start = time.fromisoformat(behavior["permitted_call_start"])
                end = time.fromisoformat(behavior["permitted_call_end"])
            except (ValueError, KeyError) as error:
                raise ValueError("Campaign calling window is invalid") from error
            if start >= end:
                raise ValueError("Campaign calling window start must be before end")

        faq_answers = dict(data.get("faq_answers", {}))
        if any(
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(answer, str)
            or not answer.strip()
            for question, answer in faq_answers.items()
        ):
            raise ValueError("Campaign FAQ questions and answers must be non-empty strings")
        prohibited_statements = tuple(data.get("prohibited_statements", ()))
        if any(
            not isinstance(statement, str) or not statement.strip()
            for statement in prohibited_statements
        ):
            raise ValueError("Campaign prohibited statements must be non-empty strings")
        classification_rules = tuple(data.get("classification_rules", ()))
        if any(
            not isinstance(rule, str) or not rule.strip()
            for rule in classification_rules
        ):
            raise ValueError("Campaign classification rules must be non-empty strings")
        if any(
            not isinstance(guidance, str) or not guidance.strip()
            for guidance in outcome_guidance.values()
        ):
            raise ValueError("Campaign outcome guidance must be non-empty strings")
        conversation_brief = data.get("conversation_brief", data["objective"])
        if (
            not isinstance(conversation_brief, str)
            or not conversation_brief.strip()
        ):
            raise ValueError("Campaign conversation_brief must be a non-empty string")
        conversation_guidelines = tuple(data.get("conversation_guidelines", ()))
        if any(
            not isinstance(guideline, str) or not guideline.strip()
            for guideline in conversation_guidelines
        ):
            raise ValueError(
                "Campaign conversation_guidelines must contain non-empty strings"
            )
        raw_scenario_playbook = data.get("scenario_playbook", ())
        if not isinstance(raw_scenario_playbook, (list, tuple)):
            raise ValueError("Campaign scenario_playbook must be an array")
        invalid_scenarios = [
            index
            for index, scenario in enumerate(raw_scenario_playbook)
            if not isinstance(scenario, dict)
            or set(scenario) != {"when", "strategy"}
            or any(
                not isinstance(value, str) or not value.strip()
                for value in scenario.values()
            )
        ]
        if invalid_scenarios:
            raise ValueError(
                "Campaign scenario_playbook entries require non-empty when and "
                "strategy strings at indices: "
                + ", ".join(str(index) for index in invalid_scenarios)
            )
        scenario_playbook = tuple(
            {"when": scenario["when"], "strategy": scenario["strategy"]}
            for scenario in raw_scenario_playbook
        )
        voicemail_message = data.get("voicemail_message")
        if voicemail_message is not None and (
            not isinstance(voicemail_message, str) or not voicemail_message.strip()
        ):
            raise ValueError("Campaign voicemail_message must be null or non-empty")
        if voicemail_message is not None:
            normalized_voicemail = voicemail_message.casefold()
            missing_voicemail_disclosures = [
                disclosure
                for disclosure in required_disclosures
                if disclosure.casefold() not in normalized_voicemail
            ]
            if missing_voicemail_disclosures:
                raise ValueError(
                    "Campaign voicemail_message is missing required disclosures: "
                    + ", ".join(missing_voicemail_disclosures)
                )

        return cls(
            campaign_id=data["campaign_id"],
            name=data["name"],
            objective=data["objective"],
            introduction=data["introduction"],
            opening=data["opening"],
            opening_field=data["opening_field"],
            required_disclosures=required_disclosures,
            desired_outcomes=desired_outcomes,
            required_fields=required_fields,
            fields_by_outcome=fields_by_outcome,
            outcome_guidance=outcome_guidance,
            field_types=field_types,
            field_allowed_values=field_allowed_values,
            questions=dict(data["questions"]),
            terminal_outcomes=terminal_outcomes,
            closing_messages=closing_messages,
            transfer_unavailable_message=data["transfer_unavailable_message"],
            question_variants=question_variants,
            conversation_brief=conversation_brief,
            conversation_guidelines=conversation_guidelines,
            scenario_playbook=scenario_playbook,
            voicemail_message=voicemail_message,
            outcome_field=outcome_field,
            classification_rules=classification_rules,
            qualified_outcomes=qualified_outcomes,
            human_followup_outcomes=human_followup_outcomes,
            hard_stop_phrases=hard_stop_phrases,
            faq_answers=faq_answers,
            prohibited_statements=prohibited_statements,
            behavior=behavior,
        )


def load_campaign(path: str | Path) -> Campaign:
    with Path(path).open(encoding="utf-8") as campaign_file:
        data = json.load(campaign_file)
    if not isinstance(data, dict):
        raise ValueError("Campaign root must be a JSON object")
    return Campaign.from_dict(data)
