from __future__ import annotations

import json
from dataclasses import dataclass, field
from string import Formatter
from pathlib import Path
from typing import Any

from speaking_agent.text_safety import (
    claims_human_identity,
    normalize_match_text,
    normalize_text,
)


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
    field_extraction_hints: dict[str, tuple[str, ...]]
    field_dependencies: dict[str, dict[str, Any]]
    questions: dict[str, str]
    terminal_outcomes: tuple[str, ...]
    closing_messages: dict[str, str]
    transfer_unavailable_message: str
    question_variants: dict[str, tuple[str, ...]] = field(default_factory=dict)
    opening_variants: tuple[str, ...] = ()
    personalized_preamble: dict[str, str] = field(default_factory=dict)
    conversation_brief: str = ""
    conversation_guidelines: tuple[str, ...] = ()
    scenario_playbook: tuple[dict[str, str], ...] = ()
    voice_style: dict[str, Any] = field(default_factory=dict)
    natural_conversation_rules: tuple[str, ...] = ()
    conversation_flow: tuple[dict[str, Any], ...] = ()
    field_collection_rules: tuple[str, ...] = ()
    sample_phrases: dict[str, Any] = field(default_factory=dict)
    interruption_and_silence_handling: dict[str, Any] = field(default_factory=dict)
    hard_stop_context_rules: tuple[str, ...] = ()
    speech_recognition_context: str = ""
    voicemail_message: str | None = None
    outcome_field: str | None = None
    classification_rules: tuple[str, ...] = ()
    qualified_outcomes: tuple[str, ...] = ()
    human_followup_outcomes: tuple[str, ...] = ()
    hard_stop_phrases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    faq_answers: dict[str, str] = field(default_factory=dict)
    faq_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    faq_answer_only: tuple[str, ...] = ()
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
        if not isinstance(data["required_disclosures"], (list, tuple)):
            raise ValueError("Campaign required_disclosures must be an array")
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
        raw_field_extraction_hints = data.get("field_extraction_hints", {})
        if not isinstance(raw_field_extraction_hints, dict):
            raise ValueError("Campaign field_extraction_hints must be an object")
        unknown_hint_fields = raw_field_extraction_hints.keys() - configured_fields
        invalid_hint_fields = [
            name
            for name, values in raw_field_extraction_hints.items()
            if field_types.get(name) != "string"
            or not isinstance(values, (list, tuple))
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
        ]
        invalid_hint_fields = sorted(
            set(unknown_hint_fields) | set(invalid_hint_fields)
        )
        if invalid_hint_fields:
            raise ValueError(
                "Campaign field extraction hints are invalid for fields: "
                + ", ".join(invalid_hint_fields)
            )
        field_extraction_hints = {
            name: tuple(value.strip() for value in values)
            for name, values in raw_field_extraction_hints.items()
        }
        raw_field_dependencies = data.get("field_dependencies", {})
        if not isinstance(raw_field_dependencies, dict):
            raise ValueError("Campaign field_dependencies must be an object")
        invalid_dependencies = []
        for target, dependency in raw_field_dependencies.items():
            if (
                target not in configured_fields
                or not isinstance(dependency, dict)
                or set(dependency) != {"field", "equals"}
                or dependency.get("field") not in configured_fields
                or dependency.get("field") == target
            ):
                invalid_dependencies.append(target)
                continue
            source_type = field_types[dependency["field"]]
            expected = dependency["equals"]
            type_matches = (
                (source_type == "string" and isinstance(expected, str))
                or (source_type == "boolean" and isinstance(expected, bool))
                or (
                    source_type == "number"
                    and isinstance(expected, (int, float))
                    and not isinstance(expected, bool)
                )
            )
            if not type_matches:
                invalid_dependencies.append(target)
        if invalid_dependencies:
            raise ValueError(
                "Campaign field dependencies are invalid for fields: "
                + ", ".join(sorted(invalid_dependencies))
            )
        field_dependencies = {
            target: dict(dependency)
            for target, dependency in raw_field_dependencies.items()
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
        opening_variants = tuple(data.get("opening_variants", ()))
        invalid_opening_variants = [
            index
            for index, opening in enumerate(opening_variants)
            if not isinstance(opening, str)
            or not opening.strip()
            or any(
                disclosure.casefold() not in opening.casefold()
                for disclosure in required_disclosures
            )
        ]
        if invalid_opening_variants:
            raise ValueError(
                "Campaign opening_variants must be non-empty and include required "
                "disclosures at indices: "
                + ", ".join(str(index) for index in invalid_opening_variants)
            )
        personalized_preamble = dict(data.get("personalized_preamble", {}))
        if personalized_preamble:
            expected_preamble_keys = {
                "recipient_confirmation",
                "property_timing",
                "qualification",
            }
            if set(personalized_preamble) != expected_preamble_keys or any(
                not isinstance(value, str) or not value.strip()
                for value in personalized_preamble.values()
            ):
                raise ValueError(
                    "Campaign personalized_preamble requires recipient_confirmation, "
                    "property_timing, and qualification strings"
                )
            expected_placeholders = {
                "recipient_confirmation": {"recipient_name"},
                "property_timing": {"property_reference"},
                "qualification": set(),
            }
            for name, template in personalized_preamble.items():
                try:
                    placeholders = {
                        field_name
                        for _, field_name, _, _ in Formatter().parse(template)
                        if field_name is not None
                    }
                except ValueError as error:
                    raise ValueError(
                        f"Campaign personalized_preamble.{name} is invalid"
                    ) from error
                if placeholders != expected_placeholders[name] or not template.endswith("?"):
                    raise ValueError(
                        f"Campaign personalized_preamble.{name} has invalid placeholders or punctuation"
                    )
            normalized_confirmation = personalized_preamble[
                "recipient_confirmation"
            ].casefold()
            missing_preamble_disclosures = [
                disclosure
                for disclosure in required_disclosures
                if disclosure.casefold() not in normalized_confirmation
            ]
            if missing_preamble_disclosures:
                raise ValueError(
                    "Campaign personalized recipient confirmation is missing required "
                    "disclosures: " + ", ".join(missing_preamble_disclosures)
                )

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
        memory_turns = behavior.get("conversation_memory_turns", 12)
        if not isinstance(memory_turns, int) or isinstance(memory_turns, bool) or memory_turns < 1:
            raise ValueError(
                "Campaign behavior.conversation_memory_turns must be a positive integer"
            )
        behavior["conversation_memory_turns"] = memory_turns
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
        transcript_enabled = behavior.get("transcript_enabled", False)
        if not isinstance(transcript_enabled, bool):
            raise ValueError("Campaign behavior.transcript_enabled must be a boolean")
        behavior["transcript_enabled"] = transcript_enabled
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
        raw_faq_aliases = data.get("faq_aliases", {})
        if not isinstance(raw_faq_aliases, dict):
            raise ValueError("Campaign FAQ aliases must be an object")
        unknown_faq_aliases = raw_faq_aliases.keys() - faq_answers.keys()
        invalid_faq_aliases = [
            question
            for question, aliases in raw_faq_aliases.items()
            if not isinstance(aliases, (list, tuple))
            or not aliases
            or any(not isinstance(alias, str) or not alias.strip() for alias in aliases)
        ]
        if unknown_faq_aliases or invalid_faq_aliases:
            raise ValueError(
                "Campaign FAQ aliases are invalid for questions: "
                + ", ".join(sorted(set(unknown_faq_aliases) | set(invalid_faq_aliases)))
            )
        faq_aliases = {
            question: tuple(aliases)
            for question, aliases in raw_faq_aliases.items()
        }
        faq_routes: dict[str, str] = {}
        for question, aliases in (
            (question, (question, *faq_aliases.get(question, ())))
            for question in faq_answers
        ):
            for alias in aliases:
                normalized_alias = normalize_match_text(alias)
                existing_question = faq_routes.get(normalized_alias)
                if existing_question is not None and existing_question != question:
                    raise ValueError(
                        "Campaign FAQ aliases overlap between questions: "
                        f"{existing_question}, {question}"
                    )
                faq_routes[normalized_alias] = question
        raw_faq_answer_only = data.get("faq_answer_only", ())
        if not isinstance(raw_faq_answer_only, (list, tuple)) or any(
            not isinstance(question, str) or not question.strip()
            for question in raw_faq_answer_only
        ):
            raise ValueError(
                "Campaign faq_answer_only must contain non-empty FAQ questions"
            )
        faq_answer_only = tuple(raw_faq_answer_only)
        unknown_answer_only = set(faq_answer_only) - faq_answers.keys()
        if unknown_answer_only or len(set(faq_answer_only)) != len(faq_answer_only):
            raise ValueError(
                "Campaign faq_answer_only references invalid FAQ questions: "
                + ", ".join(sorted(unknown_answer_only or set(faq_answer_only)))
            )
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
        voice_style = dict(data.get("voice_style", {}))
        if _has_invalid_text_tree(voice_style):
            raise ValueError("Campaign voice_style contains invalid text")
        natural_conversation_rules = _text_tuple(
            data,
            "natural_conversation_rules",
        )
        field_collection_rules = _text_tuple(data, "field_collection_rules")
        hard_stop_context_rules = _text_tuple(data, "hard_stop_context_rules")
        raw_conversation_flow = data.get("conversation_flow", ())
        if not isinstance(raw_conversation_flow, (list, tuple)):
            raise ValueError("Campaign conversation_flow must be an array")
        invalid_flow_stages = [
            index
            for index, stage in enumerate(raw_conversation_flow)
            if not isinstance(stage, dict)
            or set(stage) != {"stage", "goal", "instructions", "exit_when"}
            or not isinstance(stage["stage"], str)
            or not stage["stage"].strip()
            or not isinstance(stage["goal"], str)
            or not stage["goal"].strip()
            or not isinstance(stage["exit_when"], str)
            or not stage["exit_when"].strip()
            or not isinstance(stage["instructions"], (list, tuple))
            or not stage["instructions"]
            or any(
                not isinstance(instruction, str) or not instruction.strip()
                for instruction in stage["instructions"]
            )
        ]
        if invalid_flow_stages:
            raise ValueError(
                "Campaign conversation_flow has invalid stages at indices: "
                + ", ".join(str(index) for index in invalid_flow_stages)
            )
        conversation_flow = tuple(
            {
                "stage": stage["stage"],
                "goal": stage["goal"],
                "instructions": tuple(stage["instructions"]),
                "exit_when": stage["exit_when"],
            }
            for stage in raw_conversation_flow
        )
        sample_phrases = dict(data.get("sample_phrases", {}))
        if _has_invalid_text_tree(sample_phrases):
            raise ValueError("Campaign sample_phrases contains invalid text")
        interruption_and_silence_handling = dict(
            data.get("interruption_and_silence_handling", {})
        )
        if _has_invalid_text_tree(interruption_and_silence_handling):
            raise ValueError(
                "Campaign interruption_and_silence_handling contains invalid text"
            )
        speech_recognition_context = data.get("speech_recognition_context", "")
        if not isinstance(speech_recognition_context, str):
            raise ValueError("Campaign speech_recognition_context must be a string")
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

        spoken_messages = [
            ("introduction", data["introduction"]),
            ("opening", data["opening"]),
            ("transfer_unavailable_message", data["transfer_unavailable_message"]),
            ("behavior.model_error_message", behavior["model_error_message"]),
            *(
                (f"opening_variants[{index}]", message)
                for index, message in enumerate(opening_variants)
            ),
            *(
                (f"personalized_preamble.{name}", message)
                for name, message in personalized_preamble.items()
            ),
            *((f"questions.{name}", message) for name, message in data["questions"].items()),
            *(
                (f"question_variants.{name}[{index}]", message)
                for name, variants in question_variants.items()
                for index, message in enumerate(variants)
            ),
            *(
                (f"closing_messages.{outcome}", message)
                for outcome, message in closing_messages.items()
            ),
            *(
                (f"faq_answers.{question}", answer)
                for question, answer in faq_answers.items()
            ),
        ]
        if voicemail_message is not None:
            spoken_messages.append(("voicemail_message", voicemail_message))
        unsafe_messages = [
            name
            for name, message in spoken_messages
            if _contains_prohibited_claim(message, prohibited_statements)
        ]
        if unsafe_messages:
            raise ValueError(
                "Campaign spoken text contains prohibited statements: "
                + ", ".join(sorted(unsafe_messages))
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
            field_extraction_hints=field_extraction_hints,
            field_dependencies=field_dependencies,
            questions=dict(data["questions"]),
            terminal_outcomes=terminal_outcomes,
            closing_messages=closing_messages,
            transfer_unavailable_message=data["transfer_unavailable_message"],
            question_variants=question_variants,
            opening_variants=opening_variants,
            personalized_preamble=personalized_preamble,
            conversation_brief=conversation_brief,
            conversation_guidelines=conversation_guidelines,
            scenario_playbook=scenario_playbook,
            voice_style=voice_style,
            natural_conversation_rules=natural_conversation_rules,
            conversation_flow=conversation_flow,
            field_collection_rules=field_collection_rules,
            sample_phrases=sample_phrases,
            interruption_and_silence_handling=interruption_and_silence_handling,
            hard_stop_context_rules=hard_stop_context_rules,
            speech_recognition_context=speech_recognition_context.strip(),
            voicemail_message=voicemail_message,
            outcome_field=outcome_field,
            classification_rules=classification_rules,
            qualified_outcomes=qualified_outcomes,
            human_followup_outcomes=human_followup_outcomes,
            hard_stop_phrases=hard_stop_phrases,
            faq_answers=faq_answers,
            faq_aliases=faq_aliases,
            faq_answer_only=faq_answer_only,
            prohibited_statements=prohibited_statements,
            behavior=behavior,
        )


def load_campaign(path: str | Path) -> Campaign:
    with Path(path).open(encoding="utf-8") as campaign_file:
        data = json.load(campaign_file)
    if not isinstance(data, dict):
        raise ValueError("Campaign root must be a JSON object")
    return Campaign.from_dict(data)


def _text_tuple(data: dict[str, Any], key: str) -> tuple[str, ...]:
    values = data.get(key, ())
    if not isinstance(values, (list, tuple)) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"Campaign {key} must contain non-empty strings")
    return tuple(values)


def _has_invalid_text_tree(value: Any) -> bool:
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict):
        return any(
            not isinstance(key, str)
            or not key.strip()
            or _has_invalid_text_tree(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_has_invalid_text_tree(child) for child in value)
    return value is not None and not isinstance(value, (bool, int, float))


def _contains_prohibited_claim(
    text: str,
    prohibited_statements: tuple[str, ...],
) -> bool:
    normalized = normalize_text(text)
    if any(
        " ".join(statement.casefold().split()) in normalized
        for statement in prohibited_statements
    ):
        return True
    return claims_human_identity(normalized)
