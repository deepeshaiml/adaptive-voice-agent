from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any

from speaking_agent.campaign import Campaign
from speaking_agent.domain import ConversationState
from speaking_agent.model import ModelInterpretation


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


class ConversationPolicy:
    def __init__(self, campaign: Campaign) -> None:
        self.campaign = campaign

    def hard_stop_outcome(self, utterance: str) -> str | None:
        normalized_utterance = _normalized(utterance)
        outcomes = (
            "DO_NOT_CONTACT",
            *(
                outcome
                for outcome in self.campaign.hard_stop_phrases
                if outcome != "DO_NOT_CONTACT"
            ),
        )
        for outcome in outcomes:
            phrases = self.campaign.hard_stop_phrases.get(outcome, ())
            if any(_normalized(phrase) in normalized_utterance for phrase in phrases):
                return outcome
        return None

    def apply_interpretation(
        self,
        state: ConversationState,
        interpretation: ModelInterpretation,
    ) -> bool:
        changed = False
        if interpretation.suggested_outcome in self.campaign.desired_outcomes:
            if state.outcome != interpretation.suggested_outcome:
                state.outcome = interpretation.suggested_outcome
                changed = True
            if self.campaign.outcome_field is not None:
                if state.fields.get(self.campaign.outcome_field) != state.outcome:
                    state.fields[self.campaign.outcome_field] = state.outcome
                    changed = True

        allowed_fields = self.campaign.questions.keys()
        for name, value in interpretation.field_updates.items():
            if (
                name in allowed_fields
                and name != self.campaign.outcome_field
                and value not in (None, "")
                and self._has_configured_type(name, value)
                and self._has_meaningful_field_value(name, value)
            ):
                if state.fields.get(name) != value:
                    state.fields[name] = value
                    changed = True

        if (
            interpretation.callback_requested is not None
            and state.callback_requested != interpretation.callback_requested
        ):
            state.callback_requested = interpretation.callback_requested
            changed = True
        return changed

    def validated_outcome(
        self,
        state: ConversationState,
        suggested_outcome: str | None,
        utterance: str,
    ) -> str | None:
        if suggested_outcome not in self.campaign.desired_outcomes:
            return None
        if suggested_outcome == state.outcome:
            return suggested_outcome

        normalized_utterance = _normalized(utterance)
        evidence = {
            "SELL": ("sell", "selling", "sale", "put it on the market"),
            "RENT": ("rent", "renting", "lease", "tenant"),
            "FUTURE": (
                "later",
                "future",
                "next year",
                "not right now",
                "not decided",
            ),
            "CALLBACK": ("call back", "callback", "call me later"),
        }
        if suggested_outcome == "SELL_OR_RENT":
            combined_cues = (
                "both",
                "either",
                "whichever",
                "sell or rent",
                "selling or renting",
                "selling and renting",
            )
            return (
                suggested_outcome
                if any(cue in normalized_utterance for cue in combined_cues)
                else None
            )
        required_cues = evidence.get(suggested_outcome)
        if required_cues is not None and not any(
            cue in normalized_utterance for cue in required_cues
        ):
            return None
        return suggested_outcome

    def apply_outcome(self, state: ConversationState, outcome: str) -> None:
        if state.do_not_contact and outcome != "DO_NOT_CONTACT":
            return
        state.outcome = outcome
        if self.campaign.outcome_field is not None:
            state.fields[self.campaign.outcome_field] = outcome
        if outcome == "DO_NOT_CONTACT":
            state.do_not_contact = True
            state.callback_requested = False
            state.human_transfer_requested = False
        elif outcome == "CALLBACK":
            state.callback_requested = True
        elif outcome == "HUMAN_TRANSFER":
            state.human_transfer_requested = True

    def next_missing_field(self, state: ConversationState) -> str | None:
        fields = self._deduplicated(
            (*self.campaign.required_fields, *self.campaign.fields_by_outcome.get(state.outcome, ()))
        )
        return next(
            (
                field_name
                for field_name in fields
                if field_name not in state.skipped_fields
                and (
                    field_name not in state.fields
                    or state.fields[field_name] in (None, "")
                )
            ),
            None,
        )

    def safe_answer(self, answer: str | None) -> str | None:
        if not answer:
            return None
        normalized_answer = _normalized(answer)
        if any(
            _normalized(statement) in normalized_answer
            for statement in self.campaign.prohibited_statements
        ):
            return None
        return answer.strip()

    def safe_response_prefix(self, response: str | None) -> str | None:
        safe_response = self.safe_answer(response)
        if safe_response is None:
            return None
        statements = []
        for sentence in re.split(r"(?<=[.!?])\s+", safe_response):
            if "?" in sentence:
                break
            statements.append(sentence)
        prefix = " ".join(statements).strip()
        return prefix or None

    def _has_configured_type(self, name: str, value: Any) -> bool:
        field_type = self.campaign.field_types[name]
        if field_type == "string":
            return isinstance(value, str)
        if field_type == "boolean":
            return isinstance(value, bool)
        if field_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        return False

    def _has_meaningful_field_value(self, name: str, value: Any) -> bool:
        if not isinstance(value, str):
            return True
        normalized_value = _normalized(value).strip(".,!? ")
        generic_non_values = {
            "don't know",
            "i don't know",
            "not sure",
            "something",
            "something else",
            "other",
            "unknown",
            "you know",
        }
        field_non_values = {
            "property_location": {
                "address",
                "my address",
                "the address",
                "my location",
                "the location",
                "here",
                "there",
                "same address",
                "you should know",
            },
            "property_type": {
                "property",
                "place",
                "thing",
            },
        }
        allowed_values = self.campaign.field_allowed_values.get(name)
        return (
            normalized_value not in generic_non_values
            and normalized_value not in field_non_values.get(name, set())
            and (allowed_values is None or normalized_value in allowed_values)
        )

    @staticmethod
    def _deduplicated(values: Iterable[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(values))


def has_meaningful_value(value: Any) -> bool:
    return value is not None and value != ""
