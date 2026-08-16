from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any

from speaking_agent.campaign import Campaign
from speaking_agent.domain import ConversationState
from speaking_agent.model import ModelInterpretation
from speaking_agent.text_safety import claims_human_identity, normalize_text


def _normalized(text: str) -> str:
    return normalize_text(text)


class ConversationPolicy:
    def __init__(self, campaign: Campaign) -> None:
        self.campaign = campaign

    def hard_stop_outcome(self, utterance: str) -> str | None:
        normalized_utterance = _normalized(utterance)
        dnc_matches = any(
            self._contains_phrase(normalized_utterance, phrase)
            for phrase in self.campaign.hard_stop_phrases.get("DO_NOT_CONTACT", ())
        )
        permanent_dnc = self._has_permanent_dnc_signal(normalized_utterance)
        if permanent_dnc or (
            dnc_matches and not self._is_temporary_contact_pause(normalized_utterance)
        ):
            return "DO_NOT_CONTACT"
        if (
            "HUMAN_TRANSFER" in self.campaign.desired_outcomes
            and self._is_explicit_transfer_request(normalized_utterance)
        ):
            return "HUMAN_TRANSFER"
        for outcome in self.campaign.hard_stop_phrases:
            if outcome == "DO_NOT_CONTACT":
                continue
            phrases = self.campaign.hard_stop_phrases.get(outcome, ())
            if not any(
                self._contains_phrase(normalized_utterance, phrase)
                for phrase in phrases
            ):
                continue
            if outcome == "NOT_INTERESTED" and re.search(
                r"\bbut\b.*\b(?:sell|selling|rent|renting|lease)\b",
                normalized_utterance,
            ):
                continue
            return outcome
        return None

    def direct_faq_answer(self, utterance: str) -> str | None:
        normalized_utterance = _normalized(utterance).strip(".,!? ")
        if re.search(
            r"\b(?:call me back|callback|do not call|don't call|"
            r"speak to (?:a )?(?:person|human|agent)|transfer me)\b",
            normalized_utterance,
        ):
            return None
        for question, answer in self.campaign.faq_answers.items():
            normalized_question = _normalized(question).strip(".,!? ")
            if normalized_utterance == normalized_question:
                return answer
        return None

    def apply_interpretation(
        self,
        state: ConversationState,
        interpretation: ModelInterpretation,
        utterance: str = "",
        trusted_field_names: set[str] | None = None,
    ) -> bool:
        trusted_field_names = trusted_field_names or set()
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
                and (
                    name in trusted_field_names
                    or self._field_value_supported_by_utterance(
                        name,
                        value,
                        utterance,
                    )
                )
            ):
                if state.fields.get(name) != value:
                    state.fields[name] = value
                    changed = True

        if (
            interpretation.callback_requested is not None
            and state.callback_requested != interpretation.callback_requested
            and self._callback_change_supported(
                state,
                interpretation,
                utterance,
            )
        ):
            state.callback_requested = interpretation.callback_requested
            changed = True
        return changed

    def _callback_change_supported(
        self,
        state: ConversationState,
        interpretation: ModelInterpretation,
        utterance: str,
    ) -> bool:
        normalized_utterance = _normalized(utterance)
        if interpretation.callback_requested:
            return re.search(
                r"\b(?:call me back|callback|call me later|call me tomorrow|"
                r"contact me later|contact me tomorrow)\b",
                normalized_utterance,
            ) is not None
        if not state.callback_requested:
            return False
        if re.search(
            r"\b(?:no|cancel|don't|do not|no need)\b.{0,25}"
            r"\b(?:callback|call back|call me|contact me)\b",
            normalized_utterance,
        ):
            return True
        return interpretation.suggested_outcome in {
            "SELL",
            "RENT",
            "SELL_OR_RENT",
            "FUTURE",
        }

    def _field_value_supported_by_utterance(
        self,
        name: str,
        value: Any,
        utterance: str,
    ) -> bool:
        normalized_utterance = _normalized(utterance)
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return self._contains_phrase(normalized_utterance, str(value))
        if not isinstance(value, str):
            return False
        normalized_value = _normalized(value).strip(".,!? ")
        if name == "property_location":
            if re.search(
                r"\b(?:not|isn't|is not|aren't|are not|wasn't|was not|"
                r"don't|do not|doesn't|does not|rather not say|prefer not to say|"
                r"don't want to say|do not want to say|won't say|will not say|"
                r"not sharing)\b",
                normalized_value,
            ):
                return False
            return (
                self._contains_phrase(normalized_utterance, normalized_value)
                and not self._phrase_is_negated(
                    normalized_utterance,
                    normalized_value,
                )
            )
        if re.search(
            r"\b(?:not|isn't|is not|aren't|are not|wasn't|was not|"
            r"don't|do not|doesn't|does not)\b",
            normalized_value,
        ):
            return False
        return (
            self._contains_phrase(normalized_utterance, normalized_value)
            and not self._phrase_is_negated(normalized_utterance, normalized_value)
        )

    def deterministic_field_updates(
        self,
        state: ConversationState,
        utterance: str,
    ) -> dict[str, Any]:
        normalized_utterance = _normalized(utterance)
        updates: dict[str, Any] = {}
        for field_name, allowed_values in self.campaign.field_allowed_values.items():
            matches = [
                value
                for value in sorted(allowed_values, key=len, reverse=True)
                if re.search(
                    rf"(?<!\w){re.escape(_normalized(value))}(?!\w)",
                    normalized_utterance,
                )
                and not self._phrase_is_negated(normalized_utterance, value)
            ]
            longest_matches = [
                value
                for value in matches
                if not any(
                    value != other and value in other
                    for other in matches
                )
            ]
            if len(longest_matches) == 1:
                updates[field_name] = longest_matches[0]

        for field_name, hints in self.campaign.field_extraction_hints.items():
            matches = [
                hint
                for hint in sorted(hints, key=len, reverse=True)
                if self._contains_phrase(normalized_utterance, hint)
                and not self._phrase_is_negated(normalized_utterance, hint)
            ]
            if len(matches) == 1 and self._has_meaningful_field_value(
                field_name,
                matches[0],
            ):
                updates[field_name] = matches[0]

        timeline_match = re.search(
            r"\b(?:(?:next|this)\s+(?:week|month|year)|"
            r"(?:in|within)\s+(?:about\s+)?(?:\d+|one|two|three|four|five|six)\s+"
            r"(?:days?|weeks?|months?|years?)|soon|immediately)\b",
            normalized_utterance,
        )
        if timeline_match:
            timeline = timeline_match.group(0)
            explicit_outcome = self._explicit_property_outcome(normalized_utterance)
            if explicit_outcome == "SELL" and "selling_timeline" in self.campaign.questions:
                updates["selling_timeline"] = timeline
            elif explicit_outcome == "RENT" and "availability_date" in self.campaign.questions:
                updates["availability_date"] = timeline

        last_field = state.last_asked_field
        if last_field and self.campaign.field_types.get(last_field) == "boolean":
            yes_answers = {"yes", "yes it is", "it is", "yeah", "yep"}
            no_answers = {"no", "no it isn't", "it isn't", "not yet", "nope"}
            if normalized_utterance in yes_answers:
                updates[last_field] = True
            elif normalized_utterance in no_answers:
                updates[last_field] = False
        if "currently_listed" in self.campaign.questions:
            if re.search(
                r"\b(?:not listed|isn't listed|is not listed|"
                r"not on the market|isn't on the market|is not on the market)\b",
                normalized_utterance,
            ):
                updates["currently_listed"] = False
            elif re.search(
                r"\b(?:already listed|currently listed|listed with|"
                r"on the market)\b",
                normalized_utterance,
            ):
                updates["currently_listed"] = True
        return updates

    def validated_outcome(
        self,
        state: ConversationState,
        suggested_outcome: str | None,
        utterance: str,
    ) -> str | None:
        if suggested_outcome not in self.campaign.desired_outcomes:
            return None
        normalized_utterance = _normalized(utterance)
        if (
            state.last_asked_field == self.campaign.opening_field
            and normalized_utterance in {
                "both",
                "both options",
                "either",
                "either option",
                "whichever",
            }
            and "SELL_OR_RENT" in self.campaign.desired_outcomes
        ):
            return "SELL_OR_RENT"
        if (
            self._is_temporary_contact_pause(normalized_utterance)
            and not self._has_permanent_dnc_signal(normalized_utterance)
        ):
            if "CALLBACK" in self.campaign.desired_outcomes and re.search(
                r"\b(?:call|contact)\b.*\b(?:later|tomorrow|next|after|until|before)\b|"
                r"\b(?:later|tomorrow|next\s+\w+)\b.*\b(?:call|contact)\b",
                normalized_utterance,
            ):
                return "CALLBACK"
            if suggested_outcome == "DO_NOT_CONTACT":
                return None
        if self._is_explicit_transfer_request(normalized_utterance):
            return (
                "HUMAN_TRANSFER"
                if "HUMAN_TRANSFER" in self.campaign.desired_outcomes
                else None
            )
        if suggested_outcome == "DO_NOT_CONTACT":
            permanent_dnc = self._has_permanent_dnc_signal(normalized_utterance)
            configured_dnc = any(
                self._contains_phrase(normalized_utterance, phrase)
                for phrase in self.campaign.hard_stop_phrases.get(
                    "DO_NOT_CONTACT",
                    (),
                )
            )
            return (
                "DO_NOT_CONTACT"
                if permanent_dnc
                or (
                    configured_dnc
                    and not self._is_temporary_contact_pause(normalized_utterance)
                )
                else None
            )
        explicit_property_outcome = self._explicit_property_outcome(
            normalized_utterance
        )
        if explicit_property_outcome in self.campaign.desired_outcomes:
            return explicit_property_outcome
        if suggested_outcome == state.outcome:
            return suggested_outcome

        property_outcomes = {"SELL", "RENT", "SELL_OR_RENT", "FUTURE"}
        if state.outcome in property_outcomes and suggested_outcome == "CALLBACK":
            return state.outcome
        if suggested_outcome in {"SELL", "RENT", "SELL_OR_RENT"}:
            return None
        if suggested_outcome == "HUMAN_TRANSFER":
            return None
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
            "CALLBACK": (
                "call back",
                "callback",
                "call me later",
                "call me tomorrow",
            ),
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
                if any(
                    self._contains_phrase(normalized_utterance, cue)
                    for cue in combined_cues
                )
                else None
            )
        required_cues = evidence.get(suggested_outcome)
        if required_cues is not None and not any(
            self._contains_phrase(normalized_utterance, cue)
            for cue in required_cues
        ):
            return None
        return suggested_outcome

    @classmethod
    def _explicit_property_outcome(cls, normalized_utterance: str) -> str | None:
        if cls._contains_phrase(normalized_utterance, "neither"):
            return None
        combined_cues = (
            "both",
            "either",
            "whichever",
            "sell or rent",
            "selling or renting",
            "selling and renting",
        )
        sell_cues = ("sell", "selling", "sale", "put it on the market")
        rent_cues = ("rent", "renting", "lease", "tenant")
        has_sell = any(
            cls._contains_phrase(normalized_utterance, cue)
            and not cls._phrase_is_negated(normalized_utterance, cue)
            for cue in sell_cues
        )
        has_rent = any(
            cls._contains_phrase(normalized_utterance, cue)
            and not cls._phrase_is_negated(normalized_utterance, cue)
            for cue in rent_cues
        )
        if has_sell and has_rent and any(
            cls._contains_phrase(normalized_utterance, cue)
            for cue in combined_cues
        ):
            return "SELL_OR_RENT"
        if has_sell and has_rent:
            return "SELL_OR_RENT"
        if has_sell:
            return "SELL"
        if has_rent:
            return "RENT"
        return None

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
        if claims_human_identity(normalized_answer):
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

    def safe_acknowledgement(
        self,
        acknowledgement: str | None,
        recent_dialogue: list[dict[str, str]],
    ) -> str | None:
        safe_acknowledgement = self.safe_response_prefix(acknowledgement)
        if safe_acknowledgement is None:
            return None
        normalized_acknowledgement = _normalized(safe_acknowledgement).strip(".,!? ")
        recent_openings = {
            _normalized(re.split(r"(?<=[.!?])\s+", turn["text"], maxsplit=1)[0])
            .strip(".,!? ")
            for turn in recent_dialogue[-6:]
            if turn.get("role") == "agent"
        }
        if normalized_acknowledgement in recent_openings:
            return None
        excessive_praise = {"amazing", "excellent", "fantastic", "perfect", "wonderful"}
        if normalized_acknowledgement in excessive_praise:
            return None
        return safe_acknowledgement

    def safe_dynamic_question(
        self,
        field_name: str,
        suggested_field: str | None,
        question: str | None,
        recent_dialogue: list[dict[str, str]],
    ) -> str | None:
        if suggested_field != field_name or not question:
            return None
        candidate = " ".join(question.split())
        if (
            not candidate.endswith("?")
            or candidate.count("?") != 1
            or len(candidate.split()) > 30
            or self.safe_answer(candidate) is None
        ):
            return None
        allowed_questions = (
            self.campaign.questions[field_name],
            *self.campaign.question_variants.get(field_name, ()),
        )
        if _normalized(candidate) not in {
            _normalized(allowed_question)
            for allowed_question in allowed_questions
        }:
            return None
        normalized_candidate = _normalized(candidate)
        recent_agent_text = (
            _normalized(turn["text"])
            for turn in recent_dialogue[-8:]
            if turn.get("role") == "agent"
        )
        if any(normalized_candidate in text for text in recent_agent_text):
            return None
        return candidate

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
            "i would rather not say",
            "rather not say",
            "i prefer not to say",
            "prefer not to say",
            "i don't want to say",
            "i do not want to say",
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
        if name == "property_location" and re.search(
            r"\b(?:address|name|location|here|there)\b",
            normalized_value,
        ):
            return False
        return (
            normalized_value not in generic_non_values
            and normalized_value not in field_non_values.get(name, set())
            and (allowed_values is None or normalized_value in allowed_values)
        )

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        return re.search(
            rf"(?<!\w){re.escape(_normalized(phrase))}(?!\w)",
            text,
        ) is not None

    @staticmethod
    def _is_temporary_contact_pause(text: str) -> bool:
        explicit_pause = re.search(
            r"\b(?:do not|don't|dont)\s+(?:call|contact)(?:\s+me)?(?:\s+again)?\s+"
            r"(?:now|right now|today|at the moment|later|"
            r"until\s+(?:tomorrow|next\s+\w+|\d{1,2}(?::\d{2})?\s*(?:am|pm)?)|"
            r"before\s+(?:tomorrow|next\s+\w+|\d{1,2}(?::\d{2})?\s*(?:am|pm)?)|"
            r"for\s+(?:\d+|one|two|three|four|five|six)\s+"
            r"(?:minutes?|hours?|days?|weeks?))\b",
            text,
        )
        compact_pause = re.search(
            r"\bnot\s+(?:right\s+)?now\b.*\b(?:call|contact)(?:\s+me)?\b.*"
            r"\b(?:later|tomorrow|next\s+\w+|after\s+\w+)\b",
            text,
        )
        return explicit_pause is not None or compact_pause is not None

    @staticmethod
    def _has_permanent_dnc_signal(text: str) -> bool:
        return re.search(
            r"\b(?:ever again|never (?:call|contact)|"
            r"(?:do not|don't|dont)\s+(?:call|contact)(?:\s+me)?\s+again|"
            r"remove (?:my number|me from|me off)|"
            r"take me off|stop (?:calling|contacting)|delete my number)\b",
            text,
        ) is not None

    @staticmethod
    def _is_explicit_transfer_request(text: str) -> bool:
        return re.search(
            r"^(?:please\s+)?transfer(?:\s+me)?$|"
            r"\b(?:transfer|connect)\s+me\b|"
            r"\b(?:speak|talk)\s+(?:to|with)\s+(?:a\s+)?"
            r"(?:person|human|agent|specialist|someone)\b|"
            r"\bconnect\s+me\s+(?:to|with)\s+(?:a\s+)?(?:person|human|agent|specialist|someone)\b",
            text,
        ) is not None

    @classmethod
    def _phrase_is_negated(cls, text: str, phrase: str) -> bool:
        phrase_match = re.search(
            rf"(?<!\w){re.escape(_normalized(phrase))}(?!\w)",
            text,
        )
        if phrase_match is None:
            return False
        prefix = text[max(0, phrase_match.start() - 80) : phrase_match.start()]
        prefix = re.split(
            r"\b(?:but|however|though|instead|rather)\b|[,;]|"
            r"\b(?:and\s+)?i\s+(?:want|plan|intend|hope|prefer)\s+to\b|"
            r"\b(?:and\s+)?i(?:'d| would)\s+(?:like|rather)\s+to\b|"
            r"\b(?:and\s+)?i(?:'m| am)\s+|"
            r"\b(?:and\s+)?(?:it|this|that|the property)\s+(?:is|was)\b",
            prefix,
        )[-1]
        return re.search(
            r"\b(?:not|never|isn't|is not|aren't|are not|wasn't|was not|"
            r"don't|do not|doesn't|does not|wouldn't|would not|won't|will not|"
            r"outside|away from)\b[^,.!?;]*$",
            prefix,
        ) is not None

    @staticmethod
    def _deduplicated(values: Iterable[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(values))


def has_meaningful_value(value: Any) -> bool:
    return value is not None and value != ""
