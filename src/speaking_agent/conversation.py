from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
import hashlib
import re
from uuid import uuid4

from speaking_agent.campaign import Campaign
from speaking_agent.domain import (
    AgentReply,
    ConversationContext,
    ConversationStage,
    ConversationState,
    LeadOutcome,
    SessionAction,
)
from speaking_agent.model import ConversationModel, ConversationModelError
from speaking_agent.policy import ConversationPolicy
from speaking_agent.text_safety import normalize_match_text, normalize_text


class ConversationSession:
    def __init__(
        self,
        campaign: Campaign,
        model: ConversationModel,
        *,
        call_id: str | None = None,
        session_id: str | None = None,
        delivery_tracking: bool = False,
        context: ConversationContext | None = None,
    ) -> None:
        self.campaign = campaign
        self.model = model
        self.policy = ConversationPolicy(campaign)
        self.state = ConversationState(
            call_id=call_id or str(uuid4()),
            session_id=session_id or str(uuid4()),
            campaign_id=campaign.campaign_id,
        )
        self.context = context or ConversationContext()
        if self.context.recipient_name and not campaign.personalized_preamble:
            raise ValueError(
                "Campaign does not configure a personalized preamble"
            )
        self._apply_context_fields()
        opening_choices = (campaign.opening, *campaign.opening_variants)
        opening_index = int.from_bytes(
            hashlib.sha256(self.state.session_id.encode("utf-8")).digest()[:4],
            "big",
        ) % len(opening_choices)
        self._preamble_phase: str | None = None
        if self.context.recipient_name and campaign.personalized_preamble:
            self._preamble_phase = "recipient_confirmation"
            self.opening = campaign.personalized_preamble[
                "recipient_confirmation"
            ].format(recipient_name=self.context.recipient_name)
        else:
            self.opening = opening_choices[opening_index]
        self._recipient_confirmation_delivered = self._preamble_phase is None
        self._recipient_confirmation_attempts = 0
        self._delivery_tracking = delivery_tracking
        self._started = False

    @property
    def awaiting_recipient_confirmation(self) -> bool:
        return self._preamble_phase == "recipient_confirmation"

    @property
    def recipient_confirmation_delivered(self) -> bool:
        return self._recipient_confirmation_delivered

    @property
    def recipient_confirmation_issued(self) -> bool:
        return self._started and self.awaiting_recipient_confirmation

    def classify_recipient_confirmation(self, utterance: str) -> str:
        normalized = normalize_text(utterance).strip(".,!? ")
        if self._recipient_denied(normalized):
            return "denied"
        if self._recipient_confirmed(normalized):
            return "confirmed"
        if self.policy.is_confirmation_noise(utterance):
            return "noise"
        return "unknown"

    def classify_recipient_confirmation_overlap(self, utterance: str) -> str:
        normalized = normalize_text(utterance).strip(".,!? ")
        match_text = normalize_match_text(utterance)
        normalized_opening = normalize_match_text(self.opening)
        explicit_affirmative = (
            self._is_short_affirmative_confirmation(normalized)
            or re.fullmatch(
                r"(?:that's me|that is me|yes that's me|yes that is me)",
                normalized,
            )
            is not None
        )
        if explicit_affirmative:
            return "confirmed"
        if re.search(
            rf"(?:^|\s){re.escape(match_text)}(?:$|\s)",
            normalized_opening,
        ):
            return "noise"
        classification = self.classify_recipient_confirmation(utterance)
        if classification == "confirmed":
            return "noise"
        return classification

    def register_recipient_confirmation_noise(self) -> AgentReply | None:
        if not self.awaiting_recipient_confirmation:
            return None
        self._recipient_confirmation_attempts += 1
        maximum_attempts = int(
            self.campaign.behavior.get("max_unclear_retries", 2)
        )
        if self._recipient_confirmation_attempts >= maximum_attempts:
            return self._finish("UNKNOWN")
        return None

    def start(self, *, remember_reply: bool = True) -> AgentReply:
        if self._started:
            raise RuntimeError("Conversation session has already started")
        self._started = True
        self.state.stage = (
            ConversationStage.OPENING
            if self._preamble_phase is not None
            else ConversationStage.DISCOVERY
        )
        reply = AgentReply(
            self.opening,
            question_field=(
                None
                if self._preamble_phase is not None
                else self.campaign.opening_field
            ),
        )
        if remember_reply:
            self._remember_agent_reply(reply)
            if (
                self._preamble_phase == "recipient_confirmation"
                and not self._delivery_tracking
            ):
                self._recipient_confirmation_delivered = True
        return reply

    async def receive(
        self,
        utterance: str,
        *,
        captured_confirmation: bool = False,
    ) -> AgentReply:
        if not self._started:
            raise RuntimeError("Conversation session has not started")

        hard_stop_outcome = self.policy.hard_stop_outcome(utterance)
        captured_denial = (
            captured_confirmation
            and self.classify_recipient_confirmation(utterance) == "denied"
        )
        if (
            self.state.ended
            and hard_stop_outcome is None
            and not captured_denial
        ):
            raise RuntimeError("Conversation session has ended")
        prior_model_state = deepcopy(self.state)
        self._remember_owner_utterance(utterance)
        if hard_stop_outcome is not None:
            self.policy.apply_outcome(self.state, hard_stop_outcome)
            return self._finish(hard_stop_outcome)
        if captured_denial:
            return self._finish("WRONG_NUMBER")
        skipped_field = self.policy.explicitly_skipped_field(
            self.state,
            utterance,
        )
        if skipped_field is not None:
            self.state.skipped_fields.add(skipped_field)
            self.state.unclear_turns = 0
        direct_faq_response = self.policy.direct_faq_response(utterance)
        if direct_faq_response is not None:
            safe_faq_answer = self.policy.safe_response_prefix(
                direct_faq_response.answer
            )
            if safe_faq_answer is not None:
                if self._preamble_phase is not None:
                    if direct_faq_response.resume_objective:
                        prompt = self._current_preamble_prompt()
                        return self._remember_agent_reply(
                            AgentReply(f"{safe_faq_answer} {prompt}")
                        )
                    return self._remember_agent_reply(AgentReply(safe_faq_answer))
                if direct_faq_response.resume_objective:
                    return self._ask_next(safe_faq_answer)
                return self._remember_agent_reply(AgentReply(safe_faq_answer))
        if skipped_field is not None:
            contained_faq_response = self.policy.contained_faq_response(utterance)
            if contained_faq_response is not None:
                safe_faq_answer = self.policy.safe_response_prefix(
                    contained_faq_response.answer
                )
                if safe_faq_answer is not None:
                    if contained_faq_response.resume_objective:
                        return self._ask_next(safe_faq_answer)
                    return self._remember_agent_reply(AgentReply(safe_faq_answer))
        preamble_reply = self._handle_preamble(utterance)
        if preamble_reply is not None:
            return preamble_reply
        if self.policy.is_hesitation_fragment(utterance):
            return self._remember_agent_reply(
                AgentReply(self._hesitation_response())
            )

        model_state = deepcopy(self.state)
        model_state.recent_owner_utterances = (
            prior_model_state.recent_owner_utterances
        )
        model_state.recent_dialogue = prior_model_state.recent_dialogue
        timeout_seconds = float(self.campaign.behavior.get("model_timeout_seconds", 30))
        try:
            async with asyncio.timeout(timeout_seconds):
                interpretation = await self.model.interpret(
                    utterance,
                    model_state,
                    self.campaign,
                )
        except (TimeoutError, ConversationModelError):
            self.state.model_failures += 1
            maximum_failures = int(self.campaign.behavior.get("max_model_failures", 2))
            if self.state.model_failures >= maximum_failures:
                return self._finish("UNKNOWN")
            return self._ask_next(
                self.campaign.behavior.get(
                    "model_error_message",
                    "Sorry, I could not process that response.",
                )
            )

        self.state.model_failures = 0
        validated_outcome = self.policy.validated_outcome(
            self.state,
            interpretation.suggested_outcome,
            utterance,
        )
        deterministic_updates = self.policy.deterministic_field_updates(
            self.state,
            utterance,
        )
        pending_field = self.state.last_asked_field
        pending_value_before = (
            self.state.fields.get(pending_field)
            if pending_field is not None
            else None
        )
        interpretation = replace(
            interpretation,
            suggested_outcome=validated_outcome,
            field_updates={
                **interpretation.field_updates,
                **deterministic_updates,
            },
        )
        changed = self.policy.apply_interpretation(
            self.state,
            interpretation,
            utterance,
            trusted_field_names=set(deterministic_updates),
        )
        if interpretation.suggested_outcome in self.campaign.desired_outcomes:
            self.policy.apply_outcome(self.state, interpretation.suggested_outcome)

        owner_asked_question = self.policy.is_direct_question(utterance)
        answer = (
            self.policy.safe_response_prefix(interpretation.answer)
            if owner_asked_question
            else None
        )
        acknowledgement = self.policy.safe_acknowledgement(
            interpretation.acknowledgement,
            self.state.recent_dialogue,
        )
        response_prefix = answer or acknowledgement
        pending_field_answered = (
            pending_field is not None
            and self.state.fields.get(pending_field) not in (None, "")
            and self.state.fields.get(pending_field) != pending_value_before
        )
        if (
            pending_field_answered
            or skipped_field is not None
            or (changed and pending_field is None)
        ):
            self.state.unclear_turns = 0
        elif not owner_asked_question:
            self.state.unclear_turns += 1

        if self.state.outcome in self.campaign.terminal_outcomes:
            return self._finish(self.state.outcome)

        maximum_unclear_turns = int(self.campaign.behavior.get("max_unclear_retries", 2))
        if self.state.unclear_turns >= maximum_unclear_turns:
            unclear_field = self.state.last_asked_field
            if (
                self.state.outcome != "UNKNOWN"
                and self.campaign.behavior.get("allow_secondary_field_skips", False)
                and unclear_field is not None
                and unclear_field not in self.campaign.required_fields
            ):
                self.state.skipped_fields.add(unclear_field)
                self.state.unclear_turns = 0
            else:
                return self._finish("UNKNOWN")

        next_field = self.policy.next_missing_field(self.state)
        if next_field is None and self.state.outcome != "UNKNOWN":
            return self._finish(self.state.outcome)

        return self._ask_next(
            response_prefix,
            suggested_field=interpretation.next_question_field,
            suggested_question=interpretation.next_question,
        )

    def result(self) -> LeadOutcome:
        if not self.state.ended:
            raise RuntimeError("Conversation has not produced a final outcome")
        details = ", ".join(
            f"{name}={value}" for name, value in sorted(self.state.fields.items())
        )
        summary = f"Outcome: {self.state.outcome}."
        if details:
            summary = f"{summary} Collected: {details}."
        return LeadOutcome(
            outcome=self.state.outcome,
            qualified=self.state.outcome in self.campaign.qualified_outcomes,
            summary=summary,
            fields=dict(self.state.fields),
            callback_requested=self.state.callback_requested,
            human_followup_required=(
                self.state.outcome in self.campaign.human_followup_outcomes
            ),
        )

    def abort(self, outcome: str | None = None) -> None:
        if not self.state.ended:
            if outcome is not None:
                self.policy.apply_outcome(self.state, outcome)
            self.state.stage = ConversationStage.COMPLETED
            self.state.ended = True

    def _apply_context_fields(self) -> None:
        if not self.context.known_fields:
            return
        unknown_fields = self.context.known_fields.keys() - self.campaign.questions.keys()
        if unknown_fields:
            raise ValueError(
                "Conversation metadata contains unconfigured fields: "
                + ", ".join(sorted(unknown_fields))
            )
        if (
            self.campaign.outcome_field is not None
            and self.campaign.outcome_field in self.context.known_fields
        ):
            raise ValueError("Conversation metadata cannot set the campaign outcome field")
        for name, value in self.context.known_fields.items():
            field_type = self.campaign.field_types[name]
            valid_type = (
                (field_type == "string" and isinstance(value, str))
                or (field_type == "boolean" and isinstance(value, bool))
                or (
                    field_type == "number"
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                )
            )
            allowed_values = self.campaign.field_allowed_values.get(name)
            if not valid_type or (
                allowed_values is not None
                and isinstance(value, str)
                and value.casefold() not in allowed_values
            ):
                raise ValueError(
                    f"Conversation metadata field {name!r} is invalid for this campaign"
                )
            self.state.fields[name] = value

    def _handle_preamble(self, utterance: str) -> AgentReply | None:
        if self._preamble_phase is None:
            return None
        normalized = normalize_text(utterance).strip(".,!? ")
        if self._preamble_phase == "recipient_confirmation":
            classification = self.classify_recipient_confirmation(utterance)
            if classification == "denied":
                return self._finish("WRONG_NUMBER")
            if not self._recipient_confirmation_delivered:
                return self._remember_agent_reply(AgentReply(self.opening))
            if classification == "confirmed":
                self._preamble_phase = "property_timing"
                self._recipient_confirmation_attempts = 0
                return self._remember_agent_reply(
                    AgentReply(self._current_preamble_prompt())
                )
            maximum_attempts = int(
                self.campaign.behavior.get("max_unclear_retries", 2)
            )
            if classification == "noise":
                terminal_reply = self.register_recipient_confirmation_noise()
                if terminal_reply is not None:
                    return terminal_reply
                return self._remember_agent_reply(
                    AgentReply(
                        "Sorry, I didn't catch that. Are you "
                        f"{self.context.recipient_name}?"
                    )
                )
            self._recipient_confirmation_attempts += 1
            if self._recipient_confirmation_attempts >= maximum_attempts:
                return self._finish("UNKNOWN")
            return self._remember_agent_reply(
                AgentReply(
                    f"Sorry, am I speaking with {self.context.recipient_name}?"
                )
            )

        if self._bad_time(normalized):
            self._preamble_phase = None
            self.policy.apply_outcome(self.state, "CALLBACK")
            return self._ask_next("No problem.")
        if self._good_time(normalized):
            self._preamble_phase = None
            self.state.stage = ConversationStage.DISCOVERY
            faq_response = self.policy.contained_faq_response(utterance)
            if faq_response is not None:
                safe_answer = self.policy.safe_response_prefix(faq_response.answer)
                if safe_answer is not None:
                    if faq_response.resume_objective:
                        return self._ask_next(safe_answer)
                    return self._remember_agent_reply(AgentReply(safe_answer))
            if not self._timing_only_response(normalized):
                return None
            return self._remember_agent_reply(
                AgentReply(
                    self.campaign.personalized_preamble["qualification"],
                    question_field=self.campaign.opening_field,
                )
            )
        if re.search(r"\b(?:sell|selling|rent|renting|lease)\b", normalized):
            self._preamble_phase = None
            self.state.stage = ConversationStage.DISCOVERY
            return None
        return self._remember_agent_reply(
            AgentReply("Sorry, is this a good time for a quick call?")
        )

    @staticmethod
    def _timing_only_response(text: str) -> bool:
        return re.fullmatch(
            r"(?:(?:yes|yeah|yep|okay|ok|sure|fine)[, ]*)?"
            r"(?:it\s+is|it's)?\s*(?:a\s+)?good\s+time|"
            r"(?:yes|yeah|yep|okay|ok|sure|fine)(?:[, ]+go ahead)?|"
            r"go ahead|not at all|you can talk|i can talk",
            text,
        ) is not None

    def _hesitation_response(self) -> str:
        responses = (
            "Take your time. I'm listening.",
            "Go ahead.",
            "No rush.",
        )
        fragment_count = sum(
            self.policy.is_hesitation_fragment(turn)
            for turn in self.state.recent_owner_utterances[-4:]
        )
        return responses[max(0, fragment_count - 1) % len(responses)]

    def _current_preamble_prompt(self) -> str:
        if self._preamble_phase == "recipient_confirmation":
            return self.campaign.personalized_preamble[
                "recipient_confirmation"
            ].format(recipient_name=self.context.recipient_name)
        return self.campaign.personalized_preamble["property_timing"].format(
            property_reference=self.context.property_reference
        )

    def _recipient_confirmed(self, text: str) -> bool:
        if self._is_short_affirmative_confirmation(text) or re.fullmatch(
            r"(?:(?:yes|yeah|yep)[, ]*)?"
            r"(?:speaking|that's me|that is me)",
            text,
        ):
            return True
        name_tokens = self._recipient_name_tokens()
        if not name_tokens:
            return False
        identified_tokens = self._self_identified_name_tokens(text)
        if identified_tokens is not None:
            return name_tokens == identified_tokens
        return self._name_only_confirmation(text, name_tokens)

    @staticmethod
    def _is_short_affirmative_confirmation(text: str) -> bool:
        return re.fullmatch(
            r"(?:yes|yeah|yep)(?:[, ]+(?:i|i'm|im|my|me|ah|uh|um))?",
            text,
        ) is not None

    def _recipient_denied(self, text: str) -> bool:
        if re.match(r"^(?:no|nope)\b", text) or text == "not me" or re.search(
            r"\b(?:wrong person|not that person|you have the wrong)\b",
            text,
        ):
            return True
        name_tokens = self._recipient_name_tokens()
        if any(
            re.search(
                rf"\b(?:(?:not|isn't|is not)\s+"
                rf"(?:(?:mr|mrs|ms|miss|dr)\s+)?{re.escape(token)}|"
                rf"{re.escape(token)}\s+(?:isn't|is not|ain't)\s+"
                rf"(?:here|available))\b",
                text,
            )
            for token in name_tokens
        ):
            return True
        identified_tokens = self._self_identified_name_tokens(text)
        if identified_tokens is None:
            return False
        return name_tokens != identified_tokens

    @classmethod
    def _self_identified_name_tokens(cls, text: str) -> tuple[str, ...] | None:
        patterns = (
            r"^(?:(?:yes|yeah|yep|no|actually)[, ]+)?"
            r"(?:this is|i am|i'm)\s+"
            r"(?P<name>[^\r\n]{1,100}?)"
            r"(?:[, ]+speaking)?$",
            r"^(?:(?:yes|yeah|yep|no|actually)[, ]+)?"
            r"(?P<name>[^\r\n]{1,100}?)\s+speaking$",
        )
        for pattern in patterns:
            match = re.fullmatch(pattern, text)
            if match is None:
                continue
            tokens = cls._tokenize_name(match.group("name"))
            return tokens or None
        return None

    @classmethod
    def _name_only_confirmation(
        cls,
        text: str,
        name_tokens: tuple[str, ...],
    ) -> bool:
        return cls._tokenize_name(text) == name_tokens

    def _recipient_name_tokens(self) -> tuple[str, ...]:
        return self._tokenize_name(self.context.recipient_name or "")

    @staticmethod
    def _tokenize_name(value: str) -> tuple[str, ...]:
        tokens = tuple(
            re.findall(r"[^\W_]+", normalize_text(value), flags=re.UNICODE)
        )
        if tokens and tokens[0] in {"mr", "mrs", "ms", "miss", "dr"}:
            return tokens[1:]
        return tokens

    @staticmethod
    def _bad_time(text: str) -> bool:
        if re.search(
            r"\b(?:not busy|isn't a bad time|is not a bad time|"
            r"no need to call back|don't call back|do not call back)\b",
            text,
        ):
            return False
        return text in {"no", "nope", "not now"} or re.search(
            r"\b(?:busy|bad time|not a good time|call back|call later|another time)\b",
            text,
        ) is not None

    @staticmethod
    def _good_time(text: str) -> bool:
        return text in {"yes", "yeah", "yep", "okay", "ok", "sure", "fine"} or re.search(
            r"\b(?:go ahead|not at all|good time|you can talk|i can talk|"
            r"not busy|isn't a bad time|is not a bad time)\b",
            text,
        ) is not None

    def _record_question(self, field_name: str) -> None:
        self.state.asked_fields.add(field_name)
        self.state.asked_field_counts[field_name] = (
            self.state.asked_field_counts.get(field_name, 0) + 1
        )
        self.state.last_asked_field = field_name

    def _remember_owner_utterance(self, utterance: str) -> None:
        normalized = " ".join(utterance.split())
        if not normalized:
            return
        self.state.recent_owner_utterances.append(normalized)
        memory_turns = int(
            self.campaign.behavior.get("conversation_memory_turns", 12)
        )
        del self.state.recent_owner_utterances[:-memory_turns]
        self._remember_dialogue("owner", normalized)

    def _remember_agent_reply(self, reply: AgentReply) -> AgentReply:
        delivery = "pending" if self._delivery_tracking else "delivered"
        self._remember_dialogue(
            "agent",
            reply.text,
            delivery=delivery,
            question_field=reply.question_field,
        )
        if not self._delivery_tracking and reply.question_field is not None:
            self._record_question(reply.question_field)
        return reply

    def mark_agent_reply_started(self, reply: AgentReply) -> None:
        normalized_spoken_text = " ".join(reply.text.split())
        for turn in reversed(self.state.recent_dialogue):
            if turn.get("role") != "agent":
                continue
            remembered_text = turn["text"]
            if not (
                normalized_spoken_text == remembered_text
                or normalized_spoken_text.endswith(remembered_text)
            ):
                continue
            if reply.question_field is not None and not turn.get("question_started"):
                self._record_question(reply.question_field)
                turn["question_started"] = True
            return

    def mark_agent_reply_delivery(self, spoken_text: str, delivery: str) -> None:
        if delivery not in {"pending", "interrupted", "delivered"}:
            raise ValueError("Unsupported agent reply delivery state")
        normalized_spoken_text = " ".join(spoken_text.split())
        for turn in reversed(self.state.recent_dialogue):
            if turn.get("role") != "agent":
                continue
            remembered_text = turn["text"]
            if (
                normalized_spoken_text == remembered_text
                or normalized_spoken_text.endswith(remembered_text)
            ):
                turn["delivery"] = delivery
                if (
                    self._preamble_phase == "recipient_confirmation"
                    and delivery == "delivered"
                    and (
                        normalized_spoken_text == " ".join(self.opening.split())
                        or normalized_spoken_text.endswith(
                            " ".join(self.opening.split())
                        )
                    )
                ):
                    self._recipient_confirmation_delivered = True
                return

    def _remember_dialogue(
        self,
        role: str,
        text: str,
        *,
        delivery: str | None = None,
        question_field: str | None = None,
    ) -> None:
        normalized = " ".join(text.split())
        if not normalized:
            return
        turn = {"role": role, "text": normalized}
        if delivery is not None:
            turn["delivery"] = delivery
        if question_field is not None:
            turn["question_field"] = question_field
        self.state.recent_dialogue.append(turn)
        memory_turns = int(
            self.campaign.behavior.get("conversation_memory_turns", 12)
        )
        maximum_entries = memory_turns * 2 + 1
        del self.state.recent_dialogue[:-maximum_entries]

    def _ask_next(
        self,
        prefix: str | None = None,
        *,
        suggested_field: str | None = None,
        suggested_question: str | None = None,
    ) -> AgentReply:
        next_field = self.policy.next_missing_field(self.state)
        if next_field is None:
            next_field = self.campaign.opening_field
        self.state.stage = (
            ConversationStage.DISCOVERY
            if next_field == self.campaign.opening_field
            else ConversationStage.QUALIFICATION
        )
        question = self._question_for(
            next_field,
            suggested_field=suggested_field,
            suggested_question=suggested_question,
        )
        return self._remember_agent_reply(
            AgentReply(
                f"{prefix.strip()} {question}" if prefix else question,
                question_field=next_field,
            )
        )

    def _question_for(
        self,
        field_name: str,
        *,
        suggested_field: str | None = None,
        suggested_question: str | None = None,
    ) -> str:
        dynamic_question = self.policy.safe_dynamic_question(
            field_name,
            suggested_field,
            suggested_question,
            self.state.recent_dialogue,
        )
        if dynamic_question is not None:
            return dynamic_question
        previous_asks = self.state.asked_field_counts.get(field_name, 0)
        variants = self.campaign.question_variants.get(field_name, ())
        if previous_asks == 0 or not variants:
            return self.campaign.questions[field_name]
        return variants[(previous_asks - 1) % len(variants)]

    def _finish(self, outcome: str) -> AgentReply:
        self.policy.apply_outcome(self.state, outcome)
        effective_outcome = self.state.outcome
        self.state.stage = ConversationStage.COMPLETED
        self.state.ended = True
        action = (
            SessionAction.TRANSFER
            if self.state.human_transfer_requested
            else SessionAction.HANG_UP
        )
        message = self.campaign.closing_messages.get(
            effective_outcome,
            "Thank you for your time. Goodbye.",
        )
        return self._remember_agent_reply(AgentReply(message, action))
