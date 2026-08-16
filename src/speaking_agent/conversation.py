from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
import hashlib
from uuid import uuid4

from speaking_agent.campaign import Campaign
from speaking_agent.domain import (
    AgentReply,
    ConversationStage,
    ConversationState,
    LeadOutcome,
    SessionAction,
)
from speaking_agent.model import ConversationModel, ConversationModelError
from speaking_agent.policy import ConversationPolicy


class ConversationSession:
    def __init__(
        self,
        campaign: Campaign,
        model: ConversationModel,
        *,
        call_id: str | None = None,
        session_id: str | None = None,
        delivery_tracking: bool = False,
    ) -> None:
        self.campaign = campaign
        self.model = model
        self.policy = ConversationPolicy(campaign)
        self.state = ConversationState(
            call_id=call_id or str(uuid4()),
            session_id=session_id or str(uuid4()),
            campaign_id=campaign.campaign_id,
        )
        opening_choices = (campaign.opening, *campaign.opening_variants)
        opening_index = int.from_bytes(
            hashlib.sha256(self.state.session_id.encode("utf-8")).digest()[:4],
            "big",
        ) % len(opening_choices)
        self.opening = opening_choices[opening_index]
        self._delivery_tracking = delivery_tracking
        self._started = False

    def start(self, *, remember_reply: bool = True) -> AgentReply:
        if self._started:
            raise RuntimeError("Conversation session has already started")
        self._started = True
        self.state.stage = ConversationStage.DISCOVERY
        reply = AgentReply(
            self.opening,
            question_field=self.campaign.opening_field,
        )
        if remember_reply:
            self._remember_agent_reply(reply)
        return reply

    async def receive(self, utterance: str) -> AgentReply:
        if not self._started:
            raise RuntimeError("Conversation session has not started")

        hard_stop_outcome = self.policy.hard_stop_outcome(utterance)
        if self.state.ended and hard_stop_outcome is None:
            raise RuntimeError("Conversation session has ended")
        model_state = deepcopy(self.state)
        self._remember_owner_utterance(utterance)
        if hard_stop_outcome is not None:
            self.policy.apply_outcome(self.state, hard_stop_outcome)
            return self._finish(hard_stop_outcome)
        direct_faq_answer = self.policy.direct_faq_answer(utterance)
        if direct_faq_answer is not None:
            safe_faq_answer = self.policy.safe_response_prefix(direct_faq_answer)
            if safe_faq_answer is not None:
                return self._ask_next(safe_faq_answer)

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

        answer = self.policy.safe_response_prefix(interpretation.answer)
        acknowledgement = self.policy.safe_acknowledgement(
            interpretation.acknowledgement,
            self.state.recent_dialogue,
        )
        response_prefix = answer or acknowledgement
        if changed:
            self.state.unclear_turns = 0
        elif answer is None:
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
