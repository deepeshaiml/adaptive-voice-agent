from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
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
    ) -> None:
        self.campaign = campaign
        self.model = model
        self.policy = ConversationPolicy(campaign)
        self.state = ConversationState(
            call_id=call_id or str(uuid4()),
            session_id=session_id or str(uuid4()),
            campaign_id=campaign.campaign_id,
        )
        self._started = False

    def start(self) -> AgentReply:
        if self._started:
            raise RuntimeError("Conversation session has already started")
        self._started = True
        self.state.stage = ConversationStage.DISCOVERY
        self._record_question(self.campaign.opening_field)
        return AgentReply(self.campaign.opening)

    async def receive(self, utterance: str) -> AgentReply:
        if not self._started:
            raise RuntimeError("Conversation session has not started")

        hard_stop_outcome = self.policy.hard_stop_outcome(utterance)
        if hard_stop_outcome is not None:
            self.policy.apply_outcome(self.state, hard_stop_outcome)
            return self._finish(hard_stop_outcome)
        if self.state.ended:
            raise RuntimeError("Conversation session has ended")

        timeout_seconds = float(self.campaign.behavior.get("model_timeout_seconds", 30))
        try:
            async with asyncio.timeout(timeout_seconds):
                interpretation = await self.model.interpret(
                    utterance,
                    deepcopy(self.state),
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
        interpretation = replace(
            interpretation,
            suggested_outcome=validated_outcome,
        )
        changed = self.policy.apply_interpretation(self.state, interpretation)
        if interpretation.suggested_outcome in self.campaign.desired_outcomes:
            self.policy.apply_outcome(self.state, interpretation.suggested_outcome)

        answer = self.policy.safe_response_prefix(interpretation.answer)
        acknowledgement = self.policy.safe_response_prefix(
            interpretation.acknowledgement
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

        return self._ask_next(response_prefix)

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

    def _ask_next(self, prefix: str | None = None) -> AgentReply:
        next_field = self.policy.next_missing_field(self.state)
        if next_field is None:
            next_field = self.campaign.opening_field
        self.state.stage = (
            ConversationStage.DISCOVERY
            if next_field == self.campaign.opening_field
            else ConversationStage.QUALIFICATION
        )
        question = self._question_for(next_field)
        self._record_question(next_field)
        return AgentReply(f"{prefix.strip()} {question}" if prefix else question)

    def _question_for(self, field_name: str) -> str:
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
        return AgentReply(message, action)
