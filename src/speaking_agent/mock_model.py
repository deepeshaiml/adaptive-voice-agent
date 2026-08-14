from __future__ import annotations

import re
from collections.abc import Mapping

from speaking_agent.campaign import Campaign
from speaking_agent.domain import ConversationState
from speaking_agent.model import ModelInterpretation


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


class MockConversationModel:
    """Deterministic model double with a small fallback parser for manual demos."""

    def __init__(
        self,
        scripted: Mapping[str, ModelInterpretation] | None = None,
    ) -> None:
        self._scripted = {
            _normalized(utterance): interpretation
            for utterance, interpretation in (scripted or {}).items()
        }

    async def prepare(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def interpret(
        self,
        utterance: str,
        state: ConversationState,
        campaign: Campaign,
    ) -> ModelInterpretation:
        normalized_utterance = _normalized(utterance)
        if normalized_utterance in self._scripted:
            return self._scripted[normalized_utterance]

        outcome = self._infer_outcome(normalized_utterance, state, campaign)
        answer = next(
            (
                response
                for phrase, response in campaign.faq_answers.items()
                if _normalized(phrase) in normalized_utterance
            ),
            None,
        )
        field_updates = self._infer_fields(
            utterance,
            normalized_utterance,
            state,
            campaign,
            allow_pending_field_fallback=answer is None and outcome is None,
        )
        if outcome is not None and "intent" in campaign.questions:
            field_updates["intent"] = outcome

        return ModelInterpretation(
            suggested_outcome=outcome,
            field_updates=field_updates,
            answer=answer,
            callback_requested=True if outcome == "CALLBACK" else None,
            human_transfer_requested=outcome == "HUMAN_TRANSFER",
        )

    @staticmethod
    def _infer_outcome(
        text: str,
        state: ConversationState,
        campaign: Campaign,
    ) -> str | None:
        if (
            "SELL_OR_RENT" in campaign.desired_outcomes
            and state.last_asked_field == campaign.opening_field
            and (
                text in {"both", "either", "either one", "both options"}
                or ("sell" in text and "rent" in text)
            )
        ):
            return "SELL_OR_RENT"
        cues = (
            ("HUMAN_TRANSFER", ("human", "person", "someone", "agent")),
            ("CALLBACK", ("call me back", "callback", "call later")),
            ("SELL", ("sell", "sale", "selling")),
            ("RENT", ("rent", "lease", "letting")),
            ("FUTURE", ("maybe later", "not right now", "in the future")),
        )
        for outcome, phrases in cues:
            if outcome in campaign.desired_outcomes and any(phrase in text for phrase in phrases):
                return outcome
        return None

    @staticmethod
    def _infer_fields(
        utterance: str,
        normalized_utterance: str,
        state: ConversationState,
        campaign: Campaign,
        *,
        allow_pending_field_fallback: bool,
    ) -> dict[str, object]:
        updates: dict[str, object] = {}
        allowed_fields = campaign.questions.keys()

        property_types = ("apartment", "villa", "townhouse", "house", "land", "office")
        property_type = next(
            (value for value in property_types if value in normalized_utterance),
            None,
        )
        if property_type and "property_type" in allowed_fields:
            updates["property_type"] = property_type

        if "currently_listed" in allowed_fields:
            if "not listed" in normalized_utterance:
                updates["currently_listed"] = False
            elif "listed" in normalized_utterance:
                updates["currently_listed"] = True

        timeline = re.search(
            r"\b(?:in|within)\s+(?:about\s+)?(?:\d+|one|two|three|four|five|six)\s+"
            r"(?:days?|weeks?|months?|years?)\b",
            normalized_utterance,
        )
        if timeline and "selling_timeline" in allowed_fields:
            updates["selling_timeline"] = timeline.group(0)

        location = re.search(
            r"\b(?:located in|property is in|it is in|it's in)\s+(.+?)[.!?]?$",
            utterance,
            re.IGNORECASE,
        )
        if location and "property_location" in allowed_fields:
            updates["property_location"] = location.group(1).strip()

        if (
            allow_pending_field_fallback
            and not updates
            and state.last_asked_field
            and state.last_asked_field != campaign.opening_field
            and "?" not in utterance
        ):
            value: object = utterance.strip().rstrip(".")
            if state.last_asked_field == "currently_listed":
                if normalized_utterance in {"yes", "yes it is", "it is"}:
                    value = True
                elif normalized_utterance in {"no", "no it isn't", "it isn't"}:
                    value = False
            updates[state.last_asked_field] = value

        return updates