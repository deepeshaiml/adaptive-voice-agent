from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
import importlib
import json
import re
from threading import Event
from typing import Any, Protocol

from speaking_agent.campaign import Campaign
from speaking_agent.adapters.thread_bridge import wait_for_thread_worker
from speaking_agent.domain import ConversationState
from speaking_agent.model import (
    ConversationModelError,
    ModelInterpretation,
)


DEFAULT_MODEL_PATH = "mlx-community/Qwen3-4B-Instruct-2507-4bit"


class ChatGenerationBackend(Protocol):
    async def prepare(self) -> None: ...

    async def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
    ) -> str: ...

    async def close(self) -> None: ...


class QwenMlxConversationModel:
    MAX_PROMPT_CHARS = 14_500
    _SCENARIO_STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "being",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "or",
        "owner",
        "says",
        "the",
        "their",
        "they",
        "to",
        "was",
        "what",
        "when",
        "you",
    }

    def __init__(
        self,
        backend: ChatGenerationBackend,
        *,
        max_tokens: int = 256,
    ) -> None:
        self._backend = backend
        self._max_tokens = max_tokens

    async def prepare(self) -> None:
        await self._backend.prepare()

    async def close(self) -> None:
        await self._backend.close()

    async def interpret(
        self,
        utterance: str,
        state: ConversationState,
        campaign: Campaign,
    ) -> ModelInterpretation:
        messages = self._build_messages(utterance, state, campaign)
        try:
            response = await self._backend.generate(
                messages,
                max_tokens=self._max_tokens,
            )
        except asyncio.CancelledError:
            raise
        except ConversationModelError:
            raise
        except Exception as error:
            raise ConversationModelError("Local model generation failed") from error
        return self._parse_response(response)

    @classmethod
    def _build_messages(
        cls,
        utterance: str,
        state: ConversationState,
        campaign: Campaign,
    ) -> list[dict[str, str]]:
        schema = {
            "suggested_outcome": "one configured outcome or null",
            "field_updates": {"configured_field_name": "extracted value"},
            "answer": "brief direct answer to the owner's question, otherwise null",
            "acknowledgement": "brief natural acknowledgement, otherwise null",
            "next_question_field": "configured field addressed by next_question or null",
            "next_question": "one short natural follow-up question or null",
            "callback_requested": "true, false for explicit cancellation, or null",
            "human_transfer_requested": False,
        }
        system_message = (
            "Plan one turn in an automated telephone conversation. Campaign data is "
            "flexible guidance, not a script: adapt to the latest utterance and state; "
            "never recite guidance. Use recent dialogue for references and corrections, "
            "and avoid repeating delivered agent wording. If agent speech was interrupted, "
            "respond to the interruption before revisiting anything unheard. "
            "Classify using ordered campaign rules and use only configured outcomes and "
            "fields. Extract every volunteered fact and correction, including one-word "
            "answers to last_asked_field. Classification and extraction are independent. "
            "Do not infer SELL, RENT, or SELL_OR_RENT from generic interest, yes, or maybe. "
            "Do not store placeholders such as my address, there, something else, or not "
            "sure. Never invent facts. Market data may be used only when market_data is "
            "present and available; always distinguish actual registered transactions "
            "from current asking listings. Never ignore a direct question: answer it in one or "
            "two brief natural statements using relevant FAQ facts or safe general "
            "knowledge, and state limits plainly. acknowledgement is a short optional "
            "statement, never a substitute for extraction. If continuing, propose exactly "
            "one short next_question and its configured next_question_field for the likely "
            "remaining field. Do not combine or repeat questions; application policy owns "
            "the final question. Never produce prohibited statements. Return exactly one "
            "JSON object without markdown or commentary. Keep output sparse: omit "
            "suggested_outcome when no new outcome is established; omit field_updates "
            "when empty; omit answer, acknowledgement, and callback_requested when null; "
            "omit human_transfer_requested when false; omit both next-question keys when "
            "there is no next question. When answer is set, omit acknowledgement. "
            f"The required JSON shape is: {json.dumps(schema)}"
        )
        context = {
            "campaign": {
                "objective": campaign.objective,
                "conversation_brief": campaign.conversation_brief,
                "active_guidance": cls._active_guidance(
                    utterance,
                    state,
                    campaign,
                ),
                "voice_style": cls._compact_voice_style(campaign),
                "current_flow": cls._current_flow(state, campaign),
                "relevant_scenarios": cls._relevant_scenarios(
                    utterance,
                    state,
                    campaign,
                ),
                "sample_phrases": cls._compact_sample_phrases(campaign),
                "interruption_guidance": cls._compact_interruption_guidance(campaign),
                "desired_outcomes": campaign.desired_outcomes,
                "outcome_guidance": campaign.outcome_guidance,
                "classification_rules": campaign.classification_rules,
                "required_fields": campaign.required_fields,
                "fields_by_outcome": campaign.fields_by_outcome,
                "field_types": campaign.field_types,
                "field_allowed_values": campaign.field_allowed_values,
                "field_extraction_hints": campaign.field_extraction_hints,
                "field_dependencies": campaign.field_dependencies,
                "questions": campaign.questions,
                "relevant_faq_answers": cls._relevant_faq_answers(
                    utterance,
                    state,
                    campaign,
                ),
                "prohibited_statements": campaign.prohibited_statements,
            },
            "state": {
                "stage": state.stage,
                "outcome": state.outcome,
                "known_fields": dict(state.fields),
                "skipped_fields": sorted(state.skipped_fields),
                "last_asked_field": state.last_asked_field,
                "asked_field_counts": dict(state.asked_field_counts),
                "callback_requested": state.callback_requested,
                "market_data": state.market_context,
                "recent_dialogue": [dict(turn) for turn in state.recent_dialogue],
            },
            "latest_owner_utterance": utterance,
        }
        user_content = cls._serialize_context_with_budget(system_message, context)
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]

    @classmethod
    def _serialize_context_with_budget(
        cls,
        system_message: str,
        context: dict[str, Any],
    ) -> str:
        def serialize() -> str:
            return json.dumps(context, ensure_ascii=True, separators=(",", ":"))

        user_content = serialize()
        dialogue = context["state"]["recent_dialogue"]
        while len(system_message) + len(user_content) > cls.MAX_PROMPT_CHARS and dialogue:
            dialogue.pop(0)
            user_content = serialize()

        campaign_context = context["campaign"]
        for optional_key in (
            "sample_phrases",
            "interruption_guidance",
            "relevant_scenarios",
            "voice_style",
        ):
            if len(system_message) + len(user_content) <= cls.MAX_PROMPT_CHARS:
                break
            campaign_context.pop(optional_key, None)
            user_content = serialize()

        active_guidance = campaign_context.get("active_guidance")
        while (
            len(system_message) + len(user_content) > cls.MAX_PROMPT_CHARS
            and isinstance(active_guidance, (list, tuple))
            and len(active_guidance) > 3
        ):
            active_guidance = active_guidance[:-1]
            campaign_context["active_guidance"] = active_guidance
            user_content = serialize()

        if len(system_message) + len(user_content) > cls.MAX_PROMPT_CHARS:
            raise ConversationModelError(
                "Campaign and current turn exceed the conversation prompt budget"
            )
        return user_content

    @classmethod
    def _relevant_scenarios(
        cls,
        utterance: str,
        state: ConversationState,
        campaign: Campaign,
        *,
        limit: int = 3,
    ) -> tuple[dict[str, str], ...]:
        recent_owner_text = " ".join(
            turn["text"]
            for turn in state.recent_dialogue[-6:]
            if turn.get("role") == "owner"
        )
        query_tokens = cls._keywords(f"{recent_owner_text} {utterance}")
        scored: list[tuple[int, int, dict[str, str]]] = []
        for index, scenario in enumerate(campaign.scenario_playbook):
            scenario_tokens = cls._keywords(scenario["when"])
            score = len(query_tokens & scenario_tokens)
            if score:
                scored.append((score, -index, scenario))
        scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
        return tuple(item[2] for item in scored[:limit])

    @classmethod
    def _active_guidance(
        cls,
        utterance: str,
        state: ConversationState,
        campaign: Campaign,
        *,
        relevant_limit: int = 4,
    ) -> tuple[str, ...]:
        all_guidance = tuple(
            dict.fromkeys(
                (
                    *campaign.conversation_guidelines,
                    *campaign.natural_conversation_rules,
                    *campaign.field_collection_rules,
                    *campaign.hard_stop_context_rules,
                )
            )
        )
        core_markers = (
            "respond to what",
            "one clear question",
            "do not mechanically",
            "two or more fields",
            "do not repeat",
            "asks a relevant question",
            "never ignore",
            "information is unavailable",
            "never invent",
            "clear refusal",
            "achieved its useful goal",
        )
        core = [
            guidance
            for guidance in all_guidance
            if any(marker in guidance.casefold() for marker in core_markers)
        ]
        query_text = " ".join(
            (
                utterance,
                *(
                    turn["text"]
                    for turn in state.recent_dialogue[-4:]
                    if turn.get("role") == "owner"
                ),
            )
        )
        query_tokens = cls._keywords(query_text)
        scored = sorted(
            (
                (len(query_tokens & cls._keywords(guidance)), -index, guidance)
                for index, guidance in enumerate(all_guidance)
                if guidance not in core
            ),
            reverse=True,
        )
        relevant = [
            guidance
            for score, _, guidance in scored
            if score > 0
        ][:relevant_limit]
        return tuple(dict.fromkeys((*core, *relevant)))

    @classmethod
    def _relevant_faq_answers(
        cls,
        utterance: str,
        state: ConversationState,
        campaign: Campaign,
        *,
        limit: int = 3,
    ) -> dict[str, str]:
        query_text = " ".join(
            (
                utterance,
                *(
                    turn["text"]
                    for turn in state.recent_dialogue[-4:]
                    if turn.get("role") == "owner"
                ),
            )
        )
        query_tokens = cls._keywords(query_text)
        scored = sorted(
            (
                (
                    len(
                        query_tokens
                        & cls._keywords(
                            " ".join(
                                (
                                    question,
                                    *campaign.faq_aliases.get(question, ()),
                                )
                            )
                        )
                    ),
                    -index,
                    question,
                    answer,
                )
                for index, (question, answer) in enumerate(campaign.faq_answers.items())
            ),
            reverse=True,
        )
        selected = [item for item in scored if item[0] > 0][:limit]
        return {
            question: answer
            for _, _, question, answer in selected
        }

    @classmethod
    def _keywords(cls, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9']+", text.casefold())
            if len(token) > 2 and token not in cls._SCENARIO_STOP_WORDS
        }

    @staticmethod
    def _current_flow(
        state: ConversationState,
        campaign: Campaign,
    ) -> dict[str, Any] | None:
        if state.stage.value == "COMPLETED":
            target = "CLOSE"
        elif state.outcome == "UNKNOWN":
            target = "UNDERSTAND_INTENT"
        else:
            target = "PROPERTY_BASICS"
        return next(
            (stage for stage in campaign.conversation_flow if stage["stage"] == target),
            None,
        )

    @staticmethod
    def _compact_sample_phrases(campaign: Campaign) -> dict[str, Any]:
        compact: dict[str, Any] = {}
        for name, values in campaign.sample_phrases.items():
            compact[name] = tuple(values[:2]) if isinstance(values, (list, tuple)) else values
        return compact

    @staticmethod
    def _compact_voice_style(campaign: Campaign) -> dict[str, Any]:
        keys = (
            "personality",
            "tone",
            "pacing",
            "verbosity",
            "energy",
            "acknowledgement_rule",
            "avoid",
        )
        return {key: campaign.voice_style[key] for key in keys if key in campaign.voice_style}

    @staticmethod
    def _compact_interruption_guidance(campaign: Campaign) -> dict[str, Any]:
        keys = (
            "brief_pause_behavior",
            "unclear_audio_behavior",
            "clarification_examples",
            "maximum_clarification_attempts",
        )
        return {
            key: campaign.interruption_and_silence_handling[key]
            for key in keys
            if key in campaign.interruption_and_silence_handling
        }

    @staticmethod
    def _parse_response(response: str) -> ModelInterpretation:
        payload = response.strip()
        if payload.startswith("```") and payload.endswith("```"):
            lines = payload.splitlines()
            payload = "\n".join(lines[1:-1]).strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ConversationModelError("Local model returned invalid JSON") from error
        if not isinstance(data, dict):
            raise ConversationModelError("Local model response must be a JSON object")

        suggested_outcome = data.get("suggested_outcome")
        answer = data.get("answer")
        acknowledgement = data.get("acknowledgement")
        next_question_field = data.get("next_question_field")
        next_question = data.get("next_question")
        field_updates = data.get("field_updates", {})
        callback_requested = data.get("callback_requested")
        human_transfer_requested = data.get("human_transfer_requested", False)
        if suggested_outcome is not None and not isinstance(suggested_outcome, str):
            raise ConversationModelError("suggested_outcome must be a string or null")
        if answer is not None and not isinstance(answer, str):
            raise ConversationModelError("answer must be a string or null")
        if acknowledgement is not None and not isinstance(acknowledgement, str):
            raise ConversationModelError(
                "acknowledgement must be a string or null"
            )
        if next_question_field is not None and not isinstance(next_question_field, str):
            raise ConversationModelError(
                "next_question_field must be a string or null"
            )
        if next_question is not None and not isinstance(next_question, str):
            raise ConversationModelError("next_question must be a string or null")
        if (next_question_field is None) != (next_question is None):
            raise ConversationModelError(
                "next_question_field and next_question must both be set or null"
            )
        if not isinstance(field_updates, dict):
            raise ConversationModelError("field_updates must be a JSON object")
        if callback_requested is not None and not isinstance(callback_requested, bool):
            raise ConversationModelError(
                "callback_requested must be a boolean or null"
            )
        if not isinstance(human_transfer_requested, bool):
            raise ConversationModelError("human_transfer_requested must be a boolean")

        return ModelInterpretation(
            suggested_outcome=suggested_outcome,
            field_updates=field_updates,
            answer=answer,
            acknowledgement=acknowledgement,
            next_question_field=next_question_field,
            next_question=next_question,
            callback_requested=callback_requested,
            human_transfer_requested=human_transfer_requested,
        )


class _GenerationCancelled(Exception):
    pass


class MlxLmBackend:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        *,
        cancellation_grace_seconds: float = 2.0,
    ) -> None:
        self.model_path = model_path
        self._mlx_lm: Any | None = None
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._generation_lock = asyncio.Lock()
        self._cancellation_grace_seconds = cancellation_grace_seconds
        self._unhealthy = False

    async def prepare(self) -> None:
        async with self._generation_lock:
            try:
                await asyncio.to_thread(self._ensure_loaded)
            except (ImportError, OSError, RuntimeError) as error:
                raise ConversationModelError(
                    f"Unable to load MLX model {self.model_path!r}"
                ) from error

    async def close(self) -> None:
        async with self._generation_lock:
            if self._unhealthy:
                return
            self._model = None
            self._tokenizer = None
            self._mlx_lm = None

    async def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
    ) -> str:
        if self._unhealthy:
            raise ConversationModelError(
                "Local model backend is quarantined after a cancellation timeout"
            )
        cancellation = Event()
        async with self._generation_lock:
            worker = asyncio.create_task(
                asyncio.to_thread(
                    self._generate_sync,
                    list(messages),
                    max_tokens,
                    cancellation,
                )
            )
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancellation.set()
                stopped = await wait_for_thread_worker(
                    worker,
                    timeout_seconds=self._cancellation_grace_seconds,
                )
                if not stopped:
                    self._unhealthy = True
                raise
            except _GenerationCancelled as error:
                raise ConversationModelError("Local model generation was cancelled") from error
            except (ImportError, OSError, RuntimeError) as error:
                raise ConversationModelError(
                    f"Unable to run MLX model {self.model_path!r}"
                ) from error

    def _generate_sync(
        self,
        messages: list[Mapping[str, str]],
        max_tokens: int,
        cancellation: Event,
    ) -> str:
        self._ensure_loaded()
        prompt = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        parts: list[str] = []
        for response in self._mlx_lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=max_tokens,
        ):
            if cancellation.is_set():
                raise _GenerationCancelled
            parts.append(response.text)
        return "".join(parts)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._mlx_lm = importlib.import_module("mlx_lm")
        self._model, self._tokenizer = self._mlx_lm.load(self.model_path)