from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
import importlib
import json
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
    def __init__(
        self,
        backend: ChatGenerationBackend,
        *,
        max_tokens: int = 384,
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

    @staticmethod
    def _build_messages(
        utterance: str,
        state: ConversationState,
        campaign: Campaign,
    ) -> list[dict[str, str]]:
        schema = {
            "suggested_outcome": "one configured outcome or null",
            "field_updates": {"configured_field_name": "extracted value"},
            "answer": "brief direct answer to the owner's question, otherwise null",
            "acknowledgement": "brief natural acknowledgement, otherwise null",
            "callback_requested": "true, false for explicit cancellation, or null",
            "human_transfer_requested": False,
        }
        system_message = (
            "Interpret one turn in an automated telephone conversation. "
            "Use the campaign conversation brief, guidelines, and scenario playbook "
            "as flexible guidance, not a script. Never recite them, copy their wording, "
            "or force a scenario sequence. Adapt to the latest utterance and known "
            "state while working toward the campaign objective. "
            "Extract every fact the owner volunteered, including corrections. "
            "Outcome classification and field extraction are separate tasks: after "
            "choosing an outcome, inspect the utterance again and include every stated "
            "configured value, even when that fact helped choose the outcome. "
            "When the latest utterance directly answers state.last_asked_field, always "
            "write that field to field_updates when the value matches its configured "
            "type, including one-word answers such as apartment, yes, no, both, or a "
            "place name. Do not treat a generic response such as interested, yes, or "
            "maybe as SELL, RENT, or SELL_OR_RENT unless the owner identifies the "
            "option. Do not extract references or placeholders such as my address, "
            "there, you know, something else, or not sure as field values. An "
            "acknowledgement never substitutes for extraction. "
            "Use only configured outcomes and field names. Do not invent facts. "
            "Apply the campaign classification rules in their listed order before "
            "choosing an outcome. "
            "Never ignore a direct question. Set answer to one or two brief, natural "
            "sentences that answer it. Use the campaign FAQ when relevant. For other "
            "campaign-related questions, use reliable general knowledge; if current, "
            "specific, or live information is unavailable, say so plainly instead of "
            "inventing it. Set acknowledgement to a short, natural response to an "
            "ordinary statement when it adds value, otherwise null. Answer and "
            "acknowledgement must be statements, not questions, because the application "
            "selects and adds the next question. Do not repeat, paraphrase, or anticipate "
            "that next question. Never produce prohibited statements. Return "
            "exactly one JSON object with no markdown or commentary. "
            f"The required JSON shape is: {json.dumps(schema)}"
        )
        context = {
            "campaign": {
                "objective": campaign.objective,
                "conversation_brief": campaign.conversation_brief,
                "conversation_guidelines": campaign.conversation_guidelines,
                "scenario_playbook": campaign.scenario_playbook,
                "desired_outcomes": campaign.desired_outcomes,
                "outcome_guidance": campaign.outcome_guidance,
                "classification_rules": campaign.classification_rules,
                "required_fields": campaign.required_fields,
                "fields_by_outcome": campaign.fields_by_outcome,
                "field_types": campaign.field_types,
                "field_allowed_values": campaign.field_allowed_values,
                "questions": campaign.questions,
                "faq_answers": campaign.faq_answers,
                "prohibited_statements": campaign.prohibited_statements,
            },
            "state": {
                "outcome": state.outcome,
                "known_fields": state.fields,
                "last_asked_field": state.last_asked_field,
            },
            "latest_owner_utterance": utterance,
        }
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": json.dumps(context, ensure_ascii=True)},
        ]

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