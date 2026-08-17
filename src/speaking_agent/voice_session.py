from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
import inspect
from typing import Any

from speaking_agent.answering import AnswerKind, AnsweringMachineDetector
from speaking_agent.audio_recording import ConversationAudioRecorder
from speaking_agent.campaign import Campaign
from speaking_agent.conversation import ConversationSession
from speaking_agent.domain import AgentReply, LeadOutcome, SessionAction
from speaking_agent.domain import ConversationContext
from speaking_agent.model import ConversationModel
from speaking_agent.observability import LatencyTrace, TimingEventName
from speaking_agent.speech import (
    AudioFrame,
    SpeechNotRecognizedError,
    SpeechRecognizer,
    SpeechSynthesizer,
)
from speaking_agent.transport import (
    CallTransport,
    TransportError,
    TransportEvent,
    TransportEventKind,
)
from speaking_agent.turn_detection import (
    EnergyTurnDetector,
    TurnEventKind,
)


@dataclass(frozen=True, slots=True)
class VoiceCallResult:
    lead: LeadOutcome
    answer_kind: AnswerKind
    interruptions: int
    disconnected: bool
    cleanup_errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _TurnFinished:
    reply: AgentReply | None = None
    error: BaseException | None = None
    ignored: bool = False


@dataclass(frozen=True, slots=True)
class _PlaybackFinished:
    action: SessionAction
    disclosure_delivered: bool = False
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _SilentAction:
    action: SessionAction


@dataclass(frozen=True, slots=True)
class _TransferFinished:
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _AudioTurn:
    frames: tuple[AudioFrame, ...]
    confirmation_overlap: bool = False


class CallSession:
    def __init__(
        self,
        *,
        campaign: Campaign,
        model: ConversationModel,
        recognizer: SpeechRecognizer,
        synthesizer: SpeechSynthesizer,
        transport: CallTransport,
        answering_detector: AnsweringMachineDetector | None = None,
        turn_detector: EnergyTurnDetector | None = None,
        trace: LatencyTrace | None = None,
        on_do_not_contact: Callable[[str], Awaitable[None]] | None = None,
        on_transcript: Callable[[str], Awaitable[None] | None] | None = None,
        on_agent_reply: Callable[[AgentReply], Awaitable[None] | None] | None = None,
        on_interruption: Callable[[], Awaitable[None] | None] | None = None,
        recognition_language: str | None = None,
        recognition_context: str = "",
        transfer_available: bool = True,
        conversation_context: ConversationContext | None = None,
        audio_recorder: ConversationAudioRecorder | None = None,
    ) -> None:
        self.conversation = ConversationSession(
            campaign,
            model,
            delivery_tracking=True,
            context=conversation_context,
        )
        self.model = model
        self.recognizer = recognizer
        self.synthesizer = synthesizer
        self.transport = transport
        self.answering_detector = answering_detector
        self.turn_detector = turn_detector or EnergyTurnDetector()
        self.trace = trace or LatencyTrace()
        self._on_do_not_contact = on_do_not_contact
        self._on_transcript = on_transcript
        self._on_agent_reply = on_agent_reply
        self._on_interruption = on_interruption
        self._recognition_language = recognition_language
        self._recognition_context = recognition_context
        self._do_not_contact_persisted = False
        self._disclosure_delivered = False
        self._transfer_available = transfer_available
        self.audio_recorder = audio_recorder
        if audio_recorder is not None:
            set_playout_observer = getattr(
                self.transport,
                "set_playout_observer",
                None,
            )
            if set_playout_observer is None:
                raise TypeError(
                    "Audio recording requires transport playout observation"
                )
            set_playout_observer(audio_recorder.record_agent_audio)
        self.interruptions = 0
        self._answer_kind: AnswerKind | None = (
            AnswerKind.HUMAN if answering_detector is None else None
        )
        self._queue: asyncio.Queue[
            TransportEvent
            | _TurnFinished
            | _PlaybackFinished
            | _SilentAction
            | _TransferFinished
        ] = asyncio.Queue()
        self._task_group: asyncio.TaskGroup | None = None
        self._playback_task: asyncio.Task[None] | None = None
        self._turn_task: asyncio.Task[None] | None = None
        self._transfer_task: asyncio.Task[None] | None = None
        self._active_turn: _AudioTurn | None = None
        self._pending_turns: deque[_AudioTurn] = deque()
        self._deferred_reply: AgentReply | None = None
        self._active_reply: AgentReply | None = None
        self._confirmation_overlap_turn = False
        self._speech_in_progress = False
        self._pending_terminal_action: SessionAction | None = None
        self._cleanup_errors: list[str] = []

    async def run(self) -> VoiceCallResult:
        disconnected = False
        terminal_error: BaseException | None = None
        try:
            await self._prepare()
            await self.transport.connect()
            async with asyncio.TaskGroup() as task_group:
                self._task_group = task_group
                event_task = task_group.create_task(self._pump_transport_events())
                event_loop = asyncio.get_running_loop()
                listening_deadline = (
                    event_loop.time()
                    + float(
                        self.conversation.campaign.behavior[
                            "initial_answer_timeout_seconds"
                        ]
                    )
                    if self.answering_detector is not None
                    else None
                )
                if self.answering_detector is None:
                    await self._start_playback(self.conversation.start())
                finished = False
                while not finished:
                    try:
                        if listening_deadline is None:
                            event = await self._queue.get()
                        else:
                            remaining = listening_deadline - event_loop.time()
                            if remaining <= 0:
                                raise TimeoutError
                            async with asyncio.timeout(remaining):
                                event = await self._queue.get()
                    except TimeoutError:
                        if self._pending_terminal_action is not None:
                            pending_action = self._pending_terminal_action
                            self._pending_terminal_action = None
                            listening_deadline = None
                            if pending_action == SessionAction.HANG_UP:
                                await self.transport.hang_up()
                                finished = True
                            elif pending_action == SessionAction.TRANSFER:
                                await self._start_transfer()
                            continue
                        self.conversation.abort()
                        try:
                            await self.transport.hang_up()
                        except Exception as error:
                            self._cleanup_errors.append(
                                f"hang_up:{type(error).__name__}"
                            )
                        finished = True
                        continue
                    if isinstance(event, TransportEvent):
                        if event.kind == TransportEventKind.FAILED:
                            terminal_error = TransportError(
                                event.reason or "Call transport failed"
                            )
                            self.conversation.abort()
                            try:
                                await self.transport.hang_up()
                            except Exception as error:
                                self._cleanup_errors.append(
                                    f"hang_up:{type(error).__name__}"
                                )
                            finished = True
                        elif event.kind == TransportEventKind.DISCONNECTED:
                            disconnected = True
                            try:
                                await self._process_disconnected_speech()
                            except BaseException as error:
                                terminal_error = error
                            self.conversation.abort()
                            finished = True
                        elif event.audio is not None:
                            if self.audio_recorder is not None:
                                self.audio_recorder.record_owner_audio(event.audio)
                            had_pending_speech = self.turn_detector.has_pending_speech
                            turn_events = self.turn_detector.process(event.audio)
                            if (
                                not had_pending_speech
                                and self.turn_detector.has_candidate_speech
                            ):
                                self._confirmation_overlap_turn = (
                                    self._confirmation_capture_context_active()
                                )
                            for turn_event in turn_events:
                                if turn_event.kind == TurnEventKind.SPEECH_STARTED:
                                    listening_deadline = None
                                    self._speech_in_progress = True
                                    if self._transfer_task is not None:
                                        await self._cancel_transfer_task()
                                        self._pending_terminal_action = (
                                            SessionAction.TRANSFER
                                        )
                                    self._confirmation_overlap_turn = (
                                        self._confirmation_overlap_turn
                                        or self._confirmation_capture_context_active()
                                    )
                                    protect_confirmation = (
                                        self._confirmation_playback_is_protected()
                                    )
                                    if (
                                        not protect_confirmation
                                        and await self._interrupt_playback()
                                    ):
                                        await self._notify(
                                            self._on_interruption,
                                            None,
                                        )
                                else:
                                    listening_deadline = None
                                    self._speech_in_progress = False
                                    self.trace.record(TimingEventName.SPEECH_END)
                                    if self._confirmation_overlap_turn:
                                        self._confirmation_overlap_turn = False
                                        confirmation_delivered = (
                                            self.conversation.recipient_confirmation_delivered
                                        )
                                        if confirmation_delivered:
                                            await self._start_turn(
                                                turn_event.frames,
                                                confirmation_overlap=True,
                                            )
                                        else:
                                            self._pending_turns.append(
                                                _AudioTurn(
                                                    turn_event.frames,
                                                    confirmation_overlap=True,
                                                )
                                            )
                                    else:
                                        await self._start_turn(turn_event.frames)
                            if not turn_events:
                                if not self.turn_detector.has_pending_speech:
                                    self._confirmation_overlap_turn = False
                                await self._resume_deferred_reply_if_input_idle()
                    elif isinstance(event, _TurnFinished):
                        self._turn_task = None
                        self._active_turn = None
                        if event.error is not None:
                            terminal_error = event.error
                            self.conversation.abort()
                            try:
                                await self.transport.hang_up()
                            except Exception as error:
                                self._cleanup_errors.append(
                                    f"hang_up:{type(error).__name__}"
                                )
                            finished = True
                        elif event.reply is not None:
                            self._pending_terminal_action = None
                            if (
                                self.conversation.state.outcome
                                != "HUMAN_TRANSFER"
                            ):
                                await self._cancel_transfer_task()
                            if self._has_pending_input():
                                self._deferred_reply = event.reply
                                self.conversation.mark_agent_reply_delivery(
                                    event.reply.text,
                                    "pending",
                                )
                                if self._pending_turns:
                                    await self._start_turn(
                                        self._pending_turns.popleft()
                                    )
                            else:
                                await self._start_playback(event.reply)
                        elif event.ignored:
                            if self._pending_turns:
                                await self._start_turn(self._pending_turns.popleft())
                            elif (
                                self._deferred_reply is not None
                                and not self._has_pending_input()
                            ):
                                deferred_reply = self._deferred_reply
                                self._deferred_reply = None
                                await self._start_playback(deferred_reply)
                            elif (
                                self._pending_terminal_action is not None
                                and not self.turn_detector.has_pending_speech
                            ):
                                pending_action = self._pending_terminal_action
                                self._pending_terminal_action = None
                                if pending_action == SessionAction.HANG_UP:
                                    await self.transport.hang_up()
                                    finished = True
                                elif pending_action == SessionAction.TRANSFER:
                                    await self._start_transfer()
                            else:
                                listening_deadline = event_loop.time() + float(
                                    self.conversation.campaign.behavior[
                                        "conversation_idle_timeout_seconds"
                                    ]
                                )
                    elif isinstance(event, _SilentAction):
                        self._turn_task = None
                        self._active_turn = None
                        if event.action == SessionAction.HANG_UP:
                            await self.transport.hang_up()
                            finished = True
                        elif event.action == SessionAction.TRANSFER:
                            await self._start_transfer()
                        else:
                            terminal_error = RuntimeError(
                                "Unsupported silent call action"
                            )
                            self.conversation.abort()
                            finished = True
                    elif isinstance(event, _TransferFinished):
                        self._transfer_task = None
                        if event.error is not None:
                            self._cleanup_errors.append(
                                f"transfer:{type(event.error).__name__}"
                            )
                            await self._start_playback(
                                AgentReply(
                                    self.conversation.campaign.transfer_unavailable_message,
                                    SessionAction.HANG_UP,
                                )
                            )
                        else:
                            await self._process_disconnected_speech()
                            finished = True
                    else:
                        active_reply = self._active_reply
                        self._playback_task = None
                        self._active_reply = None
                        if event.error is not None:
                            terminal_error = event.error
                            self.conversation.abort()
                            try:
                                await self.transport.hang_up()
                            except Exception as error:
                                self._cleanup_errors.append(
                                    f"hang_up:{type(error).__name__}"
                                )
                            finished = True
                        else:
                            if active_reply is not None:
                                self.conversation.mark_agent_reply_delivery(
                                    active_reply.text,
                                    "delivered",
                                )
                            if event.disclosure_delivered:
                                self._disclosure_delivered = True
                            if event.action == SessionAction.HANG_UP:
                                if self._has_pending_input():
                                    self._pending_terminal_action = (
                                        SessionAction.HANG_UP
                                    )
                                    if self._has_confirmed_input():
                                        listening_deadline = None
                                    else:
                                        listening_deadline = (
                                            event_loop.time()
                                            + self._terminal_candidate_grace_seconds()
                                        )
                                else:
                                    await self.transport.hang_up()
                                    finished = True
                            elif event.action == SessionAction.TRANSFER:
                                if self._has_pending_input():
                                    self._pending_terminal_action = (
                                        SessionAction.TRANSFER
                                    )
                                    if self._has_confirmed_input():
                                        listening_deadline = None
                                    else:
                                        listening_deadline = (
                                            event_loop.time()
                                            + self._terminal_candidate_grace_seconds()
                                        )
                                else:
                                    await self._start_transfer()
                            elif self._pending_turns:
                                listening_deadline = None
                                await self._start_turn(
                                    self._pending_turns.popleft()
                                )
                            else:
                                listening_deadline = event_loop.time() + float(
                                    self.conversation.campaign.behavior[
                                        "conversation_idle_timeout_seconds"
                                    ]
                                )
                event_task.cancel()
                await self._cancel_active_tasks()
        finally:
            await self._close()
        if terminal_error is not None:
            raise terminal_error
        return VoiceCallResult(
            lead=self.conversation.result(),
            answer_kind=self._answer_kind or AnswerKind.UNCERTAIN,
            interruptions=self.interruptions,
            disconnected=disconnected,
            cleanup_errors=tuple(self._cleanup_errors),
        )

    async def _prepare(self) -> None:
        await self.transport.prepare()
        await self.model.prepare()
        await self.recognizer.prepare()
        await self.synthesizer.prepare()
        if self.answering_detector is not None:
            await self.answering_detector.prepare()
        if self.audio_recorder is not None:
            await self.audio_recorder.prepare(
                call_id=self.conversation.state.call_id,
                campaign_id=self.conversation.campaign.campaign_id,
                retention_days=int(
                    self.conversation.campaign.behavior["data_retention_days"]
                ),
            )

    async def _close(self) -> None:
        timeout_seconds = float(
            self.conversation.campaign.behavior["cleanup_timeout_seconds"]
        )
        operations = [
            ("active_tasks", self._cancel_active_tasks),
            ("synthesizer", self.synthesizer.close),
            ("recognizer", self.recognizer.close),
        ]
        if self.answering_detector is not None:
            operations.append(("answering_detector", self.answering_detector.close))
        operations.extend(
            (
                ("model", self.model.close),
                ("transport", self.transport.close),
            )
        )
        if self.audio_recorder is not None:
            operations.append(("audio_recorder", self.audio_recorder.close))
        for name, operation in operations:
            try:
                async with asyncio.timeout(timeout_seconds):
                    await operation()
            except TimeoutError:
                self._cleanup_errors.append(f"{name}:TimeoutError")
            except Exception as error:
                self._cleanup_errors.append(f"{name}:{type(error).__name__}")

    async def _pump_transport_events(self) -> None:
        try:
            async for event in self.transport.events():
                await self._queue.put(event)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            await self._queue.put(
                TransportEvent(
                    kind=TransportEventKind.FAILED,
                    reason=str(error),
                )
            )

    async def _start_turn(
        self,
        frames: tuple[AudioFrame, ...] | _AudioTurn,
        *,
        confirmation_overlap: bool = False,
    ) -> None:
        if self._task_group is None:
            return
        turn = (
            frames
            if isinstance(frames, _AudioTurn)
            else _AudioTurn(
                frames,
                confirmation_overlap=confirmation_overlap,
            )
        )
        if self._turn_task is not None:
            self._pending_turns.append(turn)
            return
        self._active_turn = turn
        self._turn_task = self._task_group.create_task(
            self._process_turn_and_notify(turn)
        )

    async def _process_turn_and_notify(self, turn: _AudioTurn) -> None:
        try:
            final_text = await self._transcribe_turn(turn.frames)
            hard_stop = self.conversation.policy.hard_stop_outcome(final_text)
            confirmation: str | None = None
            if turn.confirmation_overlap and hard_stop is None:
                confirmation = (
                    self.conversation.classify_recipient_confirmation_overlap(
                        final_text
                    )
                )
            if self.conversation.state.ended and hard_stop is None:
                if confirmation == "denied":
                    await self._notify(self._on_transcript, final_text)
                    reply = await self.conversation.receive(
                        final_text,
                        captured_confirmation=True,
                    )
                    self._deferred_reply = None
                    await self._queue.put(_TurnFinished(reply=reply))
                else:
                    await self._queue.put(_TurnFinished(ignored=True))
                return
            if confirmation == "noise":
                terminal_reply = (
                    self.conversation.register_recipient_confirmation_noise()
                )
                await self._queue.put(
                    _TurnFinished(
                        reply=terminal_reply,
                        ignored=terminal_reply is None,
                    )
                )
                return
            await self._notify(self._on_transcript, final_text)
            if self._answer_kind is None and self.answering_detector is not None:
                self._answer_kind = (
                    AnswerKind.HUMAN
                    if hard_stop is not None
                    else await self.answering_detector.classify(final_text)
                )
                if self._answer_kind != AnswerKind.HUMAN:
                    self.conversation.abort()
                    if (
                        self._answer_kind == AnswerKind.VOICEMAIL
                        and self.conversation.campaign.voicemail_message
                    ):
                        await self._queue.put(
                            _TurnFinished(
                                reply=AgentReply(
                                    self.conversation.campaign.voicemail_message,
                                    SessionAction.HANG_UP,
                                )
                            )
                        )
                    elif self._answer_kind == AnswerKind.VOICEMAIL:
                        await self._queue.put(
                            _SilentAction(SessionAction.HANG_UP)
                        )
                    else:
                        await self._queue.put(
                            _TurnFinished(
                                reply=AgentReply("Goodbye.", SessionAction.HANG_UP)
                            )
                        )
                    return
                if (
                    hard_stop is None
                    and self.conversation.awaiting_recipient_confirmation
                ):
                    opening = self.conversation.start()
                    self._deferred_reply = None
                    await self._queue.put(_TurnFinished(reply=opening))
                    return
                self.conversation.start(remember_reply=False)
            if turn.confirmation_overlap and hard_stop is None:
                if (
                    not self.conversation.awaiting_recipient_confirmation
                    and confirmation != "denied"
                ):
                    await self._queue.put(_TurnFinished(ignored=True))
                    return
            self.trace.record(TimingEventName.LLM_START)
            reply = await self.conversation.receive(
                final_text,
                captured_confirmation=turn.confirmation_overlap,
            )
            self._deferred_reply = None
            await self._persist_do_not_contact_if_needed()
            self.trace.record(TimingEventName.LLM_FIRST_TOKEN, streaming=False)
            await self._queue.put(_TurnFinished(reply=reply))
        except asyncio.CancelledError:
            raise
        except SpeechNotRecognizedError:
            await self._queue.put(_TurnFinished(ignored=True))
        except BaseException as error:
            await self._queue.put(_TurnFinished(error=error))

    def _confirmation_playback_is_protected(self) -> bool:
        task = self._playback_task
        active_reply = self._active_reply
        normalized_opening = " ".join(self.conversation.opening.split())
        return (
            not self.conversation.state.ended
            and self.conversation.awaiting_recipient_confirmation
            and not self.conversation.recipient_confirmation_delivered
            and active_reply is not None
            and " ".join(active_reply.text.split()).endswith(normalized_opening)
            and task is not None
            and not task.done()
        )

    def _confirmation_capture_context_active(self) -> bool:
        return (
            self.conversation.recipient_confirmation_issued
            or (
                self._active_turn is not None
                and self._active_turn.confirmation_overlap
            )
            or any(turn.confirmation_overlap for turn in self._pending_turns)
        )

    def _has_pending_input(self) -> bool:
        return (
            self._speech_in_progress
            or self.turn_detector.has_pending_speech
            or self._turn_task is not None
            or bool(self._pending_turns)
        )

    def _has_confirmed_input(self) -> bool:
        return (
            self._speech_in_progress
            or self.turn_detector.is_speaking
            or self._turn_task is not None
            or bool(self._pending_turns)
        )

    def _terminal_candidate_grace_seconds(self) -> float:
        minimum_speech_seconds = (
            self.turn_detector.config.minimum_speech_ms / 1_000
        )
        return max(0.25, minimum_speech_seconds + 0.2)

    async def _resume_deferred_reply_if_input_idle(self) -> None:
        if self._playback_task is not None or self._has_pending_input():
            return
        if self._deferred_reply is not None:
            deferred_reply = self._deferred_reply
            self._deferred_reply = None
            await self._start_playback(deferred_reply)
            return
        if self._pending_terminal_action is not None:
            pending_action = self._pending_terminal_action
            self._pending_terminal_action = None
            await self._queue.put(_SilentAction(pending_action))

    async def _transcribe_turn(self, frames: tuple[AudioFrame, ...]) -> str:
        final_text = ""
        saw_partial = False
        async with asyncio.timeout(
            float(self.conversation.campaign.behavior["asr_timeout_seconds"])
        ):
            async for event in self.recognizer.transcribe(
                self._audio_frames(frames),
                language=self._recognition_language,
                context=self._recognition_context,
            ):
                if event.is_final:
                    final_text = event.text
                    self.trace.record(TimingEventName.ASR_FINAL)
                elif not saw_partial:
                    self.trace.record(TimingEventName.ASR_PARTIAL)
                    saw_partial = True
        if not final_text.strip():
            raise SpeechNotRecognizedError(
                "Speech recognizer returned no final transcript"
            )
        return final_text

    async def _process_disconnected_speech(self) -> None:
        if self.conversation.state.do_not_contact:
            await self._persist_do_not_contact_if_needed()
            return

        turns: list[_AudioTurn] = []
        if self._active_turn is not None:
            turns.append(self._active_turn)
        turns.extend(self._pending_turns)
        flushed_turn = self.turn_detector.flush()
        if flushed_turn is not None and flushed_turn.frames:
            turns.append(_AudioTurn(flushed_turn.frames))

        await self._cancel_turn_task()
        self._pending_turns.clear()
        for turn in turns:
            try:
                final_text = await self._transcribe_turn(turn.frames)
            except SpeechNotRecognizedError:
                continue
            await self._notify(self._on_transcript, final_text)
            hard_stop = self.conversation.policy.hard_stop_outcome(final_text)
            if hard_stop is None:
                continue
            if self._answer_kind is None:
                self._answer_kind = AnswerKind.HUMAN
                self.conversation.start(remember_reply=False)
            await self.conversation.receive(final_text)
            await self._persist_do_not_contact_if_needed()
            if hard_stop == "DO_NOT_CONTACT":
                return

    async def _start_playback(self, reply: AgentReply) -> None:
        await self._interrupt_playback(count=False)
        if reply.action == SessionAction.TRANSFER and not self._transfer_available:
            reply = AgentReply(
                self.conversation.campaign.transfer_unavailable_message,
                SessionAction.HANG_UP,
            )
        reply, delivers_disclosure = self._with_required_disclosure(reply)
        self.conversation.mark_agent_reply_delivery(reply.text, "pending")
        await self._notify(self._on_agent_reply, reply)
        if self._task_group is None:
            raise RuntimeError("Call session task group is not active")
        self._active_reply = reply
        self._playback_task = self._task_group.create_task(
            self._playback_and_notify(
                reply,
                delivers_disclosure=delivers_disclosure,
            )
        )

    async def _playback_and_notify(
        self,
        reply: AgentReply,
        *,
        delivers_disclosure: bool,
    ) -> None:
        try:
            self.trace.record(TimingEventName.TTS_START)
            first_frame = True
            async with asyncio.timeout(
                float(
                    self.conversation.campaign.behavior["tts_timeout_seconds"]
                )
            ):
                async for frame in self.synthesizer.synthesize(reply.text):
                    if first_frame:
                        self.trace.record(TimingEventName.TTS_FIRST_AUDIO)
                    await self.transport.send_audio(frame)
                    if first_frame:
                        self.conversation.mark_agent_reply_started(reply)
                        self.trace.record(TimingEventName.PLAYBACK_START)
                        first_frame = False
                if first_frame:
                    raise RuntimeError("Speech synthesizer returned no audio")
                await self.transport.wait_for_playout()
            await self._queue.put(
                _PlaybackFinished(
                    action=reply.action,
                    disclosure_delivered=delivers_disclosure,
                )
            )
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            await self._queue.put(
                _PlaybackFinished(action=reply.action, error=error)
            )

    def _with_required_disclosure(
        self,
        reply: AgentReply,
    ) -> tuple[AgentReply, bool]:
        if self._answer_kind != AnswerKind.HUMAN or self._disclosure_delivered:
            return reply, False
        normalized_text = reply.text.casefold()
        has_disclosure = all(
            disclosure.casefold() in normalized_text
            for disclosure in self.conversation.campaign.required_disclosures
        )
        if not has_disclosure:
            reply = AgentReply(
                f"{self.conversation.campaign.introduction} {reply.text}",
                reply.action,
                reply.question_field,
            )
        return reply, True

    async def _start_transfer(self) -> None:
        if self._task_group is None:
            raise RuntimeError("Call session task group is not active")
        if self._transfer_task is not None:
            raise RuntimeError("Call transfer is already active")
        self._transfer_task = self._task_group.create_task(
            self._transfer_and_notify()
        )

    async def _transfer_and_notify(self) -> None:
        try:
            await self.transport.transfer()
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            await self._queue.put(_TransferFinished(error=error))
        else:
            await self._queue.put(_TransferFinished())

    async def _interrupt_playback(self, *, count: bool = True) -> bool:
        task = self._playback_task
        if task is None or task.done():
            return False
        if count:
            self.interruptions += 1
            self.trace.record(TimingEventName.INTERRUPTION)
            if self.conversation.state.ended and self._active_reply is not None:
                self._deferred_reply = self._active_reply
        if self._active_reply is not None:
            self.conversation.mark_agent_reply_delivery(
                self._active_reply.text,
                "interrupted",
            )
        await self.synthesizer.cancel()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self._playback_task = None
        self._active_reply = None
        await self.transport.stop_audio()
        return True

    async def _cancel_active_tasks(self) -> None:
        await self._interrupt_playback(count=False)
        await self._cancel_turn_task()
        await self._cancel_transfer_task()
        self._pending_turns.clear()

    async def _cancel_turn_task(self) -> None:
        if self._turn_task is not None and not self._turn_task.done():
            await self.recognizer.cancel()
            self._turn_task.cancel()
            try:
                await self._turn_task
            except asyncio.CancelledError:
                pass
        self._turn_task = None
        self._active_turn = None

    async def _cancel_transfer_task(self) -> None:
        if self._transfer_task is not None and not self._transfer_task.done():
            self._transfer_task.cancel()
            try:
                await self._transfer_task
            except asyncio.CancelledError:
                pass
        self._transfer_task = None

    async def _persist_do_not_contact_if_needed(self) -> None:
        if (
            not self.conversation.state.do_not_contact
            or self._do_not_contact_persisted
            or self._on_do_not_contact is None
        ):
            return
        await self._on_do_not_contact(self.conversation.state.call_id)
        self._do_not_contact_persisted = True

    @staticmethod
    async def _audio_frames(frames: tuple[AudioFrame, ...]) -> AsyncIterator[AudioFrame]:
        for frame in frames:
            yield frame

    @staticmethod
    async def _notify(callback: Callable[..., Any] | None, value: Any) -> None:
        if callback is None:
            return
        result = callback() if value is None else callback(value)
        if inspect.isawaitable(result):
            await result