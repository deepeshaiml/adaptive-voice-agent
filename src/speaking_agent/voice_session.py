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
        self._active_turn_frames: tuple[AudioFrame, ...] | None = None
        self._pending_turns: deque[tuple[AudioFrame, ...]] = deque()
        self._deferred_reply: AgentReply | None = None
        self._active_reply: AgentReply | None = None
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
                            for turn_event in self.turn_detector.process(event.audio):
                                if turn_event.kind == TurnEventKind.SPEECH_STARTED:
                                    listening_deadline = None
                                    if await self._interrupt_playback():
                                        await self._notify(
                                            self._on_interruption,
                                            None,
                                        )
                                else:
                                    listening_deadline = None
                                    self.trace.record(TimingEventName.SPEECH_END)
                                    await self._start_turn(turn_event.frames)
                    elif isinstance(event, _TurnFinished):
                        self._turn_task = None
                        self._active_turn_frames = None
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
                            if self.conversation.state.do_not_contact:
                                await self._cancel_transfer_task()
                            if self._pending_turns:
                                self._deferred_reply = event.reply
                                self.conversation.mark_agent_reply_delivery(
                                    event.reply.text,
                                    "pending",
                                )
                                await self._start_turn(self._pending_turns.popleft())
                            else:
                                await self._start_playback(event.reply)
                        elif event.ignored:
                            if self._pending_turns:
                                await self._start_turn(self._pending_turns.popleft())
                            elif self._deferred_reply is not None:
                                deferred_reply = self._deferred_reply
                                self._deferred_reply = None
                                await self._start_playback(deferred_reply)
                            else:
                                listening_deadline = event_loop.time() + float(
                                    self.conversation.campaign.behavior[
                                        "conversation_idle_timeout_seconds"
                                    ]
                                )
                    elif isinstance(event, _SilentAction):
                        self._turn_task = None
                        self._active_turn_frames = None
                        if event.action == SessionAction.HANG_UP:
                            await self.transport.hang_up()
                            finished = True
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
                                await self.transport.hang_up()
                                finished = True
                            elif event.action == SessionAction.TRANSFER:
                                await self._start_transfer()
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

    async def _start_turn(self, frames: tuple[AudioFrame, ...]) -> None:
        if self._task_group is None:
            return
        if self._turn_task is not None:
            self._pending_turns.append(frames)
            return
        self._active_turn_frames = frames
        self._turn_task = self._task_group.create_task(
            self._process_turn_and_notify(frames)
        )

    async def _process_turn_and_notify(self, frames: tuple[AudioFrame, ...]) -> None:
        try:
            final_text = await self._transcribe_turn(frames)
            await self._notify(self._on_transcript, final_text)
            hard_stop = self.conversation.policy.hard_stop_outcome(final_text)
            if self.conversation.state.ended and hard_stop is None:
                deferred_reply = self._deferred_reply
                self._deferred_reply = None
                if deferred_reply is not None:
                    await self._queue.put(_TurnFinished(reply=deferred_reply))
                return
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
            self.trace.record(TimingEventName.LLM_START)
            reply = await self.conversation.receive(final_text)
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

        turns: list[tuple[AudioFrame, ...]] = []
        if self._active_turn_frames is not None:
            turns.append(self._active_turn_frames)
        turns.extend(self._pending_turns)
        flushed_turn = self.turn_detector.flush()
        if flushed_turn is not None and flushed_turn.frames:
            turns.append(flushed_turn.frames)

        await self._cancel_turn_task()
        self._pending_turns.clear()
        for frames in turns:
            try:
                final_text = await self._transcribe_turn(frames)
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
        self._active_turn_frames = None

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