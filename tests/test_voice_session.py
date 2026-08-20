from array import array
import asyncio
from pathlib import Path
from dataclasses import replace
import unittest

from speaking_agent.adapters.asr.mock import MockSpeechRecognizer
from speaking_agent.adapters.telephony.mock import MockCallTransport
from speaking_agent.adapters.tts.mock import MockSpeechSynthesizer
from speaking_agent.answering import AnswerKind, HeuristicAnsweringMachineDetector
from speaking_agent.campaign import load_campaign
from speaking_agent.domain import ConversationContext
from speaking_agent.mock_model import MockConversationModel
from speaking_agent.model import ModelInterpretation
from speaking_agent.observability import TimingEventName
from speaking_agent.speech import AudioFrame, PcmFormat, TranscriptEvent
from speaking_agent.turn_detection import EnergyTurnDetector, TurnDetectionConfig
from speaking_agent.transport import TransportError
from speaking_agent.voice_session import CallSession


CAMPAIGN_PATH = Path(__file__).parents[1] / "campaigns" / "property_owner.json"


def audio_frame(amplitude: int) -> AudioFrame:
    samples = array("h", [amplitude] * 320)
    return AudioFrame(data=samples.tobytes(), format=PcmFormat(16_000))


class RecorderProbe:
    def __init__(self) -> None:
        self.artifact = None
        self.prepared: tuple[str, str, int] | None = None
        self.owner_frames: list[AudioFrame] = []
        self.agent_frames: list[AudioFrame] = []
        self.interruptions = 0
        self.closed = False

    async def prepare(self, *, call_id, campaign_id, retention_days) -> None:
        self.prepared = (call_id, campaign_id, retention_days)

    def record_owner_audio(self, frame) -> None:
        self.owner_frames.append(frame)

    def record_agent_audio(self, frame, started_at_monotonic=None) -> None:
        del started_at_monotonic
        self.agent_frames.append(frame)

    async def close(self) -> None:
        self.closed = True


class VoiceCallSessionTests(unittest.IsolatedAsyncioTestCase):
    def test_recording_disabled_accepts_legacy_transport_without_observer(self) -> None:
        class LegacyTransport:
            pass

        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer([]),
            synthesizer=MockSpeechSynthesizer(),
            transport=LegacyTransport(),
        )

        self.assertIsNone(session.audio_recorder)

        with self.assertRaisesRegex(TypeError, "playout observation"):
            CallSession(
                campaign=load_campaign(CAMPAIGN_PATH),
                model=MockConversationModel(),
                recognizer=MockSpeechRecognizer([]),
                synthesizer=MockSpeechSynthesizer(),
                transport=LegacyTransport(),
                audio_recorder=RecorderProbe(),
            )

    async def test_call_session_records_both_audio_channels_and_closes(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        recorder = RecorderProbe()
        transport = MockCallTransport()
        session = CallSession(
            campaign=campaign,
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer(
                [TranscriptEvent(text="not interested", is_final=True)]
            ),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            audio_recorder=recorder,
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.first_audio_sent.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await run_task

        self.assertEqual(result.lead.outcome, "NOT_INTERESTED")
        self.assertEqual(recorder.prepared[1:], (campaign.campaign_id, 30))
        self.assertEqual(len(recorder.owner_frames), 7)
        self.assertGreater(len(recorder.agent_frames), 0)
        self.assertTrue(recorder.closed)

    def test_call_session_forwards_personalized_context(self) -> None:
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer([]),
            synthesizer=MockSpeechSynthesizer(),
            transport=MockCallTransport(),
            conversation_context=ConversationContext(
                recipient_name="Mr. Ahmed",
                property_reference="your apartment in Marina Gate",
            ),
        )

        opening = session.conversation.start()

        self.assertIn("Mr. Ahmed", opening.text)
        self.assertIn("automated assistant", opening.text)

    async def test_empty_asr_turn_is_ignored_and_listening_continues(self) -> None:
        class SequencedRecognizer:
            def __init__(self) -> None:
                self.responses = iter(("", "not interested"))
                self.calls = 0

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                return None

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                self.calls += 1
                yield TranscriptEvent(text=next(self.responses), is_final=True)

        recognizer = SequencedRecognizer()
        transcripts: list[str] = []
        transport = MockCallTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=recognizer,
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            on_transcript=transcripts.append,
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.first_audio_sent.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        while recognizer.calls < 1 or session._turn_task is not None:
            await asyncio.sleep(0)

        self.assertFalse(run_task.done())
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await run_task

        self.assertEqual(recognizer.calls, 2)
        self.assertEqual(transcripts, ["not interested"])
        self.assertEqual(result.lead.outcome, "NOT_INTERESTED")

    async def test_queued_turn_runs_after_empty_asr_turn(self) -> None:
        class BlockingEmptyThenValidRecognizer:
            def __init__(self) -> None:
                self.first_started = asyncio.Event()
                self.release_first = asyncio.Event()
                self.calls = 0

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                self.release_first.set()

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                self.calls += 1
                if self.calls == 1:
                    self.first_started.set()
                    await self.release_first.wait()
                    text = ""
                else:
                    text = "not interested"
                yield TranscriptEvent(text=text, is_final=True)

        recognizer = BlockingEmptyThenValidRecognizer()
        transport = MockCallTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=recognizer,
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.first_audio_sent.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        await recognizer.first_started.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        recognizer.release_first.set()

        result = await asyncio.wait_for(run_task, timeout=1)

        self.assertEqual(recognizer.calls, 2)
        self.assertEqual(result.lead.outcome, "NOT_INTERESTED")

    async def test_observers_receive_transcript_and_spoken_replies(self) -> None:
        transcripts: list[str] = []
        spoken_replies: list[str] = []
        campaign = load_campaign(CAMPAIGN_PATH)
        transport = MockCallTransport()
        session = CallSession(
            campaign=campaign,
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer(
                [TranscriptEvent(text="not interested", is_final=True)]
            ),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            on_transcript=transcripts.append,
            on_agent_reply=lambda reply: spoken_replies.append(reply.text),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.first_audio_sent.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        await run_task

        self.assertEqual(transcripts, ["not interested"])
        self.assertIn(spoken_replies[0], (campaign.opening, *campaign.opening_variants))
        self.assertTrue(
            spoken_replies[-1].endswith(
                campaign.closing_messages["NOT_INTERESTED"]
            )
        )

    async def test_recognizer_receives_campaign_language_and_context(self) -> None:
        class ContextRecognizer:
            def __init__(self) -> None:
                self.language = None
                self.context = None

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                return None

            async def transcribe(self, audio, *, language=None, context=""):
                async for _ in audio:
                    pass
                self.language = language
                self.context = context
                yield TranscriptEvent(text="not interested", is_final=True)

        recognizer = ContextRecognizer()
        transport = MockCallTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=recognizer,
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            recognition_language="English",
            recognition_context="Dubai Marina; Jumeirah Village Circle",
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.first_audio_sent.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        await run_task

        self.assertEqual(recognizer.language, "English")
        self.assertIn("Jumeirah Village Circle", recognizer.context)

    async def test_confirmed_barge_in_stops_audio_and_completes_hard_stop(self) -> None:
        interruptions: list[str] = []
        recorder = RecorderProbe()
        transport = MockCallTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer(
                [TranscriptEvent(text="I'm not interested", is_final=True)]
            ),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            on_interruption=lambda: interruptions.append("detected"),
            audio_recorder=recorder,
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.connected_event.wait()
        self.assertEqual(transport.sent_audio, [])
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await run_task

        self.assertEqual(result.lead.outcome, "NOT_INTERESTED")
        self.assertEqual(result.answer_kind, AnswerKind.HUMAN)
        self.assertEqual(result.interruptions, 1)
        self.assertEqual(interruptions, ["detected"])
        self.assertGreater(len(recorder.agent_frames), 0)
        agent_turns = [
            turn
            for turn in session.conversation.state.recent_dialogue
            if turn["role"] == "agent"
        ]
        self.assertEqual(agent_turns[0]["delivery"], "interrupted")
        self.assertEqual(agent_turns[-1]["delivery"], "delivered")
        self.assertGreaterEqual(transport.stop_audio_count, 1)
        self.assertTrue(transport.hung_up)
        self.assertTrue(transport.closed)
        self.assertIsNotNone(
            session.trace.latest_duration(
                TimingEventName.SPEECH_END,
                TimingEventName.PLAYBACK_START,
            )
        )

    async def test_confirmation_playback_ignores_echo_turn_without_replay(self) -> None:
        class RecordingSynthesizer(MockSpeechSynthesizer):
            def __init__(self) -> None:
                super().__init__()
                self.texts: list[str] = []

            async def synthesize(self, text, **kwargs):
                self.texts.append(text)
                async for frame in super().synthesize(text, **kwargs):
                    yield frame

        interruptions: list[str] = []
        transcripts: list[str] = []
        transport = MockCallTransport()
        transport.playout_release.clear()
        synthesizer = RecordingSynthesizer()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer(
                [TranscriptEvent(text="Ah", is_final=True)]
            ),
            synthesizer=synthesizer,
            transport=transport,
            conversation_context=ConversationContext(
                recipient_name="Mr. Ahmed",
                property_reference="your apartment in Marina Gate",
            ),
            on_interruption=lambda: interruptions.append("detected"),
            on_transcript=transcripts.append,
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        while transport.playout_wait_count < 1:
            await asyncio.sleep(0)
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        await asyncio.sleep(0)

        self.assertEqual(interruptions, [])
        self.assertEqual(transport.stop_audio_count, 0)
        self.assertEqual(len(synthesizer.texts), 1)

        while not session._pending_turns:
            await asyncio.sleep(0)
        transport.playout_release.set()
        while session._turn_task is not None or session._pending_turns:
            await asyncio.sleep(0)
        await transport.disconnect()
        result = await run_task

        self.assertEqual(result.interruptions, 0)
        self.assertEqual(len(synthesizer.texts), 1)
        self.assertEqual(transcripts, [])
        self.assertEqual(result.lead.outcome, "UNKNOWN")

    async def test_confirmation_playback_queues_meaningful_reply_until_delivered(self) -> None:
        class RecordingSynthesizer(MockSpeechSynthesizer):
            def __init__(self) -> None:
                super().__init__()
                self.texts: list[str] = []

            async def synthesize(self, text, **kwargs):
                self.texts.append(text)
                async for frame in super().synthesize(text, **kwargs):
                    yield frame

        interruptions: list[str] = []
        transport = MockCallTransport()
        transport.playout_release.clear()
        synthesizer = RecordingSynthesizer()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer(
                [TranscriptEvent(text="Yes", is_final=True)]
            ),
            synthesizer=synthesizer,
            transport=transport,
            conversation_context=ConversationContext(
                recipient_name="Mr. Ahmed",
                property_reference="your apartment in Marina Gate",
            ),
            on_interruption=lambda: interruptions.append("detected"),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        while transport.playout_wait_count < 1:
            await asyncio.sleep(0)
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        while not session._pending_turns:
            await asyncio.sleep(0)
        transport.playout_release.set()
        while transport.playout_wait_count < 2:
            await asyncio.sleep(0)

        self.assertEqual(interruptions, [])
        self.assertEqual(transport.stop_audio_count, 0)
        self.assertEqual(len(synthesizer.texts), 2)
        self.assertIn("Marina Gate", synthesizer.texts[1])

        await transport.disconnect()
        result = await run_task

        self.assertEqual(result.interruptions, 0)
        self.assertEqual(result.lead.outcome, "UNKNOWN")

    async def test_multiple_confirmation_overlaps_preserve_later_denial(self) -> None:
        class SequencedRecognizer:
            def __init__(self) -> None:
                self.responses = iter(("Yes", "This is Ali"))

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                return None

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                yield TranscriptEvent(text=next(self.responses), is_final=True)

        class RecordingSynthesizer(MockSpeechSynthesizer):
            def __init__(self) -> None:
                super().__init__()
                self.texts: list[str] = []

            async def synthesize(self, text, **kwargs):
                self.texts.append(text)
                async for frame in super().synthesize(text, **kwargs):
                    yield frame

        transport = MockCallTransport()
        transport.playout_release.clear()
        synthesizer = RecordingSynthesizer()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=SequencedRecognizer(),
            synthesizer=synthesizer,
            transport=transport,
            conversation_context=ConversationContext(
                recipient_name="Mr. Ahmed",
                property_reference="your apartment in Marina Gate",
            ),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        while transport.playout_wait_count < 1:
            await asyncio.sleep(0)
        for _ in range(2):
            for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
                await transport.emit_audio(audio_frame(amplitude))
        while len(session._pending_turns) < 2:
            await asyncio.sleep(0)
        transport.playout_release.set()
        result = await run_task

        self.assertEqual(result.lead.outcome, "WRONG_NUMBER")
        self.assertTrue(all("Marina Gate" not in text for text in synthesizer.texts))

    async def test_inflight_denial_blocks_earlier_confirmation_reply(self) -> None:
        class BlockingFirstRecognizer:
            def __init__(self) -> None:
                self.responses = iter(("Yes", "This is Ali"))
                self.first_started = asyncio.Event()
                self.release_first = asyncio.Event()
                self.calls = 0

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                self.release_first.set()

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                self.calls += 1
                if self.calls == 1:
                    self.first_started.set()
                    await self.release_first.wait()
                yield TranscriptEvent(text=next(self.responses), is_final=True)

        class RecordingSynthesizer(MockSpeechSynthesizer):
            def __init__(self) -> None:
                super().__init__()
                self.texts: list[str] = []

            async def synthesize(self, text, **kwargs):
                self.texts.append(text)
                async for frame in super().synthesize(text, **kwargs):
                    yield frame

        transport = MockCallTransport()
        synthesizer = RecordingSynthesizer()
        recognizer = BlockingFirstRecognizer()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=recognizer,
            synthesizer=synthesizer,
            transport=transport,
            conversation_context=ConversationContext(
                recipient_name="Mr. Ahmed",
                property_reference="your apartment in Marina Gate",
            ),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.first_audio_sent.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        await recognizer.first_started.wait()
        for amplitude in (5_000, 5_000):
            await transport.emit_audio(audio_frame(amplitude))
        while not session.turn_detector.has_candidate_speech:
            await asyncio.sleep(0)

        recognizer.release_first.set()
        while session._deferred_reply is None:
            await asyncio.sleep(0)
        self.assertTrue(all("Marina Gate" not in text for text in synthesizer.texts))

        for amplitude in (5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await run_task

        self.assertEqual(result.lead.outcome, "WRONG_NUMBER")
        self.assertTrue(all("Marina Gate" not in text for text in synthesizer.texts))

    async def test_ignored_overlap_does_not_release_reply_during_new_candidate(self) -> None:
        class BlockingRecognizer:
            def __init__(self) -> None:
                self.responses = iter(("Yes", "Ah", "This is Ali"))
                self.second_started = asyncio.Event()
                self.release_second = asyncio.Event()
                self.calls = 0

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                self.release_second.set()

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                self.calls += 1
                if self.calls == 2:
                    self.second_started.set()
                    await self.release_second.wait()
                yield TranscriptEvent(text=next(self.responses), is_final=True)

        class RecordingSynthesizer(MockSpeechSynthesizer):
            def __init__(self) -> None:
                super().__init__()
                self.texts: list[str] = []

            async def synthesize(self, text, **kwargs):
                self.texts.append(text)
                async for frame in super().synthesize(text, **kwargs):
                    yield frame

        transport = MockCallTransport()
        transport.playout_release.clear()
        synthesizer = RecordingSynthesizer()
        recognizer = BlockingRecognizer()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=recognizer,
            synthesizer=synthesizer,
            transport=transport,
            conversation_context=ConversationContext(
                recipient_name="Mr. Ahmed",
                property_reference="your apartment in Marina Gate",
            ),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        while transport.playout_wait_count < 1:
            await asyncio.sleep(0)
        for _ in range(2):
            for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
                await transport.emit_audio(audio_frame(amplitude))
        while len(session._pending_turns) < 2:
            await asyncio.sleep(0)
        transport.playout_release.set()
        await recognizer.second_started.wait()
        while session._deferred_reply is None:
            await asyncio.sleep(0)

        for amplitude in (5_000, 5_000):
            await transport.emit_audio(audio_frame(amplitude))
        while not session.turn_detector.has_candidate_speech:
            await asyncio.sleep(0)
        recognizer.release_second.set()
        while session._turn_task is not None:
            await asyncio.sleep(0)

        self.assertTrue(all("Marina Gate" not in text for text in synthesizer.texts))

        for amplitude in (5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await run_task

        self.assertEqual(result.lead.outcome, "WRONG_NUMBER")
        self.assertTrue(all("Marina Gate" not in text for text in synthesizer.texts))

    async def test_repeated_confirmation_echo_noise_is_bounded_without_replay(self) -> None:
        class SequencedRecognizer:
            def __init__(self) -> None:
                self.responses = iter(("Ah", "Hi"))

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                return None

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                yield TranscriptEvent(text=next(self.responses), is_final=True)

        class RecordingSynthesizer(MockSpeechSynthesizer):
            def __init__(self) -> None:
                super().__init__()
                self.texts: list[str] = []

            async def synthesize(self, text, **kwargs):
                self.texts.append(text)
                async for frame in super().synthesize(text, **kwargs):
                    yield frame

        transport = MockCallTransport()
        transport.playout_release.clear()
        synthesizer = RecordingSynthesizer()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=SequencedRecognizer(),
            synthesizer=synthesizer,
            transport=transport,
            conversation_context=ConversationContext(
                recipient_name="Mr. Ahmed",
                property_reference="your apartment in Marina Gate",
            ),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        while transport.playout_wait_count < 1:
            await asyncio.sleep(0)
        for _ in range(2):
            for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
                await transport.emit_audio(audio_frame(amplitude))
        while len(session._pending_turns) < 2:
            await asyncio.sleep(0)
        transport.playout_release.set()
        result = await run_task

        self.assertEqual(result.lead.outcome, "UNKNOWN")
        self.assertEqual(synthesizer.texts.count(session.conversation.opening), 1)
        self.assertEqual(result.interruptions, 0)
        self.assertTrue(transport.hung_up)

    async def test_dnc_interrupts_personalized_terminal_closing(self) -> None:
        class SequencedRecognizer:
            def __init__(self) -> None:
                self.responses = iter(("not interested", "do not call me again"))

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                return None

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                yield TranscriptEvent(text=next(self.responses), is_final=True)

        persisted: list[str] = []
        transport = MockCallTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=SequencedRecognizer(),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            conversation_context=ConversationContext(
                recipient_name="Mr. Ahmed",
                property_reference="your apartment in Marina Gate",
            ),
            on_do_not_contact=lambda call_id: self._append_async(
                persisted,
                call_id,
            ),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.first_audio_sent.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        while session.conversation.state.outcome != "NOT_INTERESTED":
            await asyncio.sleep(0)
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await run_task

        self.assertEqual(result.lead.outcome, "DO_NOT_CONTACT")
        self.assertTrue(session.conversation.state.do_not_contact)
        self.assertEqual(persisted, [session.conversation.state.call_id])

    async def test_late_subthreshold_dnc_delays_terminal_hangup(self) -> None:
        class SequencedRecognizer:
            def __init__(self) -> None:
                self.responses = iter(("not interested", "do not call me again"))

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                return None

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                yield TranscriptEvent(text=next(self.responses), is_final=True)

        persisted: list[str] = []
        transport = MockCallTransport()
        detector = EnergyTurnDetector(
            TurnDetectionConfig(
                energy_threshold=0.02,
                minimum_speech_ms=60,
                end_silence_ms=60,
            )
        )
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=SequencedRecognizer(),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            on_do_not_contact=lambda call_id: self._append_async(
                persisted,
                call_id,
            ),
            turn_detector=detector,
        )

        run_task = asyncio.create_task(session.run())
        while transport.playout_wait_count < 1:
            await asyncio.sleep(0)
        transport.playout_release.clear()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        while transport.playout_wait_count < 2:
            await asyncio.sleep(0)

        for _ in range(2):
            await transport.emit_audio(audio_frame(5_000))
        while not detector.has_pending_speech:
            await asyncio.sleep(0)
        transport.playout_release.set()
        await asyncio.sleep(0)

        self.assertFalse(transport.hung_up)

        for amplitude in (5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await asyncio.wait_for(run_task, timeout=1)

        self.assertEqual(result.lead.outcome, "DO_NOT_CONTACT")
        self.assertTrue(session.conversation.state.do_not_contact)
        self.assertEqual(persisted, [session.conversation.state.call_id])

    async def test_confirmed_late_dnc_asr_outlives_candidate_grace(self) -> None:
        class SlowSecondRecognizer:
            def __init__(self) -> None:
                self.calls = 0
                self.second_started = asyncio.Event()
                self.release_second = asyncio.Event()

            async def prepare(self):
                return None

            async def close(self):
                self.release_second.set()

            async def cancel(self):
                self.release_second.set()

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                self.calls += 1
                if self.calls == 1:
                    text = "not interested"
                else:
                    self.second_started.set()
                    await self.release_second.wait()
                    text = "do not call me again"
                yield TranscriptEvent(text=text, is_final=True)

        persisted: list[str] = []
        transport = MockCallTransport()
        recognizer = SlowSecondRecognizer()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=recognizer,
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            on_do_not_contact=lambda call_id: self._append_async(
                persisted,
                call_id,
            ),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )
        session._terminal_candidate_grace_seconds = lambda: 0.01

        run_task = asyncio.create_task(session.run())
        while transport.playout_wait_count < 1:
            await asyncio.sleep(0)
        transport.playout_release.clear()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        while transport.playout_wait_count < 2:
            await asyncio.sleep(0)
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        await recognizer.second_started.wait()

        transport.playout_release.set()
        asyncio.get_running_loop().call_later(
            0.03,
            recognizer.release_second.set,
        )
        result = await asyncio.wait_for(run_task, timeout=1)

        self.assertEqual(result.lead.outcome, "DO_NOT_CONTACT")
        self.assertEqual(persisted, [session.conversation.state.call_id])

    async def test_trailing_confirmation_noise_cannot_overwrite_denial(self) -> None:
        class SequencedRecognizer:
            def __init__(self) -> None:
                self.responses = iter(("Ah", "This is Ali", "Hi"))

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                return None

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                yield TranscriptEvent(text=next(self.responses), is_final=True)

        transport = MockCallTransport()
        transport.playout_release.clear()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=SequencedRecognizer(),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            conversation_context=ConversationContext(
                recipient_name="Mr. Ahmed",
                property_reference="your apartment in Marina Gate",
            ),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        while transport.playout_wait_count < 1:
            await asyncio.sleep(0)
        for _ in range(3):
            for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
                await transport.emit_audio(audio_frame(amplitude))
        while len(session._pending_turns) < 3:
            await asyncio.sleep(0)
        transport.playout_release.set()
        result = await run_task

        self.assertEqual(result.lead.outcome, "WRONG_NUMBER")

    async def test_remote_disconnect_cancels_resources_and_returns_unknown(self) -> None:
        transport = MockCallTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer([]),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
        )

        run_task = asyncio.create_task(session.run())
        await transport.first_audio_sent.wait()
        await transport.disconnect("participant left")
        result = await run_task

        self.assertEqual(result.lead.outcome, "UNKNOWN")
        self.assertTrue(result.disconnected)
        self.assertTrue(transport.closed)

    async def test_disconnect_flushes_and_persists_buffered_do_not_contact(self) -> None:
        persisted: list[str] = []
        transport = MockCallTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer(
                [TranscriptEvent(text="do not call me again", is_final=True)]
            ),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            answering_detector=HeuristicAnsweringMachineDetector(),
            on_do_not_contact=lambda call_id: self._append_async(persisted, call_id),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=600,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.connected_event.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000):
            await transport.emit_audio(audio_frame(amplitude))
        await transport.disconnect()
        result = await run_task

        self.assertEqual(result.lead.outcome, "DO_NOT_CONTACT")
        self.assertEqual(result.answer_kind, AnswerKind.HUMAN)
        self.assertTrue(result.disconnected)
        self.assertEqual(persisted, [session.conversation.state.call_id])

    async def test_voicemail_uses_separate_message_and_skips_qualification(self) -> None:
        transport = MockCallTransport()
        campaign = load_campaign(CAMPAIGN_PATH)
        session = CallSession(
            campaign=campaign,
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer(
                [
                    TranscriptEvent(
                        text="Please leave a message after the tone",
                        is_final=True,
                    )
                ]
            ),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            answering_detector=HeuristicAnsweringMachineDetector(),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.connected_event.wait()
        self.assertEqual(transport.sent_audio, [])
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await run_task

        self.assertEqual(result.answer_kind, AnswerKind.VOICEMAIL)
        self.assertEqual(result.lead.outcome, "UNKNOWN")
        self.assertTrue(transport.hung_up)
        self.assertGreater(len(transport.sent_audio), 1)

    async def test_voicemail_without_message_hangs_up_silently(self) -> None:
        transport = MockCallTransport()
        campaign = replace(load_campaign(CAMPAIGN_PATH), voicemail_message=None)
        session = CallSession(
            campaign=campaign,
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer(
                [
                    TranscriptEvent(
                        text="Please leave a message after the tone",
                        is_final=True,
                    )
                ]
            ),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            answering_detector=HeuristicAnsweringMachineDetector(),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.connected_event.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await run_task

        self.assertEqual(result.answer_kind, AnswerKind.VOICEMAIL)
        self.assertEqual(transport.sent_audio, [])
        self.assertTrue(transport.hung_up)

    async def test_ivr_answer_closes_without_entering_conversation(self) -> None:
        transport = MockCallTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer(
                [TranscriptEvent(text="Press one for sales", is_final=True)]
            ),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            answering_detector=HeuristicAnsweringMachineDetector(),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.connected_event.wait()
        self.assertEqual(transport.sent_audio, [])
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await run_task

        self.assertEqual(result.answer_kind, AnswerKind.IVR)
        self.assertEqual(result.lead.outcome, "UNKNOWN")
        self.assertTrue(transport.hung_up)

    async def test_long_human_response_is_not_treated_as_a_machine(self) -> None:
        detector = HeuristicAnsweringMachineDetector()

        result = await detector.classify(
            "Hello yes I have been thinking about selling the apartment sometime this year"
        )

        self.assertEqual(result, AnswerKind.HUMAN)

    async def test_long_do_not_contact_request_precedes_machine_detection(self) -> None:
        class RecordingSynthesizer(MockSpeechSynthesizer):
            def __init__(self) -> None:
                super().__init__()
                self.texts: list[str] = []

            async def synthesize(self, text, **kwargs):
                self.texts.append(text)
                async for frame in super().synthesize(text, **kwargs):
                    yield frame

        transport = MockCallTransport()
        synthesizer = RecordingSynthesizer()
        campaign = load_campaign(CAMPAIGN_PATH)
        session = CallSession(
            campaign=campaign,
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer(
                [
                    TranscriptEvent(
                        text=(
                            "I am not interested in this property enquiry and please "
                            "do not call me or contact this number again"
                        ),
                        is_final=True,
                    )
                ]
            ),
            synthesizer=synthesizer,
            transport=transport,
            answering_detector=HeuristicAnsweringMachineDetector(),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.connected_event.wait()
        self.assertEqual(transport.sent_audio, [])
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await run_task

        self.assertEqual(result.answer_kind, AnswerKind.HUMAN)
        self.assertEqual(result.lead.outcome, "DO_NOT_CONTACT")
        self.assertTrue(synthesizer.texts[-1].startswith(campaign.introduction))

    async def test_human_greeting_unlocks_disclosed_opening(self) -> None:
        transport = MockCallTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer(
                [TranscriptEvent(text="Hello", is_final=True)]
            ),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            answering_detector=HeuristicAnsweringMachineDetector(),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.connected_event.wait()
        self.assertEqual(transport.sent_audio, [])
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        await transport.first_audio_sent.wait()
        await transport.disconnect()
        result = await run_task

        self.assertEqual(result.answer_kind, AnswerKind.HUMAN)
        self.assertTrue(session.conversation.state.asked_fields)

    async def test_first_human_turn_preserves_volunteered_intent(self) -> None:
        transport = MockCallTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer(
                [
                    TranscriptEvent(
                        text="Hello, I might sell my apartment in two months",
                        is_final=True,
                    )
                ]
            ),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            answering_detector=HeuristicAnsweringMachineDetector(),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.connected_event.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        await transport.first_audio_sent.wait()
        await transport.disconnect()
        result = await run_task

        self.assertEqual(result.lead.outcome, "SELL")
        self.assertEqual(result.lead.fields["property_type"], "apartment")
        self.assertEqual(result.lead.fields["selling_timeline"], "in two months")

    async def test_personalized_outbound_confirms_recipient_after_human_detection(self) -> None:
        class QueuedAffirmativeRecognizer:
            def __init__(self) -> None:
                self.responses = iter(("Yes?", "Yes?", "This is Ali"))
                self.first_started = asyncio.Event()
                self.release_first = asyncio.Event()
                self.calls = 0

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                self.release_first.set()

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                self.calls += 1
                if self.calls == 1:
                    self.first_started.set()
                    await self.release_first.wait()
                yield TranscriptEvent(text=next(self.responses), is_final=True)

        class RecordingSynthesizer(MockSpeechSynthesizer):
            def __init__(self) -> None:
                super().__init__()
                self.texts: list[str] = []

            async def synthesize(self, text, **kwargs):
                self.texts.append(text)
                async for frame in super().synthesize(text, **kwargs):
                    yield frame

        transport = MockCallTransport()
        synthesizer = RecordingSynthesizer()
        recognizer = QueuedAffirmativeRecognizer()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=recognizer,
            synthesizer=synthesizer,
            transport=transport,
            answering_detector=HeuristicAnsweringMachineDetector(),
            conversation_context=ConversationContext(
                recipient_name="Mr. Ahmed",
                property_reference="your apartment in Marina Gate",
            ),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.connected_event.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        await recognizer.first_started.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        while not session._pending_turns:
            await asyncio.sleep(0)
        recognizer.release_first.set()
        while transport.playout_wait_count < 1:
            await asyncio.sleep(0)

        self.assertIn("Mr. Ahmed", synthesizer.texts[0])
        self.assertNotIn("Marina Gate", synthesizer.texts[0])

        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await run_task

        self.assertEqual(result.lead.outcome, "WRONG_NUMBER")
        self.assertTrue(all("Marina Gate" not in text for text in synthesizer.texts))

    async def test_second_outbound_human_turn_is_processed_without_reintroducing(self) -> None:
        class SequencedRecognizer:
            def __init__(self) -> None:
                self.responses = iter(("Hello, I want to sell", "not interested"))

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                return None

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                yield TranscriptEvent(text=next(self.responses), is_final=True)

        transport = MockCallTransport()
        campaign = load_campaign(CAMPAIGN_PATH)
        session = CallSession(
            campaign=campaign,
            model=MockConversationModel(),
            recognizer=SequencedRecognizer(),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            answering_detector=HeuristicAnsweringMachineDetector(),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.connected_event.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        while transport.playout_wait_count < 1:
            await asyncio.sleep(0)
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await run_task

        self.assertEqual(result.lead.outcome, "NOT_INTERESTED")
        self.assertEqual(result.interruptions, 0)

    async def test_second_utterance_queues_while_first_turn_is_processing(self) -> None:
        class SequencedRecognizer:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.release = asyncio.Event()
                self.calls = 0

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                self.release.set()

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                index = self.calls
                self.calls += 1
                if index == 0:
                    self.started.set()
                    await self.release.wait()
                    text = "I want to sell"
                else:
                    text = "Actually please do not call me again"
                yield TranscriptEvent(text=text, is_final=True)

        recognizer = SequencedRecognizer()
        transport = MockCallTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=recognizer,
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.first_audio_sent.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        await recognizer.started.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        recognizer.release.set()
        result = await run_task

        self.assertEqual(recognizer.calls, 2)
        self.assertEqual(result.lead.outcome, "DO_NOT_CONTACT")

    async def test_queued_turn_sees_prior_unsent_reply_as_pending(self) -> None:
        class SequencedRecognizer:
            def __init__(self) -> None:
                self.first_started = asyncio.Event()
                self.release_first = asyncio.Event()
                self.responses = iter(("sell", "Dubai Marina"))
                self.calls = 0

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                self.release_first.set()

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                self.calls += 1
                if self.calls == 1:
                    self.first_started.set()
                    await self.release_first.wait()
                yield TranscriptEvent(text=next(self.responses), is_final=True)

        observed_delivery: list[str | None] = []
        observed_last_fields: list[str | None] = []

        class DeliveryModel:
            async def prepare(self):
                return None

            async def close(self):
                return None

            async def interpret(self, utterance, state, campaign):
                del campaign
                if utterance == "sell":
                    return ModelInterpretation(suggested_outcome="SELL")
                observed_delivery.append(
                    next(
                        (
                            turn.get("delivery")
                            for turn in reversed(state.recent_dialogue)
                            if turn.get("role") == "agent"
                        ),
                        None,
                    )
                )
                observed_last_fields.append(state.last_asked_field)
                return ModelInterpretation(
                    field_updates={"property_location": "Dubai Marina"}
                )

        recognizer = SequencedRecognizer()
        transport = MockCallTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=DeliveryModel(),
            recognizer=recognizer,
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.first_audio_sent.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        await recognizer.first_started.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        recognizer.release_first.set()
        while recognizer.calls < 2:
            await asyncio.sleep(0)
        await transport.disconnect()
        await run_task

        self.assertEqual(observed_delivery, ["pending"])
        self.assertEqual(observed_last_fields, ["intent"])

    async def test_terminal_action_waits_for_transport_playout(self) -> None:
        transport = MockCallTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer(
                [TranscriptEvent(text="not interested", is_final=True)]
            ),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        while transport.playout_wait_count < 1:
            await asyncio.sleep(0)
        transport.playout_release.clear()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        while transport.playout_wait_count < 2:
            await asyncio.sleep(0)

        self.assertFalse(transport.hung_up)
        transport.playout_release.set()
        await run_task
        self.assertTrue(transport.hung_up)
        self.assertGreater(transport.sent_counts_at_playout[0], 1)
        self.assertEqual(
            transport.sent_counts_at_playout[-1],
            len(transport.sent_audio),
        )

    async def test_asr_failure_is_not_reported_as_completed_call(self) -> None:
        class FailingRecognizer:
            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                return None

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                raise RuntimeError("ASR failed")
                yield

        transport = MockCallTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=FailingRecognizer(),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.first_audio_sent.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))

        with self.assertRaisesRegex(RuntimeError, "ASR failed"):
            await run_task
        self.assertTrue(transport.hung_up)

    async def test_outbound_silence_is_bounded_by_initial_answer_timeout(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        campaign = replace(
            campaign,
            behavior={
                **campaign.behavior,
                "initial_answer_timeout_seconds": 0.01,
            },
        )
        transport = MockCallTransport()
        session = CallSession(
            campaign=campaign,
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer([]),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            answering_detector=HeuristicAnsweringMachineDetector(),
        )

        result = await asyncio.wait_for(session.run(), timeout=0.2)

        self.assertEqual(result.answer_kind, AnswerKind.UNCERTAIN)
        self.assertEqual(result.lead.outcome, "UNKNOWN")
        self.assertTrue(transport.hung_up)

    async def test_idle_call_after_opening_is_bounded(self) -> None:
        campaign = load_campaign(CAMPAIGN_PATH)
        campaign = replace(
            campaign,
            behavior={
                **campaign.behavior,
                "conversation_idle_timeout_seconds": 0.01,
            },
        )
        transport = MockCallTransport()
        session = CallSession(
            campaign=campaign,
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer([]),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
        )

        result = await asyncio.wait_for(session.run(), timeout=0.2)

        self.assertEqual(result.lead.outcome, "UNKNOWN")
        self.assertTrue(transport.hung_up)

    async def test_cleanup_failure_does_not_replace_valid_result(self) -> None:
        class FailingCloseTransport(MockCallTransport):
            async def close(self) -> None:
                self.closed = True
                raise RuntimeError("close failed")

        transport = FailingCloseTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer(
                [TranscriptEvent(text="not interested", is_final=True)]
            ),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.first_audio_sent.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await run_task

        self.assertEqual(result.lead.outcome, "NOT_INTERESTED")
        self.assertEqual(result.cleanup_errors, ("transport:RuntimeError",))

    async def test_do_not_contact_overrides_terminal_closing(self) -> None:
        class SequencedRecognizer:
            def __init__(self) -> None:
                self.responses = iter(("not interested", "do not call me again"))

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                return None

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                yield TranscriptEvent(text=next(self.responses), is_final=True)

        persisted: list[str] = []
        transport = MockCallTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=SequencedRecognizer(),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            on_do_not_contact=lambda call_id: self._append_async(persisted, call_id),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        while transport.playout_wait_count < 1:
            await asyncio.sleep(0)
        transport.playout_release.clear()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        while session.conversation.state.outcome != "NOT_INTERESTED":
            await asyncio.sleep(0)
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        while session.conversation.state.outcome != "DO_NOT_CONTACT":
            await asyncio.sleep(0)
        transport.playout_release.set()
        result = await run_task

        self.assertEqual(result.lead.outcome, "DO_NOT_CONTACT")
        self.assertTrue(session.conversation.state.do_not_contact)
        self.assertEqual(persisted, [session.conversation.state.call_id])

    async def test_do_not_contact_persists_before_acknowledgement_and_hangup(self) -> None:
        callback_started = asyncio.Event()
        callback_release = asyncio.Event()

        async def persist(call_id: str) -> None:
            self.assertEqual(call_id, session.conversation.state.call_id)
            callback_started.set()
            await callback_release.wait()

        transport = MockCallTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer(
                [TranscriptEvent(text="do not call me again", is_final=True)]
            ),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            on_do_not_contact=persist,
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.first_audio_sent.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        await callback_started.wait()
        audio_count_while_persisting = len(transport.sent_audio)
        await asyncio.sleep(0)

        self.assertFalse(transport.hung_up)
        self.assertEqual(
            len(transport.sent_audio),
            audio_count_while_persisting,
        )
        callback_release.set()
        result = await run_task
        self.assertEqual(result.lead.outcome, "DO_NOT_CONTACT")

    async def test_hanging_cleanup_is_bounded(self) -> None:
        class HangingCloseSynthesizer(MockSpeechSynthesizer):
            async def close(self) -> None:
                await asyncio.Event().wait()

        campaign = load_campaign(CAMPAIGN_PATH)
        campaign = replace(
            campaign,
            behavior={**campaign.behavior, "cleanup_timeout_seconds": 0.01},
        )
        transport = MockCallTransport()
        session = CallSession(
            campaign=campaign,
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer(
                [TranscriptEvent(text="not interested", is_final=True)]
            ),
            synthesizer=HangingCloseSynthesizer(),
            transport=transport,
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.first_audio_sent.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await asyncio.wait_for(run_task, timeout=0.2)

        self.assertEqual(result.lead.outcome, "NOT_INTERESTED")
        self.assertIn("synthesizer:TimeoutError", result.cleanup_errors)

    async def test_hanging_recorder_finalization_is_bounded_and_reported(self) -> None:
        class HangingRecorder(RecorderProbe):
            async def close(self) -> None:
                await asyncio.Event().wait()

        campaign = load_campaign(CAMPAIGN_PATH)
        campaign = replace(
            campaign,
            behavior={**campaign.behavior, "cleanup_timeout_seconds": 0.01},
        )
        transport = MockCallTransport()
        session = CallSession(
            campaign=campaign,
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer(
                [TranscriptEvent(text="not interested", is_final=True)]
            ),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            audio_recorder=HangingRecorder(),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.first_audio_sent.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await asyncio.wait_for(run_task, timeout=0.2)

        self.assertEqual(result.lead.outcome, "NOT_INTERESTED")
        self.assertIn("audio_recorder:TimeoutError", result.cleanup_errors)

    async def test_unavailable_transfer_falls_back_to_hangup(self) -> None:
        transport = MockCallTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(
                {
                    "transfer": ModelInterpretation(
                        suggested_outcome="HUMAN_TRANSFER",
                    )
                }
            ),
            recognizer=MockSpeechRecognizer(
                [TranscriptEvent(text="transfer", is_final=True)]
            ),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            transfer_available=False,
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.first_audio_sent.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await run_task

        self.assertEqual(result.lead.outcome, "HUMAN_TRANSFER")
        self.assertTrue(transport.hung_up)
        self.assertFalse(transport.transferred)

    async def test_first_human_unavailable_transfer_keeps_disclosure(self) -> None:
        class RecordingSynthesizer(MockSpeechSynthesizer):
            def __init__(self) -> None:
                super().__init__()
                self.texts: list[str] = []

            async def synthesize(self, text, **kwargs):
                self.texts.append(text)
                async for frame in super().synthesize(text, **kwargs):
                    yield frame

        campaign = load_campaign(CAMPAIGN_PATH)
        transport = MockCallTransport()
        synthesizer = RecordingSynthesizer()
        session = CallSession(
            campaign=campaign,
            model=MockConversationModel(
                {
                    "transfer": ModelInterpretation(
                        suggested_outcome="HUMAN_TRANSFER",
                    )
                }
            ),
            recognizer=MockSpeechRecognizer(
                [TranscriptEvent(text="transfer", is_final=True)]
            ),
            synthesizer=synthesizer,
            transport=transport,
            answering_detector=HeuristicAnsweringMachineDetector(),
            transfer_available=False,
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.connected_event.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await run_task

        self.assertEqual(result.answer_kind, AnswerKind.HUMAN)
        self.assertTrue(transport.hung_up)
        self.assertFalse(transport.transferred)
        self.assertTrue(synthesizer.texts[-1].startswith(campaign.introduction))
        self.assertIn(campaign.transfer_unavailable_message, synthesizer.texts[-1])

    async def test_interrupted_opening_repeats_disclosure_before_next_reply(self) -> None:
        class RecordingSynthesizer(MockSpeechSynthesizer):
            def __init__(self) -> None:
                super().__init__()
                self.texts: list[str] = []

            async def synthesize(self, text, **kwargs):
                self.texts.append(text)
                async for frame in super().synthesize(text, **kwargs):
                    yield frame

        campaign = load_campaign(CAMPAIGN_PATH)
        transport = MockCallTransport()
        transport.playout_release.clear()
        synthesizer = RecordingSynthesizer()
        session = CallSession(
            campaign=campaign,
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer(
                [TranscriptEvent(text="I want to sell", is_final=True)]
            ),
            synthesizer=synthesizer,
            transport=transport,
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        while transport.playout_wait_count < 1:
            await asyncio.sleep(0)
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        while transport.playout_wait_count < 2:
            await asyncio.sleep(0)
        transport.playout_release.set()
        await transport.disconnect()
        result = await run_task

        self.assertEqual(result.lead.outcome, "SELL")
        self.assertGreaterEqual(len(synthesizer.texts), 2)
        self.assertTrue(synthesizer.texts[-1].startswith(campaign.introduction))

    async def test_failed_transfer_plays_fallback_then_hangs_up(self) -> None:
        class FailingTransferTransport(MockCallTransport):
            async def transfer(self) -> None:
                self.transferred = True
                raise TransportError("REFER failed")

        class RecordingSynthesizer(MockSpeechSynthesizer):
            def __init__(self) -> None:
                super().__init__()
                self.texts: list[str] = []

            async def synthesize(self, text, **kwargs):
                self.texts.append(text)
                async for frame in super().synthesize(text, **kwargs):
                    yield frame

        campaign = load_campaign(CAMPAIGN_PATH)
        transport = FailingTransferTransport()
        synthesizer = RecordingSynthesizer()
        session = CallSession(
            campaign=campaign,
            model=MockConversationModel(
                {
                    "transfer": ModelInterpretation(
                        suggested_outcome="HUMAN_TRANSFER",
                    )
                }
            ),
            recognizer=MockSpeechRecognizer(
                [TranscriptEvent(text="transfer", is_final=True)]
            ),
            synthesizer=synthesizer,
            transport=transport,
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.first_audio_sent.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await run_task

        self.assertTrue(transport.transferred)
        self.assertTrue(transport.hung_up)
        self.assertIn(campaign.transfer_unavailable_message, synthesizer.texts)
        self.assertIn("transfer:TransportError", result.cleanup_errors)

    async def test_successful_transfer_flushes_pending_do_not_contact(self) -> None:
        class BlockingTransferTransport(MockCallTransport):
            def __init__(self) -> None:
                super().__init__()
                self.transfer_started = asyncio.Event()
                self.transfer_release = asyncio.Event()

            async def transfer(self) -> None:
                self.transferred = True
                self.transfer_started.set()
                await self.transfer_release.wait()

        class SequencedRecognizer:
            def __init__(self) -> None:
                self.responses = iter(("transfer", "do not call me again"))

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                return None

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                yield TranscriptEvent(text=next(self.responses), is_final=True)

        persisted: list[str] = []
        transport = BlockingTransferTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(
                {
                    "transfer": ModelInterpretation(
                        suggested_outcome="HUMAN_TRANSFER",
                    )
                }
            ),
            recognizer=SequencedRecognizer(),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            on_do_not_contact=lambda call_id: self._append_async(persisted, call_id),
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=600,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.first_audio_sent.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, *(0,) * 30):
            await transport.emit_audio(audio_frame(amplitude))
        await transport.transfer_started.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000):
            await transport.emit_audio(audio_frame(amplitude))
        transport.transfer_release.set()
        result = await run_task

        self.assertTrue(transport.transferred)
        self.assertEqual(result.lead.outcome, "DO_NOT_CONTACT")
        self.assertEqual(persisted, [session.conversation.state.call_id])

    async def test_wrong_recipient_override_cancels_inflight_transfer(self) -> None:
        class BlockingTransferTransport(MockCallTransport):
            def __init__(self) -> None:
                super().__init__()
                self.transfer_started = asyncio.Event()
                self.transfer_cancelled = False

            async def transfer(self) -> None:
                self.transfer_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.transfer_cancelled = True
                    raise
                self.transferred = True

        class SequencedRecognizer:
            def __init__(self) -> None:
                self.responses = iter(("transfer", "wrong number"))

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                return None

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                yield TranscriptEvent(text=next(self.responses), is_final=True)

        transport = BlockingTransferTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(
                {
                    "transfer": ModelInterpretation(
                        suggested_outcome="HUMAN_TRANSFER",
                    )
                }
            ),
            recognizer=SequencedRecognizer(),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.first_audio_sent.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        await transport.transfer_started.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await run_task

        self.assertEqual(result.lead.outcome, "WRONG_NUMBER")
        self.assertTrue(transport.transfer_cancelled)
        self.assertFalse(transport.transferred)
        self.assertTrue(transport.hung_up)

    async def test_subthreshold_wrong_number_blocks_fast_transfer_start(self) -> None:
        class SequencedRecognizer:
            def __init__(self) -> None:
                self.responses = iter(("transfer", "wrong number"))

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                return None

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                yield TranscriptEvent(text=next(self.responses), is_final=True)

        transport = MockCallTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(
                {
                    "transfer": ModelInterpretation(
                        suggested_outcome="HUMAN_TRANSFER",
                    )
                }
            ),
            recognizer=SequencedRecognizer(),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        while transport.playout_wait_count < 1:
            await asyncio.sleep(0)
        transport.playout_release.clear()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        while transport.playout_wait_count < 2:
            await asyncio.sleep(0)

        for _ in range(2):
            await transport.emit_audio(audio_frame(5_000))
        while not session.turn_detector.has_candidate_speech:
            await asyncio.sleep(0)
        transport.playout_release.set()
        await asyncio.sleep(0)

        self.assertFalse(transport.transferred)

        for amplitude in (5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await asyncio.wait_for(run_task, timeout=1)

        self.assertEqual(result.lead.outcome, "WRONG_NUMBER")
        self.assertFalse(transport.transferred)
        self.assertTrue(transport.hung_up)

    async def test_zero_frame_tts_is_a_call_failure(self) -> None:
        class EmptySynthesizer(MockSpeechSynthesizer):
            async def synthesize(self, text, **kwargs):
                del text, kwargs
                if False:
                    yield

        transport = MockCallTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer([]),
            synthesizer=EmptySynthesizer(),
            transport=transport,
        )

        with self.assertRaisesRegex(RuntimeError, "no audio"):
            await session.run()
        self.assertTrue(transport.hung_up)
        self.assertIsNone(session.conversation.state.last_asked_field)
        self.assertEqual(session.conversation.state.asked_field_counts, {})

    async def test_transport_stream_failure_is_not_a_normal_disconnect(self) -> None:
        class FailingTransport(MockCallTransport):
            async def events(self):
                raise TransportError("stream failed")
                yield

        transport = FailingTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer([]),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
        )

        with self.assertRaisesRegex(TransportError, "stream failed"):
            await session.run()
        self.assertTrue(transport.hung_up)

    async def test_non_hard_stop_terminal_interruption_resumes_closing(self) -> None:
        class SequencedRecognizer:
            def __init__(self) -> None:
                self.responses = iter(("not interested", "wait a moment"))

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                return None

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                yield TranscriptEvent(text=next(self.responses), is_final=True)

        transport = MockCallTransport()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=SequencedRecognizer(),
            synthesizer=MockSpeechSynthesizer(),
            transport=transport,
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        await transport.first_audio_sent.wait()
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        while session.conversation.state.outcome != "NOT_INTERESTED":
            await asyncio.sleep(0)
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await asyncio.wait_for(run_task, timeout=0.2)

        self.assertEqual(result.lead.outcome, "NOT_INTERESTED")
        self.assertTrue(transport.hung_up)

    async def test_repeated_terminal_interruptions_are_bounded(self) -> None:
        class SequencedRecognizer:
            def __init__(self) -> None:
                self.responses = iter(("not interested", "", ""))

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                return None

            async def transcribe(self, audio, **kwargs):
                del kwargs
                async for _ in audio:
                    pass
                yield TranscriptEvent(text=next(self.responses), is_final=True)

        class ControlledSynthesizer:
            def __init__(self) -> None:
                self.started: asyncio.Queue[tuple[str, asyncio.Event]] = (
                    asyncio.Queue()
                )
                self.texts: list[str] = []

            async def prepare(self):
                return None

            async def close(self):
                return None

            async def cancel(self):
                return None

            async def synthesize(self, text, **kwargs):
                del kwargs
                release = asyncio.Event()
                self.texts.append(text)
                yield audio_frame(0)
                await self.started.put((text, release))
                await release.wait()

        transport = MockCallTransport()
        synthesizer = ControlledSynthesizer()
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=SequencedRecognizer(),
            synthesizer=synthesizer,
            transport=transport,
            turn_detector=EnergyTurnDetector(
                TurnDetectionConfig(
                    energy_threshold=0.02,
                    minimum_speech_ms=60,
                    end_silence_ms=60,
                )
            ),
        )

        run_task = asyncio.create_task(session.run())
        _, opening_release = await asyncio.wait_for(
            synthesizer.started.get(),
            timeout=0.2,
        )
        opening_release.set()
        while session._playback_task is not None:
            await asyncio.sleep(0)

        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        await asyncio.wait_for(synthesizer.started.get(), timeout=0.2)

        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        await asyncio.wait_for(synthesizer.started.get(), timeout=0.2)

        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            await transport.emit_audio(audio_frame(amplitude))
        result = await asyncio.wait_for(run_task, timeout=0.2)

        self.assertEqual(result.lead.outcome, "NOT_INTERESTED")
        self.assertEqual(result.interruptions, 2)
        self.assertEqual(len(synthesizer.texts), 3)
        self.assertTrue(transport.hung_up)

    @staticmethod
    async def _append_async(values: list[str], value: str) -> None:
        values.append(value)


if __name__ == "__main__":
    unittest.main()