from array import array
import asyncio
import math
from pathlib import Path
import unittest

from speaking_agent.adapters.asr.mock import MockSpeechRecognizer
from speaking_agent.adapters.telephony.sounddevice_local import (
    EchoSuppressionConfig,
    OutputEchoSuppressor,
    SoundDeviceCallTransport,
)
from speaking_agent.campaign import load_campaign
from speaking_agent.mock_model import MockConversationModel
from speaking_agent.speech import AudioFrame, PcmFormat
from speaking_agent.speech import TranscriptEvent
from speaking_agent.transport import TransportEventKind
from speaking_agent.turn_detection import EnergyTurnDetector, TurnDetectionConfig
from speaking_agent.voice_session import CallSession


CAMPAIGN_PATH = Path(__file__).parents[1] / "campaigns" / "property_owner.json"


def tone(sample_rate_hz: int, frequency_hz: int, amplitude: int) -> bytes:
    samples = array(
        "h",
        (
            round(
                amplitude
                * math.sin(2 * math.pi * frequency_hz * index / sample_rate_hz)
            )
            for index in range(sample_rate_hz // 50)
        ),
    )
    return samples.tobytes()


def mixed_tones(
    sample_rate_hz: int,
    first_frequency_hz: int,
    first_amplitude: int,
    second_frequency_hz: int,
    second_amplitude: int,
) -> bytes:
    samples = array(
        "h",
        (
            max(
                -32_768,
                min(
                    32_767,
                    round(
                        first_amplitude
                        * math.sin(
                            2 * math.pi * first_frequency_hz * index / sample_rate_hz
                        )
                        + second_amplitude
                        * math.sin(
                            2 * math.pi * second_frequency_hz * index / sample_rate_hz
                        )
                    ),
                ),
            )
            for index in range(sample_rate_hz // 50)
        ),
    )
    return samples.tobytes()


class FakeStream:
    def __init__(self, *, callback=None) -> None:
        self.callback = callback
        self.started = False
        self.stopped = False
        self.closed = False
        self.aborted = False
        self.writes: list[bytes] = []
        self.latency = 0.0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    def abort(self) -> None:
        self.aborted = True
        self.stopped = True

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def pump_output(self, frame_count: int = 480) -> bytes:
        output = bytearray(frame_count * 2)
        self.callback(output, frame_count, object(), None)
        rendered = bytes(output)
        self.writes.append(rendered)
        return rendered


class FakeAudio:
    def __init__(self) -> None:
        self.input_stream: FakeStream | None = None
        self.output_stream: FakeStream | None = None
        self.input_settings = None
        self.output_settings = None

    def check_input_settings(self, **settings) -> None:
        self.input_settings = settings

    def check_output_settings(self, **settings) -> None:
        self.output_settings = settings

    def RawInputStream(self, **settings):
        self.input_stream = FakeStream(callback=settings["callback"])
        return self.input_stream

    def RawOutputStream(self, **settings):
        self.output_stream = FakeStream(callback=settings["callback"])
        return self.output_stream


class SlowSynthesizer:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self._cancelled = asyncio.Event()
        self.cancel_count = 0
        self.calls = 0

    async def prepare(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def cancel(self) -> None:
        self.cancel_count += 1
        self._cancelled.set()

    async def synthesize(self, text, **kwargs):
        del text, kwargs
        self.calls += 1
        self._cancelled.clear()
        frame = AudioFrame(
            data=tone(24_000, 500, 14_000),
            format=PcmFormat(24_000),
        )
        self.started.set()
        if self.calls > 1:
            yield frame
            return
        while not self._cancelled.is_set():
            yield frame
            await asyncio.sleep(0.01)


class OutputEchoSuppressorTests(unittest.TestCase):
    def test_correlated_speaker_echo_is_suppressed(self) -> None:
        now = [10.0]
        suppressor = OutputEchoSuppressor(
            EchoSuppressionConfig(
                barge_in_energy_threshold=0.02,
                echo_correlation_threshold=0.4,
                echo_gain=0.6,
            ),
            clock=lambda: now[0],
        )
        suppressor.add_output(
            AudioFrame(data=tone(24_000, 500, 10_000), format=PcmFormat(24_000))
        )

        filtered = suppressor.filter_input(tone(16_000, 500, 3_000))

        self.assertEqual(filtered, bytes(len(filtered)))

    def test_uncorrelated_near_end_speech_passes_during_playback(self) -> None:
        suppressor = OutputEchoSuppressor(
            EchoSuppressionConfig(
                barge_in_energy_threshold=0.02,
                echo_correlation_threshold=0.6,
                echo_gain=0.3,
            )
        )
        suppressor.add_output(
            AudioFrame(data=tone(24_000, 500, 8_000), format=PcmFormat(24_000))
        )
        near_end = tone(16_000, 1_300, 14_000)

        filtered = suppressor.filter_input(near_end)

        self.assertEqual(filtered, near_end)

    def test_default_gate_passes_speech_quieter_than_agent_output(self) -> None:
        suppressor = OutputEchoSuppressor()
        suppressor.add_output(
            AudioFrame(data=tone(24_000, 500, 14_000), format=PcmFormat(24_000))
        )
        quieter_near_end = tone(16_000, 1_300, 5_000)

        filtered = suppressor.filter_input(quieter_near_end)

        self.assertEqual(filtered, quieter_near_end)

    def test_mixed_echo_and_near_end_speech_is_not_suppressed(self) -> None:
        suppressor = OutputEchoSuppressor()
        suppressor.add_output(
            AudioFrame(data=tone(24_000, 500, 14_000), format=PcmFormat(24_000))
        )
        mixed = mixed_tones(16_000, 500, 3_000, 1_300, 5_000)

        filtered = suppressor.filter_input(mixed)

        self.assertEqual(filtered, mixed)


class SoundDeviceCallTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_drain_callback_cannot_complete_new_playout(self) -> None:
        audio = FakeAudio()
        transport = SoundDeviceCallTransport(audio=audio)
        await transport.prepare()
        await transport.connect()

        audio.output_stream.pump_output()
        frame = AudioFrame(
            data=tone(24_000, 500, 8_000),
            format=PcmFormat(24_000),
        )
        await transport.send_audio(frame)
        await asyncio.sleep(0)
        wait_task = asyncio.create_task(transport.wait_for_playout())
        await asyncio.sleep(0)

        self.assertFalse(wait_task.done())
        audio.output_stream.pump_output()
        await asyncio.wait_for(wait_task, timeout=0.2)
        await transport.close()

    async def test_near_end_speech_interrupts_active_local_playback(self) -> None:
        audio = FakeAudio()
        transport = SoundDeviceCallTransport(audio=audio)
        synthesizer = SlowSynthesizer()
        interruptions: list[str] = []
        session = CallSession(
            campaign=load_campaign(CAMPAIGN_PATH),
            model=MockConversationModel(),
            recognizer=MockSpeechRecognizer(
                [TranscriptEvent(text="not interested", is_final=True)]
            ),
            synthesizer=synthesizer,
            transport=transport,
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
        await synthesizer.started.wait()

        async def pump_output() -> None:
            while not run_task.done():
                audio.output_stream.pump_output()
                await asyncio.sleep(0.005)

        pump_task = asyncio.create_task(pump_output())
        while not any(any(data) for data in audio.output_stream.writes):
            await asyncio.sleep(0)
        for data in (
            *(tone(16_000, 1_300, 5_000) for _ in range(4)),
            *(bytes(640) for _ in range(3)),
        ):
            audio.input_stream.callback(data, 320, object(), None)
        result = await asyncio.wait_for(run_task, timeout=1)
        await pump_task

        self.assertEqual(interruptions, ["detected"])
        self.assertGreaterEqual(synthesizer.cancel_count, 1)
        self.assertTrue(audio.output_stream.aborted)
        self.assertEqual(result.interruptions, 1)
        self.assertEqual(result.lead.outcome, "NOT_INTERESTED")

    async def test_streams_input_and_output_concurrently(self) -> None:
        audio = FakeAudio()
        transport = SoundDeviceCallTransport(audio=audio)
        await transport.prepare()
        await transport.connect()

        self.assertTrue(audio.input_stream.started)
        self.assertTrue(audio.output_stream.started)
        microphone_data = tone(16_000, 900, 8_000)
        audio.input_stream.callback(microphone_data, 320, object(), None)
        event = await anext(transport.events())

        output_frame = AudioFrame(
            data=tone(24_000, 500, 8_000),
            format=PcmFormat(24_000),
        )
        await transport.send_audio(output_frame)
        audio.output_stream.pump_output()
        await transport.wait_for_playout()
        await transport.close()

        self.assertEqual(event.kind, TransportEventKind.AUDIO)
        self.assertEqual(event.audio.data, microphone_data)
        self.assertEqual(audio.output_stream.writes, [output_frame.data])
        self.assertTrue(audio.input_stream.closed)
        self.assertTrue(audio.output_stream.closed)


if __name__ == "__main__":
    unittest.main()
