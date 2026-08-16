from array import array
import asyncio
import time
import unittest

from speaking_agent.domain import AgentReply
from speaking_agent.delivery import campaign_voice_style
from speaking_agent.local_voice_chat import (
    _check_audio_settings,
    _device,
    _record_utterance,
    _speak,
    _wait_for_room_echo,
    parse_args,
)
from speaking_agent.speech import AudioFrame, PcmFormat
from speaking_agent.speech import TranscriptEvent
from speaking_agent.turn_detection import TurnDetectionConfig


def pcm(amplitude: int) -> bytes:
    return array("h", [amplitude] * 320).tobytes()


class FakeInputStream:
    def __init__(self, frames: list[bytes]) -> None:
        self.frames = iter(frames)

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self, samples: int):
        self.samples = samples
        return next(self.frames), False


class FakeOutputStream:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.closed = False
        self.data = bytearray()

    def start(self) -> None:
        self.started = True

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class FakeAudio:
    def __init__(self, input_frames: list[bytes] | None = None) -> None:
        self.input_frames = input_frames or []
        self.input_options = None
        self.output_options = None
        self.output = FakeOutputStream()

    def check_input_settings(self, **options) -> None:
        self.checked_input_options = options

    def check_output_settings(self, **options) -> None:
        self.checked_output_options = options

    def RawInputStream(self, **options):
        self.input_options = options
        return FakeInputStream(self.input_frames)

    def RawOutputStream(self, **options):
        self.output_options = options
        return self.output


class FakeSynthesizer:
    async def synthesize(self, text, **kwargs):
        del text, kwargs
        frame = AudioFrame(data=bytes(960), format=PcmFormat(24_000))
        yield frame
        await asyncio.sleep(0)
        yield frame


class ContextRecognizer:
    def __init__(self) -> None:
        self.context = None

    async def transcribe(self, audio, *, language=None, context=""):
        del language
        async for _ in audio:
            pass
        self.context = context
        yield TranscriptEvent(text="Dubai Marina", is_final=True)


class LocalVoiceChatTests(unittest.IsolatedAsyncioTestCase):
    async def test_half_duplex_transcription_receives_campaign_context(self) -> None:
        from speaking_agent.local_voice_chat import _transcribe

        recognizer = ContextRecognizer()
        frame = AudioFrame(data=bytes(640), format=PcmFormat(16_000))

        text = await _transcribe(
            recognizer,
            (frame,),
            language="English",
            context="Acme Property; Dubai Marina",
        )

        self.assertEqual(text, "Dubai Marina")
        self.assertEqual(recognizer.context, "Acme Property; Dubai Marina")

    def test_device_accepts_an_index_or_name(self) -> None:
        self.assertEqual(_device("2"), 2)
        self.assertEqual(_device("MacBook Pro Microphone"), "MacBook Pro Microphone")
        self.assertIsNone(_device(None))

    def test_speaker_mode_waits_for_echo_without_a_prompt(self) -> None:
        sleeps: list[float] = []

        _wait_for_room_echo(750, sleeps.append)

        self.assertEqual(sleeps, [0.75])
        args = parse_args(["--speaker-mode", "--speaker-settle-ms", "750"])
        self.assertTrue(args.speaker_mode)
        self.assertEqual(args.speaker_settle_ms, 750)
        self.assertIsNone(args.style)

    def test_campaign_voice_style_drives_default_delivery(self) -> None:
        from speaking_agent.campaign import load_campaign

        campaign = load_campaign("campaigns/property_owner.json")
        style = campaign_voice_style(campaign)

        self.assertIn("Professional but conversational", style)
        self.assertIn("Avoid sounding like a survey", style)

    def test_full_duplex_mode_exposes_echo_and_barge_in_tuning(self) -> None:
        args = parse_args(
            [
                "--full-duplex",
                "--barge-in-energy-threshold",
                "0.06",
                "--echo-correlation-threshold",
                "0.55",
                "--echo-gain",
                "0.4",
                "--echo-tail-ms",
                "300",
            ]
        )

        self.assertTrue(args.full_duplex)
        self.assertEqual(args.barge_in_energy_threshold, 0.06)
        self.assertEqual(args.echo_correlation_threshold, 0.55)
        self.assertEqual(args.echo_gain, 0.4)
        self.assertEqual(args.echo_tail_ms, 300)

        defaults = parse_args(["--full-duplex"])
        self.assertEqual(defaults.barge_in_energy_threshold, 0.03)
        self.assertEqual(defaults.echo_gain, 1.0)

    def test_audio_settings_use_defaults_when_devices_are_omitted(self) -> None:
        audio = FakeAudio()

        _check_audio_settings(audio, input_device=None, output_device=None)

        self.assertIsNone(audio.checked_input_options["device"])
        self.assertIsNone(audio.checked_output_options["device"])

    def test_stale_device_index_has_actionable_error(self) -> None:
        class StaleDeviceAudio(FakeAudio):
            def check_output_settings(self, **options) -> None:
                del options
                raise ValueError("Error querying device 3")

        with self.assertRaisesRegex(RuntimeError, "indices can change"):
            _check_audio_settings(
                StaleDeviceAudio(),
                input_device=2,
                output_device=3,
            )

    def test_microphone_audio_is_segmented_at_trailing_silence(self) -> None:
        audio = FakeAudio(
            [pcm(5_000), pcm(5_000), pcm(5_000), pcm(5_000), pcm(0), pcm(0), pcm(0)]
        )

        frames = _record_utterance(
            audio,
            input_device=2,
            listen_timeout_seconds=1,
            turn_config=TurnDetectionConfig(
                energy_threshold=0.02,
                minimum_speech_ms=60,
                end_silence_ms=60,
            ),
        )

        self.assertIsNotNone(frames)
        self.assertEqual(len(frames), 4)
        self.assertEqual(audio.input_options["samplerate"], 16_000)
        self.assertEqual(audio.input_options["device"], 2)

    def test_short_one_word_utterance_is_not_discarded(self) -> None:
        audio = FakeAudio(
            [pcm(5_000), pcm(5_000), pcm(5_000), pcm(0), pcm(0), pcm(0)]
        )
        args = parse_args([])

        frames = _record_utterance(
            audio,
            input_device=None,
            listen_timeout_seconds=1,
            turn_config=TurnDetectionConfig(
                energy_threshold=args.energy_threshold,
                minimum_speech_ms=args.minimum_speech_ms,
                end_silence_ms=60,
            ),
        )

        self.assertEqual(args.minimum_speech_ms, 60)
        self.assertIsNotNone(frames)
        self.assertEqual(len(frames), 3)

    async def test_tts_pcm_streams_to_the_selected_speaker(self) -> None:
        audio = FakeAudio()

        metrics = await _speak(
            audio,
            FakeSynthesizer(),
            AgentReply("Hello"),
            output_device=3,
            voice="Aiden",
            language="English",
            style=None,
            response_started_at=time.perf_counter(),
        )

        self.assertEqual(audio.output_options["samplerate"], 24_000)
        self.assertEqual(audio.output_options["device"], 3)
        self.assertEqual(len(audio.output.data), 1_920)
        self.assertTrue(audio.output.started)
        self.assertTrue(audio.output.stopped)
        self.assertTrue(audio.output.closed)
        self.assertTrue(all(value >= 0 for value in metrics))


if __name__ == "__main__":
    unittest.main()