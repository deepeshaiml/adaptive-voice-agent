from types import SimpleNamespace
import unittest

from speaking_agent.adapters.asr.qwen_mlx import QwenMlxSpeechRecognizer
from speaking_agent.adapters.tts.qwen_mlx import QwenMlxSpeechSynthesizer
from speaking_agent.speech import AudioFrame, PcmFormat, SynthesisOptions


async def audio_frames(*frames: AudioFrame):
    for frame in frames:
        yield frame


class FakeAsrModel:
    def __init__(self) -> None:
        self.audio = None
        self.kwargs = None

    def stream_transcribe(self, audio, **kwargs):
        self.audio = audio
        self.kwargs = kwargs
        yield SimpleNamespace(
            text="I might",
            is_final=False,
            language="English",
            start_time=0.0,
            end_time=0.5,
        )
        yield SimpleNamespace(
            text="I might sell",
            is_final=True,
            language="English",
            start_time=0.0,
            end_time=1.0,
        )


class FakeTtsModel:
    def __init__(self) -> None:
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        yield SimpleNamespace(audio=[-1.0, 0.0, 0.5, 1.0], sample_rate=24_000)


class QwenSpeechAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_asr_resamples_pcm_and_maps_streaming_events(self) -> None:
        model = FakeAsrModel()
        recognizer = QwenMlxSpeechRecognizer(
            model=model,
            sample_array_factory=list,
        )
        frame = AudioFrame(
            data=bytes(320),
            format=PcmFormat(sample_rate_hz=8_000),
        )

        events = [
            event
            async for event in recognizer.transcribe(
                audio_frames(frame),
                language="English",
                context="Dubai Marina",
            )
        ]

        self.assertEqual(len(model.audio), 320)
        self.assertEqual(events[0].text, "I might")
        self.assertFalse(events[0].is_final)
        self.assertEqual(events[1].text, "I might sell")
        self.assertTrue(events[1].is_final)
        self.assertEqual(model.kwargs["system_prompt"], "Dubai Marina")

    async def test_tts_maps_streaming_waveform_to_pcm_frames(self) -> None:
        model = FakeTtsModel()
        synthesizer = QwenMlxSpeechSynthesizer(model=model)

        frames = [
            frame
            async for frame in synthesizer.synthesize(
                "Hello",
                options=SynthesisOptions(
                    voice="Ryan",
                    language="English",
                    style="Warm and concise",
                ),
            )
        ]

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].format.sample_rate_hz, 24_000)
        self.assertEqual(frames[0].sample_count, 4)
        self.assertEqual(model.kwargs["voice"], "Ryan")
        self.assertEqual(model.kwargs["instruct"], "Warm and concise")
        self.assertTrue(model.kwargs["stream"])
        self.assertEqual(model.kwargs["temperature"], 0.0)
        self.assertEqual(model.kwargs["top_k"], 50)

    async def test_tts_uses_default_style_without_per_call_options(self) -> None:
        model = FakeTtsModel()
        synthesizer = QwenMlxSpeechSynthesizer(
            model=model,
            default_style="Natural telephone delivery",
        )

        frames = [frame async for frame in synthesizer.synthesize("Hello")]

        self.assertTrue(frames)
        self.assertEqual(model.kwargs["instruct"], "Natural telephone delivery")

    async def test_tts_sampling_profile_is_configurable(self) -> None:
        model = FakeTtsModel()
        synthesizer = QwenMlxSpeechSynthesizer(
            model=model,
            temperature=0.35,
            top_k=20,
        )

        frames = [frame async for frame in synthesizer.synthesize("Hello")]

        self.assertTrue(frames)
        self.assertEqual(model.kwargs["temperature"], 0.35)
        self.assertEqual(model.kwargs["top_k"], 20)


if __name__ == "__main__":
    unittest.main()