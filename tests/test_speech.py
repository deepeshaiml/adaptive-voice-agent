from pathlib import Path
import tempfile
import unittest

from speaking_agent.adapters.asr.mock import MockSpeechRecognizer
from speaking_agent.adapters.tts.mock import MockSpeechSynthesizer
from speaking_agent.speech import (
    AudioFrame,
    PcmFormat,
    SpeechOperationCancelled,
    TranscriptEvent,
)
from speaking_agent.wav_io import read_wave_frames, write_wave_frames


async def audio_frames(*frames: AudioFrame):
    for frame in frames:
        yield frame


class SpeechContractTests(unittest.IsolatedAsyncioTestCase):
    def test_audio_frame_reports_sample_count_and_duration(self) -> None:
        pcm_format = PcmFormat(sample_rate_hz=16_000)
        frame = AudioFrame(data=bytes(640), format=pcm_format)

        self.assertEqual(frame.sample_count, 320)
        self.assertEqual(frame.duration_seconds, 0.02)

    async def test_mock_recognizer_emits_partial_and_final_events(self) -> None:
        events = (
            TranscriptEvent(text="I might", is_final=False, language="English"),
            TranscriptEvent(text="I might sell", is_final=True, language="English"),
        )
        recognizer = MockSpeechRecognizer(events)
        frame = AudioFrame(data=bytes(640), format=PcmFormat(16_000))

        actual = [
            event
            async for event in recognizer.transcribe(audio_frames(frame))
        ]

        self.assertEqual(actual, list(events))

    async def test_mock_synthesizer_streams_valid_pcm_and_cancels(self) -> None:
        synthesizer = MockSpeechSynthesizer()
        stream = synthesizer.synthesize("A sentence long enough for several frames.")

        first_frame = await anext(stream)
        await synthesizer.cancel()

        self.assertEqual(first_frame.duration_seconds, 0.02)
        with self.assertRaises(SpeechOperationCancelled):
            await anext(stream)

    async def test_wave_file_round_trip_preserves_pcm(self) -> None:
        pcm_format = PcmFormat(sample_rate_hz=16_000)
        expected = AudioFrame(data=bytes(range(16)), format=pcm_format)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            await write_wave_frames(path, audio_frames(expected))

            actual = [frame async for frame in read_wave_frames(path)]

        self.assertEqual(b"".join(frame.data for frame in actual), expected.data)
        self.assertEqual(actual[0].format, pcm_format)


if __name__ == "__main__":
    unittest.main()