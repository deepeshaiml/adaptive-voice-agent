from array import array
import unittest

from speaking_agent.speech import AudioFrame, PcmFormat
from speaking_agent.turn_detection import (
    EnergyTurnDetector,
    TurnDetectionConfig,
    TurnEventKind,
)


def audio_frame(amplitude: int) -> AudioFrame:
    samples = array("h", [amplitude] * 320)
    return AudioFrame(data=samples.tobytes(), format=PcmFormat(16_000))


class EnergyTurnDetectorTests(unittest.TestCase):
    def test_ignores_short_noise_and_segments_confirmed_speech(self) -> None:
        detector = EnergyTurnDetector(
            TurnDetectionConfig(
                energy_threshold=0.02,
                minimum_speech_ms=60,
                end_silence_ms=60,
            )
        )

        tiny_sound = detector.process(audio_frame(5_000))
        detector.process(audio_frame(0))
        events = []
        for amplitude in (5_000, 5_000, 5_000, 5_000, 0, 0, 0):
            events.extend(detector.process(audio_frame(amplitude)))

        self.assertEqual(tiny_sound, ())
        self.assertEqual(
            [event.kind for event in events],
            [TurnEventKind.SPEECH_STARTED, TurnEventKind.SPEECH_ENDED],
        )
        self.assertEqual(len(events[-1].frames), 4)


if __name__ == "__main__":
    unittest.main()