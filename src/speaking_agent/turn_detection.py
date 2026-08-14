from __future__ import annotations

from array import array
from dataclasses import dataclass
from enum import StrEnum
import math
import sys

from speaking_agent.speech import AudioFrame


class TurnEventKind(StrEnum):
    SPEECH_STARTED = "SPEECH_STARTED"
    SPEECH_ENDED = "SPEECH_ENDED"


@dataclass(frozen=True, slots=True)
class TurnEvent:
    kind: TurnEventKind
    frames: tuple[AudioFrame, ...] = ()


@dataclass(frozen=True, slots=True)
class TurnDetectionConfig:
    energy_threshold: float = 0.02
    minimum_speech_ms: int = 160
    end_silence_ms: int = 600
    maximum_utterance_ms: int = 30_000


class EnergyTurnDetector:
    def __init__(self, config: TurnDetectionConfig | None = None) -> None:
        self.config = config or TurnDetectionConfig()
        self._candidate: list[AudioFrame] = []
        self._candidate_ms = 0.0
        self._utterance: list[AudioFrame] = []
        self._utterance_ms = 0.0
        self._silence_ms = 0.0
        self._trailing_silence_frames = 0
        self._speaking = False

    def process(self, frame: AudioFrame) -> tuple[TurnEvent, ...]:
        duration_ms = frame.duration_seconds * 1_000
        has_speech = self._energy(frame) >= self.config.energy_threshold
        if not self._speaking:
            if not has_speech:
                self._candidate.clear()
                self._candidate_ms = 0.0
                return ()
            self._candidate.append(frame)
            self._candidate_ms += duration_ms
            if self._candidate_ms < self.config.minimum_speech_ms:
                return ()
            self._speaking = True
            self._utterance = self._candidate
            self._utterance_ms = self._candidate_ms
            self._candidate = []
            self._candidate_ms = 0.0
            return (TurnEvent(TurnEventKind.SPEECH_STARTED),)

        self._utterance.append(frame)
        self._utterance_ms += duration_ms
        if has_speech:
            self._silence_ms = 0.0
            self._trailing_silence_frames = 0
        else:
            self._silence_ms += duration_ms
            self._trailing_silence_frames += 1

        if (
            self._silence_ms >= self.config.end_silence_ms
            or self._utterance_ms >= self.config.maximum_utterance_ms
        ):
            return (self._finish(),)
        return ()

    def flush(self) -> TurnEvent | None:
        if not self._speaking:
            self._candidate.clear()
            self._candidate_ms = 0.0
            return None
        return self._finish()

    def _finish(self) -> TurnEvent:
        if self._trailing_silence_frames:
            frames = self._utterance[: -self._trailing_silence_frames]
        else:
            frames = self._utterance
        event = TurnEvent(TurnEventKind.SPEECH_ENDED, tuple(frames))
        self._utterance = []
        self._utterance_ms = 0.0
        self._silence_ms = 0.0
        self._trailing_silence_frames = 0
        self._speaking = False
        return event

    @staticmethod
    def _energy(frame: AudioFrame) -> float:
        samples = array("h")
        samples.frombytes(frame.data)
        if sys.byteorder != "little":
            samples.byteswap()
        if not samples:
            return 0.0
        return math.sqrt(sum(sample * sample for sample in samples) / len(samples)) / 32_768
