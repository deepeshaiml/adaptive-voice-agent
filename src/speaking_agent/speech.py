from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Protocol


class SpeechError(RuntimeError):
    """A recoverable speech recognition or synthesis failure."""


class SpeechOperationCancelled(SpeechError):
    """The active speech operation was explicitly cancelled."""


class SpeechNotRecognizedError(SpeechError):
    """A detected audio turn did not contain a usable transcript."""


@dataclass(frozen=True, slots=True)
class PcmFormat:
    sample_rate_hz: int
    channels: int = 1
    sample_width_bytes: int = 2

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.sample_width_bytes != 2:
            raise ValueError("Only signed 16-bit little-endian PCM is supported")


@dataclass(frozen=True, slots=True)
class AudioFrame:
    data: bytes
    format: PcmFormat
    timestamp_seconds: float | None = None

    def __post_init__(self) -> None:
        bytes_per_sample = self.format.channels * self.format.sample_width_bytes
        if len(self.data) % bytes_per_sample:
            raise ValueError("Audio frame data must contain complete PCM samples")
        if self.timestamp_seconds is not None and self.timestamp_seconds < 0:
            raise ValueError("timestamp_seconds cannot be negative")

    @property
    def sample_count(self) -> int:
        return len(self.data) // (
            self.format.channels * self.format.sample_width_bytes
        )

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.format.sample_rate_hz


@dataclass(frozen=True, slots=True)
class TranscriptEvent:
    text: str
    is_final: bool
    language: str | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class SynthesisOptions:
    voice: str | None = None
    language: str | None = None
    style: str | None = None
    rate_wpm: int | None = None


class SpeechRecognizer(Protocol):
    async def prepare(self) -> None: ...

    def transcribe(
        self,
        audio: AsyncIterable[AudioFrame],
        *,
        language: str | None = None,
        context: str = "",
    ) -> AsyncIterator[TranscriptEvent]: ...

    async def cancel(self) -> None: ...

    async def close(self) -> None: ...


class SpeechSynthesizer(Protocol):
    async def prepare(self) -> None: ...

    def synthesize(
        self,
        text: str,
        *,
        options: SynthesisOptions | None = None,
    ) -> AsyncIterator[AudioFrame]: ...

    async def cancel(self) -> None: ...

    async def close(self) -> None: ...
