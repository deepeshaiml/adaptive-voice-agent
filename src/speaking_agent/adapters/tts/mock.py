from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from speaking_agent.speech import (
    AudioFrame,
    PcmFormat,
    SpeechOperationCancelled,
    SynthesisOptions,
)


class MockSpeechSynthesizer:
    def __init__(
        self,
        *,
        sample_rate_hz: int = 24_000,
        frame_duration_ms: int = 20,
    ) -> None:
        self.format = PcmFormat(sample_rate_hz=sample_rate_hz)
        self.frame_duration_ms = frame_duration_ms
        self._cancelled = asyncio.Event()

    async def prepare(self) -> None:
        return None

    async def close(self) -> None:
        self._cancelled.set()

    async def cancel(self) -> None:
        self._cancelled.set()

    async def synthesize(
        self,
        text: str,
        *,
        options: SynthesisOptions | None = None,
    ) -> AsyncIterator[AudioFrame]:
        del options
        if not text.strip():
            raise ValueError("Synthesis text cannot be empty")
        self._cancelled.clear()
        samples_per_frame = self.format.sample_rate_hz * self.frame_duration_ms // 1_000
        frame_count = max(1, len(text) // 8)
        frame_data = bytes(samples_per_frame * self.format.sample_width_bytes)
        for index in range(frame_count):
            if self._cancelled.is_set():
                raise SpeechOperationCancelled("Speech synthesis was cancelled")
            await asyncio.sleep(0)
            yield AudioFrame(
                data=frame_data,
                format=self.format,
                timestamp_seconds=index * self.frame_duration_ms / 1_000,
            )
