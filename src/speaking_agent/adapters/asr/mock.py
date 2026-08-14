from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Iterable

from speaking_agent.speech import (
    AudioFrame,
    SpeechOperationCancelled,
    TranscriptEvent,
)


class MockSpeechRecognizer:
    def __init__(self, events: Iterable[TranscriptEvent]) -> None:
        self._events = tuple(events)
        self._cancelled = asyncio.Event()

    async def prepare(self) -> None:
        return None

    async def close(self) -> None:
        self._cancelled.set()

    async def cancel(self) -> None:
        self._cancelled.set()

    async def transcribe(
        self,
        audio: AsyncIterable[AudioFrame],
        *,
        language: str | None = None,
        context: str = "",
    ) -> AsyncIterator[TranscriptEvent]:
        del language, context
        self._cancelled.clear()
        async for _ in audio:
            if self._cancelled.is_set():
                raise SpeechOperationCancelled("Speech recognition was cancelled")
        for event in self._events:
            if self._cancelled.is_set():
                raise SpeechOperationCancelled("Speech recognition was cancelled")
            await asyncio.sleep(0)
            yield event
