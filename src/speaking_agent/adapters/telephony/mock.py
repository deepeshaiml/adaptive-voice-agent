from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import time
from typing import Callable

from speaking_agent.speech import AudioFrame
from speaking_agent.transport import TransportEvent, TransportEventKind


class MockCallTransport:
    def __init__(self) -> None:
        self._events: asyncio.Queue[TransportEvent] = asyncio.Queue()
        self.connected = False
        self.connected_event = asyncio.Event()
        self.closed = False
        self.hung_up = False
        self.transferred = False
        self.stop_audio_count = 0
        self.sent_audio: list[AudioFrame] = []
        self.first_audio_sent = asyncio.Event()
        self.playout_release = asyncio.Event()
        self.playout_release.set()
        self.playout_wait_count = 0
        self.sent_counts_at_playout: list[int] = []
        self._playout_observer: Callable[[AudioFrame, float], None] | None = None
        self._pending_playout: list[tuple[AudioFrame, float]] = []
        self._playout_cursor = 0.0

    def set_playout_observer(
        self,
        observer: Callable[[AudioFrame, float], None] | None,
    ) -> None:
        self._playout_observer = observer

    async def prepare(self) -> None:
        return None

    async def connect(self) -> None:
        self.connected = True
        self.connected_event.set()

    async def events(self) -> AsyncIterator[TransportEvent]:
        while True:
            event = await self._events.get()
            yield event
            if event.kind == TransportEventKind.DISCONNECTED:
                return

    async def send_audio(self, frame: AudioFrame) -> None:
        self.sent_audio.append(frame)
        started_at = max(time.monotonic(), self._playout_cursor)
        self._playout_cursor = started_at + frame.duration_seconds
        self._pending_playout.append((frame, started_at))
        self.first_audio_sent.set()
        await asyncio.sleep(0)

    async def wait_for_playout(self) -> None:
        self.playout_wait_count += 1
        self.sent_counts_at_playout.append(len(self.sent_audio))
        await self.playout_release.wait()
        if self._playout_observer is not None:
            for frame, started_at in self._pending_playout:
                self._playout_observer(frame, started_at)
        self._pending_playout.clear()

    async def stop_audio(self) -> None:
        self.stop_audio_count += 1
        self._pending_playout.clear()
        self._playout_cursor = time.monotonic()

    async def hang_up(self) -> None:
        self.hung_up = True

    async def transfer(self) -> None:
        self.transferred = True

    async def close(self) -> None:
        self.closed = True

    async def emit_audio(self, frame: AudioFrame) -> None:
        await self._events.put(
            TransportEvent(kind=TransportEventKind.AUDIO, audio=frame)
        )

    async def disconnect(self, reason: str = "remote") -> None:
        await self._events.put(
            TransportEvent(
                kind=TransportEventKind.DISCONNECTED,
                reason=reason,
            )
        )
