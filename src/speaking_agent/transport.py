from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Protocol

from speaking_agent.speech import AudioFrame


class TransportError(RuntimeError):
    """A recoverable real-time call transport failure."""


class TransportEventKind(StrEnum):
    AUDIO = "AUDIO"
    DISCONNECTED = "DISCONNECTED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class TransportEvent:
    kind: TransportEventKind
    audio: AudioFrame | None = None
    reason: str | None = None


PlayoutObserver = Callable[[AudioFrame, float], None]


class CallTransport(Protocol):
    def set_playout_observer(
        self,
        observer: PlayoutObserver | None,
    ) -> None: ...

    async def prepare(self) -> None: ...

    async def connect(self) -> None: ...

    def events(self) -> AsyncIterator[TransportEvent]: ...

    async def send_audio(self, frame: AudioFrame) -> None: ...

    async def wait_for_playout(self) -> None: ...

    async def stop_audio(self) -> None: ...

    async def hang_up(self) -> None: ...

    async def transfer(self) -> None: ...

    async def close(self) -> None: ...
