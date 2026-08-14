from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import time
from typing import Any


class TimingEventName(StrEnum):
    SPEECH_END = "speech_end"
    ASR_PARTIAL = "asr_partial"
    ASR_FINAL = "asr_final"
    LLM_START = "llm_start"
    LLM_FIRST_TOKEN = "llm_first_token"
    TTS_START = "tts_start"
    TTS_FIRST_AUDIO = "tts_first_audio"
    PLAYBACK_START = "playback_start"
    INTERRUPTION = "interruption"


@dataclass(frozen=True, slots=True)
class TimingEvent:
    name: TimingEventName
    monotonic_seconds: float
    attributes: dict[str, Any] = field(default_factory=dict)


class LatencyTrace:
    def __init__(self) -> None:
        self.events: list[TimingEvent] = []

    def record(self, name: TimingEventName, **attributes: Any) -> None:
        self.events.append(
            TimingEvent(
                name=name,
                monotonic_seconds=time.perf_counter(),
                attributes=attributes,
            )
        )

    def latest_duration(
        self,
        start: TimingEventName,
        end: TimingEventName,
    ) -> float | None:
        start_event = next(
            (event for event in reversed(self.events) if event.name == start),
            None,
        )
        if start_event is None:
            return None
        end_event = next(
            (
                event
                for event in self.events
                if event.name == end
                and event.monotonic_seconds >= start_event.monotonic_seconds
            ),
            None,
        )
        if end_event is None:
            return None
        return end_event.monotonic_seconds - start_event.monotonic_seconds
