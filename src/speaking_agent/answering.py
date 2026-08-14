from __future__ import annotations

from enum import StrEnum
from typing import Protocol


class AnswerKind(StrEnum):
    HUMAN = "HUMAN"
    VOICEMAIL = "VOICEMAIL"
    IVR = "IVR"
    MACHINE_UNAVAILABLE = "MACHINE_UNAVAILABLE"
    UNCERTAIN = "UNCERTAIN"


class AnsweringMachineDetector(Protocol):
    async def prepare(self) -> None: ...

    async def classify(self, transcript: str) -> AnswerKind: ...

    async def close(self) -> None: ...


class HeuristicAnsweringMachineDetector:
    _voicemail_phrases = (
        "leave a message",
        "after the tone",
        "after the beep",
        "not available to take your call",
        "you have reached the voicemail",
        "record your message",
    )
    _unavailable_phrases = (
        "mailbox is full",
        "mailbox has not been set up",
        "cannot accept new messages",
    )
    _ivr_phrases = (
        "press one",
        "press 1",
        "select from the following options",
        "to repeat this menu",
    )

    async def prepare(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def classify(self, transcript: str) -> AnswerKind:
        normalized = " ".join(transcript.casefold().split())
        if any(phrase in normalized for phrase in self._unavailable_phrases):
            return AnswerKind.MACHINE_UNAVAILABLE
        if any(phrase in normalized for phrase in self._voicemail_phrases):
            return AnswerKind.VOICEMAIL
        if any(phrase in normalized for phrase in self._ivr_phrases):
            return AnswerKind.IVR
        return AnswerKind.HUMAN
