from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import importlib
from threading import Event
from typing import Any

from speaking_agent.adapters.audio_conversion import float_waveform_to_pcm_frames
from speaking_agent.adapters.thread_bridge import iterate_in_thread
from speaking_agent.speech import (
    AudioFrame,
    SpeechError,
    SpeechOperationCancelled,
    SynthesisOptions,
)


DEFAULT_TTS_MODEL_PATH = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"


class QwenMlxSpeechSynthesizer:
    def __init__(
        self,
        model_path: str = DEFAULT_TTS_MODEL_PATH,
        *,
        model: Any | None = None,
        default_voice: str = "Aiden",
        default_language: str = "English",
        default_style: str | None = None,
        streaming_interval_seconds: float = 0.32,
        frame_duration_ms: int = 20,
        cancellation_grace_seconds: float = 2.0,
        temperature: float = 0.0,
        top_k: int = 50,
    ) -> None:
        if not 0 <= temperature <= 2:
            raise ValueError("TTS temperature must be between 0 and 2")
        if top_k < 0:
            raise ValueError("TTS top_k cannot be negative")
        self.model_path = model_path
        self._model = model
        self.default_voice = default_voice
        self.default_language = default_language
        self.default_style = default_style
        self.streaming_interval_seconds = streaming_interval_seconds
        self.frame_duration_ms = frame_duration_ms
        self._cancellation_grace_seconds = cancellation_grace_seconds
        self.temperature = temperature
        self.top_k = top_k
        self._cancellation = Event()
        self._operation_lock = asyncio.Lock()
        self._unhealthy = False

    async def prepare(self) -> None:
        if self._model is not None:
            return
        try:
            module = importlib.import_module("mlx_audio.tts.utils")
            self._model = await asyncio.to_thread(module.load_model, self.model_path)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            raise SpeechError(f"Unable to load TTS model {self.model_path!r}") from error

    async def close(self) -> None:
        await self.cancel()
        if not self._unhealthy:
            self._model = None

    async def cancel(self) -> None:
        self._cancellation.set()

    async def synthesize(
        self,
        text: str,
        *,
        options: SynthesisOptions | None = None,
    ) -> AsyncIterator[AudioFrame]:
        if self._unhealthy:
            raise SpeechError(
                "TTS backend is quarantined after a cancellation timeout"
            )
        if self._model is None:
            raise SpeechError("Speech synthesizer has not been prepared")
        if not text.strip():
            raise ValueError("Synthesis text cannot be empty")
        options = options or SynthesisOptions()
        if options.rate_wpm is not None:
            raise SpeechError("Qwen3-TTS does not support a words-per-minute rate")
        self._cancellation = Event()
        timestamp_seconds = 0.0

        async with self._operation_lock:
            try:
                async for result in iterate_in_thread(
                    lambda: self._model.generate(
                        text=text,
                        voice=options.voice or self.default_voice,
                        instruct=options.style or self.default_style,
                        lang_code=options.language or self.default_language,
                        stream=True,
                        streaming_interval=self.streaming_interval_seconds,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        verbose=False,
                    ),
                    self._cancellation,
                    stop_timeout_seconds=self._cancellation_grace_seconds,
                    on_abandoned=self._mark_unhealthy,
                ):
                    frames = float_waveform_to_pcm_frames(
                        result.audio,
                        sample_rate_hz=int(result.sample_rate),
                        frame_duration_ms=self.frame_duration_ms,
                        timestamp_offset_seconds=timestamp_seconds,
                    )
                    for frame in frames:
                        if self._cancellation.is_set():
                            raise SpeechOperationCancelled(
                                "Speech synthesis was cancelled"
                            )
                        yield frame
                    if frames:
                        timestamp_seconds += sum(
                            frame.duration_seconds for frame in frames
                        )
            except asyncio.CancelledError:
                raise
            except SpeechOperationCancelled:
                raise
            except Exception as error:
                if self._cancellation.is_set():
                    raise SpeechOperationCancelled(
                        "Speech synthesis was cancelled"
                    ) from error
                raise SpeechError("Qwen3-TTS synthesis failed") from error
            if self._cancellation.is_set():
                raise SpeechOperationCancelled("Speech synthesis was cancelled")

    def _mark_unhealthy(self) -> None:
        self._unhealthy = True
