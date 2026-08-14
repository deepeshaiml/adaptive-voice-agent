from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Callable, Sequence
import importlib
from threading import Event
from typing import Any

from speaking_agent.adapters.audio_conversion import pcm_frames_to_mono_float
from speaking_agent.adapters.thread_bridge import iterate_in_thread
from speaking_agent.speech import (
    AudioFrame,
    SpeechError,
    SpeechOperationCancelled,
    TranscriptEvent,
)


DEFAULT_ASR_MODEL_PATH = "mlx-community/Qwen3-ASR-0.6B-8bit"


class QwenMlxSpeechRecognizer:
    def __init__(
        self,
        model_path: str = DEFAULT_ASR_MODEL_PATH,
        *,
        model: Any | None = None,
        sample_array_factory: Callable[[Sequence[float]], object] | None = None,
        max_tokens: int = 256,
        cancellation_grace_seconds: float = 2.0,
    ) -> None:
        self.model_path = model_path
        self._model = model
        self._sample_array_factory = sample_array_factory
        self._max_tokens = max_tokens
        self._cancellation_grace_seconds = cancellation_grace_seconds
        self._cancellation = Event()
        self._operation_lock = asyncio.Lock()
        self._unhealthy = False

    async def prepare(self) -> None:
        if self._model is not None:
            return
        try:
            module = importlib.import_module("mlx_audio.stt.utils")
            self._model = await asyncio.to_thread(module.load_model, self.model_path)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            raise SpeechError(f"Unable to load ASR model {self.model_path!r}") from error

    async def close(self) -> None:
        await self.cancel()
        if not self._unhealthy:
            self._model = None

    async def cancel(self) -> None:
        self._cancellation.set()

    async def transcribe(
        self,
        audio: AsyncIterable[AudioFrame],
        *,
        language: str | None = None,
        context: str = "",
    ) -> AsyncIterator[TranscriptEvent]:
        if self._unhealthy:
            raise SpeechError(
                "ASR backend is quarantined after a cancellation timeout"
            )
        if self._model is None:
            raise SpeechError("Speech recognizer has not been prepared")
        frames = [frame async for frame in audio]
        samples = pcm_frames_to_mono_float(frames, target_sample_rate_hz=16_000)
        model_audio = self._make_sample_array(samples)
        self._cancellation = Event()

        async with self._operation_lock:
            accumulated_text = ""
            try:
                async for result in iterate_in_thread(
                    lambda: self._model.stream_transcribe(
                        model_audio,
                        max_tokens=self._max_tokens,
                        language=language,
                        system_prompt=context or None,
                    ),
                    self._cancellation,
                    stop_timeout_seconds=self._cancellation_grace_seconds,
                    on_abandoned=self._mark_unhealthy,
                ):
                    provider_text = str(result.text)
                    if result.is_final:
                        event_text = provider_text or accumulated_text
                    else:
                        accumulated_text += provider_text
                        event_text = accumulated_text
                    yield TranscriptEvent(
                        text=event_text,
                        is_final=bool(result.is_final),
                        language=getattr(result, "language", None),
                        start_seconds=float(result.start_time),
                        end_seconds=float(result.end_time),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if self._cancellation.is_set():
                    raise SpeechOperationCancelled(
                        "Speech recognition was cancelled"
                    ) from error
                raise SpeechError("Qwen3-ASR transcription failed") from error
            if self._cancellation.is_set():
                raise SpeechOperationCancelled("Speech recognition was cancelled")

    def _make_sample_array(self, samples: Sequence[float]) -> object:
        if self._sample_array_factory is not None:
            return self._sample_array_factory(samples)
        numpy = importlib.import_module("numpy")
        return numpy.asarray(samples, dtype=numpy.float32)

    def _mark_unhealthy(self) -> None:
        self._unhealthy = True
