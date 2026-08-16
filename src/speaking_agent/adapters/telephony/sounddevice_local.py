from __future__ import annotations

import asyncio
from array import array
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
import math
import sys
from threading import Lock
import time
from typing import Any, Callable

from speaking_agent.speech import AudioFrame, PcmFormat
from speaking_agent.transport import (
    TransportError,
    TransportEvent,
    TransportEventKind,
)


@dataclass(frozen=True, slots=True)
class EchoSuppressionConfig:
    input_sample_rate_hz: int = 16_000
    output_sample_rate_hz: int = 24_000
    frame_duration_ms: int = 20
    barge_in_energy_threshold: float = 0.03
    echo_correlation_threshold: float = 0.45
    echo_gain: float = 1.0
    echo_tail_ms: int = 250
    max_reference_ms: int = 500

    def __post_init__(self) -> None:
        if self.input_sample_rate_hz <= 0 or self.output_sample_rate_hz <= 0:
            raise ValueError("Audio sample rates must be positive")
        if self.frame_duration_ms <= 0:
            raise ValueError("frame_duration_ms must be positive")
        if not 0 < self.barge_in_energy_threshold <= 1:
            raise ValueError("barge_in_energy_threshold must be between 0 and 1")
        if not 0 <= self.echo_correlation_threshold <= 1:
            raise ValueError("echo_correlation_threshold must be between 0 and 1")
        if self.echo_gain < 0:
            raise ValueError("echo_gain cannot be negative")
        if self.echo_tail_ms < 0 or self.max_reference_ms <= 0:
            raise ValueError("Echo timing values are invalid")


@dataclass(frozen=True, slots=True)
class _OutputReference:
    monotonic_seconds: float
    samples: tuple[float, ...]
    rms: float


class OutputEchoSuppressor:
    """Reject likely speaker loopback while retaining near-end barge-in speech."""

    def __init__(
        self,
        config: EchoSuppressionConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or EchoSuppressionConfig()
        self._clock = clock
        self._references: deque[_OutputReference] = deque()
        self._lock = Lock()
        self._playing = False
        self._active_until = 0.0

    def add_output(self, frame: AudioFrame) -> None:
        if frame.format.channels != 1:
            raise TransportError("Local output echo reference requires mono PCM")
        samples = self._pcm_float(frame.data)
        if frame.format.sample_rate_hz != self.config.input_sample_rate_hz:
            samples = self._resample(
                samples,
                frame.format.sample_rate_hz,
                self.config.input_sample_rate_hz,
            )
        reference = _OutputReference(
            monotonic_seconds=self._clock(),
            samples=tuple(samples),
            rms=self._rms(samples),
        )
        with self._lock:
            self._references.append(reference)
            self._playing = True
            self._active_until = float("inf")
            self._discard_old_locked(reference.monotonic_seconds)

    def finish_playback(self) -> None:
        now = self._clock()
        with self._lock:
            self._playing = False
            self._active_until = now + self.config.echo_tail_ms / 1_000
            self._discard_old_locked(now)

    def filter_input(self, data: bytes) -> bytes:
        now = self._clock()
        with self._lock:
            self._discard_old_locked(now)
            active = self._playing or now <= self._active_until
            references = tuple(self._references)
        if not active or not references:
            return data

        microphone = self._pcm_float(data)
        microphone_rms = self._rms(microphone)
        if microphone_rms < self.config.barge_in_energy_threshold:
            return bytes(len(data))

        best_correlation = 0.0
        best_residual_rms = microphone_rms
        maximum_reference_rms = 0.0
        for reference in references:
            maximum_reference_rms = max(maximum_reference_rms, reference.rms)
            sample_count = min(len(microphone), len(reference.samples))
            if sample_count == 0:
                continue
            input_samples = microphone[:sample_count]
            output_samples = reference.samples[:sample_count]
            output_power = sum(value * value for value in output_samples)
            input_power = sum(value * value for value in input_samples)
            if output_power <= 1e-12 or input_power <= 1e-12:
                continue
            dot_product = sum(
                input_value * output_value
                for input_value, output_value in zip(
                    input_samples,
                    output_samples,
                    strict=True,
                )
            )
            correlation = abs(dot_product) / math.sqrt(input_power * output_power)
            if correlation <= best_correlation:
                continue
            scale = dot_product / output_power
            residual = [
                input_value - scale * output_value
                for input_value, output_value in zip(
                    input_samples,
                    output_samples,
                    strict=True,
                )
            ]
            best_correlation = correlation
            best_residual_rms = self._rms(residual)

        if best_correlation >= self.config.echo_correlation_threshold:
            if best_residual_rms < self.config.barge_in_energy_threshold:
                return bytes(len(data))
        return data

    def _discard_old_locked(self, now: float) -> None:
        oldest = now - self.config.max_reference_ms / 1_000
        while self._references and self._references[0].monotonic_seconds < oldest:
            self._references.popleft()

    @staticmethod
    def _pcm_float(data: bytes) -> list[float]:
        samples = array("h")
        samples.frombytes(data)
        if sys.byteorder != "little":
            samples.byteswap()
        return [sample / 32_768.0 for sample in samples]

    @staticmethod
    def _rms(samples: list[float] | tuple[float, ...]) -> float:
        if not samples:
            return 0.0
        return math.sqrt(sum(value * value for value in samples) / len(samples))

    @staticmethod
    def _resample(
        samples: list[float],
        source_rate_hz: int,
        target_rate_hz: int,
    ) -> list[float]:
        if source_rate_hz == target_rate_hz or not samples:
            return samples
        output_length = max(1, round(len(samples) * target_rate_hz / source_rate_hz))
        ratio = source_rate_hz / target_rate_hz
        output: list[float] = []
        for output_index in range(output_length):
            source_position = output_index * ratio
            left_index = min(math.floor(source_position), len(samples) - 1)
            right_index = min(left_index + 1, len(samples) - 1)
            fraction = source_position - left_index
            output.append(
                samples[left_index] * (1 - fraction)
                + samples[right_index] * fraction
            )
        return output


class SoundDeviceCallTransport:
    def __init__(
        self,
        *,
        audio: Any,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        echo_config: EchoSuppressionConfig | None = None,
    ) -> None:
        self.audio = audio
        self.input_device = input_device
        self.output_device = output_device
        self.echo_suppressor = OutputEchoSuppressor(echo_config)
        self.config = self.echo_suppressor.config
        self._events: asyncio.Queue[TransportEvent] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._input_stream: Any | None = None
        self._output_stream: Any | None = None
        self._output_buffer = bytearray()
        self._output_lock = Lock()
        self._playout_drained: asyncio.Event | None = None
        self._connected = False
        self._closing = False
        self.last_input_status: str | None = None
        self.last_output_status: str | None = None

    async def prepare(self) -> None:
        try:
            self.audio.check_input_settings(
                device=self.input_device,
                channels=1,
                dtype="int16",
                samplerate=self.config.input_sample_rate_hz,
            )
            self.audio.check_output_settings(
                device=self.output_device,
                channels=1,
                dtype="int16",
                samplerate=self.config.output_sample_rate_hz,
            )
        except Exception as error:
            raise TransportError(
                "Full-duplex audio devices do not support the required PCM formats"
            ) from error

    async def connect(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._playout_drained = asyncio.Event()
        self._playout_drained.set()
        input_blocksize = (
            self.config.input_sample_rate_hz
            * self.config.frame_duration_ms
            // 1_000
        )
        output_blocksize = (
            self.config.output_sample_rate_hz
            * self.config.frame_duration_ms
            // 1_000
        )
        try:
            self._output_stream = self.audio.RawOutputStream(
                samplerate=self.config.output_sample_rate_hz,
                blocksize=output_blocksize,
                device=self.output_device,
                channels=1,
                dtype="int16",
                latency="low",
                callback=self._output_callback,
            )
            self._input_stream = self.audio.RawInputStream(
                samplerate=self.config.input_sample_rate_hz,
                blocksize=input_blocksize,
                device=self.input_device,
                channels=1,
                dtype="int16",
                latency="low",
                callback=self._input_callback,
            )
            self._output_stream.start()
            self._input_stream.start()
            self._connected = True
        except Exception as error:
            await self.close()
            raise TransportError("Unable to open full-duplex local audio") from error

    def _input_callback(
        self,
        input_data: object,
        frame_count: int,
        time_info: object,
        status: object,
    ) -> None:
        del frame_count, time_info
        if self._closing or self._loop is None:
            return
        if status:
            self.last_input_status = str(status)
        data = self.echo_suppressor.filter_input(bytes(input_data))
        event = TransportEvent(
            kind=TransportEventKind.AUDIO,
            audio=AudioFrame(
                data=data,
                format=PcmFormat(self.config.input_sample_rate_hz),
            ),
        )
        self._loop.call_soon_threadsafe(self._events.put_nowait, event)

    def _output_callback(
        self,
        output_data: object,
        frame_count: int,
        time_info: object,
        status: object,
    ) -> None:
        del time_info
        if status:
            self.last_output_status = str(status)
        required_bytes = frame_count * 2
        with self._output_lock:
            byte_count = min(required_bytes, len(self._output_buffer))
            played = bytes(self._output_buffer[:byte_count])
            del self._output_buffer[:byte_count]
            drained = not self._output_buffer
        payload = played + bytes(required_bytes - byte_count)
        memoryview(output_data).cast("B")[:required_bytes] = payload
        if played:
            self.echo_suppressor.add_output(
                AudioFrame(
                    data=played,
                    format=PcmFormat(self.config.output_sample_rate_hz),
                )
            )
        if drained and self._loop is not None and self._playout_drained is not None:
            self._loop.call_soon_threadsafe(self._mark_playout_drained_if_empty)

    def _mark_playout_drained_if_empty(self) -> None:
        with self._output_lock:
            drained = not self._output_buffer
        if drained and self._playout_drained is not None:
            self._playout_drained.set()

    async def events(self) -> AsyncIterator[TransportEvent]:
        if not self._connected:
            raise TransportError("Full-duplex local audio is not connected")
        while True:
            event = await self._events.get()
            yield event
            if event.kind in {
                TransportEventKind.DISCONNECTED,
                TransportEventKind.FAILED,
            }:
                return

    async def send_audio(self, frame: AudioFrame) -> None:
        if self._output_stream is None:
            raise TransportError("Full-duplex local audio is not connected")
        if frame.format != PcmFormat(self.config.output_sample_rate_hz):
            raise TransportError(
                "Full-duplex output requires 24 kHz mono signed 16-bit PCM"
            )
        if self._playout_drained is None:
            raise TransportError("Full-duplex playout state is unavailable")
        with self._output_lock:
            self._output_buffer.extend(frame.data)
        self._playout_drained.clear()

    async def wait_for_playout(self) -> None:
        output_stream = self._output_stream
        playout_drained = self._playout_drained
        if playout_drained is not None:
            await playout_drained.wait()
        if output_stream is not None:
            latency = getattr(output_stream, "latency", 0.0)
            if isinstance(latency, (int, float)) and latency > 0:
                await asyncio.sleep(float(latency))
        self.echo_suppressor.finish_playback()

    async def stop_audio(self) -> None:
        with self._output_lock:
            self._output_buffer.clear()
        if self._playout_drained is not None:
            self._playout_drained.set()
        output_stream = self._output_stream
        if output_stream is not None and hasattr(output_stream, "abort"):
            await asyncio.to_thread(output_stream.abort)
            await asyncio.to_thread(output_stream.start)
        self.echo_suppressor.finish_playback()

    async def hang_up(self) -> None:
        return None

    async def transfer(self) -> None:
        raise TransportError("Human transfer is unavailable in local full-duplex mode")

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self.echo_suppressor.finish_playback()
        errors: list[Exception] = []
        for stream in (self._input_stream, self._output_stream):
            if stream is None:
                continue
            try:
                await asyncio.to_thread(stream.stop)
            except Exception as error:
                errors.append(error)
            try:
                await asyncio.to_thread(stream.close)
            except Exception as error:
                errors.append(error)
        self._input_stream = None
        self._output_stream = None
        with self._output_lock:
            self._output_buffer.clear()
        self._playout_drained = None
        self._connected = False
        if errors:
            raise TransportError(
                "Full-duplex local audio cleanup failed: "
                + ", ".join(type(error).__name__ for error in errors)
            )
