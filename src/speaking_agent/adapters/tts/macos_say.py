from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
import shutil
import tempfile
import wave

from speaking_agent.speech import (
    AudioFrame,
    PcmFormat,
    SpeechError,
    SpeechOperationCancelled,
    SynthesisOptions,
)


class MacOsSaySpeechSynthesizer:
    def __init__(self, *, frame_duration_ms: int = 20) -> None:
        self.frame_duration_ms = frame_duration_ms
        self._process: asyncio.subprocess.Process | None = None
        self._cancelled = asyncio.Event()
        self._operation_lock = asyncio.Lock()

    async def prepare(self) -> None:
        if shutil.which("say") is None:
            raise SpeechError("The macOS 'say' command is unavailable")

    async def close(self) -> None:
        await self.cancel()

    async def cancel(self) -> None:
        self._cancelled.set()
        process = self._process
        if process is not None and process.returncode is None:
            process.terminate()
            await process.wait()

    async def synthesize(
        self,
        text: str,
        *,
        options: SynthesisOptions | None = None,
    ) -> AsyncIterator[AudioFrame]:
        if not text.strip():
            raise ValueError("Synthesis text cannot be empty")
        options = options or SynthesisOptions()
        async with self._operation_lock:
            self._cancelled.clear()
            with tempfile.TemporaryDirectory(prefix="speaking-agent-tts-") as directory:
                output_path = Path(directory) / "speech.wav"
                command = [
                    "say",
                    "--output-file",
                    str(output_path),
                    "--file-format",
                    "WAVE",
                    "--data-format",
                    "LEI16@24000",
                    "--channels",
                    "1",
                ]
                if options.voice:
                    command.extend(("--voice", options.voice))
                if options.rate_wpm is not None:
                    command.extend(("--rate", str(options.rate_wpm)))
                command.append(text)
                self._process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    _, stderr = await self._process.communicate()
                except asyncio.CancelledError:
                    await self.cancel()
                    raise
                finally:
                    process = self._process
                    self._process = None

                if self._cancelled.is_set():
                    raise SpeechOperationCancelled("Speech synthesis was cancelled")
                if process is None or process.returncode != 0:
                    message = stderr.decode("utf-8", errors="replace").strip()
                    raise SpeechError(f"macOS speech synthesis failed: {message}")

                frames = await asyncio.to_thread(self._read_wave, output_path)
                for frame in frames:
                    if self._cancelled.is_set():
                        raise SpeechOperationCancelled("Speech synthesis was cancelled")
                    yield frame

    def _read_wave(self, path: Path) -> list[AudioFrame]:
        with wave.open(str(path), "rb") as audio_file:
            pcm_format = PcmFormat(
                sample_rate_hz=audio_file.getframerate(),
                channels=audio_file.getnchannels(),
                sample_width_bytes=audio_file.getsampwidth(),
            )
            samples_per_frame = max(
                1,
                pcm_format.sample_rate_hz * self.frame_duration_ms // 1_000,
            )
            frames: list[AudioFrame] = []
            sample_offset = 0
            while data := audio_file.readframes(samples_per_frame):
                frames.append(
                    AudioFrame(
                        data=data,
                        format=pcm_format,
                        timestamp_seconds=sample_offset / pcm_format.sample_rate_hz,
                    )
                )
                sample_offset += len(data) // (
                    pcm_format.channels * pcm_format.sample_width_bytes
                )
            return frames
