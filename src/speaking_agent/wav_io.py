from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
import wave

from speaking_agent.speech import AudioFrame, PcmFormat, SpeechError


async def read_wave_frames(
    path: str | Path,
    *,
    frame_duration_ms: int = 20,
) -> AsyncIterator[AudioFrame]:
    with wave.open(str(path), "rb") as audio_file:
        pcm_format = PcmFormat(
            sample_rate_hz=audio_file.getframerate(),
            channels=audio_file.getnchannels(),
            sample_width_bytes=audio_file.getsampwidth(),
        )
        if audio_file.getcomptype() != "NONE":
            raise SpeechError("Only uncompressed PCM WAV input is supported")
        samples_per_frame = max(
            1,
            pcm_format.sample_rate_hz * frame_duration_ms // 1_000,
        )
        sample_offset = 0
        while data := audio_file.readframes(samples_per_frame):
            yield AudioFrame(
                data=data,
                format=pcm_format,
                timestamp_seconds=sample_offset / pcm_format.sample_rate_hz,
            )
            sample_offset += len(data) // (
                pcm_format.channels * pcm_format.sample_width_bytes
            )


async def write_wave_frames(
    path: str | Path,
    frames: AsyncIterator[AudioFrame],
) -> tuple[PcmFormat, int]:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_file: wave.Wave_write | None = None
    pcm_format: PcmFormat | None = None
    sample_count = 0
    try:
        async for frame in frames:
            if pcm_format is None:
                pcm_format = frame.format
                audio_file = wave.open(str(output_path), "wb")
                audio_file.setnchannels(pcm_format.channels)
                audio_file.setsampwidth(pcm_format.sample_width_bytes)
                audio_file.setframerate(pcm_format.sample_rate_hz)
            elif frame.format != pcm_format:
                raise SpeechError("Audio format cannot change within one WAV file")
            audio_file.writeframesraw(frame.data)
            sample_count += frame.sample_count
    finally:
        if audio_file is not None:
            audio_file.close()
    if pcm_format is None:
        raise SpeechError("No audio frames were produced")
    return pcm_format, sample_count
