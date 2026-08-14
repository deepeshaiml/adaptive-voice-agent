from __future__ import annotations

from array import array
from collections.abc import Iterable, Sequence
import math
import sys
from typing import cast

from speaking_agent.speech import AudioFrame, PcmFormat, SpeechError


def pcm_frames_to_mono_float(
    frames: Sequence[AudioFrame],
    *,
    target_sample_rate_hz: int,
) -> list[float]:
    if not frames:
        raise SpeechError("No audio frames were provided")
    source_format = frames[0].format
    if any(frame.format != source_format for frame in frames):
        raise SpeechError("Audio format cannot change within one utterance")

    samples = array("h")
    for frame in frames:
        samples.frombytes(frame.data)
    if sys.byteorder != "little":
        samples.byteswap()

    mono = [
        sum(samples[index : index + source_format.channels])
        / (source_format.channels * 32_768.0)
        for index in range(0, len(samples), source_format.channels)
    ]
    if source_format.sample_rate_hz == target_sample_rate_hz:
        return mono
    return _linear_resample(
        mono,
        source_format.sample_rate_hz,
        target_sample_rate_hz,
    )


def float_waveform_to_pcm_frames(
    waveform: object,
    *,
    sample_rate_hz: int,
    frame_duration_ms: int,
    timestamp_offset_seconds: float,
) -> list[AudioFrame]:
    values = (
        waveform.tolist()
        if hasattr(waveform, "tolist")
        else list(cast(Iterable[float], waveform))
    )
    while values and isinstance(values[0], list):
        values = values[0]
    pcm = array(
        "h",
        (
            round(max(-1.0, min(1.0, float(value))) * 32_767)
            for value in values
        ),
    )
    if sys.byteorder != "little":
        pcm.byteswap()
    pcm_format = PcmFormat(sample_rate_hz=sample_rate_hz)
    samples_per_frame = max(1, sample_rate_hz * frame_duration_ms // 1_000)
    sample_width = pcm_format.sample_width_bytes
    data = pcm.tobytes()
    chunk_bytes = samples_per_frame * sample_width
    frames: list[AudioFrame] = []
    for byte_offset in range(0, len(data), chunk_bytes):
        sample_offset = byte_offset // sample_width
        frames.append(
            AudioFrame(
                data=data[byte_offset : byte_offset + chunk_bytes],
                format=pcm_format,
                timestamp_seconds=(
                    timestamp_offset_seconds + sample_offset / sample_rate_hz
                ),
            )
        )
    return frames


def _linear_resample(
    samples: Sequence[float],
    source_rate_hz: int,
    target_rate_hz: int,
) -> list[float]:
    if not samples:
        return []
    output_length = max(1, round(len(samples) * target_rate_hz / source_rate_hz))
    ratio = source_rate_hz / target_rate_hz
    output: list[float] = []
    for output_index in range(output_length):
        source_position = output_index * ratio
        left_index = min(math.floor(source_position), len(samples) - 1)
        right_index = min(left_index + 1, len(samples) - 1)
        fraction = source_position - left_index
        output.append(
            samples[left_index] * (1 - fraction) + samples[right_index] * fraction
        )
    return output
