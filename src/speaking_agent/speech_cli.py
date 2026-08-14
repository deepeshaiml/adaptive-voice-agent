from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Sequence
import json
from pathlib import Path
import sys
import time

from speaking_agent.adapters.asr.qwen_mlx import (
    DEFAULT_ASR_MODEL_PATH,
    QwenMlxSpeechRecognizer,
)
from speaking_agent.adapters.tts.macos_say import MacOsSaySpeechSynthesizer
from speaking_agent.adapters.tts.mock import MockSpeechSynthesizer
from speaking_agent.adapters.tts.qwen_mlx import (
    DEFAULT_TTS_MODEL_PATH,
    QwenMlxSpeechSynthesizer,
)
from speaking_agent.speech import (
    AudioFrame,
    SpeechError,
    SpeechSynthesizer,
    SynthesisOptions,
)
from speaking_agent.wav_io import read_wave_frames, write_wave_frames


async def synthesize(args: argparse.Namespace) -> int:
    synthesizer: SpeechSynthesizer
    if args.adapter == "qwen":
        synthesizer = QwenMlxSpeechSynthesizer(
            model_path=args.model_path or DEFAULT_TTS_MODEL_PATH
        )
    elif args.adapter == "system":
        synthesizer = MacOsSaySpeechSynthesizer()
    else:
        synthesizer = MockSpeechSynthesizer()

    started_at = time.perf_counter()
    first_audio_seconds: float | None = None

    async def measured_frames() -> AsyncIterator[AudioFrame]:
        nonlocal first_audio_seconds
        async for frame in synthesizer.synthesize(
            args.text,
            options=SynthesisOptions(
                voice=args.voice,
                language=args.language,
                style=args.style,
            ),
        ):
            if first_audio_seconds is None:
                first_audio_seconds = time.perf_counter() - started_at
            yield frame

    try:
        await synthesizer.prepare()
        started_at = time.perf_counter()
        pcm_format, sample_count = await write_wave_frames(
            args.output,
            measured_frames(),
        )
    finally:
        await synthesizer.close()

    print(
        json.dumps(
            {
                "output": str(args.output),
                "sample_rate_hz": pcm_format.sample_rate_hz,
                "audio_duration_seconds": sample_count / pcm_format.sample_rate_hz,
                "tts_first_audio_seconds": first_audio_seconds,
                "tts_total_seconds": time.perf_counter() - started_at,
            },
            indent=2,
        )
    )
    return 0


async def transcribe(args: argparse.Namespace) -> int:
    recognizer = QwenMlxSpeechRecognizer(
        model_path=args.model_path or DEFAULT_ASR_MODEL_PATH
    )
    started_at = time.perf_counter()
    first_partial_seconds: float | None = None
    final_seconds: float | None = None
    final_text = ""
    events = []
    try:
        await recognizer.prepare()
        started_at = time.perf_counter()
        async for event in recognizer.transcribe(
            read_wave_frames(args.input),
            language=args.language,
            context=args.context,
        ):
            elapsed = time.perf_counter() - started_at
            if first_partial_seconds is None:
                first_partial_seconds = elapsed
            if event.is_final:
                final_seconds = elapsed
                final_text = event.text
            events.append(
                {
                    "text": event.text,
                    "is_final": event.is_final,
                    "language": event.language,
                }
            )
    finally:
        await recognizer.close()

    print(
        json.dumps(
            {
                "input": str(args.input),
                "text": final_text,
                "asr_first_partial_seconds": first_partial_seconds,
                "asr_final_seconds": final_seconds,
                "events": events,
            },
            indent=2,
        )
    )
    return 0


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local speech adapter harness")
    commands = parser.add_subparsers(dest="command", required=True)

    synthesize_parser = commands.add_parser("synthesize")
    synthesize_parser.add_argument("--text", required=True)
    synthesize_parser.add_argument("--output", type=Path, required=True)
    synthesize_parser.add_argument(
        "--adapter",
        choices=("mock", "system", "qwen"),
        default="system",
    )
    synthesize_parser.add_argument("--model-path")
    synthesize_parser.add_argument("--voice")
    synthesize_parser.add_argument("--language")
    synthesize_parser.add_argument("--style")

    transcribe_parser = commands.add_parser("transcribe")
    transcribe_parser.add_argument("--input", type=Path, required=True)
    transcribe_parser.add_argument("--model-path")
    transcribe_parser.add_argument("--language")
    transcribe_parser.add_argument("--context", default="")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        operation = synthesize(args) if args.command == "synthesize" else transcribe(args)
        return asyncio.run(operation)
    except SpeechError as error:
        print(f"Speech error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())