from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from threading import Event
import time
from typing import Any

from speaking_agent.adapters.asr.qwen_mlx import (
    DEFAULT_ASR_MODEL_PATH,
    QwenMlxSpeechRecognizer,
)
from speaking_agent.adapters.llm.qwen_mlx import (
    DEFAULT_MODEL_PATH,
    MlxLmBackend,
    QwenMlxConversationModel,
)
from speaking_agent.adapters.telephony.sounddevice_local import (
    EchoSuppressionConfig,
    SoundDeviceCallTransport,
)
from speaking_agent.adapters.tts.qwen_mlx import (
    DEFAULT_TTS_MODEL_PATH,
    QwenMlxSpeechSynthesizer,
)
from speaking_agent.campaign import load_campaign
from speaking_agent.conversation import ConversationSession
from speaking_agent.delivery import campaign_voice_style
from speaking_agent.domain import AgentReply, SessionAction
from speaking_agent.recording import latency_snapshot
from speaking_agent.speech import AudioFrame, PcmFormat, SynthesisOptions
from speaking_agent.turn_detection import (
    EnergyTurnDetector,
    TurnDetectionConfig,
    TurnEventKind,
)
from speaking_agent.voice_session import CallSession


@dataclass(frozen=True, slots=True)
class TurnMetrics:
    asr_seconds: float
    llm_seconds: float
    tts_first_audio_seconds: float
    response_first_audio_seconds: float
    tts_playout_seconds: float


def _device(value: str | None) -> int | str | None:
    if value is None:
        return None
    return int(value) if value.isdecimal() else value


def _check_audio_settings(
    audio: Any,
    *,
    input_device: int | str | None,
    output_device: int | str | None,
) -> None:
    try:
        audio.check_input_settings(
            device=input_device,
            channels=1,
            dtype="int16",
            samplerate=16_000,
        )
        audio.check_output_settings(
            device=output_device,
            channels=1,
            dtype="int16",
            samplerate=24_000,
        )
    except Exception as error:
        raise RuntimeError(
            "Audio device unavailable. CoreAudio indices can change when displays or "
            "headsets connect; omit --input-device/--output-device to use macOS "
            "defaults, or run --list-devices again."
        ) from error


def _wait_for_room_echo(
    settle_ms: int,
    sleep: Any = time.sleep,
) -> None:
    if settle_ms:
        sleep(settle_ms / 1_000)


def _record_utterance(
    audio: Any,
    *,
    input_device: int | str | None,
    listen_timeout_seconds: float,
    turn_config: TurnDetectionConfig,
    stop_requested: Event | None = None,
) -> tuple[AudioFrame, ...] | None:
    sample_rate_hz = 16_000
    frame_duration_ms = 20
    samples_per_frame = sample_rate_hz * frame_duration_ms // 1_000
    pcm_format = PcmFormat(sample_rate_hz=sample_rate_hz)
    detector = EnergyTurnDetector(turn_config)
    stop_requested = stop_requested or Event()
    started_at = time.monotonic()
    heard_speech = False

    with audio.RawInputStream(
        samplerate=sample_rate_hz,
        blocksize=samples_per_frame,
        device=input_device,
        channels=1,
        dtype="int16",
        latency="low",
    ) as stream:
        while not stop_requested.is_set():
            data, overflowed = stream.read(samples_per_frame)
            if overflowed:
                print("Audio warning: microphone input overflowed.", file=sys.stderr)
            frame = AudioFrame(data=bytes(data), format=pcm_format)
            for event in detector.process(frame):
                if event.kind == TurnEventKind.SPEECH_STARTED:
                    heard_speech = True
                    print("Speech detected...", flush=True)
                elif event.frames:
                    return event.frames
            if (
                not heard_speech
                and time.monotonic() - started_at >= listen_timeout_seconds
            ):
                return None
    return None


async def _capture_utterance(
    audio: Any,
    *,
    input_device: int | str | None,
    listen_timeout_seconds: float,
    turn_config: TurnDetectionConfig,
) -> tuple[AudioFrame, ...] | None:
    stop_requested = Event()
    worker = asyncio.create_task(
        asyncio.to_thread(
            _record_utterance,
            audio,
            input_device=input_device,
            listen_timeout_seconds=listen_timeout_seconds,
            turn_config=turn_config,
            stop_requested=stop_requested,
        )
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        stop_requested.set()
        await worker
        raise


async def _frames(frames: tuple[AudioFrame, ...]) -> AsyncIterator[AudioFrame]:
    for frame in frames:
        yield frame


async def _transcribe(
    recognizer: QwenMlxSpeechRecognizer,
    frames: tuple[AudioFrame, ...],
    *,
    language: str,
    context: str = "",
) -> str:
    final_text = ""
    async for event in recognizer.transcribe(
        _frames(frames),
        language=language,
        context=context,
    ):
        if event.is_final:
            final_text = event.text.strip()
    if not final_text:
        raise RuntimeError("Speech recognizer returned no final transcript")
    return final_text


async def _speak(
    audio: Any,
    synthesizer: QwenMlxSpeechSynthesizer,
    reply: AgentReply,
    *,
    output_device: int | str | None,
    voice: str,
    language: str,
    style: str | None,
    response_started_at: float,
) -> tuple[float, float, float]:
    stream: Any | None = None
    first_audio_seconds: float | None = None
    tts_started_at = time.perf_counter()
    try:
        async for frame in synthesizer.synthesize(
            reply.text,
            options=SynthesisOptions(
                voice=voice,
                language=language,
                style=style,
            ),
        ):
            if stream is None:
                stream = audio.RawOutputStream(
                    samplerate=frame.format.sample_rate_hz,
                    blocksize=0,
                    device=output_device,
                    channels=frame.format.channels,
                    dtype="int16",
                    latency="low",
                )
                stream.start()
                first_audio_seconds = time.perf_counter() - tts_started_at
                response_first_audio_seconds = (
                    time.perf_counter() - response_started_at
                )
            stream.write(frame.data)
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
    if first_audio_seconds is None:
        raise RuntimeError("Speech synthesizer returned no audio")
    return (
        first_audio_seconds,
        response_first_audio_seconds,
        time.perf_counter() - tts_started_at,
    )


async def _run_full_duplex(
    args: argparse.Namespace,
    audio: Any,
    *,
    campaign: Any,
    model: QwenMlxConversationModel,
    recognizer: QwenMlxSpeechRecognizer,
    synthesizer: QwenMlxSpeechSynthesizer,
    input_device: int | str | None,
    output_device: int | str | None,
    turn_config: TurnDetectionConfig,
) -> int:
    transport = SoundDeviceCallTransport(
        audio=audio,
        input_device=input_device,
        output_device=output_device,
        echo_config=EchoSuppressionConfig(
            barge_in_energy_threshold=args.barge_in_energy_threshold,
            echo_correlation_threshold=args.echo_correlation_threshold,
            echo_gain=args.echo_gain,
            echo_tail_ms=args.echo_tail_ms,
        ),
    )
    ready = False

    def show_reply(reply: AgentReply) -> None:
        nonlocal ready
        if not ready:
            print(
                "Ready. Full-duplex microphone and barge-in are active.",
                flush=True,
            )
            ready = True
        print(f"Agent: {reply.text}", flush=True)

    session = CallSession(
        campaign=campaign,
        model=model,
        recognizer=recognizer,
        synthesizer=synthesizer,
        transport=transport,
        turn_detector=EnergyTurnDetector(turn_config),
        on_transcript=lambda text: print(f"You: {text}", flush=True),
        on_agent_reply=show_reply,
        on_interruption=lambda: print(
            "Barge-in detected; stopping agent playback...",
            flush=True,
        ),
        recognition_language=args.language,
        recognition_context=campaign.speech_recognition_context,
        transfer_available=False,
    )
    print("Loading local Qwen LLM, ASR, and TTS models...", flush=True)
    result = await session.run()
    print(
        "Session: "
        f"interruptions={result.interruptions} "
        f"latencies={json.dumps(latency_snapshot(session.trace), sort_keys=True)}"
    )
    print("Result:")
    print(json.dumps(asdict(result.lead), indent=2))
    return 0


async def run(args: argparse.Namespace, audio: Any) -> int:
    campaign = load_campaign(args.campaign)
    voice_style = args.style or campaign_voice_style(campaign)
    model = QwenMlxConversationModel(
        MlxLmBackend(args.model_path or DEFAULT_MODEL_PATH)
    )
    recognizer = QwenMlxSpeechRecognizer(
        model_path=args.asr_model_path or DEFAULT_ASR_MODEL_PATH
    )
    synthesizer = QwenMlxSpeechSynthesizer(
        model_path=args.tts_model_path or DEFAULT_TTS_MODEL_PATH,
        default_voice=args.voice,
        default_language=args.language,
        default_style=voice_style,
    )
    session = ConversationSession(campaign, model)
    input_device = _device(args.input_device)
    output_device = _device(args.output_device)
    turn_config = TurnDetectionConfig(
        energy_threshold=args.energy_threshold,
        minimum_speech_ms=args.minimum_speech_ms,
        end_silence_ms=args.end_silence_ms,
        maximum_utterance_ms=args.maximum_utterance_ms,
    )

    if args.full_duplex:
        return await _run_full_duplex(
            args,
            audio,
            campaign=campaign,
            model=model,
            recognizer=recognizer,
            synthesizer=synthesizer,
            input_device=input_device,
            output_device=output_device,
            turn_config=turn_config,
        )

    _check_audio_settings(
        audio,
        input_device=input_device,
        output_device=output_device,
    )

    print("Loading local Qwen LLM, ASR, and TTS models...", flush=True)
    try:
        await model.prepare()
        await recognizer.prepare()
        await synthesizer.prepare()
        if args.speaker_mode:
            print(
                "Ready. Speaker mode keeps the microphone closed during playback.",
                flush=True,
            )
        else:
            print("Ready. Use headphones for the clearest quality check.", flush=True)

        opening = session.start()
        print(f"Agent: {opening.text}")
        await _speak(
            audio,
            synthesizer,
            opening,
            output_device=output_device,
            voice=args.voice,
            language=args.language,
            style=voice_style,
            response_started_at=time.perf_counter(),
        )

        while not session.state.ended:
            if args.speaker_mode:
                _wait_for_room_echo(args.speaker_settle_ms)
            print(
                f"Listening (speak within {args.listen_timeout_seconds:g}s)...",
                flush=True,
            )
            frames = await _capture_utterance(
                audio,
                input_device=input_device,
                listen_timeout_seconds=args.listen_timeout_seconds,
                turn_config=turn_config,
            )
            if frames is None:
                print("No speech detected; listening again.")
                continue

            response_started_at = time.perf_counter()
            asr_started_at = time.perf_counter()
            transcript = await _transcribe(
                recognizer,
                frames,
                language=args.language,
                context=campaign.speech_recognition_context,
            )
            asr_seconds = time.perf_counter() - asr_started_at
            print(f"You: {transcript}")

            llm_started_at = time.perf_counter()
            reply = await session.receive(transcript)
            llm_seconds = time.perf_counter() - llm_started_at
            if reply.action == SessionAction.TRANSFER:
                reply = AgentReply(
                    campaign.transfer_unavailable_message,
                    SessionAction.HANG_UP,
                )
            print(f"Agent: {reply.text}")
            (
                tts_first_audio_seconds,
                response_first_audio_seconds,
                tts_playout_seconds,
            ) = await _speak(
                audio,
                synthesizer,
                reply,
                output_device=output_device,
                voice=args.voice,
                language=args.language,
                style=voice_style,
                response_started_at=response_started_at,
            )
            metrics = TurnMetrics(
                asr_seconds=asr_seconds,
                llm_seconds=llm_seconds,
                tts_first_audio_seconds=tts_first_audio_seconds,
                response_first_audio_seconds=response_first_audio_seconds,
                tts_playout_seconds=tts_playout_seconds,
            )
            print(
                "Latency: "
                f"ASR {metrics.asr_seconds:.2f}s | "
                f"LLM {metrics.llm_seconds:.2f}s | "
                f"TTS first audio {metrics.tts_first_audio_seconds:.2f}s | "
                f"speech end to response {metrics.response_first_audio_seconds:.2f}s"
            )

        print("Result:")
        print(json.dumps(asdict(session.result()), indent=2))
        return 0
    finally:
        await synthesizer.close()
        await recognizer.close()
        await model.close()


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Talk with the local Qwen voice agent using a microphone"
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path("campaigns/property_owner.json"),
    )
    parser.add_argument("--model-path")
    parser.add_argument("--asr-model-path")
    parser.add_argument("--tts-model-path")
    parser.add_argument("--voice", default="Aiden")
    parser.add_argument("--language", default="English")
    parser.add_argument(
        "--style",
        help="Override the campaign voice style for this run",
    )
    parser.add_argument("--input-device")
    parser.add_argument("--output-device")
    parser.add_argument("--listen-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--energy-threshold", type=float, default=0.02)
    parser.add_argument(
        "--minimum-speech-ms",
        type=int,
        default=60,
        help="Minimum voiced duration; 60 ms preserves short words such as yes or no",
    )
    parser.add_argument("--end-silence-ms", type=int, default=550)
    parser.add_argument("--maximum-utterance-ms", type=int, default=30_000)
    audio_modes = parser.add_mutually_exclusive_group()
    audio_modes.add_argument(
        "--speaker-mode",
        action="store_true",
        help="Wait for room echo to settle before opening the microphone",
    )
    audio_modes.add_argument(
        "--full-duplex",
        action="store_true",
        help="Listen during playback and stop the agent on confirmed near-end speech",
    )
    parser.add_argument(
        "--speaker-settle-ms",
        type=int,
        default=500,
        help="Room-echo guard before automatic microphone listening",
    )
    parser.add_argument(
        "--barge-in-energy-threshold",
        type=float,
        default=0.03,
        help="Near-end speech energy required while speaker audio is active",
    )
    parser.add_argument(
        "--echo-correlation-threshold",
        type=float,
        default=0.45,
        help="Correlation above which microphone audio is treated as speaker echo",
    )
    parser.add_argument(
        "--echo-gain",
        type=float,
        default=1.0,
        help="Expected maximum microphone echo relative to output amplitude",
    )
    parser.add_argument(
        "--echo-tail-ms",
        type=int,
        default=250,
        help="Continue suppressing probable echo after playback stops",
    )
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args(arguments)
    if args.listen_timeout_seconds <= 0:
        parser.error("--listen-timeout-seconds must be positive")
    if not 0 < args.energy_threshold <= 1:
        parser.error("--energy-threshold must be between 0 and 1")
    if args.speaker_settle_ms < 0:
        parser.error("--speaker-settle-ms cannot be negative")
    if not 0 < args.barge_in_energy_threshold <= 1:
        parser.error("--barge-in-energy-threshold must be between 0 and 1")
    if not 0 <= args.echo_correlation_threshold <= 1:
        parser.error("--echo-correlation-threshold must be between 0 and 1")
    if args.echo_gain < 0:
        parser.error("--echo-gain cannot be negative")
    if args.echo_tail_ms < 0:
        parser.error("--echo-tail-ms cannot be negative")
    for name in (
        "minimum_speech_ms",
        "end_silence_ms",
        "maximum_utterance_ms",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        import sounddevice

        if args.list_devices:
            print(sounddevice.query_devices())
            return 0
        return asyncio.run(run(args, sounddevice))
    except KeyboardInterrupt:
        print("\nLocal voice chat stopped.")
        return 130
    except Exception as error:
        detail = str(error).strip() or repr(error)
        print(
            f"Local voice chat error ({type(error).__name__}): {detail}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())