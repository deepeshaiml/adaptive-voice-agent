from __future__ import annotations

import argparse
import asyncio
from array import array
from dataclasses import asdict, dataclass
import json
import math
from uuid import uuid4

from livekit import api, rtc
from livekit.agents.utils import wait_for_participant, wait_for_track_publication

from speaking_agent.adapters.telephony.livekit_room import LiveKitRoomTransport
from speaking_agent.speech import AudioFrame, PcmFormat


@dataclass(frozen=True)
class SmokeResult:
    room_name: str
    inbound_sample_rate_hz: int
    inbound_rms: float
    outbound_sample_rate_hz: int
    outbound_rms: float


def access_token(
    api_key: str,
    api_secret: str,
    *,
    identity: str,
    room_name: str,
) -> str:
    return (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )


def tone_frame(sample_rate_hz: int, *, frequency_hz: int = 440) -> AudioFrame:
    sample_count = sample_rate_hz // 50
    samples = array(
        "h",
        (
            round(
                12_000
                * math.sin(2 * math.pi * frequency_hz * index / sample_rate_hz)
            )
            for index in range(sample_count)
        ),
    )
    return AudioFrame(
        data=samples.tobytes(),
        format=PcmFormat(sample_rate_hz),
    )


def rms(data: bytes) -> float:
    samples = array("h")
    samples.frombytes(data)
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


async def receive_non_silent(
    stream,
    frame_getter,
    *,
    timeout_seconds: float = 5.0,
) -> tuple[object, float]:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    strongest_item = None
    strongest_rms = 0.0
    while asyncio.get_running_loop().time() < deadline:
        remaining = deadline - asyncio.get_running_loop().time()
        item = await asyncio.wait_for(anext(stream), timeout=remaining)
        level = rms(frame_getter(item))
        if level > strongest_rms:
            strongest_item = item
            strongest_rms = level
        if level >= 100:
            return item, level
    raise RuntimeError(
        f"LiveKit audio remained silent; strongest RMS was {strongest_rms:.2f}"
    )


async def run(args: argparse.Namespace) -> SmokeResult:
    room_name = f"speaking-agent-smoke-{uuid4().hex[:10]}"
    caller_identity = "smoke-caller"
    agent_identity = "smoke-agent"
    caller_room = rtc.Room()
    agent_room = rtc.Room()
    caller_source: rtc.AudioSource | None = None
    caller_output_stream: rtc.AudioStream | None = None
    transport: LiveKitRoomTransport | None = None
    transport_events = None

    try:
        await caller_room.connect(
            args.url,
            access_token(
                args.api_key,
                args.api_secret,
                identity=caller_identity,
                room_name=room_name,
            ),
        )
        caller_source = rtc.AudioSource(16_000, 1, queue_size_ms=100)
        caller_track = rtc.LocalAudioTrack.create_audio_track(
            "smoke-caller-microphone",
            caller_source,
        )
        await caller_room.local_participant.publish_track(
            caller_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )

        async def connect_agent() -> None:
            await agent_room.connect(
                args.url,
                access_token(
                    args.api_key,
                    args.api_secret,
                    identity=agent_identity,
                    room_name=room_name,
                ),
            )

        async def find_caller():
            return await wait_for_participant(
                agent_room,
                identity=caller_identity,
            )

        async def no_op() -> None:
            return None

        transport = LiveKitRoomTransport(
            room=agent_room,
            connect_room=connect_agent,
            wait_for_participant=find_caller,
            hang_up_handler=no_op,
        )
        await transport.prepare()
        await transport.connect()

        transport_events = transport.events()
        inbound_event_task = asyncio.create_task(
            receive_non_silent(
                transport_events,
                lambda event: event.audio.data if event.audio is not None else b"",
            )
        )
        for _ in range(20):
            frame = tone_frame(16_000)
            await caller_source.capture_frame(
                rtc.AudioFrame(
                    data=frame.data,
                    sample_rate=16_000,
                    num_channels=1,
                    samples_per_channel=frame.sample_count,
                )
            )
        inbound_result = await inbound_event_task
        inbound_event, inbound_rms = inbound_result
        if inbound_event.audio is None:
            raise RuntimeError("LiveKit transport returned no inbound audio")

        publication = await asyncio.wait_for(
            wait_for_track_publication(
                caller_room,
                identity=agent_identity,
                kind=rtc.TrackKind.KIND_AUDIO,
                wait_for_subscription=True,
            ),
            timeout=5,
        )
        if publication.track is None:
            raise RuntimeError("Agent audio publication has no subscribed track")
        caller_output_stream = rtc.AudioStream.from_track(
            track=publication.track,
            sample_rate=24_000,
            num_channels=1,
            frame_size_ms=20,
            capacity=50,
        )
        outbound_event_task = asyncio.create_task(
            receive_non_silent(
                caller_output_stream,
                lambda event: bytes(event.frame.data),
            )
        )
        for _ in range(20):
            await transport.send_audio(tone_frame(24_000, frequency_hz=660))
        outbound_event, outbound_rms = await outbound_event_task
        await transport.wait_for_playout()

        result = SmokeResult(
            room_name=room_name,
            inbound_sample_rate_hz=inbound_event.audio.format.sample_rate_hz,
            inbound_rms=inbound_rms,
            outbound_sample_rate_hz=outbound_event.frame.sample_rate,
            outbound_rms=outbound_rms,
        )
        return result
    finally:
        if transport_events is not None:
            await transport_events.aclose()
        if caller_output_stream is not None:
            await caller_output_stream.aclose()
        if transport is not None:
            await transport.close()
        if caller_source is not None:
            await caller_source.aclose()
        await agent_room.disconnect()
        await caller_room.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bidirectional local LiveKit audio smoke test")
    parser.add_argument("--url", default="ws://127.0.0.1:7880")
    parser.add_argument("--api-key", default="devkey")
    parser.add_argument("--api-secret", default="secret")
    return parser.parse_args()


def main() -> int:
    result = asyncio.run(run(parse_args()))
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
