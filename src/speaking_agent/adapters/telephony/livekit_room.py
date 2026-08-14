from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
import importlib
from typing import Any

from speaking_agent.speech import AudioFrame, PcmFormat
from speaking_agent.transport import (
    TransportError,
    TransportEvent,
    TransportEventKind,
)


AsyncAction = Callable[[], Awaitable[Any]]


class LiveKitRoomTransport:
    def __init__(
        self,
        *,
        room: Any,
        connect_room: AsyncAction,
        wait_for_participant: AsyncAction,
        hang_up_handler: AsyncAction,
        transfer_handler: AsyncAction | None = None,
        input_sample_rate_hz: int = 16_000,
        output_sample_rate_hz: int = 24_000,
        frame_duration_ms: int = 20,
        participant_timeout_seconds: float = 10.0,
    ) -> None:
        if participant_timeout_seconds <= 0:
            raise ValueError("participant_timeout_seconds must be positive")
        self.room = room
        self._connect_room = connect_room
        self._wait_for_participant = wait_for_participant
        self._hang_up_handler = hang_up_handler
        self._transfer_handler = transfer_handler
        self.input_sample_rate_hz = input_sample_rate_hz
        self.output_sample_rate_hz = output_sample_rate_hz
        self.frame_duration_ms = frame_duration_ms
        self.participant_timeout_seconds = participant_timeout_seconds
        self._rtc: Any | None = None
        self._audio_source: Any | None = None
        self._audio_stream: Any | None = None
        self._publication: Any | None = None
        self._connected = False
        self._closing = False

    async def prepare(self) -> None:
        try:
            self._rtc = importlib.import_module("livekit.rtc")
        except ImportError as error:
            raise TransportError(
                "LiveKit is unavailable; install the 'livekit' project extra"
            ) from error

    async def connect(self) -> None:
        if self._rtc is None:
            raise TransportError("LiveKit transport has not been prepared")
        try:
            await self._connect_room()
            self._connected = True
            try:
                async with asyncio.timeout(self.participant_timeout_seconds):
                    participant = await self._wait_for_participant()
            except TimeoutError as error:
                raise TransportError(
                    "Timed out waiting for the call participant"
                ) from error
            self._audio_source = self._rtc.AudioSource(
                self.output_sample_rate_hz,
                1,
                queue_size_ms=200,
            )
            track = self._rtc.LocalAudioTrack.create_audio_track(
                "speaking-agent",
                self._audio_source,
            )
            options = self._rtc.TrackPublishOptions(
                source=self._rtc.TrackSource.SOURCE_MICROPHONE
            )
            self._publication = await self.room.local_participant.publish_track(
                track,
                options,
            )
            self._audio_stream = self._rtc.AudioStream.from_participant(
                participant=participant,
                track_source=self._rtc.TrackSource.SOURCE_MICROPHONE,
                sample_rate=self.input_sample_rate_hz,
                num_channels=1,
                frame_size_ms=self.frame_duration_ms,
                capacity=50,
            )
        except BaseException:
            try:
                await self.close()
            except Exception:
                pass
            try:
                await self._hang_up_handler()
            except Exception:
                pass
            raise

    async def events(self) -> AsyncIterator[TransportEvent]:
        if self._audio_stream is None:
            raise TransportError("LiveKit transport is not connected")
        try:
            async for frame_event in self._audio_stream:
                frame = frame_event.frame
                yield TransportEvent(
                    kind=TransportEventKind.AUDIO,
                    audio=AudioFrame(
                        data=bytes(frame.data),
                        format=PcmFormat(
                            sample_rate_hz=frame.sample_rate,
                            channels=frame.num_channels,
                        ),
                    ),
                )
        except Exception as error:
            if not self._closing:
                raise TransportError("LiveKit input audio stream failed") from error
        if not self._closing:
            yield TransportEvent(
                kind=TransportEventKind.DISCONNECTED,
                reason="LiveKit participant audio ended",
            )

    async def send_audio(self, frame: AudioFrame) -> None:
        if self._audio_source is None:
            raise TransportError("LiveKit transport is not connected")
        if frame.format.sample_rate_hz != self.output_sample_rate_hz:
            raise TransportError(
                f"LiveKit output requires {self.output_sample_rate_hz} Hz PCM"
            )
        if frame.format.channels != 1:
            raise TransportError("LiveKit output requires mono PCM")
        livekit_frame = self._rtc.AudioFrame(
            data=frame.data,
            sample_rate=frame.format.sample_rate_hz,
            num_channels=frame.format.channels,
            samples_per_channel=frame.sample_count,
        )
        await self._audio_source.capture_frame(livekit_frame)

    async def wait_for_playout(self) -> None:
        if self._audio_source is not None:
            await self._audio_source.wait_for_playout()

    async def stop_audio(self) -> None:
        if self._audio_source is not None:
            self._audio_source.clear_queue()

    async def hang_up(self) -> None:
        await self._hang_up_handler()

    async def transfer(self) -> None:
        if self._transfer_handler is None:
            raise TransportError("No LiveKit transfer handler is configured")
        await self._transfer_handler()

    async def close(self) -> None:
        self._closing = True
        errors: list[Exception] = []
        if self._audio_stream is not None:
            try:
                await self._audio_stream.aclose()
            except Exception as error:
                errors.append(error)
            self._audio_stream = None
        if self._publication is not None:
            try:
                await self.room.local_participant.unpublish_track(
                    self._publication.sid
                )
            except Exception as error:
                errors.append(error)
            self._publication = None
        if self._audio_source is not None:
            try:
                await self._audio_source.aclose()
            except Exception as error:
                errors.append(error)
            self._audio_source = None
        self._connected = False
        if errors:
            raise TransportError(
                "LiveKit transport cleanup failed: "
                + ", ".join(type(error).__name__ for error in errors)
            )
