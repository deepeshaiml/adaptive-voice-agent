import asyncio
import unittest

from speaking_agent.adapters.telephony.livekit_room import LiveKitRoomTransport
from speaking_agent.speech import AudioFrame, PcmFormat
from speaking_agent.transport import TransportError


class FakeAudioSource:
    def __init__(self) -> None:
        self.frames = []
        self.cleared = False

    async def capture_frame(self, frame) -> None:
        self.frames.append(frame)

    def clear_queue(self) -> None:
        self.cleared = True

    async def wait_for_playout(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class FakeRtcFrame:
    def __init__(self, *, data, sample_rate, num_channels, samples_per_channel):
        self.data = data
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = samples_per_channel


class FakePublication:
    sid = "TR_test"


class FakeLocalParticipant:
    def __init__(self) -> None:
        self.unpublished: list[str] = []

    async def publish_track(self, track, options):
        del track, options
        return FakePublication()

    async def unpublish_track(self, sid: str) -> None:
        self.unpublished.append(sid)


class FakeRoom:
    def __init__(self) -> None:
        self.local_participant = FakeLocalParticipant()


class LiveKitTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_participant_wait_timeout_hangs_up(self) -> None:
        hung_up = False

        async def connect_room() -> None:
            return None

        async def wait_forever():
            await asyncio.Event().wait()

        async def hang_up() -> None:
            nonlocal hung_up
            hung_up = True

        transport = LiveKitRoomTransport(
            room=object(),
            connect_room=connect_room,
            wait_for_participant=wait_forever,
            hang_up_handler=hang_up,
            participant_timeout_seconds=0.01,
        )
        transport._rtc = object()

        with self.assertRaisesRegex(TransportError, "participant"):
            await transport.connect()

        self.assertTrue(hung_up)

    async def test_sends_pcm_and_clears_provider_queue(self) -> None:
        async def action():
            return None

        transport = LiveKitRoomTransport(
            room=object(),
            connect_room=action,
            wait_for_participant=action,
            hang_up_handler=action,
        )
        transport._rtc = type("Rtc", (), {"AudioFrame": FakeRtcFrame})
        source = FakeAudioSource()
        transport._audio_source = source
        frame = AudioFrame(data=bytes(960), format=PcmFormat(24_000))

        await transport.send_audio(frame)
        await transport.stop_audio()

        self.assertEqual(source.frames[0].samples_per_channel, 480)
        self.assertTrue(source.cleared)

    async def test_reports_only_completed_playout_and_discards_stopped_audio(self) -> None:
        async def action():
            return None

        transport = LiveKitRoomTransport(
            room=object(),
            connect_room=action,
            wait_for_participant=action,
            hang_up_handler=action,
        )
        transport._rtc = type("Rtc", (), {"AudioFrame": FakeRtcFrame})
        transport._audio_source = FakeAudioSource()
        played: list[AudioFrame] = []
        transport.set_playout_observer(
            lambda frame, started_at: played.append(frame)
        )
        interrupted = AudioFrame(data=bytes(960), format=PcmFormat(24_000))

        await transport.send_audio(interrupted)
        await transport.stop_audio()
        await transport.wait_for_playout()

        self.assertEqual(played, [])

        delivered = AudioFrame(data=bytes(960), format=PcmFormat(24_000))
        await transport.send_audio(delivered)
        await transport.wait_for_playout()

        self.assertEqual(played, [delivered])

    async def test_rejects_wrong_output_sample_rate(self) -> None:
        async def action():
            return None

        transport = LiveKitRoomTransport(
            room=object(),
            connect_room=action,
            wait_for_participant=action,
            hang_up_handler=action,
        )
        transport._rtc = type("Rtc", (), {"AudioFrame": FakeRtcFrame})
        transport._audio_source = FakeAudioSource()

        with self.assertRaisesRegex(TransportError, "24000"):
            await transport.send_audio(
                AudioFrame(data=bytes(640), format=PcmFormat(16_000))
            )

    async def test_partial_connect_failure_unpublishes_and_hangs_up(self) -> None:
        connected = False
        hung_up = False
        source = FakeAudioSource()
        source.closed = False

        async def close_source() -> None:
            source.closed = True

        source.aclose = close_source

        async def connect_room() -> None:
            nonlocal connected
            connected = True

        async def participant():
            return object()

        async def hang_up() -> None:
            nonlocal hung_up
            hung_up = True

        class AudioSourceFactory:
            def __new__(cls, *args, **kwargs):
                del args, kwargs
                return source

        class LocalAudioTrack:
            @staticmethod
            def create_audio_track(name, audio_source):
                del name, audio_source
                return object()

        class AudioStream:
            @staticmethod
            def from_participant(**kwargs):
                del kwargs
                raise RuntimeError("stream setup failed")

        rtc = type(
            "Rtc",
            (),
            {
                "AudioSource": AudioSourceFactory,
                "LocalAudioTrack": LocalAudioTrack,
                "TrackPublishOptions": lambda **kwargs: kwargs,
                "TrackSource": type("TrackSource", (), {"SOURCE_MICROPHONE": 1}),
                "AudioStream": AudioStream,
            },
        )
        room = FakeRoom()
        transport = LiveKitRoomTransport(
            room=room,
            connect_room=connect_room,
            wait_for_participant=participant,
            hang_up_handler=hang_up,
        )
        transport._rtc = rtc

        with self.assertRaisesRegex(RuntimeError, "stream setup failed"):
            await transport.connect()

        self.assertTrue(connected)
        self.assertTrue(hung_up)
        self.assertTrue(source.closed)
        self.assertEqual(room.local_participant.unpublished, ["TR_test"])


if __name__ == "__main__":
    unittest.main()