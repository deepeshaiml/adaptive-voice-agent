from __future__ import annotations

from array import array
import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import wave

from speaking_agent.audio_recording import (
    AudioRecordingError,
    RecordingConsent,
    WaveConversationRecorder,
    purge_expired_recordings,
)
from speaking_agent.speech import AudioFrame, PcmFormat
from speaking_agent.retention_worker import parse_args as parse_retention_args


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def pcm_frame(amplitude: int, samples: int, sample_rate_hz: int = 8_000) -> AudioFrame:
    return AudioFrame(
        data=array("h", [amplitude] * samples).tobytes(),
        format=PcmFormat(sample_rate_hz),
    )


class ConversationAudioRecordingTests(unittest.IsolatedAsyncioTestCase):
    def test_retention_worker_has_recording_directory(self) -> None:
        args = parse_retention_args(
            ["--once", "--recording-directory", "private/recordings"]
        )

        self.assertEqual(args.recording_directory, Path("private/recordings"))

    async def test_writes_aligned_private_stereo_recording_and_manifest(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            clock = FakeClock()
            started_at = datetime(2026, 8, 16, 10, 0, tzinfo=timezone.utc)
            recorder = WaveConversationRecorder(
                temporary_directory,
                RecordingConsent("consent-ticket-42"),
                sample_rate_hz=8_000,
                clock=clock,
                now=lambda: started_at + timedelta(seconds=clock.value),
            )
            await recorder.prepare(
                call_id="call-1",
                campaign_id="campaign-1",
                retention_days=30,
            )

            recorder.record_owner_audio(pcm_frame(1_000, 80))
            clock.value = 0.01
            recorder.record_agent_audio(pcm_frame(2_000, 80))
            await recorder.close()

            artifact = recorder.artifact
            self.assertIsNotNone(artifact)
            with wave.open(str(artifact.audio_path), "rb") as audio_file:
                self.assertEqual(audio_file.getnchannels(), 2)
                self.assertEqual(audio_file.getframerate(), 8_000)
                samples = array("h")
                samples.frombytes(audio_file.readframes(audio_file.getnframes()))
            self.assertEqual(samples[0:2], array("h", [1_000, 0]))
            self.assertEqual(samples[160:162], array("h", [0, 2_000]))

            manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["purpose"], "quality_and_training")
            self.assertEqual(manifest["consent_reference"], "consent-ticket-42")
            self.assertEqual(manifest["channels"], {"1": "owner", "2": "agent"})
            self.assertFalse(manifest["contains_transcript"])
            self.assertNotIn("phone_number", manifest)
            self.assertNotIn("recipient_name", manifest)
            self.assertNotIn("property_reference", manifest)
            self.assertEqual(
                os.stat(artifact.audio_path).st_mode & 0o777,
                0o600,
            )
            self.assertEqual(
                os.stat(artifact.manifest_path).st_mode & 0o777,
                0o600,
            )
            self.assertEqual(
                os.stat(Path(temporary_directory)).st_mode & 0o777,
                0o700,
            )

    async def test_delayed_playout_confirmation_preserves_original_timeline(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            clock = FakeClock()
            started_at = datetime(2026, 8, 16, 10, 0, tzinfo=timezone.utc)
            recorder = WaveConversationRecorder(
                temporary_directory,
                RecordingConsent("delayed-playout"),
                sample_rate_hz=8_000,
                clock=clock,
                now=lambda: started_at + timedelta(seconds=clock.value),
            )
            await recorder.prepare(
                call_id="call-delayed",
                campaign_id="campaign-1",
                retention_days=1,
            )
            clock.value = 1.1
            recorder.record_owner_audio(pcm_frame(1_000, 800))
            clock.value = 1.2

            recorder.record_agent_audio(
                pcm_frame(2_000, 8_000),
                started_at_monotonic=0.0,
            )
            await recorder.close()

            with wave.open(str(recorder.artifact.audio_path), "rb") as audio_file:
                samples = array("h")
                samples.frombytes(audio_file.readframes(audio_file.getnframes()))
            self.assertEqual(samples[0:2], array("h", [0, 2_000]))
            owner_offset = round(1.1 * 8_000) * 2
            self.assertEqual(
                samples[owner_offset : owner_offset + 2],
                array("h", [1_000, 0]),
            )

    async def test_purge_uses_manifest_expiry_and_removes_audio(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            clock = FakeClock()
            started_at = datetime(2026, 8, 1, tzinfo=timezone.utc)
            recorder = WaveConversationRecorder(
                temporary_directory,
                RecordingConsent("retention-test"),
                sample_rate_hz=8_000,
                clock=clock,
                now=lambda: started_at,
            )
            await recorder.prepare(
                call_id="call-3",
                campaign_id="campaign-1",
                retention_days=1,
            )
            recorder.record_owner_audio(pcm_frame(1_000, 80))
            await recorder.close()
            audio_path = recorder.artifact.audio_path
            manifest_path = recorder.artifact.manifest_path

            self.assertEqual(
                purge_expired_recordings(
                    temporary_directory,
                    now=started_at + timedelta(hours=12),
                ),
                0,
            )
            self.assertEqual(
                purge_expired_recordings(
                    temporary_directory,
                    now=started_at + timedelta(days=2),
                ),
                1,
            )
            self.assertFalse(audio_path.exists())
            self.assertFalse(manifest_path.exists())

    async def test_rejects_invalid_consent_and_duplicate_artifacts(self) -> None:
        with self.assertRaisesRegex(ValueError, "consent reference"):
            RecordingConsent("contains spaces")

        with TemporaryDirectory() as temporary_directory:
            recorder = WaveConversationRecorder(
                temporary_directory,
                RecordingConsent("duplicate-test"),
                now=lambda: datetime(2026, 8, 16, tzinfo=timezone.utc),
            )
            await recorder.prepare(
                call_id="call-4",
                campaign_id="campaign-1",
                retention_days=1,
            )
            recorder.record_owner_audio(pcm_frame(1_000, 80))
            await recorder.close()
            duplicate = WaveConversationRecorder(
                temporary_directory,
                RecordingConsent("duplicate-test"),
                now=lambda: datetime(2026, 8, 16, tzinfo=timezone.utc),
            )

            with self.assertRaises(AudioRecordingError):
                await duplicate.prepare(
                    call_id="call-4",
                    campaign_id="campaign-1",
                    retention_days=1,
                )

    async def test_no_audio_creates_no_artifact_or_directory(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "recordings"
            recorder = WaveConversationRecorder(
                root,
                RecordingConsent("no-audio"),
                now=lambda: datetime(2026, 8, 16, tzinfo=timezone.utc),
            )
            await recorder.prepare(
                call_id="call-no-audio",
                campaign_id="campaign-1",
                retention_days=1,
            )
            recorder.record_owner_audio(
                AudioFrame(data=b"", format=PcmFormat(16_000))
            )

            await recorder.close()

            self.assertIsNone(recorder.artifact)
            self.assertEqual(list(root.rglob("*")), [])

    async def test_recording_and_retention_refuse_symlinked_storage(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            external = base / "external"
            external.mkdir()
            root_link = base / "recordings"
            root_link.symlink_to(external, target_is_directory=True)
            recorder = WaveConversationRecorder(
                root_link,
                RecordingConsent("symlink-root"),
                now=lambda: datetime(2026, 8, 16, tzinfo=timezone.utc),
            )

            with self.assertRaisesRegex(AudioRecordingError, "symlink"):
                await recorder.prepare(
                    call_id="call-symlink",
                    campaign_id="campaign-1",
                    retention_days=1,
                )
            self.assertEqual(list(external.iterdir()), [])

            safe_root = base / "safe-recordings"
            safe_root.mkdir()
            day_link = safe_root / "2026-08-16"
            day_link.symlink_to(external, target_is_directory=True)
            day_recorder = WaveConversationRecorder(
                safe_root,
                RecordingConsent("symlink-day"),
                now=lambda: datetime(2026, 8, 16, tzinfo=timezone.utc),
            )
            with self.assertRaisesRegex(AudioRecordingError, "symlink"):
                await day_recorder.prepare(
                    call_id="call-day-symlink",
                    campaign_id="campaign-1",
                    retention_days=1,
                )

            external_audio = external / "outside.wav"
            external_audio.write_bytes(b"outside")
            with self.assertRaisesRegex(AudioRecordingError, "symlink"):
                purge_expired_recordings(
                    safe_root,
                    now=datetime(2026, 8, 20, tzinfo=timezone.utc),
                )
            self.assertEqual(external_audio.read_bytes(), b"outside")

    async def test_retention_does_not_remove_active_recording_directory(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "recordings"
            recorder = WaveConversationRecorder(
                root,
                RecordingConsent("active-call"),
                now=lambda: datetime(2026, 8, 16, tzinfo=timezone.utc),
            )
            await recorder.prepare(
                call_id="call-active",
                campaign_id="campaign-1",
                retention_days=1,
            )

            purge_expired_recordings(
                root,
                now=datetime(2026, 8, 16, 0, 30, tzinfo=timezone.utc),
            )
            recorder.record_owner_audio(pcm_frame(1_000, 80))
            await recorder.close()

            self.assertIsNotNone(recorder.artifact)
            self.assertTrue(recorder.artifact.audio_path.exists())

    async def test_cancelled_finalization_removes_partial_and_final_artifacts(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "recordings"
            recorder = WaveConversationRecorder(
                root,
                RecordingConsent("cancel-test"),
                sample_rate_hz=8_000,
                now=lambda: datetime(2026, 8, 16, tzinfo=timezone.utc),
            )
            await recorder.prepare(
                call_id="call-cancel",
                campaign_id="campaign-1",
                retention_days=1,
            )
            recorder.record_owner_audio(pcm_frame(1_000, 80_000))
            close_task = asyncio.create_task(recorder.close())
            await asyncio.sleep(0)

            close_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await close_task

            self.assertEqual(list(root.glob("**/*")), [])

    async def test_partial_collision_and_stale_orphans_fail_or_purge_safely(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "recordings"
            day = root / "2026-08-16"
            day.mkdir(parents=True)
            partial = day / "call-partial.wav.partial"
            partial.write_bytes(b"sensitive")
            recorder = WaveConversationRecorder(
                root,
                RecordingConsent("partial-test"),
                now=lambda: datetime(2026, 8, 16, tzinfo=timezone.utc),
            )

            with self.assertRaises(AudioRecordingError):
                await recorder.prepare(
                    call_id="call-partial",
                    campaign_id="campaign-1",
                    retention_days=1,
                )

            orphan = day / "orphan.wav"
            orphan.write_bytes(b"sensitive")
            recent = day / "active.wav.partial"
            recent.write_bytes(b"active")
            old_timestamp = datetime(2026, 8, 1, tzinfo=timezone.utc).timestamp()
            os.utime(partial, (old_timestamp, old_timestamp))
            os.utime(orphan, (old_timestamp, old_timestamp))

            purge_expired_recordings(
                root,
                now=datetime(2026, 8, 16, tzinfo=timezone.utc),
                orphan_grace_seconds=60,
            )

            self.assertFalse(partial.exists())
            self.assertFalse(orphan.exists())
            self.assertTrue(recent.exists())


if __name__ == "__main__":
    unittest.main()
