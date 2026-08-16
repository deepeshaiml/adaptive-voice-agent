from __future__ import annotations

import asyncio
from array import array
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from itertools import chain
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
from typing import Callable, Protocol
import wave

from speaking_agent.adapters.audio_conversion import pcm_frames_to_mono_float
from speaking_agent.speech import AudioFrame


class AudioRecordingError(RuntimeError):
    """The consented conversation recording could not be completed safely."""


@dataclass(frozen=True, slots=True)
class RecordingConsent:
    reference: str
    purpose: str = "quality_and_training"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,99}", self.reference):
            raise ValueError(
                "Recording consent reference must be 1-100 safe identifier characters"
            )
        if self.purpose != "quality_and_training":
            raise ValueError("Unsupported recording purpose")


@dataclass(frozen=True, slots=True)
class RecordingArtifact:
    audio_path: Path
    manifest_path: Path
    duration_seconds: float


class ConversationAudioRecorder(Protocol):
    artifact: RecordingArtifact | None

    async def prepare(
        self,
        *,
        call_id: str,
        campaign_id: str,
        retention_days: int,
    ) -> None: ...

    def record_owner_audio(self, frame: AudioFrame) -> None: ...

    def record_agent_audio(
        self,
        frame: AudioFrame,
        started_at_monotonic: float,
    ) -> None: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _AudioSegment:
    start_sample: int
    samples: array[int]

    @property
    def end_sample(self) -> int:
        return self.start_sample + len(self.samples)


_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _canonical_storage_path(path: Path) -> Path:
    absolute_path = path.absolute()
    if sys.platform != "darwin":
        return absolute_path
    for alias_text in ("/var", "/tmp", "/etc"):
        alias = Path(alias_text)
        try:
            relative = absolute_path.relative_to(alias)
        except ValueError:
            continue
        expected_target = Path("/private") / alias.name
        try:
            if alias.resolve(strict=True) != expected_target:
                continue
        except OSError:
            continue
        return expected_target / relative
    return absolute_path


def _open_secure_directory(path: Path, *, create: bool) -> int:
    if os.name != "posix":
        raise AudioRecordingError(
            "Secure recording storage currently requires POSIX filesystem semantics"
        )
    absolute_path = _canonical_storage_path(path)
    descriptor = os.open(absolute_path.anchor or "/", _DIRECTORY_FLAGS)
    try:
        for component in absolute_path.parts[1:]:
            if component in {"", ".", ".."}:
                raise AudioRecordingError("Recording directory path is unsafe")
            if create:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            next_descriptor = os.open(
                component,
                _DIRECTORY_FLAGS,
                dir_fd=descriptor,
            )
            if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                os.close(next_descriptor)
                raise AudioRecordingError("Recording path component is not a directory")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError as error:
        os.close(descriptor)
        raise AudioRecordingError(
            f"Recording directory is unavailable or contains a symlink: {path}"
        ) from error
    except BaseException:
        os.close(descriptor)
        raise


def _open_child_directory(parent_fd: int, name: str) -> int:
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as error:
        raise AudioRecordingError(
            f"Recording day directory is unavailable or is a symlink: {name}"
        ) from error
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise AudioRecordingError("Recording day path is not a directory")
    return descriptor


def _entry_exists(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _create_private_file(directory_fd: int, name: str) -> int:
    return os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FILE_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )


class WaveConversationRecorder:
    """Build an aligned stereo WAV: owner on channel 1, agent on channel 2."""

    def __init__(
        self,
        root_directory: str | Path,
        consent: RecordingConsent,
        *,
        sample_rate_hz: int = 16_000,
        maximum_duration_seconds: float = 3_600,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("Recording sample rate must be positive")
        if maximum_duration_seconds <= 0:
            raise ValueError("Maximum recording duration must be positive")
        self.root_directory = Path(root_directory)
        self._storage_root = _canonical_storage_path(self.root_directory)
        self.consent = consent
        self.sample_rate_hz = sample_rate_hz
        self.maximum_samples = round(maximum_duration_seconds * sample_rate_hz)
        self._clock = clock
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._owner_segments: list[_AudioSegment] = []
        self._agent_segments: list[_AudioSegment] = []
        self._owner_cursor = 0
        self._agent_cursor = 0
        self._started_clock: float | None = None
        self._audio_started_clock: float | None = None
        self._started_at: datetime | None = None
        self._call_id: str | None = None
        self._campaign_id: str | None = None
        self._retention_days: int | None = None
        self._audio_path: Path | None = None
        self._manifest_path: Path | None = None
        self._root_fd: int | None = None
        self._day_fd: int | None = None
        self._day_name: str | None = None
        self._audio_name: str | None = None
        self._manifest_name: str | None = None
        self._active_name: str | None = None
        self._closed = False
        self.artifact: RecordingArtifact | None = None

    async def prepare(
        self,
        *,
        call_id: str,
        campaign_id: str,
        retention_days: int,
    ) -> None:
        if self._started_clock is not None:
            raise AudioRecordingError("Conversation recorder is already prepared")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", call_id):
            raise AudioRecordingError("Call ID is unsafe for recording storage")
        if not isinstance(campaign_id, str) or not campaign_id.strip():
            raise AudioRecordingError("Campaign ID is required for recording provenance")
        if not isinstance(retention_days, int) or retention_days < 1:
            raise AudioRecordingError("Recording retention must be a positive day count")

        started_at = self._now()
        if started_at.tzinfo is None:
            raise AudioRecordingError("Recording timestamps must be timezone-aware")
        day_name = started_at.date().isoformat()
        day_directory = self._storage_root / day_name
        audio_name = f"{call_id}.wav"
        manifest_name = f"{call_id}.json"
        active_name = f"{call_id}.active"
        audio_path = day_directory / audio_name
        manifest_path = day_directory / manifest_name
        root_fd: int | None = None
        day_fd: int | None = None
        try:
            root_fd = _open_secure_directory(self._storage_root, create=True)
            try:
                os.mkdir(day_name, mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
            day_fd = _open_child_directory(root_fd, day_name)
            os.fchmod(root_fd, 0o700)
            os.fchmod(day_fd, 0o700)
            artifact_names = (
                audio_name,
                manifest_name,
                f"{audio_name}.partial",
                f"{manifest_name}.partial",
                active_name,
            )
            if any(_entry_exists(day_fd, name) for name in artifact_names):
                raise AudioRecordingError(
                    "Recording artifact already exists for this call"
                )
            active_fd = _create_private_file(day_fd, active_name)
            with os.fdopen(active_fd, "w", encoding="utf-8") as active_file:
                active_file.write(self.consent.reference)
                active_file.write("\n")
        except Exception:
            if day_fd is not None:
                os.close(day_fd)
            if root_fd is not None:
                os.close(root_fd)
            raise

        self._call_id = call_id
        self._campaign_id = campaign_id.strip()
        self._retention_days = retention_days
        self._started_at = started_at.astimezone(timezone.utc)
        self._started_clock = self._clock()
        self._audio_path = audio_path
        self._manifest_path = manifest_path
        self._root_fd = root_fd
        self._day_fd = day_fd
        self._day_name = day_name
        self._audio_name = audio_name
        self._manifest_name = manifest_name
        self._active_name = active_name

    def record_owner_audio(self, frame: AudioFrame) -> None:
        self._record(frame, owner=True, started_at_monotonic=self._clock())

    def record_agent_audio(
        self,
        frame: AudioFrame,
        started_at_monotonic: float | None = None,
    ) -> None:
        self._record(
            frame,
            owner=False,
            started_at_monotonic=(
                self._clock()
                if started_at_monotonic is None
                else started_at_monotonic
            ),
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._started_clock is None:
            return
        if self._audio_started_clock is None:
            self._remove_active_marker()
            self._remove_empty_day_directory()
            self._release_storage()
            return
        try:
            self.artifact = await self._write_artifact()
        except asyncio.CancelledError:
            self._remove_partial_artifacts()
            self._remove_active_marker()
            self._remove_empty_day_directory()
            raise
        except Exception as error:
            self._remove_partial_artifacts()
            self._remove_active_marker()
            self._remove_empty_day_directory()
            if isinstance(error, AudioRecordingError):
                raise
            raise AudioRecordingError("Unable to finalize conversation recording") from error
        else:
            self._remove_active_marker()
        finally:
            self._release_storage()

    def _record(
        self,
        frame: AudioFrame,
        *,
        owner: bool,
        started_at_monotonic: float,
    ) -> None:
        self._ensure_active()
        waveform = pcm_frames_to_mono_float(
            (frame,),
            target_sample_rate_hz=self.sample_rate_hz,
        )
        samples = array(
            "h",
            (
                round(max(-1.0, min(1.0, sample)) * 32_767)
                for sample in waveform
            ),
        )
        if not samples:
            return
        current_clock = self._clock()
        if self._audio_started_clock is None:
            self._audio_started_clock = started_at_monotonic
            self._started_at = (
                self._now().astimezone(timezone.utc)
                - timedelta(seconds=max(0.0, current_clock - started_at_monotonic))
            )
        elif started_at_monotonic < self._audio_started_clock:
            self._move_origin_earlier(started_at_monotonic)
        elapsed_sample = self._sample_for_clock(started_at_monotonic)
        cursor = self._owner_cursor if owner else self._agent_cursor
        start_sample = max(elapsed_sample, cursor)
        end_sample = start_sample + len(samples)
        if end_sample > self.maximum_samples:
            raise AudioRecordingError("Conversation recording exceeded its duration limit")
        segment = _AudioSegment(start_sample, samples)
        if owner:
            self._owner_segments.append(segment)
            self._owner_cursor = end_sample
        else:
            self._agent_segments.append(segment)
            self._agent_cursor = end_sample

    def _ensure_active(self) -> None:
        if self._started_clock is None:
            raise AudioRecordingError("Conversation recorder is not prepared")
        if self._closed:
            raise AudioRecordingError("Conversation recorder is closed")

    def _sample_for_clock(self, clock_value: float) -> int:
        if self._audio_started_clock is None:
            return 0
        return min(
            self.maximum_samples,
            max(
                0,
                round(
                    (clock_value - self._audio_started_clock)
                    * self.sample_rate_hz
                ),
            ),
        )

    def _move_origin_earlier(self, started_at_monotonic: float) -> None:
        if self._audio_started_clock is None or self._started_at is None:
            return
        shift_seconds = self._audio_started_clock - started_at_monotonic
        shift_samples = round(shift_seconds * self.sample_rate_hz)
        self._owner_segments = [
            _AudioSegment(segment.start_sample + shift_samples, segment.samples)
            for segment in self._owner_segments
        ]
        self._agent_segments = [
            _AudioSegment(segment.start_sample + shift_samples, segment.samples)
            for segment in self._agent_segments
        ]
        if self._owner_segments:
            self._owner_cursor += shift_samples
        if self._agent_segments:
            self._agent_cursor += shift_samples
        self._audio_started_clock = started_at_monotonic
        self._started_at -= timedelta(seconds=shift_seconds)

    async def _write_artifact(self) -> RecordingArtifact:
        if (
            self._audio_path is None
            or self._manifest_path is None
            or self._started_at is None
            or self._call_id is None
            or self._campaign_id is None
            or self._retention_days is None
            or self._day_fd is None
            or self._audio_name is None
            or self._manifest_name is None
        ):
            raise AudioRecordingError("Conversation recorder provenance is incomplete")
        total_samples = max(self._owner_cursor, self._agent_cursor)
        temporary_audio = f"{self._audio_name}.partial"
        temporary_manifest = f"{self._manifest_name}.partial"
        await self._write_stereo_wave(
            temporary_audio,
            self._owner_segments,
            self._agent_segments,
            total_samples,
        )
        audio_digest = await self._sha256(temporary_audio)
        completed_at = self._now().astimezone(timezone.utc)
        manifest = {
            "schema_version": 1,
            "call_id": self._call_id,
            "campaign_id": self._campaign_id,
            "purpose": self.consent.purpose,
            "consent_reference": self.consent.reference,
            "started_at": self._started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "retention_until": (
                self._started_at + timedelta(days=self._retention_days)
            ).isoformat(),
            "audio_file": self._audio_name,
            "audio_sha256": audio_digest,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": {"1": "owner", "2": "agent"},
            "duration_seconds": total_samples / self.sample_rate_hz,
            "contains_transcript": False,
        }
        self._write_json(temporary_manifest, manifest)
        await asyncio.sleep(0)
        os.replace(
            temporary_audio,
            self._audio_name,
            src_dir_fd=self._day_fd,
            dst_dir_fd=self._day_fd,
        )
        os.replace(
            temporary_manifest,
            self._manifest_name,
            src_dir_fd=self._day_fd,
            dst_dir_fd=self._day_fd,
        )
        os.chmod(
            self._audio_name,
            0o600,
            dir_fd=self._day_fd,
            follow_symlinks=False,
        )
        os.chmod(
            self._manifest_name,
            0o600,
            dir_fd=self._day_fd,
            follow_symlinks=False,
        )
        return RecordingArtifact(
            audio_path=self._audio_path,
            manifest_path=self._manifest_path,
            duration_seconds=total_samples / self.sample_rate_hz,
        )

    async def _write_stereo_wave(
        self,
        name: str,
        owner_segments: list[_AudioSegment],
        agent_segments: list[_AudioSegment],
        total_samples: int,
    ) -> None:
        if self._day_fd is None:
            raise AudioRecordingError("Recording day directory is unavailable")
        descriptor = _create_private_file(self._day_fd, name)
        with os.fdopen(descriptor, "wb") as raw_file:
            with wave.open(raw_file, "wb") as audio_file:
                audio_file.setnchannels(2)
                audio_file.setsampwidth(2)
                audio_file.setframerate(self.sample_rate_hz)
                owner_index = 0
                agent_index = 0
                for offset in range(0, total_samples, 4_096):
                    end = min(offset + 4_096, total_samples)
                    owner_track, owner_index = self._render_chunk(
                        owner_segments,
                        offset,
                        end,
                        owner_index,
                    )
                    agent_track, agent_index = self._render_chunk(
                        agent_segments,
                        offset,
                        end,
                        agent_index,
                    )
                    stereo = array(
                        "h",
                        chain.from_iterable(zip(owner_track, agent_track)),
                    )
                    if sys.byteorder != "little":
                        stereo.byteswap()
                    audio_file.writeframesraw(stereo.tobytes())
                    await asyncio.sleep(0)
        os.chmod(
            name,
            0o600,
            dir_fd=self._day_fd,
            follow_symlinks=False,
        )

    @staticmethod
    def _render_chunk(
        segments: list[_AudioSegment],
        start: int,
        end: int,
        first_index: int,
    ) -> tuple[array[int], int]:
        while (
            first_index < len(segments)
            and segments[first_index].end_sample <= start
        ):
            first_index += 1
        track = array("h", [0]) * (end - start)
        segment_index = first_index
        while (
            segment_index < len(segments)
            and segments[segment_index].start_sample < end
        ):
            segment = segments[segment_index]
            overlap_start = max(start, segment.start_sample)
            overlap_end = min(end, segment.end_sample)
            if overlap_start < overlap_end:
                source_start = overlap_start - segment.start_sample
                source_end = overlap_end - segment.start_sample
                track[overlap_start - start : overlap_end - start] = (
                    segment.samples[source_start:source_end]
                )
            segment_index += 1
        return track, first_index

    def _write_json(self, name: str, payload: dict[str, object]) -> None:
        if self._day_fd is None:
            raise AudioRecordingError("Recording day directory is unavailable")
        descriptor = _create_private_file(self._day_fd, name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as manifest_file:
            json.dump(payload, manifest_file, indent=2, sort_keys=True)
            manifest_file.write("\n")
        os.chmod(
            name,
            0o600,
            dir_fd=self._day_fd,
            follow_symlinks=False,
        )

    async def _sha256(self, name: str) -> str:
        if self._day_fd is None:
            raise AudioRecordingError("Recording day directory is unavailable")
        digest = hashlib.sha256()
        descriptor = os.open(
            name,
            os.O_RDONLY | _FILE_NOFOLLOW,
            dir_fd=self._day_fd,
        )
        with os.fdopen(descriptor, "rb") as audio_file:
            while chunk := audio_file.read(64 * 1_024):
                digest.update(chunk)
                await asyncio.sleep(0)
        return digest.hexdigest()

    def _remove_partial_artifacts(self) -> None:
        if self._day_fd is None:
            return
        for name in (self._audio_name, self._manifest_name):
            if name is None:
                continue
            for candidate in (name, f"{name}.partial"):
                try:
                    os.unlink(candidate, dir_fd=self._day_fd)
                except FileNotFoundError:
                    pass

    def _remove_active_marker(self) -> None:
        if self._day_fd is None or self._active_name is None:
            return
        try:
            os.unlink(self._active_name, dir_fd=self._day_fd)
        except FileNotFoundError:
            pass

    def _remove_empty_day_directory(self) -> None:
        if (
            self._day_fd is None
            or self._root_fd is None
            or self._day_name is None
        ):
            return
        with os.scandir(self._day_fd) as entries:
            if any(entries):
                return
        try:
            os.rmdir(self._day_name, dir_fd=self._root_fd)
        except OSError:
            pass

    def _release_storage(self) -> None:
        for attribute in ("_day_fd", "_root_fd"):
            descriptor = getattr(self, attribute)
            if descriptor is not None:
                os.close(descriptor)
                setattr(self, attribute, None)


def purge_expired_recordings(
    root_directory: str | Path,
    *,
    now: datetime | None = None,
    orphan_grace_seconds: float = 7_200,
) -> int:
    if orphan_grace_seconds < 0:
        raise ValueError("Recording orphan grace period cannot be negative")
    root = _canonical_storage_path(Path(root_directory))
    if not os.path.lexists(root):
        return 0
    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    stale_timestamp = current_time.timestamp() - orphan_grace_seconds
    purged = 0
    root_fd = _open_secure_directory(root, create=False)
    try:
        with os.scandir(root_fd) as root_entries:
            day_names = sorted(entry.name for entry in root_entries)
        for day_name in day_names:
            day_stat = os.stat(
                day_name,
                dir_fd=root_fd,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(day_stat.st_mode):
                raise AudioRecordingError(
                    f"Recording retention refuses symlinked day directory: {day_name}"
                )
            if not stat.S_ISDIR(day_stat.st_mode):
                continue
            day_fd = _open_child_directory(root_fd, day_name)
            try:
                with os.scandir(day_fd) as entries:
                    entry_stats = {
                        entry.name: entry.stat(follow_symlinks=False)
                        for entry in entries
                    }
                symlinks = [
                    name
                    for name, entry_stat in entry_stats.items()
                    if stat.S_ISLNK(entry_stat.st_mode)
                ]
                if symlinks:
                    raise AudioRecordingError(
                        "Recording retention refuses symlinked artifacts: "
                        + ", ".join(sorted(symlinks))
                    )
                active_calls: set[str] = set()
                for name, entry_stat in entry_stats.items():
                    if not name.endswith(".active"):
                        continue
                    call_id = name.removesuffix(".active")
                    if entry_stat.st_mtime > stale_timestamp:
                        active_calls.add(call_id)
                    else:
                        os.unlink(name, dir_fd=day_fd)

                referenced_audio: set[str] = set()
                for manifest_name in sorted(
                    name
                    for name in entry_stats
                    if name.endswith(".json")
                ):
                    manifest = _read_json_file(day_fd, manifest_name)
                    try:
                        retention_until = datetime.fromisoformat(
                            manifest["retention_until"]
                        )
                        audio_name = manifest["audio_file"]
                    except (KeyError, TypeError, ValueError) as error:
                        raise AudioRecordingError(
                            "Invalid recording retention manifest: "
                            f"{day_name}/{manifest_name}"
                        ) from error
                    expected_audio = f"{manifest_name.removesuffix('.json')}.wav"
                    if audio_name != expected_audio:
                        raise AudioRecordingError(
                            "Recording manifest has an unsafe audio filename: "
                            f"{day_name}/{manifest_name}"
                        )
                    if retention_until.tzinfo is None:
                        raise AudioRecordingError(
                            "Recording retention timestamp is not timezone-aware: "
                            f"{day_name}/{manifest_name}"
                        )
                    if retention_until.astimezone(timezone.utc) > current_time:
                        referenced_audio.add(audio_name)
                        continue
                    try:
                        os.unlink(audio_name, dir_fd=day_fd)
                    except FileNotFoundError:
                        pass
                    os.unlink(manifest_name, dir_fd=day_fd)
                    purged += 1

                with os.scandir(day_fd) as remaining_entries:
                    remaining_stats = {
                        entry.name: entry.stat(follow_symlinks=False)
                        for entry in remaining_entries
                    }
                for name, entry_stat in remaining_stats.items():
                    call_id = name.split(".", 1)[0]
                    if call_id in active_calls:
                        continue
                    is_partial = name.endswith(".partial")
                    is_orphan_audio = (
                        name.endswith(".wav") and name not in referenced_audio
                    )
                    if (
                        (is_partial or is_orphan_audio)
                        and entry_stat.st_mtime <= stale_timestamp
                    ):
                        os.unlink(name, dir_fd=day_fd)

                with os.scandir(day_fd) as final_entries:
                    day_is_empty = not any(final_entries)
            finally:
                os.close(day_fd)
            if day_is_empty:
                os.rmdir(day_name, dir_fd=root_fd)
    finally:
        os.close(root_fd)
    return purged


def _read_json_file(directory_fd: int, name: str) -> dict[str, object]:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | _FILE_NOFOLLOW,
            dir_fd=directory_fd,
        )
        with os.fdopen(descriptor, "r", encoding="utf-8") as manifest_file:
            payload = json.load(manifest_file)
    except (OSError, json.JSONDecodeError) as error:
        raise AudioRecordingError(
            f"Invalid recording retention manifest: {name}"
        ) from error
    if not isinstance(payload, dict):
        raise AudioRecordingError(f"Invalid recording retention manifest: {name}")
    return payload
