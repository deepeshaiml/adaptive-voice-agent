from pathlib import Path
from datetime import datetime, timedelta, timezone
import sqlite3
import stat
import tempfile
import unittest

from speaking_agent.adapters.storage.sqlite import SQLiteCallRepository
from speaking_agent.records import (
    CallRecord,
    RetentionMigrationRequiredError,
    SuppressionKeyMismatchError,
)
from speaking_agent.suppression import (
    ContactAttemptLimitError,
    ContactAttemptPolicy,
    ContactSuppressionService,
    ContactSuppressedError,
    phone_fingerprint,
)


class SQLiteCallRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_saves_and_reads_structured_call_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SQLiteCallRepository(Path(directory) / "calls.db")
            await repository.prepare()
            expected = CallRecord(
                call_id="call-1",
                session_id="session-1",
                campaign_id="campaign-1",
                connection_result="ANSWERED_HUMAN",
                outcome="SELL",
                qualified=True,
                summary="Owner may sell.",
                fields={"intent": "SELL", "currently_listed": False},
                phone_number_masked="***0123",
                latencies={"speech_end_to_playback": 0.42},
            )

            await repository.save(expected)
            actual = await repository.get("call-1")
            recent = await repository.list_recent()

        self.assertEqual(actual, expected)
        self.assertEqual(recent, [expected])

    async def test_upsert_replaces_result_without_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SQLiteCallRepository(Path(directory) / "calls.db")
            await repository.prepare()
            base = CallRecord(
                call_id="call-1",
                session_id="session-1",
                campaign_id="campaign-1",
                connection_result="FAILED",
                outcome="UNKNOWN",
                qualified=False,
                summary="Failed.",
            )
            await repository.save(base)
            await repository.save(
                CallRecord(
                    call_id="call-1",
                    session_id="session-1",
                    campaign_id="campaign-1",
                    connection_result="ANSWERED_HUMAN",
                    outcome="RENT",
                    qualified=True,
                    summary="Rental lead.",
                )
            )

            records = await repository.list_recent()

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].outcome, "RENT")

    async def test_suppression_uses_keyed_fingerprint_without_raw_number(self) -> None:
        phone_number = "+15105550123"
        secret = "a" * 32
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.db"
            repository = SQLiteCallRepository(path)
            await repository.prepare()
            service = ContactSuppressionService(repository, secret)

            await service.ensure_allowed(phone_number)
            await service.suppress(phone_number, "call-1")

            with self.assertRaises(ContactSuppressedError):
                await service.ensure_allowed(phone_number)
            database_bytes = path.read_bytes()

        self.assertNotIn(phone_number.encode(), database_bytes)
        self.assertNotEqual(phone_fingerprint(phone_number, secret), phone_number)

    def test_suppression_key_must_be_at_least_32_bytes(self) -> None:
        with self.assertRaisesRegex(ValueError, "32 bytes"):
            phone_fingerprint("+15105550123", "too-short")

    async def test_suppression_key_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SQLiteCallRepository(Path(directory) / "calls.db")
            await repository.prepare()
            first = ContactSuppressionService(repository, "a" * 32)
            second = ContactSuppressionService(repository, "b" * 32)
            await first.prepare()
            await first.suppress("+15105550123", "call-1")

            with self.assertRaisesRegex(SuppressionKeyMismatchError, "does not match"):
                await second.ensure_allowed("+15105550123")

    async def test_legacy_suppression_without_key_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SQLiteCallRepository(Path(directory) / "calls.db")
            await repository.prepare()
            await repository.suppress_contact("legacy-fingerprint", "call-1")

            with self.assertRaisesRegex(SuppressionKeyMismatchError, "no key identifier"):
                await ContactSuppressionService(repository, "a" * 32).prepare()

    async def test_database_file_is_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.db"
            repository = SQLiteCallRepository(path)
            await repository.prepare()

            mode = stat.S_IMODE(path.stat().st_mode)

        self.assertEqual(mode, 0o600)

    async def test_purges_expired_call_records_but_keeps_suppression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SQLiteCallRepository(Path(directory) / "calls.db")
            await repository.prepare()
            old_time = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
            await repository.save(
                CallRecord(
                    call_id="old",
                    session_id="session-old",
                    campaign_id="campaign-1",
                    connection_result="ANSWERED_HUMAN",
                    outcome="SELL",
                    qualified=True,
                    summary="Old record.",
                    completed_at=old_time,
                )
            )
            service = ContactSuppressionService(repository, "a" * 32)
            await service.suppress("+15105550123", "old")

            purged = await repository.purge_expired()
            records = await repository.list_recent()

            self.assertEqual(purged, 1)
            self.assertEqual(records, [])
            with self.assertRaises(ContactSuppressedError):
                await service.ensure_allowed("+15105550123")

    async def test_migrates_legacy_database_with_disconnected_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.db"
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    CREATE TABLE call_records (
                        call_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        campaign_id TEXT NOT NULL,
                        connection_result TEXT NOT NULL,
                        outcome TEXT NOT NULL,
                        qualified INTEGER NOT NULL,
                        summary TEXT NOT NULL,
                        fields_json TEXT NOT NULL,
                        callback_requested INTEGER NOT NULL,
                        human_followup_required INTEGER NOT NULL,
                        answer_kind TEXT,
                        phone_number_masked TEXT,
                        interruptions INTEGER NOT NULL,
                        duration_seconds REAL NOT NULL,
                        latencies_json TEXT NOT NULL,
                        completed_at TEXT NOT NULL,
                        error TEXT
                    )
                    """
                )
            repository = SQLiteCallRepository(path)

            await repository.prepare()
            with sqlite3.connect(path) as connection:
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(call_records)")
                }

        self.assertIn("disconnected", columns)

    async def test_populated_legacy_retention_requires_explicit_horizon(self) -> None:
        attempted_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.db"
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    CREATE TABLE contact_attempts (
                        call_id TEXT PRIMARY KEY,
                        phone_fingerprint TEXT NOT NULL,
                        attempted_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO contact_attempts (
                        call_id, phone_fingerprint, attempted_at
                    ) VALUES (?, ?, ?)
                    """,
                    ("legacy-attempt", "fingerprint", attempted_at.isoformat()),
                )

            with self.assertRaises(RetentionMigrationRequiredError):
                await SQLiteCallRepository(path).prepare()

            repository = SQLiteCallRepository(
                path,
                legacy_retention_days=90,
            )
            await repository.prepare()
            purged = await repository.purge_expired(
                now=attempted_at + timedelta(days=31)
            )
            with sqlite3.connect(path) as connection:
                row = connection.execute(
                    """
                    SELECT expires_at FROM contact_attempts
                    WHERE call_id = 'legacy-attempt'
                    """
                ).fetchone()

        self.assertEqual(purged, 0)
        self.assertIsNotNone(row)
        self.assertEqual(
            row[0],
            (attempted_at + timedelta(days=90)).isoformat(),
        )

    async def test_contact_attempt_limits_and_interval(self) -> None:
        now = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            repository = SQLiteCallRepository(Path(directory) / "calls.db")
            await repository.prepare()
            policy = ContactAttemptPolicy(
                repository,
                "a" * 32,
                maximum_attempts=2,
                window_hours=24,
                minimum_interval_minutes=15,
            )
            await policy.reserve("+15105550123", "call-1", now=now)

            with self.assertRaisesRegex(ContactAttemptLimitError, "interval"):
                await policy.reserve(
                    "+15105550123",
                    "call-too-soon",
                    now=now + timedelta(minutes=5),
                )

            await policy.reserve(
                "+15105550123",
                "call-2",
                now=now + timedelta(minutes=20),
            )
            with self.assertRaisesRegex(ContactAttemptLimitError, "Maximum"):
                await policy.reserve(
                    "+15105550123",
                    "call-3",
                    now=now + timedelta(minutes=40),
                )

    async def test_attempt_reservation_is_atomic_across_repositories(self) -> None:
        now = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.db"
            first_repository = SQLiteCallRepository(path)
            second_repository = SQLiteCallRepository(path)
            await first_repository.prepare()
            first_policy = ContactAttemptPolicy(
                first_repository,
                "a" * 32,
                maximum_attempts=1,
                window_hours=24,
                minimum_interval_minutes=1,
            )
            second_policy = ContactAttemptPolicy(
                second_repository,
                "a" * 32,
                maximum_attempts=1,
                window_hours=24,
                minimum_interval_minutes=1,
            )

            results = await __import__("asyncio").gather(
                first_policy.reserve("+15105550123", "call-1", now=now),
                second_policy.reserve("+15105550123", "call-2", now=now),
                return_exceptions=True,
            )

        self.assertEqual(sum(result is None for result in results), 1)
        self.assertEqual(
            sum(isinstance(result, ContactAttemptLimitError) for result in results),
            1,
        )

    async def test_atomic_reservation_rechecks_suppression_without_attempt(self) -> None:
        now = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.db"
            repository = SQLiteCallRepository(path)
            await repository.prepare()
            suppression = ContactSuppressionService(repository, "a" * 32)
            attempts = ContactAttemptPolicy(
                repository,
                "a" * 32,
                maximum_attempts=3,
                window_hours=24,
                minimum_interval_minutes=15,
            )
            await suppression.suppress("+15105550123", "source-call")

            with self.assertRaises(ContactSuppressedError):
                await attempts.reserve(
                    "+15105550123",
                    "blocked-call",
                    now=now,
                )
            with sqlite3.connect(path) as connection:
                count = connection.execute(
                    "SELECT COUNT(*) FROM contact_attempts"
                ).fetchone()[0]

        self.assertEqual(count, 0)

    async def test_each_campaign_retains_rows_for_its_own_horizon(self) -> None:
        completed_at = datetime(2026, 8, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.db"
            short_retention = SQLiteCallRepository(path, retention_days=1)
            long_retention = SQLiteCallRepository(path, retention_days=30)
            await short_retention.prepare()
            await long_retention.prepare()
            await short_retention.save(
                CallRecord(
                    call_id="short",
                    session_id="session-short",
                    campaign_id="short-campaign",
                    connection_result="ANSWERED_HUMAN",
                    outcome="UNKNOWN",
                    qualified=False,
                    summary="Short retention.",
                    completed_at=completed_at.isoformat(),
                )
            )
            await long_retention.save(
                CallRecord(
                    call_id="long",
                    session_id="session-long",
                    campaign_id="long-campaign",
                    connection_result="ANSWERED_HUMAN",
                    outcome="UNKNOWN",
                    qualified=False,
                    summary="Long retention.",
                    completed_at=completed_at.isoformat(),
                )
            )
            await ContactAttemptPolicy(
                short_retention,
                "a" * 32,
                maximum_attempts=3,
                window_hours=24,
                minimum_interval_minutes=15,
            ).reserve("+15105550123", "short-attempt", now=completed_at)
            await ContactAttemptPolicy(
                long_retention,
                "a" * 32,
                maximum_attempts=3,
                window_hours=24,
                minimum_interval_minutes=15,
            ).reserve("+15105550124", "long-attempt", now=completed_at)

            purged = await short_retention.purge_expired(
                now=completed_at + timedelta(days=2)
            )
            records = await short_retention.list_recent()
            with sqlite3.connect(path) as connection:
                attempt_ids = [
                    row[0]
                    for row in connection.execute(
                        "SELECT call_id FROM contact_attempts ORDER BY call_id"
                    )
                ]

        self.assertEqual(purged, 2)
        self.assertEqual([record.call_id for record in records], ["long"])
        self.assertEqual(attempt_ids, ["long-attempt"])


if __name__ == "__main__":
    unittest.main()