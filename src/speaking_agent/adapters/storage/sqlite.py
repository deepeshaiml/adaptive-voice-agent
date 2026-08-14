from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hmac
import json
import os
from pathlib import Path
import sqlite3
from typing import Any

from speaking_agent.records import (
    AttemptReservationStatus,
    CallRecord,
    RetentionMigrationRequiredError,
    SuppressionKeyMismatchError,
)


class SQLiteCallRepository:
    def __init__(
        self,
        path: str | Path,
        *,
        retention_days: int = 30,
        legacy_retention_days: int | None = None,
    ) -> None:
        if retention_days < 1:
            raise ValueError("retention_days must be positive")
        if legacy_retention_days is not None and legacy_retention_days < 1:
            raise ValueError("legacy_retention_days must be positive")
        self.path = Path(path)
        self.retention_days = retention_days
        self.legacy_retention_days = legacy_retention_days
        self._lock = asyncio.Lock()

    async def prepare(self) -> None:
        parent_existed = self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not parent_existed:
            self.path.parent.chmod(0o700)
        async with self._lock:
            await asyncio.to_thread(self._initialize)

    async def close(self) -> None:
        return None

    async def save(self, record: CallRecord) -> None:
        async with self._lock:
            await asyncio.to_thread(self._save, record)

    async def get(self, call_id: str) -> CallRecord | None:
        async with self._lock:
            row = await asyncio.to_thread(self._get, call_id)
        return self._from_row(row) if row is not None else None

    async def list_recent(self, limit: int = 50) -> list[CallRecord]:
        if limit < 1 or limit > 1_000:
            raise ValueError("limit must be between 1 and 1000")
        async with self._lock:
            rows = await asyncio.to_thread(self._list_recent, limit)
        return [self._from_row(row) for row in rows]

    async def suppress_contact(
        self,
        phone_fingerprint: str,
        source_call_id: str,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._suppress_contact,
                phone_fingerprint,
                source_call_id,
            )

    async def is_contact_suppressed(self, phone_fingerprint: str) -> bool:
        async with self._lock:
            return await asyncio.to_thread(
                self._is_contact_suppressed,
                phone_fingerprint,
            )

    async def ensure_suppression_key(self, key_identifier: str) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._ensure_suppression_key,
                key_identifier,
            )

    async def purge_expired(self, *, now: datetime | None = None) -> int:
        cutoff = now or datetime.now(timezone.utc)
        async with self._lock:
            return await asyncio.to_thread(
                self._purge_expired,
                cutoff.isoformat(),
            )

    async def reserve_contact_attempt(
        self,
        phone_fingerprint: str,
        call_id: str,
        attempted_at: str,
        count_since: str,
        interval_since: str,
        maximum_attempts: int,
    ) -> AttemptReservationStatus:
        async with self._lock:
            return await asyncio.to_thread(
                self._reserve_contact_attempt,
                phone_fingerprint,
                call_id,
                attempted_at,
                count_since,
                interval_since,
                maximum_attempts,
            )

    def _connect(self) -> sqlite3.Connection:
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(descriptor)
        self.path.chmod(0o600)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        self._secure_database_files()
        return connection

    def _secure_database_files(self) -> None:
        for path in (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
        ):
            if path.exists():
                path.chmod(0o600)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS call_records (
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
                    disconnected INTEGER NOT NULL DEFAULT 0,
                    duration_seconds REAL NOT NULL,
                    latencies_json TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_call_records_completed_at
                    ON call_records(completed_at DESC);
                CREATE INDEX IF NOT EXISTS idx_call_records_campaign
                    ON call_records(campaign_id, completed_at DESC);
                CREATE TABLE IF NOT EXISTS contact_suppressions (
                    phone_fingerprint TEXT PRIMARY KEY,
                    source_call_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS contact_attempts (
                    call_id TEXT PRIMARY KEY,
                    phone_fingerprint TEXT NOT NULL,
                    attempted_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_contact_attempts_fingerprint_time
                    ON contact_attempts(phone_fingerprint, attempted_at DESC);
                CREATE TABLE IF NOT EXISTS repository_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(call_records)")
            }
            if "disconnected" not in columns:
                connection.execute(
                    """
                    ALTER TABLE call_records
                    ADD COLUMN disconnected INTEGER NOT NULL DEFAULT 0
                    """
                )
            if "expires_at" not in columns:
                connection.execute(
                    "ALTER TABLE call_records ADD COLUMN expires_at TEXT"
                )
            attempt_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(contact_attempts)")
            }
            if "expires_at" not in attempt_columns:
                connection.execute(
                    "ALTER TABLE contact_attempts ADD COLUMN expires_at TEXT"
                )
            legacy_call_rows = connection.execute(
                "SELECT COUNT(*) FROM call_records WHERE expires_at IS NULL"
            ).fetchone()[0]
            legacy_attempt_rows = connection.execute(
                "SELECT COUNT(*) FROM contact_attempts WHERE expires_at IS NULL"
            ).fetchone()[0]
            if legacy_call_rows or legacy_attempt_rows:
                if self.legacy_retention_days is None:
                    raise RetentionMigrationRequiredError(
                        "Legacy rows require an explicit legacy retention horizon"
                    )
                self._backfill_expiration(
                    connection,
                    retention_days=self.legacy_retention_days,
                )

    def _ensure_suppression_key(self, key_identifier: str) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT value FROM repository_metadata
                WHERE key = 'suppression_key_identifier'
                """
            ).fetchone()
            if row is None:
                existing_private_rows = connection.execute(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM contact_suppressions) +
                        (SELECT COUNT(*) FROM contact_attempts)
                    """
                ).fetchone()[0]
                if existing_private_rows:
                    raise SuppressionKeyMismatchError(
                        "Existing suppression data has no key identifier"
                    )
                connection.execute(
                    """
                    INSERT INTO repository_metadata (key, value)
                    VALUES ('suppression_key_identifier', ?)
                    """,
                    (key_identifier,),
                )
                connection.commit()
                return
            if not hmac.compare_digest(row["value"], key_identifier):
                raise SuppressionKeyMismatchError(
                    "Suppression key does not match this database"
                )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _save(self, record: CallRecord) -> None:
        expires_at = self._expiration_for(
            record.completed_at,
            self.retention_days,
        )
        values = (
            record.call_id,
            record.session_id,
            record.campaign_id,
            record.connection_result,
            record.outcome,
            int(record.qualified),
            record.summary,
            json.dumps(record.fields, sort_keys=True),
            int(record.callback_requested),
            int(record.human_followup_required),
            record.answer_kind,
            record.phone_number_masked,
            record.interruptions,
            int(record.disconnected),
            record.duration_seconds,
            json.dumps(record.latencies, sort_keys=True),
            record.completed_at,
            expires_at,
            record.error,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO call_records (
                    call_id, session_id, campaign_id, connection_result,
                    outcome, qualified, summary, fields_json,
                    callback_requested, human_followup_required, answer_kind,
                    phone_number_masked, interruptions, disconnected, duration_seconds,
                    latencies_json, completed_at, expires_at, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(call_id) DO UPDATE SET
                    connection_result=excluded.connection_result,
                    outcome=excluded.outcome,
                    qualified=excluded.qualified,
                    summary=excluded.summary,
                    fields_json=excluded.fields_json,
                    callback_requested=excluded.callback_requested,
                    human_followup_required=excluded.human_followup_required,
                    answer_kind=excluded.answer_kind,
                    phone_number_masked=excluded.phone_number_masked,
                    interruptions=excluded.interruptions,
                    disconnected=excluded.disconnected,
                    duration_seconds=excluded.duration_seconds,
                    latencies_json=excluded.latencies_json,
                    completed_at=excluded.completed_at,
                    expires_at=excluded.expires_at,
                    error=excluded.error
                """,
                values,
            )

    def _get(self, call_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM call_records WHERE call_id = ?",
                (call_id,),
            ).fetchone()

    def _list_recent(self, limit: int) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM call_records ORDER BY completed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

    def _suppress_contact(
        self,
        phone_fingerprint: str,
        source_call_id: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO contact_suppressions (
                    phone_fingerprint, source_call_id, created_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(phone_fingerprint) DO NOTHING
                """,
                (
                    phone_fingerprint,
                    source_call_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def _is_contact_suppressed(self, phone_fingerprint: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM contact_suppressions
                WHERE phone_fingerprint = ?
                """,
                (phone_fingerprint,),
            ).fetchone()
        return row is not None

    def _purge_expired(self, cutoff: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM call_records WHERE expires_at <= ?",
                (cutoff,),
            )
            attempt_cursor = connection.execute(
                "DELETE FROM contact_attempts WHERE expires_at <= ?",
                (cutoff,),
            )
            return cursor.rowcount + attempt_cursor.rowcount

    def _reserve_contact_attempt(
        self,
        phone_fingerprint: str,
        call_id: str,
        attempted_at: str,
        count_since: str,
        interval_since: str,
        maximum_attempts: int,
    ) -> AttemptReservationStatus:
        expires_at = self._expiration_for(
            attempted_at,
            self.retention_days,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            suppressed = connection.execute(
                """
                SELECT 1 FROM contact_suppressions
                WHERE phone_fingerprint = ?
                LIMIT 1
                """,
                (phone_fingerprint,),
            ).fetchone()
            if suppressed is not None:
                connection.rollback()
                return AttemptReservationStatus.SUPPRESSED

            attempt_count = connection.execute(
                """
                SELECT COUNT(*) FROM contact_attempts
                WHERE phone_fingerprint = ? AND attempted_at >= ?
                """,
                (phone_fingerprint, count_since),
            ).fetchone()[0]
            if attempt_count >= maximum_attempts:
                connection.rollback()
                return AttemptReservationStatus.MAXIMUM_REACHED

            too_recent = connection.execute(
                """
                SELECT 1 FROM contact_attempts
                WHERE phone_fingerprint = ? AND attempted_at >= ?
                LIMIT 1
                """,
                (phone_fingerprint, interval_since),
            ).fetchone()
            if too_recent is not None:
                connection.rollback()
                return AttemptReservationStatus.TOO_SOON

            connection.execute(
                """
                INSERT INTO contact_attempts (
                    call_id, phone_fingerprint, attempted_at, expires_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(call_id) DO NOTHING
                """,
                (call_id, phone_fingerprint, attempted_at, expires_at),
            )
            connection.commit()
            return AttemptReservationStatus.RESERVED
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _expiration_for(timestamp: str, retention_days: int) -> str:
        return (
            datetime.fromisoformat(timestamp) + timedelta(days=retention_days)
        ).isoformat()

    def _backfill_expiration(
        self,
        connection: sqlite3.Connection,
        *,
        retention_days: int,
    ) -> None:
        for table, timestamp_column in (
            ("call_records", "completed_at"),
            ("contact_attempts", "attempted_at"),
        ):
            rows = connection.execute(
                f"SELECT rowid, {timestamp_column} FROM {table} WHERE expires_at IS NULL"
            ).fetchall()
            connection.executemany(
                f"UPDATE {table} SET expires_at = ? WHERE rowid = ?",
                [
                    (
                        self._expiration_for(
                            row[timestamp_column],
                            retention_days,
                        ),
                        row["rowid"],
                    )
                    for row in rows
                ],
            )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> CallRecord:
        values: dict[str, Any] = dict(row)
        return CallRecord(
            call_id=values["call_id"],
            session_id=values["session_id"],
            campaign_id=values["campaign_id"],
            connection_result=values["connection_result"],
            outcome=values["outcome"],
            qualified=bool(values["qualified"]),
            summary=values["summary"],
            fields=json.loads(values["fields_json"]),
            callback_requested=bool(values["callback_requested"]),
            human_followup_required=bool(values["human_followup_required"]),
            answer_kind=values["answer_kind"],
            phone_number_masked=values["phone_number_masked"],
            interruptions=values["interruptions"],
            disconnected=bool(values["disconnected"]),
            duration_seconds=values["duration_seconds"],
            latencies=json.loads(values["latencies_json"]),
            completed_at=values["completed_at"],
            error=values["error"],
        )
