"""Durable operator-control command record owned by the live agent.

Only the running agent process opens this SQLite store; the ``agentctl``
CLI talks to the agent over a private Unix socket. Each accepted command
carries the CLI-supplied unique request ID and a monotonically increasing
durable sequence assigned at commit time. Retrying the same request ID with
an identical payload returns the stored acknowledgement without applying
the command twice; reusing an ID with different content is rejected.

This slice implements ``stop``. The record already stores the command kind
and canonical payload so later commands (resume, stop-after, next-issue,
handoff) can reuse the same ordering, retry, and acknowledgement rules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
import json
import sqlite3
import threading
from pathlib import Path


class IntakeState(StrEnum):
    """Durable issue-intake state of one agent instance."""

    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


class CommandAcknowledgement(StrEnum):
    """Durable lifecycle acknowledgement of one operator command."""

    ACCEPTED = "accepted"
    COMPLETED = "completed"
    SUPERSEDED = "superseded"
    NOT_FULFILLED = "not fulfilled"


class ControlStoreError(RuntimeError):
    """Raised when the durable control record cannot be read or committed."""


class RequestIdError(ValueError):
    """Raised when a request ID is missing or malformed (never stored)."""


class PayloadMismatchError(ValueError):
    """Raised when a request ID is reused with different content (never re-applied)."""


@dataclass(frozen=True)
class CommandRecord:
    """One durably recorded operator command."""

    request_id: str
    sequence: int
    kind: str
    acknowledgement: CommandAcknowledgement
    detail: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ActiveAttemptInfo:
    """The in-flight attempt a status snapshot was taken against."""

    issue_number: int
    branch: str
    phase: str
    started_at: str


_STORE_FILENAME = "control.sqlite3"
_STOP_KIND = "stop"


class ControlStore:
    """SQLite-backed command record and intake state under ``$DATA_DIR/state``.

    All public methods share one re-entrant lock so the lifecycle can hold
    the same lock across the claim boundary: either a stop commits first and
    no claim begins, or a claim begins first and the stop applies to the
    active attempt.
    """

    def __init__(
        self, data_dir: Path, *, clock: Callable[[], datetime] | None = None
    ) -> None:
        self._path = data_dir / "state" / _STORE_FILENAME
        self._clock = clock or (lambda: datetime.now(UTC))
        self.lock = threading.RLock()
        with self.lock:
            self._initialize()

    # -- commands ---------------------------------------------------------

    def submit_stop(self, request_id: str, *, has_active_attempt: bool) -> CommandRecord:
        """Durably record ``stop`` and apply its immediate control change.

        Without an active attempt (or when already stopped) the command
        completes immediately; with an active attempt it is accepted as
        pending until the attempt fully finishes.
        """

        _check_request_id(request_id)
        with self.lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    existing = self._find_locked(connection, request_id)
                    if existing is not None:
                        self._ensure_payload_identical(
                            connection, existing, _STOP_KIND, {}
                        )
                        connection.execute("ROLLBACK")
                        return existing
                    intake = self._intake_locked(connection)
                    if intake is IntakeState.STOPPED:
                        acknowledgement = CommandAcknowledgement.COMPLETED
                        detail = "intake already stopped"
                        pending = self._pending_locked(connection)
                    elif intake is IntakeState.STOPPING:
                        # A stop is already pending for the active attempt:
                        # accept this one alongside it without touching the
                        # pending slot. Every accepted stop completes at the
                        # attempt's completion boundary; nothing reports
                        # `completed` before the effect is durably reached.
                        acknowledgement = CommandAcknowledgement.ACCEPTED
                        detail = "stop already pending; completes when the active attempt finishes"
                        pending = self._pending_locked(connection)
                    elif has_active_attempt:
                        acknowledgement = CommandAcknowledgement.ACCEPTED
                        detail = "finish active attempt, then stop intake"
                        intake = IntakeState.STOPPING
                        pending = request_id
                    else:
                        acknowledgement = CommandAcknowledgement.COMPLETED
                        detail = "intake stopped"
                        intake = IntakeState.STOPPED
                        pending = None
                    record = self._insert_locked(
                        connection,
                        request_id,
                        _STOP_KIND,
                        {},
                        acknowledgement,
                        detail,
                    )
                    self._set_intake_locked(connection, intake, pending)
                    connection.execute("COMMIT")
                    return record
            except (PayloadMismatchError, RequestIdError):
                raise
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be committed") from error

    def submit_command(self, kind: str, request_id: str, payload: dict) -> CommandRecord:
        """Generic command entry used to detect request-ID reuse.

        Only ``stop`` applies a control change in this slice; any other kind
        with a fresh ID is rejected without altering control state, while a
        repeated ID returns (or rejects on mismatch) the stored record.
        """

        _check_request_id(request_id)
        if not kind or not kind.strip():
            raise RequestIdError("Command kind must be a non-empty string")
        with self.lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    existing = self._find_locked(connection, request_id)
                    if existing is not None:
                        self._ensure_payload_identical(connection, existing, kind, payload)
                        connection.execute("ROLLBACK")
                        return existing
                    connection.execute("ROLLBACK")
            except (PayloadMismatchError, RequestIdError):
                raise
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be committed") from error
        raise RequestIdError(f"Unsupported command kind: {kind}")

    def get_command(self, request_id: str) -> CommandRecord | None:
        """Return the durable record for one request ID, if any."""

        with self.lock:
            try:
                with self._connect() as connection:
                    return self._find_locked(connection, request_id)
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be read") from error

    def recent_commands(self, limit: int = 10) -> list[CommandRecord]:
        """Return the most recently committed commands, newest last."""

        with self.lock:
            try:
                with self._connect() as connection:
                    rows = connection.execute(
                        "SELECT request_id, sequence, kind, acknowledgement, detail,"
                        " created_at, updated_at FROM commands"
                        " ORDER BY sequence DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
            except sqlite3.Error as error:
                raise ControlStoreError("Control commands could not be read") from error
        return [self._record_from_row(row) for row in reversed(rows)]

    # -- intake -----------------------------------------------------------

    def intake_state(self) -> IntakeState:
        """Return the durable intake state."""

        with self.lock:
            try:
                with self._connect() as connection:
                    return self._intake_locked(connection)
            except sqlite3.Error as error:
                raise ControlStoreError("Control state could not be read") from error

    def pending_command_id(self) -> str | None:
        """Return the request ID of the pending stop plan, if any."""

        with self.lock:
            try:
                with self._connect() as connection:
                    return self._pending_locked(connection)
            except sqlite3.Error as error:
                raise ControlStoreError("Control state could not be read") from error

    def complete_pending_stop(self) -> list[CommandRecord]:
        """Complete every accepted stop after its attempt fully finalized.

        Returns the completed records, or an empty list when no stop is
        pending. A held finalization (unconfirmed publication, release, or
        cleanup) must not call this: the stops stay accepted and intake
        stays ``stopping`` until the attempt is durably finished.
        """

        with self.lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    pending = self._pending_locked(connection)
                    if pending is None:
                        connection.execute("ROLLBACK")
                        return []
                    timestamp = self._timestamp()
                    rows = connection.execute(
                        "SELECT request_id, sequence, kind, acknowledgement, detail,"
                        " created_at, updated_at FROM commands"
                        " WHERE kind = ? AND acknowledgement = ? ORDER BY sequence",
                        (_STOP_KIND, CommandAcknowledgement.ACCEPTED.value),
                    ).fetchall()
                    connection.execute(
                        "UPDATE commands SET acknowledgement = ?, detail = ?,"
                        " updated_at = ? WHERE kind = ? AND acknowledgement = ?",
                        (
                            CommandAcknowledgement.COMPLETED.value,
                            "intake stopped after active attempt",
                            timestamp,
                            _STOP_KIND,
                            CommandAcknowledgement.ACCEPTED.value,
                        ),
                    )
                    connection.execute(
                        "UPDATE control_state SET intake = ?, pending_command_id = NULL"
                        " WHERE id = 1",
                        (IntakeState.STOPPED.value,),
                    )
                    connection.execute("COMMIT")
                    return [
                        CommandRecord(
                            request_id=row[0],
                            sequence=int(row[1]),
                            kind=row[2],
                            acknowledgement=CommandAcknowledgement.COMPLETED,
                            detail="intake stopped after active attempt",
                            created_at=row[5],
                            updated_at=timestamp,
                        )
                        for row in rows
                    ]
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be committed") from error

    # -- internals --------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            connection = sqlite3.connect(str(self._path), timeout=30.0)
        except sqlite3.Error as error:
            raise ControlStoreError("Control store could not be opened") from error
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
        except sqlite3.Error as error:
            connection.close()
            raise ControlStoreError("Control store could not be configured") from error
        return connection

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS commands ("
                    " request_id TEXT PRIMARY KEY,"
                    " sequence INTEGER NOT NULL UNIQUE,"
                    " kind TEXT NOT NULL,"
                    " payload TEXT NOT NULL,"
                    " acknowledgement TEXT NOT NULL,"
                    " detail TEXT NOT NULL,"
                    " created_at TEXT NOT NULL,"
                    " updated_at TEXT NOT NULL"
                    ")"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS control_state ("
                    " id INTEGER PRIMARY KEY CHECK (id = 1),"
                    " intake TEXT NOT NULL,"
                    " pending_command_id TEXT NULL"
                    ")"
                )
                connection.execute(
                    "INSERT OR IGNORE INTO control_state (id, intake, pending_command_id)"
                    " VALUES (1, ?, NULL)",
                    (IntakeState.RUNNING.value,),
                )
                connection.commit()
        except sqlite3.Error as error:
            raise ControlStoreError("Control store could not be initialized") from error

    def _intake_locked(self, connection: sqlite3.Connection) -> IntakeState:
        try:
            row = connection.execute(
                "SELECT intake FROM control_state WHERE id = 1"
            ).fetchone()
        except sqlite3.Error as error:
            raise ControlStoreError("Control state could not be read") from error
        if row is None:  # pragma: no cover - initialized above
            raise ControlStoreError("Control state is missing")
        try:
            return IntakeState(row[0])
        except ValueError as error:
            raise ControlStoreError("Control state is malformed") from error

    def _pending_locked(self, connection: sqlite3.Connection) -> str | None:
        try:
            row = connection.execute(
                "SELECT pending_command_id FROM control_state WHERE id = 1"
            ).fetchone()
        except sqlite3.Error as error:
            raise ControlStoreError("Control state could not be read") from error
        if row is None:  # pragma: no cover - initialized above
            raise ControlStoreError("Control state is missing")
        value = row[0]
        if value is not None and not isinstance(value, str):
            raise ControlStoreError("Control state is malformed")
        return value

    def _set_intake_locked(
        self, connection: sqlite3.Connection, intake: IntakeState, pending: str | None
    ) -> None:
        connection.execute(
            "UPDATE control_state SET intake = ?, pending_command_id = ? WHERE id = 1",
            (intake.value, pending),
        )

    def _find_locked(
        self, connection: sqlite3.Connection, request_id: str
    ) -> CommandRecord | None:
        row = connection.execute(
            "SELECT request_id, sequence, kind, acknowledgement, detail,"
            " created_at, updated_at, payload FROM commands WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            return None
        record = self._record_from_row(row[:7])
        return record

    def _insert_locked(
        self,
        connection: sqlite3.Connection,
        request_id: str,
        kind: str,
        payload: dict,
        acknowledgement: CommandAcknowledgement,
        detail: str,
    ) -> CommandRecord:
        row = connection.execute("SELECT COALESCE(MAX(sequence), 0) FROM commands").fetchone()
        sequence = int(row[0]) + 1
        timestamp = self._timestamp()
        connection.execute(
            "INSERT INTO commands (request_id, sequence, kind, payload,"
            " acknowledgement, detail, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                request_id,
                sequence,
                kind,
                json.dumps(payload, sort_keys=True),
                acknowledgement.value,
                detail,
                timestamp,
                timestamp,
            ),
        )
        return CommandRecord(
            request_id=request_id,
            sequence=sequence,
            kind=kind,
            acknowledgement=acknowledgement,
            detail=detail,
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _ensure_payload_identical(
        self,
        connection: sqlite3.Connection,
        existing: CommandRecord,
        kind: str,
        payload: dict,
    ) -> None:
        if existing.kind != kind:
            raise PayloadMismatchError(
                f"Request ID {existing.request_id} was already used for a different command"
            )
        row = connection.execute(
            "SELECT payload FROM commands WHERE request_id = ?",
            (existing.request_id,),
        ).fetchone()
        if row is None:  # pragma: no cover - just selected above
            raise ControlStoreError("Control command is missing")
        if row[0] != json.dumps(payload, sort_keys=True):
            raise PayloadMismatchError(
                f"Request ID {existing.request_id} was already used with a different payload"
            )

    def _record_from_row(self, row: tuple) -> CommandRecord:
        request_id, sequence, kind, acknowledgement, detail, created_at, updated_at = row[:7]
        try:
            ack = CommandAcknowledgement(acknowledgement)
        except ValueError as error:
            raise ControlStoreError("Control command is malformed") from error
        return CommandRecord(
            request_id=request_id,
            sequence=int(sequence),
            kind=kind,
            acknowledgement=ack,
            detail=detail,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _timestamp(self) -> str:
        value = self._clock()
        if value.tzinfo is None:
            raise ControlStoreError("Control clock must include a timezone")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _check_request_id(request_id: object) -> None:
    if not isinstance(request_id, str) or not request_id.strip():
        raise RequestIdError("Request ID must be a non-empty string")
    if len(request_id) > 128:
        raise RequestIdError("Request ID must be at most 128 characters")


def command_to_json(record: CommandRecord) -> dict:
    """Serialize one command record for socket replies and status snapshots."""

    return {
        "request_id": record.request_id,
        "sequence": record.sequence,
        "kind": record.kind,
        "acknowledgement": record.acknowledgement.value,
        "detail": record.detail,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def build_status(
    *,
    repository: str,
    intake: IntakeState,
    recovering: bool,
    active_attempt: ActiveAttemptInfo | None,
    pending_command_id: str | None,
    get_command: Callable[[str], CommandRecord | None],
    recent_commands: Callable[[], list[CommandRecord]],
) -> dict:
    """Build one consistent live status snapshot.

    Read-only: it never opens the database directly but renders records the
    caller already serialized under the control lock.
    """

    pending = get_command(pending_command_id) if pending_command_id else None
    return {
        "repository": repository,
        "intake": "recovering" if recovering else intake.value,
        "recovering": recovering,
        "active_attempt": (
            {
                "issue_number": active_attempt.issue_number,
                "branch": active_attempt.branch,
                "phase": active_attempt.phase,
                "started_at": active_attempt.started_at,
            }
            if active_attempt is not None
            else None
        ),
        "pending_command": command_to_json(pending) if pending is not None else None,
        "commands": {record.request_id: command_to_json(record) for record in recent_commands()},
    }
