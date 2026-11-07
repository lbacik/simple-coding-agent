"""Read-only view of a stopped instance's durable control state.

The control store is SQLite in WAL mode. A live engine's committed rows
may live only in the ``-wal`` file, which cannot be opened normally on a
read-only mount (SQLite needs to create ``-shm`` next to it), while an
``immutable`` open ignores the WAL and would silently return stale rows.
The offline reader therefore never opens the volume files with SQLite:
it copies the main database file and its ``-wal`` file (when one exists)
into a writable temporary directory and opens only the copy, where the
WAL is checkpointed and verified before any row is returned.

The reader never takes the volume lock and never writes to the volume:
the source files are opened read-only for a byte copy, so no ``-wal``
or ``-shm`` file is created or modified next to them. When the committed
state cannot be read reliably the reader raises :class:`OfflineReadError`
instead of returning older rows; the contract adapter reports that as
``internal``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

from simple_coding_agent.control import (
    _STORE_FILENAME,
    CommandAcknowledgement,
    CommandRecord,
    IntakeState,
)


class OfflineReadError(RuntimeError):
    """Raised when the stopped instance's committed state cannot be read reliably."""


@dataclass(frozen=True)
class OfflineControlSnapshot:
    """The committed control state of a stopped instance (Intake, Operator Commands)."""

    intake: IntakeState
    pending_command_id: str | None
    commands: tuple[CommandRecord, ...]
    payloads: dict | None = None
    stop_after_requested: int | None = None
    stop_after_remaining: int | None = None
    stop_after_active_attempt_id: str | None = None
    latest_counted_attempt_id: str | None = None
    latest_counted_outcome: str | None = None
    next_issue_number: int | None = None
    next_issue_request_id: str | None = None
    handoff_request_id: str | None = None
    handoff_accepted_at: str | None = None
    handoff_begun: bool = False
    handoff_phase: str | None = None

    def payload_for(self, request_id: str) -> dict:
        """Return the canonical payload for one command, or ``{}``."""

        if not isinstance(self.payloads, dict):
            return {}
        payload = self.payloads.get(request_id)
        return dict(payload) if isinstance(payload, dict) else {}


def read_offline_control(data_dir: Path) -> OfflineControlSnapshot:
    """Return every committed control row without touching the volume.

    Copies ``state/control.sqlite3`` and its ``-wal`` file (when present)
    to a temporary directory, checkpoints and verifies the copy, and
    returns Intake plus all commands oldest first. Raises
    :class:`OfflineReadError` when the committed state cannot be read
    reliably instead of returning older rows.
    """

    state_dir = Path(data_dir) / "state"
    source_db = state_dir / _STORE_FILENAME
    source_wal = source_db.with_name(source_db.name + "-wal")
    if not source_db.is_file():
        raise OfflineReadError(
            f"Control store {source_db} is missing; the committed state cannot be read."
        )
    try:
        before = _source_fingerprint(source_db, source_wal)
    except OSError as error:
        raise OfflineReadError(
            f"Control store {source_db} could not be read: {error}."
        ) from error
    try:
        with tempfile.TemporaryDirectory(prefix="control-offline-") as work:
            copy_db = Path(work) / _STORE_FILENAME
            _copy_source(source_db, source_wal, copy_db)
            _check_source_stable(source_db, source_wal, before)
            snapshot = _read_copy(copy_db)
    except OfflineReadError:
        raise
    except OSError as error:
        raise OfflineReadError(
            f"Control store {source_db} could not be copied for offline reading: {error}."
        ) from error
    return snapshot


def _source_fingerprint(db: Path, wal: Path) -> tuple[tuple[int, int] | None, ...]:
    """Record the source sizes and mtimes so a torn copy is never trusted."""

    return (_file_identity(db), _file_identity(wal))


def _file_identity(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_size, stat.st_mtime_ns)


def _copy_source(source_db: Path, source_wal: Path, copy_db: Path) -> None:
    """Copy the main file and, when present, its WAL next to the copy.

    The ``-shm`` file is deliberately never copied: it is only a live
    WAL index that SQLite rebuilds from the WAL itself. A ``-wal`` file
    that exists but cannot be copied is fatal: falling back to the main
    file alone would silently return stale rows.
    """

    try:
        shutil.copyfile(source_db, copy_db)
    except FileNotFoundError as error:
        raise OfflineReadError(
            f"Control store {source_db} is missing; the committed state cannot be read."
        ) from error
    if os.path.lexists(source_wal) and not source_wal.is_file():
        raise OfflineReadError(
            f"Control WAL {source_wal} is not a readable file; refusing to"
            " return the main file's older rows as current."
        )
    if source_wal.is_file():
        try:
            shutil.copyfile(source_wal, copy_db.with_name(copy_db.name + "-wal"))
        except OSError as error:
            raise OfflineReadError(
                f"Control WAL {source_wal} could not be copied; refusing to"
                " return the main file's older rows as current."
            ) from error


def _check_source_stable(
    source_db: Path,
    source_wal: Path,
    before: tuple[tuple[int, int] | None, ...],
) -> None:
    """Refuse a copy taken while the source changed under the read."""

    try:
        after = _source_fingerprint(source_db, source_wal)
    except OSError as error:
        raise OfflineReadError(
            f"Control store {source_db} could not be re-checked during offline reading: {error}."
        ) from error
    if after != before:
        raise OfflineReadError(
            f"Control store {source_db} changed during offline reading;"
            " the volume may still be held by a live engine."
        )


def _read_copy(copy_db: Path) -> OfflineControlSnapshot:
    """Checkpoint, verify, then read the writable copy; never stale, never silent."""

    import json as _json

    try:
        connection = sqlite3.connect(str(copy_db), timeout=30.0)
    except sqlite3.Error as error:
        raise OfflineReadError(
            "The copied control store could not be opened; the committed"
            " state cannot be read reliably."
        ) from error
    try:
        try:
            checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        except sqlite3.Error as error:
            raise OfflineReadError(
                "The control WAL could not be applied; refusing to return older rows."
            ) from error
        if checkpoint is None or int(checkpoint[0]) != 0 or int(checkpoint[2]) != int(checkpoint[1]):
            raise OfflineReadError(
                "The control WAL could not be fully applied; refusing to return older rows."
            )
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
        except sqlite3.Error as error:
            raise OfflineReadError(
                "The copied control store failed its integrity check."
            ) from error
        if integrity is None or integrity[0] != "ok":
            raise OfflineReadError("The copied control store failed its integrity check.")
        try:
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(control_state)"
                ).fetchall()
            }
        except sqlite3.Error as error:
            raise OfflineReadError("The control state could not be read.") from error
        wanted = (
            "intake",
            "pending_command_id",
            "stop_after_requested",
            "stop_after_remaining",
            "stop_after_active_attempt_id",
            "latest_counted_attempt_id",
            "latest_counted_outcome",
            "next_issue_number",
            "next_issue_request_id",
            "handoff_request_id",
            "handoff_accepted_at",
            "handoff_begun",
            "handoff_phase",
        )
        available = [name for name in wanted if name in columns]
        if "intake" not in available or "pending_command_id" not in available:
            raise OfflineReadError("The control state is malformed.")
        try:
            state_row = connection.execute(
                f"SELECT {', '.join(available)} FROM control_state WHERE id = 1"
            ).fetchone()
        except sqlite3.Error as error:
            raise OfflineReadError("The control state could not be read.") from error
        if state_row is None:
            raise OfflineReadError("The control state is missing.")
        state = dict(zip(available, state_row, strict=True))
        try:
            intake = IntakeState(state.get("intake"))
        except ValueError as error:
            raise OfflineReadError("The control state is malformed.") from error
        pending = state.get("pending_command_id")
        if pending is not None and not isinstance(pending, str):
            raise OfflineReadError("The control state is malformed.")
        extra = _read_extra_state(state)
        try:
            rows = connection.execute(
                "SELECT request_id, sequence, kind, acknowledgement, detail,"
                " created_at, updated_at FROM commands ORDER BY sequence"
            ).fetchall()
        except sqlite3.Error as error:
            raise OfflineReadError("The control commands could not be read.") from error
        commands = tuple(_record_from_row(row) for row in rows)
        try:
            payload_rows = connection.execute(
                "SELECT request_id, payload FROM commands"
            ).fetchall()
        except sqlite3.Error as error:
            raise OfflineReadError("The control commands could not be read.") from error
        payloads: dict[str, dict] = {}
        for request_id, raw_payload in payload_rows:
            try:
                payload = _json.loads(raw_payload)
            except (TypeError, ValueError) as error:
                raise OfflineReadError("A control command is malformed.") from error
            if not isinstance(payload, dict):
                raise OfflineReadError("A control command is malformed.")
            if not isinstance(request_id, str):
                raise OfflineReadError("A control command is malformed.")
            payloads[request_id] = payload
    finally:
        connection.close()
    return OfflineControlSnapshot(
        intake=intake,
        pending_command_id=pending,
        commands=commands,
        payloads=payloads,
        **extra,
    )


def _read_extra_state(state: dict) -> dict:
    """Validate the optional control-state columns; fail closed on malformed."""

    def _optional_int(name: str) -> int | None:
        value = state.get(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise OfflineReadError("The control state is malformed.")
        return value

    def _optional_str(name: str) -> str | None:
        value = state.get(name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise OfflineReadError("The control state is malformed.")
        return value

    requested = _optional_int("stop_after_requested")
    remaining = _optional_int("stop_after_remaining")
    active_attempt = _optional_str("stop_after_active_attempt_id")
    latest_id = _optional_str("latest_counted_attempt_id")
    latest_outcome = _optional_str("latest_counted_outcome")
    next_number_raw = state.get("next_issue_number")
    next_number: int | None = None
    if next_number_raw is not None:
        if isinstance(next_number_raw, bool) or not isinstance(
            next_number_raw, int
        ):
            raise OfflineReadError("The control state is malformed.")
        next_number = next_number_raw
    next_request = _optional_str("next_issue_request_id")
    if (next_number is None) != (next_request is None):
        raise OfflineReadError("The control state is malformed.")
    handoff_request = _optional_str("handoff_request_id")
    handoff_accepted = _optional_str("handoff_accepted_at")
    begun_raw = state.get("handoff_begun")
    begun = False
    if begun_raw is not None:
        if isinstance(begun_raw, bool) or not isinstance(begun_raw, int):
            raise OfflineReadError("The control state is malformed.")
        begun = bool(begun_raw)
    handoff_phase = _optional_str("handoff_phase")
    return {
        "stop_after_requested": requested,
        "stop_after_remaining": remaining,
        "stop_after_active_attempt_id": active_attempt,
        "latest_counted_attempt_id": latest_id,
        "latest_counted_outcome": latest_outcome,
        "next_issue_number": next_number,
        "next_issue_request_id": next_request,
        "handoff_request_id": handoff_request,
        "handoff_accepted_at": handoff_accepted,
        "handoff_begun": begun,
        "handoff_phase": handoff_phase,
    }


def _record_from_row(row: tuple) -> CommandRecord:
    request_id, sequence, kind, acknowledgement, detail, created_at, updated_at = row[:7]
    for field in (request_id, kind, detail, created_at, updated_at):
        if not isinstance(field, str):
            raise OfflineReadError("A control command is malformed.")
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise OfflineReadError("A control command is malformed.")
    try:
        ack = CommandAcknowledgement(acknowledgement)
    except (ValueError, TypeError) as error:
        raise OfflineReadError("A control command is malformed.") from error
    ordered = sequence
    return CommandRecord(
        request_id=request_id,
        sequence=ordered,
        kind=kind,
        acknowledgement=ack,
        detail=detail,
        created_at=created_at,
        updated_at=updated_at,
    )
