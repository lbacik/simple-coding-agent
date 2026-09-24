"""Durable operator-control command record owned by the live agent.

Only the running agent process opens this SQLite store; the ``agentctl``
CLI talks to the agent over a private Unix socket. Each accepted command
carries the CLI-supplied unique request ID and a monotonically increasing
durable sequence assigned at commit time. Retrying the same request ID with
an identical payload returns the stored acknowledgement without applying
the command twice; reusing an ID with different content is rejected.

This slice implements ``stop``, ``resume``, and ``stop --after N``. The
record already stores the command kind and canonical payload so later
commands (next-issue, handoff) can reuse the same ordering, retry, and
acknowledgement rules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
import json
import re
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


class ResumeBlockedError(ValueError):
    """Raised when ``resume`` is rejected by an unresolved recovery hold.

    The hold (startup reconciliation or retained work/finalization) leaves
    intake and the command record unchanged, and the message names the
    affected attempt and the reason.
    """


class StopAfterRejectedError(ValueError):
    """Raised when ``stop --after N`` is rejected without a control change.

    Nonpositive or noninteger counts and a stopped instance are rejected;
    neither the intake state nor an existing stop plan is altered.
    """


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


@dataclass(frozen=True)
class RecoveryHold:
    """An unresolved recovery state that blocks ``resume`` without claiming."""

    issue_number: int | None
    attempt_id: str | None
    branch: str | None
    phase: str | None
    reason: str


_STORE_FILENAME = "control.sqlite3"
_STOP_KIND = "stop"
_STOP_AFTER_KIND = "stop_after"
_RESUME_KIND = "resume"
# Stop-plan kinds a newer ``resume`` (or a newer stop plan) replaces. Later
# commands extend this tuple without changing the replacement rules.
_STOP_PLAN_KINDS = (_STOP_KIND, _STOP_AFTER_KIND)


def parse_stop_after(value: object) -> int:
    """Validate a ``stop --after N`` count; raise without side effects."""

    if isinstance(value, bool):
        raise StopAfterRejectedError(
            f"Invalid --after value {value!r}: must be a positive integer number"
            " of attempts. No control change was accepted."
        )
    if isinstance(value, int):
        count = value
    elif isinstance(value, str):
        if re.fullmatch(r"[+-]?\d+", value.strip()) is None:
            raise StopAfterRejectedError(
                f"Invalid --after value {value!r}: must be a positive integer number"
                " of attempts. No control change was accepted."
            )
        count = int(value.strip(), 10)
    else:
        raise StopAfterRejectedError(
            f"Invalid --after value {value!r}: must be a positive integer number"
            " of attempts. No control change was accepted."
        )
    if count <= 0:
        raise StopAfterRejectedError(
            f"Invalid --after value {value!r}: must be a positive integer number"
            " of attempts. No control change was accepted."
        )
    return count


def _attempts_noun(count: int) -> str:
    return "1 attempt" if count == 1 else f"{count} attempts"


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
                        superseded_after = self._supersede_plans_locked(
                            connection, (_STOP_AFTER_KIND,), request_id
                        )
                        if superseded_after:
                            # A stop-after plan waited for this same attempt:
                            # the plain stop takes over the wait without
                            # consuming a count.
                            pending = request_id
                            detail += (
                                "; stop plan " + ", ".join(superseded_after) + " superseded"
                            )
                            self._clear_stop_plan_locked(connection)
                    elif has_active_attempt:
                        acknowledgement = CommandAcknowledgement.ACCEPTED
                        detail = "finish active attempt, then stop intake"
                        superseded_after = self._supersede_plans_locked(
                            connection, (_STOP_AFTER_KIND,), request_id
                        )
                        if superseded_after:
                            detail += (
                                "; stop plan " + ", ".join(superseded_after) + " superseded"
                            )
                            self._clear_stop_plan_locked(connection)
                        intake = IntakeState.STOPPING
                        pending = request_id
                    else:
                        acknowledgement = CommandAcknowledgement.COMPLETED
                        detail = "intake stopped"
                        superseded_after = self._supersede_plans_locked(
                            connection, (_STOP_AFTER_KIND,), request_id
                        )
                        if superseded_after:
                            detail += (
                                "; stop plan " + ", ".join(superseded_after) + " superseded"
                            )
                            self._clear_stop_plan_locked(connection)
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

    def submit_resume(
        self,
        request_id: str,
        *,
        has_active_attempt: bool,
        recovery_hold: RecoveryHold | None = None,
    ) -> CommandRecord:
        """Durably record ``resume`` and permit later issue intake.

        From ``stopped`` the instance enters ``running``; while a stop plan
        is pending the plan is durably replaced (the earlier accepted stops
        become ``superseded`` naming this request) and the active attempt
        continues with its ordinary outcome. In ``running`` with no pending
        plan the command completes idempotently. A ``recovery_hold`` rejects
        the command without changing intake or the command record, and the
        rejection names the affected attempt and reason. Resume never
        requeues an issue: it only changes the intake state.
        """

        _check_request_id(request_id)
        with self.lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    existing = self._find_locked(connection, request_id)
                    if existing is not None:
                        self._ensure_payload_identical(
                            connection, existing, _RESUME_KIND, {}
                        )
                        connection.execute("ROLLBACK")
                        return existing
                    if recovery_hold is not None:
                        connection.execute("ROLLBACK")
                        raise ResumeBlockedError(_describe_hold(recovery_hold))
                    intake = self._intake_locked(connection)
                    pending = self._pending_locked(connection)
                    superseded_ids = self._accepted_stop_plans_locked(connection)
                    if (
                        intake is IntakeState.RUNNING
                        and pending is None
                        and not superseded_ids
                    ):
                        acknowledgement = CommandAcknowledgement.COMPLETED
                        detail = "intake already running"
                    else:
                        timestamp = self._timestamp()
                        if superseded_ids:
                            connection.execute(
                                "UPDATE commands SET acknowledgement = ?, detail = ?,"
                                " updated_at = ? WHERE kind IN (%s)"
                                " AND acknowledgement = ?"
                                % ",".join("?" * len(_STOP_PLAN_KINDS)),
                                (
                                    CommandAcknowledgement.SUPERSEDED.value,
                                    f"superseded by {request_id}",
                                    timestamp,
                                    *_STOP_PLAN_KINDS,
                                    CommandAcknowledgement.ACCEPTED.value,
                                ),
                            )
                        if pending is not None and pending not in superseded_ids:
                            superseded_ids = [pending, *superseded_ids]
                        if superseded_ids:
                            detail = (
                                "intake resumed; stop plan "
                                f"{', '.join(superseded_ids)} superseded"
                            )
                        else:
                            detail = "intake resumed"
                        if has_active_attempt:
                            detail += "; active attempt continues"
                        acknowledgement = CommandAcknowledgement.COMPLETED
                        intake = IntakeState.RUNNING
                        pending = None
                    record = self._insert_locked(
                        connection,
                        request_id,
                        _RESUME_KIND,
                        {},
                        acknowledgement,
                        detail,
                    )
                    self._set_intake_locked(connection, intake, pending)
                    self._clear_stop_plan_locked(connection)
                    connection.execute("COMMIT")
                    return record
            except (PayloadMismatchError, RequestIdError, ResumeBlockedError):
                raise
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be committed") from error

    def submit_stop_after(
        self,
        request_id: str,
        after: object,
        *,
        has_active_attempt: bool,
        active_attempt_id: str | None = None,
    ) -> CommandRecord:
        """Durably record ``stop --after N`` and persist its countdown plan.

        The attempt active at acceptance, if any, is count one; otherwise
        counting starts with the next attempt. With ``N == 1`` and an active
        attempt intake enters ``stopping`` at acceptance, otherwise it stays
        ``running`` with a visible pending plan until the final counted
        attempt is claimed. A new accepted plan replaces any pending stop
        plan and resets the count; ``stop``, ``resume``, and ``handoff now``
        replace it the same way without consuming a count. Rejections leave
        intake and any existing plan unchanged.
        """

        _check_request_id(request_id)
        count = parse_stop_after(after)
        with self.lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    existing = self._find_locked(connection, request_id)
                    if existing is not None:
                        self._ensure_payload_identical(
                            connection, existing, _STOP_AFTER_KIND, {"after": count}
                        )
                        connection.execute("ROLLBACK")
                        return existing
                    if has_active_attempt and (
                        not isinstance(active_attempt_id, str)
                        or not active_attempt_id.strip()
                    ):
                        connection.execute("ROLLBACK")
                        raise ControlStoreError(
                            "An active attempt must carry a stable attempt identity"
                        )
                    intake = self._intake_locked(connection)
                    if intake is IntakeState.STOPPED:
                        connection.execute("ROLLBACK")
                        raise StopAfterRejectedError(
                            "stop --after N is rejected while intake is stopped;"
                            " resume first. No control change was accepted."
                        )
                    superseded_ids = self._supersede_plans_locked(
                        connection, _STOP_PLAN_KINDS, request_id
                    )
                    if has_active_attempt:
                        assert active_attempt_id is not None
                        detail = (
                            f"stop after {_attempts_noun(count)};"
                            f" active attempt {active_attempt_id} is count one"
                        )
                    else:
                        detail = (
                            f"stop after {_attempts_noun(count)};"
                            " counting starts with the next attempt"
                        )
                    if superseded_ids:
                        detail += "; stop plan " + ", ".join(superseded_ids) + " superseded"
                    record = self._insert_locked(
                        connection,
                        request_id,
                        _STOP_AFTER_KIND,
                        {"after": count},
                        CommandAcknowledgement.ACCEPTED,
                        detail,
                    )
                    if has_active_attempt and count == 1:
                        intake = IntakeState.STOPPING
                    else:
                        intake = IntakeState.RUNNING
                    self._set_intake_locked(connection, intake, request_id)
                    self._set_stop_plan_locked(
                        connection,
                        requested=count,
                        remaining=count,
                        active_attempt_id=active_attempt_id if has_active_attempt else None,
                    )
                    connection.execute("COMMIT")
                    return record
            except (PayloadMismatchError, RequestIdError, StopAfterRejectedError):
                raise
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be committed") from error

    def submit_command(self, kind: str, request_id: str, payload: dict) -> CommandRecord:
        """Generic command entry used to detect request-ID reuse.

        Only ``stop``, ``resume``, and ``stop_after`` apply a control change;
        any other kind with a fresh ID is rejected without altering control
        state, while a repeated ID returns (or rejects on mismatch) the
        stored record.
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
        stays ``stopping`` until the attempt is durably finished. A pending
        ``stop --after`` plan is never completed here; it is counted down by
        :meth:`record_attempt_finalized` instead.
        """

        with self.lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    pending = self._pending_locked(connection)
                    if pending is None:
                        connection.execute("ROLLBACK")
                        return []
                    pending_record = self._find_locked(connection, pending)
                    if pending_record is None or pending_record.kind != _STOP_KIND:
                        connection.execute("ROLLBACK")
                        return []
                    return self._complete_plain_stops_in_txn(connection)
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be committed") from error

    # -- stop-after accounting ------------------------------------------------

    def record_attempt_finalized(self, attempt_id: str, outcome: str) -> list[CommandRecord]:
        """Count one fully finalized attempt against the pending stop plan.

        Every terminal attempt outcome counts exactly once: the decrement of
        the current plan, the ``stopped`` transition, and the completion of
        the stop command commit in one SQLite transaction keyed by the
        stable attempt ID. Replaying an already counted attempt returns an
        empty list without changing anything, so restart reconciliation can
        neither lose nor double-count it. A held finalization (unconfirmed
        publication, release, or cleanup) must not call this. Returns the
        stop-plan commands completed by this count, if any.
        """

        if not isinstance(attempt_id, str) or not attempt_id.strip():
            raise ControlStoreError("A finalized attempt must carry a stable attempt identity")
        if not isinstance(outcome, str) or not outcome.strip():
            raise ControlStoreError("A finalized attempt must carry a terminal outcome")
        with self.lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    already = connection.execute(
                        "SELECT 1 FROM counted_attempts WHERE attempt_id = ?",
                        (attempt_id,),
                    ).fetchone()
                    if already is not None:
                        connection.execute("ROLLBACK")
                        return []
                    row = connection.execute(
                        "SELECT intake, pending_command_id, stop_after_requested,"
                        " stop_after_remaining FROM control_state WHERE id = 1"
                    ).fetchone()
                    if row is None:  # pragma: no cover - initialized above
                        raise ControlStoreError("Control state is missing")
                    intake_value, pending, requested, remaining = row
                    if requested is not None and pending is None:
                        # Unreachable through the public API (the plan and its
                        # pending pointer are always written and cleared
                        # together); fail closed without touching the count.
                        connection.execute("ROLLBACK")
                        return []
                    if requested is not None:
                        try:
                            intake = IntakeState(intake_value)
                        except ValueError as error:
                            raise ControlStoreError("Control state is malformed") from error
                        timestamp = self._timestamp()
                        connection.execute(
                            "INSERT INTO counted_attempts"
                            " (attempt_id, plan_request_id, outcome, counted_at)"
                            " VALUES (?, ?, ?, ?)",
                            (attempt_id, pending, outcome, timestamp),
                        )
                        left = int(remaining) - 1
                        if left <= 0:
                            detail = (
                                f"stopped after {_attempts_noun(int(requested))};"
                                f" last counted attempt {attempt_id} ({outcome})"
                            )
                            if pending is not None:
                                connection.execute(
                                    "UPDATE commands SET acknowledgement = ?, detail = ?,"
                                    " updated_at = ? WHERE request_id = ?"
                                    " AND acknowledgement = ?",
                                    (
                                        CommandAcknowledgement.COMPLETED.value,
                                        detail,
                                        timestamp,
                                        pending,
                                        CommandAcknowledgement.ACCEPTED.value,
                                    ),
                                )
                                completed_row = connection.execute(
                                    "SELECT request_id, sequence, kind, acknowledgement,"
                                    " detail, created_at, updated_at FROM commands"
                                    " WHERE request_id = ?",
                                    (pending,),
                                ).fetchone()
                            else:  # pragma: no cover - plan always has a command
                                completed_row = None
                            connection.execute(
                                "UPDATE control_state SET intake = ?,"
                                " pending_command_id = NULL,"
                                " stop_after_requested = NULL,"
                                " stop_after_remaining = NULL,"
                                " stop_after_active_attempt_id = NULL,"
                                " latest_counted_attempt_id = ?,"
                                " latest_counted_outcome = ? WHERE id = 1",
                                (IntakeState.STOPPED.value, attempt_id, outcome),
                            )
                            connection.execute("COMMIT")
                            if completed_row is None:
                                return []
                            return [self._record_from_row(completed_row)]
                        connection.execute(
                            "UPDATE control_state SET stop_after_remaining = ?,"
                            " latest_counted_attempt_id = ?,"
                            " latest_counted_outcome = ? WHERE id = 1",
                            (left, attempt_id, outcome),
                        )
                        if intake is not IntakeState.STOPPED:
                            # The counted attempt just finished, so no attempt
                            # is active: intake runs until the final claim.
                            connection.execute(
                                "UPDATE control_state SET intake = ? WHERE id = 1",
                                (IntakeState.RUNNING.value,),
                            )
                        connection.execute("COMMIT")
                        return []
                    if pending is None:
                        connection.execute("ROLLBACK")
                        return []
                    pending_record = self._find_locked(connection, pending)
                    if pending_record is None or pending_record.kind != _STOP_KIND:
                        connection.execute("ROLLBACK")
                        return []
                    completed = self._complete_plain_stops_in_txn(connection)
                    return completed
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be committed") from error
        raise AssertionError("unreachable")  # pragma: no cover

    def _complete_plain_stops_in_txn(
        self, connection: sqlite3.Connection
    ) -> list[CommandRecord]:
        """Complete accepted plain stops; caller holds the lock and a transaction."""

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

    def enter_stopping_for_final_claim(self) -> bool:
        """Enter ``stopping`` when the final counted attempt is claimed.

        Returns True when the transition was applied. While more than the
        final attempt remains, or with no pending ``stop --after`` plan,
        intake stays ``running`` and an empty runnable queue preserves the
        remaining count untouched.
        """

        with self.lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    row = connection.execute(
                        "SELECT intake, pending_command_id, stop_after_requested,"
                        " stop_after_remaining FROM control_state WHERE id = 1"
                    ).fetchone()
                    if row is None:  # pragma: no cover - initialized above
                        raise ControlStoreError("Control state is missing")
                    intake_value, pending, requested, remaining = row
                    if (
                        pending is not None
                        and requested is not None
                        and int(remaining) == 1
                        and intake_value == IntakeState.RUNNING.value
                    ):
                        connection.execute(
                            "UPDATE control_state SET intake = ? WHERE id = 1",
                            (IntakeState.STOPPING.value,),
                        )
                        connection.execute("COMMIT")
                        return True
                    connection.execute("ROLLBACK")
                    return False
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be committed") from error

    def stop_plan_snapshot(self) -> dict | None:
        """Return the pending ``stop --after`` plan for status, if any."""

        with self.lock:
            try:
                with self._connect() as connection:
                    row = connection.execute(
                        "SELECT pending_command_id, stop_after_requested,"
                        " stop_after_remaining, stop_after_active_attempt_id,"
                        " latest_counted_attempt_id, latest_counted_outcome"
                        " FROM control_state WHERE id = 1"
                    ).fetchone()
            except sqlite3.Error as error:
                raise ControlStoreError("Control state could not be read") from error
        if row is None:  # pragma: no cover - initialized above
            raise ControlStoreError("Control state is missing")
        pending, requested, remaining, active_attempt_id, latest_id, latest_outcome = row
        if requested is None or pending is None:
            return None
        return {
            "request_id": pending,
            "kind": _STOP_AFTER_KIND,
            "requested": int(requested),
            "remaining": int(remaining),
            "includes_active_attempt": active_attempt_id is not None,
            "active_attempt_id": active_attempt_id,
            "latest_counted_attempt_id": latest_id,
            "latest_counted_outcome": latest_outcome,
        }

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
                for column_ddl in (
                    "stop_after_requested INTEGER NULL",
                    "stop_after_remaining INTEGER NULL",
                    "stop_after_active_attempt_id TEXT NULL",
                    "latest_counted_attempt_id TEXT NULL",
                    "latest_counted_outcome TEXT NULL",
                ):
                    column = column_ddl.split(" ", 1)[0]
                    known = {
                        row[1]
                        for row in connection.execute(
                            "PRAGMA table_info(control_state)"
                        ).fetchall()
                    }
                    if column not in known:
                        connection.execute(
                            f"ALTER TABLE control_state ADD COLUMN {column_ddl}"
                        )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS counted_attempts ("
                    " attempt_id TEXT PRIMARY KEY,"
                    " plan_request_id TEXT NOT NULL,"
                    " outcome TEXT NOT NULL,"
                    " counted_at TEXT NOT NULL"
                    ")"
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

    def _accepted_stop_plans_locked(
        self, connection: sqlite3.Connection
    ) -> list[str]:
        """Return the request IDs of accepted stop-plan commands, oldest first."""

        return self._accepted_plans_of_kinds_locked(connection, _STOP_PLAN_KINDS)

    def _accepted_plans_of_kinds_locked(
        self, connection: sqlite3.Connection, kinds: tuple[str, ...]
    ) -> list[str]:
        """Return accepted command IDs for the given stop-plan kinds, oldest first."""

        rows = connection.execute(
            "SELECT request_id FROM commands WHERE kind IN (%s)"
            " AND acknowledgement = ? ORDER BY sequence" % ",".join("?" * len(kinds)),
            (*kinds, CommandAcknowledgement.ACCEPTED.value),
        ).fetchall()
        return [row[0] for row in rows]

    def _supersede_plans_locked(
        self, connection: sqlite3.Connection, kinds: tuple[str, ...], replacement_id: str
    ) -> list[str]:
        """Mark accepted plans of the given kinds superseded; return their IDs."""

        superseded = self._accepted_plans_of_kinds_locked(connection, kinds)
        if superseded:
            connection.execute(
                "UPDATE commands SET acknowledgement = ?, detail = ?,"
                " updated_at = ? WHERE kind IN (%s) AND acknowledgement = ?"
                % ",".join("?" * len(kinds)),
                (
                    CommandAcknowledgement.SUPERSEDED.value,
                    f"superseded by {replacement_id}",
                    self._timestamp(),
                    *kinds,
                    CommandAcknowledgement.ACCEPTED.value,
                ),
            )
        return superseded

    def _set_stop_plan_locked(
        self,
        connection: sqlite3.Connection,
        *,
        requested: int,
        remaining: int,
        active_attempt_id: str | None,
    ) -> None:
        """Persist a fresh ``stop --after`` countdown, clearing past progress."""

        connection.execute(
            "UPDATE control_state SET stop_after_requested = ?,"
            " stop_after_remaining = ?, stop_after_active_attempt_id = ?,"
            " latest_counted_attempt_id = NULL, latest_counted_outcome = NULL"
            " WHERE id = 1",
            (requested, remaining, active_attempt_id),
        )

    def _clear_stop_plan_locked(self, connection: sqlite3.Connection) -> None:
        """Remove the pending ``stop --after`` countdown, if any."""

        connection.execute(
            "UPDATE control_state SET stop_after_requested = NULL,"
            " stop_after_remaining = NULL, stop_after_active_attempt_id = NULL"
            " WHERE id = 1"
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


def _describe_hold(hold: RecoveryHold) -> str:
    """Render a resume rejection naming the affected attempt and reason."""

    if hold.issue_number is not None:
        where = f"issue #{hold.issue_number}"
        if hold.attempt_id:
            where += f" (attempt {hold.attempt_id})"
        if hold.branch:
            where += f" on {hold.branch}"
        if hold.phase:
            where += f" (phase {hold.phase})"
        return (
            f"Resume is blocked by the retained attempt for {where}:"
            f" {hold.reason}. Resolve recovery before resuming intake;"
            " no control change was accepted."
        )
    return (
        f"Resume is blocked: {hold.reason}."
        " No control change was accepted."
    )


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
    stop_plan: dict | None = None,
) -> dict:
    """Build one consistent live status snapshot.

    Read-only: it never opens the database directly but renders records the
    caller already serialized under the control lock. ``stop_plan`` carries
    the pending ``stop --after`` countdown (requested and remaining count,
    active-attempt inclusion, and latest counted attempt and outcome).
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
        "stop_plan": dict(stop_plan) if stop_plan is not None else None,
        "commands": {record.request_id: command_to_json(record) for record in recent_commands()},
    }
