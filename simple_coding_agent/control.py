"""Durable operator-control command record owned by the live agent.

Only the running agent process opens this SQLite store; the ``agentctl``
CLI talks to the agent over a private Unix socket. Each accepted command
carries the CLI-supplied unique request ID and a monotonically increasing
durable sequence assigned at commit time. Retrying the same request ID with
an identical payload returns the stored acknowledgement without applying
the command twice; reusing an ID with different content is rejected.

This slice implements ``stop``, ``resume``, ``stop --after N``, and ``next issue``. The
record already stores the command kind and canonical payload so later
commands (handoff) can reuse the same ordering, retry, and
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


class NextIssueRejectedError(ValueError):
    """Raised when ``next issue`` is rejected without a control change.

    An unknown, invalid, ineligible, or unverifiable target leaves the
    intake state and any existing priority intact, and no command is
    recorded.
    """


class HandoffRejectedError(ValueError):
    """Raised when ``handoff now`` is rejected without a control change.

    Without an active attempt there is nothing to hand off, and once the
    model has begun the handoff skill a second request cannot undo it;
    neither case alters the intake state or an existing plan.
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


# Operator-handoff timing from the durable acceptance commit: the model must
# finish its local handoff work within 240 seconds, and the driving process
# gets at most a further 120 seconds to confirm publication (360 total).
# Restart never resets these bounds; they are recomputed from the original
# ``accepted_at`` on every read.
HANDOFF_MODEL_DEADLINE_SECONDS = 240
HANDOFF_PUBLICATION_ALLOWANCE_SECONDS = 120
HANDOFF_ABSOLUTE_DEADLINE_SECONDS = (
    HANDOFF_MODEL_DEADLINE_SECONDS + HANDOFF_PUBLICATION_ALLOWANCE_SECONDS
)
_STORE_FILENAME = "control.sqlite3"
_STOP_KIND = "stop"
_STOP_AFTER_KIND = "stop_after"
_RESUME_KIND = "resume"
_NEXT_ISSUE_KIND = "next_issue"
_HANDOFF_KIND = "handoff"
_RECOVERY_RETRY_KIND = "recovery_retry"
_RECOVERY_RELEASE_KIND = "recovery_release"
# Stop-plan kinds a newer ``resume`` (or a newer stop plan) replaces. Later
# commands extend this tuple without changing the replacement rules.
# ``next issue`` is intentionally absent: it neither replaces a stop plan
# nor is replaced by one.
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


def _invalid_issue_message(value: object) -> str:
    return (
        f"Invalid issue number {value!r}: must be a positive integer issue"
        " number. No control change was accepted."
    )


def parse_next_issue(value: object) -> int:
    """Validate a ``next issue`` target; raise without side effects."""

    if isinstance(value, bool):
        raise NextIssueRejectedError(_invalid_issue_message(value))
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        if re.fullmatch(r"[+-]?\d+", value.strip()) is None:
            raise NextIssueRejectedError(_invalid_issue_message(value))
        number = int(value.strip(), 10)
    else:
        raise NextIssueRejectedError(_invalid_issue_message(value))
    if number <= 0:
        raise NextIssueRejectedError(_invalid_issue_message(value))
    return number


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
                        if (
                            pending is not None
                            and self._is_tracked_handoff_locked(connection, pending)
                            and not self._handoff_begun_locked(connection)
                        ):
                            # A not-yet-begun handoff owns the wait: the plain
                            # stop replaces it and takes over the wait.
                            superseded_handoff = self._supersede_plans_locked(
                                connection, (_HANDOFF_KIND,), request_id
                            )
                            if superseded_handoff:
                                pending = request_id
                                detail += (
                                    "; stop plan "
                                    + ", ".join(superseded_handoff)
                                    + " superseded"
                                )
                                self._set_handoff_locked(
                                    connection, None, None, begun=False, phase=None
                                )
                    elif has_active_attempt:
                        acknowledgement = CommandAcknowledgement.ACCEPTED
                        detail = "finish active attempt, then stop intake"
                        supersede_kinds: tuple[str, ...] = (_STOP_AFTER_KIND,)
                        if not self._handoff_begun_locked(connection):
                            # A begun handoff already owns the model and cannot
                            # be replaced; this branch only runs while intake
                            # is running, where that cannot happen, but the
                            # guard keeps the replacement fail-closed.
                            supersede_kinds = (_STOP_AFTER_KIND, _HANDOFF_KIND)
                        superseded_after = self._supersede_plans_locked(
                            connection,
                            supersede_kinds,
                            request_id,
                        )
                        if superseded_after:
                            detail += (
                                "; stop plan " + ", ".join(superseded_after) + " superseded"
                            )
                            self._clear_stop_plan_locked(connection)
                            self._set_handoff_locked(
                                connection, None, None, begun=False, phase=None
                            )
                        intake = IntakeState.STOPPING
                        pending = request_id
                    else:
                        acknowledgement = CommandAcknowledgement.COMPLETED
                        detail = "intake stopped"
                        supersede_kinds = (_STOP_AFTER_KIND,)
                        if not self._handoff_begun_locked(connection):
                            supersede_kinds = (_STOP_AFTER_KIND, _HANDOFF_KIND)
                        superseded_after = self._supersede_plans_locked(
                            connection,
                            supersede_kinds,
                            request_id,
                        )
                        if superseded_after:
                            detail += (
                                "; stop plan " + ", ".join(superseded_after) + " superseded"
                            )
                            self._clear_stop_plan_locked(connection)
                            self._set_handoff_locked(
                                connection, None, None, begun=False, phase=None
                            )
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
                    if self._handoff_begun_locked(connection):
                        # The model already owns the handoff: a newer resume
                        # cannot undo it or its publication. Record the resume
                        # without touching the pending handoff or the intake
                        # state; it takes effect on later intake once the
                        # handoff finalizes and stops intake.
                        pending_handoff = self._pending_locked(connection)
                        detail = (
                            "handoff"
                            f" {pending_handoff} already begun; resume takes effect"
                            " after handoff finalization, handoff continues"
                        )
                        record = self._insert_locked(
                            connection,
                            request_id,
                            _RESUME_KIND,
                            {},
                            CommandAcknowledgement.COMPLETED,
                            detail,
                        )
                        connection.execute("COMMIT")
                        return record
                    intake = self._intake_locked(connection)
                    pending = self._pending_locked(connection)
                    superseded_ids = self._accepted_stop_plans_locked(connection)
                    superseded_handoff = self._supersede_plans_locked(
                        connection, (_HANDOFF_KIND,), request_id
                    )
                    if superseded_handoff:
                        self._set_handoff_locked(
                            connection, None, None, begun=False, phase=None
                        )
                    superseded_ids = [*superseded_handoff, *superseded_ids]
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
                    if self._handoff_begun_locked(connection):
                        connection.execute("ROLLBACK")
                        raise StopAfterRejectedError(
                            "stop --after N is rejected while a handoff owns the"
                            " model; the handoff cannot be replaced once begun."
                            " No control change was accepted."
                        )
                    superseded_ids = self._supersede_plans_locked(
                        connection,
                        (*_STOP_PLAN_KINDS, _HANDOFF_KIND),
                        request_id,
                    )
                    tracked = connection.execute(
                        "SELECT handoff_request_id FROM control_state WHERE id = 1"
                    ).fetchone()
                    if tracked is not None and tracked[0] in superseded_ids:
                        self._set_handoff_locked(
                            connection, None, None, begun=False, phase=None
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


    def submit_next_issue(self, request_id: str, issue: object) -> CommandRecord:
        """Durably record a one-shot ``next issue`` priority.

        The target must already have been verified eligible by the caller
        (open, unassigned, labelled ``ready-for-agent``, with a non-empty
        body and no open blocker); this method validates only the issue
        number itself. Acceptance never changes the intake state and never
        touches a pending stop plan. A newer accepted priority supersedes
        only the earlier priority: the older command becomes terminal
        ``superseded`` naming this request, and the new detail names the
        replaced command. Retrying the same request ID with the identical
        payload returns the stored record without applying the command
        twice; reuse with different content is rejected.
        """

        _check_request_id(request_id)
        number = parse_next_issue(issue)
        with self.lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    existing = self._find_locked(connection, request_id)
                    if existing is not None:
                        self._ensure_payload_identical(
                            connection, existing, _NEXT_ISSUE_KIND, {"issue": number}
                        )
                        connection.execute("ROLLBACK")
                        return existing
                    current_number, current_request = self._next_issue_locked(connection)
                    timestamp = self._timestamp()
                    if current_request is not None:
                        connection.execute(
                            "UPDATE commands SET acknowledgement = ?, detail = ?,"
                            " updated_at = ? WHERE request_id = ?"
                            " AND acknowledgement = ?",
                            (
                                CommandAcknowledgement.SUPERSEDED.value,
                                f"superseded by {request_id}",
                                timestamp,
                                current_request,
                                CommandAcknowledgement.ACCEPTED.value,
                            ),
                        )
                        detail = (
                            f"next issue #{number} prioritized for the next"
                            f" permitted claim; supersedes {current_request}"
                            f" (was #{current_number})"
                        )
                    else:
                        detail = (
                            f"next issue #{number} prioritized for the next"
                            " permitted claim"
                        )
                    record = self._insert_locked(
                        connection,
                        request_id,
                        _NEXT_ISSUE_KIND,
                        {"issue": number},
                        CommandAcknowledgement.ACCEPTED,
                        detail,
                    )
                    connection.execute(
                        "UPDATE control_state SET next_issue_number = ?,"
                        " next_issue_request_id = ? WHERE id = 1",
                        (number, request_id),
                    )
                    connection.execute("COMMIT")
                    return record
            except (PayloadMismatchError, RequestIdError, NextIssueRejectedError):
                raise
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be committed") from error

    def submit_handoff(self, request_id: str, *, has_active_attempt: bool) -> CommandRecord:
        """Durably record ``handoff now`` against the active attempt.

        Without an active attempt the command is rejected with no control
        change. With one, it replaces any pending stop plan, enters
        ``stopping``, and waits for the bounded model handoff and its
        publication: the request is delivered at the next safe model
        boundary (or its single fallback) and completes only after the
        handoff outcome, publication, issue release, and cleanup are
        durably finished. A second request while a begun handoff owns the
        model is rejected so the handoff cannot be undone mid-flight.
        """

        _check_request_id(request_id)
        with self.lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    existing = self._find_locked(connection, request_id)
                    if existing is not None:
                        self._ensure_payload_identical(
                            connection, existing, _HANDOFF_KIND, {}
                        )
                        connection.execute("ROLLBACK")
                        return existing
                    if not has_active_attempt:
                        connection.execute("ROLLBACK")
                        raise HandoffRejectedError(
                            "handoff now is rejected without an active attempt;"
                            " no control change was accepted."
                        )
                    if self._handoff_begun_locked(connection):
                        connection.execute("ROLLBACK")
                        raise HandoffRejectedError(
                            "handoff now is rejected: the model has already begun"
                            " the handoff skill and a second request cannot undo it."
                            " No control change was accepted."
                        )
                    superseded = self._supersede_plans_locked(
                        connection,
                        (_STOP_KIND, _STOP_AFTER_KIND, _HANDOFF_KIND),
                        request_id,
                    )
                    self._clear_stop_plan_locked(connection)
                    detail = (
                        "handoff requested; delivers at the next safe model"
                        " boundary or its single fallback"
                    )
                    if superseded:
                        detail += "; stop plan " + ", ".join(superseded) + " superseded"
                    record = self._insert_locked(
                        connection,
                        request_id,
                        _HANDOFF_KIND,
                        {},
                        CommandAcknowledgement.ACCEPTED,
                        detail,
                    )
                    self._set_intake_locked(connection, IntakeState.STOPPING, request_id)
                    self._set_handoff_locked(
                        connection, request_id, self._timestamp(), begun=False, phase="accepted"
                    )
                    connection.execute("COMMIT")
                    return record
            except (PayloadMismatchError, RequestIdError, HandoffRejectedError):
                raise
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be committed") from error

    def mark_handoff_delivering(self, request_id: str) -> bool:
        """Record that the handoff instruction reached the model.

        Delivery alone does not start the handoff: the command stays
        replaceable until the model begins the handoff skill. Returns True
        when the tracked request was updated.
        """

        with self.lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    if not self._is_tracked_handoff_locked(connection, request_id):
                        connection.execute("ROLLBACK")
                        return False
                    connection.execute(
                        "UPDATE control_state SET handoff_phase = ? WHERE id = 1",
                        ("delivering",),
                    )
                    connection.execute("COMMIT")
                    return True
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be committed") from error

    def mark_handoff_begun(self, request_id: str) -> bool:
        """Record that the model began the handoff skill for this request.

        From here a newer stop-plan command can no longer replace the
        handoff; it controls only subsequent intake. Returns True when the
        tracked request was updated.
        """

        with self.lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    if not self._is_tracked_handoff_locked(connection, request_id):
                        connection.execute("ROLLBACK")
                        return False
                    connection.execute(
                        "UPDATE control_state SET handoff_begun = 1,"
                        " handoff_phase = ? WHERE id = 1",
                        ("model_work",),
                    )
                    connection.execute("COMMIT")
                    return True
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be committed") from error

    def complete_handoff(self, request_id: str, detail: str) -> CommandRecord | None:
        """Complete a handoff after its publication, release, and cleanup.

        Intake enters ``stopped`` with no further claim permitted; accepted
        plain stops waiting alongside the handoff complete on the same
        boundary. Returns the terminal record, or ``None`` when the request
        is unknown.
        """

        if not isinstance(detail, str) or not detail.strip():
            raise ControlStoreError("A completed handoff must carry a detail")
        with self.lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    existing = self._find_locked(connection, request_id)
                    if existing is None:
                        connection.execute("ROLLBACK")
                        return None
                    if existing.acknowledgement is not CommandAcknowledgement.ACCEPTED:
                        connection.execute("ROLLBACK")
                        return existing
                    timestamp = self._timestamp()
                    connection.execute(
                        "UPDATE commands SET acknowledgement = ?, detail = ?,"
                        " updated_at = ? WHERE request_id = ?",
                        (
                            CommandAcknowledgement.COMPLETED.value,
                            detail.strip(),
                            timestamp,
                            request_id,
                        ),
                    )
                    pending = self._pending_locked(connection)
                    if pending == request_id:
                        self._set_intake_locked(connection, IntakeState.STOPPED, None)
                    if existing.kind == _HANDOFF_KIND:
                        connection.execute(
                            "UPDATE control_state SET handoff_phase = ? WHERE id = 1",
                            ("completed",),
                        )
                    if pending == request_id:
                        plain = connection.execute(
                            "SELECT request_id FROM commands WHERE kind = ?"
                            " AND acknowledgement = ? ORDER BY sequence",
                            (_STOP_KIND, CommandAcknowledgement.ACCEPTED.value),
                        ).fetchall()
                        if plain:
                            connection.execute(
                                "UPDATE commands SET acknowledgement = ?, detail = ?,"
                                " updated_at = ? WHERE kind = ? AND acknowledgement = ?",
                                (
                                    CommandAcknowledgement.COMPLETED.value,
                                    "intake stopped after handoff attempt",
                                    timestamp,
                                    _STOP_KIND,
                                    CommandAcknowledgement.ACCEPTED.value,
                                ),
                            )
                    row = connection.execute(
                        "SELECT request_id, sequence, kind, acknowledgement,"
                        " detail, created_at, updated_at FROM commands"
                        " WHERE request_id = ?",
                        (request_id,),
                    ).fetchone()
                    connection.execute("COMMIT")
                    if row is None:  # pragma: no cover - just updated above
                        raise ControlStoreError("Control command is missing")
                    return self._record_from_row(row)
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be committed") from error

    def fail_handoff(self, request_id: str, reason: str) -> CommandRecord | None:
        """Mark a handoff ``not fulfilled`` with its reason and actual outcome.

        Intake remains stopped and the retained branch, checkpoint, and
        working tree stay preserved for recovery. Returns the terminal
        record, or ``None`` when the request is unknown.
        """

        if not isinstance(reason, str) or not reason.strip():
            raise ControlStoreError("A not-fulfilled handoff must carry a reason")
        with self.lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    existing = self._find_locked(connection, request_id)
                    if existing is None:
                        connection.execute("ROLLBACK")
                        return None
                    if existing.acknowledgement is not CommandAcknowledgement.ACCEPTED:
                        connection.execute("ROLLBACK")
                        return existing
                    timestamp = self._timestamp()
                    connection.execute(
                        "UPDATE commands SET acknowledgement = ?, detail = ?,"
                        " updated_at = ? WHERE request_id = ?",
                        (
                            CommandAcknowledgement.NOT_FULFILLED.value,
                            reason.strip(),
                            timestamp,
                            request_id,
                        ),
                    )
                    pending = self._pending_locked(connection)
                    if pending == request_id:
                        self._set_intake_locked(connection, IntakeState.STOPPED, None)
                    if existing.kind == _HANDOFF_KIND:
                        connection.execute(
                            "UPDATE control_state SET handoff_phase = ? WHERE id = 1",
                            ("not_fulfilled",),
                        )
                    if pending == request_id:
                        plain = connection.execute(
                            "SELECT request_id FROM commands WHERE kind = ?"
                            " AND acknowledgement = ? ORDER BY sequence",
                            (_STOP_KIND, CommandAcknowledgement.ACCEPTED.value),
                        ).fetchall()
                        if plain:
                            connection.execute(
                                "UPDATE commands SET acknowledgement = ?, detail = ?,"
                                " updated_at = ? WHERE kind = ? AND acknowledgement = ?",
                                (
                                    CommandAcknowledgement.COMPLETED.value,
                                    "intake stopped after handoff attempt",
                                    timestamp,
                                    _STOP_KIND,
                                    CommandAcknowledgement.ACCEPTED.value,
                                ),
                            )
                    row = connection.execute(
                        "SELECT request_id, sequence, kind, acknowledgement,"
                        " detail, created_at, updated_at FROM commands"
                        " WHERE request_id = ?",
                        (request_id,),
                    ).fetchone()
                    connection.execute("COMMIT")
                    if row is None:  # pragma: no cover - just updated above
                        raise ControlStoreError("Control command is missing")
                    return self._record_from_row(row)
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be committed") from error

    def handoff_snapshot(self) -> dict | None:
        """Return the tracked handoff request for status, if any."""

        with self.lock:
            try:
                with self._connect() as connection:
                    row = connection.execute(
                        "SELECT handoff_request_id, handoff_accepted_at,"
                        " handoff_begun, handoff_phase FROM control_state"
                        " WHERE id = 1"
                    ).fetchone()
            except sqlite3.Error as error:
                raise ControlStoreError("Control state could not be read") from error
        if row is None:  # pragma: no cover - initialized above
            raise ControlStoreError("Control state is missing")
        request_id, accepted_at, begun, phase = row
        if request_id is None:
            return None
        record = self.get_command(request_id)
        deadlines = handoff_deadlines(accepted_at)
        return {
            "request_id": request_id,
            "accepted_at": accepted_at,
            "begun": bool(begun),
            "phase": phase or "accepted",
            "acknowledgement": record.acknowledgement.value if record is not None else None,
            "detail": record.detail if record is not None else None,
            "model_deadline_at": deadlines.get("model_deadline_at"),
            "publication_deadline_at": deadlines.get("publication_deadline_at"),
        }

    def next_issue_snapshot(self) -> dict | None:
        """Return the pending one-shot priority for status, if any."""

        with self.lock:
            try:
                with self._connect() as connection:
                    number, request_id = self._next_issue_locked(connection)
            except sqlite3.Error as error:
                raise ControlStoreError("Control state could not be read") from error
        if request_id is None or number is None:
            return None
        return {"request_id": request_id, "issue_number": int(number)}

    def complete_next_issue(self, issue_number: int) -> CommandRecord | None:
        """Consume the pending priority after its successful assignment.

        Marks the pending ``next issue`` command terminal ``completed`` and
        clears the one-shot priority. The stored target must match the
        claimed issue; a mismatch leaves the priority intact and returns
        ``None`` so a confused claim is never reported as completed.
        """

        with self.lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    stored_number, stored_request = self._next_issue_locked(connection)
                    if stored_request is None or stored_number is None:
                        connection.execute("ROLLBACK")
                        return None
                    if int(stored_number) != int(issue_number):
                        connection.execute("ROLLBACK")
                        return None
                    timestamp = self._timestamp()
                    connection.execute(
                        "UPDATE commands SET acknowledgement = ?, detail = ?,"
                        " updated_at = ? WHERE request_id = ?"
                        " AND acknowledgement = ?",
                        (
                            CommandAcknowledgement.COMPLETED.value,
                            f"claimed issue #{issue_number} as the prioritized next issue",
                            timestamp,
                            stored_request,
                            CommandAcknowledgement.ACCEPTED.value,
                        ),
                    )
                    row = connection.execute(
                        "SELECT request_id, sequence, kind, acknowledgement,"
                        " detail, created_at, updated_at FROM commands"
                        " WHERE request_id = ?",
                        (stored_request,),
                    ).fetchone()
                    connection.execute(
                        "UPDATE control_state SET next_issue_number = NULL,"
                        " next_issue_request_id = NULL WHERE id = 1"
                    )
                    connection.execute("COMMIT")
                    if row is None:  # pragma: no cover - just updated above
                        raise ControlStoreError("Control command is missing")
                    return self._record_from_row(row)
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be committed") from error

    def fail_next_issue(self, reason: str) -> CommandRecord | None:
        """Clear the pending priority as terminal ``not fulfilled``.

        Records the observed loss-of-eligibility reason so status and
        request-ID lookup expose it; the caller falls back to the ordinary
        FIFO queue in the same intake cycle. Returns the terminal record,
        or ``None`` when no priority was pending.
        """

        if not isinstance(reason, str) or not reason.strip():
            raise ControlStoreError("A not-fulfilled priority must carry a reason")
        with self.lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    stored_number, stored_request = self._next_issue_locked(connection)
                    if stored_request is None or stored_number is None:
                        connection.execute("ROLLBACK")
                        return None
                    timestamp = self._timestamp()
                    connection.execute(
                        "UPDATE commands SET acknowledgement = ?, detail = ?,"
                        " updated_at = ? WHERE request_id = ?"
                        " AND acknowledgement = ?",
                        (
                            CommandAcknowledgement.NOT_FULFILLED.value,
                            f"next issue #{stored_number} not fulfilled:"
                            f" {reason.strip()}",
                            timestamp,
                            stored_request,
                            CommandAcknowledgement.ACCEPTED.value,
                        ),
                    )
                    row = connection.execute(
                        "SELECT request_id, sequence, kind, acknowledgement,"
                        " detail, created_at, updated_at FROM commands"
                        " WHERE request_id = ?",
                        (stored_request,),
                    ).fetchone()
                    connection.execute(
                        "UPDATE control_state SET next_issue_number = NULL,"
                        " next_issue_request_id = NULL WHERE id = 1"
                    )
                    connection.execute("COMMIT")
                    if row is None:  # pragma: no cover - just updated above
                        raise ControlStoreError("Control command is missing")
                    return self._record_from_row(row)
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be committed") from error

    # -- recovery ---------------------------------------------------------

    def submit_recovery_retry(self, request_id: str, attempt_id: str) -> CommandRecord:
        """Durably record ``recovery retry`` as accepted for one attempt identity.

        The command itself never changes intake; the lifecycle applies the
        safe finalization after acceptance and then marks this record
        ``completed`` (hold cleared) or ``not fulfilled`` (hold remains).
        Retrying the same request ID with the identical attempt returns the
        stored record; reuse with different content is rejected.
        """

        from simple_coding_agent.recovery import parse_attempt_id

        _check_request_id(request_id)
        parsed = parse_attempt_id(attempt_id)
        with self.lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    existing = self._find_locked(connection, request_id)
                    if existing is not None:
                        self._ensure_payload_identical(
                            connection,
                            existing,
                            _RECOVERY_RETRY_KIND,
                            {"attempt_id": parsed},
                        )
                        connection.execute("ROLLBACK")
                        return existing
                    record = self._insert_locked(
                        connection,
                        request_id,
                        _RECOVERY_RETRY_KIND,
                        {"attempt_id": parsed},
                        CommandAcknowledgement.ACCEPTED,
                        f"recovery retry for attempt {parsed} accepted",
                    )
                    connection.execute("COMMIT")
                    return record
            except (PayloadMismatchError, RequestIdError):
                raise
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be committed") from error

    def submit_recovery_release(
        self, request_id: str, attempt_id: str, saved_at: str
    ) -> CommandRecord:
        """Durably record ``recovery release`` as accepted for one attempt.

        The operator-provided ``saved-at`` reference is stored in the payload
        and surfaced in the detail so it remains visible in the command
        record. Like retry, the lifecycle completes or fails the record after
        attempting the abandonment path.
        """

        from simple_coding_agent.recovery import parse_attempt_id, parse_saved_at

        _check_request_id(request_id)
        parsed_attempt = parse_attempt_id(attempt_id)
        parsed_saved = parse_saved_at(saved_at)
        payload = {"attempt_id": parsed_attempt, "saved_at": parsed_saved}
        with self.lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    existing = self._find_locked(connection, request_id)
                    if existing is not None:
                        self._ensure_payload_identical(
                            connection, existing, _RECOVERY_RELEASE_KIND, payload
                        )
                        connection.execute("ROLLBACK")
                        return existing
                    record = self._insert_locked(
                        connection,
                        request_id,
                        _RECOVERY_RELEASE_KIND,
                        payload,
                        CommandAcknowledgement.ACCEPTED,
                        f"recovery release for attempt {parsed_attempt} accepted;"
                        f" retained work secured at {parsed_saved}",
                    )
                    connection.execute("COMMIT")
                    return record
            except (PayloadMismatchError, RequestIdError):
                raise
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be committed") from error

    def complete_recovery_command(self, request_id: str, detail: str) -> CommandRecord | None:
        """Mark an accepted recovery command ``completed`` with its outcome detail."""

        if not isinstance(detail, str) or not detail.strip():
            raise ControlStoreError("A completed recovery command must carry a detail")
        with self.lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    existing = self._find_locked(connection, request_id)
                    if existing is None:
                        connection.execute("ROLLBACK")
                        return None
                    if existing.acknowledgement is not CommandAcknowledgement.ACCEPTED:
                        connection.execute("ROLLBACK")
                        return existing
                    timestamp = self._timestamp()
                    connection.execute(
                        "UPDATE commands SET acknowledgement = ?, detail = ?,"
                        " updated_at = ? WHERE request_id = ?",
                        (
                            CommandAcknowledgement.COMPLETED.value,
                            detail.strip(),
                            timestamp,
                            request_id,
                        ),
                    )
                    connection.execute("COMMIT")
                    return self._find_locked(self._connect(), request_id)
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be committed") from error

    def fail_recovery_command(self, request_id: str, detail: str) -> CommandRecord | None:
        """Mark an accepted recovery command ``not fulfilled`` with the hold reason."""

        if not isinstance(detail, str) or not detail.strip():
            raise ControlStoreError("A not-fulfilled recovery command must carry a reason")
        with self.lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    existing = self._find_locked(connection, request_id)
                    if existing is None:
                        connection.execute("ROLLBACK")
                        return None
                    if existing.acknowledgement is not CommandAcknowledgement.ACCEPTED:
                        connection.execute("ROLLBACK")
                        return existing
                    timestamp = self._timestamp()
                    connection.execute(
                        "UPDATE commands SET acknowledgement = ?, detail = ?,"
                        " updated_at = ? WHERE request_id = ?",
                        (
                            CommandAcknowledgement.NOT_FULFILLED.value,
                            detail.strip(),
                            timestamp,
                            request_id,
                        ),
                    )
                    connection.execute("COMMIT")
                    return self._find_locked(self._connect(), request_id)
            except sqlite3.Error as error:
                raise ControlStoreError("Control command could not be committed") from error

    def submit_command(self, kind: str, request_id: str, payload: dict) -> CommandRecord:
        """Generic command entry used to detect request-ID reuse.

        Only ``stop``, ``resume``, and ``stop_after`` apply a control change
        through this entry point (``next issue`` has its own
        ``submit_next_issue``);
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
                    "next_issue_number INTEGER NULL",
                    "next_issue_request_id TEXT NULL",
                    "handoff_request_id TEXT NULL",
                    "handoff_accepted_at TEXT NULL",
                    "handoff_begun INTEGER NULL",
                    "handoff_phase TEXT NULL",
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

    def _set_handoff_locked(
        self,
        connection: sqlite3.Connection,
        request_id: str | None,
        accepted_at: str | None,
        *,
        begun: bool,
        phase: str | None,
    ) -> None:
        """Track (or clear) the handoff request owning the active attempt."""

        connection.execute(
            "UPDATE control_state SET handoff_request_id = ?,"
            " handoff_accepted_at = ?, handoff_begun = ?, handoff_phase = ?"
            " WHERE id = 1",
            (request_id, accepted_at, 1 if begun else 0, phase),
        )

    def _handoff_begun_locked(self, connection: sqlite3.Connection) -> bool:
        """Whether the tracked handoff already owns the model (not replaceable)."""

        try:
            row = connection.execute(
                "SELECT handoff_request_id, handoff_begun FROM control_state WHERE id = 1"
            ).fetchone()
        except sqlite3.Error as error:
            raise ControlStoreError("Control state could not be read") from error
        if row is None:  # pragma: no cover - initialized above
            raise ControlStoreError("Control state is missing")
        request_id, begun = row
        if request_id is None:
            return False
        record = self._find_locked(connection, request_id)
        if record is None or record.acknowledgement is not CommandAcknowledgement.ACCEPTED:
            return False
        return bool(begun)

    def _is_tracked_handoff_locked(
        self, connection: sqlite3.Connection, request_id: str
    ) -> bool:
        """Whether ``request_id`` is the tracked, still-accepted handoff."""

        try:
            row = connection.execute(
                "SELECT handoff_request_id FROM control_state WHERE id = 1"
            ).fetchone()
        except sqlite3.Error as error:
            raise ControlStoreError("Control state could not be read") from error
        if row is None:  # pragma: no cover - initialized above
            raise ControlStoreError("Control state is missing")
        if row[0] != request_id:
            return False
        record = self._find_locked(connection, request_id)
        return record is not None and record.acknowledgement is CommandAcknowledgement.ACCEPTED

    def _next_issue_locked(
        self, connection: sqlite3.Connection
    ) -> tuple[int | None, str | None]:
        """Return the pending priority ``(issue_number, request_id)``, if any."""

        try:
            row = connection.execute(
                "SELECT next_issue_number, next_issue_request_id"
                " FROM control_state WHERE id = 1"
            ).fetchone()
        except sqlite3.Error as error:
            raise ControlStoreError("Control state could not be read") from error
        if row is None:  # pragma: no cover - initialized above
            raise ControlStoreError("Control state is missing")
        number, request_id = row
        if number is None or request_id is None:
            return (None, None)
        if not isinstance(request_id, str):
            raise ControlStoreError("Control state is malformed")
        try:
            parsed = int(number)
        except (TypeError, ValueError) as error:
            raise ControlStoreError("Control state is malformed") from error
        return (parsed, request_id)

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


def handoff_deadlines(accepted_at: object) -> dict[str, str | None]:
    """Compute the original handoff deadlines from the acceptance timestamp.

    Returns ``model_deadline_at`` (acceptance + 240s) and
    ``publication_deadline_at`` (acceptance + 360s) as ``Z``-suffixed ISO
    strings, or ``None`` values when the acceptance time cannot be parsed.
    The bounds always derive from the original durable commit, so a process
    restart can neither extend nor renew them.
    """

    from datetime import timedelta

    parsed = _parse_control_timestamp(accepted_at)
    if parsed is None:
        return {"model_deadline_at": None, "publication_deadline_at": None}
    return {
        "model_deadline_at": _format_control_timestamp(
            parsed + timedelta(seconds=HANDOFF_MODEL_DEADLINE_SECONDS)
        ),
        "publication_deadline_at": _format_control_timestamp(
            parsed + timedelta(seconds=HANDOFF_ABSOLUTE_DEADLINE_SECONDS)
        ),
    }


def _parse_control_timestamp(value: object) -> datetime | None:
    """Parse a durable ``Z``-suffixed ISO timestamp; ``None`` when unreadable."""

    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def _format_control_timestamp(value: datetime) -> str:
    """Render a timestamp the way the control store records them."""

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
    next_issue: dict | None = None,
    stop_plan: dict | None = None,
    recovery: dict | None = None,
    handoff: dict | None = None,
) -> dict:
    """Build one consistent live status snapshot.

    Read-only: it never opens the database directly but renders records the
    caller already serialized under the control lock. ``next_issue`` carries
    the pending one-shot priority (``request_id`` and ``issue_number``), and
    ``stop_plan`` carries the pending ``stop --after`` countdown (requested
    and remaining count, active-attempt inclusion, and latest counted
    attempt and outcome). ``recovery`` carries the retained-attempt hold
    (attempt identity, phase, outcome, confirmed publication progress,
    branch/workspace/checkpoint, hold reason, and the next operator action)
    or ``None`` when no hold blocks intake. ``handoff`` carries the tracked
    operator handoff (request ID, acceptance time, skill-begun flag, phase,
    acknowledgement, and the original model/publication deadlines recomputed
    from acceptance) or ``None`` when no handoff owns the attempt.
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
        "pending_next_issue": dict(next_issue) if next_issue is not None else None,
        "stop_plan": dict(stop_plan) if stop_plan is not None else None,
        "recovery": dict(recovery) if recovery is not None else None,
        "handoff": dict(handoff) if handoff is not None else None,
        "commands": {record.request_id: command_to_json(record) for record in recent_commands()},
    }
