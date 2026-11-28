"""Durable completion boundary for one implementation attempt."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
import json
import os
from pathlib import Path
import tempfile

from simple_coding_agent.completion import AttemptOutcome


@dataclass(frozen=True)
class AttemptCompletion:
    """One fully finalized attempt, recorded exactly once."""

    attempt_id: str
    issue_number: int
    branch: str
    outcome: AttemptOutcome
    completed_at: str


class CompletionStoreError(RuntimeError):
    """Raised when the durable completion record cannot be used safely."""


class AttemptCompletionStore:
    """Atomically record fully finalized attempts keyed by stable attempt ID.

    The store is the durable completion boundary: an attempt ID is present
    only after its outcome, confirmed publication, issue release, and local
    cleanup are all durable. A leftover checkpoint whose ID is already
    recorded means the process crashed between finalization and checkpoint
    removal, so recovery must only remove the checkpoint without replaying
    any side effect or accounting.
    """

    def __init__(self, data_dir: Path, *, clock: Callable[[], datetime] | None = None) -> None:
        self._path = data_dir / "state" / "completed_attempts.json"
        self._clock = clock or (lambda: datetime.now(UTC))

    def is_finalized(self, attempt_id: str) -> bool:
        """Return True when this attempt ID already completed its boundary."""

        return attempt_id in self._read()

    def record(
        self, *, attempt_id: str, issue_number: int, branch: str, outcome: AttemptOutcome
    ) -> bool:
        """Record one finalized attempt; return False when it was already recorded."""

        records = self._read()
        if attempt_id in records:
            return False
        if isinstance(issue_number, bool) or not isinstance(issue_number, int) or issue_number <= 0:
            raise CompletionStoreError("Issue number must be a positive integer")
        if not isinstance(branch, str) or not branch.strip():
            raise CompletionStoreError("Branch must be a non-empty string")
        records[attempt_id] = AttemptCompletion(
            attempt_id=attempt_id,
            issue_number=issue_number,
            branch=branch,
            outcome=outcome,
            completed_at=_timestamp(self._clock()),
        )
        self._write(records)
        return True

    def read_all(self) -> dict[str, AttemptCompletion]:
        """Return every finalized attempt keyed by attempt ID."""

        return self._read()

    def _read(self) -> dict[str, AttemptCompletion]:
        try:
            raw = self._path.read_text()
        except FileNotFoundError:
            return {}
        except OSError as error:
            raise CompletionStoreError("Completion record could not be read") from error
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise CompletionStoreError("Completion record is malformed") from error
        if not isinstance(value, dict):
            raise CompletionStoreError("Completion record is malformed")
        records: dict[str, AttemptCompletion] = {}
        for attempt_id, entry in value.items():
            try:
                records[attempt_id] = _completion_from_json(attempt_id, entry)
            except (TypeError, ValueError) as error:
                raise CompletionStoreError("Completion record is malformed") from error
        return records

    def _write(self, records: dict[str, AttemptCompletion]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=".completed-",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary = Path(file.name)
                json.dump(
                    {
                        attempt_id: {
                            "issue_number": completion.issue_number,
                            "branch": completion.branch,
                            "outcome": completion.outcome.value,
                            "completed_at": completion.completed_at,
                        }
                        for attempt_id, completion in sorted(records.items())
                    },
                    file,
                    sort_keys=True,
                )
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self._path)
        except OSError as error:
            raise CompletionStoreError("Completion record could not be written") from error
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def _completion_from_json(attempt_id: object, value: object) -> AttemptCompletion:
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("invalid attempt id")
    if not isinstance(value, dict) or set(value) != {
        "issue_number",
        "branch",
        "outcome",
        "completed_at",
    }:
        raise ValueError("invalid completion shape")
    issue_number = value["issue_number"]
    branch = value["branch"]
    outcome = value["outcome"]
    completed_at = value["completed_at"]
    if isinstance(issue_number, bool) or not isinstance(issue_number, int) or issue_number <= 0:
        raise ValueError("invalid issue number")
    if not isinstance(branch, str) or not branch.strip():
        raise ValueError("invalid branch")
    if not isinstance(completed_at, str) or not completed_at:
        raise ValueError("invalid timestamp")
    return AttemptCompletion(
        attempt_id=attempt_id,
        issue_number=issue_number,
        branch=branch,
        outcome=AttemptOutcome(outcome),
        completed_at=completed_at,
    )


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise CompletionStoreError("Completion clock must include a timezone")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


class FinalizeStatus(StrEnum):
    """Whether the attempt crossed the durable completion boundary."""

    FINALIZED = "finalized"
    HELD = "held"


class FinalizeReason(StrEnum):
    """Stable machine-readable reason a finalization held.

    The human-readable detail rides alongside in ``FinalizeResult.detail``;
    match on the reason, show the detail.
    """

    PUBLICATION_UNCONFIRMED = "publication_unconfirmed"
    HANDOFF_HOLD = "handoff_hold"
    RELEASE_UNCONFIRMED = "release_unconfirmed"
    CLEANUP_UNCONFIRMED = "cleanup_unconfirmed"


@dataclass(frozen=True)
class FinalizeRequest:
    """Everything the boundary needs that callers already have.

    Callers differ only in how they arrived at the outcome (live model run,
    startup republication, recovery retry, recovery release); the boundary
    owns everything after that. ``checkpoint_branch`` is the durable branch
    identity; ``prepared_branch`` is ``None`` once the tree is cleaned up.
    """

    issue_number: int
    attempt_id: str
    checkpoint_branch: str
    prepared_branch: str | None
    outcome: AttemptOutcome
    comment_posted: bool
    has_prepared: bool
    publication_started_at: datetime | None = None


@dataclass(frozen=True)
class FinalizeResult:
    """The boundary's verdict: finalized, or held with a reason."""

    status: FinalizeStatus
    reason: FinalizeReason | None
    detail: str
    released: bool
    cleanup_ok: bool
    handoff: bool | None = None
    idempotent: bool = False
    failed_step: str | None = None

    @property
    def finalized(self) -> bool:
        """True when the attempt crossed the durable completion boundary."""

        return self.status is FinalizeStatus.FINALIZED


def resolve_finalize_branch(
    *, prepared_branch: str | None, checkpoint_branch: str
) -> str:
    """Resolve the branch the ledger commit records.

    The prepared branch wins while the tree is still on it; once cleanup
    returned the tree to the base, the checkpoint's durable identity wins.
    Reconstructing ``agent/issue-N`` from the issue number is never correct
    here: it drops the restoration suffixes a retained checkpoint carries.
    """

    if prepared_branch:
        return prepared_branch
    return checkpoint_branch


_HELD_DETAIL = (
    "Publication, release, or cleanup is unconfirmed, or a failed"
    " operator handoff owns this attempt; the branch, checkpoint, and"
    " working tree are preserved for recovery."
)


def finalize_attempt(
    request: FinalizeRequest,
    *,
    is_finalized: Callable[[str], bool],
    check_hold: Callable[..., bool],
    release: Callable[[], None],
    cleanup: Callable[[], None],
    commit: Callable[[str], None],
    resolve_handoff: Callable[[], bool | None],
    on_held: Callable[[FinalizeReason | None, str], None] | None = None,
    on_dirty: Callable[[Exception], None] | None = None,
) -> FinalizeResult:
    """Run the durable completion boundary in its one fixed order.

    The order is hold check, release, workspace cleanup, second hold check
    (``check_allowance=False``), ledger commit, handoff resolution. The
    second check runs only when publication, release, and cleanup all went
    durable, mirroring the pre-commit recheck after a confirmed publication.

    Release, cleanup, and commit failures propagate to the caller after the
    held outcome is reported (``try/finally`` semantics), except a dirty
    workspace, which holds via ``on_dirty`` instead of raising. A ledger
    that already holds the attempt short-circuits to finalized without
    replaying any side effect.
    """

    if is_finalized(request.attempt_id):
        return FinalizeResult(
            status=FinalizeStatus.FINALIZED,
            reason=None,
            detail="attempt already finalized; no side effect replayed",
            released=False,
            cleanup_ok=False,
            handoff=None,
            idempotent=True,
        )
    hold = request.comment_posted and bool(
        check_hold(prepared=request.has_prepared, check_allowance=True)
    )
    released = False
    release_error: BaseException | None = None
    try:
        if request.comment_posted and not hold:
            release()
            released = True
    except BaseException as error:
        # The held outcome is still reported below (cleanup runs first);
        # the failure then propagates exactly as the inline boundary did.
        release_error = error
    cleanup_ok = not request.has_prepared
    cleanup_error: BaseException | None = None
    if request.has_prepared and not hold:
        try:
            cleanup()
            cleanup_ok = True
        except BaseException as error:
            from simple_coding_agent.git_workspace import DirtyWorkspaceError

            if isinstance(error, DirtyWorkspaceError):
                if on_dirty is not None:
                    on_dirty(error)
            else:
                cleanup_error = error
    if request.comment_posted and released and cleanup_ok and not hold:
        hold = bool(check_hold(prepared=False, check_allowance=False))
    if request.comment_posted and released and cleanup_ok and not hold:
        commit(
            resolve_finalize_branch(
                prepared_branch=request.prepared_branch,
                checkpoint_branch=request.checkpoint_branch,
            )
        )
        handoff = resolve_handoff()
        return FinalizeResult(
            status=FinalizeStatus.FINALIZED,
            reason=None,
            detail="attempt finalized",
            released=True,
            cleanup_ok=True,
            handoff=handoff,
        )
    reason = _held_reason(request, hold, released, cleanup_ok)
    _report_hold(on_held, reason)
    held = FinalizeResult(
        status=FinalizeStatus.HELD,
        reason=reason,
        detail=_HELD_DETAIL,
        released=released,
        cleanup_ok=cleanup_ok,
        handoff=None,
    )
    if cleanup_error is not None:
        raise cleanup_error
    if release_error is not None:
        raise release_error
    return held


def commit_recovered_attempt(
    *,
    attempt_id: str,
    branch: str,
    is_finalized: Callable[[str], bool],
    release: Callable[[], None],
    cleanup: Callable[[], None],
    commit: Callable[[str], None],
    resolve_handoff: Callable[[], bool | None],
    on_held: Callable[[FinalizeReason | None, str], None] | None = None,
    on_dirty: Callable[[Exception], None] | None = None,
) -> FinalizeResult:
    """Run the recovery tail (release, cleanup, commit, handoff) in order.

    Recovery republication already confirmed the outcome, so unlike
    :func:`finalize_attempt` there is no handoff hold check here — the
    handoff is still resolved after the commit, at most once. Failures
    never propagate: they are captured into a held result (with the
    failing step named) so operator commands can report them and keep
    the hold.
    """
    if is_finalized(attempt_id):
        return FinalizeResult(
            status=FinalizeStatus.FINALIZED,
            reason=None,
            detail="attempt already finalized; no side effect replayed",
            released=False,
            cleanup_ok=False,
            handoff=None,
            idempotent=True,
        )
    try:
        release()
    except Exception as error:
        detail = f"issue release failed ({type(error).__name__}: {error})"
        _report_hold(on_held, FinalizeReason.RELEASE_UNCONFIRMED)
        return FinalizeResult(
            status=FinalizeStatus.HELD,
            reason=FinalizeReason.RELEASE_UNCONFIRMED,
            detail=detail,
            released=False,
            cleanup_ok=False,
            handoff=None,
            failed_step="release",
        )
    try:
        cleanup()
    except Exception as error:
        from simple_coding_agent.git_workspace import DirtyWorkspaceError

        if isinstance(error, DirtyWorkspaceError) and on_dirty is not None:
            on_dirty(error)
        detail = f"local cleanup failed ({type(error).__name__}: {error})"
        _report_hold(on_held, FinalizeReason.CLEANUP_UNCONFIRMED)
        return FinalizeResult(
            status=FinalizeStatus.HELD,
            reason=FinalizeReason.CLEANUP_UNCONFIRMED,
            detail=detail,
            released=True,
            cleanup_ok=False,
            handoff=None,
            failed_step="cleanup",
        )
    try:
        commit(branch)
        handoff = resolve_handoff()
    except SystemExit as error:
        detail = f"accounting is held ({type(error).__name__}: {error})"
        _report_hold(on_held, None)
        return FinalizeResult(
            status=FinalizeStatus.HELD,
            reason=None,
            detail=detail,
            released=True,
            cleanup_ok=True,
            handoff=None,
            failed_step="commit",
        )
    except Exception as error:
        detail = f"accounting failed ({type(error).__name__}: {error})"
        _report_hold(on_held, None)
        return FinalizeResult(
            status=FinalizeStatus.HELD,
            reason=None,
            detail=detail,
            released=True,
            cleanup_ok=True,
            handoff=None,
            failed_step="commit",
        )
    return FinalizeResult(
        status=FinalizeStatus.FINALIZED,
        reason=None,
        detail="attempt finalized",
        released=True,
        cleanup_ok=True,
        handoff=handoff,
    )


def _held_reason(
    request: FinalizeRequest | None, hold: bool, released: bool, cleanup_ok: bool
) -> FinalizeReason:
    if request is not None and not request.comment_posted:
        return FinalizeReason.PUBLICATION_UNCONFIRMED
    if hold:
        return FinalizeReason.HANDOFF_HOLD
    if not released:
        return FinalizeReason.RELEASE_UNCONFIRMED
    if not cleanup_ok:
        return FinalizeReason.CLEANUP_UNCONFIRMED
    return FinalizeReason.HANDOFF_HOLD


def _report_hold(
    on_held: Callable[[FinalizeReason | None, str], None] | None,
    reason: FinalizeReason | None,
) -> None:
    if on_held is None:
        return
    on_held(reason, _HELD_DETAIL)
