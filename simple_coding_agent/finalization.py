"""Durable completion boundary for one implementation attempt."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import tempfile
from enum import StrEnum
from typing import Protocol

from simple_coding_agent.attempt_state import AttemptCheckpoint, AttemptStateStore
from simple_coding_agent.completion import AttemptOutcome
from simple_coding_agent.git_workspace import DirtyWorkspaceError, PreparedAttempt


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

    def __init__(
        self, data_dir: Path, *, clock: Callable[[], datetime] | None = None
    ) -> None:
        self._path = data_dir / "state" / "completed_attempts.json"
        self._clock = clock or (lambda: datetime.now(UTC))

    def is_finalized(self, attempt_id: str) -> bool:
        """Return True when this attempt ID already completed its boundary."""

        return self.get(attempt_id) is not None

    def get(self, attempt_id: str) -> AttemptCompletion | None:
        """Read the recorded verdict for a finalized attempt."""

        return self._read().get(attempt_id)

    def record(
        self,
        *,
        attempt_id: str,
        issue_number: int,
        branch: str,
        outcome: AttemptOutcome,
    ) -> bool:
        """Record one finalized attempt; return False when it was already recorded."""

        records = self._read()
        if attempt_id in records:
            return False
        if (
            isinstance(issue_number, bool)
            or not isinstance(issue_number, int)
            or issue_number <= 0
        ):
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
            raise CompletionStoreError(
                "Completion record could not be written"
            ) from error
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
    if (
        isinstance(issue_number, bool)
        or not isinstance(issue_number, int)
        or issue_number <= 0
    ):
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


class FinalizationMode(StrEnum):
    NORMAL = "normal"
    RECOVERY_RETRY = "recovery_retry"
    RECOVERY_RELEASE = "recovery_release"


class FinalizationStage(StrEnum):
    BEFORE_RELEASE = "before_release"
    RELEASE = "release"
    CLEANUP = "cleanup"
    AFTER_CLEANUP = "after_cleanup"
    ACCOUNTING = "accounting"
    HANDOFF = "handoff"


class HeldReason(StrEnum):
    PUBLICATION_UNCONFIRMED = "publication_unconfirmed"
    HANDOFF_NOT_FULFILLED = "handoff_not_fulfilled"
    RELEASE_FAILED = "release_failed"
    CLEANUP_FAILED = "cleanup_failed"
    ACCOUNTING_FAILED = "accounting_failed"


@dataclass(frozen=True)
class PublicationState:
    confirmed: bool
    started_at: datetime | None = None


@dataclass(frozen=True)
class FinalizeRequest:
    outcome: AttemptOutcome
    publication: PublicationState
    prepared: PreparedAttempt | None = None
    base_branch: str = "main"
    retain_branch: bool = False
    mode: FinalizationMode = FinalizationMode.NORMAL


@dataclass(frozen=True)
class Finalized:
    outcome: AttemptOutcome


@dataclass(frozen=True)
class Held:
    reason: HeldReason
    stage: FinalizationStage
    detail: str
    error: BaseException | None = None


class HandoffPolicy(Protocol):
    def __call__(
        self,
        *,
        outcome: AttemptOutcome,
        prepared: PreparedAttempt | None,
        publication_started_at: datetime | None,
        check_allowance: bool,
    ) -> bool | None: ...


@dataclass(frozen=True)
class FinalizationEffects:
    release: Callable[[AttemptOutcome, FinalizationMode], None]
    cleanup: Callable[[PreparedAttempt, str, bool, FinalizationMode], None]
    check_handoff: HandoffPolicy
    resolve_handoff: HandoffPolicy
    write_outcome: Callable[[AttemptOutcome], str]
    finish_completion: Callable[[AttemptOutcome, str, bool], None]
    record_control: Callable[[AttemptOutcome], None]
    handoff_owned: Callable[[], bool]
    finish_replay: Callable[[], None]
    workspace_held: Callable[[DirtyWorkspaceError, FinalizationMode], None]
    report_hold: Callable[[str], None]
    resolution_unavailable: Callable[[Exception], None]


@dataclass(frozen=True)
class AttemptFinalizer:
    """Close one attempt using its checkpoint and the verdict publication supplied."""

    checkpoint: AttemptCheckpoint
    ledger: AttemptCompletionStore | None
    attempt_state: AttemptStateStore
    effects: FinalizationEffects

    def finalize(self, request: FinalizeRequest) -> Finalized | Held:
        replayed = self.replay()
        if replayed is not None:
            return replayed

        normal = request.mode is FinalizationMode.NORMAL
        publication = request.publication
        if (
            normal
            and publication.confirmed
            and self.effects.check_handoff(
                outcome=request.outcome,
                prepared=request.prepared,
                publication_started_at=publication.started_at,
                check_allowance=True,
            )
        ):
            return self._hold(
                request,
                HeldReason.HANDOFF_NOT_FULFILLED,
                FinalizationStage.BEFORE_RELEASE,
                "an operator handoff requires recovery",
            )

        release_error: BaseException | None = None
        if publication.confirmed:
            try:
                self.effects.release(request.outcome, request.mode)
            except BaseException as error:
                release_error = error
                if not normal:
                    if not isinstance(error, Exception):
                        raise
                    return self._release_failed(request, error)

        if request.prepared is not None:
            try:
                self.effects.cleanup(
                    request.prepared,
                    request.base_branch,
                    request.retain_branch
                    or not publication.confirmed
                    or request.mode is FinalizationMode.RECOVERY_RELEASE,
                    request.mode,
                )
            except DirtyWorkspaceError as error:
                self.effects.workspace_held(error, request.mode)
                return self._hold(
                    request,
                    HeldReason.CLEANUP_FAILED,
                    FinalizationStage.CLEANUP,
                    f"local cleanup failed ({_error_detail(error)})",
                    release_error if normal else None,
                )
            except BaseException as error:
                if not normal and not isinstance(error, Exception):
                    raise
                return self._hold(
                    request,
                    HeldReason.CLEANUP_FAILED,
                    FinalizationStage.CLEANUP,
                    f"local cleanup failed ({_error_detail(error)})",
                    error,
                )

        if release_error is not None:
            return self._release_failed(request, release_error)
        if not publication.confirmed:
            return self._hold(
                request,
                HeldReason.PUBLICATION_UNCONFIRMED,
                FinalizationStage.BEFORE_RELEASE,
                "publication is unconfirmed",
            )
        if normal and self.effects.check_handoff(
            outcome=request.outcome,
            prepared=None,
            publication_started_at=publication.started_at,
            check_allowance=False,
        ):
            return self._hold(
                request,
                HeldReason.HANDOFF_NOT_FULFILLED,
                FinalizationStage.AFTER_CLEANUP,
                "an operator handoff failed after cleanup",
            )

        branch = (
            request.prepared.branch
            if request.prepared is not None
            else self.checkpoint.branch
        )
        try:
            # Write the outcome first. A ledger entry then proves release and
            # cleanup finished, so a restart only replays idempotent bookkeeping.
            completed_at = self.effects.write_outcome(request.outcome)
            if self.ledger is not None:
                self.ledger.record(
                    attempt_id=self.checkpoint.started_at,
                    issue_number=self.checkpoint.issue_number,
                    branch=branch,
                    outcome=request.outcome,
                )
            self.effects.finish_completion(
                request.outcome,
                completed_at,
                request.mode is FinalizationMode.RECOVERY_RELEASE,
            )
        except SystemExit as error:
            if normal:
                raise
            return self._hold(
                request,
                HeldReason.ACCOUNTING_FAILED,
                FinalizationStage.ACCOUNTING,
                f"accounting is held ({_error_detail(error)})",
                error,
            )
        except Exception as error:
            return self._hold(
                request,
                HeldReason.ACCOUNTING_FAILED,
                FinalizationStage.ACCOUNTING,
                f"accounting failed ({_error_detail(error)})",
                error,
            )
        self._resolve_handoff(
            request.outcome,
            prepared=None if normal else request.prepared,
            started_at=publication.started_at,
            check_allowance=not normal,
        )
        return Finalized(request.outcome)

    def replay(self) -> Finalized | Held | None:
        """Settle an already recorded attempt without replaying its remote effects."""

        previous = (
            self.ledger.get(self.checkpoint.started_at)
            if self.ledger is not None
            else None
        )
        if previous is None:
            return None

        retained = self.attempt_state.read()
        if retained is None or retained.started_at != previous.attempt_id:
            return Finalized(previous.outcome)

        self.effects.record_control(previous.outcome)
        resolved = self._resolve_handoff(previous.outcome)
        if resolved is False and self.effects.handoff_owned():
            detail = (
                "A failed operator handoff owns this attempt; the branch, "
                "checkpoint, and working tree are preserved for recovery."
            )
            self.effects.report_hold(detail)
            return Held(
                HeldReason.HANDOFF_NOT_FULFILLED, FinalizationStage.HANDOFF, detail
            )

        self.attempt_state.delete()
        self.effects.finish_replay()
        return Finalized(previous.outcome)

    def _resolve_handoff(
        self,
        outcome: AttemptOutcome,
        *,
        prepared: PreparedAttempt | None = None,
        started_at: datetime | None = None,
        check_allowance: bool = False,
    ) -> bool | None:
        try:
            return self.effects.resolve_handoff(
                outcome=outcome,
                prepared=prepared,
                publication_started_at=started_at,
                check_allowance=check_allowance,
            )
        except Exception as error:
            self.effects.resolution_unavailable(error)
            return None

    def _release_failed(self, request: FinalizeRequest, error: BaseException) -> Held:
        return self._hold(
            request,
            HeldReason.RELEASE_FAILED,
            FinalizationStage.RELEASE,
            f"issue release failed ({_error_detail(error)})",
            error,
        )

    def _hold(
        self,
        request: FinalizeRequest,
        reason: HeldReason,
        stage: FinalizationStage,
        detail: str,
        error: BaseException | None = None,
    ) -> Held:
        if request.mode is FinalizationMode.NORMAL:
            self.effects.report_hold(
                "Publication, release, or cleanup is unconfirmed, or a failed"
                " operator handoff owns this attempt; the branch, checkpoint, "
                "and working tree are preserved for recovery."
            )
        return Held(reason, stage, detail, error)


def _error_detail(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
