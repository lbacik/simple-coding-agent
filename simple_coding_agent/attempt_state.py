"""Durable filesystem checkpoints for one implementation attempt."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
import json
import os
from pathlib import Path
import tempfile


#: The attempt kinds recorded on a checkpoint. ``ordinary`` is the default
#: model-driven implementation attempt; ``conflict`` is a conflict attempt
#: that only brings an already-published branch up to date. Checkpoints
#: written before the kind existed read as ``ordinary``.
ATTEMPT_KIND_ORDINARY = "ordinary"
ATTEMPT_KIND_CONFLICT = "conflict"

ATTEMPT_KINDS = frozenset({ATTEMPT_KIND_ORDINARY, ATTEMPT_KIND_CONFLICT})


class AttemptStateError(RuntimeError):
    """Raised when an implementation-attempt checkpoint cannot be used safely."""


class AttemptPhase(StrEnum):
    """Ordered stages of an active implementation attempt."""

    CLAIMED = "claimed"
    SETUP = "setup"
    MODEL_RUNNING = "model_running"
    PUSHING = "pushing"
    PUBLISHING = "publishing"


_NEXT_PHASE = {
    AttemptPhase.CLAIMED: AttemptPhase.SETUP,
    AttemptPhase.SETUP: AttemptPhase.MODEL_RUNNING,
    AttemptPhase.MODEL_RUNNING: AttemptPhase.PUSHING,
    AttemptPhase.PUSHING: AttemptPhase.PUBLISHING,
}
_STATE_DIRECTORY = "state"
_ATTEMPT_FILENAME = "attempt.json"


@dataclass(frozen=True)
class AttemptCheckpoint:
    """The durable identity and current phase of an implementation attempt."""

    issue_number: int
    branch: str
    phase: AttemptPhase
    started_at: str
    updated_at: str
    auto_merge_opt_in: bool = False
    requeue_source: str = "human"
    issue_title: str | None = None
    continuation_index: int | None = None
    attempt_kind: str = ATTEMPT_KIND_ORDINARY
    #: The remote revision of the attempt branch observed at attempt start
    #: (``None`` when the branch was absent). ``claim_lease_recorded`` is
    #: ``False`` for checkpoints written before the lease existed, which must
    #: fall back to today's observed lease instead of the recorded value.
    claim_branch_revision: str | None = None
    claim_lease_recorded: bool = False


class AttemptStateStore:
    """Reads and atomically updates the active-attempt checkpoint under DATA_DIR."""

    def __init__(
        self, data_dir: Path, *, clock: Callable[[], datetime] | None = None
    ) -> None:
        self._directory = data_dir / _STATE_DIRECTORY
        self._path = self._directory / _ATTEMPT_FILENAME
        self._clock = clock or _utc_now
        self._last_timestamp: datetime | None = None

    def read(self) -> AttemptCheckpoint | None:
        """Return the active checkpoint, or ``None`` when no attempt is active."""

        try:
            raw_checkpoint = self._path.read_text()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise AttemptStateError("Attempt checkpoint could not be read") from error

        try:
            value = json.loads(raw_checkpoint)
            return _checkpoint_from_json(value)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise AttemptStateError("Attempt checkpoint is malformed") from error

    def start(
        self,
        *,
        issue_number: int,
        branch: str,
        auto_merge_opt_in: bool = False,
        requeue_source: str = "human",
        issue_title: str | None = None,
        continuation_index: int | None = None,
        attempt_kind: str = ATTEMPT_KIND_ORDINARY,
    ) -> AttemptCheckpoint:
        """Record the initial claimed phase for a new implementation attempt.

        ``auto_merge_opt_in`` captures whether the claimed issue carried the
        ``agent-auto-merge`` label at claim time, so recovery keeps the
        claim-time decision instead of re-reading the current labels.
        ``requeue_source`` captures how the attempt was requeued (``human``
        when the issue carried ``ready-for-agent`` at claim time,
        ``automatic`` for an automatic continuation, ``conflict`` for a
        conflict requeue), so every attempt-result comment records the actual
        reason for the claim even across recovery.
        ``issue_title`` and ``continuation_index`` are computed once at claim
        so telemetry needs no tracker lookup for them; ``None`` means unknown
        (including checkpoints written before this change).
        ``attempt_kind`` captures whether the claim routed to a conflict
        attempt (``conflict``) or the ordinary model-driven path
        (``ordinary``), so startup recovery publishes a conflict result
        (which counts toward the conflict-attempt budget) instead of an
        ordinary one after a restart.
        """

        if self.read() is not None:
            raise AttemptStateError("An implementation attempt is already active")
        if isinstance(issue_number, bool) or not isinstance(issue_number, int) or issue_number <= 0:
            raise AttemptStateError("Issue number must be a positive integer")
        if not isinstance(branch, str) or not branch.strip():
            raise AttemptStateError("Branch must be a non-empty string")
        if not isinstance(auto_merge_opt_in, bool):
            raise AttemptStateError("Auto-merge opt-in must be a boolean")
        if requeue_source not in ("human", "automatic", "conflict"):
            raise AttemptStateError("Requeue source must be 'human', 'automatic', or 'conflict'")
        if issue_title is not None and not isinstance(issue_title, str):
            raise AttemptStateError("Issue title must be a string or None")
        if continuation_index is not None and (
            isinstance(continuation_index, bool)
            or not isinstance(continuation_index, int)
            or continuation_index < 0
        ):
            raise AttemptStateError("Continuation index must be a non-negative integer or None")
        if attempt_kind not in ATTEMPT_KINDS:
            raise AttemptStateError("Attempt kind must be 'ordinary' or 'conflict'")

        timestamp = self._next_timestamp()
        checkpoint = AttemptCheckpoint(
            issue_number=issue_number,
            branch=branch,
            phase=AttemptPhase.CLAIMED,
            started_at=timestamp,
            updated_at=timestamp,
            auto_merge_opt_in=auto_merge_opt_in,
            requeue_source=requeue_source,
            issue_title=issue_title,
            continuation_index=continuation_index,
            attempt_kind=attempt_kind,
        )
        self._write(checkpoint)
        return checkpoint

    def transition(self, phase: AttemptPhase) -> AttemptCheckpoint:
        """Advance the active attempt to its next lifecycle phase."""

        checkpoint = self.read()
        if checkpoint is None:
            raise AttemptStateError("No implementation attempt is active")
        expected_phase = _NEXT_PHASE.get(checkpoint.phase)
        if phase is not expected_phase:
            raise AttemptStateError(
                f"Cannot transition from {checkpoint.phase} to {phase}"
            )

        transitioned = AttemptCheckpoint(
            issue_number=checkpoint.issue_number,
            branch=checkpoint.branch,
            phase=phase,
            started_at=checkpoint.started_at,
            updated_at=self._next_timestamp(),
            auto_merge_opt_in=checkpoint.auto_merge_opt_in,
            requeue_source=checkpoint.requeue_source,
            issue_title=checkpoint.issue_title,
            continuation_index=checkpoint.continuation_index,
            attempt_kind=checkpoint.attempt_kind,
            claim_branch_revision=checkpoint.claim_branch_revision,
            claim_lease_recorded=checkpoint.claim_lease_recorded,
        )
        self._write(transitioned)
        return transitioned

    def record_claim_lease(self, revision: str | None) -> AttemptCheckpoint:
        """Durably record the remote revision observed at attempt start.

        ``revision`` is the observed remote sha of the attempt branch, or
        ``None`` when the branch was absent. Only the first recording wins:
        recovery must keep the original lease instead of overwriting it with
        a later observation. Returns the stored checkpoint.
        """

        checkpoint = self.read()
        if checkpoint is None:
            raise AttemptStateError("No implementation attempt is active")
        _validate_claim_lease(revision)
        if checkpoint.claim_lease_recorded:
            return checkpoint
        recorded = AttemptCheckpoint(
            issue_number=checkpoint.issue_number,
            branch=checkpoint.branch,
            phase=checkpoint.phase,
            started_at=checkpoint.started_at,
            updated_at=self._next_timestamp(),
            auto_merge_opt_in=checkpoint.auto_merge_opt_in,
            requeue_source=checkpoint.requeue_source,
            issue_title=checkpoint.issue_title,
            continuation_index=checkpoint.continuation_index,
            attempt_kind=checkpoint.attempt_kind,
            claim_branch_revision=revision,
            claim_lease_recorded=True,
        )
        self._write(recorded)
        return recorded

    def advance_claim_lease(self, revision: str) -> AttemptCheckpoint:
        """Advance the recorded lease to a revision this attempt just pushed.

        Called after each own successful push so later pushes in the same
        attempt use the new revision as their lease. Requires a non-empty
        revision; records the lease when an older checkpoint never stored one.
        """

        checkpoint = self.read()
        if checkpoint is None:
            raise AttemptStateError("No implementation attempt is active")
        _validate_claim_lease(revision)
        if revision is None:
            raise AttemptStateError("Claim lease advance requires a revision")
        advanced = AttemptCheckpoint(
            issue_number=checkpoint.issue_number,
            branch=checkpoint.branch,
            phase=checkpoint.phase,
            started_at=checkpoint.started_at,
            updated_at=self._next_timestamp(),
            auto_merge_opt_in=checkpoint.auto_merge_opt_in,
            requeue_source=checkpoint.requeue_source,
            issue_title=checkpoint.issue_title,
            continuation_index=checkpoint.continuation_index,
            attempt_kind=checkpoint.attempt_kind,
            claim_branch_revision=revision,
            claim_lease_recorded=True,
        )
        self._write(advanced)
        return advanced

    def delete(self) -> None:
        """Remove the active checkpoint after its lifecycle cleanup completes."""

        try:
            self._path.unlink()
        except FileNotFoundError:
            return
        except OSError as error:
            raise AttemptStateError("Attempt checkpoint could not be deleted") from error

    def _write(self, checkpoint: AttemptCheckpoint) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self._directory, prefix=".attempt-", suffix=".tmp", delete=False
            ) as file:
                temporary = Path(file.name)
                json.dump(_checkpoint_to_json(checkpoint), file, sort_keys=True)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self._path)
        except OSError as error:
            raise AttemptStateError("Attempt checkpoint could not be written") from error
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def _next_timestamp(self) -> str:
        timestamp = _as_utc(self._clock())
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            timestamp = self._last_timestamp + timedelta(microseconds=1)
        self._last_timestamp = timestamp
        return _timestamp(timestamp)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: datetime) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise AttemptStateError("Attempt timestamp must include a timezone")
    return value.astimezone(UTC)


def _validate_claim_lease(revision: str | None) -> None:
    if revision is None:
        return
    if not isinstance(revision, str) or not revision.strip():
        raise AttemptStateError("Claim lease revision must be a non-empty string or None")
    if any(character.isspace() for character in revision):
        raise AttemptStateError("Claim lease revision must not contain whitespace")


def _checkpoint_to_json(checkpoint: AttemptCheckpoint) -> dict[str, bool | int | str | None]:
    return {
        "issue_number": checkpoint.issue_number,
        "branch": checkpoint.branch,
        "phase": checkpoint.phase.value,
        "started_at": checkpoint.started_at,
        "updated_at": checkpoint.updated_at,
        "auto_merge_opt_in": checkpoint.auto_merge_opt_in,
        "requeue_source": checkpoint.requeue_source,
        "issue_title": checkpoint.issue_title,
        "continuation_index": checkpoint.continuation_index,
        "attempt_kind": checkpoint.attempt_kind,
        "claim_branch_revision": checkpoint.claim_branch_revision,
        "claim_lease_recorded": checkpoint.claim_lease_recorded,
    }


def _checkpoint_from_json(value: object) -> AttemptCheckpoint:
    _required = {"issue_number", "branch", "phase", "started_at", "updated_at"}
    _optional = {
        "auto_merge_opt_in",
        "requeue_source",
        "issue_title",
        "continuation_index",
        "attempt_kind",
        # Checkpoints written before the kind existed carry neither key and
        # read as ordinary; checkpoints written while the conflict claim was
        # recorded under ``is_conflict_claim`` map that flag onto the kind.
        "is_conflict_claim",
        # Checkpoints written before the claim lease carry neither key and
        # read as unrecorded, so recovery falls back to today's observed lease.
        "claim_branch_revision",
        "claim_lease_recorded",
    }
    if not isinstance(value, dict):
        raise ValueError("invalid checkpoint shape")
    keys = set(value)
    if not _required <= keys <= (_required | _optional):
        raise ValueError("invalid checkpoint shape")

    issue_number = value["issue_number"]
    branch = value["branch"]
    phase = value["phase"]
    started_at = value["started_at"]
    updated_at = value["updated_at"]
    auto_merge_opt_in = value.get("auto_merge_opt_in", False)
    requeue_source = value.get("requeue_source", "human")
    issue_title = value.get("issue_title", None)
    continuation_index = value.get("continuation_index", None)
    attempt_kind = value.get("attempt_kind", None)
    if attempt_kind is None:
        # The kind postdates both earlier shapes: checkpoints with no kind
        # read as ordinary, except ones whose conflict claim was recorded
        # under the earlier boolean flag, which map onto a conflict kind so
        # a restart still recovers them as conflict attempts.
        if value.get("is_conflict_claim", False) is True:
            attempt_kind = ATTEMPT_KIND_CONFLICT
        else:
            attempt_kind = ATTEMPT_KIND_ORDINARY
    if isinstance(issue_number, bool) or not isinstance(issue_number, int) or issue_number <= 0:
        raise ValueError("invalid issue number")
    if not isinstance(branch, str) or not branch.strip():
        raise ValueError("invalid branch")
    if not isinstance(phase, str):
        raise ValueError("invalid phase")
    if not isinstance(started_at, str) or not isinstance(updated_at, str):
        raise ValueError("invalid timestamps")
    if not isinstance(auto_merge_opt_in, bool):
        raise ValueError("invalid auto-merge opt-in")
    if requeue_source not in ("human", "automatic", "conflict"):
        raise ValueError("invalid requeue source")
    if issue_title is not None and not isinstance(issue_title, str):
        raise ValueError("invalid issue title")
    if continuation_index is not None and (
        isinstance(continuation_index, bool)
        or not isinstance(continuation_index, int)
        or continuation_index < 0
    ):
        raise ValueError("invalid continuation index")
    if attempt_kind not in ATTEMPT_KINDS:
        raise ValueError("invalid attempt kind")
    if "is_conflict_claim" in value and value["is_conflict_claim"] not in (True, False):
        raise ValueError("invalid conflict-claim flag")
    if "claim_lease_recorded" in value:
        claim_lease_recorded = value["claim_lease_recorded"]
    else:
        claim_lease_recorded = False
    if "claim_branch_revision" in value:
        claim_branch_revision = value["claim_branch_revision"]
    else:
        claim_branch_revision = None
    if not isinstance(claim_lease_recorded, bool):
        raise ValueError("invalid claim lease flag")
    if claim_branch_revision is not None:
        if (
            not isinstance(claim_branch_revision, str)
            or not claim_branch_revision.strip()
            or any(character.isspace() for character in claim_branch_revision)
        ):
            raise ValueError("invalid claim lease revision")
    if not claim_lease_recorded:
        # Checkpoints written before the lease existed carry neither key;
        # a stray revision without the recorded flag reads as unrecorded so
        # recovery falls back to today's observed lease.
        claim_branch_revision = None
    _parse_timestamp(started_at)
    _parse_timestamp(updated_at)
    return AttemptCheckpoint(
        issue_number=issue_number,
        branch=branch,
        phase=AttemptPhase(phase),
        started_at=started_at,
        updated_at=updated_at,
        auto_merge_opt_in=auto_merge_opt_in,
        requeue_source=requeue_source,
        issue_title=issue_title,
        continuation_index=continuation_index,
        attempt_kind=attempt_kind,
        claim_branch_revision=claim_branch_revision,
        claim_lease_recorded=claim_lease_recorded,
    )


def _parse_timestamp(value: str) -> None:
    if not value.endswith("Z"):
        raise ValueError("timestamp must be UTC")
    datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
