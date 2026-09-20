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

    def start(self, *, issue_number: int, branch: str) -> AttemptCheckpoint:
        """Record the initial claimed phase for a new implementation attempt."""

        if self.read() is not None:
            raise AttemptStateError("An implementation attempt is already active")
        if isinstance(issue_number, bool) or not isinstance(issue_number, int) or issue_number <= 0:
            raise AttemptStateError("Issue number must be a positive integer")
        if not isinstance(branch, str) or not branch.strip():
            raise AttemptStateError("Branch must be a non-empty string")

        timestamp = self._next_timestamp()
        checkpoint = AttemptCheckpoint(
            issue_number=issue_number,
            branch=branch,
            phase=AttemptPhase.CLAIMED,
            started_at=timestamp,
            updated_at=timestamp,
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
        )
        self._write(transitioned)
        return transitioned

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


def _checkpoint_to_json(checkpoint: AttemptCheckpoint) -> dict[str, int | str]:
    return {
        "issue_number": checkpoint.issue_number,
        "branch": checkpoint.branch,
        "phase": checkpoint.phase.value,
        "started_at": checkpoint.started_at,
        "updated_at": checkpoint.updated_at,
    }


def _checkpoint_from_json(value: object) -> AttemptCheckpoint:
    if not isinstance(value, dict) or set(value) != {
        "issue_number",
        "branch",
        "phase",
        "started_at",
        "updated_at",
    }:
        raise ValueError("invalid checkpoint shape")

    issue_number = value["issue_number"]
    branch = value["branch"]
    phase = value["phase"]
    started_at = value["started_at"]
    updated_at = value["updated_at"]
    if isinstance(issue_number, bool) or not isinstance(issue_number, int) or issue_number <= 0:
        raise ValueError("invalid issue number")
    if not isinstance(branch, str) or not branch.strip():
        raise ValueError("invalid branch")
    if not isinstance(phase, str):
        raise ValueError("invalid phase")
    if not isinstance(started_at, str) or not isinstance(updated_at, str):
        raise ValueError("invalid timestamps")
    _parse_timestamp(started_at)
    _parse_timestamp(updated_at)
    return AttemptCheckpoint(
        issue_number=issue_number,
        branch=branch,
        phase=AttemptPhase(phase),
        started_at=started_at,
        updated_at=updated_at,
    )


def _parse_timestamp(value: str) -> None:
    if not value.endswith("Z"):
        raise ValueError("timestamp must be UTC")
    datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
