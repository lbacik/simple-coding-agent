"""Durable completion boundary for one implementation attempt."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
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
