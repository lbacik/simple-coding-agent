"""Durable process-wide operating controls."""

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
class ConsecutiveErrorState:
    """The restart-safe state of the infrastructure-error circuit breaker."""

    count: int
    last_success_at: str | None
    last_attempt_id: str | None


class ConsecutiveErrorStore:
    """Atomically record terminal outcomes without counting an attempt twice."""

    def __init__(self, data_dir: Path, *, clock: Callable[[], datetime] | None = None) -> None:
        self._path = data_dir / "state" / "consecutive_errors.json"
        self._clock = clock or (lambda: datetime.now(UTC))

    def read(self) -> ConsecutiveErrorState:
        try:
            raw = json.loads(self._path.read_text())
        except FileNotFoundError:
            return ConsecutiveErrorState(0, None, None)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Consecutive-error state could not be read") from error
        if not isinstance(raw, dict):
            raise RuntimeError("Consecutive-error state is malformed")
        count = raw.get("count")
        last_success_at = raw.get("last_success_at")
        last_attempt_id = raw.get("last_attempt_id")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or last_success_at is not None and not isinstance(last_success_at, str)
            or last_attempt_id is not None and not isinstance(last_attempt_id, str)
        ):
            raise RuntimeError("Consecutive-error state is malformed")
        return ConsecutiveErrorState(count, last_success_at, last_attempt_id)

    def record(self, outcome: AttemptOutcome, attempt_id: str) -> int:
        """Record an outcome and return the resulting infrastructure-error count."""

        state = self.read()
        if state.last_attempt_id == attempt_id:
            return state.count
        if outcome is AttemptOutcome.INFRASTRUCTURE_ERROR:
            updated = ConsecutiveErrorState(state.count + 1, state.last_success_at, attempt_id)
        else:
            updated = ConsecutiveErrorState(0, _timestamp(self._clock()), attempt_id)
        self._write(updated)
        return updated.count

    def _write(self, state: ConsecutiveErrorState) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self._path.parent, prefix=".consecutive-", suffix=".tmp", delete=False
            ) as file:
                temporary = Path(file.name)
                json.dump(
                    {
                        "count": state.count,
                        "last_success_at": state.last_success_at,
                        "last_attempt_id": state.last_attempt_id,
                    },
                    file,
                    sort_keys=True,
                )
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self._path)
        except OSError as error:
            raise RuntimeError("Consecutive-error state could not be written") from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise RuntimeError("Operating-control clock must include a timezone")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
