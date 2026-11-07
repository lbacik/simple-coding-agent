"""Durable halt record for engine-decided exits (operator contract startup).

Before each engine-decided exit the engine writes a Halt record
(``reason``, ``detail``, ``at``) atomically in the state directory:

- provenance verification failure,
- control endpoint failure,
- the consecutive infrastructure-error limit.

The next start reports it as ``last_halt`` in ``status``; a later
successful start does not erase it (it stays until replaced by a newer
halt). A configuration error before the volume lock cannot be recorded;
the engine exits non-zero with a stderr diagnostic instead. This is a
documented limitation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import tempfile


#: The halt record inside the state directory (``<data_dir>/state/...``).
HALT_FILENAME = "halt.json"

#: Engine-decided exit that failed provenance verification.
HALT_PROVENANCE_FAILED = "provenance_verification_failed"
#: Engine-decided exit that could not open the operator control endpoint.
HALT_CONTROL_ENDPOINT_UNAVAILABLE = "control_endpoint_unavailable"
#: Engine-decided exit on the consecutive infrastructure-error limit.
HALT_CONSECUTIVE_ERROR_LIMIT = "consecutive_error_limit_reached"


@dataclass(frozen=True)
class Halt:
    """One durable engine-decided exit record."""

    reason: str
    detail: str
    at: str


class HaltStoreError(RuntimeError):
    """Raised when the halt record cannot be read or written."""


class HaltStore:
    """Atomically record engine-decided exits and report the last one."""

    def __init__(
        self, data_dir: Path, *, clock: Callable[[], datetime] | None = None
    ) -> None:
        self._path = Path(data_dir) / "state" / HALT_FILENAME
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def path(self) -> Path:
        """The halt file (created on the first recorded halt)."""

        return self._path

    def read(self) -> Halt | None:
        """Return the last recorded halt, or ``None`` when none was recorded.

        A missing file reads as no halt; a successful start never erases it.
        """

        try:
            raw = self._path.read_text()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise HaltStoreError("Halt record could not be read") from error
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise HaltStoreError("Halt record is malformed") from error
        if not isinstance(payload, dict):
            raise HaltStoreError("Halt record is malformed")
        reason = payload.get("reason")
        detail = payload.get("detail")
        at = payload.get("at")
        if (
            not isinstance(reason, str)
            or not reason
            or not isinstance(detail, str)
            or not isinstance(at, str)
            or not at
        ):
            raise HaltStoreError("Halt record is malformed")
        return Halt(reason=reason, detail=detail, at=at)

    def record(self, reason: str, detail: str) -> Halt:
        """Atomically record a halt and return it (replacing any previous one)."""

        if not reason.strip():
            raise HaltStoreError("Halt reason must not be empty")
        halt = Halt(reason=reason, detail=detail, at=_timestamp(self._clock()))
        self._write(halt)
        return halt

    def _write(self, halt: Halt) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=".halt-",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary = Path(file.name)
                json.dump(
                    {"reason": halt.reason, "detail": halt.detail, "at": halt.at},
                    file,
                    sort_keys=True,
                )
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self._path)
        except OSError as error:
            raise HaltStoreError("Halt record could not be written") from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def halt_to_json(halt: Halt | None) -> dict | None:
    """Project one halt onto the contract ``last_halt`` shape, if any."""

    if halt is None:
        return None
    return {"reason": halt.reason, "detail": halt.detail, "at": halt.at}


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise HaltStoreError("Halt clock must include a timezone")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
