"""Structured process events and durable, redacted attempt evidence."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import TextIO


class Redactor:
    """Apply write-time credential redaction to all operational evidence."""

    def __init__(self, values: Iterable[str]) -> None:
        self._values = tuple(value for value in values if value)

    def redact(self, value: object) -> object:
        if isinstance(value, str):
            for secret in self._values:
                value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, tuple):
            return [self.redact(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self.redact(item) for key, item in value.items()}
        return value


AttemptSink = Callable[[int], "AttemptArchive | None"]


class JsonEventLogger:
    """Write Docker/systemd-friendly JSON lines with the canonical fields.

    Each emitted record is also mirrored into the active attempt's
    ``agent_output.json`` (when ``attempt_sink`` resolves one), so the same
    output an operator sees on stdout is durably readable next to the
    attempt's other evidence files.
    """

    def __init__(
        self,
        stream: TextIO,
        *,
        redactions: Iterable[str] = (),
        attempt_sink: AttemptSink | None = None,
    ) -> None:
        self._stream = stream
        self._redactor = Redactor(redactions)
        self._attempt_sink = attempt_sink

    def emit(
        self,
        event: str,
        *,
        issue_number: int | None = None,
        phase: str | None = None,
        detail: str = "",
        level: str = "INFO",
    ) -> None:
        record = self._redactor.redact(
            {
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "level": level,
                "event": event,
                "issue_number": issue_number,
                "phase": phase,
                "detail": detail,
            }
        )
        json.dump(record, self._stream, sort_keys=True)
        self._stream.write("\n")
        self._stream.flush()
        if issue_number is not None and self._attempt_sink is not None:
            archive = self._attempt_sink(issue_number)
            if archive is not None:
                archive.append_event(record)


class AttemptArchive:
    """Own the append-only evidence directory for one implementation attempt."""

    def __init__(
        self, data_dir: Path, *, issue_number: int, started_at: str, redactions: Iterable[str] = ()
    ) -> None:
        safe_timestamp = started_at.replace(":", "-")
        self.directory = data_dir / "logs" / str(issue_number) / safe_timestamp
        self._redactor = Redactor(redactions)

    def write_attempt(self, metadata: dict[str, object]) -> None:
        path = self.directory / "attempt.json"
        try:
            existing = json.loads(path.read_text())
        except FileNotFoundError:
            existing = {}
        except (OSError, json.JSONDecodeError):
            existing = {}
        evidence = existing if isinstance(existing, dict) else {}
        evidence.update(metadata)
        evidence["token_estimate_authority"] = "non-authoritative"
        self._write_json("attempt.json", evidence)

    def read_attempt(self) -> dict[str, object]:
        """Return the merged attempt.json record, or {} when it is missing."""

        path = self.directory / "attempt.json"
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def append_event(self, record: dict[str, object]) -> None:
        """Append one emitted log record to this attempt's agent_output.json."""

        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "agent_output.json"
        with path.open("a", encoding="utf-8") as file:
            json.dump(self._redactor.redact(record), file, sort_keys=True)
            file.write("\n")

    def write_text(self, name: str, content: str) -> Path:
        if Path(name).name != name:
            raise ValueError("Attempt artifact name must not contain a path")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / name
        path.write_text(str(self._redactor.redact(content)))
        return path

    def _write_json(self, name: str, value: dict[str, object]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / name).write_text(json.dumps(self._redactor.redact(value), sort_keys=True))
