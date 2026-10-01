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

    def copy_cli_transcripts(
        self,
        session_id: str | None,
        *,
        issue_number: int | None = None,
        event_log: Callable[..., None] | None = None,
        projects_dir: Path | None = None,
    ) -> tuple[int, int]:
        """Copy this attempt's CLI session transcripts into ``transcripts/``, redacted.

        On-disk layout under the pinned CLI (``@anthropic-ai/claude-code@2.1.276``,
        verified against ``$HOME/.claude/projects`` in the container and the
        SDK's ``session_store.file_path_to_session_key`` documentation):

        - ``$HOME/.claude/projects/<project_key>/<session_id>.jsonl`` is the
          main-loop transcript, where ``<project_key>`` is the CLI's sanitized
          form of the working directory (``/data/repo`` becomes ``-data-repo``).
        - ``$HOME/.claude/projects/<project_key>/<session_id>/subagents/``
          holds the subagent transcripts (``agent-<id>.jsonl``).
        - The ``<session_id>/`` sidecar directory also holds non-transcript
          files (for example ``tool-results/``), which are not copied.

        The project key is not derived here: every project directory is
        searched for the exact ``<session_id>.jsonl`` name (a literal
        filename comparison, never a glob, so a session id containing
        glob metacharacters cannot match another session), so only this
        attempt's session (main file plus its ``subagents/`` tree) is copied
        and no other session's transcripts are touched. Each line is passed
        through this archive's ``Redactor`` before writing. A CLI session
        belongs to exactly one project directory; if several mains ever
        matched, each would be copied and later ones would overwrite
        earlier ones under the same destination names.

        Best-effort: a missing session id, a missing directory, missing files,
        or an I/O error is reported as a WARNING ``transcripts_copy_failed``
        event (with the reason) and never raises. Success logs an INFO
        ``transcripts_copied`` event with the file count and total bytes.
        Returns ``(file_count, total_bytes)``.
        """

        def emit(event: str, detail: str, level: str) -> None:
            if event_log is not None:
                event_log(event, detail, level=level, issue_number=issue_number)

        def fail(reason: str) -> tuple[int, int]:
            emit("transcripts_copy_failed", f"session_id={session_id}; reason={reason}", "WARNING")
            return (0, 0)

        if not session_id or Path(session_id).name != session_id:
            return fail("no CLI session id was recorded for this attempt")
        base = projects_dir if projects_dir is not None else Path.home() / ".claude" / "projects"
        try:
            if not base.is_dir():
                return fail(f"CLI transcripts directory is missing: {base}")
            mains = sorted(
                candidate
                for project in base.iterdir()
                if project.is_dir()
                for candidate in (project / f"{session_id}.jsonl",)
                if candidate.is_file()
            )
            if not mains:
                return fail(f"no transcript found for this session under {base}")
            pending: list[tuple[Path, Path]] = []
            for main in mains:
                pending.append((main, Path(f"{session_id}.jsonl")))
                subagents = main.parent / session_id / "subagents"
                if subagents.is_dir():
                    for child in sorted(subagents.rglob("*.jsonl")):
                        if child.is_file():
                            pending.append(
                                (child, Path(session_id) / "subagents" / child.relative_to(subagents))
                            )
            copied = 0
            total_bytes = 0
            for source, relative in pending:
                try:
                    content = source.read_text(encoding="utf-8")
                except OSError as error:
                    return fail(f"failed to read {source}: {error}")
                redacted = "".join(
                    str(self._redactor.redact(line)) for line in content.splitlines(keepends=True)
                )
                destination = self.directory / "transcripts" / relative
                try:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(redacted, encoding="utf-8")
                except OSError as error:
                    return fail(f"failed to write {destination}: {error}")
                copied += 1
                total_bytes += len(redacted.encode("utf-8"))
        except OSError as error:
            return fail(f"failed to copy transcripts for this session: {error}")
        emit(
            "transcripts_copied",
            f"session_id={session_id}; files={copied}; bytes={total_bytes}",
            "INFO",
        )
        return (copied, total_bytes)

    def _write_json(self, name: str, value: dict[str, object]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / name).write_text(json.dumps(self._redactor.redact(value), sort_keys=True))
