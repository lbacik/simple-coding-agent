"""Structured process events and durable, redacted attempt evidence."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import threading
from typing import TYPE_CHECKING, TextIO

from simple_coding_agent.agent_version import agent_version

if TYPE_CHECKING:
    from simple_coding_agent.contract_events import ContractEventsBuffer
    from simple_coding_agent.telemetry import RecentEventsBuffer


#: The sealing record written once an Attempt's archive is final (contract
#: section 11.2). It carries ``sealed_at`` and the frozen manifest, and it
#: is never listed in a manifest or exported (it would hash itself).
SEALING_FILENAME = "sealing.json"


class AttemptArchiveSealedError(Exception):
    """A write was refused because the Attempt's archive is already Sealed."""


#: Write-through manifest snapshot per archive directory: ``{run_dir:
#: {relative_path: (size, sha256)}}``. Every :class:`AttemptArchive` mutation
#: refreshes the entries of the files it wrote, so an export can serve the
#: engine's snapshot while streaming the live bytes: a file that grows
#: through a side channel (never through these methods) surfaces as a
#: sha256 mismatch on the consumer side instead of being silently absorbed.
#: Process-local by design: a restart falls back to a freshly computed
#: manifest. Guarded by :data:`_SNAPSHOT_LOCK`.
_SNAPSHOT_REGISTRY: dict[str, dict[str, tuple[int, str]]] = {}
_SNAPSHOT_LOCK = threading.Lock()


def read_manifest_snapshot(run_dir: Path) -> dict[str, tuple[int, str]] | None:
    """Return the write-through snapshot for one archive directory, if any.

    A copy keyed by relative POSIX path with ``(size, sha256)`` values, or
    ``None`` when this process has not written to the directory yet (the
    caller falls back to a freshly computed manifest).
    """

    key = str(run_dir)
    with _SNAPSHOT_LOCK:
        snapshot = _SNAPSHOT_REGISTRY.get(key)
        return dict(snapshot) if snapshot is not None else None


def discard_manifest_snapshot(run_dir: Path) -> None:
    """Forget the write-through snapshot for one archive directory, if any.

    Called when the archive seals: the sealing record carries the frozen
    manifest from then on, so the mutable snapshot is no longer needed.
    """

    with _SNAPSHOT_LOCK:
        _SNAPSHOT_REGISTRY.pop(str(run_dir), None)


def _remember_snapshot_file(run_dir: Path, relative: str) -> None:
    """Refresh one snapshot entry from the file just written, if readable.

    Best-effort: an unreadable file drops its entry so the caller falls
    back to the live bytes.
    """

    path = run_dir / relative
    try:
        raw = path.read_bytes()
    except OSError:
        with _SNAPSHOT_LOCK:
            entries = _SNAPSHOT_REGISTRY.get(str(run_dir))
            if entries is not None:
                entries.pop(relative, None)
        return
    with _SNAPSHOT_LOCK:
        _SNAPSHOT_REGISTRY.setdefault(str(run_dir), {})[relative] = (
            len(raw),
            hashlib.sha256(raw).hexdigest(),
        )


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
    attempt's other evidence files. Each redacted record is also fed to the
    telemetry ``recent_events`` ring buffer (when one is attached), so every
    buffered entry has passed through the redactor. The same redacted
    record is mapped onto one Contract Event and fed to the contract
    ``recent_events`` ring buffer (when one is attached); issue-tagged
    events whose sink resolves an archive are additionally appended to that
    attempt's ``events.jsonl``. The stdout line itself keeps the native
    shape: stdout is not part of the contract.
    """

    def __init__(
        self,
        stream: TextIO,
        *,
        redactions: Iterable[str] = (),
        attempt_sink: AttemptSink | None = None,
        recent_events: RecentEventsBuffer | None = None,
        contract_events: ContractEventsBuffer | None = None,
    ) -> None:
        self._stream = stream
        self._redactor = Redactor(redactions)
        self._attempt_sink = attempt_sink
        self._recent_events = recent_events
        self._contract_events = contract_events

    def emit(
        self,
        event: str,
        *,
        issue_number: int | None = None,
        phase: str | None = None,
        detail: str = "",
        level: str = "INFO",
        attempt_id: str | None = None,
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
        if self._recent_events is not None:
            try:
                self._recent_events.record(record)
            except Exception:
                pass
        archive: AttemptArchive | None = None
        if issue_number is not None and self._attempt_sink is not None:
            try:
                archive = self._attempt_sink(issue_number)
            except Exception:
                archive = None
        if attempt_id is None and archive is not None:
            attempt_id = archive.started_at
        if self._contract_events is not None:
            try:
                from simple_coding_agent.contract_events import to_contract_event

                contract_event = to_contract_event(record, attempt_id=attempt_id)
                self._contract_events.record(contract_event)
                if event != "sealed" and archive is not None:
                    archive.append_contract_event(contract_event)
            except Exception:
                pass
        if event != "sealed" and archive is not None:
            archive.append_event(record)


class AttemptArchive:
    """Own the append-only evidence directory for one implementation attempt."""

    def __init__(
        self, data_dir: Path, *, issue_number: int, started_at: str, redactions: Iterable[str] = ()
    ) -> None:
        safe_timestamp = started_at.replace(":", "-")
        self.directory = data_dir / "logs" / str(issue_number) / safe_timestamp
        # The stable Attempt ID the contract events carry (resolved by the
        # event logger's attempt sink for the record's issue).
        self.started_at = started_at
        self._redactor = Redactor(redactions)

    def is_sealed(self) -> bool:
        """Whether this Attempt's archive is Sealed and immutable."""

        return (self.directory / SEALING_FILENAME).is_file()

    def _ensure_unsealed(self, operation: str) -> None:
        """Refuse a write into a Sealed archive; a later write path is a defect."""

        if self.is_sealed():
            raise AttemptArchiveSealedError(
                f"archive for attempt {self.started_at!r} is sealed;"
                f" refusing {operation}"
            )

    def write_attempt(self, metadata: dict[str, object]) -> None:
        self._ensure_unsealed("write_attempt")
        path = self.directory / "attempt.json"
        try:
            existing = json.loads(path.read_text())
        except FileNotFoundError:
            existing = {}
        except (OSError, json.JSONDecodeError):
            existing = {}
        evidence = existing if isinstance(existing, dict) else {}
        evidence.update(metadata)
        evidence.setdefault("agent_version", agent_version())
        self._write_json("attempt.json", evidence)
        _remember_snapshot_file(self.directory, "attempt.json")

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

        self._ensure_unsealed("append_event")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "agent_output.json"
        with path.open("a", encoding="utf-8") as file:
            json.dump(self._redactor.redact(record), file, sort_keys=True)
            file.write("\n")
        _remember_snapshot_file(self.directory, "agent_output.json")

    def append_contract_event(self, contract_event: dict[str, object]) -> None:
        """Append one Contract Event to this attempt's ``events.jsonl``.

        The event already passed through the logger's redactor; it is
        redacted again here so a directly appended event is safe too.
        Listed in the manifest with kind ``events``.
        """

        self._ensure_unsealed("append_contract_event")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "events.jsonl"
        with path.open("a", encoding="utf-8") as file:
            json.dump(self._redactor.redact(contract_event), file, sort_keys=True)
            file.write("\n")
        _remember_snapshot_file(self.directory, "events.jsonl")

    def write_text(self, name: str, content: str) -> Path:
        self._ensure_unsealed("write_text")
        if Path(name).name != name:
            raise ValueError("Attempt artifact name must not contain a path")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / name
        path.write_text(str(self._redactor.redact(content)))
        _remember_snapshot_file(self.directory, name)
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

        On-disk layout under the pinned CLI (``@anthropic-ai/claude-code@2.1.286``,
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


def read_latest_completed_attempt(data_dir: Path) -> tuple[int, dict[str, object]] | None:
    """Return the latest completed attempt's archive record, or ``None``.

    Scans ``data_dir/logs/<issue>/<started>/attempt.json`` and returns the
    ``(issue_number, record)`` pair whose record carries an outcome with the
    latest completion timestamp. Records that are missing, unreadable, or
    not completed (no ``outcome``) are skipped, so telemetry seeding never
    breaks startup; ``None`` means no completed attempt is on disk.
    """

    try:
        issue_dirs = sorted(path for path in (data_dir / "logs").iterdir() if path.is_dir())
    except OSError:
        return None
    best: tuple[tuple[str, str], int, dict[str, object]] | None = None
    for issue_dir in issue_dirs:
        try:
            issue_number = int(issue_dir.name)
        except ValueError:
            continue
        if issue_number <= 0:
            continue
        try:
            run_dirs = sorted(path for path in issue_dir.iterdir() if path.is_dir())
        except OSError:
            continue
        for run_dir in run_dirs:
            record = _read_completed_record(run_dir / "attempt.json")
            if record is None:
                continue
            key = (str(record["completed_at"]), str(record["started_at"]))
            if best is None or key > best[0]:
                best = (key, issue_number, record)
    if best is None:
        return None
    return (best[1], dict(best[2]))


def _read_completed_record(path: Path) -> dict[str, object] | None:
    """Return the completed attempt record at ``path``, or ``None`` to skip it."""

    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    for key in ("outcome", "started_at", "completed_at"):
        field = value.get(key)
        if not isinstance(field, str) or not field:
            return None
    return value
