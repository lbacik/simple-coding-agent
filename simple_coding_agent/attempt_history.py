"""Contract projection over an instance's Attempt history (``attempts``).

A dashboard pages through the history with ``attempts list`` and reads one
Attempt's evidence with ``attempts show``. Listing reads the archive
directories under ``<data-dir>/logs`` plus the live checkpoint, so an
Attempt is listed as soon as its Claim opens. The Attempt ID is the
checkpoint ``started_at``; an archive directory (whose name replaces ``:``
with ``-``) is mapped back to its Attempt ID through the ``started_at``
recorded in its ``attempt.json``, never by string reversal, so directories
without a recorded ``started_at`` are skipped.

An unsealed archive, including the active Attempt's, is a moving snapshot:
its summary and manifest are computed at read time. Sealing lands in a
later ticket, so ``sealed`` is always ``False`` here.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Container, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from typing import Any, BinaryIO

from simple_coding_agent.observability import (
    SEALING_FILENAME,
    discard_manifest_snapshot,
    read_manifest_snapshot,
)


#: ``attempts list`` page size when the consumer passes no ``--limit``.
ATTEMPTS_DEFAULT_LIMIT = 20
#: ``attempts list`` maximum page size (contract section 6.1).
ATTEMPTS_MAX_LIMIT = 100


class AttemptHistoryError(ValueError):
    """A rejected history read, carrying its contract error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ActiveAttempt:
    """The live checkpoint overlaid onto the archive listing."""

    attempt_id: str
    issue_number: int
    issue_title: str | None
    #: Native lifecycle state: the checkpoint phase (``claimed``, ``setup``,
    #: ``model_running``, ``pushing``, ``publishing``).
    phase: str


def encode_cursor(attempt_id: str) -> str:
    """Encode one Attempt ID as an opaque page cursor."""

    return base64.urlsafe_b64encode(attempt_id.encode("utf-8")).decode("ascii")


def decode_cursor(cursor: str) -> str:
    """Decode one opaque page cursor back to its Attempt ID.

    Raises :class:`AttemptHistoryError` with ``invalid_argument`` when the
    cursor is not a well-formed encoding of a non-empty Attempt ID.
    """

    if not isinstance(cursor, str) or not cursor:
        raise AttemptHistoryError("invalid_argument", "invalid cursor: must not be empty")
    padded = cursor + "=" * (-len(cursor) % 4)
    try:
        attempt_id = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        raise AttemptHistoryError(
            "invalid_argument", f"invalid cursor {cursor!r}: not an encoded attempt ID"
        ) from None
    if not attempt_id:
        raise AttemptHistoryError("invalid_argument", "invalid cursor: must not be empty")
    if len(attempt_id) > 256:
        raise AttemptHistoryError(
            "invalid_argument", "invalid cursor: attempt ID is too long"
        )
    return attempt_id


def list_attempts(
    data_dir: Path,
    *,
    active: ActiveAttempt | None = None,
    held: Container[str] = frozenset(),
    limit: int | None = None,
    before: str | None = None,
) -> dict[str, Any]:
    """Return ``{"attempts": [AttemptSummary], "next_cursor": str|None}``.

    Attempts sort newest first by Attempt ID. ``limit`` defaults to
    :data:`ATTEMPTS_DEFAULT_LIMIT` and must stay within ``1..100``;
    ``before`` pages after the given opaque cursor. Raises
    :class:`AttemptHistoryError` with ``invalid_argument`` for an
    out-of-range limit or a malformed cursor.
    """

    page_size = ATTEMPTS_DEFAULT_LIMIT if limit is None else limit
    if (
        isinstance(page_size, bool)
        or not isinstance(page_size, int)
        or not 1 <= page_size <= ATTEMPTS_MAX_LIMIT
    ):
        raise AttemptHistoryError(
            "invalid_argument",
            f"invalid limit {limit!r}: must be an integer 1..{ATTEMPTS_MAX_LIMIT}",
        )
    cursor_id = decode_cursor(before) if before is not None else None

    entries = _collect_entries(data_dir, active=active, held=held)
    if cursor_id is not None:
        entries = [entry for entry in entries if _sort_key(entry[0]) < _sort_key(cursor_id)]
    page = entries[:page_size]
    next_cursor = encode_cursor(page[-1][0]) if len(entries) > page_size else None
    return {
        "attempts": [summary for _, summary in page],
        "next_cursor": next_cursor,
    }


def show_attempt(
    data_dir: Path,
    attempt_id: str,
    *,
    active: ActiveAttempt | None = None,
    held: Container[str] = frozenset(),
) -> dict[str, Any]:
    """Return ``{"attempt": AttemptSummary, "manifest": [ManifestEntry]}``.

    The manifest is computed at read time, so an unsealed archive (including
    the active Attempt's) reads as a moving snapshot, and it never includes
    private engine state. Raises :class:`AttemptHistoryError` with
    ``invalid_argument`` for a missing Attempt ID and ``not_found`` for an
    unknown one.
    """

    if not isinstance(attempt_id, str) or not attempt_id:
        raise AttemptHistoryError(
            "invalid_argument", "invalid attempt ID: must not be empty"
        )
    directories = _archive_directories(data_dir)
    if active is not None and active.attempt_id == attempt_id:
        summary = _live_summary(active, held=held)
        return {"attempt": summary, "manifest": _manifest(directories, attempt_id)}
    run_dir = find_run_directory(data_dir, attempt_id)
    if run_dir is None:
        raise AttemptHistoryError(
            "not_found", f"unknown attempt {attempt_id!r}"
        )
    record = _read_record(run_dir / "attempt.json")
    sealing = read_sealing_record(run_dir)
    summary = _archived_summary(attempt_id, record, held=held, sealing=sealing)
    if sealing is not None:
        manifest = [dict(entry) for entry in sealing["manifest"]]
    else:
        manifest = _manifest(directories, attempt_id)
    return {"attempt": summary, "manifest": manifest}


def _archive_directories(data_dir: Path) -> list[Path]:
    """Every archive run directory, or ``[]`` when the volume has none."""

    try:
        issue_dirs = sorted(path for path in (data_dir / "logs").iterdir() if path.is_dir())
    except OSError:
        return []
    run_dirs: list[Path] = []
    for issue_dir in issue_dirs:
        try:
            run_dirs.extend(path for path in issue_dir.iterdir() if path.is_dir())
        except OSError:
            continue
    return run_dirs


def _manifest(directories: list[Path], attempt_id: str) -> list[dict[str, Any]]:
    """Hash every public evidence file of one Attempt, sorted by path."""

    entries: list[dict[str, Any]] = []
    for run_dir in directories:
        if _read_record(run_dir / "attempt.json").get("started_at") != attempt_id:
            continue
        try:
            children = sorted(run_dir.rglob("*"))
        except OSError:
            continue
        for path in children:
            relative = path.relative_to(run_dir).as_posix()
            if not _is_public_evidence(path, relative):
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            entries.append(
                {
                    "kind": _manifest_kind(relative),
                    "media_type": _media_type(relative),
                    "path": relative,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size": len(raw),
                }
            )
    entries.sort(key=lambda entry: entry["path"])
    return entries


def _is_public_evidence(path: Path, relative: str) -> bool:
    """Whether an archive file may appear in a manifest (never private state)."""

    if path.is_symlink() or not path.is_file():
        return False
    return _is_public_relative(relative)


def _is_public_relative(relative: str) -> bool:
    """Whether one archive-relative POSIX path may appear in a manifest."""

    if relative == SEALING_FILENAME:
        return False
    lowered = relative.lower()
    if any(part.startswith(".") for part in relative.split("/")):
        return False
    if lowered.endswith(".tmp"):
        return False
    name = Path(relative).name
    stemmed = name.lower()
    if stemmed.endswith((".sqlite3", ".sqlite3-wal", ".sqlite3-shm", ".sqlite", ".db")):
        return False
    if "credential" in stemmed or "secret" in stemmed:
        return False
    return True


def _manifest_kind(relative: str) -> str:
    """Map one archive path onto its contract manifest kind."""

    name = Path(relative).name
    if name in ("attempt.json", "agent_output.json"):
        return "record"
    if name == "events.jsonl":
        return "events"
    if name.endswith("_tool_call.json"):
        return "tool_call"
    if name.endswith("_tool_result.json"):
        return "tool_result"
    if name.endswith("_model_response.txt"):
        return "model_output"
    if name.endswith("_stdout.log") or name.endswith("_stderr.log"):
        return "command_output"
    if relative.startswith("transcripts/") and name.endswith(".jsonl"):
        return "transcript"
    return "other"


def _media_type(relative: str) -> str:
    name = Path(relative).name
    if name == "agent_output.json":
        return "application/x-ndjson"
    suffix = Path(relative).suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix in (".jsonl", ".ndjson"):
        return "application/x-ndjson"
    if suffix in (".log", ".txt"):
        return "text/plain"
    return "application/octet-stream"


def _collect_entries(
    data_dir: Path,
    *,
    active: ActiveAttempt | None,
    held: Container[str],
) -> list[tuple[str, dict[str, Any]]]:
    """Collect ``(attempt_id, summary)`` pairs, newest first."""

    archived = _read_archived_summaries(data_dir, held=held)
    if active is not None:
        archived[active.attempt_id] = _live_summary(active, held=held)
    return sorted(archived.items(), key=lambda item: _sort_key(item[0]), reverse=True)


def _read_archived_summaries(
    data_dir: Path, *, held: Container[str]
) -> dict[str, dict[str, Any]]:
    """Read one summary per archive directory carrying a recorded ID."""

    summaries: dict[str, dict[str, Any]] = {}
    try:
        issue_dirs = sorted(path for path in (data_dir / "logs").iterdir() if path.is_dir())
    except OSError:
        return summaries
    for issue_dir in issue_dirs:
        try:
            run_dirs = sorted(path for path in issue_dir.iterdir() if path.is_dir())
        except OSError:
            continue
        for run_dir in run_dirs:
            record = _read_record(run_dir / "attempt.json")
            started_at = record.get("started_at")
            if not isinstance(started_at, str) or not started_at:
                continue
            summaries[started_at] = _archived_summary(
                started_at, record, held=held, sealing=read_sealing_record(run_dir)
            )
    return summaries


def _live_summary(active: ActiveAttempt, *, held: Container[str]) -> dict[str, Any]:
    """Build the moving-snapshot summary of the active Attempt."""

    return _summary(
        attempt_id=active.attempt_id,
        issue_number=active.issue_number,
        issue_title=active.issue_title,
        started_at=active.attempt_id,
        finished_at=None,
        lifecycle_state=active.phase,
        record={},
        held=held,
    )


def _archived_summary(
    attempt_id: str,
    record: Mapping[str, Any],
    *,
    held: Container[str],
    sealing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the summary of one archived Attempt.

    Unsealed archives read as a moving snapshot (``sealed: False``); a
    present sealing record freezes ``sealed``/``sealed_at`` and the
    manifest. The lifecycle state comes from the archive record
    (``released`` for a release without a Terminal Outcome) and defaults
    to ``finished``.
    """

    issue_number = record.get("issue_number")
    if isinstance(issue_number, bool) or not isinstance(issue_number, int):
        issue_number = None
    issue_title = record.get("issue_title")
    if not isinstance(issue_title, str) or not issue_title:
        issue_title = None
    finished_at = record.get("completed_at")
    if not isinstance(finished_at, str) or not finished_at:
        finished_at = None
    lifecycle_state = record.get("lifecycle_state")
    if lifecycle_state not in ("finished", "released"):
        lifecycle_state = "finished"
    sealed_at: str | None = None
    if sealing is not None:
        candidate = sealing.get("sealed_at")
        sealed_at = candidate if isinstance(candidate, str) and candidate else None
    return _summary(
        attempt_id=attempt_id,
        issue_number=issue_number,
        issue_title=issue_title,
        started_at=attempt_id,
        finished_at=finished_at,
        lifecycle_state=lifecycle_state,
        record=record,
        held=held,
        sealed=sealing is not None,
        sealed_at=sealed_at,
    )


def _summary(
    *,
    attempt_id: str,
    issue_number: int | None,
    issue_title: str | None,
    started_at: str,
    finished_at: str | None,
    lifecycle_state: str,
    record: Mapping[str, Any],
    held: Container[str],
    sealed: bool = False,
    sealed_at: str | None = None,
) -> dict[str, Any]:
    """Build one contract AttemptSummary in envelope key order."""

    return {
        "attempt_id": attempt_id,
        "attention_required": attempt_id in held,
        "delivery": _delivery(record),
        "finished_at": finished_at,
        "issue": {"number": issue_number, "title": issue_title},
        "lifecycle_state": lifecycle_state,
        "outcome": _outcome(record),
        "sealed": sealed,
        "sealed_at": sealed_at,
        "started_at": started_at,
    }


def _outcome(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The recorded native outcome, or ``None`` while the Attempt runs."""

    native = record.get("outcome")
    if hasattr(native, "value"):
        native = native.value  # type: ignore[union-attr]
    if not isinstance(native, str) or not native:
        return None
    reason = record.get("outcome_reason")
    if not isinstance(reason, str) or not reason:
        reason = None
    return {"label": _outcome_label(native), "native": native, "reason": reason}


def _outcome_label(native: str) -> str:
    """Backend display text for one native outcome."""

    return {
        "complete": "Complete",
        "incomplete": "Incomplete",
        "infrastructure_error": "Infrastructure error",
        "no_changes": "No changes",
        "handoff": "Handoff",
    }.get(native, native)


def _delivery(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The delivered pull request, or ``None`` when there is none."""

    candidate = record.get("pull_request")
    if candidate is None:
        return None
    if isinstance(candidate, Mapping):
        number, url = candidate.get("number"), candidate.get("url")
    else:
        number, url = getattr(candidate, "number", None), getattr(candidate, "url", None)
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        return None
    if not isinstance(url, str) or not url:
        return None
    return {"pull_request": {"number": number, "url": url}}


def _read_record(path: Path) -> Mapping[str, Any]:
    """Return the JSON object at ``path``, or ``{}`` when unreadable."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, Mapping) else {}


def read_sealing_record(run_dir: Path) -> dict[str, Any] | None:
    """Return the sealing record of one archive directory, or ``None``.

    The record carries the frozen ``sealed_at`` timestamp and the frozen
    ``manifest`` list. A missing or unreadable record means the archive is
    not Sealed; a malformed one is treated the same way (never raising),
    so a half-written seal can never fake finality.
    """

    try:
        value = json.loads((run_dir / SEALING_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    sealed_at = value.get("sealed_at")
    manifest = value.get("manifest")
    if not isinstance(sealed_at, str) or not sealed_at:
        return None
    if not isinstance(manifest, list) or not all(
        isinstance(entry, dict) for entry in manifest
    ):
        return None
    return {"sealed_at": sealed_at, "manifest": [dict(entry) for entry in manifest]}


def find_run_directory(data_dir: Path, attempt_id: str) -> Path | None:
    """Return the archive run directory of one Attempt ID, or ``None``.

    Matched through the ``started_at`` recorded in ``attempt.json``, never
    by string reversal of the directory name.
    """

    for run_dir in _archive_directories(data_dir):
        if _read_record(run_dir / "attempt.json").get("started_at") == attempt_id:
            return run_dir
    return None


def seal_archive(run_dir: Path) -> dict[str, Any] | None:
    """Seal one archive directory; set-once, and never raising.

    Computes the public manifest over the directory, then writes the
    sealing record (``sealed_at`` plus the frozen manifest) atomically
    (temporary file plus fsync plus rename) and forgets the process-local
    write-through snapshot, so later exports serve the frozen manifest.
    When the archive is already Sealed the stored record is returned
    unchanged: ``sealed_at`` is set once. Returns the sealing record, or
    ``None`` when the directory is missing or the seal cannot be written
    (the caller logs a warning and carries on unsealed).
    """

    existing = read_sealing_record(run_dir)
    if existing is not None:
        return existing
    if not run_dir.is_dir():
        return None
    try:
        children = sorted(run_dir.rglob("*"))
    except OSError:
        return None
    entries: list[dict[str, Any]] = []
    for path in children:
        relative = path.relative_to(run_dir).as_posix()
        if not _is_public_evidence(path, relative):
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        entries.append(
            {
                "kind": _manifest_kind(relative),
                "media_type": _media_type(relative),
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }
        )
    entries.sort(key=lambda entry: entry["path"])
    sealed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    record = {"sealed_at": sealed_at, "manifest": entries}
    payload = json.dumps(record, sort_keys=True)
    target = run_dir / SEALING_FILENAME
    temporary: Path | None = None
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / f".{SEALING_FILENAME}.tmp", "w", encoding="utf-8") as file:
            temporary = Path(file.name)
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, target)
    except OSError:
        return None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    discard_manifest_snapshot(run_dir)
    return read_sealing_record(run_dir) or record


def export_projection(
    data_dir: Path,
    attempt_id: str,
    *,
    active: ActiveAttempt | None = None,
    held: Container[str] = frozenset(),
) -> dict[str, Any]:
    """Return ``{"attempt": AttemptSummary, "manifest": [ManifestEntry]}`` for export.

    The summary matches :func:`show_attempt`; the manifest is the frozen
    sealing record for a Sealed archive, and a snapshot-merged manifest
    for an unsealed one: a fresh scan overlaid with the process-local
    write-through snapshot's ``(size, sha256)`` values, so a file that
    grows through a side channel after the snapshot keeps the snapshot's
    entry while the streamed bytes are read live (a torn read the
    consumer detects as a ``sha256`` mismatch). Snapshot entries for
    files that are gone or no longer public are dropped. Raises
    :class:`AttemptHistoryError` with ``invalid_argument``/``not_found``
    like :func:`show_attempt`.
    """

    shown = show_attempt(data_dir, attempt_id, active=active, held=held)
    if shown["attempt"]["sealed"]:
        return shown
    run_dir = find_run_directory(data_dir, attempt_id)
    if run_dir is None:
        return shown
    live = {entry["path"]: entry for entry in shown["manifest"]}
    snapshot = read_manifest_snapshot(run_dir)
    if not snapshot:
        return shown
    merged: list[dict[str, Any]] = []
    for path, entry in live.items():
        overlay = snapshot.get(path)
        if overlay is not None:
            size, sha256 = overlay
            if isinstance(size, int) and isinstance(sha256, str):
                entry = {**entry, "size": size, "sha256": sha256}
        merged.append(entry)
    merged.sort(key=lambda entry: entry["path"])
    return {"attempt": shown["attempt"], "manifest": merged}


def write_export_tar(
    fileobj: BinaryIO,
    *,
    attempt: Mapping[str, Any],
    entries: list[Mapping[str, Any]],
    read_bytes: Callable[[str], bytes],
) -> None:
    """Stream one uncompressed pax export tar: ``manifest.json`` first.

    The manifest member carries ``{"contract": "1.0", "attempt",
    "files"}``; every file member that follows is the bytes actually
    read (its tar header size is the live byte count), so for an unsealed
    archive a file that grew after the snapshot surfaces as a ``sha256``
    mismatch on the consumer side. The stream is deterministic for a
    Sealed archive (pax format, zero mtime, fixed ids/mode, no extra pax
    headers), hence byte-identical across reads. Raises ``KeyError`` when
    ``read_bytes`` has no bytes for a listed path.
    """

    manifest_doc = {
        "contract": "1.0",
        "attempt": dict(attempt),
        "files": [dict(entry) for entry in entries],
    }
    payload = (json.dumps(manifest_doc, sort_keys=True) + "\n").encode("utf-8")

    def _tarinfo(name: str, size: int) -> tarfile.TarInfo:
        info = tarfile.TarInfo(name)
        info.size = size
        info.mtime = 0
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mode = 0o444
        info.pax_headers = {}
        return info

    with tarfile.open(fileobj=fileobj, mode="w|", format=tarfile.PAX_FORMAT) as archive:
        archive.addfile(_tarinfo("manifest.json", len(payload)), io.BytesIO(payload))
        for entry in entries:
            path = str(entry["path"])
            data = read_bytes(path)
            archive.addfile(_tarinfo(path, len(data)), io.BytesIO(data))


def _sort_key(value: str) -> Any:
    """Order Attempt IDs newest last; unparseable IDs sort as the oldest."""

    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except (ValueError, AttributeError):
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
