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
from collections.abc import Container, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


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
    else:
        record: Mapping[str, Any] | None = None
        for run_dir in directories:
            candidate = _read_record(run_dir / "attempt.json")
            if candidate.get("started_at") == attempt_id:
                record = candidate
                break
        if record is None:
            raise AttemptHistoryError(
                "not_found", f"unknown attempt {attempt_id!r}"
            )
        summary = _archived_summary(attempt_id, record, held=held)
    return {"attempt": summary, "manifest": _manifest(directories, attempt_id)}


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
            summaries[started_at] = _archived_summary(started_at, record, held=held)
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
    attempt_id: str, record: Mapping[str, Any], *, held: Container[str]
) -> dict[str, Any]:
    """Build the moving-snapshot summary of one archived Attempt."""

    issue_number = record.get("issue_number")
    if isinstance(issue_number, bool) or not isinstance(issue_number, int):
        issue_number = None
    issue_title = record.get("issue_title")
    if not isinstance(issue_title, str) or not issue_title:
        issue_title = None
    finished_at = record.get("completed_at")
    if not isinstance(finished_at, str) or not finished_at:
        finished_at = None
    return _summary(
        attempt_id=attempt_id,
        issue_number=issue_number,
        issue_title=issue_title,
        started_at=attempt_id,
        finished_at=finished_at,
        lifecycle_state="finished",
        record=record,
        held=held,
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
        "sealed": False,
        "sealed_at": None,
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


def _sort_key(value: str) -> Any:
    """Order Attempt IDs newest last; unparseable IDs sort as the oldest."""

    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except (ValueError, AttributeError):
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
