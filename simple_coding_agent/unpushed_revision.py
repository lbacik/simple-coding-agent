"""Durable Unpushed-revision records for failed implementation-attempt pushes.

When an implementation attempt's branch push fails (retries ran out, or the
remote moved since the claim), the attempt's local commits are the only copy
of that work. The publisher stores an :class:`UnpushedRevisionRecord` for the
issue in the Worker's own data directory, next to its other attempt state.
On that Worker's next live implementation claim of the same issue the
lifecycle publishes the recorded Unpushed revision when the remote branch
still stands at the recorded claim-time lease, and otherwise keeps it as a
Preserved revision under a local ref for the operator, so the next attempt's
preparation never silently orphans it.

Glossary (``CONTEXT.md``): an **Unpushed revision** is the local
attempt-branch tip a failed push could not publish; a **Preserved revision**
is an Unpushed revision kept under a local reference because it could not be
safely published at the next claim. Recovering one never changes the failed
attempt's Recorded verdict (ADR 0010).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile


#: Outcome of a recovery try that pushed (or found already pushed) the
#: recorded Unpushed revision.
RECOVERY_RECOVERED = "recovered"
#: Outcome of a recovery try that found the remote already holding the
#: recorded Unpushed revision, so nothing was pushed.
RECOVERY_ALREADY_PUBLISHED = "already_published"
#: Outcome of a recovery try that kept the revision under a Preserved ref.
RECOVERY_PRESERVED = "preserved"

RECOVERY_OUTCOMES = frozenset(
    {RECOVERY_RECOVERED, RECOVERY_ALREADY_PUBLISHED, RECOVERY_PRESERVED}
)


class UnpushedRevisionStoreError(RuntimeError):
    """Raised when the durable Unpushed-revision records cannot be used safely."""


@dataclass(frozen=True)
class UnpushedRevisionRecord:
    """One issue's Unpushed revision, recorded where its push failed.

    ``revision`` is the full sha of the local branch tip the failed push
    tried to publish. ``lease_revision`` is that attempt's claim-time lease
    (``None`` when the attempt branch was absent at claim time); a record
    always carries a known lease, never an unknown one. ``attempt_id`` is
    the failed attempt's checkpoint ``started_at``.
    """

    issue_number: int
    revision: str
    lease_revision: str | None
    attempt_id: str


@dataclass(frozen=True)
class UnpushedRecovery:
    """The outcome of one recovery try, for events and the result comment."""

    outcome: str
    revision: str
    source_attempt_id: str
    ref: str | None = None
    reason: str | None = None
    comment_line: str = ""


class UnpushedRevisionStore:
    """Atomically persist one Unpushed-revision record per issue.

    A newer record for the same issue replaces the old one. The store lives
    next to the other attempt state (``state/unpushed_revisions.json`` under
    the data directory) and survives restarts.
    """

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "state" / "unpushed_revisions.json"

    def read(self, issue_number: int) -> UnpushedRevisionRecord | None:
        """Return the record for ``issue_number``, or ``None`` when absent."""

        return self.read_all().get(issue_number)

    def read_all(self) -> dict[int, UnpushedRevisionRecord]:
        """Return every record keyed by issue number."""

        try:
            raw = self._path.read_text()
        except FileNotFoundError:
            return {}
        except OSError as error:
            raise UnpushedRevisionStoreError(
                "Unpushed-revision records could not be read"
            ) from error
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise UnpushedRevisionStoreError(
                "Unpushed-revision records are malformed"
            ) from error
        if not isinstance(value, dict):
            raise UnpushedRevisionStoreError(
                "Unpushed-revision records are malformed"
            )
        records: dict[int, UnpushedRevisionRecord] = {}
        for key, entry in value.items():
            try:
                issue_number = _issue_from_json_key(key)
                records[issue_number] = _record_from_json(issue_number, entry)
            except (TypeError, ValueError) as error:
                raise UnpushedRevisionStoreError(
                    "Unpushed-revision records are malformed"
                ) from error
        return records

    def save(self, record: UnpushedRevisionRecord) -> None:
        """Store ``record``, replacing any older record for the same issue."""

        try:
            _validate_record(record)
        except ValueError as error:
            raise UnpushedRevisionStoreError(
                f"Unpushed-revision record is invalid: {error}"
            ) from error
        records = self.read_all()
        records[record.issue_number] = record
        self._write(records)

    def clear(self, issue_number: int) -> bool:
        """Drop the record for ``issue_number``; return True when one existed."""

        records = self.read_all()
        if issue_number not in records:
            return False
        del records[issue_number]
        self._write(records)
        return True

    def _write(self, records: dict[int, UnpushedRevisionRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=".unpushed-revisions-",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary = Path(file.name)
                json.dump(
                    {
                        str(issue_number): {
                            "revision": record.revision,
                            "lease_revision": record.lease_revision,
                            "attempt_id": record.attempt_id,
                        }
                        for issue_number, record in sorted(records.items())
                    },
                    file,
                    sort_keys=True,
                )
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self._path)
        except OSError as error:
            raise UnpushedRevisionStoreError(
                "Unpushed-revision records could not be written"
            ) from error
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def preserved_ref_for_issue_attempt(issue_number: int, attempt_id: str) -> str:
    """The deterministic Preserved-revision ref for one failed attempt.

    ``refs/agent/preserved/issue-<n>/<attempt-id>`` in a dedicated,
    non-branch namespace that names the issue and the failed attempt id, so
    an operator can find the kept commits. The attempt id (a ``started_at``
    timestamp) carries ref-unsafe characters such as ``:``, so anything
    outside ``[A-Za-z0-9._-]`` is replaced with ``-``; the scheme stays
    deterministic for a given ``(issue_number, attempt_id)`` pair.
    """

    if isinstance(issue_number, bool) or not isinstance(issue_number, int) or issue_number <= 0:
        raise ValueError(f"Invalid issue number {issue_number!r}")
    if not isinstance(attempt_id, str) or not attempt_id.strip():
        raise ValueError(f"Invalid attempt id {attempt_id!r}")
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "-", attempt_id.strip()).strip(".-")
    if not sanitized:
        raise ValueError(f"Invalid attempt id {attempt_id!r}")
    return f"refs/agent/preserved/issue-{issue_number}/{sanitized}"


def recovered_comment_line(revision: str, attempt_id: str, *, already_published: bool = False) -> str:
    """The result-comment line for a recovered Unpushed revision."""

    line = f"**Recovered unpushed revision**: `{revision}` from attempt `{attempt_id}`"
    if already_published:
        line += " (already published)"
    return line


def preserved_comment_line(revision: str, attempt_id: str, ref: str, reason: str) -> str:
    """The result-comment line for a Preserved revision."""

    return (
        f"**Preserved unpushed revision**: `{revision}` from attempt `{attempt_id}`"
        f" at `{ref}` ({reason})"
    )


def _validate_record(record: UnpushedRevisionRecord) -> None:
    if (
        isinstance(record.issue_number, bool)
        or not isinstance(record.issue_number, int)
        or record.issue_number <= 0
    ):
        raise ValueError("issue number must be a positive integer")
    _validate_revision(record.revision, "revision")
    if record.lease_revision is not None:
        _validate_revision(record.lease_revision, "lease revision")
    if (
        not isinstance(record.attempt_id, str)
        or not record.attempt_id.strip()
        or any(character.isspace() for character in record.attempt_id)
    ):
        raise ValueError("attempt id must be a non-empty string without whitespace")


def _validate_revision(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{name} must be a non-empty string without whitespace")


def _issue_from_json_key(key: object) -> int:
    if not isinstance(key, str) or not key.isdigit():
        raise ValueError("invalid issue number")
    issue_number = int(key)
    if issue_number <= 0:
        raise ValueError("invalid issue number")
    return issue_number


def _record_from_json(issue_number: int, value: object) -> UnpushedRevisionRecord:
    if not isinstance(value, dict) or set(value) != {
        "revision",
        "lease_revision",
        "attempt_id",
    }:
        raise ValueError("invalid record shape")
    record = UnpushedRevisionRecord(
        issue_number=issue_number,
        revision=value["revision"],
        lease_revision=value["lease_revision"],
        attempt_id=value["attempt_id"],
    )
    _validate_record(record)
    return record
