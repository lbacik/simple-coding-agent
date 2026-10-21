"""The single discoverable handoff record per implementation issue (issue #282).

A handoff publishes two comments while the additive rollout lasts: the
existing detailed per-attempt result comment (legacy bookkeeping, unchanged)
and one short shared comment linking to the current Handoff note on the
attempt branch. Later handoffs update that same shared comment instead of
creating another one; the comment is located through its explicit versioned
machine marker, never by loose text match.

The visible text links to the note through the branch (``blob/<branch>/...``)
so it keeps displaying subsequent committed replacements; the hidden marker
additionally identifies the immutable published branch and note revisions,
the published attempt ID, the claim-time requeue source, and the
automatic-continuation standing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile

from simple_coding_agent.recovery import (
    HANDOFF_RECORD_MARKER_PREFIX,
    HANDOFF_RECORD_VERSION,
    REQUEUE_SOURCES,
)


@dataclass(frozen=True)
class HandoffRecord:
    """The hidden versioned payload of the shared handoff comment."""

    version: int
    repository: str
    issue_number: int
    attempt_id: str
    requeue_source: str
    continuation_count: int
    branch: str
    branch_sha: str
    note_sha: str


def handoff_record_marker(record: HandoffRecord) -> str:
    """Build the hidden versioned machine marker for a handoff record."""

    _validate_record(record)
    return (
        f"{HANDOFF_RECORD_MARKER_PREFIX} "
        f"v={record.version} "
        f"repo={record.repository} "
        f"issue={record.issue_number} "
        f"attempt={record.attempt_id} "
        f"source={record.requeue_source} "
        f"count={record.continuation_count} "
        f"branch={record.branch} "
        f"branch_sha={record.branch_sha} "
        f"note_sha={record.note_sha} -->"
    )


def parse_handoff_record(body: str) -> HandoffRecord | None:
    """Return the handoff record carried by a comment body, if any.

    Unknown keys are preserved for forward compatibility by ignoring them;
    a body without a well-formed marker reads as ``None``.
    """

    start = body.find(HANDOFF_RECORD_MARKER_PREFIX)
    if start < 0:
        return None
    end = body.find("-->", start)
    if end < 0:
        return None
    fields: dict[str, str] = {}
    for token in body[start + len(HANDOFF_RECORD_MARKER_PREFIX) : end].split():
        key, separator, value = token.partition("=")
        if separator and key and value:
            fields[key] = value
    try:
        record = HandoffRecord(
            version=int(fields["v"]),
            repository=fields["repo"],
            issue_number=int(fields["issue"]),
            attempt_id=fields["attempt"],
            requeue_source=fields["source"],
            continuation_count=int(fields["count"]),
            branch=fields["branch"],
            branch_sha=fields["branch_sha"],
            note_sha=fields["note_sha"],
        )
    except (KeyError, ValueError):
        return None
    try:
        _validate_record(record)
    except ValueError:
        return None
    return record


def records_match(first: HandoffRecord, second: HandoffRecord) -> bool:
    """Whether two parsed records describe the same published handoff."""

    return first == second


def is_newer_attempt(candidate: HandoffRecord, baseline: HandoffRecord) -> bool:
    """Whether ``candidate`` was published by a later attempt than ``baseline``.

    Attempt IDs are the checkpoints' UTC ``started_at`` stamps, which sort
    chronologically in string order; a later claim always carries a greater
    ID because claims are serialized and timestamps strictly increase.
    """

    return candidate.attempt_id > baseline.attempt_id


def handoff_note_blob_url(repository: str, branch: str, issue_number: int) -> str:
    """The branch-following URL of an issue's Handoff note file."""

    return (
        f"https://github.com/{repository}/blob/{branch}"
        f"/.agent/handoff/{issue_number}.md"
    )


def build_handoff_record_body(record: HandoffRecord) -> str:
    """Build the short shared comment: one link plus the hidden marker.

    The body deliberately carries none of the legacy headers or markers, so
    the legacy attempt-result readers never mistake it for another
    per-attempt result comment, and the model-facing trusted comments skip it.
    """

    marker = handoff_record_marker(record)
    url = handoff_note_blob_url(record.repository, record.branch, record.issue_number)
    return (
        "## Handoff note\n\n"
        "The current Handoff note for this issue is preserved on"
        f" `{record.branch}`:"
        f" [`.agent/handoff/{record.issue_number}.md`]({url}).\n\n"
        f"{marker}"
    )


def _validate_record(record: HandoffRecord) -> None:
    if record.version != HANDOFF_RECORD_VERSION:
        raise ValueError(f"Unsupported handoff record version {record.version!r}")
    if not isinstance(record.repository, str) or not record.repository.strip():
        raise ValueError("Repository must be a non-empty string")
    if (
        isinstance(record.issue_number, bool)
        or not isinstance(record.issue_number, int)
        or record.issue_number <= 0
    ):
        raise ValueError("Issue number must be a positive integer")
    for name in ("attempt_id", "branch", "branch_sha", "note_sha"):
        value = getattr(record, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        if any(character.isspace() for character in value):
            raise ValueError(f"{name} must not contain whitespace")
    if record.requeue_source not in REQUEUE_SOURCES:
        raise ValueError(f"Invalid requeue source {record.requeue_source!r}")
    if (
        isinstance(record.continuation_count, bool)
        or not isinstance(record.continuation_count, int)
        or record.continuation_count < 0
    ):
        raise ValueError("Continuation count must be a non-negative integer")


class HandoffPublicationStoreError(RuntimeError):
    """Raised when the durable handoff-record intent cannot be used safely."""


class MemoryHandoffPublicationStore:
    """Process-local handoff-record intents with the durable store's first-wins rule.

    Used when no data directory is wired (unit tests, legacy callers): remote
    idempotency still holds within the process, but a restart cannot reuse
    the intent. Production always wires :class:`HandoffPublicationStore`.
    """

    def __init__(self) -> None:
        self._records: dict[str, HandoffRecord] = {}

    def load(self, attempt_id: str) -> HandoffRecord | None:
        """Return the recorded intent for an attempt, if any."""

        return self._records.get(attempt_id)

    def read_all(self) -> dict[str, HandoffRecord]:
        """Return every recorded intent keyed by attempt ID."""

        return dict(self._records)

    def save_if_absent(self, record: HandoffRecord) -> HandoffRecord:
        """Record the intent unless this attempt already has one; return the winner."""

        try:
            _validate_record(record)
        except ValueError as error:
            raise HandoffPublicationStoreError(
                f"Handoff record is invalid: {error}"
            ) from error
        return self._records.setdefault(record.attempt_id, record)


class HandoffPublicationStore:
    """Durably persist the intended shared-record payload before publication.

    The first intent saved for an attempt ID wins: a retry after a timeout or
    a restart reuses the stored attempt and continuation standing instead of
    recomputing (and incrementing) the count. Intents are kept after success
    so a repeated confirmation still compares against the published record.
    """

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "state" / "handoff_records.json"

    def load(self, attempt_id: str) -> HandoffRecord | None:
        """Return the persisted intent for an attempt, if any."""

        return self.read_all().get(attempt_id)

    def read_all(self) -> dict[str, HandoffRecord]:
        """Return every persisted intent keyed by attempt ID."""

        try:
            raw = self._path.read_text()
        except FileNotFoundError:
            return {}
        except OSError as error:
            raise HandoffPublicationStoreError(
                "Handoff record store could not be read"
            ) from error
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise HandoffPublicationStoreError(
                "Handoff record store is malformed"
            ) from error
        if not isinstance(value, dict):
            raise HandoffPublicationStoreError("Handoff record store is malformed")
        records: dict[str, HandoffRecord] = {}
        for attempt_id, entry in value.items():
            try:
                records[attempt_id] = _record_from_json(attempt_id, entry)
            except (TypeError, ValueError) as error:
                raise HandoffPublicationStoreError(
                    "Handoff record store is malformed"
                ) from error
        return records

    def save_if_absent(self, record: HandoffRecord) -> HandoffRecord:
        """Persist the intent unless this attempt already has one; return the winner."""

        try:
            _validate_record(record)
        except ValueError as error:
            raise HandoffPublicationStoreError(
                f"Handoff record is invalid: {error}"
            ) from error
        records = self.read_all()
        existing = records.get(record.attempt_id)
        if existing is not None:
            return existing
        records[record.attempt_id] = record
        self._write(records)
        return record

    def _write(self, records: dict[str, HandoffRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=".handoff-records-",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary = Path(file.name)
                json.dump(
                    {
                        attempt_id: {
                            "version": item.version,
                            "repository": item.repository,
                            "issue_number": item.issue_number,
                            "requeue_source": item.requeue_source,
                            "continuation_count": item.continuation_count,
                            "branch": item.branch,
                            "branch_sha": item.branch_sha,
                            "note_sha": item.note_sha,
                        }
                        for attempt_id, item in sorted(records.items())
                    },
                    file,
                    sort_keys=True,
                )
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self._path)
        except OSError as error:
            raise HandoffPublicationStoreError(
                "Handoff record store could not be written"
            ) from error
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def _record_from_json(attempt_id: object, value: object) -> HandoffRecord:
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("invalid attempt id")
    if not isinstance(value, dict) or set(value) != {
        "version",
        "repository",
        "issue_number",
        "requeue_source",
        "continuation_count",
        "branch",
        "branch_sha",
        "note_sha",
    }:
        raise ValueError("invalid handoff record shape")
    record = HandoffRecord(
        version=value["version"],
        repository=value["repository"],
        issue_number=value["issue_number"],
        attempt_id=attempt_id,
        requeue_source=value["requeue_source"],
        continuation_count=value["continuation_count"],
        branch=value["branch"],
        branch_sha=value["branch_sha"],
        note_sha=value["note_sha"],
    )
    _validate_record(record)
    return record
