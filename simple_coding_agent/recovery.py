"""Operator recovery for retained implementation attempts (issue #84).

A retained attempt is a checkpoint whose durable completion boundary was
never reached: its outcome was not published with a confirmed comment, the
issue was not released, local cleanup is not durable, or accounting never
committed. While such a checkpoint (or unexplained dirty work) exists,
intake stays blocked and ``resume`` is rejected.

This module holds the small, testable recovery helpers. The lifecycle owns
the serialized execution; the control store owns the durable command
record; the CLI and socket only transport requests.
"""

from __future__ import annotations

from collections.abc import Container, Iterable
import re


class RecoveryRejectedError(ValueError):
    """Raised when a recovery command is rejected without a control change.

    Unknown attempt IDs, mismatched identities, already-finalized attempts,
    concurrently changing attempts, and invalid ``--saved-at`` values leave
    intake and the command record unchanged.
    """


def parse_attempt_id(value: object) -> str:
    """Validate a recovery ``<attempt-id>``; raise without side effects."""

    if not isinstance(value, str) or not value.strip():
        raise RecoveryRejectedError(
            f"Invalid attempt ID {value!r}: must be the stable attempt identity"
            " (the checkpoint's started_at). No control change was accepted."
        )
    if len(value) > 256:
        raise RecoveryRejectedError(
            f"Invalid attempt ID {value!r}: must be at most 256 characters."
            " No control change was accepted."
        )
    return value.strip()


def parse_saved_at(value: object) -> str:
    """Validate a ``--saved-at <path-or-url>`` reference; raise without side effects."""

    if not isinstance(value, str) or not value.strip():
        raise RecoveryRejectedError(
            f"Invalid --saved-at value {value!r}: must be a non-empty path or URL"
            " where the retained work was secured. No control change was accepted."
        )
    if len(value) > 1024:
        raise RecoveryRejectedError(
            f"Invalid --saved-at value {value!r}: must be at most 1024 characters."
            " No control change was accepted."
        )
    return value.strip()


def attempt_marker(attempt_id: str) -> str:
    """The durable issue-comment marker identifying one attempt's result."""

    return f"<!-- agent-attempt: {attempt_id} -->"


#: The claim-time reason an attempt was requeued, recorded as a hidden HTML
#: marker in every attempt-result comment. ``human`` means the issue carried
#: ``ready-for-agent`` when it was claimed; ``automatic`` means it carried
#: only ``round-finished`` (an automatic continuation without a human requeue);
#: ``conflict`` means it carried ``merge-conflict`` (a conflict requeue that
#: only brings an already-published branch up to date).
REQUEUE_SOURCE_HUMAN = "human"
REQUEUE_SOURCE_AUTOMATIC = "automatic"
REQUEUE_SOURCE_CONFLICT = "conflict"

REQUEUE_SOURCES = frozenset(
    {REQUEUE_SOURCE_HUMAN, REQUEUE_SOURCE_AUTOMATIC, REQUEUE_SOURCE_CONFLICT}
)


def requeue_source_marker(source: str) -> str:
    """The durable issue-comment marker recording one attempt's requeue source."""

    if source not in REQUEUE_SOURCES:
        raise ValueError(
            f"Invalid requeue source {source!r}: must be 'human', 'automatic', or 'conflict'"
        )
    return f"<!-- requeue-source: {source} -->"


def parse_requeue_source(body: str) -> str | None:
    """Return the requeue source recorded in a comment body, if any.

    Comments written before the marker existed carry no marker and read as
    ``None`` (a legacy result, counted as a human requeue).
    """

    if requeue_source_marker(REQUEUE_SOURCE_CONFLICT) in body:
        return REQUEUE_SOURCE_CONFLICT
    if requeue_source_marker(REQUEUE_SOURCE_AUTOMATIC) in body:
        return REQUEUE_SOURCE_AUTOMATIC
    if requeue_source_marker(REQUEUE_SOURCE_HUMAN) in body:
        return REQUEUE_SOURCE_HUMAN
    return None


def requeue_source_for_labels(labels: Container[str]) -> str:
    """Decide the requeue source from the claim snapshot's labels.

    ``ready-for-agent`` present means a human requeued the issue; otherwise
    the claim was an automatic continuation from ``round-finished``.
    """

    # The literal matches READY_FOR_AGENT in github_tracker without importing
    # it (this module must stay importable from the tracker and publication).
    return REQUEUE_SOURCE_HUMAN if "ready-for-agent" in labels else REQUEUE_SOURCE_AUTOMATIC


#: The single discoverable handoff record per implementation issue
#: (issue #282). The shared comment carries one short branch-following link
#: plus a hidden versioned marker; it never carries the per-attempt
#: ``agent-attempt`` or ``requeue-source`` markers, keeping it invisible to
#: the legacy attempt-result readers.
HANDOFF_RECORD_VERSION = 1
HANDOFF_RECORD_MARKER_PREFIX = "<!-- handoff-record:"


def is_handoff_record(body: str) -> bool:
    """Whether a comment body carries the shared handoff-record marker."""

    start = body.find(HANDOFF_RECORD_MARKER_PREFIX)
    if start < 0:
        return False
    return body.find("-->", start) >= 0


class AmbiguousContinuationError(ValueError):
    """The shared handoff metadata cannot be trusted for accounting.

    Raised when the trusted comment history carries more than one shared
    handoff record, or a shared body that is malformed or uses an
    unsupported version. Callers must retain the issue for operator repair
    instead of silently granting a fresh budget.
    """


_ATTEMPT_MARKER_RE = re.compile(r"<!--\s*agent-attempt:\s*(.*?)\s*-->")


def parse_comment_attempt_id(body: str) -> str | None:
    """Return the ``<!-- agent-attempt: ... -->`` identity in a comment, if any."""

    match = _ATTEMPT_MARKER_RE.search(body or "")
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _attempt_id_sort_key(value: str) -> tuple[int, object]:
    """Order attempt IDs chronologically; unparseable IDs sort as strings.

    Checkpoint IDs are UTC ``started_at`` stamps, so parsed datetimes order
    correctly even when microseconds are present (plain string comparison
    misorders ``...00Z`` past ``...00.000001Z``). Unparseable IDs fall back
    to string order and never match a well-formed shared record.
    """

    from datetime import datetime

    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except (ValueError, AttributeError):
        return (1, value)
    return (0, parsed)


def _attempt_id_less(first: str, second: str) -> bool:
    return _attempt_id_sort_key(first) < _attempt_id_sort_key(second)


def attempt_id_greater(first: str, second: str) -> bool:
    """Whether one attempt ID orders chronologically after another.

    Public so the shared-record publisher and the recovery hold share the
    same chronological ordering as the accounting resolver.
    """

    return _attempt_id_sort_key(first) > _attempt_id_sort_key(second)


def resolve_automatic_continuation_count(
    bodies_newest_first: tuple[str, ...] | list[str],
    *,
    issue_number: int | None = None,
    repository: str | None = None,
) -> int:
    """Resolve the automatic-continuation standing from the shared record.

    The single trusted shared handoff record (issue #283) is authoritative
    when present: its ``continuation_count`` already encodes the row up to
    its own attempt, and ordinary non-handoff results newer than that
    attempt (in attempt-ID order) are reconciled on top so a newer failure
    or human requeue cannot be hidden by older handoff metadata. An
    overlapping legacy result for the same attempt counts once (the shared
    record wins and the duplicate is ignored). Conflict-attempt results
    stay transparent: they neither consume nor reset the budget.

    With no shared metadata the count initialises from trusted historical
    result comments using the legacy semantics
    (:func:`count_automatic_continuations`), so older issues migrate
    without comment changes. Missing shared metadata alone is not
    ambiguous: it reads as the legacy count.

    More than one trusted shared body, or any trusted shared body that is
    malformed or uses an unsupported version, raises
    :class:`AmbiguousContinuationError`: the history cannot verify the
    standing, so callers must retain the issue instead of granting a fresh
    budget. A record naming a different issue or repository than the
    caller expects is likewise ambiguous.
    """

    shared_bodies = [body for body in bodies_newest_first if is_handoff_record(body)]
    if not shared_bodies:
        return count_automatic_continuations(bodies_newest_first)
    # Local import: handoff_record imports this module for its markers.
    from simple_coding_agent.handoff_record import parse_handoff_record

    records = []
    for body in shared_bodies:
        record = parse_handoff_record(body)
        if record is None:
            raise AmbiguousContinuationError(
                "the shared handoff record is malformed or uses an unsupported version;"
                " the standing cannot be verified"
            )
        records.append(record)
    if len(records) != 1 or len(shared_bodies) != 1:
        raise AmbiguousContinuationError(
            "more than one shared handoff comment is present;"
            " the standing cannot be verified"
        )
    record = records[0]
    if issue_number is not None and record.issue_number != issue_number:
        raise AmbiguousContinuationError(
            "the shared handoff record names a different issue;"
            " the standing cannot be verified"
        )
    if repository is not None and record.repository != repository:
        raise AmbiguousContinuationError(
            "the shared handoff record names a different repository;"
            " the standing cannot be verified"
        )
    shared_attempt = record.attempt_id
    newer_automatics = 0
    for body in bodies_newest_first:
        if is_handoff_record(body):
            continue
        if is_conflict_attempt(body):
            continue
        attempt_id = parse_comment_attempt_id(body)
        if attempt_id is None:
            # Unmarked legacy results predate the shared-record era, which
            # the shared count already summarizes; ignore them.
            continue
        if attempt_id == shared_attempt:
            # Overlapping legacy and shared records for the same attempt
            # count once: the shared record already carries it.
            continue
        if not attempt_id_greater(attempt_id, shared_attempt):
            # Older than the shared record: already summarized by it.
            continue
        if parse_requeue_source(body) != REQUEUE_SOURCE_AUTOMATIC:
            # A newer human requeue (or legacy unmarked result with an ID,
            # or any newer ordinary result) resets the row: only automatics
            # after it count, and the older shared standing is hidden.
            return newer_automatics
        newer_automatics += 1
    return record.continuation_count + newer_automatics


def count_automatic_continuations(bodies_newest_first: tuple[str, ...] | list[str]) -> int:
    """Count automatic-requeue result comments in a row from the newest.

    Stops at the most recent human-requeue result; a legacy unmarked comment
    counts as a human requeue and also stops the count. Conflict-attempt
    result comments are transparent: they are skipped without counting and
    without stopping the row, so a conflict resolution neither consumes nor
    resets the automatic-continuation budget. The shared handoff record is
    likewise transparent: it is skipped without counting and without stopping
    the row, so it never reads as an extra legacy attempt-result comment.
    """

    count = 0
    for body in bodies_newest_first:
        if is_handoff_record(body):
            continue
        if is_conflict_attempt(body):
            continue
        if parse_requeue_source(body) != REQUEUE_SOURCE_AUTOMATIC:
            break
        count += 1
    return count


#: Machine-readable outcomes carried by the conflict-attempt marker. Every
#: conflict attempt posts exactly one result comment stamped with one of
#: these; ``skipped`` names its entry-check reason separately.
CONFLICT_OUTCOME_SUCCESS = "success"
CONFLICT_OUTCOME_SKIPPED = "skipped"
CONFLICT_OUTCOME_REBASE_CONFLICT = "rebase_conflict"
CONFLICT_OUTCOME_SETUP_FAILED = "setup_failed"
CONFLICT_OUTCOME_CHECK_FAILED = "check_failed"
CONFLICT_OUTCOME_PUSH_FAILED = "push_failed"
CONFLICT_OUTCOME_ERROR = "error"
#: The single model resolution session reached its ordinary turn, time, or
#: token limit instead of resolving the conflict (issue #271). An ordinary
#: failure: nothing is pushed and no handoff note is written.
CONFLICT_OUTCOME_MODEL_LIMIT = "model_limit"
#: The model resolution changed the rebased commit count (a squash, drop, or
#: amend of the pull request's own commits) instead of adding fixes only on
#: top (issue #271). An ordinary failure: nothing is pushed.
CONFLICT_OUTCOME_RESOLUTION_FAILED = "resolution_failed"
#: A ``success``-path push that left the remote head unchanged: the rebase
#: applied cleanly and the check passed, but nothing new reached the pull
#: request. It counts as a failure toward the conflict-attempt budget.
CONFLICT_OUTCOME_NO_CHANGE = "no_change"

CONFLICT_OUTCOMES = frozenset(
    {
        CONFLICT_OUTCOME_SUCCESS,
        CONFLICT_OUTCOME_SKIPPED,
        CONFLICT_OUTCOME_REBASE_CONFLICT,
        CONFLICT_OUTCOME_SETUP_FAILED,
        CONFLICT_OUTCOME_CHECK_FAILED,
        CONFLICT_OUTCOME_PUSH_FAILED,
        CONFLICT_OUTCOME_ERROR,
        CONFLICT_OUTCOME_MODEL_LIMIT,
        CONFLICT_OUTCOME_RESOLUTION_FAILED,
        CONFLICT_OUTCOME_NO_CHANGE,
    }
)

#: Entry-check reasons for a ``skipped`` conflict attempt.
CONFLICT_REASON_NO_OPEN_PR = "no_open_pr"
CONFLICT_REASON_DRAFT = "draft"
CONFLICT_REASON_NOT_CONFLICTING = "not_conflicting"
CONFLICT_REASON_BASE_MISMATCH = "base_mismatch"

#: Marker keys reserved for the model resolution follow-up (#271): the
#: conflicted files, the files the resolution changed outside them, and the
#: fix commits it publishes.
_CONFLICT_MARKER_PREFIX = "<!-- conflict-attempt:"


def conflict_attempt_marker(
    outcome: str,
    *,
    sha: str | None = None,
    reason: str | None = None,
    files: tuple[str, ...] | list[str] = (),
    fix_commits: tuple[str, ...] | list[str] = (),
    extra_files: tuple[str, ...] | list[str] = (),
) -> str:
    """The durable issue-comment marker recording one conflict attempt's outcome.

    ``outcome`` is one of :data:`CONFLICT_OUTCOMES`; ``sha`` is the pushed
    rebased revision (``None`` renders as ``none`` when nothing was pushed).
    ``reason`` names the entry-check reason for a ``skipped`` outcome.
    ``files`` names the conflicted files, ``extra_files`` the files the model
    resolution changed outside them, and ``fix_commits`` the fix commits the
    model resolution added on top; each is omitted when empty. Values are
    restricted to marker-safe characters so the comment stays greppable.
    """

    if outcome not in CONFLICT_OUTCOMES:
        raise ValueError(f"Invalid conflict outcome {outcome!r}")
    values = [outcome, sha or "none"]
    if reason:
        values.append(reason)
    values.extend(files)
    values.extend(fix_commits)
    values.extend(extra_files)
    for value in values:
        if not value or any(character.isspace() for character in value):
            raise ValueError(f"Invalid conflict marker value {value!r}")
    parts = [f"outcome={outcome}", f"sha={sha or 'none'}"]
    if reason:
        parts.append(f"reason={reason}")
    if files:
        parts.append(f"files={','.join(files)}")
    if fix_commits:
        parts.append(f"fix_commits={','.join(fix_commits)}")
    if extra_files:
        parts.append(f"extra_files={','.join(extra_files)}")
    return f"{_CONFLICT_MARKER_PREFIX} {' '.join(parts)} -->"


def parse_conflict_attempt(body: str) -> dict[str, str] | None:
    """Return the conflict-attempt marker fields in a comment body, if any.

    Unknown keys are preserved so newer writers stay readable to older
    readers; a body without the marker reads as ``None``.
    """

    start = body.find(_CONFLICT_MARKER_PREFIX)
    if start < 0:
        return None
    end = body.find("-->", start)
    if end < 0:
        return None
    fields: dict[str, str] = {}
    for token in body[start + len(_CONFLICT_MARKER_PREFIX) : end].split():
        key, separator, value = token.partition("=")
        if separator and key and value:
            fields[key] = value
    if "outcome" not in fields:
        return None
    return fields


def is_conflict_attempt(body: str) -> bool:
    """Whether a comment body carries a conflict-attempt marker."""

    return parse_conflict_attempt(body) is not None


#: Hidden marker identifying the conflict-attempt result that escalated its
#: issue to a human. The escalating attempt's own result comment carries it,
#: so the streak stops there: after a human removes ``ready-for-human`` the
#: next poll requeues with the full budget.
CONFLICT_ESCALATION_MARKER = "<!-- conflict-escalation -->"


def is_conflict_escalation(body: str) -> bool:
    """Whether a comment body is the escalation comment of its streak."""

    return CONFLICT_ESCALATION_MARKER in body


def is_conflict_failure(body: str) -> bool:
    """Whether a comment body is a conflict-attempt failure counting toward the budget.

    Failures are an unresolved conflict, a red check, an infrastructure
    error (setup, push, or unexpected error), ``no_change`` (a push that
    left the remote head unchanged), and any unknown conflict outcome
    (including a model-limit or restart-recovery error published as a
    conflict result). A successful push of a new sha, a ``skipped``
    result, and an escalation comment are not failures: they stop the
    streak. A body without a conflict-attempt marker is not a conflict
    failure either (an ordinary attempt result stops the streak).
    """

    if is_conflict_escalation(body):
        return False
    parsed = parse_conflict_attempt(body)
    if parsed is None:
        return False
    outcome = parsed.get("outcome", "")
    if outcome == CONFLICT_OUTCOME_SKIPPED:
        return False
    if outcome == CONFLICT_OUTCOME_SUCCESS:
        # A pushed sha of ``none`` means nothing reached the remote branch,
        # so the attempt changed nothing despite its outcome.
        return parsed.get("sha", "none") in ("", "none")
    return True


def count_conflict_failures(bodies_newest_first: tuple[str, ...] | list[str]) -> int:
    """Count consecutive conflict-attempt failures from the newest comment.

    The streak stops at a successful conflict attempt (it pushed a new
    sha), an ordinary attempt result (an ``agent-attempt`` marker without
    the conflict marker), a ``skipped`` result, or the latest escalation
    comment. Any other comment without a conflict-attempt marker also
    stops the streak. The shared handoff record is transparent instead: it
    is skipped without counting and without stopping the streak, so it never
    reads as an extra legacy attempt-result comment. Only the failure rows
    themselves are counted; the boundary comment that stops the row never is.
    """

    count = 0
    for body in bodies_newest_first:
        if is_conflict_escalation(body):
            break
        if is_handoff_record(body):
            continue
        if not is_conflict_attempt(body):
            break
        if not is_conflict_failure(body):
            break
        count += 1
    return count


def release_comment_body(
    *,
    issue_number: int,
    attempt_id: str,
    actual_outcome: str,
    failure_reason: str,
    saved_at: str,
    branch: str,
    requeue_source: str = REQUEUE_SOURCE_HUMAN,
) -> str:
    """Build the abandonment comment for ``recovery release``.

    The comment states the actual attempt outcome and the failure, records
    where the operator secured the work, and never claims a successful
    handoff or an unpublished branch as a success.
    """

    marker = attempt_marker(attempt_id)
    source_marker = requeue_source_marker(requeue_source)
    return (
        f"## Agent Attempt Result: {actual_outcome}\n\n"
        f"**Branch**: `{branch}` (retained work secured by the operator; not published as a success)\n\n"
        "### Outcome\n"
        f"The attempt `{attempt_id}` for issue #{issue_number} did not complete"
        f" finalization and was explicitly released by the operator.\n\n"
        "### Failure\n"
        f"{failure_reason}\n\n"
        "### Saved work\n"
        f"Retained work was secured at: `{saved_at}`\n\n"
        "This release does not claim a successful handoff and does not"
        " requeue the issue.\n\n"
        f"{marker}\n"
        f"{source_marker}"
    )


def retry_next_action(attempt_id: str) -> str:
    """The operator action clearing a retryable hold."""

    return (
        f"Run `agentctl recovery retry {attempt_id}` to retry the unconfirmed"
        " publication, release, cleanup, and accounting steps; if the retained"
        " work is already secured elsewhere, run"
        f" `agentctl recovery release {attempt_id} --saved-at <path-or-url>` instead."
    )


def release_next_action() -> str:
    """The operator action after securing retained work."""

    return (
        "Secure the retained branch, worktree, and checkpoint first, then run"
        " `agentctl recovery release <attempt-id> --saved-at <path-or-url>`."
    )


def describe_hold(
    *,
    issue_number: int | None,
    attempt_id: str | None,
    phase: str | None,
    reason: str,
) -> str:
    """Render the hold reason naming the affected attempt, for status and resume."""

    if issue_number is not None and attempt_id:
        where = f"issue #{issue_number} (attempt {attempt_id})"
        if phase:
            where += f" (phase {phase})"
        return f"Retained attempt for {where}: {reason}"
    if attempt_id:
        return f"Retained attempt {attempt_id}: {reason}"
    return reason


#: Structural signal of a generated handoff attempt-result comment, modern or
#: legacy: the result header with the ``handoff`` outcome. Markers
#: (``agent-attempt``, ``requeue-source``) are not required, so comments
#: posted before the markers existed are still recognized. Human prose is
#: never matched: the header is only ever rendered by publication.
_HANDOFF_RESULT_HEADER_RE = re.compile(
    r"^#{1,6}\s*Agent Attempt Result:\s*handoff\b", re.MULTILINE
)

#: Structural signal of any generated attempt-result comment: the ordinary
#: result header (whatever the outcome), the conflict-attempt header, or one
#: of the hidden machine markers publication stamps onto its shared comment.
_GENERATED_RESULT_HEADER_RE = re.compile(
    r"^#{1,6}\s*(Agent Attempt Result:|Conflict Attempt Result\b)", re.MULTILINE
)
_GENERATED_RESULT_MARKER_RES = (
    "<!-- agent-attempt:",
    "<!-- conflict-attempt:",
    "<!-- conflict-escalation -->",
    "<!-- requeue-source:",
)

#: Machine bookkeeping lines publication renders onto its shared result
#: comment. They route humans and machines to the published work (branch, PR,
#: note commit, merge, conflict standing) or carry the budget standing; none
#: of them is guidance for the next implementation.
_BOOKKEEPING_LINE_RES = (
    re.compile(r"^\*\*(Branch|PR|Handoff note commit|Merge|Conflict attempts)\*\*:"),
    re.compile(r"^Automatic continuation \d+/\d+ used;"),
)

#: Any hidden HTML marker, wherever it sits on the line.
_HIDDEN_MARKER_RE = re.compile(r"<!--.*?-->")


def is_generated_handoff_result(body: str) -> bool:
    """Whether a comment body is a generated handoff attempt-result.

    Structural, never identity-based: the handoff result header identifies
    the comment no matter who authored it, so a human note that happens to
    share the agent's login is never mistaken for one (and vice versa).
    """

    return _HANDOFF_RESULT_HEADER_RE.search(body or "") is not None


def is_generated_result_comment(body: str) -> bool:
    """Whether a comment body is any generated attempt-result comment."""

    text = body or ""
    if _GENERATED_RESULT_HEADER_RE.search(text) is not None:
        return True
    return any(marker in text for marker in _GENERATED_RESULT_MARKER_RES)


def strip_result_bookkeeping(body: str) -> str:
    """Remove machine bookkeeping from a generated result comment.

    Drops the hidden markers, the branch/PR/note-commit/merge/conflict
    standing lines, and the automatic-continuation standing line, keeping
    the human-readable evidence and details prose. Blank fallout is
    collapsed so the remainder stays readable in a prompt.
    """

    lines: list[str] = []
    for line in (body or "").splitlines():
        stripped = _HIDDEN_MARKER_RE.sub("", line)
        if not stripped.strip():
            continue
        if any(pattern.match(stripped.strip()) for pattern in _BOOKKEEPING_LINE_RES):
            continue
        lines.append(stripped.rstrip())
    return "\n".join(lines).strip()


def filter_prompt_comments(bodies: Iterable[str]) -> tuple[str, ...]:
    """Filter trusted comments down to what a starting prompt may carry.

    Generated handoff results are superseded by the replacement Handoff
    note committed on the resumed branch, so they (including their note
    payloads) are dropped wholesale. Remaining generated results keep
    their human-readable prose with the machine bookkeeping stripped.
    Ordinary human guidance passes through verbatim in its existing order.

    Presentation-only: detection, budget counting, and publication
    recovery keep reading the unfiltered history.
    """

    filtered: list[str] = []
    for body in bodies:
        if is_generated_handoff_result(body):
            continue
        if is_generated_result_comment(body):
            stripped = strip_result_bookkeeping(body)
            if stripped:
                filtered.append(stripped)
            continue
        filtered.append(body)
    return tuple(filtered)
