"""Operator recovery for retained implementation attempts (issue #84).

A retained attempt is a checkpoint whose durable completion boundary was
never reached: its outcome was not published with a confirmed comment, the
issue was not released, local cleanup is not durable, or accounting never
committed. While such a checkpoint (or unexplained dirty work) exists,
intake stays blocked and ``resume`` is rejected.

This module owns retained-attempt inference: given a retained checkpoint
and workspace evidence, what happened and what may safely be redone.
Startup reconciliation and the operator recovery retry/release commands
both infer through this module; the lifecycle keeps only locking,
control-store bookkeeping, and side-effecting execution. The control
store owns the durable command record; the CLI and socket only transport
requests.
"""

from __future__ import annotations

from collections.abc import Container, Iterable, Mapping
from dataclasses import dataclass
import re

from simple_coding_agent.attempt_state import AttemptPhase
from simple_coding_agent.completion import (
    AttemptOutcome,
    CompletionDecision,
    PublicationPath,
)


class RecoveryRejectedError(ValueError):
    """Raised when a recovery command is rejected without a control change.

    Unknown attempt IDs, mismatched identities, already-finalized attempts,
    concurrently changing attempts, and invalid ``--saved-at`` values leave
    intake and the command record unchanged.
    """


class RecoveryAttemptNotFoundError(RecoveryRejectedError):
    """The attempt ID matches neither the retained nor any recorded attempt.

    The operator contract answers ``not_found`` for it; the legacy path
    reports it like any other rejection. Nothing is stored.
    """


class RecoveryAlreadyFinalizedError(RecoveryRejectedError):
    """The attempt ID belongs to an attempt recorded as finalized.

    ``recovery release`` rejects it; ``recovery retry`` completes as a
    no-op, since no safe finalization step remains to retry.
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


#: The per-Worker identity recorded in every attempt-result comment when the
#: operator runs a shared-login pool with ``WORKER_ID`` set (issue #301).
#: The visible ``Worker: <id>`` line sits next to the existing outcome
#: details and the hidden marker sits next to the existing
#: ``<!-- agent-attempt: ... -->`` markers. Without ``WORKER_ID`` neither is
#: rendered, keeping the comment byte-for-byte identical to today. Existing
#: marker parsers match their own literal markers only, so this marker is
#: transparent to them by construction.
#: Prefix of the durable per-attempt marker. Every published attempt-result
#: comment carries it alone on its own line (see :func:`attempt_marker`).
AGENT_ATTEMPT_MARKER_PREFIX = "<!-- agent-attempt:"

AGENT_WORKER_MARKER_PREFIX = "<!-- agent-worker:"


def worker_marker(worker_id: str) -> str:
    """The durable issue-comment marker recording one attempt's Worker."""

    return f"<!-- agent-worker: {(worker_id or '').strip()} -->"


_WORKER_MARKER_RE = re.compile(re.escape(AGENT_WORKER_MARKER_PREFIX) + r"\s*(.*?)\s*-->")


def parse_agent_worker(body: str | None) -> str | None:
    """Return the ``<!-- agent-worker: ... -->`` identity in a comment, if any."""

    if not body:
        return None
    match = _WORKER_MARKER_RE.search(body)
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def worker_line(worker_id: str) -> str:
    """The visible Worker line for an attempt-result comment, or empty."""

    cleaned = (worker_id or "").strip()
    if not cleaned:
        return ""
    return f"Worker: {cleaned}"


def worker_blocks(worker_id: str) -> tuple[str, str]:
    """The visible line block and hidden marker block for one Worker.

    Returns ``(line_block, marker_block)``, each either empty (no
    ``WORKER_ID``) or terminated with a newline so callers can splice them
    next to the existing outcome details and ``agent-attempt`` marker without
    changing any other byte.
    """

    cleaned = (worker_id or "").strip()
    if not cleaned:
        return ("", "")
    return (f"{worker_line(cleaned)}\n", f"{worker_marker(cleaned)}\n")


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


def find_block_marker(body: str, prefix: str) -> str | None:
    """Return the first hidden marker sitting alone on its own line, if any.

    A marker counts only when a line outside fenced code blocks starts with
    ``prefix`` with at most three leading spaces or tabs (mirroring the ATX
    heading rule, so indented code blocks never match) and closes on the same
    line, so prose, list items, blockquotes, and inline code spans that merely
    quote a marker never match, and neither does a marker quoted inside a
    fenced block. Returns the stripped marker line, or ``None``.
    """

    in_fence = False
    for line in (body or "").splitlines():
        stripped = line.strip()
        leading = line[: len(line) - len(line.lstrip())]
        if len(leading) > 3 or any(char not in " \t" for char in leading):
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith(prefix) and "-->" in stripped:
            return stripped
    return None


def is_handoff_record(body: str) -> bool:
    """Whether a comment body is the shared handoff record.

    Structural: the body must start (after up to three leading spaces) with
    the record heading (``## Handoff note``) or with the hidden
    ``handoff-record`` marker itself (bare-marker shorthand used in older
    tests), and a complete hidden marker must sit alone on its own line
    outside any fenced block. Human prose, lists, code spans, blockquotes,
    or fenced blocks that merely quote the marker elsewhere never match.
    """

    text = body or ""
    starts_heading = _HANDOFF_NOTE_HEADING_TOP_RE.search(text) is not None
    starts_marker = text.lstrip().startswith(HANDOFF_RECORD_MARKER_PREFIX)
    if not (starts_heading or starts_marker):
        return False
    return find_block_marker(text, HANDOFF_RECORD_MARKER_PREFIX) is not None


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
        if is_pr_head_stale_notice(body):
            continue
        if not is_attempt_result_comment(body):
            # Not a structural ordinary result (human prose, quoted markers,
            # or legacy bodies without an attempt marker): ignore it so quoted
            # markers never count or reset the standing.
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
        if is_pr_head_stale_notice(body):
            continue
        if not is_attempt_result_comment(body):
            # Human prose or quoted markers are not results: stop the row
            # instead of counting a quoted automatic marker.
            break
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
#: The pull request's recorded head differs from the published branch tip
#: observed at claim: GitHub stopped syncing the pull request with its
#: branch, so rebasing and pushing would change nothing on the pull request.
CONFLICT_REASON_STALE_HEAD = "stale_head"

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
    """Whether a comment body is a structural conflict-attempt result.

    The body must start (after up to three leading spaces) with the
    conflict-attempt header and carry the conflict marker alone on its own
    line outside any fenced block, or be the bare marker itself (shorthand
    used in older tests). Human prose quoting the marker elsewhere never
    matches.
    """

    text = body or ""
    if find_block_marker(text, _CONFLICT_MARKER_PREFIX) is None:
        return False
    if text.lstrip().startswith(_CONFLICT_MARKER_PREFIX):
        return True
    return _CONFLICT_HEADER_TOP_RE.search(text) is not None


def is_attempt_result_comment(
    body: str, kinds: tuple[str, ...] | list[str] | None = None
) -> bool:
    """Whether a comment body is a structural ordinary attempt-result comment.

    The body must start (after up to three leading spaces) with the ordinary
    result header (``## Agent Attempt Result: <kind>``) and carry a
    well-formed ``agent-attempt`` marker with a non-empty id alone on its own
    line outside any fenced block. When ``kinds`` is given the header kind
    must be one of them (e.g. ``("handoff", "incomplete")`` for continuation
    detection). Human prose quoting the header or the marker elsewhere never
    matches.
    """

    text = body or ""
    match = _ORDINARY_HEADER_TOP_RE.search(text)
    if match is None:
        return False
    if kinds is not None and match.group(1) not in kinds:
        return False
    marker = find_block_marker(text, AGENT_ATTEMPT_MARKER_PREFIX)
    if marker is None:
        return False
    parsed = _ATTEMPT_MARKER_RE.search(marker)
    return parsed is not None and bool(parsed.group(1).strip())


#: Hidden marker identifying the conflict-attempt result that escalated its
#: issue to a human. The escalating attempt's own result comment carries it,
#: so the streak stops there: after a human removes ``ready-for-human`` the
#: next poll requeues with the full budget.
CONFLICT_ESCALATION_MARKER = "<!-- conflict-escalation -->"


def is_conflict_escalation(body: str) -> bool:
    """Whether a comment body is the escalation comment of its streak.

    Structural: the body must start with the conflict-attempt header and
    carry the escalation marker alone on its own line outside any fenced
    block. Human prose quoting the marker elsewhere never matches.
    """

    text = body or ""
    if find_block_marker(text, CONFLICT_ESCALATION_MARKER) is None:
        return False
    return _CONFLICT_HEADER_TOP_RE.search(text) is not None


#: Hidden marker identifying a pull-request-head-out-of-sync notice. The
#: detector posts one notice comment per stale state on the issue behind the
#: branch; the marker names the pull request, its recorded head, and the
#: branch tip, so a later poll can tell whether the state changed (a new
#: comment) or is already reported (no comment).
_PR_HEAD_STALE_MARKER_PREFIX = "<!-- pr-head-stale:"


def pr_head_stale_marker(pr_number: int, head_sha: str, branch_tip: str) -> str:
    """The durable marker recording which stale head state a notice reported."""

    for value in (head_sha, branch_tip):
        if not value or any(character.isspace() for character in value):
            raise ValueError(f"Invalid stale-head marker value {value!r}")
    if isinstance(pr_number, bool) or not isinstance(pr_number, int) or pr_number <= 0:
        raise ValueError(f"Invalid stale-head pull request number {pr_number!r}")
    return (
        f"{_PR_HEAD_STALE_MARKER_PREFIX} pr={pr_number}"
        f" head={head_sha} tip={branch_tip} -->"
    )


def parse_pr_head_stale(body: str) -> dict[str, str] | None:
    """Return the stale-head marker fields in a comment body, if any.

    A body without the marker reads as ``None``.
    """

    start = body.find(_PR_HEAD_STALE_MARKER_PREFIX)
    if start < 0:
        return None
    end = body.find("-->", start)
    if end < 0:
        return None
    fields: dict[str, str] = {}
    for token in body[start + len(_PR_HEAD_STALE_MARKER_PREFIX) : end].split():
        key, separator, value = token.partition("=")
        if separator and key and value:
            fields[key] = value
    if not all(key in fields for key in ("pr", "head", "tip")):
        return None
    return fields


def is_pr_head_stale_notice(body: str) -> bool:
    """Whether a comment body is a pull-request-head-out-of-sync notice."""

    return parse_pr_head_stale(body) is not None


def pr_head_stale_notice_body(
    *,
    repository: str,
    branch: str,
    pr_number: int,
    head_sha: str,
    branch_tip: str,
) -> str:
    """Build the one-per-state detector notice for a stale pull request head.

    The body names both shas, links the pull request, and carries the
    recovery that worked: never close and reopen the pull request (GitHub
    refuses the reopen after a force-push), and open a replacement pull
    request from the same branch instead. The hidden marker identifies the
    exact stale state so later polls deduplicate.
    """

    pr_link = f"https://github.com/{repository}/pull/{pr_number}"
    return (
        "## Pull request head out of sync\n\n"
        f"The pull request [#{pr_number}]({pr_link}) behind `{branch}` reports"
        f" merge conflicts, but its recorded head `{head_sha}` differs from"
        f" the tip of the branch (`{branch_tip}`). The detector skipped the"
        " `merge-conflict` requeue, so no conflict attempt will run until a"
        " human resolves the desync.\n\n"
        "Do not close and reopen the pull request: GitHub refuses the reopen"
        " with `422 ... branch was force-pushed or recreated`. Open a"
        f" replacement pull request from the same `{branch}` branch"
        " instead.\n\n"
        f"{pr_head_stale_marker(pr_number, head_sha, branch_tip)}"
    )


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
    if not is_conflict_attempt(body):
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
        if is_pr_head_stale_notice(body):
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
    worker_id: str = "",
    verdict_recorded: bool = False,
) -> str:
    """Build the abandonment comment for ``recovery release``.

    The comment states the actual attempt outcome and the failure, records
    where the operator secured the work, and never claims a successful
    handoff or an unpublished branch as a success. With
    ``verdict_recorded`` the outcome is the verdict publication recorded
    (ADR 0010), which the release keeps; only a recorded ``complete``, whose
    pull request already exists, drops the "not published as a success"
    note from the branch line. Without it the outcome was inferred and the
    comment is byte-for-byte the legacy body. With ``worker_id`` set the comment
    carries the visible ``Worker:`` line and the hidden ``agent-worker``
    marker.
    """

    marker = attempt_marker(attempt_id)
    source_marker = requeue_source_marker(requeue_source)
    worker_block, worker_marker_block = worker_blocks(worker_id)
    branch_note = "retained work secured by the operator; not published as a success"
    if verdict_recorded:
        if actual_outcome == "complete":
            branch_note = "retained work secured by the operator"
        outcome_text = (
            f"The attempt `{attempt_id}` for issue #{issue_number} recorded the"
            f" verdict `{actual_outcome}` when it was published, but did not"
            " complete finalization and was explicitly released by the"
            " operator. The release keeps the recorded verdict."
        )
    else:
        outcome_text = (
            f"The attempt `{attempt_id}` for issue #{issue_number} did not complete"
            " finalization and was explicitly released by the operator."
        )
    return (
        f"## Agent Attempt Result: {actual_outcome}\n\n"
        f"**Branch**: `{branch}` ({branch_note})\n"
        f"{worker_block}\n"
        "### Outcome\n"
        f"{outcome_text}\n\n"
        "### Failure\n"
        f"{failure_reason}\n\n"
        "### Saved work\n"
        f"Retained work was secured at: `{saved_at}`\n\n"
        "This release does not claim a successful handoff and does not"
        " requeue the issue.\n\n"
        f"{marker}\n"
        f"{worker_marker_block}"
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
#: legacy: the result header with the ``handoff`` outcome at the top of the
#: body (after up to three leading spaces, so indented code blocks never
#: match). Markers (``agent-attempt``,
#: ``requeue-source``) are not required, so comments posted before the markers
#: existed are still recognized. Human prose quoting the header anywhere else
#: (prose, lists, code spans, blockquotes, fenced blocks) is never matched:
#: the header is only ever rendered by publication at the top.
_ORDINARY_HEADER_TOP_RE = re.compile(
    r"\A[ \t]{0,3}#{1,6}\s*Agent Attempt Result:\s*([A-Za-z_]+)"
)

_CONFLICT_HEADER_TOP_RE = re.compile(
    r"\A[ \t]{0,3}#{1,6}\s*Conflict Attempt Result\b"
)

_HANDOFF_NOTE_HEADING_TOP_RE = re.compile(
    r"\A[ \t]{0,3}#{1,6}\s*Handoff note\b"
)

#: Machine bookkeeping lines publication renders onto its shared result
#: comment. They route humans and machines to the published work (branch, PR,
#: note commit, merge, conflict standing), name the Worker that wrote the
#: attempt, or carry the budget standing; none of them is guidance for the
#: next implementation.
_BOOKKEEPING_LINE_RES = (
    re.compile(r"^\*\*(Branch|PR|Handoff note commit|Merge|Conflict attempts)\*\*:"),
    re.compile(r"^Automatic continuation \d+/\d+ used;"),
    re.compile(r"^Worker:"),
)

#: Any hidden HTML marker, wherever it sits on the line.
_HIDDEN_MARKER_RE = re.compile(r"<!--.*?-->")


def is_generated_handoff_result(body: str) -> bool:
    """Whether a comment body is a generated handoff attempt-result.

    Structural, never identity-based: the handoff result header at the top
    identifies the comment no matter who authored it, so a human note that
    happens to share the agent's login is never mistaken for one (and vice
    versa). Legacy bodies starting with the header but predating markers
    still match, so old handoffs stay excluded from prompts.
    """

    text = body or ""
    match = _ORDINARY_HEADER_TOP_RE.search(text)
    return match is not None and match.group(1) == "handoff"


def is_generated_result_comment(body: str) -> bool:
    """Whether a comment body is any generated attempt-result comment.

    Structural: the ordinary or conflict header must start the body. Hidden
    markers alone never identify a result (including the transparent
    ``agent-worker`` marker), so human prose quoting a marker passes through
    verbatim and is never bookkeeping-stripped.
    """

    text = body or ""
    if _ORDINARY_HEADER_TOP_RE.search(text) is not None:
        return True
    return _CONFLICT_HEADER_TOP_RE.search(text) is not None


def strip_result_bookkeeping(body: str) -> str:
    """Remove machine bookkeeping from a generated result comment.

    Drops the hidden markers (including the transparent ``agent-worker``
    marker), the branch/PR/note-commit/merge/conflict standing lines, the
    Worker line, and the automatic-continuation standing line, keeping
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


# ---------------------------------------------------------------------------
# Retained-attempt inference (issue #111)
# ---------------------------------------------------------------------------
#
# Single source of truth for "checkpoint phase + commits + archive markers
# -> what happened, what may safely be redone". Both startup reconciliation
# and ``recovery retry`` infer through :func:`infer_retry_decision`;
# ``recovery release`` infers through :func:`infer_release_outcome`.
# Conflict detection (:func:`infer_conflict_reason`) and the status hold
# snapshot builders below live here for the same reason: one set of
# recovery rules, testable without constructing a lifecycle.
#
# The functions are pure: workspace reads (commits, handoff-note commits,
# dirty state) and archive reads happen at the call site through the
# attempt-workspace seam and are passed in as plain evidence. The one
# lifecycle side effect the old inference performed inline -- advancing a
# ``MODEL_RUNNING`` checkpoint with preserved commits to ``PUSHING`` -- is
# reported as :attr:`RetryInference.advance_to_pushing` so the caller
# performs it alongside its event log.

#: Reason when the final check started but never finished.
RETRY_FINAL_CHECK_INTERRUPTED_REASON = (
    "Final check was interrupted before completing;"
    " committed work is preserved on the branch."
)

#: Reason when model execution left preserved commits behind.
RETRY_MODEL_INTERRUPTED_WITH_COMMITS_REASON = (
    "Model execution was interrupted; preserved committed work."
)

#: Reason when model execution left no commits behind.
RETRY_MODEL_INTERRUPTED_NO_COMMITS_REASON = (
    "Model execution was interrupted without commits."
)

#: Reason when the attempt never reached model execution.
RETRY_BEFORE_MODEL_REASON = "Attempt was interrupted before model execution."

#: Reason when pushing/publishing left completed local evidence behind.
RETRY_RECOVERED_EVIDENCE_REASON = (
    "Recovered previously completed local evidence for publication."
)

#: Reason when pushing/publishing left a valid handoff note behind.
RETRY_RECOVERED_HANDOFF_REASON = (
    "Recovered a previously committed handoff note for publication."
)


def _infrastructure_decision(reason: str) -> CompletionDecision:
    """An infrastructure-error decision that publishes nothing."""

    return CompletionDecision(
        AttemptOutcome.INFRASTRUCTURE_ERROR, False, PublicationPath.NONE, (reason,)
    )


@dataclass(frozen=True)
class RetryEvidence:
    """Workspace evidence for one retained-attempt retry inference."""

    phase: AttemptPhase
    has_commits: bool
    final_check_interrupted: bool = False
    note_revision: str | None = None
    note_valid: bool | None = None
    note_reason: str | None = None
    note_details: str | None = None


@dataclass(frozen=True)
class RetryInference:
    """What a retained attempt means and what retry may safely redo."""

    decision: CompletionDecision
    details: str
    note_sha: str | None
    has_commits: bool
    advance_to_pushing: bool


def infer_retry_decision(evidence: RetryEvidence) -> RetryInference:
    """Infer the safe republication decision for a retained attempt.

    Pure: every workspace/archive read is already reflected in
    ``evidence``. ``advance_to_pushing`` is True exactly when the caller
    should advance a ``MODEL_RUNNING`` checkpoint with preserved commits
    (and no interrupted final check) to ``PUSHING`` before publishing.
    """

    if evidence.phase is AttemptPhase.MODEL_RUNNING:
        if evidence.has_commits:
            if evidence.final_check_interrupted:
                decision = CompletionDecision(
                    AttemptOutcome.INFRASTRUCTURE_ERROR,
                    False,
                    PublicationPath.PARTIAL,
                    (RETRY_FINAL_CHECK_INTERRUPTED_REASON,),
                )
                return RetryInference(
                    decision, decision.reasons[0], None, True, False
                )
            decision = CompletionDecision(
                AttemptOutcome.INCOMPLETE,
                False,
                PublicationPath.PARTIAL,
                (RETRY_MODEL_INTERRUPTED_WITH_COMMITS_REASON,),
            )
            return RetryInference(
                decision, decision.reasons[0], None, True, True
            )
        decision = _infrastructure_decision(
            RETRY_MODEL_INTERRUPTED_NO_COMMITS_REASON
        )
        return RetryInference(decision, decision.reasons[0], None, False, False)
    if evidence.phase in (AttemptPhase.PUSHING, AttemptPhase.PUBLISHING):
        if evidence.note_revision is not None:
            if evidence.note_valid:
                decision = CompletionDecision(
                    AttemptOutcome.HANDOFF,
                    False,
                    PublicationPath.PARTIAL,
                    (RETRY_RECOVERED_HANDOFF_REASON,),
                )
                details = evidence.note_details or decision.reasons[0]
                return RetryInference(
                    decision, details, evidence.note_revision,
                    evidence.has_commits, False,
                )
            reason = evidence.note_reason or (
                "Handoff was requested, but no handoff note commit"
                " is on the branch."
            )
            decision = CompletionDecision(
                AttemptOutcome.INCOMPLETE,
                False,
                PublicationPath.PARTIAL,
                (reason,),
            )
            return RetryInference(
                decision, reason, None, evidence.has_commits, False
            )
        decision = CompletionDecision(
            None,
            True,
            PublicationPath.COMPLETE,
            (RETRY_RECOVERED_EVIDENCE_REASON,),
        )
        return RetryInference(
            decision, decision.reasons[0], None, evidence.has_commits, False
        )
    decision = _infrastructure_decision(RETRY_BEFORE_MODEL_REASON)
    return RetryInference(
        decision, decision.reasons[0], None, evidence.has_commits, False
    )


def final_check_interrupted(record: Mapping[str, object] | None) -> bool:
    """Whether an archive record shows a final check that never finished.

    The runner records ``final_check_started`` immediately before running
    the check and ``final_check_finished`` right after it returns, so a
    marker without its finish means the process never survived the check.
    ``None`` (no archive) reads as not interrupted.
    """

    if record is None:
        return False
    return bool(record.get("final_check_started")) and not record.get(
        "final_check_finished"
    )


@dataclass(frozen=True)
class ReleaseEvidence:
    """Workspace evidence for one retained-attempt release inference."""

    phase: AttemptPhase
    has_commits: bool
    note_valid: bool = False


@dataclass(frozen=True)
class ReleaseInference:
    """The actual outcome a ``recovery release`` records."""

    outcome: AttemptOutcome
    reason: str


def infer_release_outcome(evidence: ReleaseEvidence) -> ReleaseInference:
    """Infer the actual outcome recorded by ``recovery release``.

    Pure: ``note_valid`` is whether a committed handoff note validated,
    ``has_commits`` is whether any work commit is on the branch.
    """

    phase_name = evidence.phase.value
    if evidence.note_valid:
        return ReleaseInference(
            AttemptOutcome.HANDOFF,
            "The handoff note was committed but finalization never"
            f" completed in phase {phase_name}; the handoff"
            " is recorded as not fulfilled (not a successful handoff),"
            " and the retained branch is not published as a success.",
        )
    if evidence.has_commits:
        return ReleaseInference(
            AttemptOutcome.INCOMPLETE,
            "Local work was preserved on the branch but finalization never"
            f" completed in phase {phase_name}; the operator"
            " secured the retained work and released the attempt.",
        )
    if evidence.phase in (AttemptPhase.CLAIMED, AttemptPhase.SETUP):
        return ReleaseInference(
            AttemptOutcome.INFRASTRUCTURE_ERROR,
            "The attempt was interrupted before model execution produced"
            f" commits (phase {phase_name}); the operator"
            " secured the retained state and released the attempt.",
        )
    return ReleaseInference(
        AttemptOutcome.INFRASTRUCTURE_ERROR,
        "Finalization was interrupted before its completion boundary"
        f" (phase {phase_name}); the operator secured the"
        " retained work and released the attempt.",
    )


@dataclass(frozen=True)
class ConflictEvidence:
    """Remote-claim evidence for the pre-retry conflict probe."""

    issue_number: int
    fetch_error: str | None = None
    missing: bool = False
    assignment_error: str | None = None
    self_assigned: bool | None = None
    holders: tuple[str, ...] = ()
    foreign_marker: str = ""


def infer_conflict_reason(evidence: ConflictEvidence) -> str | None:
    """Detect conflicting assignment before a retry.

    Returns a hold reason, or ``None`` when no conflict blocks the retry.
    A fetch failure or an unreadable assignment is ambiguous (the hold
    remains); a confidently observed assignment to someone else is
    conflicting (the retry must not overwrite it).
    """

    if evidence.fetch_error is not None:
        return (
            "remote issue state is ambiguous"
            f" ({evidence.fetch_error}); the retry must not replay"
            " a side effect it cannot verify."
        )
    if evidence.missing:
        return (
            f"issue #{evidence.issue_number} is unavailable on GitHub;"
            " the retry must not replay a side effect it cannot verify."
        )
    if evidence.assignment_error is not None:
        return (
            "remote assignment is ambiguous"
            f" ({evidence.assignment_error}); the retry must not replay"
            " a side effect it cannot verify."
        )
    if not evidence.self_assigned:
        if evidence.holders:
            holders = ", ".join(evidence.holders)
            return (
                f"issue #{evidence.issue_number} shows conflicting"
                " assignment changes (assigned to another assignee:"
                f" {holders}{evidence.foreign_marker}); the retry"
                " must not overwrite them and requires operator repair."
            )
        return (
            f"issue #{evidence.issue_number} shows conflicting"
            " assignment changes (it is assigned to another assignee,"
            f" no longer this agent{evidence.foreign_marker}); the"
            " retry must not overwrite them and requires operator repair."
        )
    return None


def publication_progress(
    phase: str,
    *,
    comment_confirmed: bool | None,
    pull_request: dict | None,
) -> dict:
    """Best-effort confirmed publication progress for status (read-only)."""

    return {
        "phase": phase,
        "comment_confirmed": comment_confirmed,
        "pull_request": pull_request,
    }


def startup_unreadable_snapshot(
    *,
    workspace: str,
    workspace_dirty: bool | None,
    reason: str,
) -> dict:
    """Status ``recovery`` section when no checkpoint is readable."""

    return {
        "attempt_id": None,
        "issue_number": None,
        "phase": None,
        "outcome": None,
        "branch": None,
        "workspace": workspace,
        "workspace_dirty": workspace_dirty,
        "checkpoint": False,
        "publication": {
            "phase": None,
            "comment_confirmed": None,
            "pull_request": None,
        },
        "hold_reason": reason,
        "next_action": (
            "Repair or remove the unreadable state file described"
            " in the hold reason; intake resumes automatically"
            " once it reads cleanly."
        ),
    }


def dirty_tree_snapshot(*, workspace: str, reason: str) -> dict:
    """Status ``recovery`` section for unexplained dirty work alone."""

    return {
        "attempt_id": None,
        "issue_number": None,
        "phase": None,
        "outcome": None,
        "branch": None,
        "workspace": workspace,
        "workspace_dirty": True,
        "checkpoint": False,
        "publication": {
            "phase": None,
            "comment_confirmed": None,
            "pull_request": None,
        },
        "hold_reason": reason,
        "next_action": (
            "Inspect the working tree and repair or remove the"
            " unexplained changes before intake can resume."
        ),
    }


def retained_snapshot(
    *,
    attempt_id: str,
    issue_number: int,
    phase: str,
    outcome: str | None,
    branch: str,
    workspace: str,
    workspace_dirty: bool | None,
    publication: dict,
    hold_reason: str,
    recovering: bool,
    finalized: bool,
) -> dict:
    """Status ``recovery`` section for one retained attempt.

    ``hold_reason`` is the raw hold (without the ``Retained attempt ...``
    prefix); ``finalized`` is whether the completion ledger already
    records the attempt. ``recovering`` selects the startup wait message.
    """

    if recovering:
        next_action = (
            "Wait for startup reconciliation to finish; do not run recovery"
            " commands while it is still running."
        )
    elif finalized:
        next_action = (
            "The attempt is finalized in the ledger; wait for the leftover"
            " checkpoint to be removed on the next intake cycle."
        )
    else:
        next_action = retry_next_action(attempt_id)
    return {
        "attempt_id": attempt_id,
        "issue_number": issue_number,
        "phase": phase,
        "outcome": outcome,
        "branch": branch,
        "workspace": workspace,
        "workspace_dirty": workspace_dirty,
        "checkpoint": True,
        "publication": publication,
        "hold_reason": describe_hold(
            issue_number=issue_number,
            attempt_id=attempt_id,
            phase=phase,
            reason=hold_reason,
        ),
        "next_action": (
            next_action
            if (recovering or finalized)
            else f"{next_action} {release_next_action()}"
        ),
    }
