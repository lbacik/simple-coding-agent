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

from collections.abc import Container


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
#: only ``round-finished`` (an automatic continuation without a human requeue).
REQUEUE_SOURCE_HUMAN = "human"
REQUEUE_SOURCE_AUTOMATIC = "automatic"


def requeue_source_marker(source: str) -> str:
    """The durable issue-comment marker recording one attempt's requeue source."""

    if source not in (REQUEUE_SOURCE_HUMAN, REQUEUE_SOURCE_AUTOMATIC):
        raise ValueError(f"Invalid requeue source {source!r}: must be 'human' or 'automatic'")
    return f"<!-- requeue-source: {source} -->"


def parse_requeue_source(body: str) -> str | None:
    """Return the requeue source recorded in a comment body, if any.

    Comments written before the marker existed carry no marker and read as
    ``None`` (a legacy result, counted as a human requeue).
    """

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


def count_automatic_continuations(bodies_newest_first: tuple[str, ...] | list[str]) -> int:
    """Count automatic-requeue result comments in a row from the newest.

    Stops at the most recent human-requeue result; a legacy unmarked comment
    counts as a human requeue and also stops the count. Conflict-attempt
    result comments are transparent: they are skipped without counting and
    without stopping the row, so a conflict resolution neither consumes nor
    resets the automatic-continuation budget.
    """

    count = 0
    for body in bodies_newest_first:
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
