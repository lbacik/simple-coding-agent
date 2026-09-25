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


def release_comment_body(
    *,
    issue_number: int,
    attempt_id: str,
    actual_outcome: str,
    failure_reason: str,
    saved_at: str,
    branch: str,
) -> str:
    """Build the abandonment comment for ``recovery release``.

    The comment states the actual attempt outcome and the failure, records
    where the operator secured the work, and never claims a successful
    handoff or an unpublished branch as a success.
    """

    marker = attempt_marker(attempt_id)
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
        f"{marker}"
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
