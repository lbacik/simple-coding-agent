"""Conflict requeue: bringing a conflicted published branch up to date.

A conflict requeue returns an issue to the queue because the pull request a
worker already published for it reports merge conflicts with the base
branch. The conflict attempt that follows only brings that pull request up
to date and does no new work: no model call runs. This module holds the
small, testable conflict helpers; the lifecycle owns the serialized
execution.
"""

from __future__ import annotations

from collections.abc import Container, Sequence

from simple_coding_agent.completion import AttemptOutcome
from simple_coding_agent.github_tracker import (
    MERGE_CONFLICT,
    READY_FOR_AGENT,
    READY_FOR_HUMAN,
    ROUND_FINISHED,
)
from simple_coding_agent.publication import PullRequest
from simple_coding_agent.recovery import (
    CONFLICT_OUTCOME_CHECK_FAILED,
    CONFLICT_OUTCOME_ERROR,
    CONFLICT_OUTCOME_REBASE_CONFLICT,
    CONFLICT_OUTCOME_SETUP_FAILED,
    CONFLICT_OUTCOME_SKIPPED,
    CONFLICT_OUTCOME_SUCCESS,
    CONFLICT_REASON_BASE_MISMATCH,
    CONFLICT_REASON_DRAFT,
    CONFLICT_REASON_NOT_CONFLICTING,
    CONFLICT_REASON_NO_OPEN_PR,
)


def is_conflict_claim(labels: Container[str]) -> bool:
    """Whether a claim snapshot routes to a conflict attempt.

    ``merge-conflict`` without ``ready-for-agent``, ``round-finished`` or
    ``ready-for-human``: the first two take the ordinary (or continuation)
    path instead with the stale conflict signal removed at the claim, and
    a human-gated issue is never a conflict attempt.
    """

    return (
        MERGE_CONFLICT in labels
        and READY_FOR_AGENT not in labels
        and ROUND_FINISHED not in labels
        and READY_FOR_HUMAN not in labels
    )


def attempt_branch(issue_number: int) -> str:
    """The published branch a conflict attempt brings up to date."""

    return f"agent/issue-{issue_number}"


def entry_skip_reason(pr: PullRequest | None, *, branch: str) -> tuple[str, str] | None:
    """The ``(reason, detail)`` ending a conflict attempt as skipped, if any.

    Runs after the claim and before any git work, from a fresh read of the
    open pull request behind ``branch``. ``UNKNOWN`` mergeability proceeds:
    the local rebase is the source of truth.
    """

    if pr is None:
        return (
            CONFLICT_REASON_NO_OPEN_PR,
            f"No open pull request for branch `{branch}`; there is nothing to bring up to date.",
        )
    if pr.draft:
        return (
            CONFLICT_REASON_DRAFT,
            f"Pull request #{pr.number} is a draft; a conflict attempt never publishes draft work.",
        )
    if pr.mergeable not in ("CONFLICTING", "UNKNOWN"):
        return (
            CONFLICT_REASON_NOT_CONFLICTING,
            f"Pull request #{pr.number} reports `{pr.mergeable}`, not `CONFLICTING`; "
            "there is nothing to bring up to date.",
        )
    return None


def base_mismatch_reason(pr: PullRequest, base_branch: str) -> tuple[str, str] | None:
    """The ``(reason, detail)`` when the pull request targets another base."""

    if pr.base_ref == base_branch:
        return None
    return (
        CONFLICT_REASON_BASE_MISMATCH,
        f"Pull request #{pr.number} targets `{pr.base_ref or 'unknown'}`, not the profile "
        f"base branch `{base_branch}`; a conflict attempt only rebases onto its own base.",
    )


def lifecycle_outcome(conflict_outcome: str) -> AttemptOutcome:
    """Map a conflict marker outcome to the lifecycle outcome it finalizes as.

    Success is complete work; skipped ends quietly; a conflicting rebase or
    a red check is ordinary incompleteness (the branch stays published but
    conflicting, and the consecutive-error guard must not trip on it); setup
    and push failures are infrastructure errors, as in the ordinary path.
    """

    if conflict_outcome == CONFLICT_OUTCOME_SUCCESS:
        return AttemptOutcome.COMPLETE
    if conflict_outcome == CONFLICT_OUTCOME_SKIPPED:
        return AttemptOutcome.NO_CHANGES
    if conflict_outcome in (CONFLICT_OUTCOME_REBASE_CONFLICT, CONFLICT_OUTCOME_CHECK_FAILED):
        return AttemptOutcome.INCOMPLETE
    if conflict_outcome in (
        CONFLICT_OUTCOME_SETUP_FAILED,
        CONFLICT_OUTCOME_ERROR,
    ):
        return AttemptOutcome.INFRASTRUCTURE_ERROR
    return AttemptOutcome.INFRASTRUCTURE_ERROR


def rebase_conflict_details(branch: str, base_branch: str, files: Sequence[str]) -> str:
    """Human details for a rebase that conflicted without model resolution."""

    listing = f" Conflicting files: {', '.join(files)}." if files else ""
    return (
        f"Rebase of `{branch}` onto `{base_branch}` hit conflicts "
        f"({CONFLICT_OUTCOME_REBASE_CONFLICT}).{listing} No model resolution runs yet; "
        "the remote branch is left untouched so the pull request stays `CONFLICTING`."
    )


def _failed_command_text(result: object) -> str:
    """Name the command a setup/check failure blames, with its exit."""

    commands = getattr(result, "commands", ())
    failed = next(
        (command for command in commands if getattr(command, "exit_code", 0) != 0),
        None,
    )
    if failed is None and commands:
        failed = commands[-1]
    if failed is None:
        return "no command ran"
    command = getattr(failed, "command", "?")
    if getattr(failed, "timed_out", False) or getattr(failed, "exit_code", None) is None:
        return f"`{command}` (timed out)"
    return f"`{command}` (exit {getattr(failed, 'exit_code', None)})"


def setup_failed_details(result: object) -> str:
    """Human details for setup failing on the rebased branch."""

    return (
        "Setup did not succeed on the rebased branch; the remote branch is left "
        f"untouched so the pull request stays `CONFLICTING`. Failing command: "
        f"{_failed_command_text(result)}."
    )


def check_failed_details(result: object) -> str:
    """Human details for the single profile check failing on the rebased branch."""

    return (
        "The profile check did not pass on the rebased branch; the remote branch "
        f"is left untouched so the pull request stays `CONFLICTING`. Failing command: "
        f"{_failed_command_text(result)}."
    )


def success_details(branch: str, base_branch: str, sha: str | None = None) -> str:
    """Human details for a rebased, checked, and pushed branch."""

    pushed = f" Pushed as `{sha}`." if sha else ""
    return (
        f"Rebased `{branch}` onto `{base_branch}` with no conflicts; setup and the "
        f"profile check passed, so the rebased branch was force-pushed.{pushed}"
    )
