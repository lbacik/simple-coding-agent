"""Conflict requeue: bringing a conflicted published branch up to date.

A conflict requeue returns an issue to the queue because the pull request a
worker already published for it reports merge conflicts with the base
branch. The conflict attempt that follows only brings that pull request up
to date and does no new work: a clean rebase plus setup and one profile
check pushes directly, while a real rebase conflict (or a red check after a
clean rebase) runs at most one narrow model resolution session under the
ordinary model limits. This module holds the small, testable conflict
helpers; the lifecycle owns the serialized execution.
"""

from __future__ import annotations

from collections.abc import Container, Sequence
from pathlib import Path
import subprocess

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
    CONFLICT_OUTCOME_MODEL_LIMIT,
    CONFLICT_OUTCOME_REBASE_CONFLICT,
    CONFLICT_OUTCOME_RESOLUTION_FAILED,
    CONFLICT_OUTCOME_SETUP_FAILED,
    CONFLICT_OUTCOME_SKIPPED,
    CONFLICT_OUTCOME_SUCCESS,
    CONFLICT_REASON_BASE_MISMATCH,
    CONFLICT_REASON_DRAFT,
    CONFLICT_REASON_NOT_CONFLICTING,
    CONFLICT_REASON_NO_OPEN_PR,
)


from simple_coding_agent.github_tracker import (
    conflict_branch_issue_number as conflict_branch_issue_number,
    is_conflict_detection_eligible as is_conflict_detection_eligible,
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
    if conflict_outcome in (
        CONFLICT_OUTCOME_REBASE_CONFLICT,
        CONFLICT_OUTCOME_CHECK_FAILED,
        CONFLICT_OUTCOME_MODEL_LIMIT,
        CONFLICT_OUTCOME_RESOLUTION_FAILED,
    ):
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
        f"({CONFLICT_OUTCOME_REBASE_CONFLICT}).{listing} No model resolution ran; "
        "the remote branch is left untouched so the pull request stays `CONFLICTING`."
    )


def model_unresolved_details(
    branch: str,
    base_branch: str,
    files: Sequence[str],
    *,
    after_check: bool = False,
) -> str:
    """Human details for a model session that left conflicts unresolved."""

    listing = f" Conflicting files: {', '.join(files)}." if files else ""
    origin = (
        "the profile check stayed red, so the model session ran but the branch "
        "is still not up to date"
        if after_check
        else "the rebase conflicted, so one model resolution session ran but "
        "the rebase is still unresolved"
    )
    return (
        f"Rebase of `{branch}` onto `{base_branch}` hit conflicts "
        f"({CONFLICT_OUTCOME_REBASE_CONFLICT}).{listing} {origin.capitalize()}; "
        "the remote branch is left untouched so the pull request stays `CONFLICTING`."
    )


def model_limit_details(branch: str, base_branch: str, files: Sequence[str]) -> str:
    """Human details for a model resolution session that reached its limits."""

    listing = f" Conflicting files: {', '.join(files)}." if files else ""
    return (
        f"The single model resolution session for `{branch}` onto `{base_branch}` "
        f"reached its ordinary turn, time, or token limit ({CONFLICT_OUTCOME_MODEL_LIMIT})."
        f"{listing} No handoff note was written and no further session runs; "
        "the remote branch is left untouched so the pull request stays `CONFLICTING`."
    )


def commit_count_changed_details(
    branch: str, base_branch: str, *, expected: int, actual: int
) -> str:
    """Human details for a model resolution that rewrote the pull request's history."""

    return (
        f"The model resolution for `{branch}` onto `{base_branch}` changed the "
        f"rebased commit count ({CONFLICT_OUTCOME_RESOLUTION_FAILED}): the pull "
        f"request carried {expected} commit(s) and now carries {actual} outside "
        "its fix commits, so commits were squashed, dropped, or amended instead "
        "of adding fixes only on top. The remote branch is left untouched so the "
        "pull request stays `CONFLICTING`."
    )


def model_error_details(branch: str, base_branch: str, detail: str) -> str:
    """Human details for a model resolution session that errored."""

    return (
        f"The single model resolution session for `{branch}` onto `{base_branch}` "
        f"could not finish: {detail} The remote branch is left untouched so the "
        "pull request stays `CONFLICTING`."
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


def model_success_details(
    branch: str,
    base_branch: str,
    *,
    files: Sequence[str] = (),
    extra_files: Sequence[str] = (),
    fix_commits: Sequence[str] = (),
    sha: str | None = None,
) -> str:
    """Human details for a model-resolved, checked, and pushed branch."""

    pushed = f" Pushed as `{sha}`." if sha else ""
    listing = f" Conflicted files: {', '.join(files)}." if files else ""
    extra = (
        f" Files changed outside the conflicted set: {', '.join(extra_files)}."
        if extra_files
        else ""
    )
    fixes = f" Fix commits: {', '.join(fix_commits)}." if fix_commits else ""
    return (
        f"One model resolution session brought `{branch}` onto `{base_branch}`; "
        "the rebase completed with the pull request's commit count preserved, "
        "fixes landed only as commits on top, and setup plus the profile check "
        f"passed, so the resolved branch was force-pushed.{pushed}"
        f"{listing}{extra}{fixes}"
    )


def fix_commit_count(
    before_subjects: Sequence[str], after_subjects: Sequence[str]
) -> int | None:
    """Count the fix commits on top of a preserved rebased history.

    ``before_subjects`` are the pull request's commit subjects oldest-first as
    seen before the model session; ``after_subjects`` the same listing after
    the session and its seal. Returns the number of newest commits that are
    fixes on top, or ``None`` when the rebased history itself changed (fewer
    commits than before, or the oldest subjects no longer match in order, so
    commits were squashed, dropped, or amended).
    """

    before = list(before_subjects)
    after = list(after_subjects)
    if len(after) < len(before):
        return None
    if after[: len(before)] != before:
        return None
    return len(after) - len(before)


def files_outside_conflict(
    fix_files: Sequence[str], conflicted_files: Sequence[str]
) -> tuple[str, ...]:
    """Files the resolution changed outside the conflicted set, sorted."""

    conflicted = set(conflicted_files)
    return tuple(sorted({path for path in fix_files if path and path not in conflicted}))


#: Per-stream cap for check output quoted in the conflict resolution prompt.
#: Matches the baseline prompt's scale: bounded, but enough to diagnose the
#: failure. Output is cut from the front so the tail survives.
_CHECK_PROMPT_STREAM_LIMIT = 4000


def _tail_stream(output: object, *, limit: int = _CHECK_PROMPT_STREAM_LIMIT) -> str:
    """Keep the end of a command stream, marking front-truncation visibly."""

    text = output if isinstance(output, str) else (output or "")
    if not str(text).strip():
        return "(no output)"
    text = str(text)
    if len(text) <= limit:
        return text
    return f"...[truncated; showing the last {limit} characters]\n{text[-limit:]}"


def format_check_output(result: object) -> str:
    """Render a red profile check for the conflict resolution prompt.

    Names the failing command with its exit and quotes the tail of both
    streams; the output is already credential-scrubbed by the command runner.
    """

    commands = getattr(result, "commands", ())
    failed = next(
        (command for command in commands if getattr(command, "exit_code", 0) != 0),
        None,
    )
    if failed is None and commands:
        failed = commands[-1]
    if failed is None:
        return "The profile check failed before any command ran."
    command = getattr(failed, "command", "?")
    if getattr(failed, "timed_out", False) or getattr(failed, "exit_code", None) is None:
        result_text = "timed out"
    else:
        result_text = f"exit {getattr(failed, 'exit_code', None)}"
    return (
        f"Failing command: `{command}` ({result_text}).\n\n"
        "Failing command stdout (tail):\n"
        f"```\n{_tail_stream(getattr(failed, 'stdout', ''))}\n```\n\n"
        "Failing command stderr (tail):\n"
        f"```\n{_tail_stream(getattr(failed, 'stderr', ''))}\n```"
    )


def base_changes_text(
    working_directory: Path,
    base_branch: str,
    files: Sequence[str],
    *,
    limit: int = 20,
) -> str:
    """Summarize base-branch commits since the merge-base that touched ``files``.

    Best-effort: any git failure degrades to an ``(unavailable)`` placeholder
    instead of failing, so the resolution session still runs. ``limit`` caps
    the listed commits.
    """

    if not files:
        return "(no conflicted files; nothing to look up)"
    try:
        merge_base = _git_output(
            working_directory, "merge-base", "HEAD", f"origin/{base_branch}"
        )
        log = _git_output(
            working_directory,
            "log",
            f"--max-count={limit}",
            "--format=%h %s",
            f"{merge_base}..origin/{base_branch}",
            "--",
            *files,
        )
    except Exception:
        return (
            f"(base changes unavailable: could not read commits on "
            f"`origin/{base_branch}` since the merge-base)"
        )
    if not log.strip():
        return (
            f"(no commits on `origin/{base_branch}` since the merge-base touched "
            f"these files)"
        )
    return log.strip()


def _git_output(working_directory: Path, *arguments: str) -> str:
    """Run one read-only git command for prompt evidence."""

    completed = subprocess.run(
        ("git", *arguments),
        cwd=working_directory,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(arguments)} failed")
    return completed.stdout.strip()


def build_resolution_prompt(
    *,
    issue_body: str,
    comments: Sequence[str],
    conflicted_files: Sequence[str],
    base_changes: str,
    check_output: str | None,
    gate_section: str,
) -> str:
    """Compose the narrow conflict-resolution prompt for the single model session.

    The issue body and trusted comments ride along explicitly marked as
    context only: the session brings the published branch up to date and must
    not implement the issue. The full-implementation continuation instruction
    and the ``code-review`` instruction are deliberately absent: review is not
    required for a conflict resolution.
    """

    context_lines = [
        "## Context only — do not implement",
        "",
        "The issue body and trusted comments below are context for understanding",
        "the conflicting change. Do NOT implement the issue, do NOT start new work,",
        "and do NOT invoke the `code-review` skill: this session only brings the",
        "published branch up to date.",
        "",
        "### Issue body",
        "",
        issue_body.strip() or "(empty issue body)",
    ]
    for index, comment in enumerate(comments, start=1):
        context_lines += ["", f"### Trusted comment {index}", "", comment.strip()]
    files_lines = [
        "## Conflicts to resolve",
        "",
    ]
    if conflicted_files:
        files_lines.append(
            f"Resolve the conflicts in: {', '.join(conflicted_files)}."
        )
    else:
        files_lines.append("Resolve the remaining rebase conflicts.")
    files_lines += [
        "Inspect with `git status`, edit the conflicted files, stage with `git add`,",
        "and run `git rebase --continue` until the rebase completes.",
        "Keep the pull request's commits exactly as they are: do not squash, drop,",
        "amend, or reorder them. Once the rebase completes, any further fix that the",
        "check gate needs goes only as a new commit on top — never by rewriting",
        "the rebased commits.",
        "",
        "## Base changes since the merge-base",
        "",
        base_changes.strip() or "(unavailable)",
    ]
    parts = ["\n".join(context_lines), "\n".join(files_lines)]
    if check_output is not None:
        parts.append(
            "## Failing check output (semantic conflict)\n\n"
            "The cleanly rebased branch failed the profile check below. "
            "Fix only what this output requires, as commits on top.\n\n" + check_output.strip()
        )
    parts.append(gate_section)
    return "\n\n".join(parts)


def short_revision(revision: str, *, length: int = 12) -> str:
    """A readable prefix of a commit revision for prose (never the marker)."""

    return revision[:length] if len(revision) > length else revision
