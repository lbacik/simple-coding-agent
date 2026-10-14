"""Deterministic setup/check evidence and attempt-completion evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from simple_coding_agent.command_runner import CommandResult, CommandRunner
from simple_coding_agent.config import ProfileError, RepositoryProfile
from simple_coding_agent.model_execution import ModelExecutionStatus


class AttemptOutcome(StrEnum):
    """Terminal outcomes defined by the attempt lifecycle."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    NO_CHANGES = "no_changes"
    HANDOFF = "handoff"


class PublicationPath(StrEnum):
    """The publisher-owned path permitted by deterministic attempt evidence."""

    NONE = "none"
    COMPLETE = "complete"
    PARTIAL = "partial"


@dataclass(frozen=True)
class PreparationEvidence:
    """Setup and the mandatory uncached baseline check on a prepared branch."""

    setup: CommandResult
    baseline: CommandResult | None


@dataclass(frozen=True)
class ReviewFinding:
    """One structured finding reported by the final review."""

    severity: str
    summary: str


@dataclass(frozen=True)
class ReviewEvidence:
    """Final review findings and the number of attempted repair cycles."""

    findings: tuple[ReviewFinding, ...]
    repair_cycles: int


def final_check_dirty_reason(paths: Sequence[str]) -> str:
    """The cleanup-only reason for a final check that dirtied the workspace.

    The implementation and the final check both succeeded; only the check's
    side effects need tidying. No re-implementation is needed: a short
    cleanup round commits the paths in the form the toolchain writes them,
    or adds the generated ones to ``.gitignore``, then finishes.
    """

    listed = ", ".join(paths)
    return (
        "The implementation and the final check succeeded, but the final check"
        f" left uncommitted changes: {listed}. No re-implementation is needed,"
        " only a short cleanup round: commit these files in the form the"
        " toolchain writes them, or add the generated ones to `.gitignore`,"
        " then finish."
    )


def deferred_baseline_failed_reason(baseline: object | None) -> str:
    """The incomplete reason for a baseline that stayed red after conflict resolution.

    When setup was deferred past unresolved rebase conflicts, the baseline
    check runs only after the model resolved them and its session has ended.
    A red deferred baseline ends the attempt ``incomplete`` (the final check
    would rerun the same gate on the same tree), and the next round must
    start by making the check pass on the resolved branch. The failure detail
    keeps the same form as the pre-model baseline-failure prompt section, so
    the two read consistently when both reach the next round.
    """

    return (
        "The rebase conflicts were resolved, but the baseline check run after"
        " the resolution failed; the next round must start by making the check"
        " pass on the resolved branch.\n\n"
        + _build_baseline_failure_section(baseline, after_conflict_resolution=True)
    )


def _select_failed_command(result: object):
    """Return the command execution the failure detail blames.

    The first command with a non-zero exit wins; when every command exited
    cleanly (a timeout leaves the trailing command with ``exit_code None``)
    the last command is blamed instead. ``None`` means no command ran.
    """

    commands = getattr(result, "commands", ())
    failed = next((command for command in commands if getattr(command, "exit_code", 0) != 0), None)
    if failed is None and commands:
        failed = commands[-1]
    return failed


#: Per-stream cap for baseline failure output quoted in the model starting
#: prompt. Matches the scale of the event-detail limit: bounded, but enough
#: for the model to diagnose the failure. Output is cut from the front so the
#: tail (usually the summary a linter prints last) survives.
_BASELINE_PROMPT_STREAM_LIMIT = 4000


def _tail_stream(output: object, *, limit: int = _BASELINE_PROMPT_STREAM_LIMIT) -> str:
    """Keep the end of a command stream, marking front-truncation visibly."""

    text = output if isinstance(output, str) else (output or "")
    if not text.strip():
        return "(no output)"
    if len(text) <= limit:
        return text
    return f"...[truncated; showing the last {limit} characters]\n{text[-limit:]}"


def _build_baseline_failure_section(
    baseline: object, *, after_conflict_resolution: bool = False
) -> str:
    """Render a red baseline for the continuation starting prompt.

    The baseline output is already credential-scrubbed by the command runner;
    this only selects and bounds what the model sees. Both streams are quoted:
    linters such as ``php-cs-fixer --dry-run --diff`` report on stdout, so
    stderr alone is not enough to diagnose the failure.

    The ``after_conflict_resolution`` form describes a deferred baseline that
    failed on the resolved branch (used by
    :func:`deferred_baseline_failed_reason`); the default form describes a
    pre-model baseline that failed before the attempt started.
    """

    header = "## Baseline check failure (fix before finishing)\n\n"
    if after_conflict_resolution:
        header += (
            "The baseline check — the same gate the final check uses — failed"
            " on the resolved branch after the rebase conflicts were resolved"
        )
        instruction = (
            "The conflicts are resolved, so the next round must start by making"
            " this check pass on the resolved branch: fix the underlying"
            " failure as part of that attempt rather than working around it."
        )
    else:
        header += (
            "The baseline check — the same gate the final check uses — failed on the "
            "current branch before this attempt started"
        )
        instruction = (
            "This failure most likely comes from the earlier attempt's own commits on "
            "this branch. You must pass this check before the work is considered done: "
            "fix the underlying failure as part of this attempt rather than working "
            "around it."
        )
    failed = _select_failed_command(baseline)
    if failed is None:
        return f"{header} (it timed out before any command ran).\n\n{instruction}"
    command = getattr(failed, "command", "")
    if getattr(failed, "timed_out", False) or getattr(failed, "exit_code", None) is None:
        result = "timed out"
    else:
        result = f"exit {getattr(failed, 'exit_code', None)}"
    return (
        f"{header}: `{command}` ({result}).\n\n"
        f"{instruction}\n\n"
        "Failing command stdout (tail):\n"
        f"```\n{_tail_stream(getattr(failed, 'stdout', ''))}\n```\n\n"
        "Failing command stderr (tail):\n"
        f"```\n{_tail_stream(getattr(failed, 'stderr', ''))}\n```"
    )


@dataclass(frozen=True)
class CompletionDecision:
    """Outcome or eligibility evidence for the publisher-owned final operations."""

    outcome: AttemptOutcome | None
    publication_eligible: bool
    publication_path: PublicationPath
    reasons: tuple[str, ...]


class VerificationOrderError(RuntimeError):
    """Raised when a caller skips the required model/review gate."""


class VerificationRunner:
    """Run setup, baseline, and final checks in their required lifecycle order."""

    def __init__(self, command_runner: CommandRunner) -> None:
        self._command_runner = command_runner
        self._baseline_passed = False
        self._review_completed = False

    def prepare(
        self,
        profile: RepositoryProfile,
        working_directory: Path,
        *,
        continuation: bool = False,
    ) -> PreparationEvidence:
        """Run setup and, only after success, an uncached baseline check."""

        self._baseline_passed = False
        self._review_completed = False
        setup = self._command_runner.run(
            profile.setup,
            timeout=profile.setup_timeout,
            cwd=working_directory,
            environment=profile.env,
        )
        if not setup.succeeded:
            self._baseline_passed = False
            return PreparationEvidence(setup=setup, baseline=None)
        baseline = self._run_check(profile, working_directory)
        self._baseline_passed = baseline.succeeded or continuation
        return PreparationEvidence(setup=setup, baseline=baseline)

    def mark_review_complete(
        self,
        *,
        model_status: ModelExecutionStatus,
        commit_count: int,
        review: ReviewEvidence,
    ) -> None:
        """Record observed model, commit, and final-review evidence before final check."""

        if not self._baseline_passed:
            raise VerificationOrderError("A successful baseline check is required before review completion.")
        if model_status is not ModelExecutionStatus.SUCCEEDED:
            raise VerificationOrderError("Final check requires successful model execution.")
        if commit_count <= 0:
            raise VerificationOrderError("Final check requires at least one local commit.")
        if review.repair_cycles > 2:
            raise VerificationOrderError("Final check requires a review within the repair-cycle limit.")
        self._review_completed = True

    def final_check(self, profile: RepositoryProfile, working_directory: Path) -> CommandResult:
        """Run the separate authoritative check after model and review work."""

        if not self._review_completed:
            raise VerificationOrderError("Final check requires completed model review.")
        return self._run_check(profile, working_directory)

    def _run_check(self, profile: RepositoryProfile, working_directory: Path) -> CommandResult:
        return self._command_runner.run(
            profile.check,
            timeout=profile.timeout,
            cwd=working_directory,
            environment=profile.env,
        )


class CompletionEvaluator:
    """Apply the completion contract without performing publication itself."""

    def __init__(self, blocking_severities: frozenset[str] | set[str]) -> None:
        self._blocking_severities = frozenset(blocking_severities)

    def evaluate(
        self,
        *,
        setup: CommandResult | None,
        baseline: CommandResult | None,
        model_status: ModelExecutionStatus | None,
        commit_count: int,
        acceptance_criteria_satisfied: bool,
        review: ReviewEvidence,
        final_check: CommandResult | None,
        push_succeeded: bool | None = None,
        pr_exists: bool | None = None,
        profile_error: ProfileError | None = None,
        continuation: bool = False,
        final_check_dirty_paths: Sequence[str] | None = None,
        deferred_baseline_failed: bool = False,
    ) -> CompletionDecision:
        """Classify observed evidence; a local pass is eligibility, not completion."""

        if profile_error is not None:
            return _decision(
                AttemptOutcome.INFRASTRUCTURE_ERROR,
                "Repository profile could not be loaded.",
                commit_count,
            )
        if setup is None or not setup.succeeded:
            return _decision(
                AttemptOutcome.INFRASTRUCTURE_ERROR, "Setup did not succeed.", commit_count
            )
        if baseline is None:
            return _decision(
                AttemptOutcome.INFRASTRUCTURE_ERROR,
                "Baseline check was not observed.",
                commit_count,
            )
        if not baseline.succeeded and not continuation:
            return _decision(
                AttemptOutcome.INCOMPLETE,
                "Baseline check failed before model execution.",
                commit_count,
            )
        if model_status is ModelExecutionStatus.INFRASTRUCTURE_ERROR:
            return _decision(
                AttemptOutcome.INFRASTRUCTURE_ERROR,
                "Model execution had an infrastructure error.",
                commit_count,
            )
        if model_status is ModelExecutionStatus.MODEL_LIMIT_REACHED:
            return _decision(
                AttemptOutcome.INCOMPLETE, "Model execution reached its limit.", commit_count
            )
        if model_status is ModelExecutionStatus.HANDOFF_REQUESTED:
            if commit_count <= 0:
                return _decision(
                    AttemptOutcome.NO_CHANGES,
                    "Handoff was requested without any commits on the branch.",
                    commit_count,
                )
            return CompletionDecision(
                outcome=AttemptOutcome.HANDOFF,
                publication_eligible=False,
                publication_path=PublicationPath.PARTIAL,
                reasons=("Handoff was requested with committed progress to preserve.",),
            )
        if model_status is ModelExecutionStatus.OPERATOR_HANDOFF_EXPIRED:
            if commit_count <= 0:
                return _decision(
                    AttemptOutcome.NO_CHANGES,
                    "Operator handoff reached its model deadline without any commits"
                    " on the branch.",
                    commit_count,
                )
            return CompletionDecision(
                outcome=AttemptOutcome.INCOMPLETE,
                publication_eligible=False,
                publication_path=PublicationPath.PARTIAL,
                reasons=(
                    "Operator handoff reached its model deadline without the model"
                    " invoking the handoff skill; preserved work is reported with"
                    " its ordinary outcome.",
                ),
            )
        if model_status is None:
            return _decision(
                AttemptOutcome.INCOMPLETE, "Model execution was not observed.", commit_count
            )
        if commit_count == 0:
            return _decision(
                AttemptOutcome.NO_CHANGES,
                "The skill workflow completed without commits.",
                commit_count,
            )
        if commit_count < 0:
            return _decision(
                AttemptOutcome.INFRASTRUCTURE_ERROR, "Commit count is invalid.", commit_count
            )
        if (
            deferred_baseline_failed
            and model_status is ModelExecutionStatus.SUCCEEDED
            and commit_count > 0
        ):
            # Setup was deferred past unresolved rebase conflicts, so the
            # baseline ran only after the model resolved them: the final check
            # would rerun the same gate on the same tree, and the next round
            # must start by making it pass.
            return _decision(
                AttemptOutcome.INCOMPLETE,
                deferred_baseline_failed_reason(baseline),
                commit_count,
            )
        if not acceptance_criteria_satisfied:
            return _decision(
                AttemptOutcome.INCOMPLETE,
                "Acceptance criteria are not satisfied.",
                commit_count,
            )
        blocking = tuple(finding for finding in review.findings if finding.severity in self._blocking_severities)
        if blocking:
            return _decision(
                AttemptOutcome.INCOMPLETE, "Final review has blocking findings.", commit_count
            )
        if review.repair_cycles > 2:
            return _decision(
                AttemptOutcome.INCOMPLETE,
                "The review repair-cycle limit was exceeded.",
                commit_count,
            )
        if final_check is None or not final_check.succeeded:
            return _decision(
                AttemptOutcome.INCOMPLETE,
                "Final authoritative check did not succeed.",
                commit_count,
            )
        if final_check_dirty_paths:
            return _decision(
                AttemptOutcome.INCOMPLETE,
                final_check_dirty_reason(tuple(sorted(final_check_dirty_paths))),
                commit_count,
            )

        if push_succeeded is False:
            return _decision(
                AttemptOutcome.INFRASTRUCTURE_ERROR,
                "Push failed after local completion evidence.",
                commit_count,
            )
        if pr_exists is False:
            return _decision(
                AttemptOutcome.INFRASTRUCTURE_ERROR,
                "Pull request creation failed after push.",
                commit_count,
            )
        if push_succeeded is True and pr_exists is True:
            return _decision(
                AttemptOutcome.COMPLETE, "All six completion conditions were observed.", commit_count
            )
        return CompletionDecision(
            outcome=None,
            publication_eligible=True,
            publication_path=PublicationPath.COMPLETE,
            reasons=("Local completion evidence is sufficient to begin publication.",),
        )


def _decision(outcome: AttemptOutcome, reason: str, commit_count: int) -> CompletionDecision:
    path = (
        PublicationPath.PARTIAL
        if outcome is AttemptOutcome.INCOMPLETE and commit_count > 0
        else PublicationPath.NONE
    )
    return CompletionDecision(
        outcome=outcome,
        publication_eligible=False,
        publication_path=path,
        reasons=(reason,),
    )
