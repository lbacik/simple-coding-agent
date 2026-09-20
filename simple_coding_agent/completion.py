"""Deterministic setup/check evidence and attempt-completion evaluation."""

from __future__ import annotations

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

    def prepare(self, profile: RepositoryProfile, working_directory: Path) -> PreparationEvidence:
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
        self._baseline_passed = baseline.succeeded
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
        if not baseline.succeeded:
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
