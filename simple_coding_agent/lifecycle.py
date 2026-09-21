"""Single-process orchestration of one implementation attempt at a time."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
import time
from typing import Protocol

from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.completion import (
    AttemptOutcome,
    CompletionDecision,
    CompletionEvaluator,
    PublicationPath,
    ReviewEvidence,
    VerificationOrderError,
    VerificationRunner,
)
from simple_coding_agent.config import RepositoryProfile
from simple_coding_agent.github_tracker import Claim
from simple_coding_agent.model_execution import ModelExecutionStatus, ModelExecutor
from simple_coding_agent.observability import AttemptArchive
from simple_coding_agent.operating import ConsecutiveErrorStore
from simple_coding_agent.publication import PublicationRequest


class LifecycleStatus(StrEnum):
    """Whether a single polling iteration found work to attempt."""

    ATTEMPTED = "attempted"
    IDLE = "idle"


@dataclass(frozen=True)
class LifecycleResult:
    """The externally observable result of one polling iteration."""

    status: LifecycleStatus
    outcome: AttemptOutcome | None = None


@dataclass(frozen=True)
class AttemptEvidence:
    """The model/check/review evidence consumed by the publication boundary."""

    decision: CompletionDecision
    check_command: str
    check_exit_code: int | None
    review_cycles: int
    review_findings: str
    details: str


class AttemptTracker(Protocol):
    """Claim and idempotently release the GitHub state owned by this agent."""

    def claim_next(self) -> Claim | None: ...

    def recover_claim(self, issue_number: int) -> Claim | None: ...

    def release_attempt(self, issue_number: int, label: str, assignee_id: str) -> None: ...


class Workspace(Protocol):
    """Workspace operations needed by the orchestration seam."""

    working_directory: Path

    def prepare_attempt(self, *, base_branch: str, issue_number: int): ...

    def cleanup(self, *, base_branch: str, prepared: object, retain_branch: bool) -> None: ...


class Publisher(Protocol):
    """Publication boundary, including the result-comment operation."""

    def publish(self, request: PublicationRequest): ...


class ModelAttemptRunner:
    """Adapt setup, SDK execution, review evidence, and final checks to publication."""

    def __init__(
        self,
        *,
        attempt_state: AttemptStateStore,
        workspace: object,
        verifier: VerificationRunner,
        evaluator: CompletionEvaluator,
        model_executor: ModelExecutor,
        acceptance_criteria_satisfied: Callable[[Claim], bool] = lambda claim: True,
        attempt_archive_factory: Callable[[int, str], AttemptArchive] | None = None,
    ) -> None:
        self._attempt_state = attempt_state
        self._workspace = workspace
        self._verifier = verifier
        self._evaluator = evaluator
        self._model_executor = model_executor
        self._acceptance_criteria_satisfied = acceptance_criteria_satisfied
        self._attempt_archive_factory = attempt_archive_factory

    def __call__(self, claim: Claim, profile: RepositoryProfile, prepared: object) -> AttemptEvidence:
        """Run the local evidence gates in their mandated phase order."""

        checkpoint = self._attempt_state.read()
        archive = (
            self._attempt_archive_factory(claim.issue.number, checkpoint.started_at)
            if checkpoint is not None and self._attempt_archive_factory is not None
            else None
        )
        preparation = self._verifier.prepare(profile, getattr(self._workspace, "working_directory"))
        _archive_commands(archive, "setup", preparation.setup)
        if preparation.baseline is not None:
            _archive_commands(archive, "baseline_check", preparation.baseline)
        if not preparation.setup.succeeded or preparation.baseline is None or not preparation.baseline.succeeded:
            decision = self._evaluator.evaluate(
                setup=preparation.setup, baseline=preparation.baseline, model_status=None,
                commit_count=0, acceptance_criteria_satisfied=False,
                review=ReviewEvidence((), 0), final_check=None,
            )
            return _attempt_evidence(decision, profile, None, 0, "not run")

        self._attempt_state.transition(AttemptPhase.MODEL_RUNNING)
        execution = asyncio.run(
            self._model_executor.execute(
                issue_body=claim.issue.body, working_directory=getattr(self._workspace, "working_directory")
            )
        )
        if archive is not None:
            archive.write_attempt(
                {
                    "model_usage": execution.model_usage,
                    "skill_events": [event.__dict__ for event in execution.skill_events],
                    "model_stop_reason": execution.stop_reason,
                }
            )
        commits = getattr(self._workspace, "commits_added")(prepared)
        review_count = sum(
            event.name == "code-review" and event.phase == "PreToolUse"
            for event in execution.skill_events
        )
        review_cycles = max(0, review_count - 1)
        review = ReviewEvidence((), review_cycles)
        final_check = None
        if execution.status is ModelExecutionStatus.SUCCEEDED and commits:
            try:
                self._verifier.mark_review_complete(
                    model_status=execution.status, commit_count=len(commits), review=review
                )
                final_check = self._verifier.final_check(profile, getattr(self._workspace, "working_directory"))
            except VerificationOrderError:
                final_check = None
        if final_check is not None:
            _archive_commands(archive, "check", final_check)
        decision = self._evaluator.evaluate(
            setup=preparation.setup,
            baseline=preparation.baseline,
            model_status=execution.status,
            commit_count=len(commits),
            acceptance_criteria_satisfied=(
                review_count > 0 and self._acceptance_criteria_satisfied(claim)
            ),
            review=review,
            final_check=final_check,
        )
        if decision.publication_eligible or decision.publication_path is PublicationPath.PARTIAL:
            self._attempt_state.transition(AttemptPhase.PUSHING)
        findings = "all clear" if not review.findings else "; ".join(finding.summary for finding in review.findings)
        return _attempt_evidence(decision, profile, final_check, review_cycles, findings, execution.explanation)


class AgentLifecycle:
    """Connect claim, checkpoint, setup, publication, cleanup, and polling.

    The model/review handoff is intentionally injected: its structured evidence
    is produced by the model boundary, while this class owns lifecycle order.
    """

    def __init__(
        self,
        *,
        tracker: AttemptTracker,
        attempt_state: AttemptStateStore,
        workspace: Workspace,
        profile_loader: Callable[[Path], RepositoryProfile],
        publisher: Publisher,
        attempt_runner: Callable[[Claim, RepositoryProfile, object], AttemptEvidence] | None = None,
        poll_interval: int = 60,
        sleeper: Callable[[float], None] = time.sleep,
        error_store: ConsecutiveErrorStore | None = None,
        max_consecutive_errors: int = 3,
        event_log: Callable[[str, str], None] = lambda event, detail="": None,
        attempt_archive_factory: Callable[[int, str], AttemptArchive] | None = None,
    ) -> None:
        self._tracker = tracker
        self._attempt_state = attempt_state
        self._workspace = workspace
        self._profile_loader = profile_loader
        self._publisher = publisher
        self._attempt_runner = attempt_runner
        self._poll_interval = poll_interval
        self._sleeper = sleeper
        self._error_store = error_store
        self._max_consecutive_errors = max_consecutive_errors
        self._event_log = event_log
        self._attempt_archive_factory = attempt_archive_factory
        self._startup_reconciled = False

    def run_once(self) -> LifecycleResult:
        """Claim and process one issue, or sleep once when the queue is empty."""

        if not self._startup_reconciled:
            self._startup_reconciled = True
            recovered = self._reconcile_startup()
            if recovered is not None:
                return LifecycleResult(LifecycleStatus.ATTEMPTED, recovered)

        claim = self._tracker.claim_next()
        if claim is None:
            self._sleeper(self._poll_interval)
            return LifecycleResult(LifecycleStatus.IDLE)

        checkpoint = self._attempt_state.start(
            issue_number=claim.issue.number, branch=f"agent/issue-{claim.issue.number}"
        )
        archive = self._archive_for(claim.issue.number, checkpoint.started_at)
        prepared: object | None = None
        profile: RepositoryProfile | None = None
        outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
        retain_branch = False
        comment_posted = True
        try:
            prepared = self._workspace.prepare_attempt(base_branch="main", issue_number=claim.issue.number)
            profile = self._profile_loader(self._workspace.working_directory)
            if profile.base_branch != "main":
                self._workspace.cleanup(
                    base_branch="main", prepared=prepared, retain_branch=False
                )
                prepared = self._workspace.prepare_attempt(
                    base_branch=profile.base_branch, issue_number=claim.issue.number
                )
            self._attempt_state.transition(AttemptPhase.SETUP)
            if self._attempt_runner is None:
                outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
                published = self._publish_terminal(
                    claim, checkpoint.started_at, prepared, profile, outcome
                )
                comment_posted = getattr(published, "comment_posted", True)
            else:
                evidence = self._attempt_runner(claim, profile, prepared)
                published = self._publisher.publish(
                    PublicationRequest(
                        issue_number=claim.issue.number,
                        issue_title=claim.issue.title,
                        branch=getattr(prepared, "branch"),
                        started_at=checkpoint.started_at,
                        decision=evidence.decision,
                        check_command=evidence.check_command,
                        check_exit_code=evidence.check_exit_code,
                        review_cycles=evidence.review_cycles,
                        review_findings=evidence.review_findings,
                        details=evidence.details,
                        base_branch=profile.base_branch,
                    )
                )
                outcome = published.outcome
                comment_posted = getattr(published, "comment_posted", True)
                retain_branch = (
                    outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
                    and getattr(published, "branch_url", None) is None
                )
        except Exception as error:
            self._event_log("attempt_exception", _exception_detail(error))
            published = self._publish_terminal(
                claim, checkpoint.started_at, prepared, profile, outcome
            )
            comment_posted = getattr(published, "comment_posted", True)
        finally:
            remote_cleanup_complete = False
            try:
                if comment_posted:
                    self._tracker.release_attempt(
                        claim.issue.number, "ready-for-agent", claim.assignment.assignee_id
                    )
                    remote_cleanup_complete = True
            finally:
                try:
                    if prepared is not None:
                        self._workspace.cleanup(
                            base_branch=profile.base_branch if profile is not None else "main",
                            prepared=prepared,
                            retain_branch=retain_branch or not comment_posted,
                        )
                finally:
                    if remote_cleanup_complete:
                        self._attempt_state.delete()
        self._write_outcome(archive, outcome, checkpoint.started_at)
        self._record_terminal_outcome(outcome, checkpoint.started_at)
        return LifecycleResult(LifecycleStatus.ATTEMPTED, outcome)

    def _reconcile_startup(self) -> AttemptOutcome | None:
        """Finish one durable attempt without recreating its SDK execution context."""

        checkpoint = self._attempt_state.read()
        if checkpoint is None:
            return None
        claim = self._tracker.recover_claim(checkpoint.issue_number)
        if claim is None:
            raise RuntimeError("Interrupted attempt issue is unavailable for recovery")
        archive = self._archive_for(claim.issue.number, checkpoint.started_at)

        prepared: object | None = None
        profile: RepositoryProfile | None = None
        outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
        comment_posted = True
        retain_branch = False
        has_commits = False
        try:
            prepared = self._workspace.prepare_attempt(base_branch="main", issue_number=claim.issue.number)
            profile = self._profile_loader(self._workspace.working_directory)
            if profile.base_branch != "main":
                self._workspace.cleanup(base_branch="main", prepared=prepared, retain_branch=False)
                prepared = self._workspace.prepare_attempt(
                    base_branch=profile.base_branch, issue_number=claim.issue.number
                )
            if checkpoint.phase is AttemptPhase.MODEL_RUNNING:
                commits = getattr(self._workspace, "commits_added")(prepared)
                has_commits = bool(commits)
                if commits:
                    self._attempt_state.transition(AttemptPhase.PUSHING)
                    decision = CompletionDecision(
                        AttemptOutcome.INCOMPLETE, False, PublicationPath.PARTIAL,
                        ("Model execution was interrupted; preserved committed work.",),
                    )
                else:
                    decision = _infrastructure_decision("Model execution was interrupted without commits.")
            elif checkpoint.phase in (AttemptPhase.PUSHING, AttemptPhase.PUBLISHING):
                has_commits = bool(getattr(self._workspace, "commits_added")(prepared))
                decision = CompletionDecision(
                    None, True, PublicationPath.COMPLETE,
                    ("Recovered previously completed local evidence for publication.",),
                )
            else:
                decision = _infrastructure_decision("Attempt was interrupted before model execution.")
            published = self._publisher.publish(
                PublicationRequest(
                    issue_number=claim.issue.number,
                    issue_title=claim.issue.title,
                    branch=getattr(prepared, "branch", checkpoint.branch),
                    started_at=checkpoint.started_at,
                    decision=decision,
                    check_command="not rerun during startup recovery",
                    check_exit_code=None,
                    review_cycles=0,
                    review_findings="not rerun during startup recovery",
                    details=decision.reasons[0],
                    base_branch=profile.base_branch,
                )
            )
            outcome = published.outcome
            comment_posted = getattr(published, "comment_posted", True)
            retain_branch = (
                outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
                and getattr(published, "branch_url", None) is None
                and has_commits
            )
        except Exception as error:
            self._event_log("attempt_exception", _exception_detail(error))
            published = self._publish_terminal(
                claim, checkpoint.started_at, prepared, profile, AttemptOutcome.INFRASTRUCTURE_ERROR
            )
            outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
            comment_posted = getattr(published, "comment_posted", True)
        finally:
            remote_cleanup_complete = False
            try:
                if comment_posted:
                    self._tracker.release_attempt(
                        claim.issue.number, "ready-for-agent", claim.assignment.assignee_id
                    )
                    remote_cleanup_complete = True
            finally:
                try:
                    if prepared is not None:
                        self._workspace.cleanup(
                            base_branch=profile.base_branch if profile is not None else "main",
                            prepared=prepared,
                            retain_branch=retain_branch or not comment_posted,
                        )
                finally:
                    if remote_cleanup_complete:
                        self._attempt_state.delete()
        self._write_outcome(archive, outcome, checkpoint.started_at)
        self._record_terminal_outcome(outcome, checkpoint.started_at)
        return outcome

    def run_forever(self, *, stop: Callable[[], bool]) -> None:
        """Poll sequentially until the injected deterministic stop boundary fires."""

        while not stop():
            self.run_once()

    def _publish_terminal(
        self,
        claim: Claim,
        started_at: str,
        prepared: object | None,
        profile: RepositoryProfile | None,
        outcome: AttemptOutcome,
    ):
        branch = getattr(prepared, "branch", f"agent/issue-{claim.issue.number}")
        base_branch = profile.base_branch if profile is not None else "main"
        return self._publisher.publish(
            PublicationRequest(
                issue_number=claim.issue.number,
                issue_title=claim.issue.title,
                branch=branch,
                started_at=started_at,
                decision=CompletionDecision(outcome, False, PublicationPath.NONE, ("Attempt could not complete.",)),
                check_command="not run",
                check_exit_code=None,
                review_cycles=0,
                review_findings="not run",
                details="Repository profile could not be loaded or the attempt could not start.",
                base_branch=base_branch,
            )
        )

    def _record_terminal_outcome(self, outcome: AttemptOutcome, attempt_id: str) -> None:
        """Advance the process-wide guard only after attempt cleanup is complete."""

        if self._error_store is None:
            return
        count = self._error_store.record(outcome, attempt_id)
        if count >= self._max_consecutive_errors:
            self._event_log("consecutive_error_limit_reached")
            raise SystemExit(1)

    def _archive_for(self, issue_number: int, started_at: str) -> AttemptArchive | None:
        return (
            self._attempt_archive_factory(issue_number, started_at)
            if self._attempt_archive_factory is not None
            else None
        )

    @staticmethod
    def _write_outcome(
        archive: AttemptArchive | None, outcome: AttemptOutcome, started_at: str
    ) -> None:
        if archive is not None:
            completed_at = datetime.now(UTC)
            started = datetime.fromisoformat(started_at.removesuffix("Z") + "+00:00")
            archive.write_attempt(
                {
                    "outcome": outcome.value,
                    "started_at": started_at,
                    "completed_at": completed_at.isoformat().replace("+00:00", "Z"),
                    "duration_seconds": (completed_at - started).total_seconds(),
                }
            )


def _attempt_evidence(
    decision: CompletionDecision,
    profile: RepositoryProfile,
    final_check: object | None,
    review_cycles: int,
    review_findings: str,
    details: str = "Attempt did not reach local completion evidence.",
) -> AttemptEvidence:
    return AttemptEvidence(
        decision=decision,
        check_command=" && ".join(profile.check),
        check_exit_code=getattr(final_check, "exit_code", None),
        review_cycles=review_cycles,
        review_findings=review_findings,
        details=details,
    )


def _archive_commands(archive: AttemptArchive | None, name: str, result: object) -> None:
    """Keep setup, baseline, and final-check evidence in distinct artifacts."""

    if archive is None:
        return
    commands = getattr(result, "commands", ())
    stdout = "".join(getattr(command, "stdout", "") for command in commands)
    stderr = "".join(getattr(command, "stderr", "") for command in commands)
    archive.write_text(f"{name}_stdout.log", stdout)
    archive.write_text(f"{name}_stderr.log", stderr)


def _infrastructure_decision(reason: str) -> CompletionDecision:
    return CompletionDecision(
        AttemptOutcome.INFRASTRUCTURE_ERROR, False, PublicationPath.NONE, (reason,)
    )


def _exception_detail(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"
