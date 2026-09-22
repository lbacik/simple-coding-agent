"""Single-process orchestration of one implementation attempt at a time."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
import re
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
from simple_coding_agent.git_workspace import GitWorkspaceError, GitWorkspaceRecoveryError
from simple_coding_agent.github_tracker import ROUND_FINISHED, Claim, TrackerIssue
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
    note_commit_sha: str | None = None


class AttemptTracker(Protocol):
    """Claim and idempotently release the GitHub state owned by this agent."""

    def claim_next(self) -> Claim | None: ...

    def recover_claim(self, issue_number: int) -> Claim | None: ...

    def release_attempt(self, issue_number: int, label: str, assignee_id: str) -> None: ...

    def release_handoff(self, issue_number: int, assignee_id: str) -> None: ...


class Workspace(Protocol):
    """Workspace operations needed by the orchestration seam."""

    working_directory: Path

    def prepare_attempt(self, *, base_branch: str, issue_number: int): ...

    def cleanup(self, *, base_branch: str, prepared: object, retain_branch: bool) -> None: ...

    def commit_dirty_work(self, message: str) -> bool: ...


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
        issue_comments: Callable[[TrackerIssue], tuple[str, ...]] = lambda issue: (),
        attempt_archive_factory: Callable[[int, str], AttemptArchive] | None = None,
        event_log: Callable[..., None] = (
            lambda event, detail="", level="INFO", issue_number=None: None
        ),
    ) -> None:
        self._attempt_state = attempt_state
        self._workspace = workspace
        self._verifier = verifier
        self._evaluator = evaluator
        self._model_executor = model_executor
        self._acceptance_criteria_satisfied = acceptance_criteria_satisfied
        self._issue_comments = issue_comments
        self._attempt_archive_factory = attempt_archive_factory
        self._event_log = event_log

    def __call__(self, claim: Claim, profile: RepositoryProfile, prepared: object) -> AttemptEvidence:
        """Run the local evidence gates in their mandated phase order."""

        if not getattr(prepared, "restored_from_remote", False) and _continuation_expected(
            claim.issue, self._issue_comments
        ):
            branch = getattr(prepared, "branch", f"agent/issue-{claim.issue.number}")
            details = (
                f"Expected continuation branch `{branch}` was not found on origin, so it "
                "could not be restored. A human should inspect this issue and decide next steps."
            )
            self._event_log(
                "continuation_branch_missing", details, level="ERROR", issue_number=claim.issue.number
            )
            decision = CompletionDecision(AttemptOutcome.INCOMPLETE, False, PublicationPath.NONE, (details,))
            return _attempt_evidence(decision, profile, None, 0, "not run", details)

        checkpoint = self._attempt_state.read()
        archive = (
            self._attempt_archive_factory(claim.issue.number, checkpoint.started_at)
            if checkpoint is not None and self._attempt_archive_factory is not None
            else None
        )
        self._event_log("setup_started", "", level="INFO", issue_number=claim.issue.number)
        preparation = self._verifier.prepare(profile, getattr(self._workspace, "working_directory"))
        _archive_commands(archive, "setup", preparation.setup)
        if not preparation.setup.succeeded:
            self._event_log(
                "setup_failed",
                _command_failure_detail(preparation.setup),
                level="ERROR",
                issue_number=claim.issue.number,
            )
        else:
            self._event_log("setup_succeeded", "", level="INFO", issue_number=claim.issue.number)
        if preparation.baseline is not None:
            _archive_commands(archive, "baseline_check", preparation.baseline)
            if not preparation.baseline.succeeded:
                self._event_log(
                    "baseline_check_failed",
                    _command_failure_detail(preparation.baseline),
                    level="ERROR",
                    issue_number=claim.issue.number,
                )
            else:
                self._event_log(
                    "baseline_check_succeeded", "", level="INFO", issue_number=claim.issue.number
                )
        if not preparation.setup.succeeded or preparation.baseline is None or not preparation.baseline.succeeded:
            decision = self._evaluator.evaluate(
                setup=preparation.setup, baseline=preparation.baseline, model_status=None,
                commit_count=0, acceptance_criteria_satisfied=False,
                review=ReviewEvidence((), 0), final_check=None,
            )
            return _attempt_evidence(decision, profile, None, 0, "not run")

        self._attempt_state.transition(AttemptPhase.MODEL_RUNNING)
        # Snapshot before dispatch: any commit already on the branch at this
        # point is carried over from a resumed attempt, not produced by this
        # run. `commits_added` is read again after the model runs to count
        # what this run itself added.
        continuation = bool(getattr(self._workspace, "commits_added")(prepared))
        prompt_body = _build_starting_prompt(
            claim.issue.body,
            self._issue_comments(claim.issue),
            continuation=continuation,
            issue_number=claim.issue.number,
        )
        self._event_log(
            "model_dispatch_starting",
            f"continuation={continuation}",
            level="INFO",
            issue_number=claim.issue.number,
        )
        execution = asyncio.run(
            self._model_executor.execute(
                issue_body=prompt_body,
                working_directory=getattr(self._workspace, "working_directory"),
                issue_number=claim.issue.number,
                archive=archive,
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
        if execution.status is ModelExecutionStatus.HANDOFF_REQUESTED:
            # Always preserve dirty/untracked work as a commit rather than
            # letting it get silently discarded by cleanup()'s hard reset.
            # This can leave the note no longer the final commit; that is
            # caught below as an invalid handoff rather than papered over,
            # since a continuation must be able to trust that the note is
            # genuinely the last word on the branch.
            try:
                getattr(self._workspace, "commit_dirty_work")(
                    "Preserve uncommitted work before handoff"
                )
            except GitWorkspaceError as error:
                # A decision here (rather than letting this propagate to the
                # generic exception handler) routes through the normal
                # infrastructure_error retain_branch logic, so the branch and
                # whatever did commit are kept locally for inspection instead
                # of being deleted as an assumed-discardable failed attempt.
                decision = CompletionDecision(
                    outcome=AttemptOutcome.INFRASTRUCTURE_ERROR,
                    publication_eligible=False,
                    publication_path=PublicationPath.NONE,
                    reasons=(f"Failed to preserve dirty work before handoff: {error}",),
                )
                return _attempt_evidence(decision, profile, None, 0, "not run", str(error))
        commits = getattr(self._workspace, "commits_added")(prepared)
        if execution.status is ModelExecutionStatus.HANDOFF_REQUESTED:
            # The handoff note commit is not preserved work by itself; a
            # request that only produced the note downgrades to no_changes.
            effective_commit_count = sum(
                1 for commit in commits if not _is_handoff_note_commit(commit)
            )
        else:
            effective_commit_count = len(commits)
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
            if not final_check.succeeded:
                self._event_log(
                    "final_check_failed",
                    _command_failure_detail(final_check),
                    level="ERROR",
                    issue_number=claim.issue.number,
                )
        decision = self._evaluator.evaluate(
            setup=preparation.setup,
            baseline=preparation.baseline,
            model_status=execution.status,
            commit_count=effective_commit_count,
            acceptance_criteria_satisfied=(
                review_count > 0 and self._acceptance_criteria_satisfied(claim)
            ),
            review=review,
            final_check=final_check,
        )
        handoff_rejection_reason: str | None = None
        if decision.outcome is AttemptOutcome.HANDOFF:
            validation = _validate_handoff_note(
                commits,
                claim.issue.number,
                getattr(self._workspace, "working_directory"),
                getattr(prepared, "base_revision"),
            )
            if not validation.valid:
                decision = CompletionDecision(
                    outcome=AttemptOutcome.INFRASTRUCTURE_ERROR,
                    publication_eligible=False,
                    publication_path=PublicationPath.NONE,
                    reasons=(validation.reason,),
                )
                handoff_rejection_reason = validation.reason
        if decision.publication_eligible or decision.publication_path is PublicationPath.PARTIAL:
            self._attempt_state.transition(AttemptPhase.PUSHING)
        findings = "all clear" if not review.findings else "; ".join(finding.summary for finding in review.findings)
        details = execution.explanation
        note_commit_sha = None
        if decision.outcome is AttemptOutcome.HANDOFF:
            details = _read_handoff_note(
                getattr(self._workspace, "working_directory"), claim.issue.number
            )
            if commits and _is_handoff_note_commit(commits[0]):
                note_commit_sha = commits[0].revision
        elif handoff_rejection_reason is not None:
            details = handoff_rejection_reason
        return _attempt_evidence(
            decision, profile, final_check, review_cycles, findings, details, note_commit_sha
        )


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
        event_log: Callable[..., None] = (
            lambda event, detail="", level="INFO", issue_number=None: None
        ),
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

        self._event_log("polling_for_issue", "", level="INFO")
        claim = self._tracker.claim_next()
        if claim is None:
            self._event_log(
                "no_eligible_issue_found",
                f"no ready-for-agent issue available; sleeping {self._poll_interval}s",
                level="INFO",
            )
            self._sleeper(self._poll_interval)
            return LifecycleResult(LifecycleStatus.IDLE)

        self._event_log(
            "issue_claimed",
            f"title={claim.issue.title!r}",
            level="INFO",
            issue_number=claim.issue.number,
        )
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
            self._event_log(
                "workspace_prepared",
                f"branch={getattr(prepared, 'branch', None)}; base_branch=main",
                level="INFO",
                issue_number=claim.issue.number,
            )
            profile = self._profile_loader(self._workspace.working_directory)
            self._event_log(
                "profile_loaded",
                f"base_branch={profile.base_branch}",
                level="INFO",
                issue_number=claim.issue.number,
            )
            if profile.base_branch != "main":
                self._workspace.cleanup(
                    base_branch="main", prepared=prepared, retain_branch=False
                )
                prepared = self._workspace.prepare_attempt(
                    base_branch=profile.base_branch, issue_number=claim.issue.number
                )
                self._event_log(
                    "workspace_reprepared",
                    f"branch={getattr(prepared, 'branch', None)}; base_branch={profile.base_branch}",
                    level="INFO",
                    issue_number=claim.issue.number,
                )
            self._attempt_state.transition(AttemptPhase.SETUP)
            self._event_log(
                "attempt_phase_transitioned",
                f"phase={AttemptPhase.SETUP.value}",
                level="INFO",
                issue_number=claim.issue.number,
            )
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
                        note_commit_sha=evidence.note_commit_sha,
                    )
                )
                outcome = published.outcome
                comment_posted = getattr(published, "comment_posted", True)
                retain_branch = (
                    outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
                    and getattr(published, "branch_url", None) is None
                )
        except GitWorkspaceRecoveryError as error:
            self._event_log(
                "git_workspace_unrecoverable",
                _exception_detail(error),
                level="ERROR",
                issue_number=claim.issue.number,
            )
            comment_posted = False
            prepared = None
            raise SystemExit(1) from error
        except Exception as error:
            self._event_log(
                "attempt_exception", _exception_detail(error), level="ERROR", issue_number=claim.issue.number
            )
            published = self._publish_terminal(
                claim, checkpoint.started_at, prepared, profile, outcome
            )
            comment_posted = getattr(published, "comment_posted", True)
        finally:
            remote_cleanup_complete = False
            try:
                if comment_posted:
                    self._release(claim, outcome)
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
        self._record_terminal_outcome(outcome, checkpoint.started_at, issue_number=claim.issue.number)
        return LifecycleResult(LifecycleStatus.ATTEMPTED, outcome)

    def _release(self, claim: Claim, outcome: AttemptOutcome) -> None:
        """Apply the label/assignee release sequence required by the outcome.

        A handoff must add ``round-finished`` for human-controlled requeue
        instead of the plain ready-for-agent/assignee release every other
        outcome uses.
        """

        if outcome is AttemptOutcome.HANDOFF:
            self._tracker.release_handoff(claim.issue.number, claim.assignment.assignee_id)
        else:
            self._tracker.release_attempt(
                claim.issue.number, "ready-for-agent", claim.assignment.assignee_id
            )

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
                recovered_commits = getattr(self._workspace, "commits_added")(prepared)
                has_commits = bool(recovered_commits)
                if _is_handoff_recovery(recovered_commits):
                    validation = _validate_handoff_note(
                        recovered_commits,
                        claim.issue.number,
                        self._workspace.working_directory,
                        prepared.base_revision,
                    )
                    if validation.valid:
                        decision = CompletionDecision(
                            AttemptOutcome.HANDOFF, False, PublicationPath.PARTIAL,
                            ("Recovered a previously committed handoff note for publication.",),
                        )
                    else:
                        decision = CompletionDecision(
                            AttemptOutcome.INFRASTRUCTURE_ERROR, False, PublicationPath.NONE,
                            (validation.reason,),
                        )
                else:
                    decision = CompletionDecision(
                        None, True, PublicationPath.COMPLETE,
                        ("Recovered previously completed local evidence for publication.",),
                    )
            else:
                decision = _infrastructure_decision("Attempt was interrupted before model execution.")
            details = decision.reasons[0]
            note_commit_sha = None
            if decision.outcome is AttemptOutcome.HANDOFF:
                details = _read_handoff_note(self._workspace.working_directory, claim.issue.number)
                note_commit_sha = recovered_commits[0].revision
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
                    details=details,
                    base_branch=profile.base_branch,
                    note_commit_sha=note_commit_sha,
                )
            )
            outcome = published.outcome
            comment_posted = getattr(published, "comment_posted", True)
            retain_branch = (
                outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
                and getattr(published, "branch_url", None) is None
                and has_commits
            )
        except GitWorkspaceRecoveryError as error:
            self._event_log(
                "git_workspace_unrecoverable",
                _exception_detail(error),
                level="ERROR",
                issue_number=claim.issue.number,
            )
            comment_posted = False
            prepared = None
            raise SystemExit(1) from error
        except Exception as error:
            self._event_log(
                "attempt_exception", _exception_detail(error), level="ERROR", issue_number=claim.issue.number
            )
            published = self._publish_terminal(
                claim, checkpoint.started_at, prepared, profile, AttemptOutcome.INFRASTRUCTURE_ERROR
            )
            outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
            comment_posted = getattr(published, "comment_posted", True)
        finally:
            remote_cleanup_complete = False
            try:
                if comment_posted:
                    self._release(claim, outcome)
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
        self._record_terminal_outcome(outcome, checkpoint.started_at, issue_number=claim.issue.number)
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

    def _record_terminal_outcome(
        self, outcome: AttemptOutcome, attempt_id: str, *, issue_number: int | None = None
    ) -> None:
        """Advance the process-wide guard only after attempt cleanup is complete."""

        if self._error_store is None:
            return
        count = self._error_store.record(outcome, attempt_id)
        if count >= self._max_consecutive_errors:
            self._event_log("consecutive_error_limit_reached", level="ERROR", issue_number=issue_number)
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
    note_commit_sha: str | None = None,
) -> AttemptEvidence:
    return AttemptEvidence(
        decision=decision,
        check_command=" && ".join(profile.check),
        check_exit_code=getattr(final_check, "exit_code", None),
        review_cycles=review_cycles,
        review_findings=review_findings,
        details=details,
        note_commit_sha=note_commit_sha,
    )


def _command_failure_detail(result: object, *, limit: int = 4000) -> str:
    """Surface the failing command and its stderr for the live event stream."""

    commands = getattr(result, "commands", ())
    failed = next((command for command in commands if getattr(command, "exit_code", 0) != 0), None)
    if failed is None and commands:
        failed = commands[-1]
    if failed is None:
        return "timed out before any command ran"
    stderr = (getattr(failed, "stderr", "") or "").strip()
    detail = f"`{getattr(failed, 'command', '')}` (exit {getattr(failed, 'exit_code', None)}): {stderr}"
    return detail[:limit]


def _archive_commands(archive: AttemptArchive | None, name: str, result: object) -> None:
    """Keep setup, baseline, and final-check evidence in distinct artifacts."""

    if archive is None:
        return
    commands = getattr(result, "commands", ())
    stdout = "".join(getattr(command, "stdout", "") for command in commands)
    stderr = "".join(getattr(command, "stderr", "") for command in commands)
    archive.write_text(f"{name}_stdout.log", stdout)
    archive.write_text(f"{name}_stderr.log", stderr)


def _read_handoff_note(working_directory: Path, issue_number: int) -> str:
    """Read the model-authored handoff note verbatim, for the result comment.

    The comment copy is the durable, human-visible record; the file on the
    branch is what a continuation is pointed at (see _build_starting_prompt).
    Missing is reported rather than raised: the note is written by the model
    in the SDK session this function has no control over.
    """

    note_path = working_directory / ".agent" / "handoff" / f"{issue_number}.md"
    try:
        return note_path.read_text()
    except OSError:
        return "Handoff was requested, but the handoff note file could not be read."


def _build_starting_prompt(
    issue_body: str, comments: Sequence[str], *, continuation: bool, issue_number: int
) -> str:
    """Compose the model's starting context from the issue body and trusted comments.

    Trusted comments are appended verbatim in the chronological order the
    tracker already returns them in. A continuation adds only a pointer
    sentence: the handoff note's own content is never fetched or digested
    here, since it is already reachable on the resumed branch.
    """

    parts = [issue_body, *comments]
    if continuation:
        parts.append(
            "This is a continued attempt; if it exists, the handoff note from the "
            f"previous attempt is at .agent/handoff/{issue_number}.md."
        )
    return "\n\n".join(parts)


_HANDOFF_NOTE_SUBJECT_PREFIX = "Handoff note:"

_HANDOFF_RESULT_MARKERS = (
    "Agent Attempt Result: incomplete",
    "Agent Attempt Result: handoff",
)


def _continuation_expected(
    issue: TrackerIssue, issue_comments: Callable[[TrackerIssue], tuple[str, ...]]
) -> bool:
    """Whether this issue carries (or carried) a signal that a continuation was expected.

    The ``round-finished`` label is the current signal; a prior incomplete or
    handoff attempt-result comment is the historical one, since either is what
    a handoff publishes. Checked in that order so a present label never
    triggers a needless comment fetch, and so a real handoff comment is still
    recognized once the label has been removed (e.g. on claim) or the remote
    branch is missing.
    """

    if ROUND_FINISHED in issue.labels:
        return True
    return any(
        marker in comment for comment in issue_comments(issue) for marker in _HANDOFF_RESULT_MARKERS
    )


def _is_handoff_note_commit(commit: object) -> bool:
    """Whether a commit is the handoff note commit, by its fixed subject prefix."""

    return getattr(commit, "subject", "").startswith(_HANDOFF_NOTE_SUBJECT_PREFIX)


def _is_handoff_recovery(commits: Sequence[object]) -> bool:
    """Whether the most recent commit on the branch is a handoff note commit.

    Only the most recent commit is checked: the note is the required final
    commit of a handoff attempt, so its presence there (rather than anywhere
    in history) is what distinguishes a recovered handoff from a recovered
    ordinary complete attempt.
    """

    if not commits:
        return False
    return _is_handoff_note_commit(commits[0])


_ALLOWED_HANDOFF_REASONS = frozenset({"cost_soft_threshold", "turn_limit", "time_limit"})

_HANDOFF_NOTE_TITLE_PATTERN = re.compile(r"^#\s*Handoff note:\s*issue #(\d+)")
_HANDOFF_NOTE_FIELD_PATTERN = re.compile(r"^-\s*(issue|started_at|reason|last_work_commit):\s*(.+?)\s*$")


@dataclass(frozen=True)
class _HandoffNoteValidation:
    """Whether a handoff note commit is trustworthy enough to publish."""

    valid: bool
    reason: str = ""


def _validate_handoff_note(
    commits: Sequence[object],
    issue_number: int,
    working_directory: Path,
    base_revision: str,
) -> _HandoffNoteValidation:
    """Confirm a handoff has a trustworthy, committed note before it may publish.

    Recognizing a commit by its subject prefix only proves a commit with that
    subject exists; it does not prove the note file was actually written,
    belongs to this issue and attempt, or reflects the work actually
    preserved. All of that is checked here, since a continuation trusts the
    note and a human trusts the ``round-finished`` label that only a valid
    handoff may add.
    """

    if not commits or not _is_handoff_note_commit(commits[0]):
        return _HandoffNoteValidation(
            False,
            "Handoff was requested, but no committed handoff note is the final "
            "commit on the branch.",
        )
    # The note commit being the newest commit (just checked above) means the
    # checked-out working tree is clean and matches that commit's tree, so
    # reading the file here reads the committed blob, not uncommitted state.
    note_path = working_directory / ".agent" / "handoff" / f"{issue_number}.md"
    try:
        text = note_path.read_text()
    except OSError:
        return _HandoffNoteValidation(
            False,
            "Handoff was requested, but the handoff note file could not be read "
            "from the committed branch state.",
        )
    fields = _parse_handoff_note_fields(text)
    if fields.get("title_issue") != str(issue_number):
        return _HandoffNoteValidation(
            False, "Handoff note does not carry the required title for this issue."
        )
    if fields.get("issue") != str(issue_number):
        return _HandoffNoteValidation(
            False, "Handoff note is missing a matching `issue` field."
        )
    started_at = fields.get("started_at", "")
    if not started_at or not _is_iso8601(started_at):
        return _HandoffNoteValidation(
            False, "Handoff note is missing a valid `started_at` field."
        )
    if fields.get("reason") not in _ALLOWED_HANDOFF_REASONS:
        return _HandoffNoteValidation(
            False, "Handoff note is missing a valid `reason` field."
        )
    last_work_commit = fields.get("last_work_commit", "")
    expected_last_work_commit = commits[1].revision if len(commits) > 1 else base_revision
    if (
        len(last_work_commit) < 7
        or not expected_last_work_commit.startswith(last_work_commit)
    ):
        return _HandoffNoteValidation(
            False,
            "Handoff note's `last_work_commit` does not match the last preserved "
            "work commit.",
        )
    return _HandoffNoteValidation(True)


def _parse_handoff_note_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        title_match = _HANDOFF_NOTE_TITLE_PATTERN.match(line)
        if title_match and "title_issue" not in fields:
            fields["title_issue"] = title_match.group(1)
            continue
        field_match = _HANDOFF_NOTE_FIELD_PATTERN.match(line)
        if field_match:
            fields[field_match.group(1)] = field_match.group(2)
    return fields


def _is_iso8601(value: str) -> bool:
    candidate = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return True


def _infrastructure_decision(reason: str) -> CompletionDecision:
    return CompletionDecision(
        AttemptOutcome.INFRASTRUCTURE_ERROR, False, PublicationPath.NONE, (reason,)
    )


def _exception_detail(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"
