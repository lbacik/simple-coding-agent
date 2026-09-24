"""Single-process orchestration of one implementation attempt at a time."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
import os
from pathlib import Path
import re
import signal
import threading
import time
from typing import Protocol

from simple_coding_agent.attempt_state import (
    AttemptCheckpoint,
    AttemptPhase,
    AttemptStateError,
    AttemptStateStore,
)
from simple_coding_agent.control import (
    ActiveAttemptInfo,
    CommandRecord,
    ControlStore,
    ControlStoreError,
    IntakeState,
    RecoveryHold,
    build_status,
)
from simple_coding_agent.completion import (
    AttemptOutcome,
    CompletionDecision,
    CompletionEvaluator,
    PreparationEvidence,
    PublicationPath,
    ReviewEvidence,
    VerificationOrderError,
    VerificationRunner,
)
from simple_coding_agent.config import RepositoryProfile
from simple_coding_agent.finalization import AttemptCompletionStore, CompletionStoreError
from simple_coding_agent.git_workspace import (
    DirtyWorkspaceError,
    GitWorkspaceError,
    GitWorkspaceRecoveryError,
    RebaseConflictError,
)
from simple_coding_agent.github_tracker import ROUND_FINISHED, Claim, TrackerIssue
from simple_coding_agent.model_execution import ModelExecutionStatus, ModelExecutor
from simple_coding_agent.observability import AttemptArchive
from simple_coding_agent.operating import ConsecutiveErrorStore
from simple_coding_agent.publication import PublicationRequest


class AttemptInterruptionHandler:
    """Turn SIGTERM/SIGINT into one terminal event instead of a silent death.

    The interpreter's default SIGTERM action terminates the process without
    running any handler, and ``except Exception`` does not cover
    KeyboardInterrupt — either gap lets a stop/restart during the final check
    end the attempt with no issue-tagged event at all. This handler emits a
    single best-effort ``attempt_interrupted`` event, then resignals the
    process so the usual exit code/signal semantics are preserved.
    """

    def __init__(
        self,
        *,
        event_log: Callable[..., None],
        issue_number_provider: Callable[[], int | None],
        resignal: Callable[[int], None] | None = None,
    ) -> None:
        self._event_log = event_log
        self._issue_number_provider = issue_number_provider
        self._resignal = resignal or _resignal_self

    def install(
        self, signals: Sequence[int] = (signal.SIGTERM, signal.SIGINT)
    ) -> Callable[[], None]:
        """Install the handler; return a callable restoring prior handlers."""

        previous = {signum: signal.signal(signum, self) for signum in signals}

        def restore() -> None:
            for signum, handler in previous.items():
                signal.signal(signum, handler)

        return restore

    def __call__(self, signum: int, frame: object) -> None:
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        try:
            issue_number = self._issue_number_provider()
        except Exception:
            issue_number = None
        try:
            self._event_log(
                "attempt_interrupted",
                f"Process received {name} during the attempt.",
                level="ERROR",
                issue_number=issue_number,
            )
        except Exception:
            pass
        for candidate in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(candidate, signal.SIG_DFL)
            except Exception:
                pass
        self._resignal(signum)


def _resignal_self(signum: int) -> None:
    os.kill(os.getpid(), signum)


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

    def prepare_for_profile_read(self, *, base_branch: str = "main"): ...

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
        initial_commits = (
            getattr(self._workspace, "commits_added")(prepared)
            if hasattr(self._workspace, "commits_added")
            else ()
        )
        continuation = bool(initial_commits)
        # A retained branch rebased onto its base can be left with conflict
        # markers for the model to resolve. Setup commands parse every source
        # file, so they must never run against that half-rebased tree; they
        # are deferred until after the model resolves the conflicts.
        setup_deferred = _has_unresolved_conflicts(self._workspace)
        preparation: PreparationEvidence | None = None
        if setup_deferred:
            self._event_log(
                "setup_deferred_unresolved_conflicts",
                f"The attempt branch holds unresolved merge conflicts from rebasing onto "
                f"`{profile.base_branch}`; setup and the baseline check run after the "
                "model resolves them.",
                level="WARNING",
                issue_number=claim.issue.number,
            )
        else:
            early = self._run_preparation(
                claim, profile, archive, continuation=continuation, deferred=False
            )
            if isinstance(early, AttemptEvidence):
                return early
            preparation = early

        self._attempt_state.transition(AttemptPhase.MODEL_RUNNING)
        # Snapshot before dispatch: any commit already on the branch at this
        # point is carried over from a resumed attempt, not produced by this
        # run. `commits_added` is read again after the model runs to count
        # what this run itself added.
        prompt_body = _build_starting_prompt(
            claim.issue.body,
            self._issue_comments(claim.issue),
            continuation=continuation,
            issue_number=claim.issue.number,
        )
        if setup_deferred:
            prompt_body = (
                f"{prompt_body}\n\nThe attempt branch has unresolved merge conflicts from rebasing onto "
                f"`{profile.base_branch}`. Resolve them first (inspect with git status, edit the conflicted "
                "files, stage with git add, and run git rebase --continue), commit the resolution, then "
                "continue the implementation."
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
        if execution.status in (
            ModelExecutionStatus.HANDOFF_REQUESTED,
            ModelExecutionStatus.MODEL_LIMIT_REACHED,
            ModelExecutionStatus.SUCCEEDED,
        ):
            # Always preserve dirty/untracked work as a commit rather than
            # letting it get silently discarded. This also covers a model
            # that reports success but stops before its own final commit:
            # without this, commit_count and the final check would be
            # evaluated against work that never makes it into the push.
            commit_fn = getattr(self._workspace, "commit_dirty_work", None)
            if callable(commit_fn):
                message = (
                    "Preserve uncommitted work before handoff"
                    if execution.status is not ModelExecutionStatus.SUCCEEDED
                    else "Preserve uncommitted work left after model completion"
                )
                try:
                    commit_fn(message)
                except GitWorkspaceError as error:
                    decision = CompletionDecision(
                        outcome=AttemptOutcome.INFRASTRUCTURE_ERROR,
                        publication_eligible=False,
                        publication_path=PublicationPath.NONE,
                        reasons=(f"Failed to preserve dirty work after model completion: {error}",),
                    )
                    return _attempt_evidence(decision, profile, None, 0, "not run", str(error))
        if setup_deferred:
            # Setup was skipped before the model so it would not parse a
            # half-rebased tree; run it now that the model had its chance to
            # resolve the conflicts. The deferred baseline is not a true
            # pre-work baseline, so its failure stays tolerated.
            early = self._run_preparation(
                claim, profile, archive, continuation=True, deferred=True
            )
            if isinstance(early, AttemptEvidence):
                return early
            preparation = early
        commits = getattr(self._workspace, "commits_added")(prepared)
        if execution.status is ModelExecutionStatus.MODEL_LIMIT_REACHED:
            # Emergency handoff: synthesize and commit a handoff note so
            # progress is preserved for human review with round-finished.
            if not commits or not _is_handoff_note_commit(commits[0]):
                note_content = _build_emergency_handoff_note(
                    issue_number=claim.issue.number,
                    started_at=checkpoint.started_at,
                    reason=_classify_handoff_reason(execution),
                    last_work_commit=getattr(commits[0], "revision", str(commits[0])) if commits else getattr(prepared, "base_revision", "0" * 40),
                    explanation=execution.explanation,
                )
                working_dir = getattr(self._workspace, "working_directory", None)
                if working_dir is not None:
                    try:
                        note_path = working_dir / ".agent" / "handoff" / f"{claim.issue.number}.md"
                        note_path.parent.mkdir(parents=True, exist_ok=True)
                        note_path.write_text(note_content)
                    except OSError:
                        pass
                commit_fn = getattr(self._workspace, "commit_dirty_work", None)
                if callable(commit_fn):
                    commit_fn(f"Handoff note: issue #{claim.issue.number}")
                commits = getattr(self._workspace, "commits_added")(prepared)
            execution = replace(execution, status=ModelExecutionStatus.HANDOFF_REQUESTED)
        commit_count = len(commits)
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
                self._event_log(
                    "final_check_started",
                    f"check={' && '.join(profile.check)}",
                    level="INFO",
                    issue_number=claim.issue.number,
                )
                if archive is not None:
                    archive.write_attempt({"final_check_started": _utc_timestamp()})
                final_check = self._verifier.final_check(profile, getattr(self._workspace, "working_directory"))
                if archive is not None:
                    archive.write_attempt({"final_check_finished": _utc_timestamp()})
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
            else:
                self._event_log(
                    "final_check_succeeded", "", level="INFO", issue_number=claim.issue.number
                )
        try:
            decision = self._evaluator.evaluate(
                setup=preparation.setup,
                baseline=preparation.baseline,
                model_status=execution.status,
                commit_count=commit_count,
                acceptance_criteria_satisfied=(
                    review_count > 0 and self._acceptance_criteria_satisfied(claim)
                ),
                review=review,
                final_check=final_check,
                continuation=continuation,
            )
        except TypeError:
            decision = self._evaluator.evaluate(
                setup=preparation.setup,
                baseline=preparation.baseline,
                model_status=execution.status,
                commit_count=commit_count,
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

    def _run_preparation(
        self,
        claim: Claim,
        profile: RepositoryProfile,
        archive: AttemptArchive | None,
        *,
        continuation: bool,
        deferred: bool,
    ) -> PreparationEvidence | AttemptEvidence:
        """Run setup and the baseline check once; return early evidence on failure."""

        deferral = " (deferred until conflicts were resolved)" if deferred else ""
        self._event_log(
            "setup_started", deferral.strip(), level="INFO", issue_number=claim.issue.number
        )
        try:
            preparation = self._verifier.prepare(
                profile,
                getattr(self._workspace, "working_directory"),
                continuation=continuation,
            )
        except TypeError:
            preparation = self._verifier.prepare(
                profile,
                getattr(self._workspace, "working_directory"),
            )
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
                    level="WARNING" if continuation else "ERROR",
                    issue_number=claim.issue.number,
                )
            else:
                self._event_log(
                    "baseline_check_succeeded", "", level="INFO", issue_number=claim.issue.number
                )
        if (
            not preparation.setup.succeeded
            or preparation.baseline is None
            or (not preparation.baseline.succeeded and not continuation)
        ):
            try:
                decision = self._evaluator.evaluate(
                    setup=preparation.setup,
                    baseline=preparation.baseline,
                    model_status=None,
                    commit_count=0,
                    acceptance_criteria_satisfied=False,
                    review=ReviewEvidence((), 0),
                    final_check=None,
                    continuation=continuation,
                )
            except TypeError:
                decision = self._evaluator.evaluate(
                    setup=preparation.setup,
                    baseline=preparation.baseline,
                    model_status=None,
                    commit_count=0,
                    acceptance_criteria_satisfied=False,
                    review=ReviewEvidence((), 0),
                    final_check=None,
                )
            details = decision.reasons[0] if decision.reasons else None
            if not preparation.setup.succeeded:
                failure_detail = _command_failure_detail(preparation.setup)
            elif preparation.baseline is not None and not preparation.baseline.succeeded:
                failure_detail = _command_failure_detail(preparation.baseline)
            else:
                failure_detail = None
            if details and failure_detail:
                details = f"{details} {failure_detail}"
            elif failure_detail:
                details = failure_detail
            if details:
                return _attempt_evidence(decision, profile, None, 0, "not run", details)
            return _attempt_evidence(decision, profile, None, 0, "not run")
        return preparation


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
        completion_store: AttemptCompletionStore | None = None,
        control_store: ControlStore | None = None,
        repository: str = "",
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
        self._completion_store = completion_store
        self._control_store = control_store
        self._repository = repository
        # The claim boundary and command acceptance share this lock: either
        # a stop commits first and no claim begins, or a claim begins first
        # and the stop applies to the resulting active attempt.
        self._control_lock = (
            control_store.lock if control_store is not None else threading.RLock()
        )
        self._startup_reconciled = False
        self._recovering = False
        self._active_issue_number: int | None = None
        # True while this process is working through a claimed attempt (from
        # the serialized claim until its processing finishes, including
        # startup reconciliation). A checkpoint without in-progress work is a
        # retained hold: resume must be rejected until recovery finishes it.
        self._attempt_processing = False

    @property
    def active_issue_number(self) -> int | None:
        """The issue owned by the in-flight attempt, for interruption reporting."""

        return self._active_issue_number

    @property
    def repository(self) -> str:
        """The configured ``TARGET_REPO`` identifying this agent instance."""

        return self._repository

    def set_attempt_runner(
        self,
        attempt_runner: Callable[[Claim, RepositoryProfile, object], AttemptEvidence] | None,
    ) -> None:
        """Replace the injected model/review workflow (a test seam)."""

        self._attempt_runner = attempt_runner

    # -- operator control -------------------------------------------------

    def submit_stop(self, request_id: str) -> CommandRecord:
        """Durably record ``stop`` against the current attempt liveness."""

        if self._control_store is None:
            raise ControlStoreError("Operator control is not configured")
        with self._control_lock:
            has_active_attempt = self._attempt_state.read() is not None
            return self._control_store.submit_stop(
                request_id, has_active_attempt=has_active_attempt
            )

    def submit_stop_after(self, request_id: str, after: object) -> CommandRecord:
        """Durably record ``stop --after N`` against the current attempt liveness.

        The attempt active at acceptance, if any, is count one under the new
        plan; otherwise counting starts with the next attempt. A replacement
        plan starts a fresh count without interrupting the active attempt.
        Invalid counts and a stopped instance are rejected without changing
        an existing plan.
        """

        if self._control_store is None:
            raise ControlStoreError("Operator control is not configured")
        with self._control_lock:
            checkpoint = self._attempt_state.read()
            return self._control_store.submit_stop_after(
                request_id,
                after,
                has_active_attempt=checkpoint is not None,
                active_attempt_id=checkpoint.started_at if checkpoint is not None else None,
            )

    def submit_resume(self, request_id: str) -> CommandRecord:
        """Durably record ``resume`` and permit later issue intake.

        A pending stop plan is replaced without interrupting the active
        attempt, which continues with its ordinary outcome. While startup
        reconciliation or retained work/finalization remains unresolved the
        command is rejected with the affected attempt and reason, and intake
        is unchanged. Resume never requeues an issue: the next claim still
        goes through the serialized intake check and the ordinary runnable
        queue, so a ``round-finished`` issue stays ineligible until a
        separate human-approved requeue.
        """

        if self._control_store is None:
            raise ControlStoreError("Operator control is not configured")
        with self._control_lock:
            checkpoint = self._attempt_state.read()
            return self._control_store.submit_resume(
                request_id,
                has_active_attempt=checkpoint is not None,
                recovery_hold=self._recovery_hold_locked(checkpoint),
            )

    def _recovery_hold_locked(
        self, checkpoint: AttemptCheckpoint | None
    ) -> RecoveryHold | None:
        """Return the hold blocking ``resume``, if any (caller holds the lock)."""

        if self._recovering:
            if checkpoint is not None:
                return RecoveryHold(
                    issue_number=checkpoint.issue_number,
                    attempt_id=checkpoint.started_at,
                    branch=checkpoint.branch,
                    phase=checkpoint.phase.value,
                    reason=(
                        "startup reconciliation is still running for"
                        f" issue #{checkpoint.issue_number}; intake is held"
                        " until its outcome, publication, release, cleanup,"
                        " and accounting are durable"
                    ),
                )
            return RecoveryHold(
                issue_number=None,
                attempt_id=None,
                branch=None,
                phase=None,
                reason="startup reconciliation is still running",
            )
        if checkpoint is not None and not self._attempt_processing:
            return RecoveryHold(
                issue_number=checkpoint.issue_number,
                attempt_id=checkpoint.started_at,
                branch=checkpoint.branch,
                phase=checkpoint.phase.value,
                reason=(
                    "attempt finalization is unresolved; the branch,"
                    " checkpoint, and working tree are preserved for recovery"
                ),
            )
        if not self._attempt_processing and self._workspace_is_dirty():
            # No checkpoint and no work in flight, yet the tree is dirty:
            # unexplained work that requires manual repair before intake.
            # (While an attempt is processing, dirt is its own work and
            # resume stays permitted.)
            return RecoveryHold(
                issue_number=None,
                attempt_id=None,
                branch=None,
                phase=None,
                reason=(
                    "the working tree holds unexplained dirty or untracked"
                    " work with no active attempt; manual repair is required"
                    " before intake"
                ),
            )
        return None

    def _workspace_is_dirty(self) -> bool:
        """Whether the tree holds work outside any active attempt.

        Mirrors the pre-claim dirt guard in ``run_once``: only workspaces
        exposing ``is_clean`` are inspected. An unreadable tree fails closed
        as a hold so resume never bypasses it.
        """

        probe = getattr(self._workspace, "is_clean", None)
        if not callable(probe):
            return False
        try:
            return not probe()
        except Exception:
            return True

    def get_command(self, request_id: str) -> CommandRecord | None:
        """Return one command's current durable acknowledgement."""

        if self._control_store is None:
            raise ControlStoreError("Operator control is not configured")
        with self._control_lock:
            return self._control_store.get_command(request_id)

    def control_status(self, repository: str | None = None) -> dict:
        """Return one consistent live snapshot for ``status`` and the socket."""

        if self._control_store is None:
            raise ControlStoreError("Operator control is not configured")
        with self._control_lock:
            store = self._control_store
            checkpoint = self._attempt_state.read()
            active = (
                ActiveAttemptInfo(
                    issue_number=checkpoint.issue_number,
                    branch=checkpoint.branch,
                    phase=checkpoint.phase.value,
                    started_at=checkpoint.started_at,
                )
                if checkpoint is not None
                else None
            )
            return build_status(
                repository=repository or self._repository,
                intake=store.intake_state(),
                recovering=self._recovering,
                active_attempt=active,
                pending_command_id=store.pending_command_id(),
                get_command=store.get_command,
                recent_commands=store.recent_commands,
                stop_plan=store.stop_plan_snapshot(),
            )

    def run_once(self) -> LifecycleResult:
        """Claim and process one issue, or sleep once when the queue is empty."""

        if not self._startup_reconciled:
            self._startup_reconciled = True
            self._recovering = True
            try:
                recovered = self._reconcile_startup()
            finally:
                self._recovering = False
            if recovered is not None:
                return LifecycleResult(LifecycleStatus.ATTEMPTED, recovered)

        if hasattr(self._workspace, "is_clean") and not self._workspace.is_clean():
            self._event_log(
                "working_tree_dirty",
                "Repository working tree contains uncommitted or untracked changes before claim; human cleanup is required.",
                level="WARNING",
            )
            raise SystemExit(1)

        claim, checkpoint = self._gated_claim()
        if claim is None:
            self._sleeper(self._poll_interval)
            return LifecycleResult(LifecycleStatus.IDLE)

        self._event_log(
            "issue_claimed",
            f"title={claim.issue.title!r}",
            level="INFO",
            issue_number=claim.issue.number,
        )
        self._active_issue_number = claim.issue.number
        assert checkpoint is not None  # _gated_claim starts it atomically with the claim
        archive = self._archive_for(claim.issue.number, checkpoint.started_at)
        prepared: object | None = None
        profile: RepositoryProfile | None = None
        outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
        retain_branch = False
        comment_posted = True
        try:
            # Bootstrap only checks out the default base so the profile can be
            # read. It must never touch the attempt branch: rebasing here onto
            # "main" would corrupt (or fail on) branches built on the profile's
            # real base branch before the profile is even loaded. The attempt
            # is prepared exactly once below, onto profile.base_branch.
            self._workspace.prepare_for_profile_read()
            self._event_log(
                "workspace_prepared_for_profile_read",
                "base_branch=main",
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
            prepared = self._workspace.prepare_attempt(
                base_branch=profile.base_branch, issue_number=claim.issue.number
            )
            self._event_log(
                "workspace_prepared",
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
                retain_branch = getattr(published, "branch_url", None) is None and (
                    outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
                    or (
                        outcome is AttemptOutcome.INCOMPLETE
                        and evidence.decision.publication_path is PublicationPath.PARTIAL
                    )
                )
        except DirtyWorkspaceError as error:
            # Unexplained dirty or untracked work: hold intake and preserve
            # the branch, checkpoint, and entire working tree for inspection.
            # Publishing or releasing here would discard evidence about work
            # this attempt never produced.
            self._event_log(
                "workspace_hold",
                f"{_exception_detail(error)} The branch, checkpoint, and working tree "
                "are preserved for inspection; issue intake is held until the "
                "workspace is repaired by an operator.",
                level="WARNING",
                issue_number=claim.issue.number,
            )
            comment_posted = False
            prepared = None
            raise SystemExit(1) from error
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
        except SystemExit:
            raise
        except RebaseConflictError as error:
            published = self._publish_rebase_conflict(
                claim, checkpoint.started_at, prepared, profile, error
            )
            outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
            comment_posted = getattr(published, "comment_posted", True)
        except Exception as error:
            self._event_log(
                "attempt_exception", _exception_detail(error), level="ERROR", issue_number=claim.issue.number
            )
            published = self._publish_terminal(
                claim, checkpoint.started_at, prepared, profile, outcome
            )
            comment_posted = getattr(published, "comment_posted", True)
        except BaseException as error:
            # KeyboardInterrupt and friends bypass `except Exception` (which is
            # deliberately ordered first); without this the attempt would die
            # with no terminal event or comment. The original is always
            # re-raised after best-effort reporting.
            self._event_log(
                "attempt_exception", _exception_detail(error), level="ERROR", issue_number=claim.issue.number
            )
            try:
                published = self._publish_terminal(
                    claim, checkpoint.started_at, prepared, profile, outcome
                )
                comment_posted = getattr(published, "comment_posted", True)
            except Exception:
                comment_posted = False
            raise
        finally:
            # Durable completion boundary: the attempt is finalized only when
            # its outcome is published (confirmed comment), the issue is
            # released, and local cleanup is durable. Anything short of that
            # keeps the checkpoint, branch, and entire working tree —
            # including dirty and untracked work — for recovery or inspection.
            # Cleanup itself never deletes files or branches.
            released = False
            try:
                if comment_posted:
                    self._release(claim, outcome)
                    released = True
            finally:
                cleanup_ok = prepared is None
                try:
                    if prepared is not None:
                        self._workspace.cleanup(
                            base_branch=profile.base_branch if profile is not None else "main",
                            prepared=prepared,
                            retain_branch=retain_branch or not comment_posted,
                        )
                        cleanup_ok = True
                finally:
                    if comment_posted and released and cleanup_ok:
                        self._commit_finalization(
                            checkpoint, claim.issue.number, getattr(
                                prepared, "branch", f"agent/issue-{claim.issue.number}"
                            ) if prepared is not None else f"agent/issue-{claim.issue.number}",
                            outcome, archive,
                        )
                    else:
                        self._event_log(
                            "attempt_finalization_held",
                            "Publication, release, or cleanup is unconfirmed; the branch, "
                            "checkpoint, and working tree are preserved for recovery.",
                            level="WARNING",
                            issue_number=claim.issue.number,
                        )
                    # The attempt's processing is finished (finalized or held
                    # for recovery); a later resume sees a retained checkpoint
                    # here instead of an active attempt.
                    with self._control_lock:
                        self._attempt_processing = False
        self._active_issue_number = None
        return LifecycleResult(LifecycleStatus.ATTEMPTED, outcome)

    def _gated_claim(self) -> tuple[Claim | None, AttemptCheckpoint | None]:
        """Claim the next issue unless operator control forbids intake.

        The intake check, the GitHub claim, and the checkpoint start share
        the control lock with command acceptance: if a stop commits first,
        no claim begins afterward; if a claim began first, a later stop
        applies to that claim as the active attempt. Returns the claim and
        its checkpoint, or ``(None, None)`` when idle; the caller sleeps
        outside the lock so the control socket stays responsive.
        """

        with self._control_lock:
            if self._control_store is not None:
                intake = self._control_store.intake_state()
                if intake is not IntakeState.RUNNING:
                    if (
                        intake is IntakeState.STOPPING
                        and self._attempt_state.read() is None
                    ):
                        # The waited-for attempt is gone without finalizing
                        # through this path (e.g. it was reconciled on a
                        # previous start); finish the stop instead of holding.
                        self._control_store.complete_pending_stop()
                    self._event_log(
                        "intake_stopped",
                        f"issue intake is {self._control_store.intake_state().value};"
                        f" sleeping {self._poll_interval}s",
                        level="INFO",
                    )
                    return (None, None)
            self._event_log("polling_for_issue", "", level="INFO")
            claim = self._tracker.claim_next()
            if claim is None:
                self._event_log(
                    "no_eligible_issue_found",
                    f"no ready-for-agent issue available; sleeping {self._poll_interval}s",
                    level="INFO",
                )
                return (None, None)
            checkpoint = self._attempt_state.start(
                issue_number=claim.issue.number, branch=f"agent/issue-{claim.issue.number}"
            )
            self._attempt_processing = True
            if self._control_store is not None:
                # The final counted attempt of a ``stop --after`` plan enters
                # ``stopping`` at claim; earlier claims leave intake running
                # and an empty queue preserves the remaining count untouched.
                self._control_store.enter_stopping_for_final_claim()
            return (claim, checkpoint)

    def _record_control_finalization(
        self, attempt_id: str, outcome: AttemptOutcome
    ) -> None:
        """Count one fully finalized attempt against the pending stop plan.

        Called only on the durable completion boundary (outcome published,
        issue released, cleanup durable). The at-most-once decrement, the
        ``stopped`` transition, and the stop-command completion commit in one
        control transaction keyed by the stable attempt ID, so no claim can
        intervene and a replayed attempt cannot count twice. A held
        finalization never reaches this, so its stop plan stays pending and
        intake stays ``stopping`` until recovery finishes the attempt.
        """

        if self._control_store is None:
            return
        with self._control_lock:
            completed = self._control_store.record_attempt_finalized(
                attempt_id, outcome.value
            )
        for record in completed:
            self._event_log(
                "intake_stopped",
                f"stop plan {record.request_id} completed after the final counted attempt finished",
                level="INFO",
            )

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

    def _commit_finalization(
        self,
        checkpoint: object,
        issue_number: int,
        branch: str,
        outcome: AttemptOutcome,
        archive: AttemptArchive | None,
    ) -> None:
        """Record the durable completion boundary, then remove the checkpoint.

        Accounting and the outcome archive precede the ledger entry, which is
        written immediately before the control accounting transaction. The
        at-most-once stop-plan decrement, the ``stopped`` transition, and the
        stop-command completion commit in one control transaction keyed by
        the stable attempt ID, immediately before checkpoint removal. A crash
        before the ledger entry replays safely on restart: publication is
        deduplicated by the attempt marker, release re-reads remote state
        before mutating, and accounting is keyed by the stable attempt ID. A
        crash between the ledger entry or the control transaction and
        checkpoint removal is reconciled as "already finalized, only count
        once and delete" instead of replaying the result comment, the issue
        release, or the accounting. Checkpoint deletion alone is never
        evidence of completion.
        """

        attempt_id = checkpoint.started_at
        self._record_terminal_outcome(outcome, attempt_id, issue_number=issue_number)
        self._write_outcome(archive, outcome, attempt_id)
        if self._completion_store is not None:
            self._completion_store.record(
                attempt_id=attempt_id,
                issue_number=issue_number,
                branch=branch,
                outcome=outcome,
            )
        self._record_control_finalization(attempt_id, outcome)
        self._attempt_state.delete()

    def _reconcile_startup(self) -> AttemptOutcome | None:
        """Finish one durable attempt without recreating its SDK execution context."""

        try:
            checkpoint = self._attempt_state.read()
        except AttemptStateError as error:
            self._event_log(
                "attempt_checkpoint_unusable",
                f"{_exception_detail(error)} The checkpoint is preserved for inspection; "
                "issue intake is held until it is repaired or removed by an operator.",
                level="ERROR",
            )
            raise SystemExit(1) from error
        if checkpoint is None:
            return None
        try:
            finalized = (
                self._completion_store is not None
                and self._completion_store.is_finalized(checkpoint.started_at)
            )
        except CompletionStoreError as error:
            self._event_log(
                "attempt_completion_unusable",
                f"{_exception_detail(error)} The completion ledger is preserved for "
                "inspection; issue intake is held until it is repaired or removed "
                "by an operator.",
                level="ERROR",
            )
            raise SystemExit(1) from error
        if finalized:
            # The previous process finalized every boundary step and recorded
            # the attempt, then crashed before removing the checkpoint.
            # Replaying any side effect here would duplicate the result
            # comment, the issue release, or the completed-attempt accounting,
            # so the saved accounting is applied at most once and only the
            # leftover checkpoint is removed. A stop plan that waited for
            # this attempt completes here: its whole boundary is durable.
            if self._completion_store is not None:
                completions = self._completion_store.read_all()
                terminal = completions.get(checkpoint.started_at)
                if terminal is not None:
                    self._record_control_finalization(
                        checkpoint.started_at, terminal.outcome
                    )
            self._attempt_state.delete()
            self._active_issue_number = None
            return None
        claim = self._tracker.recover_claim(checkpoint.issue_number)
        if claim is None:
            raise RuntimeError("Interrupted attempt issue is unavailable for recovery")
        self._active_issue_number = claim.issue.number
        with self._control_lock:
            self._attempt_processing = True
            if self._control_store is not None:
                # A crash between the final claim and its ``stopping``
                # transition must not lose the boundary: the recovered final
                # counted attempt enters ``stopping`` here instead.
                self._control_store.enter_stopping_for_final_claim()
        archive = self._archive_for(claim.issue.number, checkpoint.started_at)

        prepared: object | None = None
        profile: RepositoryProfile | None = None
        outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
        comment_posted = True
        retain_branch = False
        has_commits = False
        try:
            self._workspace.prepare_for_profile_read()
            profile = self._profile_loader(self._workspace.working_directory)
            prepared = self._workspace.prepare_attempt(
                base_branch=profile.base_branch, issue_number=claim.issue.number
            )
            if checkpoint.phase is AttemptPhase.MODEL_RUNNING:
                commits = getattr(self._workspace, "commits_added")(prepared)
                has_commits = bool(commits)
                if commits:
                    if _final_check_was_interrupted(archive):
                        # The previous process logged final_check_started but
                        # never finished the check (kill, crash, or container
                        # restart). Report that honestly instead of blaming
                        # model execution; the branch is retained for resume.
                        self._event_log(
                            "final_check_interrupted",
                            "The final check started but never finished; the attempt process "
                            "died mid-check.",
                            level="ERROR",
                            issue_number=claim.issue.number,
                        )
                        decision = CompletionDecision(
                            AttemptOutcome.INFRASTRUCTURE_ERROR,
                            False,
                            PublicationPath.PARTIAL,
                            (
                                "Final check was interrupted before completing; committed work "
                                "is preserved on the branch.",
                            ),
                        )
                    else:
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
                getattr(published, "branch_url", None) is None
                and has_commits
                and (
                    outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
                    or (
                        outcome is AttemptOutcome.INCOMPLETE
                        and decision.publication_path is PublicationPath.PARTIAL
                    )
                )
            )
        except DirtyWorkspaceError as error:
            # Unexplained dirty or untracked work: hold intake and preserve
            # the branch, checkpoint, and entire working tree for inspection.
            # Publishing or releasing here would discard evidence about work
            # this attempt never produced.
            self._event_log(
                "workspace_hold",
                f"{_exception_detail(error)} The branch, checkpoint, and working tree "
                "are preserved for inspection; issue intake is held until the "
                "workspace is repaired by an operator.",
                level="WARNING",
                issue_number=claim.issue.number,
            )
            comment_posted = False
            prepared = None
            raise SystemExit(1) from error
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
        except SystemExit:
            raise
        except RebaseConflictError as error:
            published = self._publish_rebase_conflict(
                claim, checkpoint.started_at, prepared, profile, error
            )
            outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
            comment_posted = getattr(published, "comment_posted", True)
        except Exception as error:
            self._event_log(
                "attempt_exception", _exception_detail(error), level="ERROR", issue_number=claim.issue.number
            )
            published = self._publish_terminal(
                claim, checkpoint.started_at, prepared, profile, AttemptOutcome.INFRASTRUCTURE_ERROR
            )
            outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
            comment_posted = getattr(published, "comment_posted", True)
        except BaseException as error:
            self._event_log(
                "attempt_exception", _exception_detail(error), level="ERROR", issue_number=claim.issue.number
            )
            try:
                published = self._publish_terminal(
                    claim, checkpoint.started_at, prepared, profile, AttemptOutcome.INFRASTRUCTURE_ERROR
                )
                comment_posted = getattr(published, "comment_posted", True)
            except Exception:
                comment_posted = False
            outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
            raise
        finally:
            # Same durable completion boundary as the live-attempt path (see
            # run_once): only a confirmed comment plus a completed release
            # plus durable cleanup may record the attempt and remove its
            # checkpoint. Anything else preserves everything for recovery.
            released = False
            try:
                if comment_posted:
                    self._release(claim, outcome)
                    released = True
            finally:
                cleanup_ok = prepared is None
                try:
                    if prepared is not None:
                        self._workspace.cleanup(
                            base_branch=profile.base_branch if profile is not None else "main",
                            prepared=prepared,
                            retain_branch=retain_branch or not comment_posted,
                        )
                        cleanup_ok = True
                finally:
                    if comment_posted and released and cleanup_ok:
                        self._commit_finalization(
                            checkpoint, claim.issue.number, getattr(
                                prepared, "branch", checkpoint.branch
                            ) if prepared is not None else checkpoint.branch,
                            outcome, archive,
                        )
                    else:
                        self._event_log(
                            "attempt_finalization_held",
                            "Publication, release, or cleanup is unconfirmed; the branch, "
                            "checkpoint, and working tree are preserved for recovery.",
                            level="WARNING",
                            issue_number=claim.issue.number,
                        )
                    # Reconciliation is finished (finalized or held); a later
                    # resume sees a retained checkpoint here, not an active
                    # attempt.
                    with self._control_lock:
                        self._attempt_processing = False
        self._active_issue_number = None
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
        details: str = "Repository profile could not be loaded or the attempt could not start.",
    ):
        branch = getattr(prepared, "branch", f"agent/issue-{claim.issue.number}")
        base_branch = profile.base_branch if profile is not None else "main"
        return self._publisher.publish(
            PublicationRequest(
                issue_number=claim.issue.number,
                issue_title=claim.issue.title,
                branch=branch,
                started_at=started_at,
                decision=CompletionDecision(outcome, False, PublicationPath.NONE, (details,)),
                check_command="not run",
                check_exit_code=None,
                review_cycles=0,
                review_findings="not run",
                details=details,
                base_branch=base_branch,
            )
        )

    def _publish_rebase_conflict(
        self,
        claim: Claim,
        started_at: str,
        prepared: object | None,
        profile: RepositoryProfile | None,
        error: RebaseConflictError,
    ):
        """Report an aborted rebase conflict as the terminal infrastructure error.

        The conflicting rebase was already aborted by workspace preparation,
        so setup never ran: the failure is classified here with its
        ``rebase_conflict`` cause instead of surfacing later as an
        unrelated setup error against a half-merged tree.
        """

        self._event_log(
            "rebase_conflict",
            _exception_detail(error),
            level="ERROR",
            issue_number=claim.issue.number,
        )
        return self._publish_terminal(
            claim,
            started_at,
            prepared,
            profile,
            AttemptOutcome.INFRASTRUCTURE_ERROR,
            details=(
                f"{error} "
                "Setup was not started so the failure is reported here "
                "instead of as an unrelated setup error."
            ),
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


def _final_check_was_interrupted(archive: AttemptArchive | None) -> bool:
    """Whether a previous process died after starting the final check.

    The runner records ``final_check_started`` immediately before running the
    check and ``final_check_finished`` right after it returns, so a marker
    without its finish means the process never survived the check — the one
    signature a SIGKILL/OOM leaves behind.
    """

    if archive is None:
        return False
    record = archive.read_attempt()
    return bool(record.get("final_check_started")) and not record.get("final_check_finished")


def _has_unresolved_conflicts(workspace: object) -> bool:
    """Whether the workspace holds a rebase/merge the model must resolve first."""

    probe = getattr(workspace, "has_unresolved_conflicts", None)
    if not callable(probe):
        return False
    return bool(probe())


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


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


_ALLOWED_HANDOFF_REASONS = frozenset(
    {"cost_soft_threshold", "cost_hard_limit", "turn_limit", "time_limit"}
)


def _classify_handoff_reason(execution: object) -> str:
    explanation = getattr(execution, "explanation", "").lower()
    terminal_reason = (getattr(execution, "terminal_reason", None) or "").lower()
    if "budget" in terminal_reason or "hard cost ceiling" in explanation or "budget" in explanation:
        return "cost_hard_limit"
    if "turn" in terminal_reason or "max_turns" in explanation:
        return "turn_limit"
    if "timeout" in explanation or "time" in terminal_reason:
        return "time_limit"
    return "cost_hard_limit"


def _build_emergency_handoff_note(
    *,
    issue_number: int,
    started_at: str,
    reason: str,
    last_work_commit: str,
    explanation: str,
) -> str:
    return (
        f"# Handoff note: issue #{issue_number}\n\n"
        f"- issue: {issue_number}\n"
        f"- started_at: {started_at}\n"
        f"- reason: {reason}\n"
        f"- last_work_commit: {last_work_commit}\n\n"
        "## Summary\n\n"
        f"{explanation}\n\n"
        "## Remaining work\n\n"
        "Review progress preserved on this branch and continue implementation.\n"
    )

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
