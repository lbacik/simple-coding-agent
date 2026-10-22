"""Single-process orchestration of one implementation attempt at a time."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Collection, Mapping, Sequence
import copy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from typing import Any, Protocol

from simple_coding_agent.attempt_state import (
    ATTEMPT_KIND_CONFLICT,
    ATTEMPT_KIND_ORDINARY,
    AttemptCheckpoint,
    AttemptPhase,
    AttemptStateError,
    AttemptStateStore,
)
from simple_coding_agent.attempt_workspace import AttemptWorkspace, ConflictState
from simple_coding_agent.control import (
    ActiveAttemptInfo,
    CommandAcknowledgement,
    CommandRecord,
    ControlStore,
    ControlStoreError,
    HANDOFF_ABSOLUTE_DEADLINE_SECONDS,
    HANDOFF_PUBLICATION_ALLOWANCE_SECONDS,
    IntakeState,
    NextIssueRejectedError,
    RecoveryHold,
    build_status,
    parse_next_issue,
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
    _build_baseline_failure_section,
    _select_failed_command,
    deferred_baseline_failed_reason,
    final_check_dirty_reason,
)
from simple_coding_agent.config import RepositoryProfile
from simple_coding_agent.finalization import AttemptCompletionStore, CompletionStoreError
from simple_coding_agent.git_workspace import (
    AttemptCommit,
    DirtyWorkspaceError,
    GitWorkspaceError,
    GitWorkspaceRecoveryError,
    PreparedAttempt,
    RebaseConflictError,
    handoff_note_relpath,
)
from simple_coding_agent.conflict_attempt import (
    attempt_branch as _conflict_attempt_branch,
)
from simple_coding_agent.conflict_attempt import (
    base_changes_text as _conflict_base_changes_text,
)
from simple_coding_agent.conflict_attempt import (
    base_mismatch_reason as _conflict_base_mismatch_reason,
)
from simple_coding_agent.conflict_attempt import (
    build_resolution_prompt as _conflict_resolution_prompt,
)
from simple_coding_agent.conflict_attempt import (
    check_failed_details as _conflict_check_failed_details,
)
from simple_coding_agent.conflict_attempt import (
    commit_count_changed_details as _conflict_count_changed_details,
)
from simple_coding_agent.conflict_attempt import (
    entry_skip_reason as _conflict_entry_skip_reason,
)
from simple_coding_agent.conflict_attempt import (
    failure_output_tail as _conflict_failure_output_tail,
)
from simple_coding_agent.conflict_attempt import (
    files_outside_conflict as _conflict_files_outside,
)
from simple_coding_agent.conflict_attempt import (
    fix_commit_count as _conflict_fix_commit_count,
)
from simple_coding_agent.conflict_attempt import (
    format_check_output as _conflict_format_check_output,
)
from simple_coding_agent.conflict_attempt import (
    is_conflict_claim as _is_conflict_claim,
)
from simple_coding_agent.conflict_attempt import (
    lifecycle_outcome as _conflict_lifecycle_outcome,
)
from simple_coding_agent.conflict_attempt import (
    model_error_details as _conflict_model_error_details,
)
from simple_coding_agent.conflict_attempt import (
    model_limit_details as _conflict_model_limit_details,
)
from simple_coding_agent.conflict_attempt import (
    model_success_details as _conflict_model_success_details,
)
from simple_coding_agent.conflict_attempt import (
    model_unresolved_details as _conflict_model_unresolved_details,
)
from simple_coding_agent.conflict_attempt import (
    rebase_conflict_details as _conflict_rebase_conflict_details,
)
from simple_coding_agent.conflict_attempt import (
    setup_failed_details as _conflict_setup_failed_details,
)
from simple_coding_agent.conflict_attempt import (
    short_revision as _conflict_short_revision,
)
from simple_coding_agent.conflict_attempt import (
    success_details as _conflict_success_details,
)
from simple_coding_agent.github_tracker import (
    MERGE_CONFLICT,
    ROUND_FINISHED,
    Claim,
    ClaimLostError,
    TrackerIssue,
    ineligibility_reason,
)
from simple_coding_agent.model_execution import ModelExecutionStatus, ModelExecutor, OperatorHandoff, ToolCallRecord
from simple_coding_agent.observability import AttemptArchive, Redactor
from simple_coding_agent.operating import ConsecutiveErrorStore
from simple_coding_agent.publication import (
    AUTO_MERGE_LABEL,
    ConflictPublicationRequest,
    PublicationRequest,
    PullRequest,
    confirm_shared_handoff,
)
from simple_coding_agent.recovery import (
    CONFLICT_OUTCOME_CHECK_FAILED,
    CONFLICT_OUTCOME_ERROR,
    CONFLICT_OUTCOME_MODEL_LIMIT,
    CONFLICT_OUTCOME_REBASE_CONFLICT,
    CONFLICT_OUTCOME_RESOLUTION_FAILED,
    CONFLICT_OUTCOME_SETUP_FAILED,
    CONFLICT_OUTCOME_SKIPPED,
    CONFLICT_OUTCOME_SUCCESS,
    REQUEUE_SOURCE_AUTOMATIC,
    REQUEUE_SOURCE_CONFLICT,
    REQUEUE_SOURCE_HUMAN,
    filter_prompt_comments,
    requeue_source_for_labels,
)
from simple_coding_agent.telemetry import (
    CONTROL_LOCK_TIMEOUT_SECONDS,
    SnapshotFailedError,
    TelemetryReporter,
    format_timestamp,
    load_last_finished,
)


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
    reviews_performed: int = 0
    #: The profile check steps in run order, for the ``K/M`` position of a
    #: failure in the published result comment.
    check_steps: tuple[str, ...] = ()
    #: The one profile check step that failed, and its 1-based position in
    #: ``check_steps``. Set even when the exit code is missing (a timeout);
    #: ``None`` when the check passed or never ran.
    check_failed_step: str | None = None
    check_failed_index: int | None = None
    #: Why the check never ran. ``None`` renders a bare ``Check: not run``.
    check_not_run_reason: str | None = None


#: The handoff-note reason an operator ``handoff now`` request requires.
_OPERATOR_HANDOFF_REASON = "operator_request"

#: How many offending paths the dirty-tree hold names in its log and status
#: detail. The full list is whatever ``git status --short`` reports; only
#: the first few are quoted so the detail stays a one-screen operator hint.
_DIRTY_PATH_DISPLAY_LIMIT = 5


class OperatorHandoffControl(Protocol):
    """The control boundary a model attempt uses to serve one handoff request."""

    def snapshot(self) -> dict | None: ...

    def mark_delivering(self, request_id: str) -> bool: ...

    def mark_begun(self, request_id: str) -> bool: ...

    def complete(self, request_id: str, detail: str) -> None: ...

    def fail(self, request_id: str, reason: str) -> None: ...


def _parse_operator_accepted_at(value: object) -> datetime | None:
    """Parse a control ``accepted_at`` timestamp for executor windows."""

    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


class _ControlOperatorHandoff:
    """Adapt the durable control store to the runner's operator boundary."""

    def __init__(self, store: ControlStore, event_log: Callable[..., None]) -> None:
        self._store = store
        self._event_log = event_log

    def snapshot(self) -> dict | None:
        return self._store.handoff_snapshot()

    def mark_delivering(self, request_id: str) -> bool:
        return self._store.mark_handoff_delivering(request_id)

    def mark_begun(self, request_id: str) -> bool:
        return self._store.mark_handoff_begun(request_id)

    def complete(self, request_id: str, detail: str) -> None:
        record = self._store.complete_handoff(request_id, detail)
        if record is None:
            self._event_log(
                "operator_handoff_resolution_unknown",
                f"request={request_id}; the handoff command is no longer recorded",
                level="WARNING",
            )

    def fail(self, request_id: str, reason: str) -> None:
        record = self._store.fail_handoff(request_id, reason)
        if record is None:
            self._event_log(
                "operator_handoff_resolution_unknown",
                f"request={request_id}; the handoff command is no longer recorded",
                level="WARNING",
            )


class AttemptTracker(Protocol):
    """Claim and idempotently release the GitHub state owned by this agent."""

    def claim_next(self) -> Claim | None: ...

    def fetch_issue(self, number: int) -> TrackerIssue | None: ...

    def is_self_assigned(self, issue: TrackerIssue) -> bool: ...

    def claim_verified(self, issue: TrackerIssue) -> Claim: ...

    def automatic_continuation_count(self, issue: TrackerIssue) -> int: ...

    def conflict_failure_streak(self, issue: TrackerIssue) -> int: ...

    def escalate_conflict_to_human(self, issue: TrackerIssue) -> None: ...

    def recover_claim(self, issue_number: int) -> Claim | None: ...

    def release_attempt(self, issue_number: int, label: str, assignee_id: str) -> None: ...

    def release_handoff(self, issue_number: int, assignee_id: str) -> None: ...


class Publisher(Protocol):
    """Publication boundary, including the result-comment operation."""

    def publish(self, request: PublicationRequest): ...

    def find_pull_request(self, head: str) -> PullRequest | None: ...

    def publish_conflict(self, request: ConflictPublicationRequest): ...


class ConflictCheckRunner(Protocol):
    """Run one profile command sequence for a conflict attempt's fast path."""

    def run(
        self,
        commands: Sequence[str],
        *,
        timeout: float,
        cwd: Path,
        environment: Mapping[str, str] | None = None,
    ): ...


class ModelAttemptRunner:
    """Adapt setup, SDK execution, review evidence, and final checks to publication."""

    def __init__(
        self,
        *,
        attempt_state: AttemptStateStore,
        workspace: AttemptWorkspace,
        verifier: VerificationRunner,
        evaluator: CompletionEvaluator,
        model_executor: ModelExecutor,
        acceptance_criteria_satisfied: Callable[[Claim], bool] = lambda claim: True,
        issue_comments: Callable[[TrackerIssue], tuple[str, ...]] = lambda issue: (),
        prompt_comments: Callable[[TrackerIssue], tuple[str, ...]] | None = None,
        attempt_archive_factory: Callable[[int, str], AttemptArchive] | None = None,
        event_log: Callable[..., None] = (
            lambda event, detail="", level="INFO", issue_number=None: None
        ),
        operator_handoff: OperatorHandoffControl | None = None,
        prompt_redactions: Sequence[str] = (),
    ) -> None:
        self._attempt_state = attempt_state
        self._workspace = workspace
        self._verifier = verifier
        self._evaluator = evaluator
        self._model_executor = model_executor
        self._acceptance_criteria_satisfied = acceptance_criteria_satisfied
        self._issue_comments = issue_comments
        if prompt_comments is None:
            # The default prompt view filters the raw comment feed: historical
            # handoff results are superseded by the replacement Handoff note
            # on the resumed branch, and machine bookkeeping is stripped from
            # the generated results that remain. Detection, budget counting,
            # and publication recovery keep reading the unfiltered feed.
            raw_comments = issue_comments

            def default_prompt_comments(issue: TrackerIssue) -> tuple[str, ...]:
                return filter_prompt_comments(raw_comments(issue))

            prompt_comments = default_prompt_comments

        self._prompt_comments = prompt_comments
        self._attempt_archive_factory = attempt_archive_factory
        self._event_log = event_log
        self._operator_handoff = operator_handoff
        self._prompt_redactions = tuple(prompt_redactions)

    def set_operator_handoff(self, control: OperatorHandoffControl | None) -> None:
        """Attach (or detach) the operator handoff served by the next execution.

        A request accepted during setup waits here: the provider is only
        consulted once model execution starts, so setup-phase requests are
        delivered at the first model boundary instead of interrupting setup.
        """

        self._operator_handoff = control

    def __call__(self, claim: Claim, profile: RepositoryProfile, prepared: PreparedAttempt) -> AttemptEvidence:
        """Run the local evidence gates in their mandated phase order."""

        if not prepared.restored_from_remote and _continuation_expected(
            claim.issue, self._issue_comments
        ):
            branch = prepared.branch
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
        attempt_id = _attempt_id(checkpoint)
        archive = (
            self._attempt_archive_factory(claim.issue.number, checkpoint.started_at)
            if checkpoint is not None and self._attempt_archive_factory is not None
            else None
        )
        initial_commits = self._workspace.commits_added(prepared)
        continuation = bool(initial_commits)
        # A retained branch rebased onto its base can be left with conflict
        # markers for the model to resolve. Setup commands parse every source
        # file, so they must never run against that half-rebased tree; they
        # are deferred until after the model resolves the conflicts.
        pre_state = self._workspace.conflict_state()
        setup_deferred = pre_state.in_progress
        preparation: PreparationEvidence | None = None
        if setup_deferred:
            resolution_files = _remaining_conflicts(pre_state, prepared)
            self._event_log(
                "rebase_resolution_started",
                "The model resolves the rebase conflicts first, before setup runs."
                f"{_format_conflicted_files(resolution_files)}",
                level="INFO",
                issue_number=claim.issue.number,
            )
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
                # Setup (or the mandatory baseline) failed before model
                # execution: a pending operator handoff can never be served,
                # so it is settled here with the ordinary outcome and its
                # reason instead of leaking as still-accepted.
                served, late = self._read_operator_request()
                self._settle_operator_request(claim, served, late, early.decision)
                return early
            preparation = early

        self._attempt_state.transition(AttemptPhase.MODEL_RUNNING)
        # Snapshot before dispatch: any commit already on the branch at this
        # point is carried over from a resumed attempt, not produced by this
        # run. `commits_added` is read again after the model runs to count
        # what this run itself added.
        # A tolerated red baseline on a continuation is the model's to fix
        # (it almost always comes from the earlier attempt's own commits), so
        # the failure rides along in the starting prompt. A red first-attempt
        # baseline never reaches here: preparation returns early above.
        failed_baseline = None
        if (
            preparation is not None
            and continuation
            and preparation.baseline is not None
            and not preparation.baseline.succeeded
        ):
            failed_baseline = preparation.baseline
        prompt_body = _build_starting_prompt(
            claim.issue.body,
            self._prompt_comments(claim.issue),
            continuation=continuation,
            issue_number=claim.issue.number,
            profile=profile,
            redactions=self._prompt_redactions,
            setup_deferred=setup_deferred,
            baseline=failed_baseline,
        )
        self._event_log(
            "model_dispatch_starting",
            f"continuation={continuation}",
            level="INFO",
            issue_number=claim.issue.number,
        )
        self._wire_operator_handoff()
        self._wire_review_gate(prepared, initial_commits)
        try:
            execution = asyncio.run(
                _execute_model(
                    self._model_executor,
                    issue_body=prompt_body,
                    working_directory=self._workspace.working_directory,
                    issue_number=claim.issue.number,
                    archive=archive,
                    attempt_id=attempt_id,
                    branch=prepared.branch,
                    base_revision=prepared.base_revision,
                )
            )
            self._archive_cli_transcripts(archive, claim.issue.number)
            if archive is not None:
                archive.write_attempt(
                    {
                        "model_usage": execution.model_usage,
                        "skill_events": [event.__dict__ for event in execution.skill_events],
                        "model_stop_reason": execution.stop_reason,
                        "token_budget": execution.token_budget,
                    }
                )
            served_operator_request, late_operator_snapshot = self._read_operator_request()
        except BaseException:
            # Model execution, or recording its result, failed: still archive
            # whatever transcripts the model left behind and seal whatever
            # work is on the branch, then propagate so the attempt finalizes
            # as an infrastructure error with the work sealed on the branch.
            # A seal failure is only logged here. It refuses on
            # an unresolved rebase, which cleanup then aborts; any other
            # dirty tree makes cleanup hold the workspace for the operator.
            self._archive_cli_transcripts(archive, claim.issue.number)
            try:
                self._workspace.seal(
                    issue_number=claim.issue.number,
                    attempt_id=attempt_id,
                    status=ModelExecutionStatus.INFRASTRUCTURE_ERROR,
                )
            except Exception as seal_error:
                self._event_log(
                    "seal_attempt_work_failed",
                    _exception_detail(seal_error),
                    level="ERROR",
                    issue_number=claim.issue.number,
                )
            raise
        if setup_deferred:
            # The model had its chance to resolve the conflicts left in
            # place by preparation. Setup must never run against a
            # half-merged tree, so the resolution is verified
            # deterministically first: any remaining rebase/merge state,
            # unmerged path, conflict marker, or whitespace error falls
            # back to the pre-rebase branch state and the attempt reports
            # a rebase_conflict infrastructure error, exactly as if the
            # rebase had been aborted up front.
            post_state = self._workspace.conflict_state()
            if post_state.resolution_problems:
                remaining = _remaining_conflicts(post_state, prepared)
                branch = prepared.branch
                try:
                    self._workspace.abort_unresolved_rebase(prepared)
                except GitWorkspaceError:
                    self._event_log(
                        "rebase_resolution_failed",
                        f"{' '.join(post_state.resolution_problems)} The branch could not be restored.",
                        level="ERROR",
                        issue_number=claim.issue.number,
                    )
                    raise
                self._event_log(
                    "rebase_resolution_failed",
                    f"{' '.join(post_state.resolution_problems)}"
                    " The rebase was aborted and the branch restored to its"
                    " pre-rebase state.",
                    level="ERROR",
                    issue_number=claim.issue.number,
                )
                files = f" Conflicting files: {', '.join(remaining)}." if remaining else ""
                raise RebaseConflictError(
                    f"Rebase of {branch} onto {profile.base_branch} hit conflicts"
                    f" (rebase_conflict).{files} The model left conflicts"
                    " unresolved, so the rebase was aborted and setup was not started.",
                    branch=branch,
                    base_branch=profile.base_branch,
                    conflicted_files=remaining,
                )
            self._event_log(
                "rebase_resolution_succeeded",
                "The model resolved the rebase conflicts; setup runs normally."
                f"{_format_conflicted_files(resolution_files)}",
                level="INFO",
                issue_number=claim.issue.number,
            )
        # Seal the attempt's work unconditionally: every normal exit from
        # model execution leaves dirty and untracked changes in one commit
        # attributed to this attempt, whatever the execution status. A new
        # ``ModelExecutionStatus`` therefore cannot silently drop work the
        # way the old enumerated status tuple did. The unresolved-conflict
        # path above aborts and restores instead of committing conflict
        # markers; the exception path seals before propagating.
        try:
            try:
                pre_seal_dirty: tuple[str, ...] | None = self._workspace.dirty_paths()
            except Exception:
                pre_seal_dirty = None
            seal_commit = self._workspace.seal(
                issue_number=claim.issue.number,
                attempt_id=attempt_id,
                status=execution.status,
            )
        except GitWorkspaceError as error:
            decision = CompletionDecision(
                outcome=AttemptOutcome.INFRASTRUCTURE_ERROR,
                publication_eligible=False,
                publication_path=PublicationPath.NONE,
                reasons=(f"Failed to seal attempt work after model execution: {error}",),
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
        commits = self._workspace.commits_added(prepared)
        if execution.status is ModelExecutionStatus.MODEL_LIMIT_REACHED:
            # Emergency handoff: synthesize and commit a handoff note so
            # progress is preserved for human review with round-finished.
            # The model's work is already sealed above; the note commit
            # stages only the note file on top of it.
            note_commit = self._workspace.handoff_note_commit(
                prepared, claim.issue.number, attempt_id=attempt_id
            )
            if note_commit is None:
                reason = _classify_handoff_reason(execution)
                note_content = _build_emergency_handoff_note(
                    issue_number=claim.issue.number,
                    started_at=attempt_id,
                    reason=reason,
                    last_work_commit=commits[0].revision if commits else prepared.base_revision,
                    explanation=execution.explanation,
                    diff_stat=_git_diff_stat(
                        self._workspace.working_directory,
                        prepared.base_revision,
                        seal_commit.revision if seal_commit is not None else prepared.branch,
                    ),
                    attempt_commits=tuple(
                        f"{commit.revision[:7]} {commit.subject}" for commit in commits
                    ),
                    sealed_files=(
                        tuple(pre_seal_dirty)
                        if seal_commit is not None and pre_seal_dirty is not None
                        else None
                    ),
                    edited_files=tuple(execution.edited_files),
                    last_assistant_text=execution.last_assistant_text,
                    handoff_skill_invoked=_main_thread_handoff_invoked(
                        execution.skill_events
                    ),
                    tool_calls=tuple(execution.tool_calls),
                    last_check_call=_last_check_call(
                        tuple(execution.bash_calls), profile.check
                    ),
                    baseline_result=(
                        preparation.baseline if preparation is not None else None
                    ),
                    token_budget=execution.token_budget,
                    archive_dir=(
                        str(archive.directory) if archive is not None else None
                    ),
                    session_id=getattr(
                        self._model_executor, "cli_session_id", None
                    ),
                    previous_note=_previous_complete_note(
                        self._workspace, prepared, claim.issue.number
                    ),
                )
                self._workspace.commit_handoff_note(claim.issue.number, note_content)
                commits = self._workspace.commits_added(prepared)
            execution = replace(execution, status=ModelExecutionStatus.HANDOFF_REQUESTED)
        commit_count = len(commits)
        # A red deferred baseline means the same gate the final check runs
        # already failed on this tree after conflict resolution, with no new
        # work in between: running the final check would only fail the same
        # way. The attempt ends incomplete with a named reason instead, and
        # the next round starts by making the check pass. This applies only
        # when model execution succeeded with commits (when the final check
        # would have run); handoff, limit, setup-failure, and no-change paths
        # keep their current behavior.
        deferred_baseline: object | None = None
        if (
            setup_deferred
            and preparation is not None
            and preparation.baseline is not None
            and not preparation.baseline.succeeded
            and execution.status is ModelExecutionStatus.SUCCEEDED
            and commits
        ):
            deferred_baseline = preparation.baseline
        if deferred_baseline is not None:
            deferred_reason = deferred_baseline_failed_reason(deferred_baseline)
            self._event_log(
                "deferred_baseline_failed",
                deferred_reason,
                level="ERROR",
                issue_number=claim.issue.number,
            )
            if archive is not None:
                archive.write_attempt(
                    {"deferred_baseline_failed": {"reason": deferred_reason}}
                )
        review_count = sum(
            event.name == "code-review" and event.phase == "PreToolUse"
            for event in execution.skill_events
        )
        review_cycles = max(0, review_count - 1)
        review = ReviewEvidence((), review_cycles)
        final_check = None
        if execution.status is ModelExecutionStatus.SUCCEEDED and commits and deferred_baseline is None:
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
                final_check = self._verifier.final_check(profile, self._workspace.working_directory)
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
        # A check command may rewrite tracked files or generate non-ignored
        # files (``next build`` rewrites ``tsconfig.json`` and creates
        # ``next-env.d.ts``). The attempt work was sealed before the check, so
        # any residue is the check's own side effect: detect it here, before
        # any push, PR, or result comment, instead of surfacing it later as a
        # cleanup hold after a ``complete`` publication.
        final_check_dirty_paths: tuple[str, ...] = ()
        preservation_commit: AttemptCommit | None = None
        if final_check is not None and final_check.succeeded:
            try:
                final_check_dirty_paths = tuple(self._workspace.dirty_paths())
            except Exception:
                final_check_dirty_paths = ()
            if final_check_dirty_paths:
                dirty_reason = final_check_dirty_reason(tuple(sorted(final_check_dirty_paths)))
                self._event_log(
                    "final_check_dirtied_workspace",
                    dirty_reason,
                    level="WARNING",
                    issue_number=claim.issue.number,
                )
                try:
                    preservation_commit = self._workspace.seal(
                        issue_number=claim.issue.number,
                        attempt_id=attempt_id,
                        status=execution.status,
                    )
                except Exception as seal_error:
                    self._event_log(
                        "seal_attempt_work_failed",
                        _exception_detail(seal_error),
                        level="ERROR",
                        issue_number=claim.issue.number,
                    )
                    preservation_commit = None
                if archive is not None:
                    archive.write_attempt(
                        {
                            "final_check_dirtied_workspace": {
                                "paths": sorted(final_check_dirty_paths),
                                "reason": dirty_reason,
                                "preservation_commit": (
                                    preservation_commit.revision
                                    if preservation_commit is not None
                                    else None
                                ),
                            }
                        }
                    )
                commits = self._workspace.commits_added(prepared)
                commit_count = len(commits)
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
                final_check_dirty_paths=final_check_dirty_paths,
                deferred_baseline_failed=deferred_baseline is not None,
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
            if deferred_baseline is not None:
                # An older evaluator without the deferred-baseline seam: apply
                # the same incomplete decision here so the attempt still ends
                # with the named reason instead of the generic final-check one.
                decision = CompletionDecision(
                    outcome=AttemptOutcome.INCOMPLETE,
                    publication_eligible=False,
                    publication_path=(
                        PublicationPath.PARTIAL if commit_count > 0 else PublicationPath.NONE
                    ),
                    reasons=(
                        deferred_baseline_failed_reason(deferred_baseline),
                    ),
                )
            elif final_check_dirty_paths:
                # An older evaluator without the dirtied-workspace seam: apply
                # the same incomplete decision here so the attempt still stops
                # before publication.
                decision = CompletionDecision(
                    outcome=AttemptOutcome.INCOMPLETE,
                    publication_eligible=False,
                    publication_path=(
                        PublicationPath.PARTIAL if commit_count > 0 else PublicationPath.NONE
                    ),
                    reasons=(
                        final_check_dirty_reason(tuple(sorted(final_check_dirty_paths))),
                    ),
                )
        handoff_rejection_reason: str | None = None
        if decision.outcome is AttemptOutcome.HANDOFF:
            validation = _validate_handoff_note(
                self._workspace.handoff_note_commit(
                    prepared, claim.issue.number, attempt_id=attempt_id
                ),
                commits,
                claim.issue.number,
                self._workspace.working_directory,
                prepared.base_revision,
                note_commit_revisions=_this_attempt_note_commit_revisions(
                    self._workspace, prepared, claim.issue.number, attempt_id
                ),
            )
            _emit_handoff_note_events(
                self._event_log, claim.issue.number, validation
            )
            if not validation.valid:
                # A committed-but-untrustworthy note is ordinary evidence,
                # not an infrastructure failure: the branch still publishes
                # for human review when it carries work. Only commit, push,
                # or transport failures stay infrastructure errors.
                decision = CompletionDecision(
                    outcome=AttemptOutcome.INCOMPLETE,
                    publication_eligible=False,
                    publication_path=PublicationPath.PARTIAL if commits else PublicationPath.NONE,
                    reasons=(validation.reason,),
                )
                handoff_rejection_reason = validation.reason
        self._settle_operator_request(
            claim,
            served_operator_request,
            late_operator_snapshot,
            decision,
        )
        _log_completion_decision(self._event_log, claim.issue.number, decision)
        if decision.publication_eligible or decision.publication_path is PublicationPath.PARTIAL:
            self._attempt_state.transition(AttemptPhase.PUSHING)
        findings = _review_findings_text(review, review_count)
        details = execution.explanation
        note_commit_sha = None
        if decision.outcome is AttemptOutcome.HANDOFF:
            details = _read_handoff_note(
                self._workspace.working_directory, claim.issue.number
            )
            note_commit = self._workspace.handoff_note_commit(
                prepared, claim.issue.number, attempt_id=attempt_id
            )
            if note_commit is not None:
                note_commit_sha = note_commit.revision
        elif handoff_rejection_reason is not None:
            details = handoff_rejection_reason
        elif (
            decision.outcome in (AttemptOutcome.INCOMPLETE, AttemptOutcome.NO_CHANGES)
            and decision.reasons
        ):
            # The evaluator's reason (for example a missing ``code-review``)
            # is the human-visible cause; the generic SDK explanation would
            # otherwise hide it behind a green check.
            details = "; ".join(decision.reasons)
            if final_check_dirty_paths and preservation_commit is not None:
                details = (
                    f"{details} The check's side effects were preserved on the"
                    f" attempt branch in commit {preservation_commit.revision}."
                )
        return _attempt_evidence(
            decision,
            profile,
            final_check,
            review_cycles,
            findings,
            details,
            note_commit_sha,
            reviews_performed=review_count,
        )

    def _wire_review_gate(self, prepared: object, initial_commits: Sequence[object]) -> None:
        """Let the model execution nudge the model toward ``code-review``.

        The stop gate blocks the first finish attempt while the branch holds
        commits added by this run and no review was observed. ``prepared``
        and the pre-dispatch commit snapshot delimit "added by this run";
        executors without the seam (older fakes) are left alone.
        """

        set_provider = getattr(self._model_executor, "set_new_commits_provider", None)
        if not callable(set_provider):
            return
        baseline = len(initial_commits)

        def new_commits() -> int:
            return len(self._workspace.commits_added(prepared)) - baseline  # type: ignore[arg-type]

        set_provider(new_commits)

    def _wire_operator_handoff(self) -> None:
        """Serve the pending operator handoff through this model execution.

        The provider reads the live control snapshot on every poll, so a
        request accepted during setup is picked up once execution starts;
        executors without the operator seams (older fakes) are left alone.
        """

        control = self._operator_handoff
        if control is None:
            return

        def provider() -> OperatorHandoff | None:
            try:
                snapshot = control.snapshot()
            except Exception:
                return None
            if snapshot is None or snapshot.get("acknowledgement") != "accepted":
                return None
            request_id = snapshot.get("request_id")
            accepted_at = _parse_operator_accepted_at(snapshot.get("accepted_at"))
            if not isinstance(request_id, str) or not request_id or accepted_at is None:
                return None
            return OperatorHandoff(request_id=request_id, accepted_at=accepted_at)

        def reporter(event: str, request_id: str) -> None:
            try:
                if event == "delivered":
                    control.mark_delivering(request_id)
                elif event == "begun":
                    control.mark_begun(request_id)
            except Exception as error:
                self._event_log(
                    "operator_handoff_report_failed",
                    f"event={event}; request={request_id}; {_exception_detail(error)}",
                    level="WARNING",
                )

        set_provider = getattr(
            self._model_executor, "set_operator_handoff_provider", None
        )
        if callable(set_provider):
            set_provider(provider)
        set_reporter = getattr(
            self._model_executor, "set_operator_handoff_reporter", None
        )
        if callable(set_reporter):
            set_reporter(reporter)

    def _archive_cli_transcripts(
        self, archive: AttemptArchive | None, issue_number: int
    ) -> None:
        """Copy this attempt's CLI session transcripts into the attempt archive.

        Best-effort: the copy never changes the attempt's status or outcome.
        Executors without the session seam (older fakes) and archives without
        the copy seam are left alone.
        """

        if archive is None:
            return
        copy = getattr(archive, "copy_cli_transcripts", None)
        if not callable(copy):
            return
        try:
            copy(
                getattr(self._model_executor, "cli_session_id", None),
                issue_number=issue_number,
                event_log=self._event_log,
            )
        except Exception as error:
            self._event_log(
                "transcripts_copy_failed",
                f"reason={_exception_detail(error)}",
                level="WARNING",
                issue_number=issue_number,
            )

    def _read_operator_request(
        self,
    ) -> tuple[str | None, dict | None]:
        """Split the pending handoff into served vs never-delivered.

        Returns the served request ID (this execution latched it) and, when a
        request is still accepted but this execution never observed it — it
        arrived after model execution or the executor has no operator seams —
        that late snapshot for ordinary handling.
        """

        control = self._operator_handoff
        if control is None:
            return None, None
        try:
            snapshot = control.snapshot()
        except Exception:
            return None, None
        if snapshot is None or snapshot.get("acknowledgement") != "accepted":
            return None, None
        latched = getattr(self._model_executor, "operator_request_id", None)
        if isinstance(latched, str) and latched == snapshot.get("request_id"):
            return snapshot["request_id"], None
        return None, snapshot

    def _settle_operator_request(
        self,
        claim: Claim,
        served_request_id: str | None,
        late_snapshot: dict | None,
        decision: CompletionDecision,
    ) -> None:
        """Fail operator handoffs this attempt cannot fulfill.

        A served request whose valid ``operator_request`` note published as a
        handoff stays pending: the lifecycle completes it only after the
        durable publication boundary. Everything else is failed here with its
        reason and the ordinary outcome, so intake still stops with the work
        retained. A late request never changes the attempt decision.
        """

        control = self._operator_handoff
        if control is None:
            return
        outcome_name = (
            decision.outcome.value if decision.outcome is not None else "unknown"
        )
        if late_snapshot is not None:
            request_id = late_snapshot.get("request_id", "unknown")
            cause = f"the attempt finalized with the ordinary outcome {outcome_name}"
            if decision.reasons:
                cause += f": {decision.reasons[0]}"
            control.fail(
                request_id,
                f"Operator handoff {request_id} was not delivered to model"
                f" execution, so it is not fulfilled; {cause}.",
            )
            return
        if served_request_id is None:
            return
        if decision.outcome is AttemptOutcome.HANDOFF:
            reason = self._operator_note_reason(claim.issue.number)
            if reason != _OPERATOR_HANDOFF_REASON:
                control.fail(
                    served_request_id,
                    f"Operator handoff {served_request_id} is not fulfilled: the"
                    f" committed handoff note cites reason {reason!r} instead of"
                    f" `{_OPERATOR_HANDOFF_REASON}`; the valid handoff still"
                    " publishes for human-approved continuation.",
                )
            return
        control.fail(
            served_request_id,
            f"Operator handoff {served_request_id} is not fulfilled: model"
            f" execution ended with the ordinary outcome {outcome_name}"
            " without a valid operator handoff note.",
        )

    def _operator_note_reason(self, issue_number: int) -> str | None:
        """Read the committed handoff note's reason, if it can be read."""

        try:
            text = _read_handoff_note(
                self._workspace.working_directory, issue_number
            )
        except Exception:
            return None
        if not isinstance(text, str) or not text:
            return None
        return _parse_handoff_note_fields(text).get("reason")

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
                self._workspace.working_directory,
                continuation=continuation,
            )
        except TypeError:
            preparation = self._verifier.prepare(
                profile,
                self._workspace.working_directory,
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
            _log_completion_decision(self._event_log, claim.issue.number, decision)
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
        workspace: AttemptWorkspace,
        profile_loader: Callable[[Path], RepositoryProfile],
        publisher: Publisher,
        attempt_runner: Callable[[Claim, RepositoryProfile, object], AttemptEvidence] | None = None,
        poll_interval: int = 60,
        sleeper: Callable[[float], None] = time.sleep,
        error_store: ConsecutiveErrorStore | None = None,
        max_consecutive_errors: int = 3,
        max_auto_continuations: int = 0,
        max_conflict_attempts: int = 0,
        command_runner: ConflictCheckRunner | None = None,
        conflict_model_executor: object | None = None,
        conflict_issue_comments: Callable[[TrackerIssue], tuple[str, ...]] | None = None,
        conflict_prompt_redactions: Sequence[str] = (),
        event_log: Callable[..., None] = (
            lambda event, detail="", level="INFO", issue_number=None: None
        ),
        attempt_archive_factory: Callable[[int, str], AttemptArchive] | None = None,
        completion_store: AttemptCompletionStore | None = None,
        control_store: ControlStore | None = None,
        repository: str = "",
        clock: Callable[[], datetime] | None = None,
        telemetry: TelemetryReporter | None = None,
        data_dir: Path | None = None,
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
        self._max_auto_continuations = max_auto_continuations
        self._max_conflict_attempts = max_conflict_attempts
        self._command_runner = command_runner
        self._conflict_model_executor = conflict_model_executor
        self._conflict_issue_comments = conflict_issue_comments
        self._conflict_prompt_redactions = tuple(conflict_prompt_redactions)
        self._event_log = event_log
        self._attempt_archive_factory = attempt_archive_factory
        self._completion_store = completion_store
        self._control_store = control_store
        self._repository = repository
        # Wall clock for operator-handoff deadline checks (acceptance + 240s
        # model work, + 120s publication, 360s absolute). Injected in tests;
        # the control store's own commit clock stamps the acceptance itself.
        self._clock = clock or (lambda: datetime.now(UTC))
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
        # True while a ``recovery retry`` or ``recovery release`` command is
        # executing (it unlocks during execution so status stays live). The
        # retained checkpoint is still a hold then: resume must stay rejected
        # until the command finishes, even though _attempt_processing is set.
        self._recovery_executing = False
        # Whether the pre-claim dirty-tree hold already logged
        # ``working_tree_dirty`` for the current dirty episode. The hold logs
        # once (naming the first offending paths) instead of on every poll,
        # and resets when the tree reads clean again.
        self._dirty_tree_reported = False
        # A startup hold entered when the attempt checkpoint or the
        # completion ledger cannot be read (``attempt_checkpoint_unusable`` /
        # ``attempt_completion_unusable``). While set, intake stays held
        # without exiting: the process keeps polling and serving the
        # control socket until the unreadable state is repaired or removed.
        # ``_startup_hold_reported`` rate-limits the intake heartbeat to
        # once per episode, mirroring the dirty-tree hold.
        self._startup_hold_detail: str | None = None
        self._startup_hold_kind: str | None = None
        self._startup_hold_reported = False
        self._telemetry = telemetry
        self._data_dir = data_dir
        # Seed the last-finished view from disk so it survives restarts;
        # a failure only leaves it null, never breaks startup.
        self._seed_telemetry_last_finished()
        # The last control section built for telemetry, with its
        # ``control_observed_at``. Served again while the control lock is
        # busy; ``None`` until the first successful build.
        self._telemetry_control: dict | None = None
        # Last publication progress and workspace-dirty state the lifecycle
        # itself observed, keyed by stable attempt ID. Publication flows
        # (live attempts, startup reconciliation, recovery retry) record the
        # publisher's verified result here; dirty handlers and cleanup
        # outcomes record the workspace state. Telemetry serves these
        # remembered values without network or git I/O; they stay absent
        # (``None``) until something is observed in this process. ``status``
        # keeps its live lookups and never reads these.
        self._observed_publication: dict[str, dict] = {}
        self._observed_workspace_dirty: dict[str, bool] = {}

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
            try:
                checkpoint = self._attempt_state.read()
            except AttemptStateError as error:
                # The checkpoint cannot be read, so resume cannot verify a
                # safe intake state: reject with the startup hold instead of
                # raising, keeping the control socket usable while held.
                self._enter_checkpoint_hold_locked(error)
                # Bound locally: the control path must not depend on an
                # assert (which vanishes under -O and would crash the
                # socket it is meant to keep serving if it ever fired).
                detail = self._startup_hold_detail or _checkpoint_hold_detail(error)
                return self._control_store.submit_resume(
                    request_id,
                    has_active_attempt=True,
                    recovery_hold=RecoveryHold(
                        issue_number=None,
                        attempt_id=None,
                        branch=None,
                        phase=None,
                        reason=detail,
                    ),
                )
            return self._control_store.submit_resume(
                request_id,
                has_active_attempt=checkpoint is not None,
                recovery_hold=self._recovery_hold_locked(checkpoint),
            )

    def submit_handoff(self, request_id: str) -> CommandRecord:
        """Durably record ``handoff now`` against the active attempt.

        Without an active attempt the command is rejected with no control
        change. With one, intake enters ``stopping`` and the request waits
        for the bounded model handoff: the model attempt runner delivers it
        at the next safe SDK boundary (or its single fallback) and observes
        the project-owned ``handoff`` skill. The command completes only after
        the handoff outcome, publication, issue release, and cleanup are
        durably finished; anything short of that marks it ``not fulfilled``
        with the branch, checkpoint, and working tree retained for recovery.
        """

        if self._control_store is None:
            raise ControlStoreError("Operator control is not configured")
        with self._control_lock:
            has_active_attempt = self._attempt_state.read() is not None
            return self._control_store.submit_handoff(
                request_id, has_active_attempt=has_active_attempt
            )

    def submit_next_issue(self, request_id: str, issue: object) -> CommandRecord:
        """Durably prioritize one eligible issue for the next permitted claim.

        The target is fetched and checked against the ordinary eligibility
        rules (open, unassigned, labelled ``ready-for-agent``, with a
        non-empty body and no open blocker) or, within the operator's
        automatic-continuation budget, the ``round-finished`` continuation
        rule before anything is recorded; an invalid, ineligible, or
        unverifiable target is rejected without changing the intake state or
        an existing priority. A newer accepted priority supersedes only the
        earlier priority, never a stop plan, and never resumes intake.
        Retrying the same request ID with the identical payload returns the
        stored acknowledgement without re-fetching GitHub or applying the
        command twice.
        """

        if self._control_store is None:
            raise ControlStoreError("Operator control is not configured")
        with self._control_lock:
            store = self._control_store
            if not isinstance(request_id, str) or not request_id.strip():
                return store.submit_next_issue(request_id, issue)
            try:
                existing = store.get_command(request_id)
            except ControlStoreError:
                raise
            except Exception as error:
                raise ControlStoreError("Control command could not be read") from error
            if existing is not None:
                # Idempotent retry: let the store enforce the
                # identical-payload rule without touching GitHub, so an
                # eligibility change after acceptance cannot alter history.
                return store.submit_next_issue(request_id, issue)
            number = parse_next_issue(issue)
            try:
                fresh = self._tracker.fetch_issue(number)
            except NextIssueRejectedError:
                raise
            except Exception as error:
                raise NextIssueRejectedError(
                    f"issue #{number} could not be verified: {error}."
                    " No control change was accepted."
                ) from error
            # The continuation count needs a comment read, so it is only
            # consulted when it can matter: a positive budget and a target
            # carrying only the continuation signal. Anything else follows
            # the ordinary rule with the count at zero, exactly as before.
            auto_count = 0
            if (
                fresh is not None
                and self._max_auto_continuations > 0
                and fresh.is_continuation_signal
            ):
                try:
                    auto_count = self._auto_continuation_count(fresh)
                except NextIssueRejectedError:
                    raise
                except Exception as error:
                    raise NextIssueRejectedError(
                        f"issue #{number} could not be verified: {error}."
                        " No control change was accepted."
                    ) from error
            reason = ineligibility_reason(
                fresh,
                number,
                max_auto_continuations=self._max_auto_continuations,
                auto_continuation_count=auto_count,
                max_conflict_attempts=self._max_conflict_attempts,
            )
            if reason is not None:
                raise NextIssueRejectedError(
                    f"{reason}. No control change was accepted."
                )
            return store.submit_next_issue(request_id, number)

    # -- operator recovery --------------------------------------------------

    def submit_recovery_retry(self, request_id: str, attempt_id: object) -> CommandRecord:
        """Recheck evidence and retry only unconfirmed safe finalization steps.

        The command runs through the live channel with a durable request ID.
        It never invokes the model, renews a handoff deadline, or overwrites
        ambiguous or conflicting evidence. Unknown, mismatched,
        already-finalized, or concurrently changing attempt IDs are rejected
        without a control change. On success the hold is cleared and the
        command completes; otherwise the remaining reason is reported and
        intake stays blocked.
        """

        from simple_coding_agent.recovery import parse_attempt_id

        if self._control_store is None:
            raise ControlStoreError("Operator control is not configured")
        parsed = parse_attempt_id(attempt_id)
        _check_recovery_request_id(request_id)
        with self._control_lock:
            store = self._control_store
            try:
                existing = store.get_command(request_id)
            except ControlStoreError:
                raise
            except Exception as error:
                raise ControlStoreError("Control command could not be read") from error
            if existing is not None:
                # Idempotent retry: the store enforces the identical-payload
                # rule; a terminal record is returned without re-executing.
                record = store.submit_recovery_retry(request_id, parsed)
                if record.acknowledgement is not CommandAcknowledgement.ACCEPTED:
                    return record
                # An accepted record for the same ID means this process already
                # accepted it but never finished: fall through and finish it
                # once instead of recording twice.
                checkpoint = self._validated_recovery_checkpoint_locked(parsed)
                self._attempt_processing = True
                self._recovery_executing = True
            else:
                checkpoint = self._validated_recovery_checkpoint_locked(parsed)
                record = store.submit_recovery_retry(request_id, parsed)
                self._attempt_processing = True
                self._recovery_executing = True
        try:
            success, detail = self._retry_retained_attempt(checkpoint)
        finally:
            with self._control_lock:
                self._attempt_processing = False
                self._recovery_executing = False
        with self._control_lock:
            if success:
                completed = store.complete_recovery_command(request_id, detail)
                return completed if completed is not None else store.get_command(request_id)  # type: ignore[return-value]
            failed = store.fail_recovery_command(request_id, detail)
            return failed if failed is not None else store.get_command(request_id)  # type: ignore[return-value]

    def submit_recovery_release(
        self, request_id: str, attempt_id: object, saved_at: object
    ) -> CommandRecord:
        """Abandon a retained attempt after the operator secured its work.

        The operator-provided ``saved-at`` reference is durably recorded with
        the attempt ID. A deduplicated issue comment stating the actual
        attempt outcome and failure is confirmed before any label or assignee
        change; a successful handoff or an unpublished branch is never
        claimed. Only then do the idempotent release, local cleanup,
        checkpoint finalization, and at-most-once accounting run. Any failure
        keeps the hold; work is never silently discarded. The issue is not
        requeued.
        """

        from simple_coding_agent.recovery import (
            parse_attempt_id,
            parse_saved_at,
        )

        if self._control_store is None:
            raise ControlStoreError("Operator control is not configured")
        parsed_attempt = parse_attempt_id(attempt_id)
        parsed_saved = parse_saved_at(saved_at)
        _check_recovery_request_id(request_id)
        with self._control_lock:
            store = self._control_store
            try:
                existing = store.get_command(request_id)
            except ControlStoreError:
                raise
            except Exception as error:
                raise ControlStoreError("Control command could not be read") from error
            if existing is not None:
                record = store.submit_recovery_release(
                    request_id, parsed_attempt, parsed_saved
                )
                if record.acknowledgement is not CommandAcknowledgement.ACCEPTED:
                    return record
                checkpoint = self._validated_recovery_checkpoint_locked(parsed_attempt)
                self._attempt_processing = True
                self._recovery_executing = True
            else:
                checkpoint = self._validated_recovery_checkpoint_locked(parsed_attempt)
                record = store.submit_recovery_release(
                    request_id, parsed_attempt, parsed_saved
                )
                self._attempt_processing = True
                self._recovery_executing = True
        try:
            success, detail = self._release_retained_attempt(checkpoint, parsed_saved)
        finally:
            with self._control_lock:
                self._attempt_processing = False
                self._recovery_executing = False
        with self._control_lock:
            if success:
                completed = store.complete_recovery_command(request_id, detail)
                return completed if completed is not None else store.get_command(request_id)  # type: ignore[return-value]
            failed = store.fail_recovery_command(request_id, detail)
            return failed if failed is not None else store.get_command(request_id)  # type: ignore[return-value]

    def _validated_recovery_checkpoint_locked(
        self, attempt_id: str
    ) -> AttemptCheckpoint:
        """Return the retained checkpoint for ``attempt_id`` or reject the command.

        Caller holds the control lock. Rejections cover unknown IDs (no
        checkpoint, or a workspace hold without a recoverable identity),
        mismatched identities, already-finalized attempts, and attempts that
        are concurrently changing (startup reconciliation or active work).
        """

        from simple_coding_agent.recovery import RecoveryRejectedError

        if self._recovering:
            raise RecoveryRejectedError(
                "Recovery is rejected while startup reconciliation is still"
                " running; intake is held until it finishes."
                " No control change was accepted."
            )
        if self._attempt_processing:
            raise RecoveryRejectedError(
                "Recovery is rejected while the attempt is concurrently"
                " changing; wait until the active work finishes."
                " No control change was accepted."
            )
        try:
            checkpoint = self._attempt_state.read()
        except AttemptStateError as error:
            raise RecoveryRejectedError(
                f"Recovery is rejected: the attempt checkpoint cannot be read ({error})."
                " No control change was accepted."
            ) from error
        if checkpoint is None:
            if self._workspace_is_dirty():
                raise RecoveryRejectedError(
                    "Recovery is rejected: the working tree holds unexplained"
                    " dirty or untracked work with no recoverable attempt"
                    " identity; manual repair is required before intake."
                    " No control change was accepted."
                )
            raise RecoveryRejectedError(
                f"Recovery is rejected: unknown attempt ID {attempt_id!r};"
                " no retained attempt matches it."
                " No control change was accepted."
            )
        if checkpoint.started_at != attempt_id:
            raise RecoveryRejectedError(
                f"Recovery is rejected: attempt ID {attempt_id!r} does not match"
                f" the retained attempt {checkpoint.started_at!r} for issue"
                f" #{checkpoint.issue_number}. No control change was accepted."
            )
        if self._completion_store is not None:
            try:
                if self._completion_store.is_finalized(attempt_id):
                    raise RecoveryRejectedError(
                        f"Recovery is rejected: attempt {attempt_id!r} is already"
                        " finalized; no retained work remains."
                        " No control change was accepted."
                    )
            except RecoveryRejectedError:
                raise
            except Exception as error:
                raise RecoveryRejectedError(
                    f"Recovery is rejected: the completion ledger cannot be read ({error})."
                    " No control change was accepted."
                ) from error
        return checkpoint

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
            reason = self._startup_hold_detail or (
                "attempt finalization is unresolved; the branch,"
                " checkpoint, and working tree are preserved for recovery"
            )
            if self._workspace_is_dirty():
                # Name the offending paths, as the intake dirty-tree hold
                # does, so the operator sees what blocks the hold.
                suffix = self._dirty_paths_suffix()
                reason += " The working tree is also dirty" + (suffix if suffix else ".")
            return RecoveryHold(
                issue_number=checkpoint.issue_number,
                attempt_id=checkpoint.started_at,
                branch=checkpoint.branch,
                phase=checkpoint.phase.value,
                reason=reason,
            )
        if checkpoint is not None and self._recovery_executing:
            # A recovery command owns the retained attempt right now (it
            # unlocks during execution so status stays live): resume stays
            # rejected until the command finishes instead of slipping
            # through the in-progress gap.
            return RecoveryHold(
                issue_number=checkpoint.issue_number,
                attempt_id=checkpoint.started_at,
                branch=checkpoint.branch,
                phase=checkpoint.phase.value,
                reason=(
                    "a recovery command is running for the retained attempt;"
                    " the branch, checkpoint, and working tree are preserved"
                    " until it finishes"
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
                reason=self._dirty_tree_detail(),
            )
        return None

    def _workspace_is_dirty(self) -> bool:
        """Whether the tree holds work outside any active attempt.

        Mirrors the pre-claim dirt guard in ``run_once``. An unreadable
        tree fails closed as a hold so resume never bypasses it.
        """

        try:
            return not self._workspace.is_clean()
        except Exception:
            return True

    def _remember_publication_result(
        self, attempt_id: str, published: object, phase: str | None = None
    ) -> None:
        """Remember the publisher's verified result for telemetry.

        Records ``comment_confirmed`` from ``comment_posted`` and the pull
        request number/url when the publisher returns one. Called in the
        publication flows (live attempts, startup reconciliation, recovery
        retry) where the lifecycle already holds the verified result, so
        telemetry can later serve it without network I/O. Never raises.
        """

        if not isinstance(attempt_id, str) or not attempt_id:
            return
        try:
            try:
                comment_confirmed: bool | None = bool(published.comment_posted)
            except AttributeError:
                comment_confirmed = None
            raw_pr = getattr(published, "pull_request", None)
            pull_request: dict | None = None
            if raw_pr is not None:
                if isinstance(raw_pr, dict):
                    number = raw_pr.get("number")
                    url = raw_pr.get("url")
                else:
                    number = getattr(raw_pr, "number", None)
                    url = getattr(raw_pr, "url", None)
                # A publisher that returns a PR object without a number still
                # counts as an observation (known, but unnumbered).
                pull_request = {"number": number, "url": url}
            resolved_phase = phase
            if resolved_phase is None:
                try:
                    checkpoint = self._attempt_state.read()
                    if (
                        checkpoint is not None
                        and checkpoint.started_at == attempt_id
                    ):
                        resolved_phase = checkpoint.phase.value
                except Exception:
                    resolved_phase = None
            self._observed_publication[attempt_id] = {
                "phase": resolved_phase,
                "comment_confirmed": comment_confirmed,
                "pull_request": copy.deepcopy(pull_request)
                if pull_request is not None
                else None,
            }
        except Exception:
            return

    def _archive_pull_request(
        self, issue_number: int, started_at: str, published: object
    ) -> None:
        """Write the published pull request into the attempt archive; never raises.

        Called at publication (in every flow that publishes), so the archive
        record carries the PR number and URL for the telemetry seed. An
        attempt without a pull request records an explicit ``None``.
        """

        try:
            archive = self._archive_for(issue_number, started_at)
            if archive is None:
                return
            archive.write_attempt(
                {"pull_request": _published_pull_request_value(published)}
            )
        except Exception as error:
            self._event_log(
                "archive_pull_request_failed",
                f"the pull request could not be archived: {error}",
                level="WARNING",
                issue_number=issue_number,
            )

    def _record_attempt_kind(self, issue_number: int, checkpoint: AttemptCheckpoint) -> None:
        """Record a conflict attempt's kind in its archive record; never raises.

        Called at claim time, so ``attempt.json`` carries
        ``attempt_kind: conflict`` for a conflict attempt even if the
        checkpoint is later lost. Ordinary attempts leave the archive
        untouched.
        """

        if (
            getattr(checkpoint, "attempt_kind", ATTEMPT_KIND_ORDINARY)
            != ATTEMPT_KIND_CONFLICT
        ):
            return
        try:
            archive = self._archive_for(issue_number, checkpoint.started_at)
            if archive is None:
                return
            archive.write_attempt({"attempt_kind": ATTEMPT_KIND_CONFLICT})
        except Exception as error:
            self._event_log(
                "archive_attempt_kind_failed",
                f"the attempt kind could not be archived: {error}",
                level="WARNING",
                issue_number=issue_number,
            )

    def _remember_workspace_dirty(self, attempt_id: str, dirty: bool) -> None:
        """Remember the workspace-dirty state for telemetry (never raises)."""

        if not isinstance(attempt_id, str) or not attempt_id:
            return
        try:
            self._observed_workspace_dirty[attempt_id] = bool(dirty)
        except Exception:
            return

    def _observe_and_remember_workspace_dirty(self, attempt_id: str) -> None:
        """Best-effort dirty check after publication; telemetry remembers it.

        Runs only in publication flows (which already perform git I/O), never
        on the telemetry path. An unreadable tree records nothing, leaving
        the remembered value ``None`` (not known yet).
        """

        try:
            dirty = not self._workspace.is_clean()
        except Exception:
            return
        self._remember_workspace_dirty(attempt_id, dirty)

    def _dirty_tree_hold_locked(self) -> bool:
        """Block the claim while the tree holds unexplained work (no exit).

        The caller holds the control lock with intake already permitted.
        Returns True when the claim must wait: the branch, checkpoint
        (none here), and working tree are left untouched for the
        operator, and the caller sleeps the normal poll interval with
        the control socket still serving. Logs ``working_tree_dirty``
        once per dirty episode (naming the first offending paths) and
        ``working_tree_clean`` on recovery, so a stopped instance stays
        quiet and a dirty poll loop stays quiet after the first report.
        An unreadable tree fails closed as a hold.
        """

        try:
            clean = self._workspace.is_clean()
        except Exception:
            clean = False
        if clean:
            if self._dirty_tree_reported:
                self._dirty_tree_reported = False
                self._event_log(
                    "working_tree_clean",
                    "repository working tree is clean again; intake hold cleared",
                    level="INFO",
                )
            return False
        if not self._dirty_tree_reported:
            self._dirty_tree_reported = True
            self._event_log(
                "working_tree_dirty",
                self._dirty_tree_detail(),
                level="WARNING",
            )
        return True

    def _dirty_tree_paths_locked(self) -> tuple[str, ...]:
        """Best-effort offending paths for the dirty-tree hold detail.

        Never raises: an unreadable tree (or a workspace without the
        ``dirty_paths`` seam) yields no paths, and the hold still blocks
        with the base message so the tree fails closed.
        """

        describe = getattr(self._workspace, "dirty_paths", None)
        if not callable(describe):
            return ()
        try:
            return tuple(describe())
        except Exception:
            return ()

    def _dirty_tree_detail(self) -> str:
        """Operator-facing dirty-tree hold detail naming offending paths."""

        base = (
            "repository working tree holds unexplained dirty or untracked"
            " work with no active attempt; manual repair is required before"
            " intake"
        )
        suffix = self._dirty_paths_suffix()
        if not suffix:
            return base + "."
        return base + suffix

    def _dirty_paths_suffix(self) -> str:
        """Best-effort ``; offending paths: ...`` suffix for hold details.

        Returns ``""`` when the tree reads clean or its paths cannot be
        read, so callers fail closed on the base message. Shared by the
        intake dirty-tree hold, the workspace-hold event, and the retained
        recovery hold, so every dirty-tree hold names the same paths.
        """

        listed = self._dirty_path_list()
        if not listed:
            return ""
        return f"; offending paths: {listed}."

    def _dirty_path_list(self) -> str:
        """Best-effort ``a, b (and N more)`` dirty-path list, or ``""``."""

        paths = self._dirty_tree_paths_locked()
        if not paths:
            return ""
        shown = ", ".join(paths[:_DIRTY_PATH_DISPLAY_LIMIT])
        extra = len(paths) - _DIRTY_PATH_DISPLAY_LIMIT
        return shown + (f" (and {extra} more)" if extra > 0 else "")

    def _enter_checkpoint_hold_locked(self, error: AttemptStateError) -> None:
        """Enter the startup hold for an unreadable checkpoint (caller holds the lock).

        Logs ``attempt_checkpoint_unusable`` once (when the hold is first
        entered) and the ``intake_held_for_recovery`` heartbeat once per
        episode, mirroring the dirty-tree hold. Later polls stay silent
        until the state reads cleanly again.
        """

        if self._startup_hold_detail is None:
            self._startup_hold_kind = "checkpoint"
            self._startup_hold_detail = _checkpoint_hold_detail(error)
            self._event_log(
                "attempt_checkpoint_unusable",
                self._startup_hold_detail,
                level="ERROR",
            )
        self._note_startup_hold_locked(self._startup_hold_detail)

    def _note_startup_hold_locked(self, detail: str, issue_number: int | None = None) -> None:
        """Log the startup-hold heartbeat once per episode (caller holds the lock)."""

        if self._startup_hold_reported:
            return
        self._startup_hold_reported = True
        self._event_log(
            "intake_held_for_recovery",
            f"{detail} sleeping {self._poll_interval}s",
            level="WARNING",
            issue_number=issue_number,
        )

    def _reread_after_startup_hold(self) -> tuple[bool, AttemptCheckpoint | None]:
        """Probe whether a startup hold cleared; return ``(cleared, checkpoint)``.

        A checkpoint hold clears once the checkpoint reads again; a
        completion-ledger hold additionally requires the ledger to read, so
        repairing only the checkpoint never drops a ledger hold. Any read
        failure keeps the hold (fails closed). Never raises.
        """

        try:
            checkpoint = self._attempt_state.read()
            if self._startup_hold_kind == "completion" and self._completion_store is not None:
                self._completion_store.read_all()
        except Exception:
            return (False, None)
        return (True, checkpoint)

    def _clear_startup_hold_locked(self) -> tuple[bool, AttemptCheckpoint | None]:
        """Clear a repaired startup hold and log the recovery (caller holds the lock).

        Returns ``(cleared, checkpoint)``: ``cleared`` is False while the
        state still fails to read (the hold stays, silently after the first
        heartbeat); on success the hold is reset and the re-read checkpoint
        (possibly ``None`` when the operator removed it) is returned for the
        caller to continue with.
        """

        cleared, checkpoint = self._reread_after_startup_hold()
        if not cleared:
            return (False, None)
        self._startup_hold_detail = None
        self._startup_hold_kind = None
        self._startup_hold_reported = False
        self._event_log(
            "startup_hold_cleared",
            "the unreadable state now reads cleanly; the startup hold is cleared",
            level="INFO",
        )
        return (True, checkpoint)

    # -- recovery execution -------------------------------------------------

    def _retry_retained_attempt(
        self, checkpoint: AttemptCheckpoint
    ) -> tuple[bool, str]:
        """Retry only unconfirmed safe steps for a retained checkpoint.

        Returns ``(cleared, detail)``: ``cleared`` is True only when the hold
        is fully cleared (outcome published with a confirmed comment, issue
        released, cleanup durable, accounting committed, checkpoint removed).
        Ambiguous remote state, conflicting changes, and persistent failures
        keep the hold with a reason. Never invokes the model.
        """

        from simple_coding_agent.recovery import retry_next_action

        attempt_id = checkpoint.started_at
        issue_number = checkpoint.issue_number
        hold_suffix = retry_next_action(attempt_id)

        try:
            claim = self._tracker.recover_claim(issue_number)
        except Exception as error:
            return False, (
                f"recovery retry for attempt {attempt_id} not fulfilled:"
                f" issue #{issue_number} is unavailable for recovery"
                f" ({_exception_detail(error)}); the hold remains. {hold_suffix}"
            )
        if claim is None:
            return False, (
                f"recovery retry for attempt {attempt_id} not fulfilled:"
                f" issue #{issue_number} is unavailable for recovery;"
                f" the hold remains. {hold_suffix}"
            )
        conflict = self._recovery_conflict_reason(claim, checkpoint)
        if conflict is not None:
            return False, (
                f"recovery retry for attempt {attempt_id} not fulfilled:"
                f" {conflict} The hold remains. {hold_suffix}"
            )
        try:
            self._workspace.prepare_for_profile_read()
            profile = self._profile_loader(self._workspace.working_directory)
            prepared = self._workspace.prepare_attempt(
                base_branch=profile.base_branch, issue_number=claim.issue.number
            )
        except DirtyWorkspaceError as error:
            self._remember_workspace_dirty(attempt_id, True)
            return False, (
                f"recovery retry for attempt {attempt_id} not fulfilled:"
                f" {_exception_detail(error)} The branch, checkpoint, and"
                f" working tree are preserved for inspection. {hold_suffix}"
            )
        except Exception as error:
            return False, (
                f"recovery retry for attempt {attempt_id} not fulfilled:"
                f" the workspace cannot be safely prepared"
                f" ({_exception_detail(error)}); the hold remains. {hold_suffix}"
            )
        if self._workspace.conflict_state().in_progress:
            return False, (
                f"recovery retry for attempt {attempt_id} not fulfilled:"
                " the worktree holds unresolved merge conflicts that the"
                " retry must not overwrite; manual repair is required."
                f" {hold_suffix}"
            )
        try:
            decision, details, note_sha, has_commits = self._infer_retry_decision(
                checkpoint, claim, prepared
            )
        except Exception as error:
            return False, (
                f"recovery retry for attempt {attempt_id} not fulfilled:"
                f" local evidence is missing or invalid"
                f" ({_exception_detail(error)}); the hold remains. {hold_suffix}"
            )
        shared_hold = self._recovery_shared_hold(
            claim.issue.number, attempt_id, decision.outcome
        )
        if shared_hold is not None:
            return False, (
                f"recovery retry for attempt {attempt_id} not fulfilled:"
                f" {shared_hold} the hold remains. {hold_suffix}"
            )
        try:
            published = self._publisher.publish(
                PublicationRequest(
                    issue_number=claim.issue.number,
                    issue_title=claim.issue.title,
                    branch=prepared.branch,
                    started_at=attempt_id,
                    decision=decision,
                    check_command="not run",
                    check_exit_code=None,
                    check_not_run_reason="not rerun during recovery retry",
                    review_cycles=0,
                    review_findings="not rerun during recovery retry",
                    details=self._handoff_publication_details(
                        claim, checkpoint, details, decision.outcome
                    ),
                    base_branch=profile.base_branch,
                    note_commit_sha=note_sha,
                    auto_merge_opt_in=_claim_time_opt_in(claim, checkpoint),
                    requeue_source=_claim_time_requeue_source(claim, checkpoint),
                    handoff_continuation_count=(
                        self._handoff_continuation_used(claim, checkpoint)
                        if decision.outcome is AttemptOutcome.HANDOFF
                        else None
                    ),
                    handoff_notify=(
                        decision.outcome is AttemptOutcome.HANDOFF
                        and self._handoff_needs_notification(claim, checkpoint)
                    ),
                )
            )
        except Exception as error:
            return False, (
                f"recovery retry for attempt {attempt_id} not fulfilled:"
                f" publication failed ({_exception_detail(error)});"
                " the hold remains."
                f" {hold_suffix}"
            )
        self._remember_publication_result(attempt_id, published)
        self._archive_pull_request(claim.issue.number, attempt_id, published)
        self._observe_and_remember_workspace_dirty(attempt_id)
        outcome = published.outcome
        comment_posted = _publication_confirmed(published)
        if not comment_posted and outcome is AttemptOutcome.HANDOFF and getattr(
            published, "handoff_posted", False
        ):
            # The shared handoff record is authoritative: a verified shared
            # publication confirms the handoff even when no individual
            # handoff result comment exists (deleted, or never posted).
            # The branch push behind it was already verified by publication,
            # so outstanding release/cleanup work can resume without
            # rerunning the model or changing the outcome. The confirmation
            # is re-read from the remote record (not trusted from the
            # in-memory flag alone) and bound to the handoff note revision
            # this retry published, whenever the publisher exposes the seam.
            if self._confirm_shared_handoff_for_retry(
                claim.issue.number, attempt_id, note_sha
            ):
                comment_posted = True
        if not comment_posted:
            return False, (
                f"recovery retry for attempt {attempt_id} not fulfilled:"
                " publication is unconfirmed (the result comment or the shared"
                " handoff record could not be verified); the hold remains."
                f" {hold_suffix}"
            )
        try:
            self._release(claim, outcome, checkpoint)
        except Exception as error:
            return False, (
                f"recovery retry for attempt {attempt_id} not fulfilled:"
                f" issue release failed ({_exception_detail(error)});"
                " the hold remains."
                f" {hold_suffix}"
            )
        try:
            retain_branch = getattr(published, "branch_url", None) is None and (
                has_commits
                and (
                    outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
                    or (
                        outcome is AttemptOutcome.INCOMPLETE
                        and decision.publication_path is PublicationPath.PARTIAL
                    )
                )
            )
            self._workspace.cleanup(
                base_branch=profile.base_branch,
                prepared=prepared,
                # comment_posted is True here (the unconfirmed path returned
                # above), so only genuinely unpublished work retains.
                retain_branch=bool(retain_branch),
            )
            self._observe_and_remember_workspace_dirty(attempt_id)
        except DirtyWorkspaceError as error:
            # Cleanup refused to leave the attempt branch while the tree is
            # dirty: the hold remains with everything preserved for
            # operator inspection. Unlike the attempt paths this does not
            # exit the process: recovery retry is an operator command, and
            # the failed result below already keeps the hold in place.
            self._emit_workspace_hold(claim.issue.number, error)
            return False, (
                f"recovery retry for attempt {attempt_id} not fulfilled:"
                f" local cleanup failed ({_exception_detail(error)});"
                " the hold remains."
                f" {hold_suffix}"
            )
        except Exception as error:
            return False, (
                f"recovery retry for attempt {attempt_id} not fulfilled:"
                f" local cleanup failed ({_exception_detail(error)});"
                " the hold remains."
                f" {hold_suffix}"
            )
        try:
            archive = self._archive_for(claim.issue.number, attempt_id)
            self._commit_finalization(
                checkpoint,
                claim.issue.number,
                prepared.branch,
                outcome,
                archive,
            )
            self._resolve_pending_handoff(
                issue_number=claim.issue.number,
                attempt_id=attempt_id,
                outcome=outcome,
                prepared=prepared,
            )
        except SystemExit as error:
            return False, (
                f"recovery retry for attempt {attempt_id} not fulfilled:"
                f" accounting is held ({_exception_detail(error)});"
                " the hold remains."
                f" {hold_suffix}"
            )
        except Exception as error:
            return False, (
                f"recovery retry for attempt {attempt_id} not fulfilled:"
                f" accounting failed ({_exception_detail(error)});"
                " the hold remains."
                f" {hold_suffix}"
            )
        return True, (
            f"recovery retry for attempt {attempt_id} completed;"
            f" issue #{issue_number} finalized with outcome {outcome.value};"
            " the hold is cleared"
        )

    def _release_retained_attempt(
        self, checkpoint: AttemptCheckpoint, saved_at: str
    ) -> tuple[bool, str]:
        """Abandon a retained checkpoint after the operator secured its work.

        Confirms a deduplicated issue comment stating the actual outcome and
        failure before any label or assignee change, never claiming a
        successful handoff or an unpublished branch. Only then runs the
        idempotent release, local cleanup, checkpoint finalization, and
        at-most-once accounting. Any failure keeps the hold.
        """

        from simple_coding_agent.recovery import (
            attempt_marker,
            release_comment_body,
        )

        attempt_id = checkpoint.started_at
        issue_number = checkpoint.issue_number
        try:
            claim = self._tracker.recover_claim(issue_number)
        except Exception as error:
            return False, (
                f"recovery release for attempt {attempt_id} not fulfilled:"
                f" issue #{issue_number} is unavailable for recovery"
                f" ({_exception_detail(error)}); the hold remains and no work"
                " was discarded."
            )
        if claim is None:
            return False, (
                f"recovery release for attempt {attempt_id} not fulfilled:"
                f" issue #{issue_number} is unavailable for recovery;"
                " the hold remains and no work was discarded."
            )
        try:
            self._workspace.prepare_for_profile_read()
            profile = self._profile_loader(self._workspace.working_directory)
            prepared = self._workspace.prepare_attempt(
                base_branch=profile.base_branch, issue_number=claim.issue.number
            )
        except DirtyWorkspaceError as error:
            self._remember_workspace_dirty(attempt_id, True)
            return False, (
                f"recovery release for attempt {attempt_id} not fulfilled:"
                f" {_exception_detail(error)} The branch, checkpoint, and"
                " working tree are preserved; no work was discarded."
            )
        except Exception as error:
            return False, (
                f"recovery release for attempt {attempt_id} not fulfilled:"
                f" the workspace cannot be safely prepared"
                f" ({_exception_detail(error)}); no work was discarded."
            )
        actual_outcome, failure_reason = self._infer_release_outcome(
            checkpoint, claim, prepared
        )
        branch = prepared.branch
        marker = attempt_marker(attempt_id)
        source = _claim_time_requeue_source(claim, checkpoint)
        comment_body = release_comment_body(
            issue_number=issue_number,
            attempt_id=attempt_id,
            actual_outcome=actual_outcome.value,
            failure_reason=failure_reason,
            saved_at=saved_at,
            branch=branch,
            requeue_source=source,
        )
        confirmed = self._confirm_release_comment(
            claim, marker, comment_body, attempt_id, requeue_source=source
        )
        if confirmed is None:
            return False, (
                f"recovery release for attempt {attempt_id} not fulfilled:"
                " remote state is ambiguous and the result comment could not"
                " be verified; the hold remains and no work was discarded."
            )
        if not confirmed:
            return False, (
                f"recovery release for attempt {attempt_id} not fulfilled:"
                " the result comment could not be confirmed on GitHub;"
                " the hold remains and no work was discarded."
            )
        try:
            # Plain release only: never add ``round-finished`` here, so the
            # release never claims a successful handoff. The comment above
            # already states the actual outcome and the failure.
            self._tracker.release_attempt(
                claim.issue.number, "ready-for-agent", claim.assignment.assignee_id
            )
        except Exception as error:
            return False, (
                f"recovery release for attempt {attempt_id} not fulfilled:"
                f" issue release failed ({_exception_detail(error)});"
                " the hold remains and no work was discarded."
            )
        try:
            self._workspace.cleanup(
                base_branch=profile.base_branch,
                prepared=prepared,
                retain_branch=True,
            )
        except Exception as error:
            return False, (
                f"recovery release for attempt {attempt_id} not fulfilled:"
                f" local cleanup failed ({_exception_detail(error)});"
                " the hold remains and no work was discarded."
            )
        try:
            archive = self._archive_for(claim.issue.number, attempt_id)
            self._commit_finalization(
                checkpoint, claim.issue.number, branch, actual_outcome, archive
            )
            self._resolve_pending_handoff(
                issue_number=claim.issue.number,
                attempt_id=attempt_id,
                outcome=actual_outcome,
                prepared=prepared,
            )
        except SystemExit as error:
            return False, (
                f"recovery release for attempt {attempt_id} not fulfilled:"
                f" accounting is held ({_exception_detail(error)});"
                " the hold remains and no work was discarded."
            )
        except Exception as error:
            return False, (
                f"recovery release for attempt {attempt_id} not fulfilled:"
                f" accounting failed ({_exception_detail(error)});"
                " the hold remains and no work was discarded."
            )
        return True, (
            f"recovery release for attempt {attempt_id} completed;"
            f" issue #{issue_number} released with outcome {actual_outcome.value};"
            f" retained work secured at {saved_at}; the hold is cleared"
        )

    def _infer_retry_decision(
        self, checkpoint: AttemptCheckpoint, claim: Claim, prepared: PreparedAttempt
    ) -> tuple[CompletionDecision, str, str | None, bool]:
        """Infer the safe republication decision without invoking the model."""

        archive = self._archive_for(claim.issue.number, checkpoint.started_at)
        commits = self._workspace.commits_added(prepared)
        has_commits = bool(commits)
        if checkpoint.phase is AttemptPhase.MODEL_RUNNING:
            if commits:
                if _final_check_was_interrupted(archive):
                    decision = CompletionDecision(
                        AttemptOutcome.INFRASTRUCTURE_ERROR,
                        False,
                        PublicationPath.PARTIAL,
                        (
                            "Final check was interrupted before completing;"
                            " committed work is preserved on the branch.",
                        ),
                    )
                else:
                    try:
                        self._attempt_state.transition(AttemptPhase.PUSHING)
                    except AttemptStateError:
                        pass
                    decision = CompletionDecision(
                        AttemptOutcome.INCOMPLETE,
                        False,
                        PublicationPath.PARTIAL,
                        ("Model execution was interrupted; preserved committed work.",),
                    )
            else:
                decision = _infrastructure_decision(
                    "Model execution was interrupted without commits."
                )
            return (decision, decision.reasons[0], None, has_commits)
        if checkpoint.phase in (AttemptPhase.PUSHING, AttemptPhase.PUBLISHING):
            note_commit = self._workspace.handoff_note_commit(
                prepared, claim.issue.number, attempt_id=checkpoint.started_at
            )
            if note_commit is not None:
                validation = _validate_handoff_note(
                    note_commit,
                    commits,
                    claim.issue.number,
                    self._workspace.working_directory,
                    prepared.base_revision,
                    note_commit_revisions=_this_attempt_note_commit_revisions(
                        self._workspace,
                        prepared,
                        claim.issue.number,
                        checkpoint.started_at,
                    ),
                )
                _emit_handoff_note_events(
                    self._event_log, claim.issue.number, validation
                )
                if validation.valid:
                    decision = CompletionDecision(
                        AttemptOutcome.HANDOFF,
                        False,
                        PublicationPath.PARTIAL,
                        (
                            "Recovered a previously committed handoff note"
                            " for publication.",
                        ),
                    )
                    details = _read_handoff_note(
                        self._workspace.working_directory, claim.issue.number
                    )
                    return (decision, details, note_commit.revision, has_commits)
                # An invalid recovered note is ordinary incomplete work, not
                # an infrastructure failure: the preserved commits still
                # publish on the partial path with the validation reason.
                decision = CompletionDecision(
                    AttemptOutcome.INCOMPLETE,
                    False,
                    PublicationPath.PARTIAL,
                    (validation.reason,),
                )
                return (decision, validation.reason, None, has_commits)
            decision = CompletionDecision(
                None,
                True,
                PublicationPath.COMPLETE,
                ("Recovered previously completed local evidence for publication.",),
            )
            return (decision, decision.reasons[0], None, has_commits)
        decision = _infrastructure_decision(
            "Attempt was interrupted before model execution."
        )
        return (decision, decision.reasons[0], None, has_commits)

    def _infer_release_outcome(
        self, checkpoint: AttemptCheckpoint, claim: Claim, prepared: PreparedAttempt
    ) -> tuple[AttemptOutcome, str]:
        """Infer the actual outcome recorded by ``recovery release``."""

        try:
            commits = self._workspace.commits_added(prepared)
        except Exception:
            commits = ()
        try:
            note_commit = self._workspace.handoff_note_commit(
                prepared, claim.issue.number, attempt_id=checkpoint.started_at
            )
        except Exception:
            note_commit = None
        if note_commit is not None:
            try:
                validation = _validate_handoff_note(
                    note_commit,
                    commits,
                    claim.issue.number,
                    self._workspace.working_directory,
                    prepared.base_revision,
                    note_commit_revisions=_this_attempt_note_commit_revisions(
                        self._workspace,
                        prepared,
                        claim.issue.number,
                        checkpoint.started_at,
                    ),
                )
            except Exception:
                validation = None
            if validation is not None:
                _emit_handoff_note_events(
                    self._event_log, claim.issue.number, validation
                )
            if validation is not None and validation.valid:
                return (
                    AttemptOutcome.HANDOFF,
                    "The handoff note was committed but finalization never"
                    f" completed in phase {checkpoint.phase.value}; the handoff"
                    " is recorded as not fulfilled (not a successful handoff),"
                    " and the retained branch is not published as a success.",
                )
        if commits:
            return (
                AttemptOutcome.INCOMPLETE,
                "Local work was preserved on the branch but finalization never"
                f" completed in phase {checkpoint.phase.value}; the operator"
                " secured the retained work and released the attempt.",
            )
        if checkpoint.phase in (AttemptPhase.CLAIMED, AttemptPhase.SETUP):
            return (
                AttemptOutcome.INFRASTRUCTURE_ERROR,
                "The attempt was interrupted before model execution produced"
                f" commits (phase {checkpoint.phase.value}); the operator"
                " secured the retained state and released the attempt.",
            )
        return (
            AttemptOutcome.INFRASTRUCTURE_ERROR,
            "Finalization was interrupted before its completion boundary"
            f" (phase {checkpoint.phase.value}); the operator secured the"
            " retained work and released the attempt.",
        )

    def _confirm_release_comment(
        self, claim: Claim, marker: str, body: str, attempt_id: str, *, requeue_source: str = "human"
    ) -> bool | None:
        """Confirm the deduplicated release comment; ``None`` means ambiguous.

        Uses the publisher's GitHub transport when it exposes the
        comment operations; otherwise falls back to the publisher boundary
        (which owns the same deduplication). Never claims a successful
        handoff or an unpublished branch.
        """

        github = getattr(self._publisher, "_github", None)
        repository = getattr(self._publisher, "_repository", None) or self._repository
        find = getattr(github, "find_attempt_comment", None)
        add = getattr(github, "add_comment", None)
        if callable(find) and callable(add):
            try:
                if find(repository, claim.issue.number, marker):
                    return True
            except Exception:
                return None
            try:
                add(repository, claim.issue.number, body)
            except Exception:
                return False
            try:
                return bool(find(repository, claim.issue.number, marker))
            except Exception:
                return None
        # Test fakes (and transports without comment operations): delegate to
        # the publisher boundary, which deduplicates by the same marker.
        try:
            published = self._publisher.publish(
                PublicationRequest(
                    issue_number=claim.issue.number,
                    issue_title=claim.issue.title,
                    branch=getattr(claim, "branch", f"agent/issue-{claim.issue.number}"),
                    started_at=attempt_id,
                    decision=CompletionDecision(
                        AttemptOutcome.INFRASTRUCTURE_ERROR,
                        False,
                        PublicationPath.NONE,
                        (body,),
                    ),
                    check_command="not run",
                    check_exit_code=None,
                    check_not_run_reason="not run during recovery release",
                    review_cycles=0,
                    review_findings="not run during recovery release",
                    details=body,
                    base_branch="main",
                    requeue_source=requeue_source,
                )
            )
        except Exception:
            return False
        return bool(getattr(published, "comment_posted", True))

    def _recovery_conflict_reason(
        self, claim: Claim, checkpoint: AttemptCheckpoint
    ) -> str | None:
        """Detect conflicting assignment or label changes before a retry.

        Returns a hold reason, or ``None`` when no conflict blocks the retry.
        A fetch failure is ambiguous (the hold remains); a confidently
        observed change by someone else (assignment or labels moved by
        another worker or a human) is conflicting (the retry must not
        overwrite it).
        """

        try:
            fresh = self._tracker.fetch_issue(checkpoint.issue_number)
        except Exception as error:
            return (
                "remote issue state is ambiguous"
                f" ({_exception_detail(error)}); the retry must not replay"
                " a side effect it cannot verify."
            )
        if fresh is None:
            return (
                f"issue #{checkpoint.issue_number} is unavailable on GitHub;"
                " the retry must not replay a side effect it cannot verify."
            )
        try:
            self_assigned = bool(self._tracker.is_self_assigned(fresh))
        except Exception as error:
            return (
                "remote assignment is ambiguous"
                f" ({_exception_detail(error)}); the retry must not replay"
                " a side effect it cannot verify."
            )
        if not self_assigned:
            holders = ", ".join(getattr(fresh, "assignee_logins", ()) or ())
            if holders:
                return (
                    f"issue #{checkpoint.issue_number} shows conflicting"
                    f" assignment changes (assigned to another assignee:"
                    f" {holders}); the retry"
                    " must not overwrite them and requires operator repair."
                )
            return (
                f"issue #{checkpoint.issue_number} shows conflicting"
                " assignment changes (it is assigned to another assignee,"
                " no longer this agent); the"
                " retry must not overwrite them and requires operator repair."
            )
        return None

    def _shared_record_seam(self) -> tuple[object, str] | None:
        """The shared-record lookup seam, or ``None`` when absent.

        Returns the ``(github, repository)`` pair backing
        ``find_handoff_record``. Publishers without the seam (older test
        doubles) read as absent and callers skip the shared-record check.
        """

        github = getattr(self._publisher, "_github", None)
        repository = getattr(self._publisher, "_repository", None) or self._repository
        if not callable(getattr(github, "find_handoff_record", None)):
            return None
        return (github, repository)

    def _recovery_shared_hold(
        self, issue_number: int, attempt_id: str, outcome: AttemptOutcome
    ) -> str | None:
        """Detect a shared handoff record that blocks a recovery retry.

        Returns a hold reason, or ``None`` when no shared record blocks the
        retry. Publishers without the ``find_handoff_record`` seam (older
        test doubles) skip the check. An unreadable remote state, an
        unparseable (or unsupported) record, a record naming a different
        issue or repository, or a record owned by a newer attempt all hold
        the retry: an older retained checkpoint never overwrites a newer
        shared record and is never treated as published merely because its
        own record was superseded. A record naming this attempt while the
        inferred outcome is not a handoff likewise holds: a confirmed
        handoff is never changed into another outcome. Anything else (no
        remote record, or an older record this retry may update) proceeds.
        """

        seam = self._shared_record_seam()
        if seam is None:
            return None
        github, repository = seam
        find = getattr(github, "find_handoff_record", None)
        try:
            existing = find(repository, issue_number)
        except Exception as error:
            return (
                "shared handoff record is ambiguous"
                f" ({_exception_detail(error)}); the retry must not replay"
                " a side effect it cannot verify."
            )
        if existing is None:
            # No remote record: the migration path creates it on republication.
            return None
        from simple_coding_agent.handoff_record import parse_handoff_record

        parsed = parse_handoff_record(getattr(existing, "body", "") or "")
        if parsed is None:
            return (
                "shared handoff record is unreadable; it is left untouched"
                " and the retry requires operator repair."
            )
        if parsed.issue_number != issue_number or parsed.repository != repository:
            return (
                "shared handoff record names a different issue or repository;"
                " the retry requires operator repair."
            )
        from simple_coding_agent import recovery as recovery_module

        if recovery_module.attempt_id_greater(parsed.attempt_id, attempt_id):
            return (
                "a newer attempt already owns the shared handoff record;"
                " the older retained attempt must not overwrite it and is"
                " not treated as published."
            )
        if parsed.attempt_id == attempt_id and outcome is not AttemptOutcome.HANDOFF:
            return (
                "the shared handoff record already confirms a handoff for"
                " this attempt; the retry must not change it into another"
                " outcome and requires operator repair."
            )
        return None

    def _confirm_shared_handoff_for_retry(
        self, issue_number: int, attempt_id: str, note_sha: str | None = None
    ) -> bool:
        """Independently confirm the shared handoff record during recovery retry.

        Re-reads the remote record for this attempt and issue, bound to the
        handoff note revision this retry just published (when one was
        committed), instead of trusting the in-memory publication flag
        alone, so a confirmed handoff resumes release and cleanup even
        when its individual result comment is missing. The branch push
        behind the record stays verified by publication itself (the record
        is first-wins per attempt and ``handoff_posted`` is only set once
        the push is verified), so no local branch comparison second-guesses
        it here. Publishers without the shared-record seam (older test
        doubles) keep trusting the verified flag, which stands alone
        there. Any verification failure reads as unconfirmed so the retry
        retains the attempt instead of guessing.
        """

        seam = self._shared_record_seam()
        if seam is None:
            return True
        github, repository = seam
        try:
            return (
                confirm_shared_handoff(
                    github,
                    repository,
                    issue_number,
                    attempt_id,
                    note_sha=note_sha,
                )
                is not None
            )
        except Exception:
            return False

    def _recovery_snapshot_locked(
        self, checkpoint: AttemptCheckpoint | None
    ) -> dict | None:
        """Build the status ``recovery`` section (caller holds the lock)."""

        from simple_coding_agent.recovery import (
            describe_hold,
            release_next_action,
            retry_next_action,
        )

        if checkpoint is None:
            if self._startup_hold_detail is not None:
                # The checkpoint or completion ledger cannot be read: report
                # the startup hold (read-only; never logs from status) with
                # the dirty paths when the tree is dirty too.
                reason = self._startup_hold_detail
                workspace_dirty = self._workspace_is_dirty()
                if workspace_dirty:
                    suffix = self._dirty_paths_suffix()
                    reason += " The working tree is also dirty" + (
                        suffix if suffix else "."
                    )
                return {
                    "attempt_id": None,
                    "issue_number": None,
                    "phase": None,
                    "outcome": None,
                    "branch": None,
                    "workspace": str(self._workspace.working_directory),
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
            if not self._workspace_is_dirty():
                return None
            reason = self._dirty_tree_detail()
            return {
                "attempt_id": None,
                "issue_number": None,
                "phase": None,
                "outcome": None,
                "branch": None,
                "workspace": str(self._workspace.working_directory),
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
        hold = self._recovery_hold_locked(checkpoint)
        if hold is None and not self._recovering:
            return None
        reason = hold.reason if hold is not None else "startup reconciliation is still running"
        outcome: str | None = None
        if self._completion_store is not None:
            try:
                completions = self._completion_store.read_all()
                terminal = completions.get(checkpoint.started_at)
                if terminal is not None:
                    outcome = terminal.outcome.value
            except Exception:
                outcome = None
        publication = self._publication_progress_locked(checkpoint)
        if self._recovering:
            next_action = (
                "Wait for startup reconciliation to finish; do not run recovery"
                " commands while it is still running."
            )
        elif outcome is not None:
            next_action = (
                "The attempt is finalized in the ledger; wait for the leftover"
                " checkpoint to be removed on the next intake cycle."
            )
        else:
            next_action = retry_next_action(checkpoint.started_at)
        return {
            "attempt_id": checkpoint.started_at,
            "issue_number": checkpoint.issue_number,
            "phase": checkpoint.phase.value,
            "outcome": outcome,
            "branch": checkpoint.branch,
            "workspace": str(self._workspace.working_directory),
            "workspace_dirty": self._workspace_is_dirty(),
            "checkpoint": True,
            "publication": publication,
            "hold_reason": describe_hold(
                issue_number=checkpoint.issue_number,
                attempt_id=checkpoint.started_at,
                phase=checkpoint.phase.value,
                reason=reason,
            ),
            "next_action": next_action
            if self._recovering or outcome is not None
            else f"{next_action} {release_next_action()}",
        }

    def _publication_progress_locked(
        self, checkpoint: AttemptCheckpoint
    ) -> dict:
        """Best-effort confirmed publication progress for status (read-only)."""

        from simple_coding_agent.recovery import attempt_marker

        progress: dict = {
            "phase": checkpoint.phase.value,
            "comment_confirmed": None,
            "pull_request": None,
        }
        github = getattr(self._publisher, "_github", None)
        repository = getattr(self._publisher, "_repository", None) or self._repository
        find_comment = getattr(github, "find_attempt_comment", None)
        if callable(find_comment):
            try:
                progress["comment_confirmed"] = bool(
                    find_comment(
                        repository, checkpoint.issue_number, attempt_marker(checkpoint.started_at)
                    )
                )
            except Exception:
                progress["comment_confirmed"] = None
        find_pr = getattr(github, "find_pull_request", None)
        if callable(find_pr):
            try:
                existing = find_pr(repository, checkpoint.branch)
                if existing is not None:
                    progress["pull_request"] = {
                        "number": getattr(existing, "number", None),
                        "url": getattr(existing, "url", None),
                    }
            except Exception:
                progress["pull_request"] = None
        return progress

    def _is_self_assignment(self, found: TrackerIssue) -> bool:
        """Whether a revalidated target is assigned to this agent itself.

        A failed identity read propagates as ambiguity with the priority
        preserved.
        """

        return bool(self._tracker.is_self_assigned(found))

    def _auto_continuation_count(self, issue: TrackerIssue) -> int:
        """The issue's automatic-continuation count since the last human requeue.

        Trackers without the seam (older test doubles) read as zero, which
        keeps the continuation path disabled-safe: with a zero budget the
        count is never consulted for eligibility.
        """

        counter = getattr(self._tracker, "automatic_continuation_count", None)
        if not callable(counter):
            return 0
        return int(counter(issue))

    def _handoff_continuation_used(
        self, claim: Claim, checkpoint: AttemptCheckpoint | None
    ) -> int | None:
        """The automatic-continuation standing the shared handoff record carries.

        A human requeue (or any non-automatic source, including conflict)
        stands at zero; an automatic continuation stands one past the current
        row, matching the ``Automatic continuation k/N used`` line the legacy
        handoff comment reports. ``None`` means the count could not be read,
        so the handoff stays legacy-only instead of publishing a guessed
        standing.
        """

        if _claim_time_requeue_source(claim, checkpoint) != REQUEUE_SOURCE_AUTOMATIC:
            return 0
        try:
            stored = self._stored_handoff_continuation(checkpoint)
            if stored is not None:
                return stored
            return self._auto_continuation_count(claim.issue) + 1
        except Exception:
            return None

    def _stored_handoff_continuation(
        self, checkpoint: AttemptCheckpoint | None
    ) -> int | None:
        """The previously computed standing for this attempt, if one persists.

        The first intent saved for an attempt always wins, so a retry after
        a timeout or restart reuses the stored standing instead of
        recomputing (and never consumes the budget twice). Publishers
        without the seam (older test doubles) read as absent, and a missing
        or invalid stored count falls back to a fresh computation.
        """

        if checkpoint is None:
            return None
        intent_of = getattr(self._publisher, "handoff_intent", None)
        if not callable(intent_of):
            return None
        try:
            intent = intent_of(checkpoint.started_at)
        except Exception:
            return None
        if intent is None:
            return None
        count = getattr(intent, "continuation_count", None)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return None
        return count

    def _conflict_failure_streak(self, issue: TrackerIssue) -> int:
        """The issue's consecutive failed conflict attempts since the last stop.

        Trackers without the seam (older test doubles) read as zero, which
        keeps the budget disabled-safe: with a zero budget the count is
        never consulted for escalation.
        """

        counter = getattr(self._tracker, "conflict_failure_streak", None)
        if not callable(counter):
            return 0
        return int(counter(issue))

    def _stamp_conflict_budget(
        self, claim: Claim, request: ConflictPublicationRequest
    ) -> ConflictPublicationRequest:
        """Stamp the pre-publish budget standing onto a conflict request.

        A ``success`` or ``skipped`` outcome resets the streak to zero; any
        other outcome counts one past the current streak and escalates when
        that count reaches the operator budget. The budget is enforced here
        at escalation time, never at claim time, so lowering it below a
        running streak escalates on the next failure. A ``success``-path
        push that leaves the remote head unchanged is still reported (and
        escalated) as ``no_change`` by publication, from ``conflict_prior``.
        """

        budget = self._max_conflict_attempts
        prior = self._conflict_failure_streak(claim.issue)
        if request.conflict_outcome in (
            CONFLICT_OUTCOME_SUCCESS,
            CONFLICT_OUTCOME_SKIPPED,
        ):
            index, escalate = 0, False
        else:
            index = prior + 1
            escalate = budget > 0 and index >= budget
        return replace(
            request,
            conflict_prior=prior,
            conflict_index=index,
            conflict_budget=budget,
            escalate=escalate,
        )

    def _maybe_escalate_conflict_to_human(
        self, claim: Claim, published: object, escalate: bool
    ) -> None:
        """Hand a budget-exhausted issue to a human after its comment posts.

        Only the escalating attempt's own result comment escalates, and only
        once its comment is confirmed posted: the label (created in the
        repository when missing) gates both the detector and the queue until
        a human removes it, which requeues with the full budget. The agent
        never removes ``ready-for-human``. A label failure is a warning, not
        a failed attempt: the posted escalation comment still stops the
        streak, so the next poll requeues instead of looping.
        """

        if not escalate:
            return
        if not getattr(published, "comment_posted", True):
            return
        try:
            escalator = getattr(self._tracker, "escalate_conflict_to_human", None)
            if callable(escalator):
                escalator(claim.issue)
            self._event_log(
                "conflict_escalated",
                f"issue #{claim.issue.number}; conflict attempts exhausted; "
                "labelled ready-for-human",
                level="WARNING",
                issue_number=claim.issue.number,
            )
        except Exception as error:
            self._event_log(
                "conflict_escalation_failed",
                _exception_detail(error),
                level="WARNING",
                issue_number=claim.issue.number,
            )

    def _emit_auto_continuation_claim(
        self, issue_number: int, auto_count: int | None
    ) -> None:
        """Emit the observability event for an automatic-continuation claim."""

        budget = self._max_auto_continuations
        used = "unknown" if auto_count is None else f"{auto_count}/{budget}"
        self._event_log(
            "auto_continuation_claimed",
            f"issue #{issue_number}; automatic continuation count {used}",
            level="INFO",
            issue_number=issue_number,
        )

    def _emit_conflict_attempt_claim(self, issue_number: int) -> None:
        """Emit the observability event for a conflict-attempt claim."""

        self._event_log(
            "conflict_attempt_claimed",
            f"issue #{issue_number}; merge-conflict requeue",
            level="INFO",
            issue_number=issue_number,
        )

    def _continuation_index_for_claim(
        self, source: str, auto_count: int | None
    ) -> int | None:
        """The claim-time automatic-continuation index stored for telemetry.

        A human requeue starts a new row at zero; an automatic continuation
        continues it at ``count + 1``. An unreadable count stays unknown
        (``None``) so telemetry reports it as unavailable instead of guessing.
        """

        if source != REQUEUE_SOURCE_AUTOMATIC:
            return 0
        if not isinstance(auto_count, int) or isinstance(auto_count, bool):
            return None
        return auto_count + 1

    def _telemetry_begin_for_checkpoint(self, checkpoint: AttemptCheckpoint) -> None:
        """Seed telemetry ``current_attempt`` at claim; never raises.

        A telemetry failure logs once and keeps the previous view so it can
        never break the claim it observes.
        """

        reporter = self._telemetry
        if reporter is None:
            return
        try:
            reporter.begin_attempt(
                {
                    "attempt_id": checkpoint.started_at,
                    "issue_number": checkpoint.issue_number,
                    "issue_title": checkpoint.issue_title,
                    "continuation_index": checkpoint.continuation_index,
                    "continuation_max": self._max_auto_continuations,
                }
            )
        except Exception as error:
            self._event_log(
                "telemetry_publish_failed",
                f"telemetry begin_attempt failed: {error}; keeping previous view",
                level="WARNING",
                issue_number=checkpoint.issue_number,
            )

    def _seed_telemetry_last_finished(self) -> None:
        """Seed telemetry ``last_finished`` from disk at startup; never raises.

        Reads the latest completed attempt archive record (token budget and
        pull request included). With no completed attempt on disk the view
        stays ``None``; a failure logs once and keeps it ``None`` so
        telemetry seeding can never break startup.
        """

        reporter = self._telemetry
        data_dir = self._data_dir
        if reporter is None or data_dir is None:
            return
        try:
            summary = load_last_finished(data_dir)
        except Exception as error:
            self._event_log(
                "telemetry_seed_failed",
                f"telemetry last_finished could not be seeded from disk: {error}",
                level="WARNING",
            )
            return
        if summary is None:
            return
        self._telemetry_finish(summary)

    def _telemetry_finish(self, summary: dict | None = None) -> None:
        """Record the finished attempt in telemetry at finalisation; never raises.

        Always clears ``current_attempt``; with ``summary`` also updates
        ``last_finished``. A telemetry failure logs once and keeps the
        previous view so it can never break the finalisation it observes.
        """

        reporter = self._telemetry
        if reporter is None:
            return
        try:
            finish = getattr(reporter, "finish_attempt", None)
            if not callable(finish):
                return
            if summary is None:
                finish()
                return
            try:
                finish(summary)
            except TypeError:
                finish()
        except Exception as error:
            self._event_log(
                "telemetry_publish_failed",
                f"telemetry finish_attempt failed: {error}; keeping previous view",
                level="WARNING",
            )

    def _handoff_publication_details(
        self,
        claim: Claim,
        checkpoint: AttemptCheckpoint | None,
        details: str,
        outcome: AttemptOutcome,
    ) -> str:
        """Append the automatic-continuation standing to a handoff's details.

        Written once, when the handoff is released: with a positive operator
        budget the result comment reports how much of the budget the issue
        has used (including this handoff's own requeue source) and whether
        the agent continues automatically or needs a human requeue. Any other
        outcome, or a zero budget, passes the details through unchanged.
        """

        if outcome is not AttemptOutcome.HANDOFF:
            return details
        line = self._auto_continuation_status_line(claim, checkpoint)
        if line is None:
            return details
        return f"{details}\n\n{line}" if details else line

    def _auto_continuation_status_line(
        self, claim: Claim, checkpoint: AttemptCheckpoint | None
    ) -> str | None:
        """The handoff comment's budget-standing line, or ``None`` when silent."""

        budget = self._max_auto_continuations
        if budget <= 0:
            return None
        source = _claim_time_requeue_source(claim, checkpoint)
        try:
            count_before = self._auto_continuation_count(claim.issue)
        except Exception:
            return None
        # The line reports where the issue stands once this handoff's own
        # result comment lands: a human-stamped comment resets the count to
        # zero (even after earlier automatic rounds), while an
        # automatic-stamped one extends the current row by one.
        used = count_before + 1 if source == REQUEUE_SOURCE_AUTOMATIC else 0
        shown = min(used, budget)
        if used >= budget:
            return (
                f"Automatic continuation {shown}/{budget} used; the automatic"
                " continuation budget is exhausted and the issue now needs a"
                " human requeue (re-apply `ready-for-agent`) before the agent"
                " can continue it."
            )
        return (
            f"Automatic continuation {shown}/{budget} used; the agent will"
            " continue automatically."
        )

    def _handoff_needs_notification(
        self, claim: Claim, checkpoint: AttemptCheckpoint | None
    ) -> bool:
        """Whether this handoff needs an explicit notification comment.

        True only when the automatic-continuation budget is exhausted by
        this handoff (the issue stops and needs a human requeue). Routine
        handoffs stay quiet in ``link`` mode; a failed or ambiguous shared
        publication notifies independently of this flag. An unreadable
        count notifies rather than silently suppressing, so a stopped
        issue is never hidden. A zero budget never notifies here: the
        default human-controlled requeue needs no standing line.
        """

        budget = self._max_auto_continuations
        if budget <= 0:
            return False
        try:
            count_before = self._auto_continuation_count(claim.issue)
        except Exception:
            return True
        source = _claim_time_requeue_source(claim, checkpoint)
        used = count_before + 1 if source == REQUEUE_SOURCE_AUTOMATIC else 0
        return used >= budget

    def get_command(self, request_id: str) -> CommandRecord | None:
        """Return one command's current durable acknowledgement."""

        if self._control_store is None:
            raise ControlStoreError("Operator control is not configured")
        with self._control_lock:
            return self._control_store.get_command(request_id)

    def submit_errors_reset(self, request_id: str) -> CommandRecord:
        """Reset the consecutive-error guard to 0 through the live process.

        The counter is cleared in the process (never from outside it, so
        the reset cannot race ``_record_terminal_outcome``) and the reset
        is durably recorded so ``agentctl command <id>`` reports it. The
        file write precedes the command commit under the same lock, so a
        crash between them leaves no record and a retry safely re-applies
        both; a retry arriving after the commit returns the stored
        acknowledgement without touching the counter again. The reset
        applies immediately even with an active attempt: it only changes
        the counter the next terminal outcome is compared against.
        """

        if self._control_store is None:
            raise ControlStoreError("Operator control is not configured")
        with self._control_lock:
            store = self._control_store
            try:
                existing = store.get_command(request_id)
            except ControlStoreError:
                raise
            except Exception as error:
                raise ControlStoreError("Control command could not be read") from error
            if existing is not None:
                return store.submit_errors_reset(request_id)
            previous_count = self._consecutive_error_count_or_none()
            self._reset_consecutive_errors()
            record = store.submit_errors_reset(
                request_id, previous_count=previous_count
            )
            self._event_log(
                "consecutive_errors_reset",
                f"request={request_id}; previous_count={previous_count}; count=0",
                level="INFO",
            )
            return record

    def _consecutive_error_count_or_none(self) -> int | None:
        """Return the current guard count, or ``None`` when unreadable."""

        if self._error_store is None:
            return None
        try:
            return self._error_store.read().count
        except (RuntimeError, OSError):
            return None

    def _reset_consecutive_errors(self) -> None:
        """Clear the guard counter in-process; no-op when unconfigured."""

        if self._error_store is None:
            return
        self._error_store.reset()

    def _consecutive_errors_snapshot_locked(self) -> dict | None:
        """Build the status ``consecutive_errors`` section (caller holds the lock)."""

        if self._error_store is None:
            return None
        try:
            state = self._error_store.read()
        except (RuntimeError, OSError) as error:
            return {
                "count": None,
                "max_consecutive_errors": self._max_consecutive_errors,
                "last_attempt_id": None,
                "last_success_at": None,
                "error": str(error),
            }
        return {
            "count": state.count,
            "max_consecutive_errors": self._max_consecutive_errors,
            "last_attempt_id": state.last_attempt_id,
            "last_success_at": state.last_success_at,
            "error": None,
        }

    def control_status(self, repository: str | None = None) -> dict:
        """Return one consistent live snapshot for ``status`` and the socket."""

        if self._control_store is None:
            raise ControlStoreError("Operator control is not configured")
        with self._control_lock:
            store = self._control_store
            try:
                checkpoint = self._attempt_state.read()
            except AttemptStateError as error:
                # Status is read-only: record the startup hold silently (no
                # event) so the operator still sees the hold reason instead
                # of an error while the process stays up.
                if self._startup_hold_detail is None:
                    self._startup_hold_kind = "checkpoint"
                    self._startup_hold_detail = _checkpoint_hold_detail(error)
                checkpoint = None
            active = (
                ActiveAttemptInfo(
                    issue_number=checkpoint.issue_number,
                    branch=checkpoint.branch,
                    phase=checkpoint.phase.value,
                    started_at=checkpoint.started_at,
                    issue_title=checkpoint.issue_title,
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
                next_issue=store.next_issue_snapshot(),
                stop_plan=store.stop_plan_snapshot(),
                recovery=self._recovery_snapshot_locked(checkpoint),
                handoff=store.handoff_snapshot(),
                consecutive_errors=self._consecutive_errors_snapshot_locked(),
            )

    def telemetry_snapshot(self) -> dict:
        """Serve one telemetry snapshot for the ``telemetry`` socket operation.

        Read-only: assembly goes through the telemetry reporter with this
        lifecycle as the control reader. Raises :class:`SnapshotFailedError`
        when telemetry is not configured or no control section can be built.
        """

        reporter = self._telemetry
        if reporter is None:
            raise SnapshotFailedError("telemetry is not configured for this process")
        return reporter.snapshot(self)

    def telemetry_control_snapshot(self) -> dict:
        """Return the network-free control section for one telemetry snapshot.

        The same payload :meth:`control_status` builds, but without workspace
        or GitHub I/O: no dirty-tree check runs, no tracker or publisher call
        is made, and the publication progress and workspace-dirty state are
        served from what the lifecycle itself last observed (``None``/"not
        known yet" until something is observed in this process). The section
        carries ``control_observed_at`` saying when it was built.

        Waits at most ``CONTROL_LOCK_TIMEOUT_SECONDS`` for the control lock so
        telemetry never holds up other control commands. While the lock is
        busy the last built section is reused; when none has been built yet
        this raises :class:`SnapshotFailedError`.
        """

        if self._control_store is None:
            raise SnapshotFailedError("operator control is not configured")
        if not self._control_lock.acquire(timeout=CONTROL_LOCK_TIMEOUT_SECONDS):
            cached = self._telemetry_control
            if cached is None:
                raise SnapshotFailedError(
                    "the control lock is busy and no control section has been built yet"
                )
            # A copy: the served snapshot must never alias the cache, or a
            # consumer mutating it would corrupt later reuses.
            return copy.deepcopy(cached)
        try:
            section = self._build_telemetry_control_section()
        finally:
            self._control_lock.release()
        self._telemetry_control = section
        return copy.deepcopy(section)

    def _build_telemetry_control_section(self) -> dict:
        """Build a fresh network-free control section (caller holds the lock)."""

        store = self._control_store
        assert store is not None  # checked by telemetry_control_snapshot
        try:
            checkpoint = self._attempt_state.read()
        except AttemptStateError as error:
            # Read-only, as in control_status: record the startup hold
            # silently (no event) so the operator still sees the hold reason.
            if self._startup_hold_detail is None:
                self._startup_hold_kind = "checkpoint"
                self._startup_hold_detail = _checkpoint_hold_detail(error)
            checkpoint = None
        active = (
            ActiveAttemptInfo(
                issue_number=checkpoint.issue_number,
                branch=checkpoint.branch,
                phase=checkpoint.phase.value,
                started_at=checkpoint.started_at,
                issue_title=checkpoint.issue_title,
            )
            if checkpoint is not None
            else None
        )
        section = build_status(
            repository=self._repository,
            intake=store.intake_state(),
            recovering=self._recovering,
            active_attempt=active,
            pending_command_id=store.pending_command_id(),
            get_command=store.get_command,
            recent_commands=store.recent_commands,
            next_issue=store.next_issue_snapshot(),
            stop_plan=store.stop_plan_snapshot(),
            recovery=self._recovery_snapshot_network_free(checkpoint),
            handoff=store.handoff_snapshot(),
            consecutive_errors=self._consecutive_errors_snapshot_locked(),
        )
        section["control_observed_at"] = format_timestamp(self._clock())
        return section

    def _recovery_hold_network_free(
        self, checkpoint: AttemptCheckpoint | None
    ) -> RecoveryHold | None:
        """Return the hold blocking ``resume``, without workspace I/O.

        Mirrors :meth:`_recovery_hold_locked` minus the dirty-tree check: the
        tree is never inspected, so a dirty tree alone never appears here and
        no workspace call is made. Never touches the tracker or publisher.
        """

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
            reason = self._startup_hold_detail or (
                "attempt finalization is unresolved; the branch,"
                " checkpoint, and working tree are preserved for recovery"
            )
            return RecoveryHold(
                issue_number=checkpoint.issue_number,
                attempt_id=checkpoint.started_at,
                branch=checkpoint.branch,
                phase=checkpoint.phase.value,
                reason=reason,
            )
        if checkpoint is not None and self._recovery_executing:
            return RecoveryHold(
                issue_number=checkpoint.issue_number,
                attempt_id=checkpoint.started_at,
                branch=checkpoint.branch,
                phase=checkpoint.phase.value,
                reason=(
                    "a recovery command is running for the retained attempt;"
                    " the branch, checkpoint, and working tree are preserved"
                    " until it finishes"
                ),
            )
        return None

    def _recovery_snapshot_network_free(
        self, checkpoint: AttemptCheckpoint | None
    ) -> dict | None:
        """Build the status ``recovery`` section without network or git I/O.

        Mirrors :meth:`_recovery_snapshot_locked`: the workspace dirty check
        is skipped (a dirty tree alone never produces a section here) and the
        publication progress and workspace-dirty state are served from what
        the lifecycle itself last observed (``None``/"not known yet" until
        something is observed in this process) because the GitHub lookups
        never run. Never touches the tracker, publisher, or workspace
        methods. Caller holds the control lock.
        """

        from simple_coding_agent.recovery import (
            describe_hold,
            release_next_action,
            retry_next_action,
        )

        if checkpoint is None:
            if self._startup_hold_detail is None:
                return None
            return {
                "attempt_id": None,
                "issue_number": None,
                "phase": None,
                "outcome": None,
                "branch": None,
                "workspace": str(self._workspace.working_directory),
                "workspace_dirty": None,
                "checkpoint": False,
                "publication": {
                    "phase": None,
                    "comment_confirmed": None,
                    "pull_request": None,
                },
                "hold_reason": self._startup_hold_detail,
                "next_action": (
                    "Repair or remove the unreadable state file described"
                    " in the hold reason; intake resumes automatically"
                    " once it reads cleanly."
                ),
            }
        hold = self._recovery_hold_network_free(checkpoint)
        if hold is None and not self._recovering:
            return None
        reason = hold.reason if hold is not None else "startup reconciliation is still running"
        outcome: str | None = None
        if self._completion_store is not None:
            try:
                completions = self._completion_store.read_all()
                terminal = completions.get(checkpoint.started_at)
                if terminal is not None:
                    outcome = terminal.outcome.value
            except Exception:
                outcome = None
        remembered = self._observed_publication.get(checkpoint.started_at)
        if remembered is not None:
            publication = {
                "phase": remembered.get("phase")
                if remembered.get("phase") is not None
                else checkpoint.phase.value,
                "comment_confirmed": remembered.get("comment_confirmed"),
                "pull_request": copy.deepcopy(remembered.get("pull_request"))
                if remembered.get("pull_request") is not None
                else None,
            }
        else:
            publication = {
                "phase": checkpoint.phase.value,
                "comment_confirmed": None,
                "pull_request": None,
            }
        workspace_dirty = self._observed_workspace_dirty.get(checkpoint.started_at)
        if self._recovering:
            next_action = (
                "Wait for startup reconciliation to finish; do not run recovery"
                " commands while it is still running."
            )
        elif outcome is not None:
            next_action = (
                "The attempt is finalized in the ledger; wait for the leftover"
                " checkpoint to be removed on the next intake cycle."
            )
        else:
            next_action = retry_next_action(checkpoint.started_at)
        return {
            "attempt_id": checkpoint.started_at,
            "issue_number": checkpoint.issue_number,
            "phase": checkpoint.phase.value,
            "outcome": outcome,
            "branch": checkpoint.branch,
            "workspace": str(self._workspace.working_directory),
            "workspace_dirty": workspace_dirty,
            "checkpoint": True,
            "publication": publication,
            "hold_reason": describe_hold(
                issue_number=checkpoint.issue_number,
                attempt_id=checkpoint.started_at,
                phase=checkpoint.phase.value,
                reason=reason,
            ),
            "next_action": next_action
            if self._recovering or outcome is not None
            else f"{next_action} {release_next_action()}",
        }

    def run_once(self) -> LifecycleResult:
        """Claim and process one issue, or sleep once when the queue is empty.

        A dirty working tree blocks the claim but never exits the process:
        the hold lives in ``_gated_claim`` (after the intake gate, so a
        stopped instance stays quiet), logs once per episode, and sleeps
        the normal poll interval with the control socket still serving.
        """

        if not self._startup_reconciled:
            self._startup_reconciled = True
            self._recovering = True
            try:
                recovered = self._reconcile_startup()
            finally:
                self._recovering = False
            if recovered is not None:
                return LifecycleResult(LifecycleStatus.ATTEMPTED, recovered)

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
        prepared: PreparedAttempt | None = None
        profile: RepositoryProfile | None = None
        outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
        retain_branch = False
        comment_posted = True
        publication_started_at: datetime | None = None
        # A conflict requeue routes to the conflict path (a fast path plus at
        # most one narrow model resolution session) instead of the ordinary
        # model-driven path; both share the except/finally boundary that
        # follows (holds, error publication, release, cleanup, and
        # finalization).
        is_conflict_claim = _is_conflict_claim(claim.issue.labels)
        try:
            if is_conflict_claim:
                conflict_request, prepared, profile = self._execute_conflict_attempt(
                    claim, checkpoint
                )
                publication_started_at = self._clock()
                published = self._publish_conflict_bounded(conflict_request, claim)
                outcome = published.outcome
                comment_posted = getattr(published, "comment_posted", True)
                retain_branch = getattr(published, "branch_url", None) is None and (
                    outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
                )
            else:
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
                rebase_conflicts = tuple(prepared.rebase_conflicts or ())
                prepared_detail = (
                    f"branch={prepared.branch}; base_branch={profile.base_branch}"
                )
                if rebase_conflicts:
                    prepared_detail += f"; rebase_conflicts={', '.join(rebase_conflicts)}"
                self._event_log(
                    "workspace_prepared",
                    prepared_detail,
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
                    publication_started_at = self._clock()
                    published = self._publish_terminal(
                        claim, checkpoint.started_at, prepared, profile, outcome,
                        timeout=self._handoff_publish_budget(),
                        auto_merge_opt_in=_claim_time_opt_in(claim, checkpoint),
                        requeue_source=_claim_time_requeue_source(claim, checkpoint),
                    )
                    comment_posted = getattr(published, "comment_posted", True)
                else:
                    self._wire_attempt_runner_handoff()
                    self._wire_attempt_telemetry_publisher()
                    evidence = self._attempt_runner(claim, profile, prepared)
                    publication_started_at = self._clock()
                    published = self._publish_bounded(
                        PublicationRequest(
                            issue_number=claim.issue.number,
                            issue_title=claim.issue.title,
                            branch=prepared.branch,
                            started_at=checkpoint.started_at,
                            decision=evidence.decision,
                            check_command=evidence.check_command,
                            check_exit_code=evidence.check_exit_code,
                            check_steps=evidence.check_steps,
                            check_failed_step=evidence.check_failed_step,
                            check_failed_index=evidence.check_failed_index,
                            check_not_run_reason=evidence.check_not_run_reason,
                            review_cycles=evidence.review_cycles,
                            reviews_performed=evidence.reviews_performed,
                            review_findings=evidence.review_findings,
                            details=self._handoff_publication_details(
                                claim,
                                checkpoint,
                                evidence.details,
                                evidence.decision.outcome,
                            ),
                            base_branch=profile.base_branch,
                            note_commit_sha=evidence.note_commit_sha,
                            auto_merge_opt_in=_claim_time_opt_in(claim, checkpoint),
                            requeue_source=_claim_time_requeue_source(claim, checkpoint),
                            handoff_continuation_count=(
                                self._handoff_continuation_used(claim, checkpoint)
                                if evidence.decision.outcome is AttemptOutcome.HANDOFF
                                else None
                            ),
                            handoff_notify=(
                                evidence.decision.outcome is AttemptOutcome.HANDOFF
                                and self._handoff_needs_notification(claim, checkpoint)
                            ),
                        )
                    )
                    outcome = published.outcome
                    comment_posted = _publication_confirmed(published)
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
            # this attempt never produced. The attempt stays held (no exit):
            # the finally boundary below logs ``attempt_finalization_held``
            # and retains the checkpoint, so the next poll holds on the
            # retained attempt instead of claiming new work.
            self._remember_workspace_dirty(checkpoint.started_at, True)
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
        except GitWorkspaceRecoveryError as error:
            self._event_log(
                "git_workspace_unrecoverable",
                _exception_detail(error),
                level="ERROR",
                issue_number=claim.issue.number,
            )
            comment_posted = False
            prepared = None
        except SystemExit:
            raise
        except RebaseConflictError as error:
            publication_started_at = self._clock()
            if is_conflict_claim:
                # The model already had its single resolution session, so a
                # rebase left in progress reaches here only defensively:
                # restore the branch best-effort and report the conflict
                # without pushing.
                if prepared is not None:
                    try:
                        self._workspace.abort_unresolved_rebase(prepared)
                    except Exception as abort_error:
                        self._event_log(
                            "conflict_abort_failed",
                            _exception_detail(abort_error),
                            level="WARNING",
                            issue_number=claim.issue.number,
                        )
                published = self._publish_conflict_bounded(
                    self._conflict_rebase_request(
                        claim, checkpoint, prepared, profile, None, error.conflicted_files
                    ),
                    claim,
                )
                outcome = published.outcome
            else:
                published = self._publish_rebase_conflict(
                    claim, checkpoint.started_at, prepared, profile, error,
                    timeout=self._handoff_publish_budget(),
                    auto_merge_opt_in=_claim_time_opt_in(claim, checkpoint),
                    requeue_source=_claim_time_requeue_source(claim, checkpoint),
                )
                outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
            comment_posted = getattr(published, "comment_posted", True)
        except Exception as error:
            self._event_log(
                "attempt_exception", _exception_detail(error), level="ERROR", issue_number=claim.issue.number
            )
            publication_started_at = self._clock()
            if is_conflict_claim:
                published = self._publish_conflict_bounded(
                    self._conflict_error_request(
                        claim, checkpoint, prepared, profile, _exception_detail(error)
                    ),
                    claim,
                )
                outcome = published.outcome
            else:
                published = self._publish_terminal(
                    claim, checkpoint.started_at, prepared, profile, outcome,
                    timeout=self._handoff_publish_budget(),
                    auto_merge_opt_in=_claim_time_opt_in(claim, checkpoint),
                    requeue_source=_claim_time_requeue_source(claim, checkpoint),
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
                publication_started_at = self._clock()
                if is_conflict_claim:
                    published = self._publish_conflict_bounded(
                        self._conflict_error_request(
                            claim, checkpoint, prepared, profile, _exception_detail(error)
                        ),
                        claim,
                    )
                    outcome = published.outcome
                else:
                    published = self._publish_terminal(
                        claim, checkpoint.started_at, prepared, profile, outcome,
                        timeout=self._handoff_publish_budget(),
                        auto_merge_opt_in=_claim_time_opt_in(claim, checkpoint),
                        requeue_source=_claim_time_requeue_source(claim, checkpoint),
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
            # Cleanup itself never deletes branches, and deletes files only
            # for reproducible artifacts the attempt branch ignored but the
            # base branch exposes as untracked; unexplained work is preserved.
            # A failed operator handoff holds the same boundary: the result
            # comment is already published truthfully, but the issue stays
            # assigned and the checkpoint, branch, and working tree stay
            # retained for `recovery retry` or `recovery release` instead of
            # finalizing. The hold is checked before release and again before
            # the ledger commit, so slow remote steps cannot slip a deadline
            # past the verdict.
            hold = comment_posted and self._hold_failed_handoff(
                issue_number=claim.issue.number,
                attempt_id=checkpoint.started_at,
                outcome=outcome,
                prepared=prepared,
                publication_started_at=publication_started_at,
            )
            released = False
            try:
                if comment_posted and not hold:
                    self._release(claim, outcome, checkpoint)
                    released = True
            finally:
                cleanup_ok = prepared is None
                try:
                    if prepared is not None and not hold:
                        self._workspace.cleanup(
                            base_branch=profile.base_branch if profile is not None else "main",
                            prepared=prepared,
                            retain_branch=retain_branch or not comment_posted,
                        )
                        cleanup_ok = True
                        self._observe_and_remember_workspace_dirty(
                            checkpoint.started_at
                        )
                except DirtyWorkspaceError as error:
                    # Cleanup refused to leave the attempt branch while the
                    # tree is dirty: hold intake with everything preserved
                    # for operator inspection instead of carrying uncommitted
                    # work onto the base or claiming further issues.
                    self._hold_dirty_workspace(claim.issue.number, error)
                finally:
                    if comment_posted and released and cleanup_ok and not hold:
                        # Publication already confirmed (comment posted) before
                        # release began: only the whole-command budget still
                        # applies, not the concluded publication window.
                        hold = self._hold_failed_handoff(
                            issue_number=claim.issue.number,
                            attempt_id=checkpoint.started_at,
                            outcome=outcome,
                            prepared=prepared,
                            publication_started_at=publication_started_at,
                            check_allowance=False,
                        )
                    if comment_posted and released and cleanup_ok and not hold:
                        self._commit_finalization(
                            checkpoint, claim.issue.number, prepared.branch
                            if prepared is not None else f"agent/issue-{claim.issue.number}",
                            outcome, archive,
                        )
                        self._resolve_pending_handoff(
                            issue_number=claim.issue.number,
                            attempt_id=checkpoint.started_at,
                            outcome=outcome,
                            prepared=prepared,
                            publication_started_at=publication_started_at,
                            check_allowance=False,
                        )
                    else:
                        self._event_log(
                            "attempt_finalization_held",
                            "Publication, release, or cleanup is unconfirmed, or a failed"
                            " operator handoff owns this attempt; the branch, "
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
        applies to that claim as the active attempt. A pending one-shot
        ``next issue`` priority is rechecked before the FIFO runnable queue
        at the next permitted claim boundary; a successful assignment
        consumes it once, while lost eligibility clears it as ``not
        fulfilled`` and falls back to FIFO in the same cycle. A retained
        checkpoint from a held attempt (startup reconciliation or a live
        attempt whose finalization is unconfirmed) blocks every claim until
        an operator clears it with ``recovery retry`` or ``recovery
        release``: claiming new work would collide with the retained
        checkpoint. Returns the
        claim and its checkpoint, or ``(None, None)`` when idle; the caller
        sleeps outside the lock so the control socket stays responsive.

        Cleanliness is checked only here, after the intake gate: a stopped
        instance sleeps quietly without ever inspecting the tree, and a
        dirty tree with intake running blocks the claim while the process
        stays up. The hold logs ``working_tree_dirty`` once per dirty
        episode (naming the first offending paths) and is visible in
        ``agentctl status`` through the ``recovery`` block, so the operator
        can repair the tree without losing the live view.
        """

        with self._control_lock:
            if self._startup_hold_detail is not None:
                # A startup hold (unreadable checkpoint or completion
                # ledger) is active, with or without operator control:
                # resume only once the state reads cleanly again
                # (operator repair); otherwise stay held silently after
                # the first heartbeat.
                cleared, _ = self._clear_startup_hold_locked()
                if not cleared:
                    return (None, None)
                # The hold cleared: fall through and read fresh state below.
            if self._control_store is not None:
                intake = self._control_store.intake_state()
                if intake is not IntakeState.RUNNING:
                    if intake is IntakeState.STOPPING:
                        try:
                            stopped_empty = self._attempt_state.read() is None
                        except AttemptStateError as error:
                            # The checkpoint cannot be read, so the
                            # waited-for attempt may still exist: hold
                            # instead of finishing the stop.
                            self._enter_checkpoint_hold_locked(error)
                            return (None, None)
                        if stopped_empty:
                            # The waited-for attempt is gone without finalizing
                            # through this path (e.g. it was reconciled on a
                            # previous start); finish the stop instead of holding.
                            # A pending handoff can never still be observed, so
                            # it is failed rather than waited on forever.
                            self._fail_orphan_handoff_locked()
                            self._control_store.complete_pending_stop()
                    self._event_log(
                        "intake_stopped",
                        f"issue intake is {self._control_store.intake_state().value};"
                        f" sleeping {self._poll_interval}s",
                        level="INFO",
                    )
                    return (None, None)
                try:
                    retained = self._attempt_state.read()
                except AttemptStateError as error:
                    # The checkpoint became unreadable after startup:
                    # hold without exiting, as startup reconciliation
                    # does, instead of claiming over unknown state.
                    self._enter_checkpoint_hold_locked(error)
                    return (None, None)
                if retained is not None:
                    # A retained checkpoint means an earlier attempt is held
                    # for recovery (or a recovery command is working through
                    # it now): no new claim may begin until the operator
                    # clears the hold, so intake sleeps instead of colliding.
                    self._event_log(
                        "intake_held_for_recovery",
                        f"retained attempt {retained.started_at!r} for issue"
                        f" #{retained.issue_number} is held for recovery;"
                        f" sleeping {self._poll_interval}s",
                        level="WARNING",
                        issue_number=retained.issue_number,
                    )
                    return (None, None)
            if self._dirty_tree_hold_locked():
                # No claim while unexplained work sits in the tree; the
                # branch, checkpoint (none here), and working tree are left
                # untouched for the operator, and the caller sleeps the
                # normal poll interval with the socket still serving.
                return (None, None)
            if self._control_store is not None:
                prioritized = self._claim_prioritized_locked()
                if prioritized is not None:
                    return prioritized
                if self._control_store.next_issue_snapshot() is not None:
                    # The priority was cleared as not fulfilled above; the
                    # FIFO selection below runs in the same intake cycle.
                    pass
            self._event_log("polling_for_issue", "", level="INFO")
            claim = self._tracker.claim_next()
            if claim is None:
                self._event_log(
                    "no_eligible_issue_found",
                    f"no eligible issue available; sleeping {self._poll_interval}s",
                    level="INFO",
                )
                return (None, None)
            source = requeue_source_for_labels(claim.issue.labels)
            is_conflict = _is_conflict_claim(claim.issue.labels)
            attempt_kind = ATTEMPT_KIND_CONFLICT if is_conflict else ATTEMPT_KIND_ORDINARY
            if is_conflict:
                # A conflict requeue is transparent to the continuation
                # budget: it records a conflict requeue source with no
                # continuation index, and never emits the automatic event.
                source = REQUEUE_SOURCE_CONFLICT
                auto_count = None
                continuation_index = None
            elif source == REQUEUE_SOURCE_AUTOMATIC:
                try:
                    auto_count = self._auto_continuation_count(claim.issue)
                except Exception:
                    auto_count = None
                continuation_index = self._continuation_index_for_claim(source, auto_count)
            else:
                auto_count = None
                continuation_index = self._continuation_index_for_claim(source, auto_count)
            checkpoint = self._attempt_state.start(
                issue_number=claim.issue.number,
                branch=f"agent/issue-{claim.issue.number}",
                auto_merge_opt_in=AUTO_MERGE_LABEL in claim.issue.labels,
                requeue_source=source,
                issue_title=claim.issue.title,
                continuation_index=continuation_index,
                attempt_kind=attempt_kind,
            )
            self._record_attempt_kind(claim.issue.number, checkpoint)
            self._attempt_processing = True
            self._telemetry_begin_for_checkpoint(checkpoint)
            if is_conflict:
                self._emit_conflict_attempt_claim(claim.issue.number)
            elif source == REQUEUE_SOURCE_AUTOMATIC:
                self._emit_auto_continuation_claim(claim.issue.number, auto_count)
            if self._control_store is not None:
                # The final counted attempt of a ``stop --after`` plan enters
                # ``stopping`` at claim; earlier claims leave intake running
                # and an empty queue preserves the remaining count untouched.
                self._control_store.enter_stopping_for_final_claim()
            return (claim, checkpoint)

    def _claim_prioritized_locked(
        self,
    ) -> tuple[Claim | None, AttemptCheckpoint | None] | None:
        """Attempt the pending priority claim; ``None`` means use FIFO.

        The caller holds the control lock with intake already permitted.
        Returns the prioritized ``(claim, checkpoint)`` on success, or
        ``None`` when no priority is pending or when the target lost
        eligibility (the priority is then terminal ``not fulfilled`` and the
        caller falls back to FIFO). A transport or assignment failure
        propagates with the priority preserved: it is never reported as
        completed and no other claim may begin until reconciled.
        """

        store = self._control_store
        if store is None:  # pragma: no cover - guarded by the caller
            return None
        snapshot = store.next_issue_snapshot()
        if snapshot is None:
            return None
        target_number = int(snapshot["issue_number"])
        try:
            fresh = self._tracker.fetch_issue(target_number)
            # As in ``submit_next_issue``: the comment read behind the count
            # only runs when it can matter, so a zero budget (or an ordinary
            # target) revalidates exactly as before.
            auto_count = 0
            if (
                fresh is not None
                and self._max_auto_continuations > 0
                and fresh.is_continuation_signal
            ):
                auto_count = self._auto_continuation_count(fresh)
        except Exception as error:
            self._event_log(
                "next_issue_claim_ambiguous",
                f"issue #{target_number} could not be revalidated: {error};"
                " the priority stays pending and no other claim may begin"
                " until reconciled",
                level="ERROR",
                issue_number=target_number,
            )
            raise
        if fresh is not None and fresh.number != target_number:  # pragma: no cover - keyed fetch
            raise ControlStoreError(
                "The tracker returned a different issue than the pending priority"
            )
        reason = ineligibility_reason(
            fresh,
            target_number,
            max_auto_continuations=self._max_auto_continuations,
            auto_continuation_count=auto_count,
            max_conflict_attempts=self._max_conflict_attempts,
        )
        if reason is not None:
            if fresh is not None and self._is_self_assignment(fresh):
                # Our own assignment is an ambiguous prior claim (the
                # target was unassigned at acceptance and this worker's
                # own assignment went through while its reply was lost),
                # never lost eligibility: hold it for reconciliation
                # instead of failing the priority and falling back to FIFO.
                self._event_log(
                    "next_issue_claim_ambiguous",
                    f"issue #{target_number} is assigned to this agent after"
                    " an unconfirmed assignment; the priority stays pending"
                    " and no other claim may begin until reconciled",
                    level="ERROR",
                    issue_number=target_number,
                )
                raise ControlStoreError(
                    f"Issue #{target_number} has an ambiguous assignment to"
                    " this agent; the priority stays pending until reconciled."
                )
            store.fail_next_issue(reason)
            self._event_log(
                "next_issue_not_fulfilled",
                f"issue #{target_number} not fulfilled: {reason};"
                " falling back to the runnable queue",
                level="WARNING",
                issue_number=target_number,
            )
            return None
        assert fresh is not None  # eligibility implies presence
        try:
            claim = self._tracker.claim_verified(fresh)
        except ClaimLostError as error:
            # Another assignee won the priority target during the settle
            # window (the tracker already removed only our assignment and
            # emitted ``claim_lost``): the priority ends terminally as not
            # fulfilled and this same cycle falls back to FIFO.
            holders = ", ".join(error.other_assignees)
            suffix = f" ({holders})" if holders else ""
            reason = (
                f"issue #{target_number} claimed by another assignee{suffix}"
            )
            store.fail_next_issue(reason)
            self._event_log(
                "next_issue_not_fulfilled",
                f"issue #{target_number} not fulfilled: {reason};"
                " falling back to the runnable queue",
                level="WARNING",
                issue_number=target_number,
            )
            return None
        except Exception as error:
            self._event_log(
                "next_issue_claim_ambiguous",
                f"issue #{target_number} assignment is ambiguous: {error};"
                " the priority stays pending and no other claim may begin"
                " until reconciled",
                level="ERROR",
                issue_number=target_number,
            )
            raise
        completed = store.complete_next_issue(claim.issue.number)
        if completed is None:
            # The claimed issue does not match the pending priority: never
            # report a confused assignment as a completed priority claim.
            self._event_log(
                "next_issue_claim_ambiguous",
                f"claimed issue #{claim.issue.number} does not match the"
                f" prioritized issue #{target_number}; the priority stays"
                " pending and no other claim may begin until reconciled",
                level="ERROR",
                issue_number=target_number,
            )
            raise ControlStoreError(
                "The prioritized claim does not match the pending priority"
            )
        self._event_log(
            "next_issue_claimed",
            f"title={claim.issue.title!r} (prioritized next issue"
            f" #{target_number})",
            level="INFO",
            issue_number=claim.issue.number,
        )
        source = requeue_source_for_labels(fresh.labels)
        is_conflict = _is_conflict_claim(fresh.labels)
        if is_conflict:
            source = REQUEUE_SOURCE_CONFLICT
            continuation_index = None
            self._emit_conflict_attempt_claim(claim.issue.number)
        elif source == REQUEUE_SOURCE_AUTOMATIC:
            self._emit_auto_continuation_claim(claim.issue.number, auto_count)
            continuation_index = self._continuation_index_for_claim(
                source, auto_count
            )
        else:
            continuation_index = 0
        checkpoint = self._attempt_state.start(
            issue_number=claim.issue.number,
            branch=f"agent/issue-{claim.issue.number}",
            auto_merge_opt_in=AUTO_MERGE_LABEL in claim.issue.labels,
            requeue_source=source,
            issue_title=claim.issue.title,
            continuation_index=continuation_index,
            attempt_kind=ATTEMPT_KIND_CONFLICT if is_conflict else ATTEMPT_KIND_ORDINARY,
        )
        self._record_attempt_kind(claim.issue.number, checkpoint)
        self._attempt_processing = True
        self._telemetry_begin_for_checkpoint(checkpoint)
        return (claim, checkpoint)

    def _wire_attempt_runner_handoff(self) -> None:
        """Attach the live operator handoff to a runner that can serve it.

        Plain-function runners (older test doubles) have no operator seam and
        are left alone: any pending handoff then resolves as unserved at the
        durable finalization boundary.
        """

        setter = getattr(self._attempt_runner, "set_operator_handoff", None)
        if not callable(setter):
            return
        if self._control_store is None:
            setter(None)
            return
        setter(_ControlOperatorHandoff(self._control_store, self._event_log))

    def _wire_attempt_telemetry_publisher(self) -> None:
        """Attach the telemetry reporter to the executor that publishes live figures.

        The executor publishes an immutable attempt view after every limit
        check; the reporter swaps the reference under a lock held only for
        the swap, so the listener thread serves live figures without ever
        reading the token ledger. Runners and executors without the seam
        (older test doubles) are left alone. Never raises.
        """

        try:
            reporter = self._telemetry
            if reporter is None:
                return
            executor = getattr(self._attempt_runner, "_model_executor", None)
            setter = getattr(executor, "set_attempt_publisher", None)
            if not callable(setter):
                return
            setter(reporter.publish_attempt_view)
        except Exception as error:
            self._event_log(
                "telemetry_publish_failed",
                f"telemetry publisher wiring failed: {error}; continuing without live figures",
                level="WARNING",
            )

    def _handoff_publish_budget(self) -> float | None:
        """Bound one publication call by the pending handoff's remaining budget.

        Returns ``min(120, 360 - elapsed-since-acceptance)`` seconds, clamped
        at zero, while an operator handoff is accepted; otherwise None, keeping
        the publisher's configured timeout. The budget is anchored at the
        original acceptance stored in control, so a restart never renews it.
        """

        store = self._control_store
        if store is None:
            return None
        try:
            snapshot = store.handoff_snapshot()
        except (ControlStoreError, RuntimeError, OSError):
            return None
        if snapshot is None or snapshot.get("acknowledgement") != "accepted":
            return None
        accepted_at = _parse_operator_accepted_at(snapshot.get("accepted_at"))
        if accepted_at is None:
            return None
        remaining = HANDOFF_ABSOLUTE_DEADLINE_SECONDS - (
            self._clock() - accepted_at
        ).total_seconds()
        return max(0.0, min(float(HANDOFF_PUBLICATION_ALLOWANCE_SECONDS), remaining))

    def _publish_bounded(self, request: PublicationRequest):
        """Publish, bounding the call by the pending handoff's remaining budget."""

        budget = self._handoff_publish_budget()
        if budget is None:
            published = self._publisher.publish(request)
        else:
            published = self._publisher.publish(request, timeout=budget)
        # Remember the verified result for telemetry (no extra lookup), plus
        # the workspace state observed right after publication. The pull
        # request is also written into the attempt's archive record here, at
        # publication, so the telemetry seed finds it after a restart.
        self._remember_publication_result(request.started_at, published)
        self._archive_pull_request(
            request.issue_number, request.started_at, published
        )
        self._observe_and_remember_workspace_dirty(request.started_at)
        return published

    def _handoff_completion_cause(
        self,
        snapshot: dict,
        *,
        issue_number: int,
        outcome: AttemptOutcome,
        prepared: PreparedAttempt | None,
        publication_started_at: datetime | None = None,
        check_allowance: bool = True,
    ) -> str | None:
        """Decide whether an accepted handoff can complete; None means it can.

        The handoff completes only when the attempt outcome is a published
        handoff, the model began its skill, the committed note is a valid
        ``operator_request`` note, publication confirmed within 120 seconds
        after the valid local handoff, and the whole command finished within
        360 seconds of its original acceptance. Otherwise returns the
        ``not fulfilled`` cause. ``check_allowance`` is False only for the
        pre-commit recheck after a confirmed publication: the 120-second
        window already concluded when the result comment posted, so only the
        360-second whole-command budget still applies.
        """

        note_valid, note_why = self._check_operator_note(issue_number, prepared)
        if outcome is not AttemptOutcome.HANDOFF:
            return f"the attempt finalized with the ordinary outcome {outcome.value}"
        if snapshot.get("begun") is not True:
            return "model execution never began the handoff skill"
        if not note_valid:
            return f"the committed handoff note is not a valid operator note ({note_why})"
        now = self._clock()
        accepted_at = _parse_operator_accepted_at(snapshot.get("accepted_at"))
        if accepted_at is None:
            return (
                "the original acceptance time is unavailable, so the"
                " 360-second budget cannot be verified"
            )
        if (now - accepted_at).total_seconds() > HANDOFF_ABSOLUTE_DEADLINE_SECONDS:
            return (
                "the 360-second budget from acceptance elapsed before"
                " publication, release, and cleanup finished"
                f" (accepted {snapshot.get('accepted_at')})"
            )
        if (
            check_allowance
            and publication_started_at is not None
            and (now - publication_started_at).total_seconds()
            > HANDOFF_PUBLICATION_ALLOWANCE_SECONDS
        ):
            return (
                "publication was not confirmed within 120 seconds after"
                " the valid local handoff"
            )
        return None

    def _handoff_owned_by_attempt(self, attempt_id: str) -> bool:
        """Whether the tracked handoff failure was recorded for this attempt.

        The hold records its retaining attempt ID in the ``not fulfilled``
        reason, so a crash replay can tell a hold it must preserve from a
        stale failure owned by an earlier attempt (which finalizes normally).
        The match is anchored on the full ``of attempt <id>.`` clause the
        hold writes, not a bare substring.
        """

        store = self._control_store
        if store is None or not attempt_id:
            return False
        try:
            snapshot = store.handoff_snapshot()
        except (ControlStoreError, RuntimeError, OSError):
            return False
        if snapshot is None or snapshot.get("acknowledgement") != "not fulfilled":
            return False
        detail = snapshot.get("detail")
        return isinstance(detail, str) and f"of attempt {attempt_id}." in detail

    def _hold_failed_handoff(
        self,
        *,
        issue_number: int,
        attempt_id: str,
        outcome: AttemptOutcome,
        prepared: PreparedAttempt | None,
        publication_started_at: datetime | None = None,
        check_allowance: bool = True,
    ) -> bool:
        """Fail the owned handoff when it cannot complete; report the hold.

        Returns True when this attempt owns a handoff that is terminally ``not
        fulfilled``: the caller must skip the issue release, workspace cleanup,
        and ledger finalization so the branch, checkpoint, and working tree
        stay retained for operator recovery. An accepted handoff that cannot
        complete (wrong outcome, unbegun, invalid note, or blown deadline) is
        marked ``not fulfilled`` here with its reason and the retaining
        attempt ID. Anything else — no handoff, a completable handoff, a
        completed one, or a failure owned by another attempt — returns False.
        ``check_allowance`` is False only for the pre-commit recheck after a
        confirmed publication (see ``_handoff_completion_cause``).
        """

        store = self._control_store
        if store is None:
            return False
        try:
            snapshot = store.handoff_snapshot()
        except (ControlStoreError, RuntimeError, OSError) as error:
            self._event_log(
                "operator_handoff_resolution_unavailable",
                f"{_exception_detail(error)} The pending handoff keeps waiting;",
                level="ERROR",
                issue_number=issue_number,
            )
            return False
        if snapshot is None:
            return False
        acknowledgement = snapshot.get("acknowledgement")
        if acknowledgement == "completed":
            return False
        request_id = snapshot.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return False
        if acknowledgement != "accepted":
            if acknowledgement != "not fulfilled":
                return False
            return self._handoff_owned_by_attempt(attempt_id)
        cause = self._handoff_completion_cause(
            snapshot,
            issue_number=issue_number,
            outcome=outcome,
            prepared=prepared,
            publication_started_at=publication_started_at,
            check_allowance=check_allowance,
        )
        if cause is None:
            return False
        store.fail_handoff(
            request_id,
            f"Operator handoff {request_id} is not fulfilled: {cause}; the"
            " branch, checkpoint, and working tree are retained for recovery"
            f" of attempt {attempt_id}.",
        )
        self._event_log(
            "operator_handoff_not_fulfilled",
            f"request={request_id}; {cause}",
            level="WARNING",
            issue_number=issue_number,
        )
        return True

    def _resolve_pending_handoff(
        self,
        *,
        issue_number: int,
        outcome: AttemptOutcome,
        prepared: PreparedAttempt | None,
        attempt_id: str | None = None,
        publication_started_at: datetime | None = None,
        check_allowance: bool = True,
    ) -> bool | None:
        """Complete or fail the pending operator handoff after finalization.

        Returns True when the handoff completed, False when it is (or already
        was) terminally ``not fulfilled``, and None when no handoff owns this
        attempt. A handoff completes only when the attempt outcome is a
        published handoff, the model began its skill, the committed note is a
        valid ``operator_request`` note, publication confirmed within 120
        seconds after the valid local handoff, and the whole command finished
        within 360 seconds of its original acceptance. Anything else is marked
        ``not fulfilled`` with its reason; the caller then retains the branch,
        checkpoint, and working tree for operator recovery instead of
        finalizing. Already-terminal commands are returned untouched, so crash
        replays and recovery retries resolve at most once. Callers after a
        confirmed publication pass ``check_allowance=False`` (see
        ``_handoff_completion_cause``).
        """

        store = self._control_store
        if store is None:
            return None
        try:
            snapshot = store.handoff_snapshot()
        except (ControlStoreError, RuntimeError, OSError) as error:
            self._event_log(
                "operator_handoff_resolution_unavailable",
                f"{_exception_detail(error)} The pending handoff keeps waiting;",
                level="ERROR",
                issue_number=issue_number,
            )
            return None
        if snapshot is None:
            return None
        acknowledgement = snapshot.get("acknowledgement")
        if acknowledgement == "completed":
            return True
        if acknowledgement != "accepted":
            return False if acknowledgement == "not fulfilled" else None
        request_id = snapshot.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return None
        cause = self._handoff_completion_cause(
            snapshot,
            issue_number=issue_number,
            outcome=outcome,
            prepared=prepared,
            publication_started_at=publication_started_at,
            check_allowance=check_allowance,
        )
        if cause is None:
            store.complete_handoff(
                request_id,
                f"operator handoff fulfilled for issue #{issue_number}: a valid"
                " `operator_request` handoff note was published, the issue was"
                " released, and cleanup is durable",
            )
            self._event_log(
                "operator_handoff_completed",
                f"request={request_id}; issue #{issue_number} handed off",
                level="INFO",
                issue_number=issue_number,
            )
            return True
        hold_suffix = f" of attempt {attempt_id}" if attempt_id else ""
        store.fail_handoff(
            request_id,
            f"Operator handoff {request_id} is not fulfilled: {cause}; the"
            " branch, checkpoint, and working tree are retained for recovery"
            f"{hold_suffix}.",
        )
        self._event_log(
            "operator_handoff_not_fulfilled",
            f"request={request_id}; {cause}",
            level="WARNING",
            issue_number=issue_number,
        )
        return False

    def _check_operator_note(
        self, issue_number: int, prepared: PreparedAttempt | None
    ) -> tuple[bool, str]:
        """Whether the branch carries a valid ``operator_request`` note.

        Without the prepared working tree (a crash replay after every other
        boundary went durable) prior validation is trusted: only a begun
        handoff with a handoff outcome reaches here.
        """

        if prepared is None:
            return True, "not rechecked after a crash replay"
        try:
            commits = self._workspace.commits_added(prepared)
            working_directory = self._workspace.working_directory
            attempt_id = _attempt_id(self._attempt_state.read())
        except Exception as error:
            return False, f"the branch state could not be read ({_exception_detail(error)})"
        try:
            validation = _validate_handoff_note(
                self._workspace.handoff_note_commit(
                    prepared, issue_number, attempt_id=attempt_id
                ),
                commits,
                issue_number,
                working_directory,
                prepared.base_revision,
                note_commit_revisions=_this_attempt_note_commit_revisions(
                    self._workspace, prepared, issue_number, attempt_id
                ),
            )
        except Exception as error:
            return False, f"the handoff note could not be validated ({_exception_detail(error)})"
        _emit_handoff_note_events(self._event_log, issue_number, validation)
        if not validation.valid:
            return False, validation.reason
        try:
            text = _read_handoff_note(working_directory, issue_number)
        except Exception as error:
            return False, f"the handoff note could not be read ({_exception_detail(error)})"
        fields = _parse_handoff_note_fields(text if isinstance(text, str) else "")
        if fields.get("reason") != _OPERATOR_HANDOFF_REASON:
            return (
                False,
                f"the note cites reason {fields.get('reason')!r} instead of"
                f" `{_OPERATOR_HANDOFF_REASON}`",
            )
        return True, "valid"

    def _fail_orphan_handoff_locked(self) -> None:
        """Fail a pending handoff whose active attempt is already gone.

        Caller holds the control lock. With no checkpoint left, no model
        execution can still observe the request, so it is marked ``not
        fulfilled`` and intake stops instead of waiting forever.
        """

        store = self._control_store
        if store is None:
            return
        snapshot = store.handoff_snapshot()
        if snapshot is None or snapshot.get("acknowledgement") != "accepted":
            return
        request_id = snapshot.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return
        store.fail_handoff(
            request_id,
            f"Operator handoff {request_id} is not fulfilled: the active"
            " attempt ended without finalizing through the agent, so no model"
            " execution can still observe the request.",
        )
        self._event_log(
            "operator_handoff_not_fulfilled",
            f"request={request_id}; the active attempt is gone",
            level="WARNING",
        )

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

    def _release(
        self,
        claim: Claim,
        outcome: AttemptOutcome,
        checkpoint: AttemptCheckpoint | None = None,
    ) -> None:
        """Apply the label/assignee release sequence required by the outcome.

        A handoff must add ``round-finished`` for human-controlled requeue
        instead of the plain ready-for-agent/assignee release every other
        outcome uses. A conflict attempt releases ``merge-conflict`` instead:
        every conflict outcome removes the conflict signal and the assignee,
        and never adds ``ready-for-agent`` or ``round-finished``. The
        checkpoint kind decides when one is passed (it records the
        claim-time routing, so a label change mid-attempt cannot reroute
        the release); otherwise the live labels decide.
        """

        if outcome is AttemptOutcome.HANDOFF:
            self._tracker.release_handoff(claim.issue.number, claim.assignment.assignee_id)
            return
        if checkpoint is not None:
            is_conflict = _is_conflict_attempt(checkpoint)
        else:
            is_conflict = _is_conflict_claim(claim.issue.labels)
        if is_conflict:
            self._tracker.release_attempt(
                claim.issue.number, MERGE_CONFLICT, claim.assignment.assignee_id
            )
        else:
            self._tracker.release_attempt(
                claim.issue.number, "ready-for-agent", claim.assignment.assignee_id
            )

    def _hold_dirty_workspace(self, issue_number: int, error: DirtyWorkspaceError) -> None:
        """Hold intake with a dirty tree preserved on the attempt branch.

        Cleanup refused to switch to the base branch while the tree is
        dirty, so the workspace stays on the attempt branch: emit the hold
        event and return to the poll loop instead of finalizing or claiming
        further issues. The caller retains the checkpoint (its cleanup-ok
        flag stays false), so the next poll holds on the retained attempt.
        The branch, checkpoint, and entire working tree are preserved for
        inspection.
        """

        try:
            attempt_id = _attempt_id(self._attempt_state.read())
            self._remember_workspace_dirty(attempt_id, True)
        except Exception:
            pass
        self._emit_workspace_hold(issue_number, error)

    def _emit_workspace_hold(self, issue_number: int, error: DirtyWorkspaceError) -> None:
        """Record that intake is held on a dirty attempt workspace.

        Names the offending paths, as the intake dirty-tree hold does:
        the cleanup error itself carries no paths, so they are read
        best-effort from the tree.
        """

        try:
            attempt_id = _attempt_id(self._attempt_state.read())
            self._remember_workspace_dirty(attempt_id, True)
        except Exception:
            pass
        detail = (
            f"{_exception_detail(error)} The branch, checkpoint, and working tree "
            "are preserved for inspection; issue intake is held until the "
            "workspace is repaired by an operator"
        )
        detail += self._dirty_paths_suffix() or "."
        self._event_log(
            "workspace_hold",
            detail,
            level="WARNING",
            issue_number=issue_number,
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
        issue_title = getattr(checkpoint, "issue_title", None)
        if not isinstance(issue_title, str) or not issue_title:
            issue_title = None
        completed_at = self._write_outcome(
            archive,
            outcome,
            attempt_id,
            issue_number=issue_number,
            issue_title=issue_title,
        )
        self._backfill_archived_pull_request(archive, attempt_id)
        if self._completion_store is not None:
            self._completion_store.record(
                attempt_id=attempt_id,
                issue_number=issue_number,
                branch=branch,
                outcome=outcome,
            )
        self._record_control_finalization(attempt_id, outcome)
        self._attempt_state.delete()
        self._telemetry_finish(
            self._last_finished_summary(
                attempt_id,
                issue_number,
                issue_title,
                outcome,
                completed_at,
                archive,
            )
        )

    def _backfill_archived_pull_request(
        self, archive: AttemptArchive | None, attempt_id: str
    ) -> None:
        """Ensure the archive record carries the remembered pull request.

        Publication already archives the pull request; this safety net covers
        flows whose publication result bypassed the archive write, without
        ever clobbering a value publication recorded. Never raises.
        """

        if archive is None:
            return
        try:
            remembered = self._observed_publication.get(attempt_id, {}).get(
                "pull_request"
            )
            if remembered is None:
                return
            if archive.read_attempt().get("pull_request") is not None:
                return
            archive.write_attempt({"pull_request": remembered})
        except Exception:
            return

    def _last_finished_summary(
        self,
        attempt_id: str,
        issue_number: int,
        issue_title: str | None,
        outcome: AttemptOutcome,
        completed_at: str,
        archive: AttemptArchive | None,
    ) -> dict:
        """Build the telemetry ``last_finished`` summary for a finalized attempt.

        Token budget and turns come from the attempt's archive record; the
        pull request comes from the archive (written at publication) falling
        back to the remembered publication result. Never raises.
        """

        record: dict = {}
        if archive is not None:
            try:
                record = archive.read_attempt()
            except Exception:
                record = {}
        if not isinstance(record, dict):
            record = {}
        title = issue_title
        if title is None:
            archived_title = record.get("issue_title")
            title = (
                archived_title
                if isinstance(archived_title, str) and archived_title
                else None
            )
        token_budget = record.get("token_budget")
        budget_used = None
        turns = None
        if isinstance(token_budget, dict):
            budget_used = token_budget.get("actual_tokens")
            turns = token_budget.get("turns")
        pull_request = record.get("pull_request")
        if pull_request is None:
            pull_request = self._observed_publication.get(attempt_id, {}).get(
                "pull_request"
            )
        return {
            "attempt_id": attempt_id,
            "issue_number": issue_number,
            "issue_title": title,
            "outcome": outcome.value,
            "finished_at": completed_at,
            "budget_used": budget_used,
            "turns": turns,
            "pull_request": pull_request,
        }

    def _reconcile_startup(self) -> AttemptOutcome | None:
        """Finish one durable attempt without recreating its SDK execution context."""

        try:
            checkpoint = self._attempt_state.read()
        except AttemptStateError as error:
            # An unreadable checkpoint is operator-repairable (repair or
            # remove the file), not a fatal error: hold intake without
            # exiting, keeping the process polling and the control socket
            # serving until the state reads cleanly again.
            detail = _checkpoint_hold_detail(error)
            self._event_log(
                "attempt_checkpoint_unusable",
                detail,
                level="ERROR",
            )
            # Refresh unconditionally (reconcile runs once per process):
            # the served reason always matches the logged event, while
            # the once-per-episode gating lives in the heartbeat below.
            self._startup_hold_kind = "checkpoint"
            self._startup_hold_detail = detail
            self._note_startup_hold_locked(detail)
            self._active_issue_number = None
            return None
        if checkpoint is None:
            return None
        try:
            finalized = (
                self._completion_store is not None
                and self._completion_store.is_finalized(checkpoint.started_at)
            )
        except CompletionStoreError as error:
            # An unreadable completion ledger is operator-repairable like
            # an unreadable checkpoint: hold without exiting rather than
            # crash-looping under process restart.
            detail = (
                f"{_exception_detail(error)} The completion ledger is preserved for "
                "inspection; issue intake is held until it is repaired or removed "
                "by an operator."
            )
            self._event_log(
                "attempt_completion_unusable",
                detail,
                level="ERROR",
            )
            # Refresh unconditionally, as for the checkpoint hold above.
            self._startup_hold_kind = "completion"
            self._startup_hold_detail = detail
            self._note_startup_hold_locked(
                detail, issue_number=checkpoint.issue_number
            )
            self._active_issue_number = None
            return None
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
                    # The previous process recorded every boundary but may
                    # have crashed before resolving the handoff it waited
                    # for; without a prepared tree prior validation stands.
                    handoff_failed = self._resolve_pending_handoff(
                        issue_number=checkpoint.issue_number,
                        attempt_id=checkpoint.started_at,
                        outcome=terminal.outcome,
                        prepared=None,
                    )
                    if handoff_failed is False and self._handoff_owned_by_attempt(
                        checkpoint.started_at
                    ):
                        # The handoff failed for this attempt: keep the
                        # leftover checkpoint for operator recovery instead
                        # of deleting the last handle on the retained work.
                        self._event_log(
                            "attempt_finalization_held",
                            "A failed operator handoff owns this attempt; the branch, "
                            "checkpoint, and working tree are preserved for recovery.",
                            level="WARNING",
                            issue_number=checkpoint.issue_number,
                        )
                        self._active_issue_number = None
                        return None
            self._attempt_state.delete()
            self._telemetry_finish()
            self._active_issue_number = None
            return None
        claim = self._tracker.recover_claim(checkpoint.issue_number)
        if claim is None:
            # Recorded decision for #219: a retained checkpoint whose issue
            # is gone entirely is a genuine fatal error, not a workspace
            # hold — there is no issue to hold intake for and no operator
            # workspace repair that could progress it.
            raise RuntimeError("Interrupted attempt issue is unavailable for recovery")
        self._active_issue_number = claim.issue.number
        # The attempt survived a restart: re-seed the claim-time view from
        # the checkpoint (no tracker lookup) so telemetry covers the attempt
        # from its original claim until its eventual finalisation.
        self._telemetry_begin_for_checkpoint(checkpoint)
        with self._control_lock:
            self._attempt_processing = True
            if self._control_store is not None:
                # A crash between the final claim and its ``stopping``
                # transition must not lose the boundary: the recovered final
                # counted attempt enters ``stopping`` here instead.
                self._control_store.enter_stopping_for_final_claim()
        archive = self._archive_for(claim.issue.number, checkpoint.started_at)

        prepared: PreparedAttempt | None = None
        profile: RepositoryProfile | None = None
        outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
        comment_posted = True
        retain_branch = False
        has_commits = False
        publication_started_at: datetime | None = None
        try:
            self._workspace.prepare_for_profile_read()
            profile = self._profile_loader(self._workspace.working_directory)
            prepared = self._workspace.prepare_attempt(
                base_branch=profile.base_branch, issue_number=claim.issue.number
            )
            if checkpoint.phase is AttemptPhase.MODEL_RUNNING:
                commits = self._workspace.commits_added(prepared)
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
                recovered_commits = self._workspace.commits_added(prepared)
                has_commits = bool(recovered_commits)
                recovered_note_commit = self._workspace.handoff_note_commit(
                    prepared, claim.issue.number, attempt_id=checkpoint.started_at
                )
                if recovered_note_commit is not None:
                    validation = _validate_handoff_note(
                        recovered_note_commit,
                        recovered_commits,
                        claim.issue.number,
                        self._workspace.working_directory,
                        prepared.base_revision,
                        note_commit_revisions=_this_attempt_note_commit_revisions(
                            self._workspace,
                            prepared,
                            claim.issue.number,
                            checkpoint.started_at,
                        ),
                    )
                    _emit_handoff_note_events(
                        self._event_log, claim.issue.number, validation
                    )
                    if validation.valid:
                        decision = CompletionDecision(
                            AttemptOutcome.HANDOFF, False, PublicationPath.PARTIAL,
                            ("Recovered a previously committed handoff note for publication.",),
                        )
                    else:
                        # An invalid recovered note is ordinary incomplete
                        # work, not an infrastructure failure: the preserved
                        # commits still publish on the partial path with the
                        # validation reason attached.
                        decision = CompletionDecision(
                            AttemptOutcome.INCOMPLETE, False, PublicationPath.PARTIAL,
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
                note_commit = self._workspace.handoff_note_commit(
                    prepared, claim.issue.number, attempt_id=checkpoint.started_at
                )
                if note_commit is not None:
                    note_commit_sha = note_commit.revision
            publication_started_at = self._clock()
            if _is_conflict_attempt(checkpoint):
                # A retained conflict attempt recovers as a conflict error
                # (a failure toward the budget, possibly escalating), never
                # as an ordinary result: an ordinary marker would stop the
                # streak and grant a fresh budget the attempt did not earn.
                published = self._publish_conflict_bounded(
                    self._conflict_error_request(
                        claim,
                        checkpoint,
                        prepared,
                        profile,
                        "Attempt was interrupted by a restart before a result "
                        "could be verified; the remote branch is left untouched.",
                        requeue_source=REQUEUE_SOURCE_CONFLICT,
                    ),
                    claim,
                )
                outcome = published.outcome
                comment_posted = getattr(published, "comment_posted", True)
                retain_branch = (
                    getattr(published, "branch_url", None) is None
                    and outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
                )
            else:
                published = self._publish_bounded(
                    PublicationRequest(
                        issue_number=claim.issue.number,
                        issue_title=claim.issue.title,
                        branch=prepared.branch,
                        started_at=checkpoint.started_at,
                        decision=decision,
                        check_command="not run",
                        check_exit_code=None,
                        check_not_run_reason="not rerun during startup recovery",
                        review_cycles=0,
                        review_findings="not rerun during startup recovery",
                        details=self._handoff_publication_details(
                            claim, checkpoint, details, decision.outcome
                        ),
                        base_branch=profile.base_branch,
                        note_commit_sha=note_commit_sha,
                        auto_merge_opt_in=_claim_time_opt_in(claim, checkpoint),
                        requeue_source=_claim_time_requeue_source(claim, checkpoint),
                        handoff_continuation_count=(
                            self._handoff_continuation_used(claim, checkpoint)
                            if decision.outcome is AttemptOutcome.HANDOFF
                            else None
                        ),
                        handoff_notify=(
                            decision.outcome is AttemptOutcome.HANDOFF
                            and self._handoff_needs_notification(claim, checkpoint)
                        ),
                    )
                )
                outcome = published.outcome
                comment_posted = _publication_confirmed(published)
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
            # this attempt never produced. The attempt stays held (no exit):
            # the finally boundary below logs ``attempt_finalization_held``
            # and retains the checkpoint, so the next poll holds on the
            # retained attempt instead of claiming new work.
            self._remember_workspace_dirty(checkpoint.started_at, True)
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
        except GitWorkspaceRecoveryError as error:
            self._event_log(
                "git_workspace_unrecoverable",
                _exception_detail(error),
                level="ERROR",
                issue_number=claim.issue.number,
            )
            comment_posted = False
            prepared = None
        except SystemExit:
            raise
        except RebaseConflictError as error:
            publication_started_at = self._clock()
            if _is_conflict_attempt(checkpoint):
                published = self._publish_conflict_bounded(
                    self._conflict_rebase_request(
                        claim,
                        checkpoint,
                        prepared,
                        profile,
                        None,
                        error.conflicted_files,
                        requeue_source=REQUEUE_SOURCE_CONFLICT,
                    ),
                    claim,
                )
            else:
                published = self._publish_rebase_conflict(
                    claim, checkpoint.started_at, prepared, profile, error,
                    timeout=self._handoff_publish_budget(),
                    auto_merge_opt_in=_claim_time_opt_in(claim, checkpoint),
                    requeue_source=_claim_time_requeue_source(claim, checkpoint),
                )
            outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
            comment_posted = getattr(published, "comment_posted", True)
        except Exception as error:
            self._event_log(
                "attempt_exception", _exception_detail(error), level="ERROR", issue_number=claim.issue.number
            )
            publication_started_at = self._clock()
            if _is_conflict_attempt(checkpoint):
                published = self._publish_conflict_bounded(
                    self._conflict_error_request(
                        claim,
                        checkpoint,
                        prepared,
                        profile,
                        _exception_detail(error),
                        requeue_source=REQUEUE_SOURCE_CONFLICT,
                    ),
                    claim,
                )
            else:
                published = self._publish_terminal(
                    claim, checkpoint.started_at, prepared, profile, AttemptOutcome.INFRASTRUCTURE_ERROR,
                    timeout=self._handoff_publish_budget(),
                    auto_merge_opt_in=_claim_time_opt_in(claim, checkpoint),
                    requeue_source=_claim_time_requeue_source(claim, checkpoint),
                )
            outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
            comment_posted = getattr(published, "comment_posted", True)
        except BaseException as error:
            self._event_log(
                "attempt_exception", _exception_detail(error), level="ERROR", issue_number=claim.issue.number
            )
            try:
                publication_started_at = self._clock()
                if _is_conflict_attempt(checkpoint):
                    published = self._publish_conflict_bounded(
                        self._conflict_error_request(
                            claim,
                            checkpoint,
                            prepared,
                            profile,
                            _exception_detail(error),
                            requeue_source=REQUEUE_SOURCE_CONFLICT,
                        ),
                        claim,
                    )
                else:
                    published = self._publish_terminal(
                        claim, checkpoint.started_at, prepared, profile, AttemptOutcome.INFRASTRUCTURE_ERROR,
                        timeout=self._handoff_publish_budget(),
                        auto_merge_opt_in=_claim_time_opt_in(claim, checkpoint),
                        requeue_source=_claim_time_requeue_source(claim, checkpoint),
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
            # A failed operator handoff holds the same boundary: the result
            # comment is already published truthfully, but the issue stays
            # assigned and the checkpoint, branch, and working tree stay
            # retained for `recovery retry` or `recovery release` instead of
            # finalizing, across restarts with the original deadlines.
            hold = comment_posted and self._hold_failed_handoff(
                issue_number=claim.issue.number,
                attempt_id=checkpoint.started_at,
                outcome=outcome,
                prepared=prepared,
                publication_started_at=publication_started_at,
            )
            released = False
            try:
                if comment_posted and not hold:
                    self._release(claim, outcome, checkpoint)
                    released = True
            finally:
                cleanup_ok = prepared is None
                try:
                    if prepared is not None and not hold:
                        self._workspace.cleanup(
                            base_branch=profile.base_branch if profile is not None else "main",
                            prepared=prepared,
                            retain_branch=retain_branch or not comment_posted,
                        )
                        cleanup_ok = True
                        self._observe_and_remember_workspace_dirty(
                            checkpoint.started_at
                        )
                except DirtyWorkspaceError as error:
                    # Cleanup refused to leave the attempt branch while the
                    # tree is dirty: hold intake with everything preserved
                    # for operator inspection instead of carrying uncommitted
                    # work onto the base or claiming further issues.
                    self._hold_dirty_workspace(claim.issue.number, error)
                finally:
                    if comment_posted and released and cleanup_ok and not hold:
                        # Publication already confirmed (comment posted) before
                        # release began: only the whole-command budget still
                        # applies, not the concluded publication window.
                        hold = self._hold_failed_handoff(
                            issue_number=claim.issue.number,
                            attempt_id=checkpoint.started_at,
                            outcome=outcome,
                            prepared=prepared,
                            publication_started_at=publication_started_at,
                            check_allowance=False,
                        )
                    if comment_posted and released and cleanup_ok and not hold:
                        self._commit_finalization(
                            checkpoint, claim.issue.number, prepared.branch
                            if prepared is not None else checkpoint.branch,
                            outcome, archive,
                        )
                        self._resolve_pending_handoff(
                            issue_number=claim.issue.number,
                            attempt_id=checkpoint.started_at,
                            outcome=outcome,
                            prepared=prepared,
                            publication_started_at=publication_started_at,
                            check_allowance=False,
                        )
                    else:
                        self._event_log(
                            "attempt_finalization_held",
                            "Publication, release, or cleanup is unconfirmed, or a failed"
                            " operator handoff owns this attempt; the branch, "
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

    # -- conflict attempts ----------------------------------------------------

    def _execute_conflict_attempt(
        self,
        claim: Claim,
        checkpoint: AttemptCheckpoint,
    ) -> tuple[ConflictPublicationRequest, PreparedAttempt | None, RepositoryProfile | None]:
        """Run the conflict fast path, with one model session on real conflicts.

        The entry check re-reads the open pull request behind the published
        branch before any git work: a missing, draft, or no-longer-conflicting
        pull request ends the attempt as ``skipped``, as does a pull request
        targeting another base branch once the profile is loaded. A clean
        rebase runs setup plus one profile check (no baseline, no model
        call); only that green path pushes directly. A real rebase conflict,
        or a red check after a clean rebase (a semantic conflict), runs at
        most one narrow model resolution session under the ordinary model
        limits; the session brings the branch up to date and does no new
        work. Returns the publication request together with the prepared
        attempt (``None`` when skipped before any git work ran) and the
        profile (``None`` when it never loaded).
        """

        issue_number = claim.issue.number
        branch = _conflict_attempt_branch(issue_number)
        pull_request = self._publisher.find_pull_request(branch)
        skip = _conflict_entry_skip_reason(pull_request, branch=branch)
        if skip is not None:
            reason, details = skip
            self._event_log(
                "conflict_attempt_skipped",
                f"{reason}: {details}",
                level="INFO",
                issue_number=issue_number,
            )
            return (
                self._build_conflict_request(
                    claim,
                    checkpoint,
                    branch,
                    "main",
                    pull_request,
                    CONFLICT_OUTCOME_SKIPPED,
                    reason,
                    details,
                    push_branch=False,
                ),
                None,
                None,
            )
        self._workspace.prepare_for_profile_read()
        self._event_log(
            "workspace_prepared_for_profile_read",
            "base_branch=main",
            level="INFO",
            issue_number=issue_number,
        )
        profile = self._profile_loader(self._workspace.working_directory)
        self._event_log(
            "profile_loaded",
            f"base_branch={profile.base_branch}",
            level="INFO",
            issue_number=issue_number,
        )
        base_skip = _conflict_base_mismatch_reason(pull_request, profile.base_branch)
        if base_skip is not None:
            reason, details = base_skip
            self._event_log(
                "conflict_attempt_skipped",
                f"{reason}: {details}",
                level="INFO",
                issue_number=issue_number,
            )
            return (
                self._build_conflict_request(
                    claim,
                    checkpoint,
                    branch,
                    profile.base_branch,
                    pull_request,
                    CONFLICT_OUTCOME_SKIPPED,
                    reason,
                    details,
                    push_branch=False,
                ),
                None,
                profile,
            )
        self._attempt_state.transition(AttemptPhase.SETUP)
        self._event_log(
            "attempt_phase_transitioned",
            f"phase={AttemptPhase.SETUP.value}",
            level="INFO",
            issue_number=issue_number,
        )
        prepared = self._workspace.prepare_attempt(
            base_branch=profile.base_branch, issue_number=issue_number
        )
        self._event_log(
            "workspace_prepared",
            f"branch={prepared.branch}; base_branch={profile.base_branch}",
            level="INFO",
            issue_number=issue_number,
        )
        state = self._workspace.conflict_state()
        conflicted = tuple(state.conflicted_files) or tuple(prepared.rebase_conflicts or ())
        if state.in_progress or prepared.rebase_conflicts:
            # A real rebase conflict: leave the rebase in place for the
            # single model resolution session instead of aborting up front.
            self._event_log(
                "conflict_rebase_conflict",
                f"branch={prepared.branch}; base_branch={profile.base_branch}"
                + (f"; files={', '.join(conflicted)}" if conflicted else ""),
                level="WARNING",
                issue_number=issue_number,
            )
            return (
                self._resolve_conflict_with_model(
                    claim,
                    checkpoint,
                    prepared,
                    profile,
                    pull_request,
                    conflicted,
                    check_output=None,
                ),
                prepared,
                profile,
            )
        runner = self._command_runner
        if runner is None:
            detail = (
                "No command runner is configured for conflict attempts; the rebased "
                "branch was not verified, so the remote branch is left untouched."
            )
            self._event_log(
                "conflict_attempt_error", detail, level="ERROR", issue_number=issue_number
            )
            return (
                self._conflict_error_request(claim, checkpoint, prepared, profile, detail),
                prepared,
                profile,
            )
        working_directory = self._workspace.working_directory
        setup_result = runner.run(
            tuple(profile.setup),
            timeout=profile.setup_timeout,
            cwd=working_directory,
            environment=profile.env,
        )
        if not setup_result.succeeded:
            detail = _conflict_setup_failed_details(setup_result)
            self._event_log(
                "conflict_setup_failed", detail, level="WARNING", issue_number=issue_number
            )
            return (
                self._build_conflict_request(
                    claim,
                    checkpoint,
                    prepared.branch,
                    profile.base_branch,
                    pull_request,
                    CONFLICT_OUTCOME_SETUP_FAILED,
                    "",
                    detail,
                    push_branch=False,
                ),
                prepared,
                profile,
            )
        self._attempt_state.transition(AttemptPhase.MODEL_RUNNING)
        self._event_log(
            "attempt_phase_transitioned",
            f"phase={AttemptPhase.MODEL_RUNNING.value}",
            level="INFO",
            issue_number=issue_number,
        )
        check_result = runner.run(
            tuple(profile.check),
            timeout=profile.timeout,
            cwd=working_directory,
            environment=profile.env,
        )
        check_command = " && ".join(profile.check)
        if not check_result.succeeded:
            # A red check after a clean rebase is a semantic conflict: the
            # single model resolution session runs with the check output in
            # its prompt instead of failing here.
            detail = _conflict_check_failed_details(check_result)
            self._event_log(
                "conflict_check_failed", detail, level="WARNING", issue_number=issue_number
            )
            return (
                self._resolve_conflict_with_model(
                    claim,
                    checkpoint,
                    prepared,
                    profile,
                    pull_request,
                    (),
                    check_output=_conflict_format_check_output(check_result),
                ),
                prepared,
                profile,
            )
        self._attempt_state.transition(AttemptPhase.PUSHING)
        self._event_log(
            "attempt_phase_transitioned",
            f"phase={AttemptPhase.PUSHING.value}",
            level="INFO",
            issue_number=issue_number,
        )
        return (
            self._build_conflict_request(
                claim,
                checkpoint,
                prepared.branch,
                profile.base_branch,
                pull_request,
                CONFLICT_OUTCOME_SUCCESS,
                "",
                _conflict_success_details(prepared.branch, profile.base_branch),
                push_branch=True,
                check_command=check_command,
                check_exit_code=check_result.exit_code,
                check_steps=tuple(profile.check),
            ),
            prepared,
            profile,
        )

    def _resolve_conflict_with_model(
        self,
        claim: Claim,
        checkpoint: AttemptCheckpoint,
        prepared: PreparedAttempt,
        profile: RepositoryProfile,
        pull_request: PullRequest | None,
        conflicted: tuple[str, ...],
        *,
        check_output: str | None,
    ) -> ConflictPublicationRequest:
        """Run the single narrow model session resolving a conflict attempt.

        Runs for a real rebase conflict (``check_output`` is ``None``) or for
        a red profile check after a clean rebase (a semantic conflict, with
        the check output in the prompt). Exactly one ``execute()`` call runs
        per conflict attempt, under the ordinary model limits. The prompt
        carries the issue body and trusted comments marked context-only plus
        the conflicted files, the base changes, and the review-free gate;
        the full-implementation continuation and the ``code-review``
        instruction are absent, and the review gate is never wired.
        ``MODEL_LIMIT_REACHED``, an unresolved rebase, a changed commit
        count, and a red verification are ordinary failures: nothing is
        pushed and no handoff note is written.
        """

        issue_number = claim.issue.number
        branch = prepared.branch
        base_branch = profile.base_branch
        working_directory = self._workspace.working_directory
        try:
            before_subjects = tuple(
                commit.subject
                for commit in reversed(self._workspace.commits_added(prepared))
            )
        except Exception as error:
            return self._conflict_error_request(
                claim,
                checkpoint,
                prepared,
                profile,
                "Could not read the pull request's commits before model resolution: "
                f"{_exception_detail(error)}",
            )
        executor = self._conflict_model_executor
        if executor is None:
            return self._conflict_error_request(
                claim,
                checkpoint,
                prepared,
                profile,
                "No model executor is configured for conflict resolution; the remote "
                "branch is left untouched so the pull request stays `CONFLICTING`.",
            )
        try:
            current = self._attempt_state.read()
        except Exception:
            current = None
        if current is not None and current.phase is AttemptPhase.SETUP:
            # The rebase-conflict path arrives here straight from setup; the
            # red-check path already moved to MODEL_RUNNING for its check.
            self._attempt_state.transition(AttemptPhase.MODEL_RUNNING)
            self._event_log(
                "attempt_phase_transitioned",
                f"phase={AttemptPhase.MODEL_RUNNING.value}",
                level="INFO",
                issue_number=issue_number,
            )
        comments = (
            self._conflict_issue_comments(claim.issue)
            if self._conflict_issue_comments is not None
            else ()
        )
        prompt = _conflict_resolution_prompt(
            issue_body=claim.issue.body,
            comments=comments,
            conflicted_files=conflicted,
            base_changes=_conflict_base_changes_text(
                working_directory, base_branch, conflicted
            ),
            check_output=check_output,
            gate_section=_gate_section(
                profile,
                self._conflict_prompt_redactions,
                include_review_instruction=False,
            ),
        )
        self._event_log(
            "conflict_model_started",
            f"branch={branch}; base_branch={base_branch}"
            + (f"; files={', '.join(conflicted)}" if conflicted else "")
            + ("; after_check=true" if check_output is not None else ""),
            level="INFO",
            issue_number=issue_number,
        )
        archive = self._archive_for(issue_number, checkpoint.started_at)
        attempt_id = _attempt_id(checkpoint)
        try:
            execution = asyncio.run(
                _execute_model(
                    executor,
                    issue_body=prompt,
                    working_directory=working_directory,
                    issue_number=issue_number,
                    archive=archive,
                    attempt_id=attempt_id,
                    branch=prepared.branch,
                    base_revision=prepared.base_revision,
                )
            )
        except Exception as error:
            detail = (
                "The conflict resolution session could not finish: "
                f"{_exception_detail(error)}"
            )
            self._event_log(
                "conflict_model_failed", detail, level="ERROR", issue_number=issue_number
            )
            self._abort_conflict_best_effort(prepared, issue_number)
            self._seal_conflict_best_effort(
                issue_number, attempt_id, ModelExecutionStatus.INFRASTRUCTURE_ERROR
            )
            return self._build_conflict_request(
                claim,
                checkpoint,
                branch,
                base_branch,
                pull_request,
                CONFLICT_OUTCOME_ERROR,
                "",
                _conflict_model_error_details(branch, base_branch, f"{detail} "),
                push_branch=False,
                conflicted_files=tuple(conflicted),
            )
        self._archive_conflict_model_result(archive, issue_number, execution)
        status = execution.status
        self._event_log(
            "conflict_model_finished",
            f"status={status}",
            level="INFO" if status is ModelExecutionStatus.SUCCEEDED else "WARNING",
            issue_number=issue_number,
        )
        if status is ModelExecutionStatus.MODEL_LIMIT_REACHED:
            self._abort_conflict_best_effort(prepared, issue_number)
            self._seal_conflict_best_effort(issue_number, attempt_id, status)
            return self._build_conflict_request(
                claim,
                checkpoint,
                branch,
                base_branch,
                pull_request,
                CONFLICT_OUTCOME_MODEL_LIMIT,
                "",
                _conflict_model_limit_details(branch, base_branch, conflicted),
                push_branch=False,
                conflicted_files=tuple(conflicted),
            )
        if status is not ModelExecutionStatus.SUCCEEDED:
            self._abort_conflict_best_effort(prepared, issue_number)
            self._seal_conflict_best_effort(issue_number, attempt_id, status)
            detail = (
                f"the model session ended with status `{status}` instead of "
                "resolving the conflict. "
            )
            return self._build_conflict_request(
                claim,
                checkpoint,
                branch,
                base_branch,
                pull_request,
                CONFLICT_OUTCOME_ERROR,
                "",
                _conflict_model_error_details(branch, base_branch, detail),
                push_branch=False,
                conflicted_files=tuple(conflicted),
            )
        post_state = self._workspace.conflict_state()
        if post_state.resolution_problems:
            remaining = tuple(post_state.conflicted_files) or tuple(conflicted)
            self._event_log(
                "conflict_model_unresolved",
                "; ".join(post_state.resolution_problems),
                level="WARNING",
                issue_number=issue_number,
            )
            self._abort_conflict_best_effort(prepared, issue_number)
            self._seal_conflict_best_effort(issue_number, attempt_id, status)
            return self._build_conflict_request(
                claim,
                checkpoint,
                branch,
                base_branch,
                pull_request,
                CONFLICT_OUTCOME_REBASE_CONFLICT,
                "",
                _conflict_model_unresolved_details(
                    branch,
                    base_branch,
                    remaining,
                    after_check=check_output is not None,
                ),
                push_branch=False,
                conflicted_files=tuple(remaining),
            )
        try:
            self._workspace.seal(
                issue_number=issue_number, attempt_id=attempt_id, status=status
            )
        except Exception as error:
            return self._conflict_error_request(
                claim,
                checkpoint,
                prepared,
                profile,
                "Failed to seal conflict resolution work: "
                f"{_exception_detail(error)}",
            )
        try:
            after_commits = tuple(self._workspace.commits_added(prepared))
        except Exception as error:
            return self._conflict_error_request(
                claim,
                checkpoint,
                prepared,
                profile,
                "Could not read the pull request's commits after model resolution: "
                f"{_exception_detail(error)}",
            )
        after_subjects = tuple(commit.subject for commit in reversed(after_commits))
        fix_count = _conflict_fix_commit_count(before_subjects, after_subjects)
        if fix_count is None:
            return self._build_conflict_request(
                claim,
                checkpoint,
                branch,
                base_branch,
                pull_request,
                CONFLICT_OUTCOME_RESOLUTION_FAILED,
                "",
                _conflict_count_changed_details(
                    branch,
                    base_branch,
                    expected=len(before_subjects),
                    actual=len(after_subjects),
                ),
                push_branch=False,
                conflicted_files=tuple(conflicted),
            )
        runner = self._command_runner
        if runner is None:
            return self._conflict_error_request(
                claim,
                checkpoint,
                prepared,
                profile,
                "No command runner is configured for conflict attempts; the resolved "
                "branch was not verified, so the remote branch is left untouched.",
            )
        setup_result = runner.run(
            tuple(profile.setup),
            timeout=profile.setup_timeout,
            cwd=working_directory,
            environment=profile.env,
        )
        if not setup_result.succeeded:
            detail = _conflict_setup_failed_details(setup_result)
            self._event_log(
                "conflict_setup_failed", detail, level="WARNING", issue_number=issue_number
            )
            return self._build_conflict_request(
                claim,
                checkpoint,
                branch,
                base_branch,
                pull_request,
                CONFLICT_OUTCOME_SETUP_FAILED,
                "",
                detail,
                push_branch=False,
                conflicted_files=tuple(conflicted),
            )
        check_result = runner.run(
            tuple(profile.check),
            timeout=profile.timeout,
            cwd=working_directory,
            environment=profile.env,
        )
        check_command = " && ".join(profile.check)
        if not check_result.succeeded:
            detail = _conflict_check_failed_details(check_result)
            self._event_log(
                "conflict_check_failed", detail, level="WARNING", issue_number=issue_number
            )
            failed_step, failed_index = _check_failure_step(check_result)
            return self._build_conflict_request(
                claim,
                checkpoint,
                branch,
                base_branch,
                pull_request,
                CONFLICT_OUTCOME_CHECK_FAILED,
                "",
                detail,
                push_branch=False,
                check_command=check_command,
                check_exit_code=check_result.exit_code,
                check_steps=tuple(profile.check),
                check_failed_step=failed_step,
                check_failed_index=failed_index,
                conflicted_files=tuple(conflicted),
                failure_output=_conflict_failure_output_tail(check_result),
            )
        self._attempt_state.transition(AttemptPhase.PUSHING)
        self._event_log(
            "attempt_phase_transitioned",
            f"phase={AttemptPhase.PUSHING.value}",
            level="INFO",
            issue_number=issue_number,
        )
        # The commit count held, so the newest ``fix_count`` commits
        # (newest-first, seal commit included) are the fixes on top.
        fix_revisions = tuple(commit.revision for commit in after_commits[:fix_count])
        fix_files: list[str] = []
        for revision in fix_revisions:
            fix_files.extend(self._workspace.files_changed(revision))
        extra_files = _conflict_files_outside(fix_files, conflicted)
        self._event_log(
            "conflict_resolution_verified",
            f"branch={branch}; fix_commits="
            f"{', '.join(_conflict_short_revision(revision) for revision in fix_revisions) or 'none'}"
            + (f"; extra_files={', '.join(extra_files)}" if extra_files else ""),
            level="INFO",
            issue_number=issue_number,
        )
        return self._build_conflict_request(
            claim,
            checkpoint,
            branch,
            base_branch,
            pull_request,
            CONFLICT_OUTCOME_SUCCESS,
            "",
            _conflict_model_success_details(
                branch,
                base_branch,
                files=tuple(conflicted),
                extra_files=extra_files,
                fix_commits=fix_revisions,
            ),
            push_branch=True,
            check_command=check_command,
            check_exit_code=check_result.exit_code,
            check_steps=tuple(profile.check),
            conflicted_files=tuple(conflicted),
            extra_files=extra_files,
            fix_commits=fix_revisions,
        )

    def _abort_conflict_best_effort(
        self, prepared: PreparedAttempt, issue_number: int
    ) -> None:
        """Abort an in-progress rebase without changing the attempt's outcome."""

        try:
            if not self._workspace.conflict_state().in_progress:
                return
        except Exception as error:
            self._event_log(
                "conflict_abort_failed",
                f"reason={_exception_detail(error)}",
                level="WARNING",
                issue_number=issue_number,
            )
            return
        try:
            self._workspace.abort_unresolved_rebase(prepared)
        except Exception as error:
            self._event_log(
                "conflict_abort_failed",
                f"reason={_exception_detail(error)}",
                level="WARNING",
                issue_number=issue_number,
            )

    def _seal_conflict_best_effort(
        self,
        issue_number: int,
        attempt_id: str,
        status: ModelExecutionStatus,
    ) -> None:
        """Seal leftover resolution work without changing the attempt's outcome."""

        try:
            self._workspace.seal(
                issue_number=issue_number, attempt_id=attempt_id, status=status
            )
        except Exception as error:
            self._event_log(
                "seal_attempt_work_failed",
                _exception_detail(error),
                level="ERROR",
                issue_number=issue_number,
            )

    def _archive_conflict_model_result(
        self, archive: object | None, issue_number: int, execution: object
    ) -> None:
        """Archive one conflict resolution session's usage, best-effort."""

        if archive is None:
            return
        write = getattr(archive, "write_attempt", None)
        if callable(write):
            try:
                write(
                    {
                        "model_usage": getattr(execution, "model_usage", None),
                        "skill_events": [
                            event.__dict__ for event in getattr(execution, "skill_events", ())
                        ],
                        "model_stop_reason": getattr(execution, "stop_reason", None),
                        "token_budget": getattr(execution, "token_budget", None),
                    }
                )
            except Exception as error:
                self._event_log(
                    "conflict_archive_failed",
                    f"reason={_exception_detail(error)}",
                    level="WARNING",
                    issue_number=issue_number,
                )
        copy = getattr(archive, "copy_cli_transcripts", None)
        if callable(copy):
            try:
                copy(
                    getattr(self._conflict_model_executor, "cli_session_id", None),
                    issue_number=issue_number,
                    event_log=self._event_log,
                )
            except Exception as error:
                self._event_log(
                    "transcripts_copy_failed",
                    f"reason={_exception_detail(error)}",
                    level="WARNING",
                    issue_number=issue_number,
                )

    def _build_conflict_request(
        self,
        claim: Claim,
        checkpoint: AttemptCheckpoint,
        branch: str,
        base_branch: str,
        pull_request: PullRequest | None,
        conflict_outcome: str,
        reason: str,
        details: str,
        *,
        push_branch: bool,
        check_command: str = "not run",
        check_exit_code: int | None = None,
        check_steps: tuple[str, ...] = (),
        check_failed_step: str | None = None,
        check_failed_index: int | None = None,
        check_not_run_reason: str | None = None,
        conflicted_files: tuple[str, ...] = (),
        extra_files: tuple[str, ...] = (),
        fix_commits: tuple[str, ...] = (),
        failure_output: str = "",
        requeue_source: str | None = None,
    ) -> ConflictPublicationRequest:
        """Build one conflict publication request from its outcome and evidence.

        ``failure_output`` is the tail of the last check output quoted in
        the escalation's collapsed section when this attempt is the one
        whose failure reaches the budget; failures without command output
        fall back to the reason and details at render time. ``requeue_source``
        records the requeue source in the result comment (``None`` renders no
        marker); only startup recovery of a retained conflict attempt sets it.
        """

        return ConflictPublicationRequest(
            issue_number=claim.issue.number,
            issue_title=claim.issue.title,
            branch=branch,
            started_at=checkpoint.started_at,
            base_branch=base_branch,
            outcome=_conflict_lifecycle_outcome(conflict_outcome),
            conflict_outcome=conflict_outcome,
            reason=reason,
            details=details,
            push_branch=push_branch,
            pull_request=pull_request,
            check_command=check_command,
            check_exit_code=check_exit_code,
            check_steps=tuple(check_steps),
            check_failed_step=check_failed_step,
            check_failed_index=check_failed_index,
            check_not_run_reason=check_not_run_reason,
            conflicted_files=tuple(conflicted_files),
            auto_merge_opt_in=_claim_time_opt_in(claim, checkpoint),
            extra_files=tuple(extra_files),
            fix_commits=tuple(fix_commits),
            failure_output=failure_output,
            requeue_source=requeue_source,
        )

    def _conflict_error_request(
        self,
        claim: Claim,
        checkpoint: AttemptCheckpoint,
        prepared: PreparedAttempt | None,
        profile: RepositoryProfile | None,
        detail: str,
        *,
        requeue_source: str | None = None,
    ) -> ConflictPublicationRequest:
        """Build the conflict error request for an unexpected failure."""

        branch = (
            prepared.branch if prepared is not None else _conflict_attempt_branch(claim.issue.number)
        )
        base_branch = profile.base_branch if profile is not None else "main"
        return self._build_conflict_request(
            claim,
            checkpoint,
            branch,
            base_branch,
            None,
            CONFLICT_OUTCOME_ERROR,
            "",
            f"Conflict attempt failed before a result could be verified: {detail}",
            push_branch=False,
            requeue_source=requeue_source,
        )

    def _conflict_rebase_request(
        self,
        claim: Claim,
        checkpoint: AttemptCheckpoint,
        prepared: PreparedAttempt | None,
        profile: RepositoryProfile | None,
        pull_request: PullRequest | None,
        conflicted_files: tuple[str, ...],
        *,
        requeue_source: str | None = None,
    ) -> ConflictPublicationRequest:
        """Build the conflict request for a rebase that hit conflicts."""

        branch = (
            prepared.branch if prepared is not None else _conflict_attempt_branch(claim.issue.number)
        )
        base_branch = profile.base_branch if profile is not None else "main"
        return self._build_conflict_request(
            claim,
            checkpoint,
            branch,
            base_branch,
            pull_request,
            CONFLICT_OUTCOME_REBASE_CONFLICT,
            "",
            _conflict_rebase_conflict_details(branch, base_branch, conflicted_files),
            push_branch=False,
            conflicted_files=tuple(conflicted_files),
            requeue_source=requeue_source,
        )

    def _publish_conflict_bounded(self, request: ConflictPublicationRequest, claim: Claim):
        """Publish a conflict attempt, bounding the call by the handoff budget.

        The request is stamped with the pre-publish budget standing first
        (the streak is read here, never at claim time); when the published
        result escalated and its comment posted, the issue is handed to a
        human with ``ready-for-human``.
        """

        stamped = self._stamp_conflict_budget(claim, request)
        budget = self._handoff_publish_budget()
        if budget is None:
            published = self._publisher.publish_conflict(stamped)
        else:
            published = self._publisher.publish_conflict(stamped, timeout=budget)
        self._maybe_escalate_conflict_to_human(
            claim, published, getattr(published, "escalated", stamped.escalate)
        )
        # Remember the verified result for telemetry (no extra lookup), plus
        # the workspace state observed right after publication. The pull
        # request is also written into the attempt's archive record here, at
        # publication, so the telemetry seed finds it after a restart.
        self._remember_publication_result(request.started_at, published)
        self._archive_pull_request(request.issue_number, request.started_at, published)
        self._observe_and_remember_workspace_dirty(request.started_at)
        return published

    def _publish_terminal(
        self,
        claim: Claim,
        started_at: str,
        prepared: PreparedAttempt | None,
        profile: RepositoryProfile | None,
        outcome: AttemptOutcome,
        details: str = "Repository profile could not be loaded or the attempt could not start.",
        timeout: float | None = None,
        auto_merge_opt_in: bool = False,
        requeue_source: str = "human",
    ):
        branch = prepared.branch if prepared is not None else f"agent/issue-{claim.issue.number}"
        base_branch = profile.base_branch if profile is not None else "main"
        request = PublicationRequest(
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
            auto_merge_opt_in=auto_merge_opt_in,
            requeue_source=requeue_source,
        )
        if timeout is None:
            published = self._publisher.publish(request)
        else:
            published = self._publisher.publish(request, timeout=timeout)
        self._remember_publication_result(started_at, published)
        self._archive_pull_request(claim.issue.number, started_at, published)
        self._observe_and_remember_workspace_dirty(started_at)
        return published

    def _publish_rebase_conflict(
        self,
        claim: Claim,
        started_at: str,
        prepared: PreparedAttempt | None,
        profile: RepositoryProfile | None,
        error: RebaseConflictError,
        timeout: float | None = None,
        auto_merge_opt_in: bool = False,
        requeue_source: str = "human",
    ):
        """Report an unresolvable rebase conflict as the terminal infrastructure error.

        The model already had its chance to resolve the conflict that
        workspace preparation left in place; the rebase was aborted and the
        branch restored to its pre-rebase state, so setup never ran against
        a half-merged tree. The failure is classified here with its
        ``rebase_conflict`` cause instead of surfacing later as an
        unrelated setup error.
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
            timeout=timeout,
            auto_merge_opt_in=auto_merge_opt_in,
            requeue_source=requeue_source,
        )

    def _record_terminal_outcome(
        self, outcome: AttemptOutcome, attempt_id: str, *, issue_number: int | None = None
    ) -> None:
        """Advance the process-wide guard only after attempt cleanup is complete."""

        if self._error_store is None:
            return
        # Share the control lock with the operator reset so a reset racing
        # a terminal outcome cannot lose its write between read and write.
        with self._control_lock:
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
        archive: AttemptArchive | None,
        outcome: AttemptOutcome,
        started_at: str,
        *,
        issue_number: int | None = None,
        issue_title: str | None = None,
    ) -> str:
        """Write the outcome record; return its ``completed_at`` timestamp.

        The issue number and title ride along so the telemetry seed can
        report the finished attempt after a restart without any lookup.
        """

        completed_at = datetime.now(UTC)
        finished_at = completed_at.isoformat().replace("+00:00", "Z")
        if archive is not None:
            started = datetime.fromisoformat(started_at.removesuffix("Z") + "+00:00")
            payload: dict[str, object] = {
                "outcome": outcome.value,
                "started_at": started_at,
                "completed_at": finished_at,
                "duration_seconds": (completed_at - started).total_seconds(),
            }
            if (
                isinstance(issue_number, int)
                and not isinstance(issue_number, bool)
                and issue_number > 0
            ):
                payload["issue_number"] = issue_number
            if isinstance(issue_title, str) and issue_title:
                payload["issue_title"] = issue_title
            archive.write_attempt(payload)
        return finished_at


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


def _format_conflicted_files(files: tuple[str, ...]) -> str:
    """Render the conflicted file list for event details, or "" when unknown."""

    if not files:
        return ""
    return f" Conflicting files: {', '.join(files)}."


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _remaining_conflicts(state: ConflictState, prepared: PreparedAttempt) -> tuple[str, ...]:
    """Conflicted files from the tree, or those preparation reported."""

    return state.conflicted_files or tuple(
        str(path) for path in (prepared.rebase_conflicts or ()) if str(path).strip()
    )


def _claim_time_opt_in(claim: Claim, checkpoint: AttemptCheckpoint | None) -> bool:
    """Whether the issue was opted into auto-merge when it was claimed.

    The checkpoint recorded the label set at claim time, so a label added
    while the attempt runs (or after it finishes) never enables that
    attempt's merge — including when a crashed attempt is recovered, where
    the re-read claim carries the current labels instead.
    """

    if checkpoint is not None:
        return bool(getattr(checkpoint, "auto_merge_opt_in", False))
    return AUTO_MERGE_LABEL in claim.issue.labels


def _claim_time_requeue_source(claim: Claim, checkpoint: AttemptCheckpoint | None) -> str:
    """How the attempt was requeued, decided once from the claim snapshot.

    The checkpoint recorded the claim snapshot's labels, so a label change
    while the attempt runs (or after it finishes) never rewrites that
    attempt's result comment — including when a crashed attempt is
    recovered, where the re-read claim carries the current labels instead.
    """

    if checkpoint is not None:
        source = getattr(checkpoint, "requeue_source", None)
        if source in (REQUEUE_SOURCE_HUMAN, REQUEUE_SOURCE_AUTOMATIC, REQUEUE_SOURCE_CONFLICT):
            return source
    return requeue_source_for_labels(claim.issue.labels)


def _publication_confirmed(published: object) -> bool:
    """Whether an attempt's publication is verified and finalizable.

    Both the legacy per-attempt result comment and the shared handoff
    record (a handoff outcome's second comment) must confirm: an
    unconfirmed shared record retains the attempt for recovery instead of
    releasing, cleaning up, and finalizing. Non-handoff outcomes and older
    publisher doubles report no shared record, which reads as confirmed.
    """

    return bool(getattr(published, "comment_posted", True)) and bool(
        getattr(published, "handoff_posted", True)
    )


def _is_conflict_attempt(checkpoint: AttemptCheckpoint | None) -> bool:
    """Whether a checkpoint belongs to a conflict attempt.

    A missing checkpoint (or a checkpoint written before the kind existed)
    reads as an ordinary attempt.
    """

    if checkpoint is None:
        return False
    return getattr(checkpoint, "attempt_kind", ATTEMPT_KIND_ORDINARY) == ATTEMPT_KIND_CONFLICT


def _attempt_id(checkpoint: AttemptCheckpoint | None) -> str:
    """The stable attempt ID carried in seal trailers and handoff notes.

    This is the checkpoint's ``started_at``. The lifecycle always starts
    the checkpoint before the runner executes; a missing checkpoint falls
    back to a placeholder so sealing still preserves the work instead of
    failing the attempt.
    """

    if checkpoint is None:
        return "unknown"
    return checkpoint.started_at


def _published_pull_request_value(published: object) -> dict | None:
    """The ``{number, url}`` pull request from a publication result, or ``None``."""

    raw = getattr(published, "pull_request", None)
    if raw is None:
        return None
    if isinstance(raw, dict):
        number = raw.get("number")
        url = raw.get("url")
    else:
        number = getattr(raw, "number", None)
        url = getattr(raw, "url", None)
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        return None
    if not isinstance(url, str) or not url:
        return None
    return {"number": number, "url": url}


def _review_findings_text(review: ReviewEvidence, review_count: int) -> str:
    """Render the published "Review findings" line without claiming uncollected work.

    Findings are never parsed out of the ``code-review`` skill output (tracked
    separately), so an empty tuple must not read as ``all clear``: when the
    skill was invoked its findings were not collected, and when it was never
    invoked no review ran.
    """

    if review.findings:
        return "; ".join(finding.summary for finding in review.findings)
    if review_count > 0:
        return "not collected"
    return "not run"


def _execute_model(
    model_executor: object,
    *,
    issue_body: str,
    working_directory: Path,
    issue_number: int | None,
    archive: object | None,
    attempt_id: str | None,
    branch: str | None = None,
    base_revision: str | None = None,
) -> Any:
    """Invoke the model executor, passing the attempt id when it accepts it.

    Older executor doubles in tests predate the ``attempt_id`` seam: they
    raise ``TypeError`` on the unexpected keyword. Fall back to the old
    call shape for them instead of failing the attempt. The branch/base
    seam (for the token soft-threshold state snapshot) falls back the same
    way.
    """

    execute = model_executor.execute  # type: ignore[attr-defined]
    try:
        return execute(
            issue_body=issue_body,
            working_directory=working_directory,
            issue_number=issue_number,
            archive=archive,
            attempt_id=attempt_id,
            branch=branch,
            base_revision=base_revision,
        )
    except TypeError:
        pass
    try:
        return execute(
            issue_body=issue_body,
            working_directory=working_directory,
            issue_number=issue_number,
            archive=archive,
            attempt_id=attempt_id,
        )
    except TypeError:
        return execute(
            issue_body=issue_body,
            working_directory=working_directory,
            issue_number=issue_number,
            archive=archive,
        )


def _emit_handoff_note_events(
    event_log: Callable[..., None],
    issue_number: int,
    validation: _HandoffNoteValidation,
) -> None:
    """Emit the WARNING events for one handoff-note verdict.

    A bad ``started_at`` never rejects the note: it emits
    ``handoff_note_started_at_invalid`` and the note is otherwise judged as
    before. Any invalid note additionally emits ``handoff_note_rejected``
    with the validation reason, so a silent downgrade to ``incomplete`` is
    always visible in the event stream.
    """

    if validation.started_at_warning:
        event_log(
            "handoff_note_started_at_invalid",
            validation.started_at_warning,
            level="WARNING",
            issue_number=issue_number,
        )
    if not validation.valid:
        event_log(
            "handoff_note_rejected",
            validation.reason,
            level="WARNING",
            issue_number=issue_number,
        )


def _check_failure_step(result: object) -> tuple[str | None, int | None]:
    """Name the failed check step and its 1-based position.

    The first command with a non-zero exit wins (a timeout leaves its
    command with ``exit_code None``, which also sorts as failed); when
    every command exited cleanly the last one is blamed instead, mirroring
    :func:`completion._select_failed_command`. ``(None, None)`` means no
    command ran.
    """

    commands = tuple(getattr(result, "commands", ()) or ())
    for index, command in enumerate(commands):
        if getattr(command, "exit_code", 0) != 0:
            return str(getattr(command, "command", "?")), index + 1
    if commands:
        last = commands[-1]
        return str(getattr(last, "command", "?")), len(commands)
    return None, None


def _attempt_evidence(
    decision: CompletionDecision,
    profile: RepositoryProfile,
    final_check: object | None,
    review_cycles: int,
    review_findings: str,
    details: str = "Attempt did not reach local completion evidence.",
    note_commit_sha: str | None = None,
    reviews_performed: int = 0,
) -> AttemptEvidence:
    steps = tuple(profile.check)
    exit_code = getattr(final_check, "exit_code", None)
    failed_step: str | None = None
    failed_index: int | None = None
    if final_check is not None and exit_code != 0:
        failed_step, failed_index = _check_failure_step(final_check)
    return AttemptEvidence(
        decision=decision,
        check_command=" && ".join(profile.check),
        check_exit_code=exit_code,
        review_cycles=review_cycles,
        review_findings=review_findings,
        details=details,
        note_commit_sha=note_commit_sha,
        reviews_performed=reviews_performed,
        check_steps=steps,
        check_failed_step=failed_step,
        check_failed_index=failed_index,
    )


def _log_completion_decision(
    event_log: Callable[..., None],
    issue_number: int,
    decision: CompletionDecision,
) -> None:
    """Emit the evaluator's outcome and reasons as one auditable event.

    A run that commits changes but never invokes ``code-review`` ends
    ``incomplete`` ("Acceptance criteria are not satisfied.") with no other
    visible cause; this event keeps that reason in ``agent_output.json``.
    """

    outcome = decision.outcome.value if decision.outcome is not None else "none"
    reasons = "; ".join(decision.reasons) if decision.reasons else "none"
    event_log(
        "completion_decided",
        f"outcome={outcome}; reasons={reasons}",
        level=(
            "WARNING"
            if decision.outcome
            in (AttemptOutcome.INCOMPLETE, AttemptOutcome.NO_CHANGES, AttemptOutcome.INFRASTRUCTURE_ERROR)
            else "INFO"
        ),
        issue_number=issue_number,
    )


def _command_failure_detail(result: object, *, limit: int = 4000) -> str:
    """Surface the failing command and its output for the live event stream."""

    failed = _select_failed_command(result)
    if failed is None:
        return "timed out before any command ran"
    stderr = (getattr(failed, "stderr", "") or "").strip()
    output = stderr or (getattr(failed, "stdout", "") or "").strip()
    detail = f"`{getattr(failed, 'command', '')}` (exit {getattr(failed, 'exit_code', None)}): {output}"
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

    note_path = working_directory / handoff_note_relpath(issue_number)
    try:
        return note_path.read_text()
    except OSError:
        return "Handoff was requested, but the handoff note file could not be read."


def _prompt_redactor(
    profile: RepositoryProfile, redactions: Sequence[str]
) -> Redactor:
    """Combine operator credentials with profile env values for prompt redaction."""

    values = [*redactions, *(value for value in profile.env.values() if value)]
    return Redactor(values)


def _gate_section(
    profile: RepositoryProfile,
    redactions: Sequence[str],
    *,
    include_review_instruction: bool = True,
) -> str:
    """Describe the repository check gate the attempt is judged by.

    Every check command is listed exactly as the profile gives it (after
    redaction), one per line and numbered, in profile order. When the
    profile carries ``env``, only the variable names are shown so the model
    can reproduce the gate's environment without seeing the values. Conflict
    resolution sessions pass ``include_review_instruction=False``: bringing
    the branch up to date needs no review, so the mandatory ``code-review``
    gate for ordinary work stays out of their prompt.
    """

    redactor = _prompt_redactor(profile, redactions)
    lines = [
        "---",
        "## Completion gate",
        "",
        "Before the attempt counts as complete, the branch must pass the "
        "following commands, in this order:",
        "",
    ]
    for index, command in enumerate(profile.check, start=1):
        redacted = redactor.redact(command)
        lines.append(f"{index}. {redacted}")
    lines += [
        "",
        "The agent runs these commands itself as the final check after the "
        "model finishes. They run in sequence and the first failure fails "
        "the gate.",
        "",
    ]
    if include_review_instruction:
        lines += [
            "Before finishing, invoke the `code-review` skill on your changes: "
            "an attempt with commits but no `code-review` invocation in this run "
            "is classified as incomplete, even when the checks pass and the "
            "change is small (for example formatting).",
        ]
    if profile.env:
        names = ", ".join(profile.env.keys())
        lines.append(
            "The gate runs with these environment variables set: "
            f"{names}. Their values are not shown here."
        )
    instructions = profile.model_instructions
    if isinstance(instructions, str) and instructions.strip():
        redacted_instructions = redactor.redact(instructions)
        lines += [
            "",
            "### Repository guidance",
            "",
            "The following repository-provided guidance also applies:",
            "",
            str(redacted_instructions),
        ]
    return "\n".join(lines)


def _build_starting_prompt(
    issue_body: str,
    comments: Sequence[str],
    *,
    continuation: bool,
    issue_number: int,
    profile: RepositoryProfile,
    redactions: Sequence[str] = (),
    setup_deferred: bool = False,
    baseline: object | None = None,
) -> str:
    """Compose the model's starting context from the issue body and trusted comments.

    Trusted comments are appended verbatim in the chronological order the
    tracker already returns them in. A continuation adds a read-first
    instruction naming the current Handoff note file as authoritative: the
    note's own content is never fetched or digested here, since it is
    already reachable on the resumed branch. A setup-deferred attempt adds
    the rebase-conflict resolution instruction.

    A failed pre-model ``baseline`` (only possible on a tolerated continuation
    baseline; a red first-attempt baseline stops before dispatch) adds the
    failure section so the model knows the branch is red and what to fix. A
    passing or absent baseline leaves the prompt byte-for-byte unchanged.

    Every prompt ends with the repository check gate so the model knows
    which commands the final check runs, in which order.
    """

    parts = [issue_body, *comments]
    if continuation:
        parts.append(
            "This is a continued attempt. Read the current Handoff note at "
            f".agent/handoff/{issue_number}.md before implementing: it is the "
            "authoritative, self-contained record of the previous attempt's "
            "state, remaining work, and evidence, and it supersedes every "
            "historical handoff result comment. "
            "Invoke the `code-review` skill on your changes before finishing, "
            "even when this run only verifies earlier work or fixes formatting: "
            "without a `code-review` invocation in this run the attempt is "
            "classified as incomplete."
        )
    if setup_deferred:
        parts.append(
            "The attempt branch has unresolved merge conflicts from rebasing onto "
            f"`{profile.base_branch}`. Resolve them first (inspect with git status, edit the conflicted "
            "files, stage with git add, and run git rebase --continue until the rebase completes), "
            "then continue the implementation."
        )
    if baseline is not None and not getattr(baseline, "succeeded", True):
        parts.append(_build_baseline_failure_section(baseline))
    parts.append(_gate_section(profile, redactions))
    return "\n\n".join(parts)


_HANDOFF_RESULT_MARKERS = (
    "Agent Attempt Result: incomplete",
    "Agent Attempt Result: handoff",
)

#: Result comments carrying this branch line never published a branch, so they
#: are not handoffs and must not trigger the continuation path (issue #84).
_NO_PUBLISHED_BRANCH_MARKER = "no branch created"

#: Detail written by the ``continuation_branch_missing`` short-circuit itself.
#: Such a comment must never count as a continuation signal, or every later
#: attempt would short-circuit the same way (issue #84).
_CONTINUATION_MISSING_MARKER = "was not found on origin"


def _continuation_expected(
    issue: TrackerIssue, issue_comments: Callable[[TrackerIssue], tuple[str, ...]]
) -> bool:
    """Whether this issue carries (or carried) a signal that a continuation was expected.

    The ``round-finished`` label is the current signal; a prior handoff or
    incomplete attempt-result comment is the historical one, but only when it
    actually published a branch. Baseline/setup failures (and the
    ``continuation_branch_missing`` short-circuit itself) post ``incomplete``
    with ``no branch created`` — none of them promises a remote branch, so
    they must not trigger the continuation path. Checked in that order so a
    present label never triggers a needless comment fetch, and so a real
    handoff comment is still recognized once the label has been removed (e.g.
    on claim) or the remote branch is missing.
    """

    if ROUND_FINISHED in issue.labels:
        return True
    for comment in issue_comments(issue):
        if not any(marker in comment for marker in _HANDOFF_RESULT_MARKERS):
            continue
        if _CONTINUATION_MISSING_MARKER in comment:
            continue
        if _NO_PUBLISHED_BRANCH_MARKER in comment:
            continue
        return True
    return False


# ``cost_soft_threshold`` is the legacy name of ``token_soft_threshold``: it is
# still accepted when reading old notes, but new notes never use it.
_ALLOWED_HANDOFF_REASONS = frozenset(
    {
        "token_soft_threshold",
        "token_hard_ceiling",
        "cost_soft_threshold",
        "cost_hard_limit",
        "turn_limit",
        "time_limit",
        "operator_request",
    }
)


def _classify_handoff_reason(execution: object) -> str:
    """Classify the emergency handoff note reason from structured signals.

    The latched token hard-ceiling flag wins over every terminal signal; the
    SDK budget signals map to ``cost_hard_limit``, turn signals to
    ``turn_limit``, and timeout signals to ``time_limit``. The explanation
    text is never read. Unknown signals default to ``cost_hard_limit``.
    """

    if bool(getattr(execution, "token_hard_ceiling_reached", False)):
        return "token_hard_ceiling"
    stop_reason = getattr(execution, "stop_reason", None) or ""
    terminal_reason = getattr(execution, "terminal_reason", None) or ""
    stop = stop_reason.lower() if isinstance(stop_reason, str) else ""
    terminal = terminal_reason.lower() if isinstance(terminal_reason, str) else ""
    if (
        terminal in ("budget_exhausted", "error_max_budget_usd")
        or "budget" in terminal
        or stop == "max_budget_usd_exceeded"
        or "budget" in stop
    ):
        return "cost_hard_limit"
    if (
        terminal in ("max_turns", "error_max_turns")
        or "turn" in terminal
        or stop == "max_turns_exceeded"
        or "turn" in stop
    ):
        return "turn_limit"
    if (
        stop == "timeout"
        or "timeout" in terminal
        or "timeout" in stop
        or stop.startswith("aborted_")
        or terminal.startswith("aborted_")
    ):
        return "time_limit"
    return "cost_hard_limit"


# Bounds for the emergency handoff note: the tool-call trail shows the last
# 15 calls, file lists show the first 50 entries plus a `+N more` line, the
# last assistant text is truncated to about 1,500 characters, and a preserved
# previous note is truncated to about 3,000 characters.
_EMERGENCY_TOOL_TRAIL_LIMIT = 15
_EMERGENCY_FILE_LIST_LIMIT = 50
_EMERGENCY_ASSISTANT_TEXT_LIMIT = 1500
_EMERGENCY_PREVIOUS_NOTE_LIMIT = 3000


@dataclass(frozen=True)
class _PreviousCompleteNote:
    """A prior round's Handoff note commit and what it held.

    ``content`` is ``None`` when the revision could not be read: the
    emergency note then names the revision explicitly instead of pretending
    the older note describes the current branch.
    """

    revision: str
    content: str | None


#: Scoping id for the previous-note lookup: unparseable by
#: ``attempt_start_epoch`` on purpose, so ``handoff_note_commits`` applies no
#: time bound and returns every round's note commits, newest first.
_PREVIOUS_NOTE_ATTEMPT_ID = "previous-rounds"

#: Marker the runtime stamps into every note it writes itself, distinguishing
#: emergency notes (runtime evidence only) from complete ones.
_RUNTIME_WRITTEN_MARKER = "written by the runtime"


def _is_complete_handoff_note(content: str, issue_number: int) -> bool:
    """Whether ``content`` is a complete note of an earlier round for this issue.

    Complete means it parses with a matching title and ``issue`` field and
    was not written by the runtime: an emergency note carries only runtime
    evidence, never a continuation-ready assessment.
    """

    fields = _parse_handoff_note_fields(content)
    return (
        fields.get("title_issue") == str(issue_number)
        and fields.get("issue") == str(issue_number)
        and _RUNTIME_WRITTEN_MARKER not in content
    )


def _previous_complete_note(
    workspace: AttemptWorkspace,
    prepared: PreparedAttempt,
    issue_number: int,
) -> _PreviousCompleteNote | None:
    """The newest complete Handoff note of an earlier round, if any.

    Never raises: an unreadable history degrades to an absent or unreadable
    note instead of failing the emergency path.
    """

    try:
        commits = workspace.handoff_note_commits(
            prepared, issue_number, attempt_id=_PREVIOUS_NOTE_ATTEMPT_ID
        )
    except Exception:
        return None
    unreadable: str | None = None
    for commit in commits:
        try:
            content = workspace.handoff_note_content_at_revision(
                commit.revision, issue_number
            )
        except Exception:
            content = None
        if content is None:
            if unreadable is None:
                unreadable = commit.revision
            continue
        if _is_complete_handoff_note(content, issue_number):
            return _PreviousCompleteNote(revision=commit.revision, content=content)
    if unreadable is not None:
        return _PreviousCompleteNote(revision=unreadable, content=None)
    return None


def _build_emergency_handoff_note(
    *,
    issue_number: int,
    started_at: str,
    reason: str,
    last_work_commit: str,
    explanation: str,
    diff_stat: str | None = None,
    attempt_commits: Sequence[str] = (),
    sealed_files: Sequence[str] | None = None,
    edited_files: Sequence[str] = (),
    last_assistant_text: str | None = None,
    handoff_skill_invoked: bool = False,
    tool_calls: Sequence[ToolCallRecord] = (),
    last_check_call: ToolCallRecord | None = None,
    baseline_result: object | None = None,
    token_budget: Mapping[str, Any] | None = None,
    archive_dir: str | None = None,
    session_id: str | None = None,
    previous_note: _PreviousCompleteNote | None = None,
) -> str:
    """Render the runtime-written handoff note for a limit stop without a model note.

    The title and four header fields are exactly as before so
    ``_validate_handoff_note`` still passes; every section below them
    degrades to ``"not available"`` / ``"none recorded"`` / ``"none"``
    instead of failing when a fact is missing.
    """

    header = (
        f"# Handoff note: issue #{issue_number}\n\n"
        f"- issue: {issue_number}\n"
        f"- started_at: {started_at}\n"
        f"- reason: {reason}\n"
        f"- last_work_commit: {last_work_commit}\n"
    )
    sections = [
        ("Provenance", _emergency_provenance(reason, explanation)),
        ("Previous complete note", _emergency_previous_note(previous_note)),
        (
            "Changes",
            _emergency_changes(
                diff_stat=diff_stat,
                attempt_commits=attempt_commits,
                sealed_files=sealed_files,
                edited_files=edited_files,
            ),
        ),
        (
            "Last model intent",
            _emergency_last_intent(
                last_assistant_text=last_assistant_text,
                handoff_skill_invoked=handoff_skill_invoked,
            ),
        ),
        (
            "Activity trail",
            _emergency_activity(
                edited_files=edited_files,
                tool_calls=tool_calls,
                last_check_call=last_check_call,
            ),
        ),
        ("Check state", _emergency_check_state(baseline_result=baseline_result)),
        ("Budget facts", _emergency_budget(token_budget=token_budget)),
        (
            "Pointers",
            _emergency_pointers(archive_dir=archive_dir, session_id=session_id),
        ),
        ("Remaining work", _emergency_remaining_work()),
    ]
    body = "".join(f"\n## {title}\n\n{content}\n" for title, content in sections)
    return f"{header}{body}"


def _emergency_provenance(reason: str, explanation: str) -> str:
    """Who wrote the note and which limit fired."""

    detail = (explanation or "").strip() or "not available"
    return (
        "This note was written by the runtime because the model did not write "
        "a handoff note before the attempt ended.\n"
        f"The limit that fired was `{reason}`.\n\n"
        "Model's terminal explanation:\n\n"
        f"{_blockquote(detail)}"
    )


def _emergency_previous_note(previous_note: _PreviousCompleteNote | None) -> str:
    """The last complete note of an earlier round, or why none is shown.

    A found note is quoted verbatim (capped) at its immutable revision and
    marked as older context, clearly distinguished from the runtime evidence
    below, which describes the current branch. A missing or unreadable note
    is said explicitly, so the continuation never mistakes the older note
    for the current branch state.
    """

    if previous_note is None:
        return (
            "No previous complete Handoff note was found on this branch: "
            "no earlier round committed a readable, complete note for this issue."
        )
    if previous_note.content is None:
        return (
            "The previous round's Handoff note commit "
            f"`{previous_note.revision}` could not be read from this branch, "
            "so its older context is unavailable; only the runtime evidence "
            "below describes the current branch."
        )
    content = previous_note.content.strip()
    if len(content) > _EMERGENCY_PREVIOUS_NOTE_LIMIT:
        content = content[:_EMERGENCY_PREVIOUS_NOTE_LIMIT] + "...[truncated]"
    return (
        "The last complete Handoff note "
        f"(revision `{previous_note.revision}`) is preserved below as older "
        "context: it describes the branch as an earlier round left it; the "
        "runtime evidence below describes the current branch.\n\n"
        f"{_blockquote(content)}"
    )


def _emergency_changes(
    *,
    diff_stat: str | None,
    attempt_commits: Sequence[str],
    sealed_files: Sequence[str] | None,
    edited_files: Sequence[str],
) -> str:
    """Preserved work: diff stat, attempt commits, sealed and edited files."""

    stat_block = (diff_stat or "").strip() or "not available"
    commits = [line for line in (str(line).strip() for line in attempt_commits) if line]
    commits_block = (
        "\n".join(f"- {line}" for line in commits) if commits else "none"
    )
    if sealed_files is None:
        sealed_block = "none"
    else:
        sealed_block = _format_capped_files(sealed_files, empty="none")
    return (
        "Changes since the verified base "
        "(`git diff --stat <base>..<seal or HEAD>`):\n\n"
        "```diff\n"
        f"{stat_block}\n"
        "```\n\n"
        "Attempt commits (`git log --oneline <base>..HEAD`):\n\n"
        f"{commits_block}\n\n"
        "Files that were uncommitted when sealing happened "
        "(committed by the seal commit):\n\n"
        f"{sealed_block}\n\n"
        "Files the model edited or created (Edit/Write calls):\n\n"
        f"{_format_capped_files(edited_files, empty='none recorded')}"
    )


def _emergency_last_intent(
    *,
    last_assistant_text: str | None,
    handoff_skill_invoked: bool,
) -> str:
    """The model's last words and whether it invoked the handoff skill."""

    detail = (last_assistant_text or "").strip()
    if not detail:
        text_block = "not available"
    elif len(detail) > _EMERGENCY_ASSISTANT_TEXT_LIMIT:
        text_block = _blockquote(
            detail[:_EMERGENCY_ASSISTANT_TEXT_LIMIT] + "...[truncated]"
        )
    else:
        text_block = _blockquote(detail)
    invoked = "invoked" if handoff_skill_invoked else "not invoked"
    return (
        "Last assistant text (truncated to about 1,500 characters):\n\n"
        f"{text_block}\n\n"
        f"Handoff skill on the main thread: {invoked}."
    )


def _emergency_activity(
    *,
    edited_files: Sequence[str],
    tool_calls: Sequence[ToolCallRecord],
    last_check_call: ToolCallRecord | None,
) -> str:
    """Edited files, the recent tool trail, and the last check-command call."""

    trail = list(tool_calls)[-_EMERGENCY_TOOL_TRAIL_LIMIT:]
    if trail:
        calls_block = "\n".join(_format_tool_call(record) for record in trail)
    else:
        calls_block = "none recorded"
    if last_check_call is None:
        check_block = "none recorded"
    else:
        command = " ".join((last_check_call.command or "").split())
        check_block = f"`{command}` ({last_check_call.outcome})"
    return (
        "Files the model edited or created (Edit/Write calls):\n\n"
        f"{_format_capped_files(edited_files, empty='none recorded')}\n\n"
        "Recent tool calls (last 15: tool, short target, ok/error flag):\n\n"
        f"{calls_block}\n\n"
        "Last Bash call running a profile `check` command:\n\n"
        f"{check_block}"
    )


def _emergency_check_state(baseline_result: object | None) -> str:
    """The final check never runs on this path; report the baseline if one ran."""

    if baseline_result is None:
        baseline_block = "not available (no baseline check ran)"
    else:
        succeeded = getattr(baseline_result, "succeeded", None)
        if succeeded is True:
            baseline_block = (
                f"passed (exit {getattr(baseline_result, 'exit_code', None)})"
            )
        elif succeeded is False:
            baseline_block = f"failed: {_command_failure_detail(baseline_result)}"
        else:
            baseline_block = "not available"
    return (
        "final check: not run (the runtime does not run the final check "
        "before writing an emergency note).\n\n"
        f"Baseline check (pre-model): {baseline_block}."
    )


def _emergency_budget(token_budget: Mapping[str, Any] | None) -> str:
    """Token-budget facts, each degrading to ``not available`` when missing."""

    budget = token_budget if isinstance(token_budget, Mapping) else {}
    soft_crossed = budget.get("soft_threshold_crossed")
    soft_at = budget.get("soft_threshold_crossed_at_tokens")
    if soft_crossed and soft_at is not None:
        soft_block = f"crossed at {soft_at} tokens"
    elif soft_crossed:
        soft_block = "crossed"
    else:
        soft_block = "not crossed"
    prompt_cache = budget.get("prompt_cache")
    hit_rate = (
        prompt_cache.get("main_hit_rate")
        if isinstance(prompt_cache, Mapping)
        else None
    )
    lines = [
        f"- budget_tokens: {_budget_value(budget.get('budget_tokens'))}",
        f"- max_budget_tokens: {_budget_value(budget.get('max_budget_tokens'))}",
        f"- soft threshold: {soft_block}",
        f"- peak_main_context_tokens: "
        f"{_budget_value(budget.get('peak_main_context_tokens'))}",
        f"- main prompt-cache hit rate: "
        f"{hit_rate if hit_rate is not None else 'not available'}",
        f"- turns: {_budget_value(budget.get('turns'))}",
        f"- elapsed: {_elapsed_value(budget.get('elapsed_seconds'))}",
    ]
    return "\n".join(lines)


def _emergency_pointers(
    *, archive_dir: str | None, session_id: str | None
) -> str:
    """Where to look next: the attempt log directory and the CLI session id."""

    directory = (archive_dir or "").strip() or "not available"
    session = (session_id or "").strip() or "not available"
    return (
        f"- attempt log directory: {directory}\n"
        f"- CLI transcript session id: {session}"
    )


def _emergency_remaining_work() -> str:
    """What is left: unknown to the runtime; the issue decides; nothing verified."""

    return (
        "The runtime could not determine what work remains: the attempt "
        "stopped at a limit without a model-written assessment of what is "
        "left.\n\n"
        "The issue's acceptance criteria are the reference for what \"done\" "
        "means. None were verified by the runtime on this path.\n\n"
        "Review the sections above and continue implementation from the "
        "preserved work on this branch."
    )


def _blockquote(text: str) -> str:
    """Quote free model text so it can never be misread as a note header field."""

    lines = text.strip().splitlines() or [text]
    return "\n".join(f"> {line}" if line.strip() else ">" for line in lines)


def _format_capped_files(files: Sequence[str], *, empty: str) -> str:
    """Render a file list capped at 50 entries with a ``+N more`` line."""

    paths = [str(path).strip() for path in files]
    paths = [path for path in paths if path]
    if not paths:
        return empty
    shown = paths[:_EMERGENCY_FILE_LIST_LIMIT]
    lines = [f"- `{path}`" for path in shown]
    remaining = len(paths) - len(shown)
    if remaining:
        lines.append(f"- +{remaining} more")
    return "\n".join(lines)


def _format_tool_call(record: ToolCallRecord) -> str:
    """Render one trail entry as tool name, short target, and ok/error flag."""

    target = f" {record.target}" if record.target else ""
    return f"- `{record.tool}`{target} ({record.outcome})"


def _budget_value(value: object) -> str:
    return "not available" if value is None else str(value)


def _elapsed_value(value: object) -> str:
    if value is None:
        return "not available"
    return f"{value}s"


def _main_thread_handoff_invoked(skill_events: object) -> bool:
    """Whether the ``handoff`` skill was invoked in the main thread."""

    events = skill_events if isinstance(skill_events, (list, tuple)) else ()
    return any(
        getattr(event, "name", None) == "handoff"
        and getattr(event, "agent_id", None) is None
        for event in events
    )


def _last_check_call(
    bash_calls: Sequence[ToolCallRecord], check_commands: Sequence[str]
) -> ToolCallRecord | None:
    """The last Bash call whose command contains a profile ``check`` command."""

    checks = tuple(
        command
        for command in check_commands
        if isinstance(command, str) and command
    )
    if not checks:
        return None
    for record in reversed(list(bash_calls)):
        command = record.command if isinstance(record, ToolCallRecord) else ""
        if any(check in command for check in checks):
            return record
    return None


def _git_diff_stat(
    working_directory: Path, base_revision: str, endpoint: str
) -> str | None:
    """Best-effort ``git diff --stat`` for the emergency note; ``None`` when unavailable."""

    try:
        completed = subprocess.run(
            ("git", "diff", "--stat", f"{base_revision}..{endpoint}"),
            cwd=working_directory,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    output = (completed.stdout or "").strip()
    return output if output else "(no differences)"

_HANDOFF_NOTE_TITLE_PATTERN = re.compile(r"^#\s*Handoff note:\s*issue #(\d+)")
_HANDOFF_NOTE_FIELD_PATTERN = re.compile(r"^-\s*(issue|started_at|reason|last_work_commit):\s*(.+?)\s*$")


@dataclass(frozen=True)
class _HandoffNoteValidation:
    """Whether a handoff note commit is trustworthy enough to publish."""

    valid: bool
    reason: str = ""
    started_at_warning: str | None = None


def _this_attempt_note_commit_revisions(
    workspace: AttemptWorkspace,
    prepared: PreparedAttempt,
    issue_number: int,
    attempt_id: str,
) -> frozenset[str]:
    """Revisions of this attempt's note-run commits for note validation.

    The run is every note-touching commit of this attempt except work
    commits that only touched the note path by deleting a stale note
    alongside a real change: those end the run, so validation expects
    them as ``last_work_commit`` instead of skipping past them. A
    commit that cannot be inspected fails closed to work (it is not
    skipped), which is exactly the pre-fix strictness: validation then
    expects the commit directly beneath the note.
    """

    try:
        commits = workspace.handoff_note_commits(
            prepared, issue_number, attempt_id=attempt_id
        )
    except Exception:
        return frozenset()
    run: list[str] = []
    for commit in commits:
        try:
            carries_work = workspace.handoff_note_commit_carries_work(
                commit.revision, issue_number
            )
        except Exception:
            continue
        if not carries_work:
            run.append(commit.revision)
    return frozenset(run)


def _validate_handoff_note(
    note_commit: AttemptCommit | None,
    commits: Sequence[AttemptCommit],
    issue_number: int,
    working_directory: Path,
    base_revision: str,
    *,
    note_commit_revisions: Collection[str] | None = None,
) -> _HandoffNoteValidation:
    """Confirm a handoff has a trustworthy, committed note before it may publish.

    The note is located by path (the newest attempt commit touching the
    issue's handoff note file), not by position: a seal commit landing on
    top of a model-committed note does not hide it. Recognizing the note
    commit only proves a commit touched that path; it does not prove the
    note file was actually written, belongs to this issue and attempt, or
    reflects the work actually preserved. All of that is checked here,
    since a continuation trusts the note and a human trusts the
    ``round-finished`` label that only a valid handoff may add.
    """

    if note_commit is None:
        return _HandoffNoteValidation(
            False,
            "Handoff was requested, but no handoff note commit is on the branch.",
        )
    # The seal commit (when the model left dirty work after committing its
    # own note) only adds other files, so reading the note file from the
    # checked-out working tree reads the committed note content.
    note_path = working_directory / handoff_note_relpath(issue_number)
    try:
        text = note_path.read_text()
    except OSError:
        return _HandoffNoteValidation(
            False,
            "Handoff was requested, but the handoff note file could not be read "
            "from the committed branch state.",
        )
    fields = _parse_handoff_note_fields(text)
    # ``started_at`` is the attempt id owned by the runner: it is
    # informational, never a gate. A missing or malformed value produces a
    # warning carried alongside the verdict instead of rejecting the note.
    started_at = fields.get("started_at", "")
    started_at_warning: str | None = None
    if not started_at:
        started_at_warning = (
            f"issue #{issue_number}: the handoff note's `started_at` field is"
            " missing; it should carry the attempt id."
        )
    elif not _is_iso8601(started_at):
        started_at_warning = (
            f"issue #{issue_number}: the handoff note's `started_at` value"
            f" {started_at!r} is not valid ISO-8601; it should carry the attempt id."
        )
    if fields.get("title_issue") != str(issue_number):
        return _HandoffNoteValidation(
            False,
            "Handoff note does not carry the required title for this issue.",
            started_at_warning,
        )
    if fields.get("issue") != str(issue_number):
        return _HandoffNoteValidation(
            False,
            "Handoff note is missing a matching `issue` field.",
            started_at_warning,
        )
    if fields.get("reason") not in _ALLOWED_HANDOFF_REASONS:
        return _HandoffNoteValidation(
            False,
            "Handoff note is missing a valid `reason` field.",
            started_at_warning,
        )
    last_work_commit = fields.get("last_work_commit", "")
    # The note describes the work preserved beneath it: the attempt commit
    # directly under the note in newest-first order, or the base revision
    # when the note is the oldest attempt commit. When the model committed
    # the note more than once, the commits beneath the newest note commit
    # are earlier note commits of this same attempt, not work: skip the
    # unbroken run of this attempt's note-run commits below the note and
    # expect the first commit beneath that run. A note-touching commit
    # that deleted the note while carrying other files is work that
    # dropped a stale note, not a note commit, so it is already excluded
    # from ``note_commit_revisions`` and ends the run. Note commits from
    # earlier rounds of a continued branch are never in
    # ``note_commit_revisions``, so a new round that names the previous
    # round's note commit (HEAD at injection) still validates.
    if note_commit_revisions is None:
        note_revisions = {note_commit.revision}
    else:
        note_revisions = set(note_commit_revisions) | {note_commit.revision}
    note_index = next(
        (index for index, commit in enumerate(commits) if commit.revision == note_commit.revision),
        None,
    )
    if note_index is None:
        expected_last_work_commit = base_revision
    else:
        cursor = note_index + 1
        while cursor < len(commits) and commits[cursor].revision in note_revisions:
            cursor += 1
        if cursor < len(commits):
            expected_last_work_commit = commits[cursor].revision
        else:
            expected_last_work_commit = base_revision
    if (
        len(last_work_commit) < 7
        or not expected_last_work_commit.startswith(last_work_commit)
    ):
        return _HandoffNoteValidation(
            False,
            "Handoff note's `last_work_commit` does not match the last preserved "
            "work commit.",
            started_at_warning,
        )
    return _HandoffNoteValidation(True, started_at_warning=started_at_warning)


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


def _checkpoint_hold_detail(error: Exception) -> str:
    """Operator-facing hold reason for an unreadable attempt checkpoint."""

    return (
        f"{_exception_detail(error)} The checkpoint is preserved for"
        " inspection; issue intake is held until it is repaired or"
        " removed by an operator."
    )


def _check_recovery_request_id(request_id: object) -> None:
    """Validate a recovery request ID without touching control state."""

    if not isinstance(request_id, str) or not request_id.strip():
        from simple_coding_agent.control import RequestIdError

        raise RequestIdError("Request ID must be a non-empty string")
    if len(request_id) > 128:
        from simple_coding_agent.control import RequestIdError

        raise RequestIdError("Request ID must be at most 128 characters")
