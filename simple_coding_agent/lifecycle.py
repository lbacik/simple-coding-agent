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
)
from simple_coding_agent.config import RepositoryProfile
from simple_coding_agent.finalization import AttemptCompletionStore, CompletionStoreError
from simple_coding_agent.git_workspace import (
    DirtyWorkspaceError,
    GitWorkspaceError,
    GitWorkspaceRecoveryError,
    RebaseConflictError,
)
from simple_coding_agent.github_tracker import (
    ROUND_FINISHED,
    Claim,
    TrackerIssue,
    ineligibility_reason,
)
from simple_coding_agent.model_execution import ModelExecutionStatus, ModelExecutor, OperatorHandoff
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


#: The handoff-note reason an operator ``handoff now`` request requires.
_OPERATOR_HANDOFF_REASON = "operator_request"


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
        operator_handoff: OperatorHandoffControl | None = None,
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
        self._operator_handoff = operator_handoff

    def set_operator_handoff(self, control: OperatorHandoffControl | None) -> None:
        """Attach (or detach) the operator handoff served by the next execution.

        A request accepted during setup waits here: the provider is only
        consulted once model execution starts, so setup-phase requests are
        delivered at the first model boundary instead of interrupting setup.
        """

        self._operator_handoff = control

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
            resolution_files = _conflict_resolution_files(self._workspace, prepared)
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
                "files, stage with git add, and run git rebase --continue until the rebase completes), "
                "then continue the implementation."
            )
        self._event_log(
            "model_dispatch_starting",
            f"continuation={continuation}",
            level="INFO",
            issue_number=claim.issue.number,
        )
        self._wire_operator_handoff()
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
        served_operator_request, late_operator_snapshot = self._read_operator_request()
        if setup_deferred:
            # The model had its chance to resolve the conflicts left in
            # place by preparation. Setup must never run against a
            # half-merged tree, so the resolution is verified
            # deterministically first: any remaining rebase/merge state,
            # unmerged path, conflict marker, or whitespace error falls
            # back to the pre-rebase branch state and the attempt reports
            # a rebase_conflict infrastructure error, exactly as if the
            # rebase had been aborted up front.
            problems = _resolution_problems(self._workspace)
            if problems:
                remaining = _conflict_resolution_files(self._workspace, prepared)
                branch = getattr(prepared, "branch", f"agent/issue-{claim.issue.number}")
                abort = getattr(self._workspace, "abort_unresolved_rebase", None)
                if callable(abort):
                    try:
                        abort(prepared)
                    except GitWorkspaceError:
                        self._event_log(
                            "rebase_resolution_failed",
                            f"{' '.join(problems)} The branch could not be restored.",
                            level="ERROR",
                            issue_number=claim.issue.number,
                        )
                        raise
                self._event_log(
                    "rebase_resolution_failed",
                    f"{' '.join(problems)}"
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
        if execution.status in (
            ModelExecutionStatus.HANDOFF_REQUESTED,
            ModelExecutionStatus.MODEL_LIMIT_REACHED,
            ModelExecutionStatus.OPERATOR_HANDOFF_EXPIRED,
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
                getattr(self._workspace, "working_directory"), issue_number
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
        clock: Callable[[], datetime] | None = None,
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
        non-empty body and no open blocker) before anything is recorded; an
        invalid, ineligible, or unverifiable target is rejected without
        changing the intake state or an existing priority. A newer accepted
        priority supersedes only the earlier priority, never a stop plan,
        and never resumes intake. Retrying the same request ID with the
        identical payload returns the stored acknowledgement without
        re-fetching GitHub or applying the command twice.
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
            fetcher = _tracker_method(self._tracker, "fetch_issue")
            if fetcher is None:
                raise NextIssueRejectedError(
                    f"issue #{number} cannot be verified: the tracker cannot"
                    " fetch issues. No control change was accepted."
                )
            try:
                fresh = fetcher(number)
            except NextIssueRejectedError:
                raise
            except Exception as error:
                raise NextIssueRejectedError(
                    f"issue #{number} could not be verified: {error}."
                    " No control change was accepted."
                ) from error
            reason = ineligibility_reason(fresh, number)
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

        from simple_coding_agent.recovery import RecoveryRejectedError, parse_attempt_id

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
            RecoveryRejectedError,
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
        if _has_unresolved_conflicts(self._workspace):
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
        try:
            published = self._publisher.publish(
                PublicationRequest(
                    issue_number=claim.issue.number,
                    issue_title=claim.issue.title,
                    branch=getattr(prepared, "branch", checkpoint.branch),
                    started_at=attempt_id,
                    decision=decision,
                    check_command="not rerun during recovery retry",
                    check_exit_code=None,
                    review_cycles=0,
                    review_findings="not rerun during recovery retry",
                    details=details,
                    base_branch=profile.base_branch,
                    note_commit_sha=note_sha,
                )
            )
        except Exception as error:
            return False, (
                f"recovery retry for attempt {attempt_id} not fulfilled:"
                f" publication failed ({_exception_detail(error)});"
                " the hold remains."
                f" {hold_suffix}"
            )
        outcome = published.outcome
        comment_posted = getattr(published, "comment_posted", True)
        if not comment_posted:
            return False, (
                f"recovery retry for attempt {attempt_id} not fulfilled:"
                " publication is unconfirmed (the result comment could not be"
                " verified); the hold remains."
                f" {hold_suffix}"
            )
        try:
            self._release(claim, outcome)
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
                getattr(prepared, "branch", checkpoint.branch),
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
        branch = getattr(prepared, "branch", checkpoint.branch)
        marker = attempt_marker(attempt_id)
        comment_body = release_comment_body(
            issue_number=issue_number,
            attempt_id=attempt_id,
            actual_outcome=actual_outcome.value,
            failure_reason=failure_reason,
            saved_at=saved_at,
            branch=branch,
        )
        confirmed = self._confirm_release_comment(
            claim, marker, comment_body, attempt_id
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
        self, checkpoint: AttemptCheckpoint, claim: Claim, prepared: object
    ) -> tuple[CompletionDecision, str, str | None, bool]:
        """Infer the safe republication decision without invoking the model."""

        archive = self._archive_for(claim.issue.number, checkpoint.started_at)
        commits_added = getattr(self._workspace, "commits_added", None)
        commits = (
            tuple(commits_added(prepared)) if callable(commits_added) else ()
        )
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
            if _is_handoff_recovery(commits):
                base_revision = getattr(prepared, "base_revision", "")
                validation = _validate_handoff_note(
                    commits,
                    claim.issue.number,
                    self._workspace.working_directory,
                    base_revision if isinstance(base_revision, str) else "",
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
                    return (decision, details, commits[0].revision, has_commits)
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
        self, checkpoint: AttemptCheckpoint, claim: Claim, prepared: object
    ) -> tuple[AttemptOutcome, str]:
        """Infer the actual outcome recorded by ``recovery release``."""

        commits_added = getattr(self._workspace, "commits_added", None)
        try:
            commits = (
                tuple(commits_added(prepared)) if callable(commits_added) else ()
            )
        except Exception:
            commits = ()
        if _is_handoff_recovery(commits):
            base_revision = getattr(prepared, "base_revision", "")
            try:
                validation = _validate_handoff_note(
                    commits,
                    claim.issue.number,
                    self._workspace.working_directory,
                    base_revision if isinstance(base_revision, str) else "",
                )
            except Exception:
                validation = None
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
        self, claim: Claim, marker: str, body: str, attempt_id: str
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
                    check_command="not run during recovery release",
                    check_exit_code=None,
                    review_cycles=0,
                    review_findings="not run during recovery release",
                    details=body,
                    base_branch="main",
                )
            )
        except Exception:
            return False
        return bool(getattr(published, "comment_posted", True))

    def _recovery_conflict_reason(
        self, claim: Claim, checkpoint: AttemptCheckpoint
    ) -> str | None:
        """Detect conflicting human changes or ambiguous state before a retry.

        Returns a hold reason, or ``None`` when no conflict blocks the retry.
        A fetch failure is ambiguous (the hold remains); a confidently
        observed human change (assignment or labels moved by someone else) is
        conflicting (the retry must not overwrite it).
        """

        fetcher = _tracker_method(self._tracker, "fetch_issue") or _tracker_method(
            self._tracker, "get_issue"
        )
        if fetcher is None:
            return None
        try:
            fresh = fetcher(checkpoint.issue_number)
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
        probe = _tracker_method(self._tracker, "is_self_assigned")
        if probe is not None:
            try:
                self_assigned = bool(probe(fresh))
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
                        f" human changes (assigned to {holders}); the retry"
                        " must not overwrite them and requires operator repair."
                    )
                return (
                    f"issue #{checkpoint.issue_number} shows conflicting human"
                    " changes (it is no longer assigned to this agent); the"
                    " retry must not overwrite them and requires operator repair."
                )
        return None

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
            if not self._workspace_is_dirty():
                return None
            reason = (
                "the working tree holds unexplained dirty or untracked work"
                " with no active attempt; manual repair is required before intake"
            )
            return {
                "attempt_id": None,
                "issue_number": None,
                "phase": None,
                "outcome": None,
                "branch": None,
                "workspace": str(
                    getattr(self._workspace, "working_directory", "")
                ),
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
            "workspace": str(getattr(self._workspace, "working_directory", "")),
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

        Missing tracker support degrades to ``False`` (the assignment is
        then treated as ordinary ineligibility); a failed identity read
        propagates as ambiguity with the priority preserved.
        """

        probe = _tracker_method(self._tracker, "is_self_assigned")
        if probe is None:
            return False
        return bool(probe(found))

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
                next_issue=store.next_issue_snapshot(),
                stop_plan=store.stop_plan_snapshot(),
                recovery=self._recovery_snapshot_locked(checkpoint),
                handoff=store.handoff_snapshot(),
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
        publication_started_at: datetime | None = None
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
            rebase_conflicts = tuple(getattr(prepared, "rebase_conflicts", None) or ())
            prepared_detail = (
                f"branch={getattr(prepared, 'branch', None)}; base_branch={profile.base_branch}"
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
                )
                comment_posted = getattr(published, "comment_posted", True)
            else:
                self._wire_attempt_runner_handoff()
                evidence = self._attempt_runner(claim, profile, prepared)
                publication_started_at = self._clock()
                published = self._publish_bounded(
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
            publication_started_at = self._clock()
            published = self._publish_rebase_conflict(
                claim, checkpoint.started_at, prepared, profile, error,
                timeout=self._handoff_publish_budget(),
            )
            outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
            comment_posted = getattr(published, "comment_posted", True)
        except Exception as error:
            self._event_log(
                "attempt_exception", _exception_detail(error), level="ERROR", issue_number=claim.issue.number
            )
            publication_started_at = self._clock()
            published = self._publish_terminal(
                claim, checkpoint.started_at, prepared, profile, outcome,
                timeout=self._handoff_publish_budget(),
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
                published = self._publish_terminal(
                    claim, checkpoint.started_at, prepared, profile, outcome,
                    timeout=self._handoff_publish_budget(),
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
                    self._release(claim, outcome)
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
                            checkpoint, claim.issue.number, getattr(
                                prepared, "branch", f"agent/issue-{claim.issue.number}"
                            ) if prepared is not None else f"agent/issue-{claim.issue.number}",
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
                retained = self._attempt_state.read()
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
        fetcher = _tracker_method(self._tracker, "fetch_issue")
        if fetcher is None:
            raise ControlStoreError(
                "The tracker cannot revalidate the prioritized issue"
            )
        try:
            fresh = fetcher(target_number)
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
        reason = ineligibility_reason(fresh, target_number)
        if reason is not None:
            if fresh is not None and self._is_self_assignment(fresh):
                # Our own assignment is an ambiguous prior claim (the
                # target was unassigned at acceptance and only this
                # process claims through the serialized boundary), never
                # lost eligibility: hold it for reconciliation instead of
                # failing the priority and falling back to FIFO.
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
        claim_verified = _tracker_method(self._tracker, "claim_verified")
        if claim_verified is None:
            raise ControlStoreError(
                "The tracker cannot claim the prioritized issue"
            )
        try:
            claim = claim_verified(fresh)
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
        checkpoint = self._attempt_state.start(
            issue_number=claim.issue.number, branch=f"agent/issue-{claim.issue.number}"
        )
        self._attempt_processing = True
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
            return self._publisher.publish(request)
        return self._publisher.publish(request, timeout=budget)

    def _handoff_completion_cause(
        self,
        snapshot: dict,
        *,
        issue_number: int,
        outcome: AttemptOutcome,
        prepared: object | None,
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
        prepared: object | None,
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
        prepared: object | None,
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
        self, issue_number: int, prepared: object | None
    ) -> tuple[bool, str]:
        """Whether the branch carries a valid ``operator_request`` note.

        Without the prepared working tree (a crash replay after every other
        boundary went durable) prior validation is trusted: only a begun
        handoff with a handoff outcome reaches here.
        """

        if prepared is None:
            return True, "not rechecked after a crash replay"
        try:
            commits = getattr(self._workspace, "commits_added")(prepared)
            working_directory = getattr(self._workspace, "working_directory")
        except Exception as error:
            return False, f"the branch state could not be read ({_exception_detail(error)})"
        try:
            validation = _validate_handoff_note(
                commits,
                issue_number,
                working_directory,
                getattr(prepared, "base_revision", ""),
            )
        except Exception as error:
            return False, f"the handoff note could not be validated ({_exception_detail(error)})"
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
        publication_started_at: datetime | None = None
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
                note_commit_sha = recovered_commits[0].revision
            publication_started_at = self._clock()
            published = self._publish_bounded(
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
            publication_started_at = self._clock()
            published = self._publish_rebase_conflict(
                claim, checkpoint.started_at, prepared, profile, error,
                timeout=self._handoff_publish_budget(),
            )
            outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
            comment_posted = getattr(published, "comment_posted", True)
        except Exception as error:
            self._event_log(
                "attempt_exception", _exception_detail(error), level="ERROR", issue_number=claim.issue.number
            )
            publication_started_at = self._clock()
            published = self._publish_terminal(
                claim, checkpoint.started_at, prepared, profile, AttemptOutcome.INFRASTRUCTURE_ERROR,
                timeout=self._handoff_publish_budget(),
            )
            outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
            comment_posted = getattr(published, "comment_posted", True)
        except BaseException as error:
            self._event_log(
                "attempt_exception", _exception_detail(error), level="ERROR", issue_number=claim.issue.number
            )
            try:
                publication_started_at = self._clock()
                published = self._publish_terminal(
                    claim, checkpoint.started_at, prepared, profile, AttemptOutcome.INFRASTRUCTURE_ERROR,
                    timeout=self._handoff_publish_budget(),
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
                    self._release(claim, outcome)
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
                            checkpoint, claim.issue.number, getattr(
                                prepared, "branch", checkpoint.branch
                            ) if prepared is not None else checkpoint.branch,
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

    def _publish_terminal(
        self,
        claim: Claim,
        started_at: str,
        prepared: object | None,
        profile: RepositoryProfile | None,
        outcome: AttemptOutcome,
        details: str = "Repository profile could not be loaded or the attempt could not start.",
        timeout: float | None = None,
    ):
        branch = getattr(prepared, "branch", f"agent/issue-{claim.issue.number}")
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
        )
        if timeout is None:
            return self._publisher.publish(request)
        return self._publisher.publish(request, timeout=timeout)

    def _publish_rebase_conflict(
        self,
        claim: Claim,
        started_at: str,
        prepared: object | None,
        profile: RepositoryProfile | None,
        error: RebaseConflictError,
        timeout: float | None = None,
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


def _tracker_method(tracker: object, name: str) -> Callable[..., object] | None:
    """Return the tracker's ``name`` method, if it provides a callable one."""

    probe = getattr(tracker, name, None)
    return probe if callable(probe) else None


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
    try:
        return bool(probe())
    except Exception:
        # An unreadable worktree is not a clean one: fail safe so setup
        # never runs against a tree whose state could not be verified.
        return True


def _conflict_resolution_files(workspace: object, prepared: object) -> tuple[str, ...]:
    """The files the model must resolve, from live probes or the prepare snapshot."""

    probe = getattr(workspace, "conflicted_files", None)
    if callable(probe):
        try:
            files = tuple(str(path) for path in probe() if str(path).strip())
        except Exception:
            files = ()
        if files:
            return files
    snapshot = getattr(prepared, "rebase_conflicts", None) or ()
    return tuple(str(path) for path in snapshot if str(path).strip())


def _resolution_problems(workspace: object) -> tuple[str, ...]:
    """Deterministic post-model verification that a left-in-place rebase resolved."""

    problems: list[str] = []
    probe = getattr(workspace, "rebase_resolution_problems", None)
    if callable(probe):
        try:
            problems.extend(str(problem) for problem in probe())
        except Exception as error:
            # Verification itself failed: fall back to abort-and-restore
            # rather than running setup against an unverified tree.
            problems.append(f"Conflict-resolution verification failed: {error}.")
    if _has_unresolved_conflicts(workspace) and not problems:
        problems.append("A rebase or merge is still in progress or paths remain unmerged.")
    return tuple(problems)


def _format_conflicted_files(files: tuple[str, ...]) -> str:
    """Render the conflicted file list for event details, or "" when unknown."""

    if not files:
        return ""
    return f" Conflicting files: {', '.join(files)}."


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
    {"cost_soft_threshold", "cost_hard_limit", "turn_limit", "time_limit", "operator_request"}
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


def _check_recovery_request_id(request_id: object) -> None:
    """Validate a recovery request ID without touching control state."""

    if not isinstance(request_id, str) or not request_id.strip():
        from simple_coding_agent.control import RequestIdError

        raise RequestIdError("Request ID must be a non-empty string")
    if len(request_id) > 128:
        from simple_coding_agent.control import RequestIdError

        raise RequestIdError("Request ID must be at most 128 characters")
