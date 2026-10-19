"""Orchestrator-owned publication of implementation-attempt results."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Protocol

from simple_coding_agent.completion import AttemptOutcome, CompletionDecision, PublicationPath
from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.attempt_workspace import BranchPusher
from simple_coding_agent.git_workspace import GitWorkspaceError
from simple_coding_agent.recovery import (
    CONFLICT_ESCALATION_MARKER,
    CONFLICT_OUTCOME_NO_CHANGE,
    CONFLICT_OUTCOME_SUCCESS,
    REQUEUE_SOURCE_HUMAN,
    conflict_attempt_marker,
    requeue_source_marker,
)


_SUCCEEDED_BUT_UNPUBLISHED = "Implementation succeeded but publication failed."


#: The issue label opting one implementation into auto-merge when the
#: operator runs with ``PR_AUTO_MERGE=label``. Like ``ready-for-agent`` it
#: sits on the collaborator trust boundary: only writers can apply it. It is
#: read once, when the issue is claimed, so a label added while the attempt
#: runs (or after it finishes) never enables that attempt's merge.
AUTO_MERGE_LABEL = "agent-auto-merge"

#: The merge result reported when ``label`` mode skips an unopted attempt.
#: The PR is still created; no merge-related GitHub call runs.
AUTO_MERGE_SKIPPED = (
    f"auto-merge skipped (issue not labelled `{AUTO_MERGE_LABEL}` at claim time)"
)

_AUTO_MERGE_MODES = frozenset({"false", "true", "label"})


def _normalize_auto_merge(value: bool | str) -> str:
    """Normalize the auto-merge mode, accepting the legacy boolean form."""

    if isinstance(value, bool):
        return "true" if value else "false"
    normalized = str(value).strip().lower()
    if normalized not in _AUTO_MERGE_MODES:
        raise ValueError(f"auto_merge must be one of {sorted(_AUTO_MERGE_MODES)}")
    return normalized


@dataclass(frozen=True)
class PullRequest:
    """The small, stable portion of a GitHub pull request used by an attempt."""

    number: int
    url: str
    node_id: str = ""
    draft: bool = False
    mergeable: str = "UNKNOWN"
    base_ref: str = ""
    head_sha: str = ""


class PublicationTransport(Protocol):
    """GitHub operations that remain callable independently during recovery."""

    def find_pull_request(self, repository: str, head: str) -> PullRequest | None: ...

    def create_pull_request(
        self, repository: str, head: str, base: str, title: str, body: str
    ) -> PullRequest: ...

    def enable_auto_merge(
        self, repository: str, pull_request: PullRequest, merge_method: str
    ) -> None: ...

    def merge_pull_request(
        self, repository: str, pull_request: PullRequest, merge_method: str
    ) -> None: ...

    def find_attempt_comment(self, repository: str, issue_number: int, marker: str) -> bool: ...

    def add_comment(self, repository: str, issue_number: int, body: str) -> None: ...


@dataclass(frozen=True)
class PublicationRequest:
    """Evidence and identity required to publish one completed local attempt."""

    issue_number: int
    issue_title: str
    branch: str
    started_at: str
    decision: CompletionDecision
    check_command: str
    check_exit_code: int | None
    review_cycles: int
    review_findings: str
    details: str
    base_branch: str
    note_commit_sha: str | None = None
    auto_merge_opt_in: bool = False
    reviews_performed: int = 0
    requeue_source: str = REQUEUE_SOURCE_HUMAN


@dataclass(frozen=True)
class PublicationResult:
    """The verified remote result, suitable for the completion and cleanup phases."""

    outcome: AttemptOutcome
    branch_url: str | None
    pull_request: PullRequest | None
    comment_posted: bool = True
    details: str = ""
    merge_result: str | None = None
    #: The conflict-attempt marker outcome actually published (``""`` for
    #: ordinary attempts). A success-path push that left the remote head
    #: unchanged publishes ``no_change`` instead of ``success``.
    conflict_outcome: str = ""
    #: Whether the published conflict comment escalated to a human.
    escalated: bool = False


@dataclass(frozen=True)
class ConflictPublicationRequest:
    """Evidence and identity required to publish one conflict attempt.

    ``conflict_outcome`` is the machine-readable marker outcome (``success``,
    ``skipped``, or a failure kind); ``outcome`` is the lifecycle outcome it
    maps to. ``push_branch`` is True only for the success path (clean rebase,
    or a model resolution that preserved the commit count, plus a green
    check): every other outcome leaves the remote branch untouched so the
    pull request stays ``CONFLICTING``. ``pull_request`` is the entry-checked
    open pull request, reused here so publication never creates one.
    ``conflicted_files`` names the files the rebase conflicted in,
    ``extra_files`` the files a model resolution changed outside them, and
    ``fix_commits`` the fix commits it added on top. The comment carries no
    requeue-source marker, keeping the attempt transparent to the
    automatic-continuation count.

    ``conflict_index``/``conflict_budget`` render the ``k/N`` standing line
    every conflict result carries (``conflict_index`` is the resulting
    consecutive-failure count, or 0 once a success or skip resets the
    streak; 0 budget omits the line). ``escalate`` marks the attempt whose
    failure reaches the budget: its own comment becomes the escalation
    comment. ``conflict_prior`` is the streak before this attempt, so a
    success-path push that leaves the remote head unchanged can still be
    reported (and escalated) as ``no_change``. ``failure_output`` is the
    tail of the last check output (or the failure reason) quoted in the
    escalation's collapsed section.
    """

    issue_number: int
    issue_title: str
    branch: str
    started_at: str
    base_branch: str
    outcome: AttemptOutcome
    conflict_outcome: str
    reason: str
    details: str
    push_branch: bool
    pull_request: PullRequest | None
    check_command: str = "not run"
    check_exit_code: int | None = None
    conflicted_files: tuple[str, ...] = ()
    auto_merge_opt_in: bool = False
    extra_files: tuple[str, ...] = ()
    fix_commits: tuple[str, ...] = ()
    conflict_index: int = 0
    conflict_budget: int = 0
    escalate: bool = False
    conflict_prior: int = 0
    failure_output: str = ""


class Publisher:
    """Own push, pull-request, and concise attempt-result comment publication."""

    def __init__(
        self,
        workspace: BranchPusher,
        github: PublicationTransport,
        repository: str,
        *,
        max_retries: int = 3,
        publish_timeout: float = 120,
        attempt_state: AttemptStateStore | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        event_log: Callable[..., None] = (
            lambda event, detail="", level="INFO", issue_number=None: None
        ),
        auto_merge: bool | str = False,
        merge_method: str = "merge",
    ) -> None:
        self._workspace = workspace
        self._github = github
        self._repository = repository
        self._max_retries = max_retries
        self._publish_timeout = publish_timeout
        self._attempt_state = attempt_state
        self._sleeper = sleeper
        self._monotonic = monotonic
        self._event_log = event_log
        self._auto_merge = _normalize_auto_merge(auto_merge)
        self._merge_method = merge_method

    def publish(
        self, request: PublicationRequest, *, timeout: float | None = None
    ) -> PublicationResult:
        """Publish the permitted path and always try to record its outcome.

        A branch is published only for a complete candidate or partial work.
        PR creation is deliberately limited to a locally complete candidate.
        ``timeout`` overrides the configured publish timeout for this call so
        a handoff publication can be bounded by its remaining acceptance
        budget (at most 120 seconds after the valid local handoff and 360
        seconds from acceptance); ``None`` keeps the configured timeout.
        """

        effective_timeout = self._publish_timeout if timeout is None else timeout
        deadline = self._monotonic() + effective_timeout
        outcome = request.decision.outcome
        branch_url: str | None = None
        pull_request: PullRequest | None = None
        needs_push = request.decision.publication_eligible or (
            outcome in (AttemptOutcome.INCOMPLETE, AttemptOutcome.HANDOFF)
            and request.decision.publication_path is PublicationPath.PARTIAL
        )
        if needs_push:
            try:
                checkpoint = self._attempt_state.read() if self._attempt_state is not None else None
                if checkpoint is None or checkpoint.phase is not AttemptPhase.PUBLISHING:
                    self._check_deadline(deadline)
                    self._workspace.push_attempt_branch(
                        request.branch, max_retries=self._max_retries, deadline=deadline
                    )
                    self._check_deadline(deadline)
                    self._mark_publishing()
                branch_url = f"https://github.com/{self._repository}/tree/{request.branch}"
            except (GitWorkspaceError, TimeoutError) as error:
                push_failure = f"Partial branch push failed: {type(error).__name__}: {error}"
                self._event_log(
                    "publication_failed",
                    push_failure,
                    level="ERROR",
                    issue_number=request.issue_number,
                )
                if outcome is AttemptOutcome.INCOMPLETE and not request.decision.publication_eligible:
                    # A failed final authoritative check stays reported as
                    # incomplete; the push failure is secondary and must not
                    # hide the evaluator's reason.
                    details = "; ".join((*request.decision.reasons, push_failure))
                else:
                    outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
                    details = _SUCCEEDED_BUT_UNPUBLISHED
                comment_posted = self._post_result(request, outcome, branch_url, None, details)
                return PublicationResult(outcome, branch_url, None, comment_posted, details)

        if request.decision.publication_eligible:
            try:
                pull_request = self._ensure_pull_request(request, deadline)
            except Exception:  # Transport implementations normalize only their own failures.
                outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
                details = _SUCCEEDED_BUT_UNPUBLISHED
                comment_posted = self._post_result(request, outcome, branch_url, None, details)
                return PublicationResult(outcome, branch_url, None, comment_posted, details)
            outcome = AttemptOutcome.COMPLETE

        merge_result: str | None = None
        if outcome is AttemptOutcome.COMPLETE and pull_request is not None:
            if self._auto_merge == "true" or (
                self._auto_merge == "label" and request.auto_merge_opt_in
            ):
                # Merging is a follow-up to publication: it never redefines
                # completion, so any failure stays reported on a complete outcome.
                merge_result = self._maybe_auto_merge(request, pull_request, deadline)
            elif self._auto_merge == "label":
                # Label-gated mode without a claim-time opt-in: the PR stands
                # for human review and no merge-related GitHub call runs.
                merge_result = AUTO_MERGE_SKIPPED

        if outcome is None:
            outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
        comment_posted = self._post_result(
            request, outcome, branch_url, pull_request, request.details, merge_result
        )
        return PublicationResult(
            outcome, branch_url, pull_request, comment_posted, request.details, merge_result
        )

    def find_pull_request(self, head: str) -> PullRequest | None:
        """Return the open pull request for an attempt branch, if any.

        Used by the conflict-attempt entry check before any git work runs.
        """

        return self._github.find_pull_request(self._repository, head)

    def publish_conflict(
        self, request: ConflictPublicationRequest, *, timeout: float | None = None
    ) -> PublicationResult:
        """Publish one conflict attempt: push only on success, then comment.

        The rebased branch is pushed only when the caller verified a clean
        rebase and a green check (``push_branch``); every other outcome posts
        its failure (or skipped) result while leaving the remote branch
        untouched, so GitHub cannot merge an unverified resolution. Auto-merge
        is re-enabled on the pushed branch under the same mode as ordinary
        publication; like there, a merge-step failure is reported but never
        redefines the outcome. The result comment carries the
        conflict-attempt marker and no requeue-source marker.
        """

        effective_timeout = self._publish_timeout if timeout is None else timeout
        deadline = self._monotonic() + effective_timeout
        outcome = request.outcome
        conflict_outcome = request.conflict_outcome
        conflict_index = request.conflict_index
        escalate = request.escalate
        branch_url: str | None = None
        pushed_sha: str | None = None
        details = request.details
        if request.push_branch:
            try:
                checkpoint = self._attempt_state.read() if self._attempt_state is not None else None
                if checkpoint is None or checkpoint.phase is not AttemptPhase.PUBLISHING:
                    self._check_deadline(deadline)
                    pushed_sha = self._workspace.push_attempt_branch(
                        request.branch, max_retries=self._max_retries, deadline=deadline
                    )
                    self._check_deadline(deadline)
                    self._mark_publishing()
                branch_url = f"https://github.com/{self._repository}/tree/{request.branch}"
            except (GitWorkspaceError, TimeoutError) as error:
                push_failure = f"Rebased branch push failed: {type(error).__name__}: {error}"
                self._event_log(
                    "publication_failed",
                    push_failure,
                    level="ERROR",
                    issue_number=request.issue_number,
                )
                outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
                conflict_outcome = "push_failed"
                details = "; ".join((request.details, push_failure)) if request.details else push_failure
                conflict_index, escalate = _conflict_failure_standing(request)
                comment_posted = self._post_conflict_result(
                    request,
                    outcome,
                    conflict_outcome,
                    branch_url,
                    None,
                    details,
                    None,
                    None,
                    conflict_index=conflict_index,
                    escalate=escalate,
                )
                return PublicationResult(
                    outcome,
                    branch_url,
                    request.pull_request,
                    comment_posted,
                    details,
                    None,
                    conflict_outcome,
                    escalate,
                )
            if (
                conflict_outcome == CONFLICT_OUTCOME_SUCCESS
                and pushed_sha
                and request.pull_request is not None
                and request.pull_request.head_sha
                and pushed_sha == request.pull_request.head_sha
            ):
                # The green push changed nothing on the remote branch: report
                # it as ``no_change`` (a failure toward the budget) while the
                # verified green outcome stands.
                conflict_outcome = CONFLICT_OUTCOME_NO_CHANGE
                conflict_index, escalate = _conflict_failure_standing(request)

        merge_result: str | None = None
        if request.push_branch and branch_url is not None and request.pull_request is not None:
            if self._auto_merge == "true" or (
                self._auto_merge == "label" and request.auto_merge_opt_in
            ):
                merge_result = self._maybe_auto_merge(request, request.pull_request, deadline)
            elif self._auto_merge == "label":
                merge_result = AUTO_MERGE_SKIPPED

        comment_posted = self._post_conflict_result(
            request,
            outcome,
            conflict_outcome,
            branch_url,
            request.pull_request,
            details,
            merge_result,
            pushed_sha,
            conflict_index=conflict_index,
            escalate=escalate,
        )
        return PublicationResult(
            outcome,
            branch_url,
            request.pull_request,
            comment_posted,
            details,
            merge_result,
            conflict_outcome,
            escalate,
        )

    def _mark_publishing(self) -> None:
        """Durably record a verified branch write before any PR write can begin."""

        if self._attempt_state is None:
            return
        checkpoint = self._attempt_state.read()
        if checkpoint is None:
            raise GitWorkspaceError("Attempt checkpoint is missing during publication")
        if checkpoint.phase is AttemptPhase.PUBLISHING:
            return
        if checkpoint.phase is not AttemptPhase.PUSHING:
            raise GitWorkspaceError("Attempt is not ready to publish")
        self._attempt_state.transition(AttemptPhase.PUBLISHING)

    def _ensure_pull_request(self, request: PublicationRequest, deadline: float) -> PullRequest:
        existing = self._github.find_pull_request(self._repository, request.branch)
        if existing is not None:
            return existing
        body = _pull_request_body(request)
        for attempt in range(self._max_retries):
            self._check_deadline(deadline)
            try:
                return self._github.create_pull_request(
                    self._repository,
                    request.branch,
                    request.base_branch,
                    request.issue_title,
                    body,
                )
            except Exception:
                existing = self._github.find_pull_request(self._repository, request.branch)
                if existing is not None:
                    return existing
                if attempt + 1 == self._max_retries:
                    raise
                self._sleeper(min(2 * 2**attempt, 30))
        raise AssertionError("unreachable")

    def _check_deadline(self, deadline: float) -> None:
        if self._monotonic() >= deadline:
            raise TimeoutError("PUBLISH_TIMEOUT exceeded")

    def _maybe_auto_merge(
        self,
        request: PublicationRequest | ConflictPublicationRequest,
        pull_request: PullRequest,
        deadline: float,
    ) -> str:
        """Enable native auto-merge, falling back to a direct merge when ready.

        Every path returns a comment-ready result; failures never raise, so
        the attempt stays ``complete`` and recovery re-runs stay idempotent.
        """

        try:
            self._check_deadline(deadline)
            self._github.enable_auto_merge(
                self._repository, pull_request, self._merge_method
            )
            result = f"auto-merge enabled ({self._merge_method})"
            self._event_log(
                "auto_merge_enabled", result, level="INFO", issue_number=request.issue_number
            )
            return result
        except Exception as error:
            if _is_already_merged(error):
                self._event_log(
                    "pull_request_merged",
                    "pull request was already merged",
                    level="INFO",
                    issue_number=request.issue_number,
                )
                return "merged"
            if _is_already_enabled(error):
                result = f"auto-merge enabled ({self._merge_method})"
                self._event_log(
                    "auto_merge_enabled",
                    result,
                    level="INFO",
                    issue_number=request.issue_number,
                )
                return result
            if _is_already_mergeable(error):
                return self._merge_directly(request, pull_request, deadline)
            return self._merge_failure(request, error)

    def _merge_directly(
        self,
        request: PublicationRequest | ConflictPublicationRequest,
        pull_request: PullRequest,
        deadline: float,
    ) -> str:
        """Merge an already-mergeable PR after auto-merge was refused."""

        try:
            self._check_deadline(deadline)
            self._github.merge_pull_request(
                self._repository, pull_request, self._merge_method
            )
            self._event_log(
                "pull_request_merged",
                f"merged with method {self._merge_method}",
                level="INFO",
                issue_number=request.issue_number,
            )
            return "merged"
        except Exception as error:
            if _is_already_merged(error):
                return "merged"
            return self._merge_failure(request, error)

    def _merge_failure(
        self, request: PublicationRequest | ConflictPublicationRequest, error: Exception
    ) -> str:
        reason = str(error).strip() or type(error).__name__
        result = f"auto-merge failed: {reason}"
        self._event_log(
            "auto_merge_failed", result, level="WARNING", issue_number=request.issue_number
        )
        return result

    def _post_result(
        self,
        request: PublicationRequest,
        outcome: AttemptOutcome,
        branch_url: str | None,
        pull_request: PullRequest | None,
        details: str,
        merge_result: str | None = None,
    ) -> bool:
        marker = f"<!-- agent-attempt: {request.started_at} -->"
        try:
            if self._github.find_attempt_comment(self._repository, request.issue_number, marker):
                return True
            self._github.add_comment(
                self._repository,
                request.issue_number,
                _result_comment(
                    request, outcome, branch_url, pull_request, details, marker, merge_result
                ),
            )
            return True
        except Exception:
            # The local result remains authoritative when GitHub is unavailable.
            return False

    def _post_conflict_result(
        self,
        request: ConflictPublicationRequest,
        outcome: AttemptOutcome,
        conflict_outcome: str,
        branch_url: str | None,
        pull_request: PullRequest | None,
        details: str,
        merge_result: str | None,
        pushed_sha: str | None,
        *,
        conflict_index: int | None = None,
        escalate: bool | None = None,
    ) -> bool:
        marker = f"<!-- agent-attempt: {request.started_at} -->"
        try:
            if self._github.find_attempt_comment(self._repository, request.issue_number, marker):
                return True
            self._github.add_comment(
                self._repository,
                request.issue_number,
                _conflict_result_comment(
                    request,
                    outcome,
                    conflict_outcome,
                    branch_url,
                    pull_request,
                    details,
                    marker,
                    merge_result,
                    pushed_sha,
                    conflict_index=(
                        request.conflict_index if conflict_index is None else conflict_index
                    ),
                    escalate=request.escalate if escalate is None else escalate,
                ),
            )
            return True
        except Exception:
            # The local result remains authoritative when GitHub is unavailable.
            return False


def _conflict_failure_standing(request: ConflictPublicationRequest) -> tuple[int, bool]:
    """The ``(index, escalate)`` for a conflict failure discovered at publish time.

    A success-path outcome that turns into a failure only once the push is
    observed (``push_failed``, or ``no_change`` when the remote head did not
    move) counts one past the pre-publish streak; it escalates when that
    count reaches the operator budget. A zero budget never escalates.
    """

    index = request.conflict_prior + 1
    return index, request.conflict_budget > 0 and index >= request.conflict_budget


def _review_lines(request: PublicationRequest) -> list[str]:
    """Report reviews performed separately from review repair cycles.

    ``review_cycles`` is ``max(0, reviews - 1)`` (repair cycles), so 0 and 1
    reviews used to look identical; the explicit "not run" line keeps a
    missing ``code-review`` visible instead of reading as satisfied.
    """

    lines = [
        f"- Reviews performed: {request.reviews_performed}",
        f"- Review repair cycles: {request.review_cycles}/2",
    ]
    if request.reviews_performed == 0:
        lines.append("- code-review was not run")
    return lines


def _pull_request_body(request: PublicationRequest) -> str:
    return (
        f"Closes #{request.issue_number}\n\n"
        "## Summary\n"
        f"{request.details}\n\n"
        "## Evidence\n"
        f"- Check: exit code {request.check_exit_code}\n"
        + "\n".join(_review_lines(request))
        + "\n\n"
        "---\n"
        "*Generated by simple-coding-agent*"
    )


def _result_comment(
    request: PublicationRequest,
    outcome: AttemptOutcome,
    branch_url: str | None,
    pull_request: PullRequest | None,
    details: str,
    marker: str,
    merge_result: str | None = None,
) -> str:
    branch = f"[{request.branch}]({branch_url})" if branch_url else "no branch created"
    pr = f"[#{pull_request.number}]({pull_request.url})" if pull_request else "none"
    commit_line = (
        f"**Handoff note commit**: `{request.note_commit_sha}`\n"
        if outcome is AttemptOutcome.HANDOFF and request.note_commit_sha
        else ""
    )
    merge_line = f"**Merge**: {merge_result}\n" if merge_result is not None else ""
    review_lines = "\n".join(_review_lines(request))
    source_marker = requeue_source_marker(request.requeue_source)
    return (
        f"## Agent Attempt Result: {outcome.value}\n\n"
        f"**Branch**: {branch}\n"
        f"**PR**: {pr}\n"
        f"{merge_line}"
        f"{commit_line}\n"
        "### Evidence\n"
        f"- Check command: `{request.check_command}` → exit code {request.check_exit_code}\n"
        f"{review_lines}\n"
        f"- Review findings: {request.review_findings}\n\n"
        "### Details\n"
        f"{details}\n\n"
        f"{marker}\n"
        f"{source_marker}"
    )


def _short_revisions(revisions: tuple[str, ...], *, length: int = 12) -> tuple[str, ...]:
    """Readable commit prefixes for prose; the marker always carries full revisions."""

    return tuple(
        revision[:length] if len(revision) > length else revision for revision in revisions
    )


#: Cap for the failure text quoted in an escalation comment when no
#: pre-trimmed check-output tail is available. Cut from the front so the
#: most recent reason survives, like the existing result-comment limits.
_ESCALATION_FAILURE_LIMIT = 4000


def _escalation_failure_text(text: str) -> str:
    """Trim a fallback escalation failure reason, keeping the tail visible."""

    if not text.strip():
        return "(no details recorded)"
    if len(text) <= _ESCALATION_FAILURE_LIMIT:
        return text
    return (
        f"...[truncated; showing the last {_ESCALATION_FAILURE_LIMIT} characters]\n"
        f"{text[-_ESCALATION_FAILURE_LIMIT:]}"
    )


def _conflict_result_comment(
    request: ConflictPublicationRequest,
    outcome: AttemptOutcome,
    conflict_outcome: str,
    branch_url: str | None,
    pull_request: PullRequest | None,
    details: str,
    marker: str,
    merge_result: str | None,
    pushed_sha: str | None,
    *,
    conflict_index: int | None = None,
    escalate: bool | None = None,
) -> str:
    """Build the conflict-attempt result comment with its machine-readable marker.

    The header names the conflict outcome (``success``, ``skipped``, or a
    failure kind), not the lifecycle outcome. The comment carries the
    attempt marker for publication dedup and the conflict-attempt marker
    (outcome plus pushed sha), and deliberately no requeue-source marker, so
    the automatic-continuation count stays transparent to conflict attempts.
    Every result carries the ``k/N`` standing line (omitted only when the
    operator budget is zero). The attempt whose failure reaches the budget
    escalates: its comment gains the human-handoff section (the failure
    tail or reason in a collapsed section, a link to the pull request, and
    the hand-back line) plus the hidden escalation marker that stops the
    streak, so removing ``ready-for-human`` requeues with the full budget.
    """

    effective_index = request.conflict_index if conflict_index is None else conflict_index
    effective_escalate = request.escalate if escalate is None else escalate
    if branch_url and pushed_sha:
        branch = f"[{request.branch}]({branch_url}) (sha `{pushed_sha}`)"
    elif branch_url:
        branch = f"[{request.branch}]({branch_url})"
    else:
        branch = f"`{request.branch}` (not pushed)"
    pr = f"[#{pull_request.number}]({pull_request.url})" if pull_request else "none"
    merge_line = f"**Merge**: {merge_result}\n" if merge_result is not None else ""
    standing_line = (
        f"**Conflict attempts**: {effective_index}/{request.conflict_budget}\n"
        if request.conflict_budget > 0
        else ""
    )
    files_line = (
        f"- Conflicted files: {', '.join(request.conflicted_files)}\n"
        if request.conflicted_files
        else ""
    )
    extra_line = (
        f"- Files changed outside the conflicted set: {', '.join(request.extra_files)}\n"
        if request.extra_files
        else ""
    )
    fixes_line = (
        f"- Fix commits: {', '.join(_short_revisions(request.fix_commits))}\n"
        if request.fix_commits
        else ""
    )
    escalation_section = ""
    if effective_escalate:
        failure_text = request.failure_output or _escalation_failure_text(
            details or request.reason
        )
        pr_link = (
            f"[#{pull_request.number}]({pull_request.url})"
            if pull_request is not None
            else "none"
        )
        escalation_section = (
            "### Human handoff\n\n"
            f"The conflict-attempt budget is exhausted "
            f"({effective_index}/{request.conflict_budget} consecutive failures). "
            "The issue is labelled `ready-for-human`.\n\n"
            "<details>\n"
            "<summary>Failure output (tail)</summary>\n\n"
            f"{failure_text}\n\n"
            "</details>\n\n"
            f"Pull request: {pr_link}\n\n"
            f"Remove the `ready-for-human` label to hand the issue back "
            f"for {request.conflict_budget} more attempts.\n\n"
            f"{CONFLICT_ESCALATION_MARKER}\n\n"
        )
    conflict_marker = conflict_attempt_marker(
        conflict_outcome,
        sha=pushed_sha,
        reason=request.reason or None,
        files=request.conflicted_files,
        fix_commits=request.fix_commits,
        extra_files=request.extra_files,
    )
    return (
        f"## Conflict Attempt Result: {conflict_outcome}\n\n"
        f"**Branch**: {branch}\n"
        f"**PR**: {pr}\n"
        f"{merge_line}"
        f"**Lifecycle outcome**: {outcome.value}\n"
        f"{standing_line}\n"
        "### Evidence\n"
        f"- Check command: `{request.check_command}` → exit code {request.check_exit_code}\n"
        f"{files_line}"
        f"{extra_line}"
        f"{fixes_line}\n"
        "### Details\n"
        f"{details}\n\n"
        f"{escalation_section}"
        f"{marker}\n"
        f"{conflict_marker}"
    )


def _error_text(error: Exception) -> str:
    return f"{type(error).__name__}: {error}".lower()


def _is_already_mergeable(error: Exception) -> bool:
    """Whether enabling auto-merge was refused because the PR can merge now.

    An unprotected base branch (``protected branch rules not configured``)
    merges directly: with no required checks to wait for, the agent's own
    final check stands as the gate, and ``_merge_directly`` still reports a
    genuinely unmergeable PR (e.g. conflicts) via ``mergePullRequest``.
    """

    text = _error_text(error)
    return (
        "mergeable" in text
        or "clean status" in text
        or "unstable status" in text
        or "protected branch rules not configured" in text
    )


def _is_already_enabled(error: Exception) -> bool:
    """Whether auto-merge was already enabled on a previous run."""

    text = _error_text(error)
    return "already" in text and ("auto-merge" in text or "auto_merge" in text or "auto merge" in text)


def _is_already_merged(error: Exception) -> bool:
    """Whether the PR was already merged on a previous run."""

    text = _error_text(error)
    return "merg" in text and (
        "already merged" in text or "was merged" in text or "is merged" in text
    )
