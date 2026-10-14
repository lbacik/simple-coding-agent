"""Orchestrator-owned publication of implementation-attempt results."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Protocol

from simple_coding_agent.completion import AttemptOutcome, CompletionDecision, PublicationPath
from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.attempt_workspace import BranchPusher
from simple_coding_agent.git_workspace import GitWorkspaceError
from simple_coding_agent.recovery import REQUEUE_SOURCE_HUMAN, requeue_source_marker


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
                    f"agent: {request.issue_title}",
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
        request: PublicationRequest,
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
        request: PublicationRequest,
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

    def _merge_failure(self, request: PublicationRequest, error: Exception) -> str:
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
