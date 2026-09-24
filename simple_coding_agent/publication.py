"""Orchestrator-owned publication of implementation-attempt results."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Protocol

from simple_coding_agent.completion import AttemptOutcome, CompletionDecision, PublicationPath
from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.git_workspace import GitWorkspace, GitWorkspaceError


_SUCCEEDED_BUT_UNPUBLISHED = "Implementation succeeded but publication failed."


@dataclass(frozen=True)
class PullRequest:
    """The small, stable portion of a GitHub pull request used by an attempt."""

    number: int
    url: str


class PublicationTransport(Protocol):
    """GitHub operations that remain callable independently during recovery."""

    def find_pull_request(self, repository: str, head: str) -> PullRequest | None: ...

    def create_pull_request(
        self, repository: str, head: str, base: str, title: str, body: str
    ) -> PullRequest: ...

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


@dataclass(frozen=True)
class PublicationResult:
    """The verified remote result, suitable for the completion and cleanup phases."""

    outcome: AttemptOutcome
    branch_url: str | None
    pull_request: PullRequest | None
    comment_posted: bool = True
    details: str = ""


class Publisher:
    """Own push, pull-request, and concise attempt-result comment publication."""

    def __init__(
        self,
        workspace: GitWorkspace,
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

    def publish(self, request: PublicationRequest) -> PublicationResult:
        """Publish the permitted path and always try to record its outcome.

        A branch is published only for a complete candidate or partial work.
        PR creation is deliberately limited to a locally complete candidate.
        """

        deadline = self._monotonic() + self._publish_timeout
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

        if outcome is None:
            outcome = AttemptOutcome.INFRASTRUCTURE_ERROR
        comment_posted = self._post_result(request, outcome, branch_url, pull_request, request.details)
        return PublicationResult(outcome, branch_url, pull_request, comment_posted, request.details)

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

    def _post_result(
        self,
        request: PublicationRequest,
        outcome: AttemptOutcome,
        branch_url: str | None,
        pull_request: PullRequest | None,
        details: str,
    ) -> bool:
        marker = f"<!-- agent-attempt: {request.started_at} -->"
        try:
            if self._github.find_attempt_comment(self._repository, request.issue_number, marker):
                return True
            self._github.add_comment(
                self._repository,
                request.issue_number,
                _result_comment(request, outcome, branch_url, pull_request, details, marker),
            )
            return True
        except Exception:
            # The local result remains authoritative when GitHub is unavailable.
            return False


def _pull_request_body(request: PublicationRequest) -> str:
    return (
        f"Closes #{request.issue_number}\n\n"
        "## Summary\n"
        f"{request.details}\n\n"
        "## Evidence\n"
        f"- Check: exit code {request.check_exit_code}\n"
        f"- Review cycles: {request.review_cycles}/2\n\n"
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
) -> str:
    branch = f"[{request.branch}]({branch_url})" if branch_url else "no branch created"
    pr = f"[#{pull_request.number}]({pull_request.url})" if pull_request else "none"
    commit_line = (
        f"**Handoff note commit**: `{request.note_commit_sha}`\n"
        if outcome is AttemptOutcome.HANDOFF and request.note_commit_sha
        else ""
    )
    return (
        f"## Agent Attempt Result: {outcome.value}\n\n"
        f"**Branch**: {branch}\n"
        f"**PR**: {pr}\n"
        f"{commit_line}\n"
        "### Evidence\n"
        f"- Check command: `{request.check_command}` → exit code {request.check_exit_code}\n"
        f"- Review cycles: {request.review_cycles}/2\n"
        f"- Review findings: {request.review_findings}\n\n"
        "### Details\n"
        f"{details}\n\n"
        f"{marker}"
    )
