"""GitHub boundary for selecting and claiming implementation issues."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime
import json
import os
import re
import time
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from simple_coding_agent.publication import PullRequest
from simple_coding_agent.recovery import (
    count_automatic_continuations,
    count_conflict_failures,
)


READY_FOR_AGENT = "ready-for-agent"
ROUND_FINISHED = "round-finished"
READY_FOR_HUMAN = "ready-for-human"
MERGE_CONFLICT = "merge-conflict"

#: Fixed colour for the auto-created ``merge-conflict`` repository label.
MERGE_CONFLICT_LABEL_COLOR = "d93f0b"
#: Fixed description for the auto-created ``merge-conflict`` repository label.
MERGE_CONFLICT_LABEL_DESCRIPTION = (
    "Agent PR has merge conflicts; queued for a Conflict attempt"
)

#: Published attempt branches, and the issue number each one belongs to. The
#: detector takes the issue number from this branch name, never from
#: ``Closes #N`` in the pull request body, which links nothing off the
#: default branch.
_AGENT_BRANCH_RE = re.compile(r"^agent/issue-(\d+)$")


def conflict_branch_issue_number(head_ref: str) -> int | None:
    """The issue number behind a published attempt branch, if it is one."""

    match = _AGENT_BRANCH_RE.fullmatch(head_ref or "")
    return int(match.group(1)) if match else None


def _peer_login_set(peer_logins: Iterable[str] | None) -> frozenset[str]:
    """Normalize the pool-peer logins whose conflict comments count as the worker's own.

    An explicit constructor value wins; otherwise the ``PEER_LOGINS``
    environment setting (comma- or space-separated, from the pool work) is
    read, so the budget keeps counting when several workers share one
    account. Absent in both places means only the worker's own login
    counts. Empty logins never count (they would match deleted accounts).
    """

    if peer_logins is not None:
        return frozenset(
            login.strip() for login in peer_logins if login and login.strip()
        )
    raw = os.environ.get("PEER_LOGINS", "")
    return frozenset(
        part for part in re.split(r"[,\s]+", raw) if part
    )


def is_conflict_detection_eligible(
    issue: TrackerIssue, *, draft: bool, mergeable: str
) -> bool:
    """Whether the detector may requeue ``issue`` for a conflicting pull request.

    The pull request must be open, non-draft, and report ``CONFLICTING``
    (``UNKNOWN`` only means GitHub is still computing mergeability, so this
    poll skips it); the issue must meet the base queue rules while carrying
    none of the queue labels (an already-queued issue is never relabelled).
    The detector never counts any budget and never removes ``merge-conflict``.
    """

    if draft or mergeable != "CONFLICTING":
        return False
    if not issue.meets_base_queue_rules:
        return False
    return not (
        issue.labels & {READY_FOR_AGENT, ROUND_FINISHED, READY_FOR_HUMAN, MERGE_CONFLICT}
    )


def ineligibility_reason(
    issue: TrackerIssue | None,
    number: int,
    *,
    max_auto_continuations: int = 0,
    auto_continuation_count: int = 0,
    max_conflict_attempts: int = 0,
) -> str | None:
    """Explain why a ``next issue`` target cannot be accepted or claimed.

    Returns ``None`` when the issue is currently eligible under the ordinary
    queue rules (open, unassigned, labelled ``ready-for-agent``, with a
    non-empty body and no open blocker), as an automatic continuation (open,
    unassigned, labelled ``round-finished`` without ``ready-for-agent``, with
    a non-empty body and no open blocker, whose automatic-continuation count
    is below ``max_auto_continuations``), or as a conflict requeue (open,
    unassigned, labelled ``merge-conflict`` without ``ready-for-agent``,
    ``round-finished`` or ``ready-for-human``, with a non-empty body and no
    open blocker, while ``max_conflict_attempts`` is positive); otherwise
    returns a visible reason naming the observed state. An over-budget
    continuation names the limit as ``(k/N)``.
    """

    if issue is None:
        return f"issue #{number} was not found"
    if issue.state != "OPEN":
        return f"issue #{number} is not open (state {issue.state})"
    if READY_FOR_AGENT not in issue.labels:
        if max_auto_continuations > 0 and ROUND_FINISHED in issue.labels:
            if issue.meets_base_queue_rules:
                if auto_continuation_count < max_auto_continuations:
                    return None
                return (
                    f"issue #{number}: automatic continuation limit reached"
                    f" ({auto_continuation_count}/{max_auto_continuations})"
                )
            # A ``round-finished`` issue that fails the base queue rules is
            # reported through the same checks as an ordinary issue below.
        elif READY_FOR_HUMAN in issue.labels:
            return f"issue #{number} requires human implementation (labelled `{READY_FOR_HUMAN}`)"
        elif (
            max_conflict_attempts > 0
            and MERGE_CONFLICT in issue.labels
            and READY_FOR_HUMAN not in issue.labels
            and ROUND_FINISHED not in issue.labels
        ):
            if issue.meets_base_queue_rules:
                return None
            # A ``merge-conflict`` issue that fails the base queue rules is
            # reported through the same checks as an ordinary issue below.
        else:
            return f"issue #{number} is not labelled `{READY_FOR_AGENT}`"
    if issue.assignee_logins:
        holders = ", ".join(issue.assignee_logins)
        return f"issue #{number} is already assigned to {holders}"
    if not issue.body.strip():
        return f"issue #{number} has an empty body"
    if issue.blocked_by != 0:
        noun = "blocker" if issue.blocked_by == 1 else "blockers"
        return f"issue #{number} has {issue.blocked_by} open {noun}"
    return None


class GitHubTrackerError(RuntimeError):
    """Raised when the GitHub tracker cannot complete an API operation."""


@dataclass(frozen=True)
class TrackerIssue:
    """The GitHub fields needed to determine issue eligibility."""

    id: str
    number: int
    title: str
    body: str
    created_at: datetime
    state: str
    labels: frozenset[str]
    assignee_logins: tuple[str, ...]
    blocked_by: int
    author_login: str

    @property
    def is_eligible(self) -> bool:
        """Whether this issue belongs in the runnable queue."""

        return (
            self.state == "OPEN"
            and READY_FOR_AGENT in self.labels
            and not self.assignee_logins
            and bool(self.body.strip())
            and self.blocked_by == 0
        )

    @property
    def meets_base_queue_rules(self) -> bool:
        """The runnable-queue rules shared by ordinary and continuation issues.

        Open, unassigned, with a non-empty body and no open native blockers;
        the queue label itself is checked separately.
        """

        return (
            self.state == "OPEN"
            and not self.assignee_logins
            and bool(self.body.strip())
            and self.blocked_by == 0
        )

    @property
    def is_continuation_signal(self) -> bool:
        """Whether the issue carries a handoff continuation signal only.

        ``round-finished`` without ``ready-for-agent``: a human requeue (which
        re-applies ``ready-for-agent``) stays an ordinary queue issue.
        """

        return ROUND_FINISHED in self.labels and READY_FOR_AGENT not in self.labels

    @property
    def is_conflict_signal(self) -> bool:
        """Whether the issue carries a conflict-requeue signal only.

        ``merge-conflict`` without ``ready-for-agent``, ``round-finished`` or
        ``ready-for-human``: an issue carrying one of those takes the
        ordinary (or continuation) path instead, and ``ready-for-human``
        always stays with a human.
        """

        return (
            MERGE_CONFLICT in self.labels
            and READY_FOR_AGENT not in self.labels
            and ROUND_FINISHED not in self.labels
            and READY_FOR_HUMAN not in self.labels
        )


@dataclass(frozen=True)
class IssuePage:
    """One cursor page of issues from the configured repository."""

    issues: tuple[TrackerIssue, ...]
    next_cursor: str | None


@dataclass(frozen=True)
class GitHubIdentity:
    """The authenticated identity used to claim an issue."""

    id: str
    login: str


@dataclass(frozen=True)
class Assignment:
    """The successful GitHub assignment that constitutes a claim."""

    issue_id: str
    assignee_id: str


@dataclass(frozen=True)
class Claim:
    """The selected issue snapshot and its successful claim result."""

    issue: TrackerIssue
    assignment: Assignment


@dataclass(frozen=True)
class IssueComment:
    """One issue comment, kept minimal to the fields prompt filtering needs."""

    author_login: str
    body: str


@dataclass(frozen=True)
class OpenPullRequest:
    """The detector fields of one open repository pull request.

    Only the scalars the conflict detector reads: no per-PR connections are
    attached, because each one costs a rate-limit point.
    """

    number: int
    head_ref: str
    base_ref: str
    head_sha: str
    draft: bool
    mergeable: str


@dataclass(frozen=True)
class PullRequestPage:
    """One cursor page of open pull requests from the configured repository."""

    pull_requests: tuple[OpenPullRequest, ...]
    next_cursor: str | None


class GitHubTransport(Protocol):
    """Transport seam; tests can provide a deterministic fake."""

    def list_issues(self, repository: str, cursor: str | None) -> IssuePage: ...

    def list_open_pull_requests(self, repository: str, cursor: str | None) -> PullRequestPage: ...

    def get_issue(self, repository: str, number: int) -> TrackerIssue | None: ...

    def viewer(self) -> GitHubIdentity: ...

    def assign_issue(self, issue_id: str, assignee_id: str) -> Assignment: ...

    def remove_assignee(self, issue_id: str, assignee_id: str) -> None: ...

    def remove_label(self, issue_id: str, label: str) -> None: ...

    def add_label(self, issue_id: str, label: str) -> None: ...

    def list_issue_comments(self, repository: str, issue_number: int) -> tuple[IssueComment, ...]: ...


class GitHubTracker:
    """Select and claim one eligible issue within exactly one repository."""

    def __init__(
        self,
        transport: GitHubTransport,
        target_repo: str,
        *,
        max_auto_continuations: int = 0,
        max_conflict_attempts: int = 0,
        peer_logins: Iterable[str] | None = None,
        event_log: Callable[..., None] = (
            lambda event, detail="", level="INFO", issue_number=None: None
        ),
    ) -> None:
        if (
            isinstance(max_auto_continuations, bool)
            or not isinstance(max_auto_continuations, int)
            or max_auto_continuations < 0
        ):
            raise GitHubTrackerError("MAX_AUTO_CONTINUATIONS must be a non-negative integer")
        if (
            isinstance(max_conflict_attempts, bool)
            or not isinstance(max_conflict_attempts, int)
            or max_conflict_attempts < 0
        ):
            raise GitHubTrackerError("MAX_CONFLICT_ATTEMPTS must be a non-negative integer")
        self._transport = transport
        self._target_repo = target_repo
        self._max_auto_continuations = max_auto_continuations
        self._max_conflict_attempts = max_conflict_attempts
        self._peer_logins = _peer_login_set(peer_logins)
        self._event_log = event_log

    @property
    def peer_logins(self) -> frozenset[str]:
        """Additional logins whose conflict-attempt comments count toward the budget.

        The worker's own login always counts; these pool peers count too.
        Empty when the pool setting is absent.
        """

        return self._peer_logins

    @property
    def max_auto_continuations(self) -> int:
        """The operator's automatic-continuation budget (0 disables the path)."""

        return self._max_auto_continuations

    @property
    def max_conflict_attempts(self) -> int:
        """The operator's conflict-requeue budget (0 disables the path)."""

        return self._max_conflict_attempts

    def is_conflict_eligible(self, issue: TrackerIssue) -> bool:
        """Whether a ``merge-conflict`` issue may be claimed for a conflict attempt.

        The issue must carry only the conflict signal (never alongside
        ``ready-for-agent`` or ``round-finished``, which take the ordinary
        path, and never with ``ready-for-human``) and meet the base queue
        rules. With a zero budget this is always False and reads no comments.
        Per-issue attempt counting arrives with the budget follow-up; today
        the setting only gates the queue.
        """

        if self._max_conflict_attempts <= 0:
            return False
        return bool(issue.is_conflict_signal and issue.meets_base_queue_rules)

    def is_auto_continuation_eligible(self, issue: TrackerIssue) -> bool:
        """Whether a ``round-finished`` issue may continue without a human requeue.

        The issue must carry only the continuation signal (never alongside
        ``ready-for-agent``, which stays ordinary), meet the base queue rules,
        and have an automatic-continuation count below the operator budget.
        With a zero budget this is always False and reads no comments.
        """

        if self._max_auto_continuations <= 0:
            return False
        if not issue.is_continuation_signal or not issue.meets_base_queue_rules:
            return False
        return self.automatic_continuation_count(issue) < self._max_auto_continuations

    def detect_conflict_requeues(
        self, by_number: dict[int, TrackerIssue]
    ) -> list[tuple[int, int, str]]:
        """Label issues whose published pull request reports merge conflicts.

        With a positive conflict budget, read the repository's open pull
        requests in one paged query and add ``merge-conflict`` to the issue
        behind every open, non-draft ``agent/issue-N`` branch that reports
        ``CONFLICTING``, when issue N is in this poll's open-issue set, meets
        the base queue rules, and carries none of the queue labels. Each
        labelled snapshot is updated in place so the issue is eligible in the
        same poll. With a zero budget (or a transport without the pull-request
        seam) no pull-request query runs and nothing is labelled. Returns the
        ``(issue, pull request, head sha)`` triples that were labelled, in
        pull-request page order.
        """

        if self._max_conflict_attempts <= 0:
            return []
        list_pull_requests = getattr(self._transport, "list_open_pull_requests", None)
        if not callable(list_pull_requests):
            return []
        detected: list[tuple[int, int, str]] = []
        cursor: str | None = None
        while True:
            page = list_pull_requests(self._target_repo, cursor)
            for pull_request in page.pull_requests:
                number = conflict_branch_issue_number(pull_request.head_ref)
                if number is None:
                    continue
                issue = by_number.get(number)
                if issue is None:
                    continue
                if not is_conflict_detection_eligible(
                    issue, draft=pull_request.draft, mergeable=pull_request.mergeable
                ):
                    continue
                self._transport.add_label(issue.id, MERGE_CONFLICT)
                by_number[number] = replace(
                    issue, labels=issue.labels | {MERGE_CONFLICT}
                )
                detected.append((number, pull_request.number, pull_request.head_sha))
                self._event_log(
                    "conflict_requeue_detected",
                    f"issue #{number}; pull request #{pull_request.number}"
                    f" ({pull_request.head_sha}) reports merge conflicts",
                    level="INFO",
                    issue_number=number,
                )
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        return detected

    def runnable_queue(self) -> tuple[TrackerIssue, ...]:
        """Return all eligible issues, conflicts then continuations then ordinary.

        Conflict requeues (``merge-conflict``) are claimed before automatic
        continuations, which are claimed before ordinary ``ready-for-agent``
        issues, so a blocked published branch does not wait behind new work;
        within each group the order is FIFO by creation date. A
        ``ready-for-agent`` issue that is also a continuation (a human requeue
        after a handoff) stays in the ordinary group, as does any issue
        carrying ``ready-for-agent`` or ``round-finished`` alongside
        ``merge-conflict``.

        Building the queue also runs the conflict detector: with a positive
        conflict budget, issues whose published pull request reports merge
        conflicts are labelled ``merge-conflict`` first, so they are eligible
        in this same poll.
        """

        by_number: dict[int, TrackerIssue] = {}
        cursor: str | None = None
        while True:
            page = self._transport.list_issues(self._target_repo, cursor)
            for issue in page.issues:
                by_number[issue.number] = issue
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        self.detect_conflict_requeues(by_number)
        conflicts: list[TrackerIssue] = []
        continuations: list[TrackerIssue] = []
        ordinary: list[TrackerIssue] = []
        for issue in by_number.values():
            if issue.is_eligible:
                ordinary.append(issue)
            elif self.is_conflict_eligible(issue):
                conflicts.append(issue)
            elif self.is_auto_continuation_eligible(issue):
                continuations.append(issue)
        by_fifo = lambda issue: (issue.created_at, issue.number)
        return tuple(
            sorted(conflicts, key=by_fifo)
            + sorted(continuations, key=by_fifo)
            + sorted(ordinary, key=by_fifo)
        )

    def fetch_issue(self, number: int) -> TrackerIssue | None:
        """Re-read one issue's current state for ``next issue`` validation."""

        return self._transport.get_issue(self._target_repo, number)

    def is_self_assigned(self, issue: TrackerIssue) -> bool:
        """Whether the issue is currently assigned to this agent's identity.

        A priority target that revalidates as self-assigned is an ambiguous
        prior assignment (it was unassigned when the priority was accepted,
        and only this process claims through the serialized boundary), never
        ordinary ineligibility: the caller must hold it for reconciliation
        instead of marking the priority ``not fulfilled``.
        """

        return self._transport.viewer().login in issue.assignee_logins

    def claim_verified(self, issue: TrackerIssue) -> Claim:
        """Assign an already revalidated priority target.

        The caller re-fetched the issue and confirmed eligibility immediately
        before this call, so no second eligibility read happens here: the
        existing assignment-based claim check runs exactly once. A stale
        ``round-finished`` label is removed as in :meth:`claim_next`.
        """

        identity = self._transport.viewer()
        assignment = self._transport.assign_issue(issue.id, identity.id)
        if ROUND_FINISHED in issue.labels:
            self._transport.remove_label(issue.id, ROUND_FINISHED)
        if MERGE_CONFLICT in issue.labels and (
            READY_FOR_AGENT in issue.labels or ROUND_FINISHED in issue.labels
        ):
            # An ordinary or continuation claim on an issue that also carries
            # ``merge-conflict``: those paths take precedence, so the stale
            # conflict signal is removed at the claim. A conflict-path claim
            # keeps the label until the conflict attempt itself finishes.
            self._transport.remove_label(issue.id, MERGE_CONFLICT)
        return Claim(issue=issue, assignment=assignment)

    def claim_next(self) -> Claim | None:
        """Claim the first issue that remains eligible at assignment time.

        This deliberately uses a check-then-set operation. The configured agent
        is a single process, so it is not presented as a distributed lock.

        A stale ``round-finished`` label from a prior handoff is removed here,
        regardless of this attempt's eventual outcome, so it never lingers
        across repeated handoff/continue cycles. The returned claim still
        carries the pre-removal issue snapshot, so callers can detect that a
        continuation was expected even after the label is gone.
        """

        queue = self.runnable_queue()
        if not queue:
            return None
        identity = self._transport.viewer()
        for candidate in queue:
            current = self._transport.get_issue(self._target_repo, candidate.number)
            # Revalidated at claim time under the same rule as the queue
            # read: an issue whose budget ran out, or that a human
            # relabelled, between the two reads is skipped.
            if current is None or (
                not current.is_eligible
                and not self.is_conflict_eligible(current)
                and not self.is_auto_continuation_eligible(current)
            ):
                continue
            assignment = self._transport.assign_issue(current.id, identity.id)
            if ROUND_FINISHED in current.labels:
                self._transport.remove_label(current.id, ROUND_FINISHED)
            if MERGE_CONFLICT in current.labels and (
                READY_FOR_AGENT in current.labels or ROUND_FINISHED in current.labels
            ):
                # The ordinary (or continuation) path takes precedence over
                # the conflict signal, so the stale label goes at the claim.
                self._transport.remove_label(current.id, MERGE_CONFLICT)
            return Claim(issue=current, assignment=assignment)
        return None

    def recover_claim(self, issue_number: int) -> Claim | None:
        """Reconstruct this agent's claim for checkpoint-based cleanup only."""

        current = self._transport.get_issue(self._target_repo, issue_number)
        if current is None:
            return None
        identity = self._transport.viewer()
        return Claim(current, Assignment(issue_id=current.id, assignee_id=identity.id))

    def release_attempt(self, issue_number: int, label: str, assignee_id: str) -> None:
        """Idempotently release only this agent's assignment and queue label.

        Each remote field is re-read immediately before its mutation, so a
        restart after a partial cleanup neither removes a human's work nor
        repeats an already-completed operation.
        """

        current = self._transport.get_issue(self._target_repo, issue_number)
        if current is None:
            return
        if label in current.labels:
            self._transport.remove_label(current.id, label)
        current = self._transport.get_issue(self._target_repo, issue_number)
        if current is None:
            return
        identity = self._transport.viewer()
        if identity.login in current.assignee_logins:
            self._transport.remove_assignee(current.id, assignee_id)

    def release_handoff(self, issue_number: int, assignee_id: str) -> None:
        """Idempotently apply the handoff label/assignee sequence in its required order.

        Each remote field is re-read immediately before its mutation, exactly
        like ``release_attempt``, so a restart between steps resumes from the
        first unconfirmed one without repeating an already-completed write or
        prematurely releasing a human's requeue signal.
        """

        current = self._transport.get_issue(self._target_repo, issue_number)
        if current is None:
            return
        if READY_FOR_AGENT in current.labels:
            self._transport.remove_label(current.id, READY_FOR_AGENT)
        current = self._transport.get_issue(self._target_repo, issue_number)
        if current is None:
            return
        if ROUND_FINISHED not in current.labels:
            self._transport.add_label(current.id, ROUND_FINISHED)
        current = self._transport.get_issue(self._target_repo, issue_number)
        if current is None:
            return
        identity = self._transport.viewer()
        if identity.login in current.assignee_logins:
            self._transport.remove_assignee(current.id, assignee_id)

    def trusted_comments(self, issue: TrackerIssue) -> tuple[str, ...]:
        """Chronologically ordered bodies of comments trusted for the model's prompt.

        Trusted means authored by the issue's own author or by this agent's
        GitHub identity (its own attempt-result and handoff comments). Any
        other commenter is excluded, so the model never sees guidance it
        cannot attribute to someone with write access to the issue.
        """

        viewer = self._transport.viewer()
        comments = self._transport.list_issue_comments(self._target_repo, issue.number)
        return tuple(
            comment.body
            for comment in comments
            # An empty login means a deleted account; never trust one by
            # incidentally matching another deleted account.
            if comment.author_login and comment.author_login in (issue.author_login, viewer.login)
        )

    def automatic_continuation_count(self, issue: TrackerIssue) -> int:
        """Count automatic continuations in a row since the last human requeue.

        Only comments written by this agent's own GitHub identity are
        examined; any other comment is ignored even if it contains the
        marker. A human-requeue result (or a legacy unmarked result from
        before the marker existed) resets the count to zero.
        """

        viewer = self._transport.viewer()
        comments = self._transport.list_issue_comments(self._target_repo, issue.number)
        own_bodies_newest_first = tuple(
            comment.body
            for comment in reversed(comments)
            # An empty login means a deleted account; never count one by
            # incidentally matching another deleted account.
            if comment.author_login and comment.author_login == viewer.login
        )
        return count_automatic_continuations(own_bodies_newest_first)

    def conflict_failure_streak(self, issue: TrackerIssue) -> int:
        """Count consecutive failed conflict attempts on ``issue`` since the last stop.

        Only comments authored by this worker's own GitHub identity (plus
        any pool-peer login) are examined; any other comment is ignored
        even if it contains the marker. The streak stops at a successful
        conflict attempt (it pushed a new sha), an ordinary attempt result,
        a ``skipped`` result, or the latest escalation comment, so removing
        ``ready-for-human`` after an escalation requeues with the full
        budget. With a zero budget this still reads the comments: the
        budget is enforced at escalation time, never at claim time.
        """

        viewer = self._transport.viewer()
        allowed = {viewer.login} | set(self._peer_logins)
        comments = self._transport.list_issue_comments(self._target_repo, issue.number)
        own_bodies_newest_first = tuple(
            comment.body
            for comment in reversed(comments)
            # An empty login means a deleted account; never count one by
            # incidentally matching another deleted account.
            if comment.author_login and comment.author_login in allowed
        )
        return count_conflict_failures(own_bodies_newest_first)

    def escalate_conflict_to_human(self, issue: TrackerIssue) -> None:
        """Hand a conflict-exhausted issue to a human with ``ready-for-human``.

        The escalating attempt's own result comment is the escalation
        comment; this only adds the gate label (created in the repository
        when missing, like the other worker-applied labels). The agent
        never removes ``ready-for-human``: a human hands the issue back by
        removing it, which requeues with the full budget.
        """

        self._transport.add_label(issue.id, READY_FOR_HUMAN)


class GitHubGraphQLTransport:
    """GitHub GraphQL transport used by the tracker in production."""

    _endpoint = "https://api.github.com/graphql"

    def __init__(self, token: str, *, max_retries: int = 3) -> None:
        self._token = token
        self._max_retries = max_retries

    def list_issues(self, repository: str, cursor: str | None) -> IssuePage:
        owner, name = _repository_parts(repository)
        data = self._execute(
            """
            query Queue($owner: String!, $name: String!, $cursor: String) {
              repository(owner: $owner, name: $name) {
                issues(first: 100, after: $cursor, states: OPEN, orderBy: {field: CREATED_AT, direction: ASC}) {
                  nodes { ...IssueFields }
                  pageInfo { hasNextPage endCursor }
                }
              }
            }
            fragment IssueFields on Issue {
              id number title body createdAt state
              author { login }
              labels(first: 100) { nodes { name } }
              assignees(first: 100) { nodes { login } }
              issueDependenciesSummary { blockedBy }
            }
            """,
            {"owner": owner, "name": name, "cursor": cursor},
        )
        try:
            connection = data["repository"]["issues"]
            page_info = connection["pageInfo"]
            return IssuePage(
                issues=tuple(_issue_from_graphql(value) for value in connection["nodes"]),
                next_cursor=page_info["endCursor"] if page_info["hasNextPage"] else None,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise GitHubTrackerError("GitHub returned an invalid issue page") from error

    def get_issue(self, repository: str, number: int) -> TrackerIssue | None:
        owner, name = _repository_parts(repository)
        data = self._execute(
            """
            query Issue($owner: String!, $name: String!, $number: Int!) {
              repository(owner: $owner, name: $name) {
                issue(number: $number) { ...IssueFields }
              }
            }
            fragment IssueFields on Issue {
              id number title body createdAt state
              author { login }
              labels(first: 100) { nodes { name } }
              assignees(first: 100) { nodes { login } }
              issueDependenciesSummary { blockedBy }
            }
            """,
            {"owner": owner, "name": name, "number": number},
        )
        try:
            value = data["repository"]["issue"]
            return None if value is None else _issue_from_graphql(value)
        except (KeyError, TypeError, ValueError) as error:
            raise GitHubTrackerError("GitHub returned an invalid issue") from error

    def list_open_pull_requests(
        self, repository: str, cursor: str | None
    ) -> PullRequestPage:
        """Return one page of the repository's open pull requests for the detector.

        The detector owns the only pull-request listing: one paged query per
        poll reading exactly ``number headRefName baseRefName headRefOid
        isDraft mergeable``, with no ``baseRefName`` filter (the profile is
        not loaded in the loop) and no per-PR connections.
        """

        owner, name = _repository_parts(repository)
        data = self._execute(
            """
            query OpenPullRequests($owner: String!, $name: String!, $cursor: String) {
              repository(owner: $owner, name: $name) {
                pullRequests(first: 100, after: $cursor, states: OPEN, orderBy: {field: CREATED_AT, direction: ASC}) {
                  nodes { number headRefName baseRefName headRefOid isDraft mergeable }
                  pageInfo { hasNextPage endCursor }
                }
              }
            }
            """,
            {"owner": owner, "name": name, "cursor": cursor},
        )
        try:
            connection = data["repository"]["pullRequests"]
            page_info = connection["pageInfo"]
            return PullRequestPage(
                pull_requests=tuple(
                    _open_pull_request_from_graphql(value)
                    for value in connection["nodes"]
                ),
                next_cursor=page_info["endCursor"] if page_info["hasNextPage"] else None,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise GitHubTrackerError("GitHub returned an invalid pull request page") from error

    def viewer(self) -> GitHubIdentity:
        data = self._execute("query Viewer { viewer { id login } }", {})
        try:
            viewer = data["viewer"]
            return GitHubIdentity(id=viewer["id"], login=viewer["login"])
        except (KeyError, TypeError) as error:
            raise GitHubTrackerError("GitHub returned an invalid viewer") from error

    def assign_issue(self, issue_id: str, assignee_id: str) -> Assignment:
        self._execute(
            """
            mutation Assign($issueId: ID!, $assigneeId: ID!) {
              addAssigneesToAssignable(input: {assignableId: $issueId, assigneeIds: [$assigneeId]}) {
                clientMutationId
              }
            }
            """,
            {"issueId": issue_id, "assigneeId": assignee_id},
        )
        return Assignment(issue_id=issue_id, assignee_id=assignee_id)

    def remove_assignee(self, issue_id: str, assignee_id: str) -> None:
        """Remove one known assignee without affecting any other assignee."""

        self._execute(
            """
            mutation RemoveAssignee($issueId: ID!, $assigneeId: ID!) {
              removeAssigneesFromAssignable(input: {assignableId: $issueId, assigneeIds: [$assigneeId]}) {
                clientMutationId
              }
            }
            """,
            {"issueId": issue_id, "assigneeId": assignee_id},
        )

    def remove_label(self, issue_id: str, label: str) -> None:
        """Remove just the queue label after resolving its repository node id."""

        data = self._execute(
            """
            query LabelId($issueId: ID!) {
              node(id: $issueId) { ... on Issue { labels(first: 100) { nodes { id name } } } }
            }
            """,
            {"issueId": issue_id},
        )
        try:
            labels = data["node"]["labels"]["nodes"]
            label_id = next(item["id"] for item in labels if item["name"] == label)
        except (KeyError, StopIteration, TypeError) as error:
            raise GitHubTrackerError("GitHub returned an invalid issue label") from error
        self._execute(
            """
            mutation RemoveLabel($issueId: ID!, $labelId: ID!) {
              removeLabelsFromLabelable(input: {labelableId: $issueId, labelIds: [$labelId]}) {
                clientMutationId
              }
            }
            """,
            {"issueId": issue_id, "labelId": label_id},
        )

    def add_label(self, issue_id: str, label: str) -> None:
        """Add a repository label to the issue, resolving its node id first.

        If the label does not exist in the repository yet, it is created
        automatically so the tracker never crashes on uninitialized labels.
        When another process creates the same label concurrently, GitHub
        rejects the creation as already existing; the existing label is then
        re-read and applied as if this process had created it.
        """

        data = self._execute(
            """
            query RepositoryLabelId($issueId: ID!, $label: String!) {
              node(id: $issueId) {
                ... on Issue {
                  repository { id label(name: $label) { id } }
                }
              }
            }
            """,
            {"issueId": issue_id, "label": label},
        )
        try:
            repo_node = data["node"]["repository"]
            label_obj = repo_node.get("label")
            if label_obj and isinstance(label_obj, dict) and "id" in label_obj:
                label_id = label_obj["id"]
            else:
                repo_id = repo_node["id"]
                try:
                    label_id = self._create_label(repo_id, label)
                except GitHubTrackerError as error:
                    if not _is_label_already_exists_error(error):
                        raise
                    label_id = self._reread_label_id_after_conflict(issue_id, label, error)
        except (KeyError, TypeError) as error:
            raise GitHubTrackerError("GitHub returned an invalid repository label") from error
        self._execute(
            """
            mutation AddLabel($issueId: ID!, $labelId: ID!) {
              addLabelsToLabelable(input: {labelableId: $issueId, labelIds: [$labelId]}) {
                clientMutationId
              }
            }
            """,
            {"issueId": issue_id, "labelId": label_id},
        )

    def _create_label(self, repository_id: str, label: str) -> str:
        if label == ROUND_FINISHED:
            color = "fbca04"
            description = "Agent round finished; requires human evaluation to continue"
        elif label == MERGE_CONFLICT:
            color = MERGE_CONFLICT_LABEL_COLOR
            description = MERGE_CONFLICT_LABEL_DESCRIPTION
        else:
            color = "ededed"
            description = ""
        data = self._execute(
            """
            mutation CreateLabel($repositoryId: ID!, $name: String!, $color: String!, $description: String) {
              createLabel(input: {repositoryId: $repositoryId, name: $name, color: $color, description: $description}) {
                label { id }
              }
            }
            """,
            {
                "repositoryId": repository_id,
                "name": label,
                "color": color,
                "description": description,
            },
        )
        try:
            return data["createLabel"]["label"]["id"]
        except (KeyError, TypeError) as error:
            raise GitHubTrackerError(f"GitHub returned an invalid repository label: {error}") from error

    def _reread_label_id_after_conflict(
        self, issue_id: str, label: str, error: GitHubTrackerError
    ) -> str:
        """Re-read the label id created concurrently by another process."""

        data = self._execute(
            """
            query RepositoryLabelId($issueId: ID!, $label: String!) {
              node(id: $issueId) {
                ... on Issue {
                  repository { id label(name: $label) { id } }
                }
              }
            }
            """,
            {"issueId": issue_id, "label": label},
        )
        try:
            label_obj = data["node"]["repository"].get("label")
            if label_obj and isinstance(label_obj, dict) and "id" in label_obj:
                return label_obj["id"]
        except (KeyError, TypeError, AttributeError):
            raise GitHubTrackerError(
                "GitHub returned an invalid repository label"
            ) from error
        raise GitHubTrackerError("GitHub returned an invalid repository label") from error

    def find_pull_request(self, repository: str, head: str) -> PullRequest | None:
        """Return an existing open PR for an attempt branch, if any.

        The entry-check fields (draft state, mergeable status, and base
        branch) ride along so a conflict attempt can decide without a second
        read; only open pull requests are returned, so a merged or closed PR
        reads as absent.
        """

        owner, name = _repository_parts(repository)
        data = self._execute(
            """
            query PullRequest($owner: String!, $name: String!, $head: String!) {
              repository(owner: $owner, name: $name) {
                pullRequests(first: 1, states: OPEN, headRefName: $head) {
                  nodes { id number url isDraft mergeable baseRefName headRefOid }
                }
              }
            }
            """,
            {"owner": owner, "name": name, "head": head},
        )
        try:
            nodes = data["repository"]["pullRequests"]["nodes"]
            return None if not nodes else _pull_request_from_graphql(nodes[0])
        except (KeyError, TypeError, ValueError) as error:
            raise GitHubTrackerError("GitHub returned an invalid pull request") from error

    def create_pull_request(
        self, repository: str, head: str, base: str, title: str, body: str
    ) -> PullRequest:
        """Create the single human-reviewable PR for a verified complete attempt."""

        owner, name = _repository_parts(repository)
        data = self._execute(
            """
            mutation CreatePullRequest(
              $repositoryId: ID!, $head: String!, $base: String!, $title: String!, $body: String!
            ) {
              createPullRequest(input: {
                repositoryId: $repositoryId, headRefName: $head, baseRefName: $base, title: $title, body: $body
              }) { pullRequest { id number url } }
            }
            """,
            {
                "repositoryId": self._repository_id(owner, name),
                "head": head,
                "base": base,
                "title": title,
                "body": body,
            },
        )
        try:
            return _pull_request_from_graphql(data["createPullRequest"]["pullRequest"])
        except (KeyError, TypeError, ValueError) as error:
            raise GitHubTrackerError("GitHub returned an invalid created pull request") from error

    def enable_auto_merge(
        self, repository: str, pull_request: PullRequest, merge_method: str
    ) -> None:
        """Enable GitHub's native auto-merge so the PR merges once protection passes."""

        _repository_parts(repository)
        pull_request_id = pull_request.node_id
        if not pull_request_id:
            raise GitHubTrackerError("GitHub returned a pull request without an id")
        self._execute(
            """
            mutation EnableAutoMerge($pullRequestId: ID!, $mergeMethod: PullRequestMergeMethod!) {
              enablePullRequestAutoMerge(
                input: {pullRequestId: $pullRequestId, mergeMethod: $mergeMethod}
              ) { pullRequest { autoMergeRequest { mergeMethod } } }
            }
            """,
            {
                "pullRequestId": pull_request_id,
                "mergeMethod": _graphql_merge_method(merge_method),
            },
        )

    def merge_pull_request(
        self, repository: str, pull_request: PullRequest, merge_method: str
    ) -> None:
        """Merge an already-mergeable PR directly with the configured method."""

        _repository_parts(repository)
        pull_request_id = pull_request.node_id
        if not pull_request_id:
            raise GitHubTrackerError("GitHub returned a pull request without an id")
        self._execute(
            """
            mutation MergePullRequest($pullRequestId: ID!, $mergeMethod: PullRequestMergeMethod!) {
              mergePullRequest(
                input: {pullRequestId: $pullRequestId, mergeMethod: $mergeMethod}
              ) { pullRequest { merged } }
            }
            """,
            {
                "pullRequestId": pull_request_id,
                "mergeMethod": _graphql_merge_method(merge_method),
            },
        )

    def find_attempt_comment(self, repository: str, issue_number: int, marker: str) -> bool:
        """Find a result only when it carries this attempt's durable marker.

        Both ordinary attempt results and conflict-attempt results carry the
        attempt marker; the header distinguishes them.
        """

        owner, name = _repository_parts(repository)
        data = self._execute(
            """
            query AttemptComments($owner: String!, $name: String!, $number: Int!) {
              repository(owner: $owner, name: $name) {
                issue(number: $number) { comments(last: 100) { nodes { body } } }
              }
            }
            """,
            {"owner": owner, "name": name, "number": issue_number},
        )
        try:
            comments = data["repository"]["issue"]["comments"]["nodes"]
            return any(
                (
                    "Agent Attempt Result" in comment["body"]
                    or "Conflict Attempt Result" in comment["body"]
                )
                and marker in comment["body"]
                for comment in comments
            )
        except (KeyError, TypeError) as error:
            raise GitHubTrackerError("GitHub returned invalid issue comments") from error

    def list_issue_comments(self, repository: str, issue_number: int) -> tuple[IssueComment, ...]:
        """Return every comment on an issue, unfiltered; trust filtering is the tracker's job."""

        owner, name = _repository_parts(repository)
        data = self._execute(
            """
            query IssueComments($owner: String!, $name: String!, $number: Int!) {
              repository(owner: $owner, name: $name) {
                issue(number: $number) {
                  comments(first: 100) { nodes { body author { login } } }
                }
              }
            }
            """,
            {"owner": owner, "name": name, "number": issue_number},
        )
        try:
            nodes = data["repository"]["issue"]["comments"]["nodes"]
            return tuple(_comment_from_graphql(value) for value in nodes)
        except (KeyError, TypeError) as error:
            raise GitHubTrackerError("GitHub returned invalid issue comments") from error

    def add_comment(self, repository: str, issue_number: int, body: str) -> None:
        """Post a concise result comment without exposing local execution data."""

        owner, name = _repository_parts(repository)
        issue_id = self._issue_id(owner, name, issue_number)
        self._execute(
            """
            mutation AddComment($subjectId: ID!, $body: String!) {
              addComment(input: {subjectId: $subjectId, body: $body}) { commentEdge { node { id } } }
            }
            """,
            {"subjectId": issue_id, "body": body},
        )

    def _repository_id(self, owner: str, name: str) -> str:
        data = self._execute(
            "query RepositoryId($owner: String!, $name: String!) { repository(owner: $owner, name: $name) { id } }",
            {"owner": owner, "name": name},
        )
        try:
            return data["repository"]["id"]
        except (KeyError, TypeError) as error:
            raise GitHubTrackerError("GitHub returned an invalid repository") from error

    def _issue_id(self, owner: str, name: str, number: int) -> str:
        data = self._execute(
            "query IssueId($owner: String!, $name: String!, $number: Int!) { repository(owner: $owner, name: $name) { issue(number: $number) { id } } }",
            {"owner": owner, "name": name, "number": number},
        )
        try:
            return data["repository"]["issue"]["id"]
        except (KeyError, TypeError) as error:
            raise GitHubTrackerError("GitHub returned an invalid issue") from error

    def _execute(self, query: str, variables: dict[str, object]) -> dict[str, Any]:
        request = Request(
            self._endpoint,
            data=json.dumps({"query": query, "variables": variables}).encode(),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        response_data: object | None = None
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                with urlopen(request) as response:  # noqa: S310 -- fixed GitHub endpoint
                    response_data = json.load(response)
                break
            except HTTPError as error:
                if error.code != 429 and not 500 <= error.code < 600:
                    raise GitHubTrackerError("GitHub request was rejected") from error
                last_error = error
                if attempt + 1 == self._max_retries:
                    raise GitHubTrackerError("GitHub request failed") from error
                time.sleep(_retry_delay(error, attempt))
            except (URLError, TimeoutError) as error:
                last_error = error
                if attempt + 1 == self._max_retries:
                    raise GitHubTrackerError("GitHub request failed") from error
                time.sleep(_retry_delay(error, attempt))
            except (OSError, ValueError) as error:
                raise GitHubTrackerError("GitHub request failed") from error
        if response_data is None:
            raise GitHubTrackerError("GitHub request failed") from last_error
        if not isinstance(response_data, dict) or response_data.get("errors"):
            raise GitHubTrackerError(
                f"GitHub GraphQL request was rejected: {_graphql_error_summary(response_data)}"
            )
        data = response_data.get("data")
        if not isinstance(data, dict):
            raise GitHubTrackerError("GitHub response has no data")
        return data


def _graphql_error_summary(response_data: object) -> str:
    """Surface GitHub's own error messages instead of hiding them behind a generic label."""

    if not isinstance(response_data, dict):
        return "malformed response"
    errors = response_data.get("errors")
    if not isinstance(errors, list) or not errors:
        return "no error details provided"
    messages = [
        error["message"]
        for error in errors
        if isinstance(error, dict) and isinstance(error.get("message"), str)
    ]
    return "; ".join(messages) if messages else "no error details provided"


def _is_label_already_exists_error(error: GitHubTrackerError) -> bool:
    """Whether a label-creation failure means another process won the race."""

    return "already exists" in str(error).lower().replace("_", " ")


def _retry_delay(error: Exception, attempt: int) -> float:
    """Honor GitHub's bounded Retry-After signal before normal backoff."""

    if isinstance(error, HTTPError):
        retry_after = error.headers.get("Retry-After") if error.headers is not None else None
        try:
            return min(float(retry_after), 300) if retry_after is not None else min(2 * 2**attempt, 30)
        except ValueError:
            return min(2 * 2**attempt, 30)
    return min(2 * 2**attempt, 30)


def _graphql_merge_method(merge_method: str) -> str:
    """Map the operator merge method to GitHub's ``PullRequestMergeMethod`` enum."""

    try:
        return {"merge": "MERGE", "squash": "SQUASH", "rebase": "REBASE"}[merge_method.strip().lower()]
    except (AttributeError, KeyError) as error:
        raise GitHubTrackerError("PR_MERGE_METHOD must be merge, squash, or rebase") from error


def _repository_parts(repository: str) -> tuple[str, str]:
    try:
        owner, name = repository.split("/", 1)
    except ValueError as error:
        raise GitHubTrackerError("TARGET_REPO must use the owner/repo format") from error
    if not owner or not name or "/" in name:
        raise GitHubTrackerError("TARGET_REPO must use the owner/repo format")
    return owner, name


def _issue_from_graphql(value: dict[str, Any]) -> TrackerIssue:
    author = value["author"]
    return TrackerIssue(
        id=value["id"],
        number=value["number"],
        title=value["title"],
        body=value["body"],
        created_at=datetime.fromisoformat(value["createdAt"].replace("Z", "+00:00")),
        state=value["state"],
        labels=frozenset(label["name"] for label in value["labels"]["nodes"]),
        assignee_logins=tuple(assignee["login"] for assignee in value["assignees"]["nodes"]),
        blocked_by=value["issueDependenciesSummary"]["blockedBy"],
        author_login=author["login"] if author is not None else "",
    )


def _comment_from_graphql(value: dict[str, Any]) -> IssueComment:
    author = value["author"]
    return IssueComment(
        author_login=author["login"] if author is not None else "",
        body=value["body"],
    )


def _open_pull_request_from_graphql(value: dict[str, Any]) -> OpenPullRequest:
    number = value["number"]
    head_ref = value["headRefName"]
    base_ref = value["baseRefName"]
    head_sha = value["headRefOid"]
    draft = value["isDraft"]
    mergeable = value["mergeable"]
    if isinstance(number, bool) or not isinstance(number, int):
        raise ValueError("invalid pull request")
    if not isinstance(head_ref, str) or not head_ref:
        raise ValueError("invalid pull request")
    if not isinstance(base_ref, str):
        raise ValueError("invalid pull request")
    if not isinstance(head_sha, str) or not head_sha:
        raise ValueError("invalid pull request")
    if not isinstance(draft, bool):
        raise ValueError("invalid pull request")
    if not isinstance(mergeable, str) or not mergeable:
        raise ValueError("invalid pull request")
    return OpenPullRequest(number, head_ref, base_ref, head_sha, draft, mergeable.upper())


def _pull_request_from_graphql(value: dict[str, Any]) -> PullRequest:
    number = value["number"]
    url = value["url"]
    node_id = value.get("id", "")
    draft = value.get("isDraft", False)
    mergeable = value.get("mergeable", "UNKNOWN")
    base_ref = value.get("baseRefName", "")
    head_sha = value.get("headRefOid", "")
    if isinstance(number, bool) or not isinstance(number, int) or not isinstance(url, str) or not url:
        raise ValueError("invalid pull request")
    if not isinstance(node_id, str):
        raise ValueError("invalid pull request")
    if not isinstance(draft, bool):
        raise ValueError("invalid pull request")
    if not isinstance(mergeable, str) or not mergeable:
        raise ValueError("invalid pull request")
    if not isinstance(base_ref, str):
        raise ValueError("invalid pull request")
    if not isinstance(head_sha, str):
        raise ValueError("invalid pull request")
    return PullRequest(number, url, node_id, draft, mergeable.upper(), base_ref, head_sha)
