"""GitHub boundary for selecting and claiming implementation issues."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import time
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from simple_coding_agent.publication import PullRequest


READY_FOR_AGENT = "ready-for-agent"


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


class GitHubTransport(Protocol):
    """Transport seam; tests can provide a deterministic fake."""

    def list_issues(self, repository: str, cursor: str | None) -> IssuePage: ...

    def get_issue(self, repository: str, number: int) -> TrackerIssue | None: ...

    def viewer(self) -> GitHubIdentity: ...

    def assign_issue(self, issue_id: str, assignee_id: str) -> Assignment: ...

    def remove_assignee(self, issue_id: str, assignee_id: str) -> None: ...

    def remove_label(self, issue_id: str, label: str) -> None: ...

    def list_issue_comments(self, repository: str, issue_number: int) -> tuple[IssueComment, ...]: ...


class GitHubTracker:
    """Select and claim one eligible issue within exactly one repository."""

    def __init__(self, transport: GitHubTransport, target_repo: str) -> None:
        self._transport = transport
        self._target_repo = target_repo

    def runnable_queue(self) -> tuple[TrackerIssue, ...]:
        """Return all eligible issues in FIFO creation order."""

        issues: list[TrackerIssue] = []
        cursor: str | None = None
        while True:
            page = self._transport.list_issues(self._target_repo, cursor)
            issues.extend(issue for issue in page.issues if issue.is_eligible)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        return tuple(sorted(issues, key=lambda issue: (issue.created_at, issue.number)))

    def claim_next(self) -> Claim | None:
        """Claim the first issue that remains eligible at assignment time.

        This deliberately uses a check-then-set operation. The configured agent
        is a single process, so it is not presented as a distributed lock.
        """

        queue = self.runnable_queue()
        if not queue:
            return None
        identity = self._transport.viewer()
        for candidate in queue:
            current = self._transport.get_issue(self._target_repo, candidate.number)
            if current is None or not current.is_eligible:
                continue
            assignment = self._transport.assign_issue(current.id, identity.id)
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

    def find_pull_request(self, repository: str, head: str) -> PullRequest | None:
        """Return an existing open PR for an attempt branch, if any."""

        owner, name = _repository_parts(repository)
        data = self._execute(
            """
            query PullRequest($owner: String!, $name: String!, $head: String!) {
              repository(owner: $owner, name: $name) {
                pullRequests(first: 1, states: OPEN, headRefName: $head) {
                  nodes { number url }
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
              }) { pullRequest { number url } }
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

    def find_attempt_comment(self, repository: str, issue_number: int, marker: str) -> bool:
        """Find a result only when it carries this attempt's durable marker."""

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
                "Agent Attempt Result" in comment["body"] and marker in comment["body"]
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


def _retry_delay(error: Exception, attempt: int) -> float:
    """Honor GitHub's bounded Retry-After signal before normal backoff."""

    if isinstance(error, HTTPError):
        retry_after = error.headers.get("Retry-After") if error.headers is not None else None
        try:
            return min(float(retry_after), 300) if retry_after is not None else min(2 * 2**attempt, 30)
        except ValueError:
            return min(2 * 2**attempt, 30)
    return min(2 * 2**attempt, 30)


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


def _pull_request_from_graphql(value: dict[str, Any]) -> PullRequest:
    number = value["number"]
    url = value["url"]
    if isinstance(number, bool) or not isinstance(number, int) or not isinstance(url, str) or not url:
        raise ValueError("invalid pull request")
    return PullRequest(number, url)
