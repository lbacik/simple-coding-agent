from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json

import pytest

from simple_coding_agent.github_tracker import (
    Assignment,
    GitHubGraphQLTransport,
    GitHubIdentity,
    GitHubTracker,
    GitHubTrackerError,
    IssueComment,
    IssuePage,
    TrackerIssue,
)
from simple_coding_agent.publication import PullRequest


def test_lists_only_eligible_open_issues() -> None:
    eligible = issue(1)
    transport = FakeTransport(
        pages=[
            IssuePage(
                issues=(
                    eligible,
                    replace(eligible, number=2, labels=frozenset()),
                    replace(eligible, number=3, assignee_logins=("person",)),
                    replace(eligible, number=4, body=" \n\t"),
                    replace(eligible, number=5, blocked_by=1),
                    replace(eligible, number=6, state="CLOSED"),
                ),
                next_cursor=None,
            )
        ]
    )

    assert GitHubTracker(transport, "octo/example").runnable_queue() == (eligible,)


def test_paginates_then_orders_the_runnable_queue_fifo() -> None:
    later = issue(2, created_at=datetime(2026, 9, 20, 14, 0, tzinfo=UTC))
    first = issue(1, created_at=datetime(2026, 9, 20, 13, 0, tzinfo=UTC))
    transport = FakeTransport(
        pages=[
            IssuePage(issues=(later,), next_cursor="next"),
            IssuePage(issues=(first,), next_cursor=None),
        ]
    )

    assert [item.number for item in GitHubTracker(transport, "octo/example").runnable_queue()] == [1, 2]
    assert transport.page_cursors == [None, "next"]


def test_claims_the_first_issue_that_remains_eligible() -> None:
    selected = issue(2, title="Selected", body="Implement this")
    transport = FakeTransport(
        pages=[
            IssuePage(
                issues=(
                    issue(1, created_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC)),
                    selected,
                ),
                next_cursor=None,
            )
        ],
        refreshed={
            1: replace(issue(1), assignee_logins=("another-agent",)),
            2: selected,
        },
    )

    claim = GitHubTracker(transport, "octo/example").claim_next()

    assert claim is not None
    assert claim.issue.number == 2
    assert claim.issue.title == "Selected"
    assert claim.issue.body == "Implement this"
    assert claim.assignment == Assignment(issue_id="issue-2", assignee_id="viewer-id")
    assert transport.assignments == [("issue-2", "viewer-id")]


@pytest.mark.parametrize(
    "change",
    [
        "assigned",
        "empty_body",
        "blocked",
        "unlabelled",
        "closed",
    ],
)
def test_does_not_assign_an_issue_that_changes_before_claim(
    change: str,
) -> None:
    candidate = issue(1)
    changed_issue = {
        "assigned": replace(candidate, assignee_logins=("human",)),
        "empty_body": replace(candidate, body="  "),
        "blocked": replace(candidate, blocked_by=1),
        "unlabelled": replace(candidate, labels=frozenset()),
        "closed": replace(candidate, state="CLOSED"),
    }[change]
    transport = FakeTransport(
        pages=[IssuePage(issues=(candidate,), next_cursor=None)],
        refreshed={1: changed_issue},
    )

    assert GitHubTracker(transport, "octo/example").claim_next() is None
    assert transport.assignments == []


def test_trusted_comments_keeps_only_the_author_and_the_viewer_in_order() -> None:
    transport = FakeCommentTransport(
        comments=(
            IssueComment(author_login="reporter", body="please also handle timeouts"),
            IssueComment(author_login="random-passerby", body="+1"),
            IssueComment(author_login="agent", body="## Agent Attempt Result: incomplete"),
        )
    )

    trusted = GitHubTracker(transport, "octo/example").trusted_comments(issue(1, author_login="reporter"))

    assert trusted == ("please also handle timeouts", "## Agent Attempt Result: incomplete")


def test_trusted_comments_degrades_to_empty_tuple_with_no_comments() -> None:
    transport = FakeCommentTransport(comments=())

    trusted = GitHubTracker(transport, "octo/example").trusted_comments(issue(1, author_login="reporter"))

    assert trusted == ()


def test_lists_issue_comments_with_author_login_via_graphql() -> None:
    transport = RecordingGraphQLTransport(
        [
            {
                "repository": {
                    "issue": {
                        "comments": {
                            "nodes": [
                                {"body": "note", "author": {"login": "reporter"}},
                                {"body": "from a deleted account", "author": None},
                            ]
                        }
                    }
                }
            }
        ]
    )

    comments = transport.list_issue_comments("octo/example", 1)

    assert comments == (
        IssueComment(author_login="reporter", body="note"),
        IssueComment(author_login="", body="from a deleted account"),
    )
    assert "comments(first: 100)" in transport.queries[0]


def test_publication_transport_uses_attempt_marker_and_reuses_an_existing_pr() -> None:
    transport = RecordingGraphQLTransport(
        [
            {"repository": {"pullRequests": {"nodes": [{"number": 8, "url": "https://example.test/8"}]}}},
            {"repository": {"issue": {"comments": {"nodes": [
                {"body": "## Agent Attempt Result: complete\n<!-- agent-attempt: now -->"},
                {"body": "## Agent Attempt Result: complete\n<!-- agent-attempt: earlier -->"},
            ]}}}},
        ]
    )

    assert transport.find_pull_request("octo/example", "agent/issue-23") == PullRequest(8, "https://example.test/8")
    assert transport.find_attempt_comment("octo/example", 23, "<!-- agent-attempt: now -->")
    assert "headRefName" in transport.queries[0]
    assert "comments(last: 100)" in transport.queries[1]


def test_releases_only_the_ready_label_and_the_authenticated_agent_assignment() -> None:
    claimed = replace(issue(24), assignee_logins=("agent", "human"), labels=frozenset({"ready-for-agent", "bug"}))
    after_label = replace(claimed, labels=frozenset({"bug"}))
    transport = FakeTransport(pages=[], refreshed={24: claimed}, releases={24: after_label})

    GitHubTracker(transport, "octo/example").release_attempt(24, "ready-for-agent", "viewer-id")

    assert transport.removed_labels == [("issue-24", "ready-for-agent")]
    assert transport.removed_assignees == [("issue-24", "viewer-id")]


def test_graphql_errors_surface_githubs_own_message(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps(
        {"errors": [{"message": "Resource not accessible by integration"}]}
    ).encode()
    monkeypatch.setattr(
        "simple_coding_agent.github_tracker.urlopen",
        lambda request: _FakeResponse(payload),
    )
    transport = GitHubGraphQLTransport("token")

    with pytest.raises(GitHubTrackerError) as excinfo:
        transport._execute("query Viewer { viewer { id login } }", {})

    assert "Resource not accessible by integration" in str(excinfo.value)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *arguments: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


class RecordingGraphQLTransport:
    def __init__(self, responses: list[dict]) -> None:
        from simple_coding_agent.github_tracker import GitHubGraphQLTransport

        self._delegate = GitHubGraphQLTransport("token")
        self._responses = iter(responses)
        self.queries: list[str] = []
        self._delegate._execute = self._execute  # type: ignore[method-assign]

    def _execute(self, query: str, variables: dict[str, object]) -> dict:
        self.queries.append(query)
        return next(self._responses)

    def find_pull_request(self, repository: str, head: str):
        return self._delegate.find_pull_request(repository, head)

    def find_attempt_comment(self, repository: str, issue_number: int, marker: str):
        return self._delegate.find_attempt_comment(repository, issue_number, marker)

    def list_issue_comments(self, repository: str, issue_number: int):
        return self._delegate.list_issue_comments(repository, issue_number)


def issue(
    number: int,
    *,
    created_at: datetime | None = None,
    title: str = "Issue title",
    body: str = "Issue body",
    author_login: str = "reporter",
) -> TrackerIssue:
    return TrackerIssue(
        id=f"issue-{number}",
        number=number,
        title=title,
        body=body,
        created_at=created_at or datetime(2026, 9, 20, 13, 0, tzinfo=UTC) + timedelta(seconds=number),
        state="OPEN",
        labels=frozenset({"ready-for-agent"}),
        assignee_logins=(),
        blocked_by=0,
        author_login=author_login,
    )


class FakeTransport:
    def __init__(
        self,
        *,
        pages: list[IssuePage],
        refreshed: dict[int, TrackerIssue] | None = None,
        releases: dict[int, TrackerIssue] | None = None,
    ) -> None:
        self._pages = pages
        self._refreshed = refreshed or {}
        self._releases = releases or {}
        self.page_cursors: list[str | None] = []
        self.assignments: list[tuple[str, str]] = []
        self.removed_labels: list[tuple[str, str]] = []
        self.removed_assignees: list[tuple[str, str]] = []

    def list_issues(self, repository: str, cursor: str | None) -> IssuePage:
        assert repository == "octo/example"
        self.page_cursors.append(cursor)
        return self._pages[len(self.page_cursors) - 1]

    def get_issue(self, repository: str, number: int) -> TrackerIssue | None:
        assert repository == "octo/example"
        return self._releases.get(number, self._refreshed.get(number)) if self.removed_labels else self._refreshed.get(number)

    def viewer(self) -> GitHubIdentity:
        return GitHubIdentity(id="viewer-id", login="agent")

    def assign_issue(self, issue_id: str, assignee_id: str) -> Assignment:
        self.assignments.append((issue_id, assignee_id))
        return Assignment(issue_id=issue_id, assignee_id=assignee_id)

    def remove_label(self, issue_id: str, label: str) -> None:
        self.removed_labels.append((issue_id, label))

    def remove_assignee(self, issue_id: str, assignee_id: str) -> None:
        self.removed_assignees.append((issue_id, assignee_id))

    def list_issue_comments(self, repository: str, issue_number: int) -> tuple[IssueComment, ...]:
        return ()


class FakeCommentTransport:
    def __init__(self, *, comments: tuple[IssueComment, ...]) -> None:
        self._comments = comments

    def viewer(self) -> GitHubIdentity:
        return GitHubIdentity(id="viewer-id", login="agent")

    def list_issue_comments(self, repository: str, issue_number: int) -> tuple[IssueComment, ...]:
        assert repository == "octo/example"
        return self._comments
