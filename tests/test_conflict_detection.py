"""Automatic conflict detection: labelling ``merge-conflict`` from open PRs.

Workers notice on their own that a published agent pull request reports merge
conflicts and requeue its issue with ``merge-conflict``, so a Conflict
attempt starts without a human. The detector runs while the runnable queue is
built (inside ``claim_next``/``runnable_queue``), which places it after every
intake gate and keeps it out of startup reconciliation and priority-claim
polls. Every poll with a positive budget reads the repository's open pull
requests in one paged GraphQL query; with a zero budget no pull-request query
runs at all.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.command_runner import (
    CommandExecution,
    CommandResult,
    CommandStatus,
)
from simple_coding_agent.completion import AttemptOutcome, CompletionDecision, PublicationPath
from simple_coding_agent.config import RepositoryProfile
from simple_coding_agent.control import ControlStore
from simple_coding_agent.github_tracker import (
    MERGE_CONFLICT_LABEL_COLOR,
    MERGE_CONFLICT_LABEL_DESCRIPTION,
    Assignment,
    Claim,
    GitHubIdentity,
    GitHubTracker,
    IssueComment,
    IssuePage,
    OpenPullRequest,
    PullRequestPage,
    TrackerIssue,
    conflict_branch_issue_number,
    is_conflict_detection_eligible,
)
from simple_coding_agent.lifecycle import AgentLifecycle, AttemptEvidence, LifecycleStatus
from simple_coding_agent.publication import PullRequest
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace


_CREATED_BASE = datetime(2026, 9, 20, 13, 0, tzinfo=UTC)


def issue(number: int, **overrides: object) -> TrackerIssue:
    """An otherwise ordinary issue with no queue label at all."""

    base: dict[str, object] = {
        "id": f"issue-{number}",
        "number": number,
        "title": f"Issue {number}",
        "body": f"Implement {number}.",
        "created_at": _CREATED_BASE + timedelta(seconds=number),
        "state": "OPEN",
        "labels": frozenset(),
        "assignee_logins": (),
        "blocked_by": 0,
        "author_login": "reporter",
    }
    base.update(overrides)
    return TrackerIssue(**base)  # type: ignore[arg-type]


def open_pr(
    number: int,
    head: str,
    *,
    mergeable: str = "CONFLICTING",
    draft: bool = False,
    base: str = "main",
    sha: str = "deadbee",
) -> OpenPullRequest:
    return OpenPullRequest(
        number=number,
        head_ref=head,
        base_ref=base,
        head_sha=sha,
        draft=draft,
        mergeable=mergeable,
    )


class DetectorTransport:
    """Transport double with issue pages, PR pages, and label mutation."""

    def __init__(
        self,
        issue_pages: list[IssuePage],
        pr_pages: list[PullRequestPage] | None = None,
    ) -> None:
        self._issue_pages = issue_pages
        self._pr_pages = list(pr_pages) if pr_pages is not None else None
        self.state = {
            found.number: found for page in issue_pages for found in page.issues
        }
        self.issue_cursors: list[str | None] = []
        self.pr_cursors: list[str | None] = []
        self.assignments: list[tuple[str, str]] = []
        self.added_labels: list[tuple[str, str]] = []
        self.removed_labels: list[tuple[str, str]] = []
        self.removed_assignees: list[tuple[str, str]] = []

    def list_issues(self, repository: str, cursor: str | None) -> IssuePage:
        assert repository == "octo/example"
        self.issue_cursors.append(cursor)
        # Every poll re-reads live state (like GitHub), so a label added by
        # an earlier poll in the same test is visible to the next build.
        return IssuePage(issues=tuple(self.state.values()), next_cursor=None)

    def list_open_pull_requests(
        self, repository: str, cursor: str | None
    ) -> PullRequestPage:
        assert repository == "octo/example"
        assert self._pr_pages is not None
        self.pr_cursors.append(cursor)
        index = min(len(self.pr_cursors) - 1, len(self._pr_pages) - 1)
        return self._pr_pages[index]

    def get_issue(self, repository: str, number: int) -> TrackerIssue | None:
        assert repository == "octo/example"
        return self.state.get(number)

    def viewer(self) -> GitHubIdentity:
        return GitHubIdentity(id="viewer-id", login="agent")

    def assign_issue(self, issue_id: str, assignee_id: str) -> Assignment:
        self.assignments.append((issue_id, assignee_id))
        return Assignment(issue_id=issue_id, assignee_id=assignee_id)

    def remove_label(self, issue_id: str, label: str) -> None:
        self.removed_labels.append((issue_id, label))

    def add_label(self, issue_id: str, label: str) -> None:
        self.added_labels.append((issue_id, label))
        for number, found in self.state.items():
            if found.id == issue_id:
                self.state[number] = replace(found, labels=found.labels | {label})

    def remove_assignee(self, issue_id: str, assignee_id: str) -> None:
        self.removed_assignees.append((issue_id, assignee_id))

    def list_issue_comments(
        self, repository: str, issue_number: int
    ) -> tuple[IssueComment, ...]:
        assert repository == "octo/example"
        return ()


def make_tracker(
    transport: DetectorTransport,
    events: list[tuple[str, str, str, int | None]],
    *,
    budget: int = 1,
) -> GitHubTracker:
    return GitHubTracker(
        transport,
        "octo/example",
        max_conflict_attempts=budget,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level, issue_number)
        ),
    )


def single_issue_transport(
    target: TrackerIssue, pull_requests: list[OpenPullRequest]
) -> DetectorTransport:
    return DetectorTransport(
        [IssuePage(issues=(target,), next_cursor=None)],
        [PullRequestPage(pull_requests=tuple(pull_requests), next_cursor=None)],
    )


# ---------------------------------------------------------------------------
# branch parsing and the labelling rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        ("agent/issue-7", 7),
        ("agent/issue-123", 123),
        ("agent/issue-0", 0),
        ("agent/issue-", None),
        ("agent/issue-abc", None),
        ("agent/issue-7-extra", None),
        ("feature/cool", None),
        ("agent/Issue-7", None),
        ("", None),
    ],
)
def test_conflict_branch_issue_number_takes_n_from_the_branch_name(
    head: str, expected: int | None
) -> None:
    assert conflict_branch_issue_number(head) is expected


def test_detector_labels_a_conflicting_pr_and_the_same_poll_claims_it() -> None:
    transport = single_issue_transport(
        issue(7), [open_pr(42, "agent/issue-7", sha="abc123")]
    )
    events: list[tuple[str, str, str, int | None]] = []
    tracker = make_tracker(transport, events)

    queue = tracker.runnable_queue()

    assert [found.number for found in queue] == [7]
    assert queue[0].labels == frozenset({"merge-conflict"})
    assert transport.added_labels == [("issue-7", "merge-conflict")]
    assert transport.pr_cursors == [None]

    claim = tracker.claim_next()

    assert claim is not None and claim.issue.number == 7
    assert "merge-conflict" in claim.issue.labels
    assert ("issue-7", "merge-conflict") not in transport.removed_labels
    assert events == [
        (
            "conflict_requeue_detected",
            "issue #7; pull request #42 (abc123) reports merge conflicts",
            "INFO",
            7,
        )
    ]


@pytest.mark.parametrize(
    ("target", "pull_requests"),
    [
        # UNKNOWN only starts GitHub's own computation; this poll skips it.
        (issue(7), [open_pr(42, "agent/issue-7", mergeable="UNKNOWN")]),
        (issue(7), [open_pr(42, "agent/issue-7", mergeable="MERGEABLE")]),
        (issue(7), [open_pr(42, "agent/issue-7", draft=True)]),
        (issue(7), [open_pr(42, "feature/cool")]),
        (issue(7), [open_pr(42, "agent/issue-9")]),
        # The issue behind the branch is missing or fails the base queue rules.
        (issue(7, assignee_logins=("human",)), [open_pr(42, "agent/issue-7")]),
        (issue(7, body="  "), [open_pr(42, "agent/issue-7")]),
        (issue(7, blocked_by=2), [open_pr(42, "agent/issue-7")]),
        # Any other queue label already on the issue blocks relabelling.
        (issue(7, labels=frozenset({"round-finished"})), [open_pr(42, "agent/issue-7")]),
        (issue(7, labels=frozenset({"ready-for-human"})), [open_pr(42, "agent/issue-7")]),
    ],
)
def test_detector_leaves_ineligible_pairs_unlabelled(
    target: TrackerIssue, pull_requests: list[OpenPullRequest]
) -> None:
    transport = single_issue_transport(target, pull_requests)
    events: list[tuple[str, str, str, int | None]] = []
    tracker = make_tracker(transport, events)

    assert tracker.runnable_queue() == ()
    assert tracker.claim_next() is None
    assert transport.added_labels == []
    assert events == []


@pytest.mark.parametrize(
    "labels",
    [
        frozenset({"ready-for-agent"}),
        frozenset({"merge-conflict"}),
    ],
)
def test_detector_never_relabels_an_already_queued_issue(
    labels: frozenset[str],
) -> None:
    """An issue carrying a queue label stays in its own group, untouched."""

    transport = single_issue_transport(
        issue(7, labels=labels), [open_pr(42, "agent/issue-7", sha="abc123")]
    )
    events: list[tuple[str, str, str, int | None]] = []
    tracker = make_tracker(transport, events)

    queue = tracker.runnable_queue()

    assert [found.number for found in queue] == [7]
    assert queue[0].labels == labels
    assert transport.added_labels == []
    assert events == []


def test_detector_skips_a_closed_issue_behind_the_branch() -> None:
    transport = single_issue_transport(
        issue(7, state="CLOSED"), [open_pr(42, "agent/issue-7")]
    )
    events: list[tuple[str, str, str, int | None]] = []
    tracker = make_tracker(transport, events)

    assert tracker.runnable_queue() == ()
    assert transport.added_labels == []
    assert events == []


def test_unknown_mergeability_labels_on_a_later_poll_once_conflicting() -> None:
    pr = open_pr(42, "agent/issue-7", mergeable="UNKNOWN", sha="abc123")
    transport = single_issue_transport(issue(7), [pr])
    events: list[tuple[str, str, str, int | None]] = []
    tracker = make_tracker(transport, events)

    assert tracker.runnable_queue() == ()
    assert events == []

    transport._pr_pages = [
        PullRequestPage(
            pull_requests=(replace(pr, mergeable="CONFLICTING"),), next_cursor=None
        )
    ]

    assert [found.number for found in tracker.runnable_queue()] == [7]
    assert events == [
        (
            "conflict_requeue_detected",
            "issue #7; pull request #42 (abc123) reports merge conflicts",
            "INFO",
            7,
        )
    ]


def test_second_poll_does_not_relabel_an_already_queued_issue() -> None:
    transport = single_issue_transport(
        issue(7), [open_pr(42, "agent/issue-7", sha="abc123")]
    )
    tracker = make_tracker(transport, [])

    assert [found.number for found in tracker.runnable_queue()] == [7]
    assert [found.number for found in tracker.runnable_queue()] == [7]
    assert transport.added_labels == [("issue-7", "merge-conflict")]


# ---------------------------------------------------------------------------
# cadence: every poll with a budget, never without one
# ---------------------------------------------------------------------------


def test_zero_budget_makes_no_pr_query_and_labels_nothing() -> None:
    transport = single_issue_transport(
        issue(7), [open_pr(42, "agent/issue-7", sha="abc123")]
    )
    events: list[tuple[str, str, str, int | None]] = []
    tracker = make_tracker(transport, events, budget=0)

    assert tracker.runnable_queue() == ()
    assert tracker.claim_next() is None
    assert transport.pr_cursors == []
    assert transport.added_labels == []
    assert events == []


def test_open_prs_are_read_in_one_paged_query_per_poll() -> None:
    first_page = PullRequestPage(
        pull_requests=(open_pr(42, "agent/issue-7", sha="aaa"),),
        next_cursor="cursor-1",
    )
    second_page = PullRequestPage(
        pull_requests=(open_pr(43, "agent/issue-9", sha="bbb"),),
        next_cursor=None,
    )
    transport = DetectorTransport(
        [IssuePage(issues=(issue(7), issue(9)), next_cursor=None)],
        [first_page, second_page, first_page, second_page],
    )
    events: list[tuple[str, str, str, int | None]] = []
    tracker = make_tracker(transport, events)

    assert [found.number for found in tracker.runnable_queue()] == [7, 9]
    assert transport.pr_cursors == [None, "cursor-1"]
    assert transport.added_labels == [
        ("issue-7", "merge-conflict"),
        ("issue-9", "merge-conflict"),
    ]
    assert [event[0] for event in events] == [
        "conflict_requeue_detected",
        "conflict_requeue_detected",
    ]

    # The next poll re-reads the listing once more and labels nothing new.
    assert [found.number for found in tracker.runnable_queue()] == [7, 9]
    assert transport.pr_cursors == [None, "cursor-1", None, "cursor-1"]
    assert transport.added_labels == [
        ("issue-7", "merge-conflict"),
        ("issue-9", "merge-conflict"),
    ]
    assert len(events) == 2


def test_detector_never_counts_the_continuation_budget() -> None:
    transport = single_issue_transport(
        issue(7), [open_pr(42, "agent/issue-7", sha="abc123")]
    )

    def _no_comment_reads(
        repository: str, issue_number: int
    ) -> tuple[IssueComment, ...]:
        raise AssertionError("the detector must never read issue comments")

    transport.list_issue_comments = _no_comment_reads  # type: ignore[method-assign]
    tracker = make_tracker(transport, [])

    assert [found.number for found in tracker.runnable_queue()] == [7]


# ---------------------------------------------------------------------------
# GraphQL transport: fields, paging, label creation
# ---------------------------------------------------------------------------


class StubGraphQLTransport:
    """Drive the production transport with canned ``_execute`` responses."""

    def __init__(self, responses: list[dict]) -> None:
        from simple_coding_agent.github_tracker import GitHubGraphQLTransport

        self._delegate = GitHubGraphQLTransport("token")
        self._responses = iter(responses)
        self.queries: list[str] = []
        self.variables: list[dict[str, object]] = []
        self._delegate._execute = self._execute  # type: ignore[method-assign]

    def _execute(self, query: str, variables: dict[str, object]) -> dict:
        self.queries.append(query)
        self.variables.append(variables)
        return next(self._responses)

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


def test_graphql_lists_open_pull_requests_with_detector_fields_only() -> None:
    transport = StubGraphQLTransport(
        [
            {
                "repository": {
                    "pullRequests": {
                        "nodes": [
                            {
                                "number": 42,
                                "headRefName": "agent/issue-7",
                                "baseRefName": "main",
                                "headRefOid": "abc123",
                                "isDraft": False,
                                "mergeable": "CONFLICTING",
                            }
                        ],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }
        ]
    )

    page = transport.list_open_pull_requests("octo/example", None)

    assert page == PullRequestPage(
        pull_requests=(
            OpenPullRequest(42, "agent/issue-7", "main", "abc123", False, "CONFLICTING"),
        ),
        next_cursor=None,
    )
    query = transport.queries[0]
    for field in (
        "number",
        "headRefName",
        "baseRefName",
        "headRefOid",
        "isDraft",
        "mergeable",
    ):
        assert field in query
    assert "states: OPEN" in query
    assert "baseRefName: $" not in query
    for connection in ("comments", "reviews", "commits", "files"):
        assert connection not in query
    assert transport.variables[0]["cursor"] is None


def test_graphql_rejects_an_invalid_pull_request_page() -> None:
    from simple_coding_agent.github_tracker import GitHubTrackerError

    transport = StubGraphQLTransport([{"repository": {"pullRequests": None}}])

    with pytest.raises(GitHubTrackerError):
        transport.list_open_pull_requests("octo/example", None)


def test_graphql_creates_a_missing_merge_conflict_label_with_its_colour() -> None:
    transport = StubGraphQLTransport(
        [
            {"node": {"repository": {"id": "repo-id-1", "label": None}}},
            {"createLabel": {"label": {"id": "created-label-id"}}},
            {},
        ]
    )

    transport.add_label("issue-id", "merge-conflict")

    assert "createLabel" in transport.queries[1]
    assert transport.variables[1]["name"] == "merge-conflict"
    assert transport.variables[1]["color"] == MERGE_CONFLICT_LABEL_COLOR
    assert transport.variables[1]["description"] == MERGE_CONFLICT_LABEL_DESCRIPTION
    assert MERGE_CONFLICT_LABEL_COLOR
    assert MERGE_CONFLICT_LABEL_DESCRIPTION == (
        "Agent PR has merge conflicts; queued for a Conflict attempt"
    )


# ---------------------------------------------------------------------------
# lifecycle: the same poll claims what the detector labelled
# ---------------------------------------------------------------------------


def profile() -> RepositoryProfile:
    return RepositoryProfile(
        setup=("setup.sh",),
        check=("pytest",),
        base_branch="main",
        timeout=300,
        setup_timeout=120,
        env={},
    )


def ok_result(command: str = "pytest") -> CommandResult:
    return CommandResult(
        CommandStatus.SUCCEEDED,
        (CommandExecution(command, 0, "", "", False),),
        0.1,
    )


class FakeCommandRunner:
    def __init__(self, results: list[CommandResult]) -> None:
        self._results = list(results)

    def run(
        self,
        commands: tuple[str, ...],
        *,
        timeout: float,
        cwd: Path,
        environment: object = None,
    ) -> CommandResult:
        return self._results.pop(0)


def raising_runner(claim: object, profile: object, prepared: object) -> object:
    raise AssertionError("a conflict attempt must never call the model")


def test_lifecycle_claims_a_detected_issue_in_the_same_poll(tmp_path: Path) -> None:
    published = PullRequest(
        42, "https://example.test/pull/42", "pr-42", False, "CONFLICTING", "main", "abc123"
    )
    tracker_events: list[tuple[str, str, str, int | None]] = []
    lifecycle_events: list[tuple[str, str, str, int | None]] = []
    tracker = FakeTracker(
        {7: issue(7)},
        fifo=[],
        open_pull_requests={"agent/issue-7": published},
        max_conflict_attempts=1,
        event_log=lambda event, detail="", level="INFO", issue_number=None: tracker_events.append(
            (event, detail, level, issue_number)
        ),
    )
    lifecycle = AgentLifecycle(
        tracker=tracker,  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: profile(),
        publisher=FakePublisher(pull_requests={"agent/issue-7": published}),
        attempt_runner=raising_runner,  # type: ignore[arg-type]
        control_store=None,
        sleeper=lambda seconds: None,
        event_log=lambda event, detail="", level="INFO", issue_number=None: lifecycle_events.append(
            (event, detail, level, issue_number)
        ),
        max_conflict_attempts=1,
        command_runner=FakeCommandRunner([ok_result("setup.sh"), ok_result()]),  # type: ignore[arg-type]
    )

    result = lifecycle.run_once()

    assert result.outcome is AttemptOutcome.COMPLETE
    assert tracker.added_labels == [("issue-7", "merge-conflict")]
    assert tracker_events == [
        (
            "conflict_requeue_detected",
            "issue #7; pull request #42 (abc123) reports merge conflicts",
            "INFO",
            7,
        )
    ]
    assert "conflict_attempt_claimed" in [event[0] for event in lifecycle_events]


def test_lifecycle_makes_no_pr_query_with_a_zero_budget(tmp_path: Path) -> None:
    tracker = FakeTracker(
        {7: issue(7)},
        fifo=[],
        open_pull_requests={
            "agent/issue-7": PullRequest(42, "https://example.test/pull/42")
        },
        max_conflict_attempts=0,
    )
    lifecycle = AgentLifecycle(
        tracker=tracker,  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: profile(),
        publisher=FakePublisher(),
        attempt_runner=raising_runner,  # type: ignore[arg-type]
        control_store=None,
        sleeper=lambda seconds: None,
        max_conflict_attempts=0,
        command_runner=FakeCommandRunner([]),  # type: ignore[arg-type]
    )

    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.IDLE
    assert tracker.pr_queries == []
    assert tracker.added_labels == []


def ordinary_workflow(
    received_claim: Claim, profile: object, prepared: object
) -> AttemptEvidence:
    return AttemptEvidence(
        decision=CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)),
        check_command="pytest",
        check_exit_code=0,
        review_cycles=1,
        review_findings="all clear",
        details="Implemented.",
    )


def test_priority_claim_poll_skips_detection(tmp_path: Path) -> None:
    tracker = FakeTracker(
        {
            7: issue(7),
            8: replace(issue(8), labels=frozenset({"ready-for-agent"})),
        },
        fifo=[],
        open_pull_requests={
            "agent/issue-7": PullRequest(
                42, "https://example.test/pull/42", "pr-42", False, "CONFLICTING", "main", "abc123"
            )
        },
        max_conflict_attempts=1,
    )
    lifecycle = AgentLifecycle(
        tracker=tracker,  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: profile(),
        publisher=FakePublisher(),
        attempt_runner=ordinary_workflow,  # type: ignore[arg-type]
        control_store=ControlStore(tmp_path),
        sleeper=lambda seconds: None,
        max_conflict_attempts=1,
        command_runner=FakeCommandRunner([]),  # type: ignore[arg-type]
    )

    lifecycle.submit_next_issue("req-1", 8)
    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.ATTEMPTED
    assert tracker.claimed_verified == [8]
    # The prioritized poll never built the runnable queue.
    assert tracker.pr_queries == []
    assert tracker.added_labels == []
    assert 7 not in tracker.claimed


def test_stopped_intake_skips_detection(tmp_path: Path) -> None:
    tracker = FakeTracker(
        {7: issue(7)},
        fifo=[7],
        open_pull_requests={
            "agent/issue-7": PullRequest(
                42, "https://example.test/pull/42", "pr-42", False, "CONFLICTING", "main", "abc123"
            )
        },
        max_conflict_attempts=1,
    )
    lifecycle = AgentLifecycle(
        tracker=tracker,  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: profile(),
        publisher=FakePublisher(),
        attempt_runner=ordinary_workflow,  # type: ignore[arg-type]
        control_store=ControlStore(tmp_path),
        sleeper=lambda seconds: None,
        max_conflict_attempts=1,
        command_runner=FakeCommandRunner([]),  # type: ignore[arg-type]
    )

    lifecycle.submit_stop("req-stop")
    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.IDLE
    assert tracker.pr_queries == []
    assert tracker.added_labels == []
    assert tracker.claimed == []


def test_startup_reconciliation_skips_detection(tmp_path: Path) -> None:
    state = AttemptStateStore(tmp_path)
    state.start(issue_number=24, branch="agent/issue-24")
    tracker = FakeTracker(
        {
            24: replace(issue(24), labels=frozenset({"ready-for-agent"})),
            7: issue(7),
        },
        fifo=[],
        open_pull_requests={
            "agent/issue-7": PullRequest(
                42, "https://example.test/pull/42", "pr-42", False, "CONFLICTING", "main", "abc123"
            )
        },
        max_conflict_attempts=1,
    )
    lifecycle = AgentLifecycle(
        tracker=tracker,  # type: ignore[arg-type]
        attempt_state=state,
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: profile(),
        publisher=FakePublisher(),
        attempt_runner=ordinary_workflow,  # type: ignore[arg-type]
        control_store=None,
        sleeper=lambda seconds: None,
        max_conflict_attempts=1,
        command_runner=FakeCommandRunner([]),  # type: ignore[arg-type]
    )

    lifecycle.run_once()

    assert tracker.pr_queries == []
    assert tracker.added_labels == []


def test_detection_predicate_matches_the_labelling_rule() -> None:
    eligible = issue(7)

    assert is_conflict_detection_eligible(eligible, draft=False, mergeable="CONFLICTING")
    assert not is_conflict_detection_eligible(
        eligible, draft=False, mergeable="UNKNOWN"
    )
    assert not is_conflict_detection_eligible(
        replace(eligible, labels=frozenset({"merge-conflict"})),
        draft=False,
        mergeable="CONFLICTING",
    )
    assert not is_conflict_detection_eligible(
        SimpleNamespace(
            meets_base_queue_rules=False, labels=frozenset()
        ),  # type: ignore[arg-type]
        draft=False,
        mergeable="CONFLICTING",
    )
