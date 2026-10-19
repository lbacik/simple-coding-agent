"""Conflict-attempt budget: escalate to a human after N failures (issue #272).

Consecutive failed conflict attempts on an issue count toward
``MAX_CONFLICT_ATTEMPTS``; the attempt whose failure reaches the budget
escalates with its own result comment and a ``ready-for-human`` label. A
human hands the issue back by removing the label, which requeues with the
full budget. The budget is enforced at escalation time, never at claim
time, and the agent never removes ``ready-for-human``.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from simple_coding_agent.attempt_state import (
    ATTEMPT_KIND_CONFLICT,
    ATTEMPT_KIND_ORDINARY,
    AttemptStateStore,
)
from simple_coding_agent.command_runner import CommandResult
from simple_coding_agent.completion import AttemptOutcome
from simple_coding_agent.config import RepositoryProfile
from simple_coding_agent.conflict_attempt import is_conflict_claim
from simple_coding_agent.github_tracker import (
    GitHubIdentity,
    GitHubTracker,
    IssueComment,
    TrackerIssue,
    ineligibility_reason,
    is_conflict_detection_eligible,
)
from simple_coding_agent.lifecycle import AgentLifecycle
from simple_coding_agent.model_execution import ModelExecutionStatus
from simple_coding_agent.publication import (
    ConflictPublicationRequest,
    PullRequest,
    Publisher,
    _conflict_result_comment,
)
from simple_coding_agent.recovery import (
    CONFLICT_ESCALATION_MARKER,
    CONFLICT_OUTCOME_NO_CHANGE,
    conflict_attempt_marker,
    count_conflict_failures,
    is_conflict_escalation,
    is_conflict_failure,
    parse_conflict_attempt,
)
from tests.fakes import (
    FakeCommandRunner,
    FakeConflictExecutor,
    FakePublisher,
    FakeTracker,
    InMemoryWorkspace,
)

CONFLICT = frozenset({"merge-conflict"})
HUMAN = frozenset({"ready-for-human"})
CONFLICT_HUMAN = frozenset({"merge-conflict", "ready-for-human"})

_CREATED_BASE = datetime(2026, 9, 20, 13, 0, tzinfo=UTC)


def issue(number: int = 7, **overrides: object) -> TrackerIssue:
    base: dict[str, object] = {
        "id": f"issue-{number}",
        "number": number,
        "title": f"Issue {number}",
        "body": f"Implement {number}.",
        "created_at": _CREATED_BASE + timedelta(seconds=number),
        "state": "OPEN",
        "labels": CONFLICT,
        "assignee_logins": (),
        "blocked_by": 0,
        "author_login": "reporter",
    }
    base.update(overrides)
    return TrackerIssue(**base)  # type: ignore[arg-type]


def pull_request(number: int = 7, **overrides: object) -> PullRequest:
    base: dict[str, object] = {
        "number": number,
        "url": f"https://example.test/pull/{number}",
        "node_id": f"pr-{number}",
        "draft": False,
        "mergeable": "CONFLICTING",
        "base_ref": "main",
    }
    base.update(overrides)
    return PullRequest(**base)  # type: ignore[arg-type]


def profile() -> RepositoryProfile:
    return RepositoryProfile(
        setup=("setup.sh",),
        check=("pytest",),
        base_branch="main",
        timeout=300,
        setup_timeout=120,
        env={},
    )


def fail_body(outcome: str = "rebase_conflict", sha: str = "none") -> str:
    """A conflict-attempt failure result carrying both markers."""

    return (
        f"## Conflict Attempt Result: {outcome}\n\nRebase hit conflicts.\n\n"
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->\n"
        f"{conflict_attempt_marker(outcome, sha=sha)}"
    )


def escalation_body(outcome: str = "rebase_conflict") -> str:
    """The escalating attempt's own result comment, carrying the stop marker."""

    return f"{fail_body(outcome)}\n{CONFLICT_ESCALATION_MARKER}"


def ordinary_body() -> str:
    """An ordinary attempt result: stops the streak without the conflict marker."""

    return (
        "## Agent Attempt Result: handoff\n\nHalf done.\n\n"
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->\n"
        "<!-- requeue-source: human -->"
    )


def raising_runner(claim: object, profile: object, prepared: object) -> object:
    raise AssertionError("a conflict attempt must never call the ordinary model path")


def failing_workspace(tmp_path: Path) -> InMemoryWorkspace:
    """A workspace whose rebase conflict the model leaves unresolved."""

    workspace = InMemoryWorkspace(tmp_path / "repo", resolve_after_model=False)
    workspace.prepare_conflicts = ("a.txt",)
    return workspace


def run_budget(
    tmp_path: Path,
    tracker: FakeTracker,
    store_dir: Path,
    *,
    workspace: InMemoryWorkspace | None = None,
    budget: int = 2,
    command_results: list[CommandResult] | None = None,
    executor_status: ModelExecutionStatus = ModelExecutionStatus.SUCCEEDED,
) -> tuple[object, ...]:
    """Claim the queued conflict issue through ``run_once`` and return the harness."""

    active_workspace = workspace or failing_workspace(tmp_path)
    publisher = FakePublisher(pull_requests={"agent/issue-7": pull_request()})
    active_runner = FakeCommandRunner(list(command_results or []))
    executor = FakeConflictExecutor(status=executor_status)
    executor.after_execute = active_workspace.note_model_finished
    events: list[tuple[str, str, str, int | None]] = []

    def event_log(
        event: str,
        detail: str = "",
        level: str = "INFO",
        issue_number: int | None = None,
    ) -> None:
        events.append((event, detail, level, issue_number))

    lifecycle = AgentLifecycle(
        tracker=tracker,  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(store_dir),
        workspace=active_workspace,
        profile_loader=lambda _: profile(),
        publisher=publisher,
        attempt_runner=raising_runner,  # type: ignore[arg-type]
        control_store=None,
        sleeper=lambda seconds: None,
        event_log=event_log,
        max_conflict_attempts=budget,
        command_runner=active_runner,  # type: ignore[arg-type]
        conflict_model_executor=executor,
        conflict_issue_comments=lambda _: (),
    )
    result = lifecycle.run_once()
    return result, tracker, publisher, active_workspace, active_runner, events


def event_names(events: list[tuple[str, str, str, int | None]]) -> list[str]:
    return [event[0] for event in events]


class BudgetTransport:
    """The comment/listing/label seams the budget needs from a transport."""

    def __init__(self, comments: tuple[IssueComment, ...] = ()) -> None:
        self._comments = comments
        self.added_labels: list[tuple[str, str]] = []

    def viewer(self) -> GitHubIdentity:
        return GitHubIdentity(id="viewer-id", login="agent")

    def list_issue_comments(self, repository: str, issue_number: int) -> tuple[IssueComment, ...]:
        assert repository == "octo/example"
        return self._comments

    def add_label(self, issue_id: str, label: str) -> None:
        self.added_labels.append((issue_id, label))


def budget_tracker(
    comments: tuple[IssueComment, ...] = (),
    transport: BudgetTransport | None = None,
    **kwargs: object,
) -> tuple[GitHubTracker, BudgetTransport]:
    active = transport if transport is not None else BudgetTransport(comments)
    return GitHubTracker(active, "octo/example", **kwargs), active  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# streak counter: failures, stops, logins
# ---------------------------------------------------------------------------


def test_no_change_counts_as_a_failure() -> None:
    assert is_conflict_failure(fail_body("no_change")) is True


def test_each_failure_kind_counts() -> None:
    for outcome in (
        "rebase_conflict",
        "check_failed",
        "setup_failed",
        "push_failed",
        "error",
        "model_limit",
        "resolution_failed",
        "no_change",
    ):
        assert is_conflict_failure(fail_body(outcome)) is True, outcome
    # A future unknown outcome is still a failure: it can never be written
    # through conflict_attempt_marker (which rejects it, see
    # test_conflict_marker_rejects_invalid_values), so build it raw.
    future_body = (
        "## Conflict Attempt Result: mystery\n\nRebase hit conflicts.\n\n"
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->\n"
        "<!-- conflict-attempt: outcome=mystery_future_outcome sha=none -->"
    )
    assert is_conflict_failure(future_body) is True


def test_success_skipped_and_escalation_are_not_failures() -> None:
    assert is_conflict_failure(fail_body("success", sha="abc123")) is False
    assert is_conflict_failure(fail_body("skipped", sha="none")) is False
    assert is_conflict_failure(escalation_body()) is False
    assert is_conflict_failure(ordinary_body()) is False


def test_success_without_a_push_counts_as_a_failure() -> None:
    assert is_conflict_failure(fail_body("success", sha="none")) is True


def test_streak_stops_at_success_ordinary_skipped_or_escalation() -> None:
    assert count_conflict_failures([fail_body(), fail_body("success", sha="x")]) == 1
    assert count_conflict_failures([fail_body(), ordinary_body()]) == 1
    assert count_conflict_failures([fail_body(), fail_body("skipped")]) == 1
    assert count_conflict_failures([fail_body("skipped"), fail_body()]) == 0
    assert (
        count_conflict_failures([fail_body(), escalation_body(), fail_body()]) == 1
    )


def test_streak_counts_consecutive_failures_only() -> None:
    assert count_conflict_failures([]) == 0
    assert count_conflict_failures([fail_body(), fail_body(), fail_body()]) == 3


def test_production_streak_ignores_other_logins_and_deleted_accounts() -> None:
    tracker, _ = budget_tracker(
        (
            IssueComment(author_login="agent", body=fail_body()),
            IssueComment(author_login="random-passerby", body=fail_body()),
            IssueComment(author_login="", body=fail_body()),
            IssueComment(author_login="agent", body=fail_body("check_failed")),
        )
    )

    assert tracker.conflict_failure_streak(issue()) == 2


def test_production_streak_counts_peer_logins(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("PEER_LOGINS", "pool-mate")
    tracker, _ = budget_tracker(
        (
            IssueComment(author_login="pool-mate", body=fail_body("check_failed")),
            IssueComment(author_login="agent", body=fail_body()),
        )
    )

    assert tracker.conflict_failure_streak(issue()) == 2


def test_production_streak_counts_explicit_peer_logins() -> None:
    tracker, _ = budget_tracker(
        (IssueComment(author_login="pool-mate", body=fail_body()),),
        peer_logins=["pool-mate"],
    )

    assert tracker.conflict_failure_streak(issue()) == 1


def test_fake_streak_mirrors_the_production_rule() -> None:
    tracker = FakeTracker(
        {7: issue()},
        comments={
            7: [
                ("agent", fail_body()),
                ("random-passerby", fail_body()),
                ("", fail_body()),
            ]
        },
    )

    assert tracker.conflict_failure_streak(issue()) == 1


# ---------------------------------------------------------------------------
# escalation: N consecutive failures hand the issue to a human, exactly once
# ---------------------------------------------------------------------------


def test_two_consecutive_failures_escalate_exactly_once(tmp_path: Path) -> None:
    tracker = FakeTracker(
        {7: issue()},
        fifo=[7],
        comments={7: [("agent", fail_body("check_failed"))]},
    )
    result, tracker, publisher, _, _, events = run_budget(
        tmp_path, tracker, tmp_path / "store1"
    )

    assert result.outcome is AttemptOutcome.INCOMPLETE
    request = publisher.conflict_requests[0]
    assert request.conflict_outcome == "rebase_conflict"
    assert request.conflict_prior == 1
    assert request.conflict_index == 2
    assert request.conflict_budget == 2
    assert request.escalate is True
    assert tracker.escalated == [7]
    assert ("issue-7", "ready-for-human") in tracker.added_labels
    assert "conflict_escalated" in event_names(events)


def test_first_failure_does_not_escalate(tmp_path: Path) -> None:
    tracker = FakeTracker({7: issue()}, fifo=[7])
    _, _, publisher, _, _, events = run_budget(tmp_path, tracker, tmp_path / "store1")

    request = publisher.conflict_requests[0]
    assert (request.conflict_prior, request.conflict_index) == (0, 1)
    assert request.escalate is False
    assert tracker.escalated == []
    assert "conflict_escalated" not in event_names(events)


def test_lowered_budget_escalates_on_the_next_failure(tmp_path: Path) -> None:
    tracker = FakeTracker(
        {7: issue()},
        fifo=[7],
        comments={7: [("agent", fail_body()), ("agent", fail_body())]},
    )
    _, _, publisher, _, _, _ = run_budget(
        tmp_path, tracker, tmp_path / "store1", budget=1
    )

    request = publisher.conflict_requests[0]
    assert request.conflict_index == 3
    assert request.escalate is True
    assert tracker.escalated == [7]


def test_success_resets_the_standing_to_zero(tmp_path: Path) -> None:
    tracker = FakeTracker(
        {7: issue()},
        fifo=[7],
        comments={7: [("agent", fail_body())]},
    )
    workspace = InMemoryWorkspace(tmp_path / "repo")
    publisher = FakePublisher(pull_requests={"agent/issue-7": pull_request()})
    events: list[tuple[str, str, str, int | None]] = []
    lifecycle = AgentLifecycle(
        tracker=tracker,  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(tmp_path / "store1"),
        workspace=workspace,
        profile_loader=lambda _: profile(),
        publisher=publisher,
        attempt_runner=raising_runner,  # type: ignore[arg-type]
        control_store=None,
        sleeper=lambda seconds: None,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level, issue_number)
        ),
        max_conflict_attempts=2,
        command_runner=FakeCommandRunner([]),
        conflict_model_executor=FakeConflictExecutor(),
        conflict_issue_comments=lambda _: (),
    )
    request = lifecycle._build_conflict_request(
        lifecycle._tracker.recover_claim(7),  # type: ignore[union-attr]
        lifecycle._attempt_state.start(
            issue_number=7,
            branch="agent/issue-7",
            requeue_source="human",
            attempt_kind=ATTEMPT_KIND_CONFLICT,
        ),
        "agent/issue-7",
        "main",
        pull_request(),
        "success",
        "",
        "Rebased cleanly.",
        push_branch=True,
    )
    stamped = lifecycle._stamp_conflict_budget(
        lifecycle._tracker.recover_claim(7),  # type: ignore[union-attr]
        request,
    )

    assert (stamped.conflict_prior, stamped.conflict_index) == (1, 0)
    assert stamped.escalate is False


def test_production_escalation_adds_the_human_label() -> None:
    tracker, transport = budget_tracker(max_conflict_attempts=2)

    tracker.escalate_conflict_to_human(issue())

    assert transport.added_labels == [("issue-7", "ready-for-human")]


# ---------------------------------------------------------------------------
# hand-back: removing the label requeues with the full budget
# ---------------------------------------------------------------------------


def test_hand_back_requeues_and_needs_n_more_failures(tmp_path: Path) -> None:
    tracker = FakeTracker(
        {7: issue()},
        fifo=[7],
        comments={7: [("agent", fail_body("check_failed"))]},
        max_conflict_attempts=2,
    )
    run_budget(tmp_path, tracker, tmp_path / "store1")
    assert tracker.escalated == [7]
    # The release dropped the conflict signal while the escalation added the
    # human gate; the detector must not relabel until a human removes it.
    tracker.issues[7] = replace(tracker.issues[7], labels=HUMAN)
    assert tracker.detect_conflict_requeues() == []
    assert "merge-conflict" not in tracker.issues[7].labels

    # The human removes the label: the escalation comment stops the streak,
    # so the next poll relabels and the next failure stands at 1/2.
    tracker.comments[7].append(("agent", escalation_body("rebase_conflict")))
    tracker.issues[7] = replace(tracker.issues[7], labels=frozenset())
    tracker.open_pull_requests = {"agent/issue-7": pull_request()}
    tracker.has_pull_request_listing = True
    assert tracker.detect_conflict_requeues() == [(7, 7, "")]
    assert "merge-conflict" in tracker.issues[7].labels

    tracker.fifo = [7]
    _, _, publisher, _, _, events = run_budget(
        tmp_path, tracker, tmp_path / "store2", workspace=failing_workspace(tmp_path)
    )
    request = publisher.conflict_requests[0]
    assert (request.conflict_index, request.conflict_budget) == (1, 2)
    assert request.escalate is False
    assert tracker.escalated == [7]
    assert "conflict_escalated" not in event_names(events)

    # One more consecutive failure reaches the budget and escalates again.
    tracker.comments[7].append(("agent", fail_body()))
    tracker.fifo = [7]
    _, _, publisher, _, _, events = run_budget(
        tmp_path, tracker, tmp_path / "store3", workspace=failing_workspace(tmp_path)
    )
    request = publisher.conflict_requests[0]
    assert request.conflict_index == 2
    assert request.escalate is True
    assert tracker.escalated == [7, 7]


# ---------------------------------------------------------------------------
# gate: ready-for-human is never labelled, claimed, or removed by the agent
# ---------------------------------------------------------------------------


def test_detector_skips_human_gated_issues() -> None:
    gated = issue(labels=CONFLICT_HUMAN)

    assert is_conflict_claim(gated.labels) is False
    assert (
        is_conflict_detection_eligible(gated, draft=False, mergeable="CONFLICTING")
        is False
    )
    assert (
        is_conflict_detection_eligible(
            issue(labels=HUMAN), draft=False, mergeable="CONFLICTING"
        )
        is False
    )


def test_queue_rejects_human_gated_issues_as_human_work() -> None:
    gated = issue(labels=CONFLICT_HUMAN)

    assert (
        ineligibility_reason(gated, 7, max_conflict_attempts=2)
        == "issue #7 requires human implementation (labelled `ready-for-human`)"
    )


def test_production_queue_never_claims_a_human_gated_issue() -> None:
    tracker, _ = budget_tracker(max_conflict_attempts=2)

    assert tracker.is_conflict_eligible(issue(labels=CONFLICT_HUMAN)) is False
    assert tracker.is_conflict_eligible(issue(labels=HUMAN)) is False


def test_escalating_attempt_never_removes_ready_for_human(tmp_path: Path) -> None:
    tracker = FakeTracker(
        {7: issue()},
        fifo=[7],
        comments={7: [("agent", fail_body("check_failed"))]},
    )
    run_budget(tmp_path, tracker, tmp_path / "store1")

    assert tracker.escalated == [7]
    assert tracker.cleanup == [(7, "merge-conflict", "agent-id")]
    assert all(label != "ready-for-human" for _, label in tracker.removed_labels)


# ---------------------------------------------------------------------------
# comment content: every result shows k/N; escalation carries the handoff
# ---------------------------------------------------------------------------


def conflict_request(**overrides: object) -> ConflictPublicationRequest:
    base: dict[str, object] = {
        "issue_number": 7,
        "issue_title": "Issue 7",
        "branch": "agent/issue-7",
        "started_at": "2026-10-11T10:00:00Z",
        "base_branch": "main",
        "outcome": AttemptOutcome.INCOMPLETE,
        "conflict_outcome": "rebase_conflict",
        "reason": "",
        "details": "Rebase hit conflicts.",
        "push_branch": False,
        "pull_request": pull_request(),
        "conflict_index": 1,
        "conflict_budget": 2,
        "escalate": False,
        "conflict_prior": 0,
        "failure_output": "",
    }
    base.update(overrides)
    return ConflictPublicationRequest(**base)  # type: ignore[arg-type]


def test_every_conflict_result_shows_the_standing() -> None:
    assert "**Conflict attempts**: 1/2" in _conflict_result_comment(
        conflict_request(),
        AttemptOutcome.INCOMPLETE,
        "rebase_conflict",
        None,
        pull_request(),
        "Rebase hit conflicts.",
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->",
        None,
        None,
    )
    assert "**Conflict attempts**: 0/2" in _conflict_result_comment(
        conflict_request(conflict_outcome="success", conflict_index=0),
        AttemptOutcome.COMPLETE,
        "success",
        "https://github.com/octo/example/tree/agent/issue-7",
        pull_request(),
        "Rebased cleanly.",
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->",
        None,
        "abc123",
    )
    assert "**Conflict attempts**: 0/2" in _conflict_result_comment(
        conflict_request(conflict_outcome="skipped", reason="draft", conflict_index=0),
        AttemptOutcome.NO_CHANGES,
        "skipped",
        None,
        None,
        "Nothing to do.",
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->",
        None,
        None,
    )


def test_zero_budget_omits_the_standing_line() -> None:
    body = _conflict_result_comment(
        conflict_request(conflict_budget=0, conflict_index=0),
        AttemptOutcome.INCOMPLETE,
        "rebase_conflict",
        None,
        pull_request(),
        "Rebase hit conflicts.",
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->",
        None,
        None,
    )

    assert "Conflict attempts" not in body


def test_escalation_comment_carries_the_handoff() -> None:
    body = _conflict_result_comment(
        conflict_request(
            escalate=True,
            conflict_index=2,
            failure_output="Failing command: `pytest`\n\nstdout (tail):\n```\nboom\n```",
        ),
        AttemptOutcome.INCOMPLETE,
        "check_failed",
        None,
        pull_request(),
        "The profile check did not pass.",
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->",
        None,
        None,
    )

    assert is_conflict_escalation(body) is True
    assert parse_conflict_attempt(body)["outcome"] == "check_failed"
    assert "**Conflict attempts**: 2/2" in body
    assert "<details>" in body
    assert "Failing command: `pytest`" in body
    assert "[#7](https://example.test/pull/7)" in body
    assert "Remove the `ready-for-human` label to hand the issue back for 2 more attempts." in body
    assert "@" not in body


def test_escalation_falls_back_to_the_failure_reason() -> None:
    body = _conflict_result_comment(
        conflict_request(escalate=True, conflict_index=2),
        AttemptOutcome.INCOMPLETE,
        "rebase_conflict",
        None,
        pull_request(),
        "Rebase hit conflicts.",
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->",
        None,
        None,
    )

    assert is_conflict_escalation(body) is True
    assert "Rebase hit conflicts." in body
    assert "ready-for-human" in body


def test_ordinary_results_carry_no_escalation_marker() -> None:
    body = _conflict_result_comment(
        conflict_request(),
        AttemptOutcome.INCOMPLETE,
        "rebase_conflict",
        None,
        pull_request(),
        "Rebase hit conflicts.",
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->",
        None,
        None,
    )

    assert is_conflict_escalation(body) is False
    assert "ready-for-human" not in body


class _ConflictGitHub:
    """The comment seam the production publisher needs for a conflict result."""

    def __init__(self) -> None:
        self.comments: list[str] = []

    def find_attempt_comment(
        self, repository: str, issue_number: int, marker: str
    ) -> bool:
        return False

    def add_comment(self, repository: str, issue_number: int, body: str) -> None:
        self.comments.append(body)


class _ConflictWorkspace:
    """A pusher double: failures never reach the push."""

    pushed_sha = "abc123"

    def push_attempt_branch(
        self, branch: str, *, max_retries: int, deadline: float | None = None
    ) -> str:
        return self.pushed_sha


def test_production_publish_posts_the_escalation_comment(tmp_path: Path) -> None:
    github = _ConflictGitHub()
    publisher = Publisher(_ConflictWorkspace(), github, "octo/example")  # type: ignore[arg-type]

    result = publisher.publish_conflict(
        conflict_request(
            escalate=True,
            conflict_index=2,
            failure_output="Failing command: `pytest`",
        )
    )

    assert result.escalated is True
    assert result.conflict_outcome == "rebase_conflict"
    assert len(github.comments) == 1
    assert is_conflict_escalation(github.comments[0]) is True
    assert "**Conflict attempts**: 2/2" in github.comments[0]


def test_production_publish_reports_an_unchanged_push_as_no_change() -> None:
    github = _ConflictGitHub()
    workspace = _ConflictWorkspace()
    workspace.pushed_sha = "same-sha"
    publisher = Publisher(workspace, github, "octo/example")  # type: ignore[arg-type]

    result = publisher.publish_conflict(
        conflict_request(
            outcome=AttemptOutcome.COMPLETE,
            conflict_outcome="success",
            push_branch=True,
            pull_request=pull_request(head_sha="same-sha"),
            conflict_index=0,
            conflict_prior=1,
            conflict_budget=2,
            details="Rebased cleanly.",
        )
    )

    assert result.conflict_outcome == CONFLICT_OUTCOME_NO_CHANGE
    assert result.escalated is True
    assert parse_conflict_attempt(github.comments[0])["outcome"] == "no_change"
    # The escalation marker stops the streak, so an escalation comment is
    # never itself a failure (see test_success_skipped_and_escalation...).
    assert is_conflict_failure(github.comments[0]) is False
    assert "**Conflict attempts**: 2/2" in github.comments[0]
    assert is_conflict_escalation(github.comments[0]) is True


# ---------------------------------------------------------------------------
# restart recovery publishes a conflict error that counts toward the budget
# ---------------------------------------------------------------------------


def test_restart_recovery_publishes_a_counted_conflict_error(tmp_path: Path) -> None:
    tracker = FakeTracker(
        {7: issue()},
        comments={7: [("agent", fail_body("check_failed"))]},
    )
    publisher = FakePublisher(pull_requests={"agent/issue-7": pull_request()})
    events: list[tuple[str, str, str, int | None]] = []
    lifecycle = AgentLifecycle(
        tracker=tracker,  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(tmp_path / "store1"),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: profile(),
        publisher=publisher,
        attempt_runner=raising_runner,  # type: ignore[arg-type]
        control_store=None,
        sleeper=lambda seconds: None,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level, issue_number)
        ),
        max_conflict_attempts=2,
        command_runner=FakeCommandRunner([]),
        conflict_model_executor=FakeConflictExecutor(),
        conflict_issue_comments=lambda _: (),
    )
    lifecycle._attempt_state.start(
        issue_number=7,
        branch="agent/issue-7",
        requeue_source="human",
        attempt_kind=ATTEMPT_KIND_CONFLICT,
    )

    assert lifecycle._reconcile_startup() is AttemptOutcome.INFRASTRUCTURE_ERROR
    request = publisher.conflict_requests[0]
    assert request.conflict_outcome == "error"
    assert (request.conflict_prior, request.conflict_index) == (1, 2)
    assert request.escalate is True
    assert tracker.escalated == [7]
    assert not publisher.requests or all(
        getattr(entry, "conflict_outcome", "error") == "error"
        for entry in publisher.conflict_requests
    )


def test_checkpoint_kind_round_trips_and_defaults_ordinary(tmp_path: Path) -> None:
    store = AttemptStateStore(tmp_path / "store1")
    checkpoint = store.start(
        issue_number=7,
        branch="agent/issue-7",
        requeue_source="human",
        attempt_kind=ATTEMPT_KIND_CONFLICT,
    )

    assert checkpoint.attempt_kind == ATTEMPT_KIND_CONFLICT
    assert store.read() is not None and store.read().attempt_kind == ATTEMPT_KIND_CONFLICT  # type: ignore[union-attr]

    legacy = AttemptStateStore(tmp_path / "store2")
    legacy.start(issue_number=7, branch="agent/issue-7", requeue_source="human")

    assert legacy.read() is not None and legacy.read().attempt_kind == ATTEMPT_KIND_ORDINARY  # type: ignore[union-attr]
