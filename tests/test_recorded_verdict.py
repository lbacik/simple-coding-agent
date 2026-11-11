"""The verdict is recorded when publication decides it (issue #390, ADR 0010)."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

from simple_coding_agent.attempt_state import (
    AttemptPhase,
    AttemptStateStore,
    RecordedVerdict,
)
from simple_coding_agent.completion import (
    AttemptOutcome,
    CompletionDecision,
    PublicationPath,
)
from simple_coding_agent.control import CommandAcknowledgement, ControlStore
from simple_coding_agent.finalization import AttemptCompletionStore
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.lifecycle import AgentLifecycle, AttemptEvidence
from simple_coding_agent.recovery import release_comment_body
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _issue(number: int) -> TrackerIssue:
    return TrackerIssue(
        id=f"issue-{number}",
        number=number,
        title="Record the verdict",
        body="Keep the verdict publication decided.",
        created_at=datetime(2026, 9, 20, tzinfo=UTC),
        state="OPEN",
        labels=frozenset({"ready-for-agent"}),
        assignee_logins=("agent",),
        blocked_by=0,
        author_login="reporter",
    )


class CommentTransport:
    """The publisher's GitHub comment seam, recording release comments."""

    def __init__(self, order: list[str]) -> None:
        self.bodies: list[str] = []
        self._order = order

    def find_attempt_comment(self, repository: str, issue_number: int, marker: str) -> bool:
        return any(marker in body for body in self.bodies)

    def add_comment(self, repository: str, issue_number: int, body: str) -> None:
        self._order.append("comment")
        self.bodies.append(body)


class GithubPublisher(FakePublisher):
    _repository = "owner/repo"

    def __init__(self, order: list[str], **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._github = CommentTransport(order)


class OrderedTracker(FakeTracker):
    def __init__(self, order: list[str], **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._order = order

    def release_attempt(self, issue_number: int, label: str, assignee_id: str) -> None:
        self._order.append("release")
        super().release_attempt(issue_number, label, assignee_id)


def _complete_evidence(claim: Claim, profile: object, prepared: object) -> AttemptEvidence:
    return AttemptEvidence(
        decision=CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)),
        check_command="pytest",
        check_exit_code=0,
        review_cycles=1,
        review_findings="all clear",
        details="Implemented the change.",
    )


def _lifecycle(
    root: Path,
    *,
    publisher: FakePublisher,
    tracker: FakeTracker,
    events: list[tuple[str, str]],
) -> AgentLifecycle:
    return AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(root),
        workspace=InMemoryWorkspace(root / "workspace", issue_number=24),
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=publisher,
        attempt_runner=_complete_evidence,
        control_store=ControlStore(root),
        completion_store=AttemptCompletionStore(root),
        sleeper=lambda seconds: None,
        repository="owner/repo",
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail)
        ),
    )


def _named(events: list[tuple[str, str]], name: str) -> list[str]:
    return [detail for event, detail in events if event == name]


def _checkpoint(root: Path, phase: AttemptPhase) -> str:
    state = AttemptStateStore(root)
    started = state.start(issue_number=24, branch="agent/issue-24")
    for next_phase in (
        AttemptPhase.SETUP,
        AttemptPhase.MODEL_RUNNING,
        AttemptPhase.PUSHING,
        AttemptPhase.PUBLISHING,
    ):
        current = state.read()
        assert current is not None
        if current.phase is phase:
            break
        state.transition(next_phase)
    return started.started_at


def _retain_a_complete_attempt(root: Path) -> tuple[str, list[tuple[str, str]]]:
    """Run one attempt whose publication is complete but unconfirmed."""

    events: list[tuple[str, str]] = []
    tracker = FakeTracker(
        issues={24: _issue(24)},
        next_claim=Claim(_issue(24), Assignment("issue-24", "agent-id")),
    )
    lifecycle = _lifecycle(
        root,
        publisher=FakePublisher(comment_posted=False),
        tracker=tracker,
        events=events,
    )
    lifecycle.run_once()
    checkpoint = AttemptStateStore(root).read()
    assert checkpoint is not None
    return checkpoint.started_at, events


# ---------------------------------------------------------------------------
# recording at publication
# ---------------------------------------------------------------------------


def test_publication_records_the_verdict_before_the_finalization_boundary(
    tmp_path: Path,
) -> None:
    attempt_id, events = _retain_a_complete_attempt(tmp_path)

    checkpoint = AttemptStateStore(tmp_path).read()
    assert checkpoint is not None
    assert checkpoint.started_at == attempt_id
    assert checkpoint.verdict == RecordedVerdict("complete", None)
    assert _named(events, "outcome_recorded") == ["outcome=complete"]
    assert _named(events, "attempt_finished") == []
    assert _named(events, "attempt_finalization_held") != []
    assert AttemptCompletionStore(tmp_path).read_all() == {}


def test_a_finalized_attempt_emits_outcome_recorded_once(tmp_path: Path) -> None:
    events: list[tuple[str, str]] = []
    tracker = FakeTracker(
        issues={24: _issue(24)},
        next_claim=Claim(_issue(24), Assignment("issue-24", "agent-id")),
    )
    lifecycle = _lifecycle(
        tmp_path, publisher=FakePublisher(), tracker=tracker, events=events
    )

    lifecycle.run_once()

    names = [event for event, _ in events]
    assert names.count("outcome_recorded") == 1
    assert names.index("outcome_recorded") < names.index("attempt_finished")
    assert AttemptStateStore(tmp_path).read() is None


def test_startup_reconciliation_keeps_a_recorded_verdict_after_a_restart(
    tmp_path: Path,
) -> None:
    # Publishing and unconfirmed: inference alone would republish this
    # attempt as complete (no handoff note), the verdict says incomplete.
    attempt_id = _checkpoint(tmp_path, AttemptPhase.PUBLISHING)
    AttemptStateStore(tmp_path).record_verdict(attempt_id, "incomplete", "tests failed")
    publisher = FakePublisher()
    events: list[tuple[str, str]] = []
    lifecycle = _lifecycle(
        tmp_path,
        publisher=publisher,
        tracker=FakeTracker(issues={24: _issue(24)}),
        events=events,
    )

    lifecycle.run_once()

    [request] = publisher.requests
    assert request.decision.outcome is AttemptOutcome.INCOMPLETE
    completion = AttemptCompletionStore(tmp_path).read_all()[attempt_id]
    assert completion.outcome is AttemptOutcome.INCOMPLETE
    assert _named(events, "outcome_recorded") == []
    assert AttemptStateStore(tmp_path).read() is None


def test_startup_reconciliation_holds_a_recorded_verdict_it_cannot_reach(
    tmp_path: Path,
) -> None:
    # Interrupted in model_running with no commits: a recorded complete can
    # no longer be republished safely, so the attempt stays retained.
    attempt_id = _checkpoint(tmp_path, AttemptPhase.MODEL_RUNNING)
    AttemptStateStore(tmp_path).record_verdict(attempt_id, "complete", None)
    publisher = FakePublisher()
    events: list[tuple[str, str]] = []
    lifecycle = _lifecycle(
        tmp_path,
        publisher=publisher,
        tracker=FakeTracker(issues={24: _issue(24)}),
        events=events,
    )

    lifecycle.run_once()

    assert publisher.requests == []
    checkpoint = AttemptStateStore(tmp_path).read()
    assert checkpoint is not None
    assert checkpoint.verdict == RecordedVerdict("complete", None)
    assert AttemptCompletionStore(tmp_path).read_all() == {}
    assert _named(events, "recorded_verdict_held") != []
    assert _named(events, "attempt_finalization_held") != []


def test_startup_reconciliation_holds_when_the_publisher_misses_the_verdict(
    tmp_path: Path,
) -> None:
    attempt_id = _checkpoint(tmp_path, AttemptPhase.PUBLISHING)
    AttemptStateStore(tmp_path).record_verdict(attempt_id, "complete", None)
    lifecycle = _lifecycle(
        tmp_path,
        publisher=FakePublisher(outcome=AttemptOutcome.INFRASTRUCTURE_ERROR),
        tracker=FakeTracker(issues={24: _issue(24)}),
        events=[],
    )

    lifecycle.run_once()

    checkpoint = AttemptStateStore(tmp_path).read()
    assert checkpoint is not None
    assert checkpoint.verdict == RecordedVerdict("complete", None)
    assert AttemptCompletionStore(tmp_path).read_all() == {}


# ---------------------------------------------------------------------------
# recovery release
# ---------------------------------------------------------------------------


def test_release_keeps_a_recorded_complete_verdict_after_a_restart(
    tmp_path: Path,
) -> None:
    attempt_id, first_events = _retain_a_complete_attempt(tmp_path)
    order: list[str] = []
    publisher = GithubPublisher(order)
    tracker = OrderedTracker(order, issues={24: _issue(24)})
    events: list[tuple[str, str]] = []
    lifecycle = _lifecycle(tmp_path, publisher=publisher, tracker=tracker, events=events)

    record = lifecycle.submit_recovery_release(
        "req-release", attempt_id, "https://example.test/saved"
    )

    assert record.acknowledgement is CommandAcknowledgement.COMPLETED, record.detail
    assert "outcome complete" in record.detail
    completion = AttemptCompletionStore(tmp_path).read_all()[attempt_id]
    assert completion.outcome is AttemptOutcome.COMPLETE
    # The comment is confirmed before the plain release, which never takes
    # the round-finished path.
    assert order == ["comment", "release"]
    assert tracker.cleanup == [(24, "ready-for-agent", "agent-id")]
    [body] = publisher._github.bodies
    assert body.startswith("## Agent Attempt Result: complete\n")
    assert "https://example.test/saved" in body
    assert "recorded" in body
    assert AttemptStateStore(tmp_path).read() is None
    # outcome_recorded was emitted once for the attempt, at publication.
    assert _named(first_events, "outcome_recorded") == ["outcome=complete"]
    assert _named(events, "outcome_recorded") == []
    assert len(_named(events, "attempt_finished")) == 1


def test_release_without_a_recorded_verdict_keeps_inferring_the_outcome(
    tmp_path: Path,
) -> None:
    # A checkpoint written before the verdict existed (no ``verdict`` key).
    path = tmp_path / "state" / "attempt.json"
    path.parent.mkdir(parents=True)
    attempt_id = "2026-09-20T13:00:00Z"
    path.write_text(
        json.dumps(
            {
                "branch": "agent/issue-24",
                "issue_number": 24,
                "phase": "publishing",
                "started_at": attempt_id,
                "updated_at": attempt_id,
            }
        )
    )
    order: list[str] = []
    publisher = GithubPublisher(order)
    events: list[tuple[str, str]] = []
    lifecycle = _lifecycle(
        tmp_path,
        publisher=publisher,
        tracker=OrderedTracker(order, issues={24: _issue(24)}),
        events=events,
    )

    record = lifecycle.submit_recovery_release("req-release", attempt_id, "/tmp/saved")

    assert record.acknowledgement is CommandAcknowledgement.COMPLETED, record.detail
    completion = AttemptCompletionStore(tmp_path).read_all()[attempt_id]
    # Inference never answers complete for an unpublished branch (#84).
    assert completion.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    [body] = publisher._github.bodies
    assert "not published as a success" in body
    # Without a recorded verdict the outcome is announced once, at finalization.
    assert _named(events, "outcome_recorded") == ["outcome=infrastructure_error"]


# ---------------------------------------------------------------------------
# recovery retry
# ---------------------------------------------------------------------------


def test_retry_finalizes_with_the_recorded_verdict_even_when_inference_differs(
    tmp_path: Path,
) -> None:
    # Publishing without a handoff note: inference alone would republish
    # this attempt as complete, while publication recorded incomplete.
    attempt_id = _checkpoint(tmp_path, AttemptPhase.PUBLISHING)
    AttemptStateStore(tmp_path).record_verdict(attempt_id, "incomplete", "tests failed")
    publisher = FakePublisher()
    events: list[tuple[str, str]] = []
    lifecycle = _lifecycle(
        tmp_path,
        publisher=publisher,
        tracker=FakeTracker(issues={24: _issue(24)}),
        events=events,
    )

    record = lifecycle.submit_recovery_retry("req-retry", attempt_id)

    assert record.acknowledgement is CommandAcknowledgement.COMPLETED, record.detail
    [request] = publisher.requests
    assert request.decision.outcome is AttemptOutcome.INCOMPLETE
    assert request.decision.publication_eligible is False
    assert request.details == "tests failed"
    completion = AttemptCompletionStore(tmp_path).read_all()[attempt_id]
    assert completion.outcome is AttemptOutcome.INCOMPLETE
    assert _named(events, "outcome_recorded") == []
    assert AttemptStateStore(tmp_path).read() is None


def test_retry_never_forces_a_recorded_complete_the_workspace_no_longer_shows(
    tmp_path: Path,
) -> None:
    # Interrupted in model_running with no commits: republishing complete
    # would push the branch, so retry answers not fulfilled instead.
    attempt_id = _checkpoint(tmp_path, AttemptPhase.MODEL_RUNNING)
    AttemptStateStore(tmp_path).record_verdict(attempt_id, "complete", None)
    publisher = FakePublisher()
    lifecycle = _lifecycle(
        tmp_path,
        publisher=publisher,
        tracker=FakeTracker(issues={24: _issue(24)}),
        events=[],
    )

    record = lifecycle.submit_recovery_retry("req-retry", attempt_id)

    assert record.acknowledgement is CommandAcknowledgement.NOT_FULFILLED
    assert "recorded verdict complete" in record.detail
    assert publisher.requests == []
    assert AttemptStateStore(tmp_path).read() is not None


def test_retry_keeps_the_hold_when_republication_misses_the_recorded_verdict(
    tmp_path: Path,
) -> None:
    attempt_id = _checkpoint(tmp_path, AttemptPhase.PUBLISHING)
    AttemptStateStore(tmp_path).record_verdict(attempt_id, "complete", None)
    lifecycle = _lifecycle(
        tmp_path,
        publisher=FakePublisher(outcome=AttemptOutcome.INFRASTRUCTURE_ERROR),
        tracker=FakeTracker(issues={24: _issue(24)}),
        events=[],
    )

    record = lifecycle.submit_recovery_retry("req-retry", attempt_id)

    assert record.acknowledgement is CommandAcknowledgement.NOT_FULFILLED
    assert "recorded verdict complete" in record.detail
    checkpoint = AttemptStateStore(tmp_path).read()
    assert checkpoint is not None
    assert checkpoint.verdict == RecordedVerdict("complete", None)
    assert AttemptCompletionStore(tmp_path).read_all() == {}


def test_retry_keeps_the_hold_for_a_recorded_handoff_it_cannot_republish(
    tmp_path: Path,
) -> None:
    attempt_id = _checkpoint(tmp_path, AttemptPhase.PUBLISHING)
    AttemptStateStore(tmp_path).record_verdict(attempt_id, "handoff", "note committed")
    publisher = FakePublisher()
    lifecycle = _lifecycle(
        tmp_path,
        publisher=publisher,
        tracker=FakeTracker(issues={24: _issue(24)}),
        events=[],
    )

    record = lifecycle.submit_recovery_retry("req-retry", attempt_id)

    assert record.acknowledgement is CommandAcknowledgement.NOT_FULFILLED
    assert "recorded verdict handoff" in record.detail
    assert publisher.requests == []
    assert AttemptStateStore(tmp_path).read() is not None


def test_retry_without_a_recorded_verdict_records_the_republished_one_once(
    tmp_path: Path,
) -> None:
    attempt_id = _checkpoint(tmp_path, AttemptPhase.PUSHING)
    events: list[tuple[str, str]] = []
    lifecycle = _lifecycle(
        tmp_path,
        publisher=FakePublisher(),
        tracker=FakeTracker(issues={24: _issue(24)}),
        events=events,
    )

    record = lifecycle.submit_recovery_retry("req-retry", attempt_id)

    assert record.acknowledgement is CommandAcknowledgement.COMPLETED, record.detail
    assert _named(events, "outcome_recorded") == ["outcome=complete"]
    names = [event for event, _ in events]
    assert names.index("outcome_recorded") < names.index("attempt_finished")


# ---------------------------------------------------------------------------
# release comment
# ---------------------------------------------------------------------------


def test_release_comment_states_a_recorded_verdict() -> None:
    body = release_comment_body(
        issue_number=24,
        attempt_id="2026-09-20T13:00:00Z",
        actual_outcome="complete",
        failure_reason="The result comment could not be confirmed.",
        saved_at="https://example.test/saved",
        branch="agent/issue-24",
        verdict_recorded=True,
    )

    assert body.startswith("## Agent Attempt Result: complete\n")
    assert "recorded" in body
    assert "`https://example.test/saved`" in body
    assert "not published as a success" not in body
    assert "does not claim a successful handoff" in body
    assert "<!-- agent-attempt: 2026-09-20T13:00:00Z -->" in body


def test_release_comment_keeps_the_unpublished_note_for_other_recorded_verdicts() -> None:
    body = release_comment_body(
        issue_number=24,
        attempt_id="2026-09-20T13:00:00Z",
        actual_outcome="infrastructure_error",
        failure_reason="The push failed.",
        saved_at="/tmp/saved",
        branch="agent/issue-24",
        verdict_recorded=True,
    )

    assert "recorded the verdict `infrastructure_error`" in body
    assert "not published as a success" in body
