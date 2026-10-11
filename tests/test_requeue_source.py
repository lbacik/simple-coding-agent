"""Claim-time requeue source for attempt-result comments.

Every attempt-result comment records how its attempt was requeued: ``human``
(the issue carried ``ready-for-agent`` at claim time) or ``automatic`` (it
carried only ``round-finished`` at claim time). The source is decided once
from the claim snapshot, so a label change while the attempt runs (or after
it finishes) never rewrites that attempt's comment — including when a
crashed attempt is recovered later.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.completion import AttemptOutcome, CompletionDecision, PublicationPath
from simple_coding_agent.control import CommandAcknowledgement, ControlStore
from simple_coding_agent.finalization import AttemptCompletionStore
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.lifecycle import AgentLifecycle, AttemptEvidence, LifecycleStatus
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace


def issue(number: int, labels: frozenset[str] = frozenset({"ready-for-agent"})) -> TrackerIssue:
    return TrackerIssue(
        id=f"issue-{number}",
        number=number,
        title="Implement the requeued feature",
        body="Connect boundaries.",
        created_at=datetime(2026, 9, 20, tzinfo=UTC),
        state="OPEN",
        labels=labels,
        assignee_logins=(),
        blocked_by=0,
        author_login="reporter",
    )


def complete_runner() -> AttemptEvidence:
    return AttemptEvidence(
        decision=CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)),
        check_command="pytest",
        check_exit_code=0,
        review_cycles=1,
        review_findings="all clear",
        details="Implemented the requeued feature.",
    )


def run_complete_attempt(tmp_path: Path, tracker: FakeTracker, runner=None):
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
    publisher = FakePublisher()
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=publisher,
        attempt_runner=runner or (lambda claim, profile, prepared: complete_runner()),
    )

    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.ATTEMPTED
    assert result.outcome is AttemptOutcome.COMPLETE
    return lifecycle, publisher


def test_claim_with_ready_for_agent_records_a_human_requeue_source(tmp_path: Path) -> None:
    tracker = FakeTracker(
        next_claim=Claim(
            issue(24, frozenset({"ready-for-agent", "round-finished"})),
            Assignment("issue-24", "agent-id"),
        )
    )

    _, publisher = run_complete_attempt(tmp_path, tracker)

    assert publisher.requests[0].requeue_source == "human"


def test_claim_without_ready_for_agent_records_an_automatic_requeue_source(
    tmp_path: Path,
) -> None:
    tracker = FakeTracker(
        issues={24: issue(24, frozenset({"round-finished"}))},
        fifo=[24],
    )

    _, publisher = run_complete_attempt(tmp_path, tracker)

    assert publisher.requests[0].requeue_source == "automatic"


def test_label_removed_after_the_claim_keeps_the_human_source(tmp_path: Path) -> None:
    tracker = FakeTracker(
        next_claim=Claim(issue(24), Assignment("issue-24", "agent-id"))
    )

    def runner(claim: Claim, profile: object, prepared: object) -> AttemptEvidence:
        current = tracker.issues.get(claim.issue.number)
        if current is None:
            current = claim.issue
        tracker.issues[claim.issue.number] = replace(current, labels=frozenset())
        return complete_runner()

    _, publisher = run_complete_attempt(tmp_path, tracker, runner)

    assert publisher.requests[0].requeue_source == "human"


def start_checkpoint(tmp_path: Path, *, requeue_source: str, phase: AttemptPhase = AttemptPhase.PUSHING):
    state = AttemptStateStore(tmp_path)
    state.start(
        issue_number=24,
        branch="agent/issue-24",
        auto_merge_opt_in=False,
        requeue_source=requeue_source,
    )
    order = (
        AttemptPhase.SETUP,
        AttemptPhase.MODEL_RUNNING,
        AttemptPhase.PUSHING,
        AttemptPhase.PUBLISHING,
    )
    for next_phase in order:
        current = state.read()
        assert current is not None
        if current.phase is phase:
            break
        state.transition(next_phase)
    return state.read()


def recovery_lifecycle(tmp_path: Path, tracker: FakeTracker):
    publisher = FakePublisher()
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "workspace", issue_number=24),
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=publisher,
        attempt_runner=lambda c, p, pr: None,
        control_store=ControlStore(tmp_path),
        completion_store=AttemptCompletionStore(tmp_path),
        sleeper=lambda seconds: None,
        repository="owner/repo",
    )
    return lifecycle, publisher


def self_assigned_tracker() -> FakeTracker:
    return FakeTracker(issues={24: replace(issue(24), assignee_logins=("agent",))})


def test_recovery_retry_keeps_the_claim_time_requeue_source(tmp_path: Path) -> None:
    checkpoint = start_checkpoint(tmp_path, requeue_source="automatic")
    assert checkpoint is not None
    lifecycle, publisher = recovery_lifecycle(tmp_path, self_assigned_tracker())

    record = lifecycle.submit_recovery_retry("req-keep-source", checkpoint.started_at)

    assert record.acknowledgement is CommandAcknowledgement.COMPLETED
    assert publisher.requests[0].requeue_source == "automatic"
