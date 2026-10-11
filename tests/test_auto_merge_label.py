"""Claim-time opt-in for label-gated auto-merge.

In ``label`` mode the ``agent-auto-merge`` label only counts when the issue
carried it at claim time: a label added while the attempt runs (or after it
finishes) must not enable the merge, and a crashed attempt recovered later
keeps the claim-time decision instead of re-reading the current labels.
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
from simple_coding_agent.publication import AUTO_MERGE_LABEL
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace


def issue(number: int, labels: frozenset[str] = frozenset({"ready-for-agent"})) -> TrackerIssue:
    return TrackerIssue(
        id=f"issue-{number}",
        number=number,
        title="Implement the labelled feature",
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
        details="Implemented the labelled feature.",
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
    return publisher


def test_claim_records_the_opt_in_when_the_label_is_present(tmp_path: Path) -> None:
    tracker = FakeTracker(
        next_claim=Claim(
            issue(24, frozenset({"ready-for-agent", AUTO_MERGE_LABEL})),
            Assignment("issue-24", "agent-id"),
        )
    )

    publisher = run_complete_attempt(tmp_path, tracker)

    assert publisher.requests[0].auto_merge_opt_in is True


def test_claim_records_no_opt_in_when_the_label_is_absent(tmp_path: Path) -> None:
    tracker = FakeTracker(
        next_claim=Claim(issue(24), Assignment("issue-24", "agent-id"))
    )

    publisher = run_complete_attempt(tmp_path, tracker)

    assert publisher.requests[0].auto_merge_opt_in is False


def test_label_added_after_the_claim_does_not_enable_auto_merge(tmp_path: Path) -> None:
    tracker = FakeTracker(
        next_claim=Claim(issue(24), Assignment("issue-24", "agent-id"))
    )

    def runner(claim: Claim, profile: object, prepared: object) -> AttemptEvidence:
        current = tracker.issues.get(claim.issue.number)
        if current is None:
            current = claim.issue
        tracker.issues[claim.issue.number] = replace(
            current, labels=current.labels | frozenset({AUTO_MERGE_LABEL})
        )
        return complete_runner()

    publisher = run_complete_attempt(tmp_path, tracker, runner)

    assert tracker.issues[24].labels >= {AUTO_MERGE_LABEL}
    assert publisher.requests[0].auto_merge_opt_in is False


def start_checkpoint(
    tmp_path: Path, *, opt_in: bool, phase: AttemptPhase = AttemptPhase.PUSHING
):
    state = AttemptStateStore(tmp_path)
    state.start(issue_number=24, branch="agent/issue-24", auto_merge_opt_in=opt_in)
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


def test_recovery_retry_keeps_an_opt_in_the_issue_no_longer_carries(
    tmp_path: Path,
) -> None:
    checkpoint = start_checkpoint(tmp_path, opt_in=True)
    assert checkpoint is not None
    lifecycle, publisher = recovery_lifecycle(tmp_path, self_assigned_tracker())

    record = lifecycle.submit_recovery_retry("req-keep-opt-in", checkpoint.started_at)

    assert record.acknowledgement is CommandAcknowledgement.COMPLETED
    assert publisher.requests[0].auto_merge_opt_in is True


def test_recovery_retry_keeps_no_opt_in_when_the_label_arrived_late(
    tmp_path: Path,
) -> None:
    checkpoint = start_checkpoint(tmp_path, opt_in=False)
    assert checkpoint is not None
    tracker = self_assigned_tracker()
    tracker.issues[24] = replace(
        tracker.issues[24],
        labels=tracker.issues[24].labels | frozenset({AUTO_MERGE_LABEL}),
    )
    lifecycle, publisher = recovery_lifecycle(tmp_path, tracker)

    record = lifecycle.submit_recovery_retry("req-keep-skipped", checkpoint.started_at)

    assert record.acknowledgement is CommandAcknowledgement.COMPLETED
    assert publisher.requests[0].auto_merge_opt_in is False


def test_startup_recovery_keeps_the_claim_time_decision(tmp_path: Path) -> None:
    checkpoint = start_checkpoint(tmp_path, opt_in=True)
    assert checkpoint is not None
    workspace = InMemoryWorkspace(tmp_path / "workspace", issue_number=24)
    publisher = FakePublisher()
    lifecycle = AgentLifecycle(
        tracker=self_assigned_tracker(),
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=publisher,
        sleeper=lambda seconds: None,
    )

    lifecycle.run_once()

    assert publisher.requests != []
    assert publisher.requests[0].auto_merge_opt_in is True
