"""Restart-safe finalization: stable identity, durable boundary, preserved work."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.completion import AttemptOutcome, CompletionDecision, PublicationPath
from simple_coding_agent.control import (
    CommandAcknowledgement,
    ControlStore,
    ResumeBlockedError,
)
from simple_coding_agent.finalization import AttemptCompletionStore
from simple_coding_agent.git_workspace import DirtyWorkspaceError, PreparedAttempt
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.lifecycle import AgentLifecycle, LifecycleStatus
from simple_coding_agent.operating import ConsecutiveErrorStore
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace


def test_completion_store_records_each_attempt_once(tmp_path: Path) -> None:
    store = AttemptCompletionStore(tmp_path)

    assert store.record(attempt_id="2026-09-20T13:00:00Z", issue_number=24, branch="agent/issue-24", outcome=AttemptOutcome.COMPLETE) is True
    assert store.record(attempt_id="2026-09-20T13:00:00Z", issue_number=24, branch="agent/issue-24", outcome=AttemptOutcome.COMPLETE) is False

    assert store.is_finalized("2026-09-20T13:00:00Z") is True
    assert store.is_finalized("2026-09-20T14:00:00Z") is False
    recorded = store.read_all()["2026-09-20T13:00:00Z"]
    assert recorded.issue_number == 24
    assert recorded.outcome is AttemptOutcome.COMPLETE


def test_completion_store_survives_restart(tmp_path: Path) -> None:
    AttemptCompletionStore(tmp_path).record(
        attempt_id="attempt-a", issue_number=24, branch="agent/issue-24", outcome=AttemptOutcome.INCOMPLETE
    )

    restarted = AttemptCompletionStore(tmp_path)

    assert restarted.is_finalized("attempt-a") is True
    assert restarted.record(attempt_id="attempt-a", issue_number=24, branch="agent/issue-24", outcome=AttemptOutcome.INCOMPLETE) is False


def test_completion_store_rejects_malformed_record(tmp_path: Path) -> None:
    path = tmp_path / "state" / "completed_attempts.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not JSON")

    from simple_coding_agent.finalization import CompletionStoreError

    with pytest.raises(CompletionStoreError, match="malformed"):
        AttemptCompletionStore(tmp_path).is_finalized("attempt-a")


def test_finalized_attempt_is_recorded_once_before_a_new_claim(
    tmp_path: Path, workspace: InMemoryWorkspace, publisher: FakePublisher
) -> None:
    """A fully finalized attempt lands in the ledger exactly once with its accounting."""
    claim = Claim(_issue(24), Assignment("issue-24", "agent-id"))
    tracker = FakeTracker(next_claim=claim)
    completions = AttemptCompletionStore(tmp_path)
    errors = ConsecutiveErrorStore(tmp_path)
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: _profile(),
        publisher=publisher,
        attempt_runner=_evidence_runner(AttemptOutcome.INCOMPLETE),
        error_store=errors,
        completion_store=completions,
    )

    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.ATTEMPTED
    # The fresh claim created a checkpoint that finalization must remove
    # after recording the attempt exactly once.
    assert completions.read_all()
    attempt_id = next(iter(completions.read_all()))
    assert completions.is_finalized(attempt_id) is True
    assert AttemptStateStore(tmp_path).read() is None
    assert errors.read().last_attempt_id == attempt_id


def test_crash_between_finalization_and_checkpoint_removal_does_not_republish(
    tmp_path: Path, workspace: InMemoryWorkspace, publisher: FakePublisher
) -> None:
    """Ledger present + checkpoint present means: only remove the checkpoint."""
    state = _state_at(tmp_path, AttemptPhase.PUBLISHING)
    checkpoint = state.read()
    assert checkpoint is not None
    completions = AttemptCompletionStore(tmp_path)
    completions.record(
        attempt_id=checkpoint.started_at, issue_number=24, branch="agent/issue-24", outcome=AttemptOutcome.COMPLETE
    )
    workspace.model_commit("completed", files=("completed",))
    tracker = FakeTracker({24: _issue(24)})
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: _profile(),
        publisher=publisher,
        completion_store=completions,
        sleeper=lambda seconds: None,
    )

    lifecycle.run_once()

    assert publisher.requests == []
    assert tracker.cleanup == []
    assert state.read() is None
    # Still exactly one ledger entry: no duplicate accounting.
    assert len(completions.read_all()) == 1


def test_release_failure_holds_the_attempt_without_accounting(
    tmp_path: Path, workspace: InMemoryWorkspace, publisher: FakePublisher
) -> None:
    state = _state_at(tmp_path, AttemptPhase.PUSHING)
    checkpoint = state.read()
    assert checkpoint is not None
    completions = AttemptCompletionStore(tmp_path)
    errors = ConsecutiveErrorStore(tmp_path)
    workspace.model_commit("completed", files=("completed",))
    lifecycle = AgentLifecycle(
        tracker=FailingReleaseTracker({24: _issue(24)}),
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: _profile(),
        publisher=publisher,
        error_store=errors,
        completion_store=completions,
    )

    with pytest.raises(OSError, match="GitHub unavailable"):
        lifecycle.run_once()

    assert state.read() is not None
    assert completions.read_all() == {}
    assert errors.read().last_attempt_id is None


def test_comment_failure_holds_without_ledger_then_recovers_once(
    tmp_path: Path, workspace: InMemoryWorkspace, publisher: FakePublisher
) -> None:
    state = _state_at(tmp_path, AttemptPhase.PUSHING)
    completions = AttemptCompletionStore(tmp_path)
    errors = ConsecutiveErrorStore(tmp_path)
    workspace.model_commit("completed", files=("completed",))
    first = AgentLifecycle(
        tracker=FakeTracker({24: _issue(24)}),
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: _profile(),
        publisher=FakePublisher(comment_posted=False),
        error_store=errors,
        completion_store=completions,
        sleeper=lambda seconds: None,
    )

    first.run_once()

    assert state.read() is not None
    assert completions.read_all() == {}
    assert errors.read().last_attempt_id is None

    tracker = FakeTracker({24: _issue(24)})
    second = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: _profile(),
        publisher=publisher,
        error_store=errors,
        completion_store=completions,
        sleeper=lambda seconds: None,
    )

    second.run_once()

    assert tracker.cleanup == [(24, "ready-for-agent", "agent-id")]
    assert state.read() is None
    assert len(completions.read_all()) == 1
    assert errors.read().last_attempt_id is not None


def test_dirty_worktree_during_startup_holds_without_publishing(
    tmp_path: Path, publisher: FakePublisher
) -> None:
    """Unexplained dirty work keeps the checkpoint, branch, and tree for inspection.

    The process stays up (no ``SystemExit``): the attempt is held with
    everything preserved, and the next poll holds on the retained attempt
    instead of claiming new work.
    """
    state = _state_at(tmp_path, AttemptPhase.MODEL_RUNNING)
    completions = AttemptCompletionStore(tmp_path)
    tracker = FakeTracker({24: _issue(24)})
    workspace = DirtyStartupWorkspace(tmp_path / "repo")
    workspace.make_dirty("work.txt", "uncommitted work")
    events: list[str] = []
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: _profile(),
        publisher=publisher,
        completion_store=completions,
        control_store=ControlStore(tmp_path),
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(event),
        sleeper=lambda seconds: None,
    )

    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.ATTEMPTED
    assert state.read() is not None
    assert publisher.requests == []
    assert tracker.cleanup == []
    assert completions.read_all() == {}
    assert "workspace_hold" in events
    assert "attempt_finalization_held" in events

    # The next poll holds on the retained attempt instead of claiming new work.
    tracker.next_claim = Claim(_issue(25), Assignment("issue-25", "agent-id"))
    followup = lifecycle.run_once()

    assert followup.status is LifecycleStatus.IDLE
    assert tracker.claimed == []
    assert "intake_held_for_recovery" in events
    assert state.read() is not None


def test_malformed_checkpoint_holds_intake_without_claiming(
    tmp_path: Path, workspace: InMemoryWorkspace, publisher: FakePublisher
) -> None:
    path = tmp_path / "state" / "attempt.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not JSON")
    tracker = FakeTracker(next_claim=Claim(_issue(25), Assignment("issue-25", "agent-id")))
    events: list[str] = []
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: _profile(),
        publisher=publisher,
        completion_store=AttemptCompletionStore(tmp_path),
        control_store=ControlStore(tmp_path),
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(event),
        sleeper=lambda seconds: None,
    )

    result = lifecycle.run_once()

    # The process stays up (no ``SystemExit``) and claims nothing: startup
    # reconciliation holds on the unreadable checkpoint.
    assert result.status is LifecycleStatus.IDLE
    assert tracker.claimed == []
    assert "attempt_checkpoint_unusable" in events

    # A repeated poll stays held without spamming the hold event again.
    lifecycle.run_once()

    assert tracker.claimed == []
    assert events.count("attempt_checkpoint_unusable") == 1

    # The operator-visible state names the hold for repair.
    snapshot = lifecycle.control_status("owner/repo")
    assert snapshot["recovery"] is not None
    assert "checkpoint" in snapshot["recovery"]["hold_reason"]
    assert snapshot["recovery"]["next_action"]

    # Removing the unreadable checkpoint clears the hold and intake resumes.
    path.unlink()
    resumed = lifecycle.run_once()

    assert resumed.status is LifecycleStatus.ATTEMPTED
    assert tracker.claimed == [25]
    assert "startup_hold_cleared" in events


def test_dirty_tree_without_checkpoint_holds_intake(
    tmp_path: Path, workspace: InMemoryWorkspace, publisher: FakePublisher
) -> None:
    """A dirty tree with no checkpoint holds the claim without exiting.

    The process stays up (no ``SystemExit``), sleeps the poll interval,
    and keeps serving the control socket, so the operator retains a live
    view while the tree awaits manual repair.
    """
    workspace.make_dirty("work.txt", "uncommitted work")
    tracker = FakeTracker(next_claim=Claim(_issue(25), Assignment("issue-25", "agent-id")))
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: _profile(),
        publisher=publisher,
        completion_store=AttemptCompletionStore(tmp_path),
        sleeper=lambda seconds: None,
    )

    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.IDLE
    assert tracker.claimed == []
    assert AttemptStateStore(tmp_path).read() is None


def test_error_limit_trips_before_the_completion_record(
    tmp_path: Path, workspace: InMemoryWorkspace, publisher: FakePublisher
) -> None:
    """The circuit breaker must fire without recording a completion it never reached.

    Accounting precedes the ledger entry so a limit trip leaves the checkpoint
    present and the attempt unrecorded; otherwise a restart would see a
    finalized ledger entry, delete the checkpoint, and claim a new issue
    despite the tripped limit.
    """
    claim = Claim(_issue(24), Assignment("issue-24", "agent-id"))
    completions = AttemptCompletionStore(tmp_path)
    errors = ConsecutiveErrorStore(tmp_path)
    lifecycle = AgentLifecycle(
        tracker=FakeTracker(next_claim=claim),
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: _profile(),
        publisher=publisher,
        attempt_runner=_evidence_runner(AttemptOutcome.INFRASTRUCTURE_ERROR),
        error_store=errors,
        max_consecutive_errors=1,
        completion_store=completions,
    )

    with pytest.raises(SystemExit):
        lifecycle.run_once()

    assert completions.read_all() == {}
    assert AttemptStateStore(tmp_path).read() is not None
    assert errors.read().count == 1


def test_error_limit_survives_restart_replay_without_a_new_claim(
    tmp_path: Path, workspace: InMemoryWorkspace, publisher: FakePublisher
) -> None:
    """Replay after a limit trip re-trips the limit instead of claiming anew."""
    claim = Claim(_issue(24), Assignment("issue-24", "agent-id"))
    completions = AttemptCompletionStore(tmp_path)
    errors = ConsecutiveErrorStore(tmp_path)
    first = AgentLifecycle(
        tracker=FakeTracker(next_claim=claim),
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: _profile(),
        publisher=publisher,
        attempt_runner=_evidence_runner(AttemptOutcome.INFRASTRUCTURE_ERROR),
        error_store=errors,
        max_consecutive_errors=1,
        completion_store=completions,
    )

    with pytest.raises(SystemExit):
        first.run_once()

    tracker = FakeTracker({24: _issue(24)})
    second = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: _profile(),
        publisher=publisher,
        error_store=errors,
        max_consecutive_errors=1,
        completion_store=completions,
    )

    with pytest.raises(SystemExit):
        second.run_once()

    assert tracker.claimed == []
    assert completions.read_all() == {}
    assert AttemptStateStore(tmp_path).read() is not None


def test_live_dirty_worktree_holds_without_publishing(
    tmp_path: Path, publisher: FakePublisher
) -> None:
    """Dirty work found while preparing a live attempt holds like recovery does.

    No result comment, no issue release, no ledger entry: the branch,
    checkpoint, and working tree stay available for inspection, the process
    stays up (no ``SystemExit``), and the next poll holds on the retained
    attempt instead of claiming new work.
    """
    claim = Claim(_issue(24), Assignment("issue-24", "agent-id"))
    tracker = FakeTracker(next_claim=claim)
    completions = AttemptCompletionStore(tmp_path)
    events: list[str] = []
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=PrepareDirtyWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: _profile(),
        publisher=publisher,
        attempt_runner=_evidence_runner(AttemptOutcome.INCOMPLETE),
        completion_store=completions,
        control_store=ControlStore(tmp_path),
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(event),
        sleeper=lambda seconds: None,
    )

    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.ATTEMPTED
    assert publisher.requests == []
    assert tracker.cleanup == []
    assert completions.read_all() == {}
    assert AttemptStateStore(tmp_path).read() is not None
    assert "workspace_hold" in events
    assert "attempt_finalization_held" in events

    tracker.next_claim = Claim(_issue(25), Assignment("issue-25", "agent-id"))
    followup = lifecycle.run_once()

    assert followup.status is LifecycleStatus.IDLE
    assert tracker.claimed == [24]
    assert "intake_held_for_recovery" in events
    assert AttemptStateStore(tmp_path).read() is not None


def test_cleanup_failure_holds_with_a_visible_event(
    tmp_path: Path, publisher: FakePublisher
) -> None:
    """A confirmed comment and release with failed cleanup keeps the checkpoint visibly."""
    claim = Claim(_issue(24), Assignment("issue-24", "agent-id"))
    events: list[str] = []
    lifecycle = AgentLifecycle(
        tracker=FakeTracker(next_claim=claim),
        attempt_state=AttemptStateStore(tmp_path),
        workspace=CleanupFailingWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: _profile(),
        publisher=publisher,
        attempt_runner=_evidence_runner(AttemptOutcome.INCOMPLETE),
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(event),
        completion_store=AttemptCompletionStore(tmp_path),
    )

    with pytest.raises(OSError, match="cleanup unavailable"):
        lifecycle.run_once()

    assert "attempt_finalization_held" in events
    assert AttemptStateStore(tmp_path).read() is not None
    assert AttemptCompletionStore(tmp_path).read_all() == {}


def test_malformed_completion_ledger_holds_intake_without_claiming(
    tmp_path: Path, workspace: InMemoryWorkspace, publisher: FakePublisher
) -> None:
    path = tmp_path / "state" / "completed_attempts.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not JSON")
    _state_at(tmp_path, AttemptPhase.PUSHING)
    events: list[str] = []
    tracker = FakeTracker({24: _issue(24)})
    workspace.model_commit("completed", files=("completed",))
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: _profile(),
        publisher=publisher,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(event),
        completion_store=AttemptCompletionStore(tmp_path),
        control_store=ControlStore(tmp_path),
        sleeper=lambda seconds: None,
    )

    result = lifecycle.run_once()

    # The process stays up (no ``SystemExit``) and claims nothing: startup
    # reconciliation holds on the unreadable ledger with the checkpoint kept.
    assert result.status is LifecycleStatus.IDLE
    assert "attempt_completion_unusable" in events
    assert tracker.claimed == []
    assert AttemptStateStore(tmp_path).read() is not None

    snapshot = lifecycle.control_status("owner/repo")
    assert snapshot["recovery"] is not None
    assert "ledger" in snapshot["recovery"]["hold_reason"]

    # Repairing only the checkpoint file never drops the ledger hold, and
    # repairing the ledger lets the retained attempt surface for recovery.
    path.unlink()
    followup = lifecycle.run_once()

    assert followup.status is LifecycleStatus.IDLE
    assert "startup_hold_cleared" in events
    assert "intake_held_for_recovery" in events
    assert tracker.claimed == []
    assert AttemptStateStore(tmp_path).read() is not None
    recovered = lifecycle.control_status("owner/repo")
    assert recovered["recovery"] is not None
    assert recovered["recovery"]["issue_number"] == 24


# --- Failure-injection doubles (single-method subclasses of shared doubles) ---


def _evidence_runner(outcome: AttemptOutcome):
    from simple_coding_agent.lifecycle import AttemptEvidence

    def run(claim: Claim, profile: object, prepared: object) -> AttemptEvidence:
        if outcome is AttemptOutcome.INCOMPLETE:
            decision = CompletionDecision(outcome, False, PublicationPath.PARTIAL, ("check failed",))
        else:
            decision = CompletionDecision(outcome, False, PublicationPath.NONE, ("done",))
        return AttemptEvidence(decision, "pytest", 1, 0, "all clear", "details")

    return run


class FailingReleaseTracker(FakeTracker):
    def release_attempt(self, issue_number: int, label: str, assignee_id: str) -> None:
        raise OSError("GitHub unavailable")

    def release_handoff(self, issue_number: int, assignee_id: str) -> None:
        raise OSError("GitHub unavailable")


class DirtyStartupWorkspace(InMemoryWorkspace):
    def prepare_for_profile_read(self, *, base_branch: str = "main") -> str:
        raise DirtyWorkspaceError("Repository working tree contains uncommitted changes")


class PrepareDirtyWorkspace(InMemoryWorkspace):
    """Clean at the pre-claim check, dirty once preparation touches the tree."""

    def prepare_attempt(self, *, base_branch: str, issue_number: int) -> PreparedAttempt:
        raise DirtyWorkspaceError("Repository working tree contains uncommitted changes")


class CleanupFailingWorkspace(InMemoryWorkspace):
    def cleanup(self, *, base_branch: str, prepared: PreparedAttempt, retain_branch: bool) -> None:
        raise OSError("cleanup unavailable")


def _profile(base_branch: str = "main"):
    return type("Profile", (), {"base_branch": base_branch})()


def _state_at(tmp_path: Path, phase: AttemptPhase) -> AttemptStateStore:
    state = AttemptStateStore(tmp_path)
    state.start(issue_number=24, branch="agent/issue-24")
    for next_phase in (AttemptPhase.SETUP, AttemptPhase.MODEL_RUNNING, AttemptPhase.PUSHING, AttemptPhase.PUBLISHING):
        if state.read() and state.read().phase is phase:  # type: ignore[union-attr]
            break
        state.transition(next_phase)
    return state


def _issue(number: int, assignees: tuple[str, ...] = ()) -> TrackerIssue:
    return TrackerIssue(
        id=f"issue-{number}",
        number=number,
        title="Implement lifecycle",
        body="Connect boundaries.",
        created_at=datetime(2026, 9, 20, tzinfo=UTC),
        state="OPEN",
        labels=frozenset({"ready-for-agent"}),
        assignee_logins=assignees,
        blocked_by=0,
        author_login="reporter",
    )


def test_restart_with_held_workspace_stays_held_without_exiting(
    tmp_path: Path, publisher: FakePublisher
) -> None:
    """A restart over a held workspace re-enters the hold instead of crash-looping.

    The issue scenario: cleanup held on a dirty tree, the process restarts,
    and startup reconciliation re-detects the same hold. Every restart must
    stay up (no ``SystemExit``) with the checkpoint, branch, and tree
    preserved.
    """
    state = _state_at(tmp_path, AttemptPhase.PUBLISHING)
    completions = AttemptCompletionStore(tmp_path)

    def build(events: list[str], tracker: FakeTracker) -> AgentLifecycle:
        return AgentLifecycle(
            tracker=tracker,
            attempt_state=state,
            workspace=DirtyStartupWorkspace(tmp_path / "repo"),
            profile_loader=lambda _: _profile(),
            publisher=publisher,
            completion_store=completions,
            control_store=ControlStore(tmp_path),
            event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(event),
            sleeper=lambda seconds: None,
        )

    first_events: list[str] = []
    first = build(first_events, FakeTracker({24: _issue(24)}))
    first_result = first.run_once()

    assert first_result.status is LifecycleStatus.ATTEMPTED
    assert state.read() is not None
    assert "workspace_hold" in first_events

    # Simulated restart: a brand-new lifecycle over the same state.
    second_events: list[str] = []
    second = build(second_events, FakeTracker({24: _issue(24)}))
    second_result = second.run_once()

    assert second_result.status is LifecycleStatus.ATTEMPTED
    assert publisher.requests == []
    assert state.read() is not None
    assert completions.read_all() == {}
    assert "workspace_hold" in second_events
    assert "attempt_finalization_held" in second_events

    # A further poll on the restarted process holds quietly on the retained
    # attempt instead of claiming new work.
    third = second.run_once()

    assert third.status is LifecycleStatus.IDLE
    assert second_events.count("workspace_hold") == 1


def test_cleanup_hold_recovers_through_recovery_retry_after_repair(
    tmp_path: Path,
) -> None:
    """Repairing a cleanup hold lets ``recovery retry`` clear it and intake resume."""
    issue_24 = _issue(24, assignees=("agent",))
    tracker = FakeTracker(
        issues={24: issue_24, 25: _issue(25, assignees=("agent",))},
        next_claim=Claim(issue_24, Assignment("issue-24", "agent-id")),
    )
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
    state = AttemptStateStore(tmp_path)
    completions = AttemptCompletionStore(tmp_path)
    publisher = FakePublisher()
    events: list[str] = []

    def run_workflow(claim: Claim, profile: object, prepared: object):
        if claim.issue.number == 24:
            workspace.make_dirty("uncommitted.py", "leftover work")
        return _evidence_runner(AttemptOutcome.COMPLETE)(claim, profile, prepared)

    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: _profile(),
        publisher=publisher,
        attempt_runner=run_workflow,
        completion_store=completions,
        control_store=ControlStore(tmp_path),
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(event),
        sleeper=lambda seconds: None,
    )

    held = lifecycle.run_once()

    assert held.status is LifecycleStatus.ATTEMPTED
    checkpoint = state.read()
    assert checkpoint is not None
    assert "workspace_hold" in events

    # The operator repairs the tree; the hold still needs its recovery step.
    workspace.dirty.clear()
    assert lifecycle.run_once().status is LifecycleStatus.IDLE
    assert state.read() is not None

    record = lifecycle.submit_recovery_retry("req-retry", checkpoint.started_at)

    assert record.acknowledgement is CommandAcknowledgement.COMPLETED
    assert state.read() is None
    assert completions.is_finalized(checkpoint.started_at)

    # Intake resumes with the next issue.
    tracker.next_claim = Claim(_issue(25), Assignment("issue-25", "agent-id"))
    resumed = lifecycle.run_once()

    assert resumed.status is LifecycleStatus.ATTEMPTED
    assert tracker.claimed == [24, 25]


def test_resume_is_rejected_while_the_startup_hold_is_active(
    tmp_path: Path, workspace: InMemoryWorkspace, publisher: FakePublisher
) -> None:
    """``resume`` names the startup hold instead of raising while held."""
    path = tmp_path / "state" / "attempt.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not JSON")
    lifecycle = AgentLifecycle(
        tracker=FakeTracker(next_claim=Claim(_issue(25), Assignment("issue-25", "agent-id"))),
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: _profile(),
        publisher=publisher,
        completion_store=AttemptCompletionStore(tmp_path),
        control_store=ControlStore(tmp_path),
        sleeper=lambda seconds: None,
    )

    assert lifecycle.run_once().status is LifecycleStatus.IDLE

    with pytest.raises(ResumeBlockedError, match="checkpoint"):
        lifecycle.submit_resume("req-resume")

    # Repairing the checkpoint lets resume succeed and intake proceed.
    path.unlink()
    resumed = lifecycle.submit_resume("req-resume-2")

    assert resumed.acknowledgement is CommandAcknowledgement.COMPLETED
