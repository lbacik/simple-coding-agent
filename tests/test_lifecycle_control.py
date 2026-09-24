"""Lifecycle integration for durable stop and live status (issue #80)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.completion import AttemptOutcome, CompletionDecision, PublicationPath
from simple_coding_agent.control import (
    CommandAcknowledgement,
    ControlStore,
    IntakeState,
    ResumeBlockedError,
)
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.lifecycle import AgentLifecycle, LifecycleStatus
from datetime import UTC, datetime


def test_idle_stop_blocks_the_next_claim(tmp_path: Path) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, next_claim=claim(24))

    record = lifecycle.submit_stop("req-1")

    assert record.acknowledgement is CommandAcknowledgement.COMPLETED
    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.IDLE
    assert tracker.claimed == []
    assert store.intake_state() is IntakeState.STOPPED


def test_active_stop_lets_the_attempt_finish_before_completing(tmp_path: Path) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, next_claim=claim(24))
    started = threading.Event()
    release = threading.Event()

    def blocking_workflow(received_claim: Claim, profile: object, prepared: object):
        started.set()
        assert release.wait(timeout=30)
        return evidence()

    lifecycle.set_attempt_runner(blocking_workflow)
    worker = threading.Thread(target=lifecycle.run_once)
    worker.start()
    assert started.wait(timeout=30)

    record = lifecycle.submit_stop("req-stop")

    assert record.acknowledgement is CommandAcknowledgement.ACCEPTED
    assert store.intake_state() is IntakeState.STOPPING
    # The attempt is still running: nothing published, released, or cleaned up.
    assert tracker.cleanup == []
    release.set()
    worker.join(timeout=30)
    assert not worker.is_alive()

    assert tracker.cleanup == [(24, "ready-for-agent", "agent-id")]
    assert workspace_of(lifecycle).cleanup_calls != []
    stored = store.get_command("req-stop")
    assert stored is not None
    assert stored.acknowledgement is CommandAcknowledgement.COMPLETED
    assert store.intake_state() is IntakeState.STOPPED


def test_active_attempt_keeps_its_ordinary_outcome_under_stop(tmp_path: Path) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, next_claim=claim(24))
    started = threading.Event()
    release = threading.Event()
    lifecycle.set_attempt_runner(blocking_evidence(started, release, AttemptOutcome.COMPLETE))
    worker = threading.Thread(target=lifecycle.run_once)
    worker.start()
    assert started.wait(timeout=30)

    lifecycle.submit_stop("req-stop")
    release.set()
    worker.join(timeout=30)

    assert tracker.cleanup == [(24, "ready-for-agent", "agent-id")]
    assert store.get_command("req-stop").acknowledgement is CommandAcknowledgement.COMPLETED


def test_claim_cannot_begin_after_an_accepted_stop_boundary(tmp_path: Path) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, next_claim=claim(24))
    store.submit_stop("req-1", has_active_attempt=False)

    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.IDLE
    assert tracker.claimed == []


def test_stop_arriving_during_a_claim_applies_to_the_claimed_attempt(
    tmp_path: Path,
) -> None:
    """A claim that began first becomes the active attempt for the command."""

    lifecycle, tracker, store = make_lifecycle(tmp_path, next_claim=claim(24))
    entered = threading.Event()
    release = threading.Event()
    original_claim = tracker.claim_next

    def blocking_claim():
        entered.set()
        assert release.wait(timeout=30)
        return original_claim()

    tracker.claim_next = blocking_claim  # type: ignore[method-assign]
    worker = threading.Thread(target=lifecycle.run_once)
    worker.start()
    assert entered.wait(timeout=30)

    submitted: dict[str, object] = {}

    def do_submit() -> None:
        submitted["record"] = lifecycle.submit_stop("req-stop")

    submitter = threading.Thread(target=do_submit)
    submitter.start()
    release.set()
    worker.join(timeout=30)
    submitter.join(timeout=30)

    record = submitted["record"]
    assert record.acknowledgement is CommandAcknowledgement.ACCEPTED
    assert tracker.claimed == [24]
    assert tracker.cleanup == [(24, "ready-for-agent", "agent-id")]
    assert store.get_command("req-stop").acknowledgement is CommandAcknowledgement.COMPLETED
    assert store.intake_state() is IntakeState.STOPPED


def test_finalized_recovery_completes_the_waiting_stop(tmp_path: Path) -> None:
    """A crash between finalization and checkpoint removal still completes stop."""

    from simple_coding_agent.finalization import AttemptCompletionStore

    state = AttemptStateStore(tmp_path)
    checkpoint = state.start(issue_number=24, branch="agent/issue-24")
    completion_store = AttemptCompletionStore(tmp_path)
    completion_store.record(
        attempt_id=checkpoint.started_at,
        issue_number=24,
        branch="agent/issue-24",
        outcome=AttemptOutcome.INCOMPLETE,
    )
    lifecycle, tracker, store = make_lifecycle(tmp_path, next_claim=claim(99))
    record = lifecycle.submit_stop("req-stop")
    assert record.acknowledgement is CommandAcknowledgement.ACCEPTED
    assert store.intake_state() is IntakeState.STOPPING

    # Simulate the restart: fresh lifecycle, same data dir. The checkpoint
    # is already finalized, so recovery only removes it — and must complete
    # the stop whose whole boundary is durable.
    from simple_coding_agent.lifecycle import AgentLifecycle

    restarted = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=workspace_of(lifecycle),
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=lambda received_claim, profile, prepared: _evidence(None),
        control_store=store,
        completion_store=completion_store,
        sleeper=lambda seconds: None,
    )
    result = restarted.run_once()

    assert result.status is LifecycleStatus.IDLE
    assert tracker.claimed == []
    assert store.get_command("req-stop").acknowledgement is CommandAcknowledgement.COMPLETED
    assert store.intake_state() is IntakeState.STOPPED


def test_status_snapshot_reports_repository_intake_attempt_and_ack(
    tmp_path: Path,
) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, next_claim=None)
    state = AttemptStateStore(tmp_path)
    state.start(issue_number=24, branch="agent/issue-24")
    state.transition(AttemptPhase.SETUP)
    lifecycle.submit_stop("req-1")

    snapshot = lifecycle.control_status("owner/repo")

    assert snapshot["repository"] == "owner/repo"
    assert snapshot["intake"] == IntakeState.STOPPING.value
    assert snapshot["recovering"] is False
    assert snapshot["active_attempt"]["issue_number"] == 24
    assert snapshot["active_attempt"]["phase"] == AttemptPhase.SETUP.value
    assert snapshot["pending_command"]["request_id"] == "req-1"
    assert snapshot["pending_command"]["acknowledgement"] == "accepted"
    assert snapshot["commands"]["req-1"]["acknowledgement"] == "accepted"


def test_status_reports_recovering_during_startup_reconciliation(
    tmp_path: Path,
) -> None:
    state = AttemptStateStore(tmp_path)
    state.start(issue_number=24, branch="agent/issue-24")
    lifecycle, tracker, store = make_lifecycle(tmp_path, next_claim=None)
    seen: dict[str, object] = {}
    entered = threading.Event()
    release = threading.Event()

    original_recover = tracker.recover_claim

    def blocking_recover(issue_number: int):
        entered.set()
        assert release.wait(timeout=30)
        return original_recover(issue_number)

    tracker.recover_claim = blocking_recover  # type: ignore[method-assign]
    worker = threading.Thread(target=lifecycle.run_once)
    worker.start()
    assert entered.wait(timeout=30)

    seen.update(lifecycle.control_status("owner/repo"))

    release.set()
    worker.join(timeout=30)

    assert seen["recovering"] is True
    assert seen["intake"] == "recovering"


def test_restart_reconciles_before_intake_and_then_holds_stopped(
    tmp_path: Path,
) -> None:
    first, tracker, store = make_lifecycle(tmp_path, next_claim=claim(24))
    first.submit_stop("req-idle")
    assert store.intake_state() is IntakeState.STOPPED

    second, tracker2, _ = make_lifecycle(tmp_path, next_claim=claim(25))
    result = second.run_once()

    assert result.status is LifecycleStatus.IDLE
    assert tracker2.claimed == []


# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------


def claim(number: int) -> Claim:
    return Claim(issue(number), Assignment(f"issue-{number}", "agent-id"))


def issue(number: int) -> TrackerIssue:
    return TrackerIssue(
        id=f"issue-{number}",
        number=number,
        title="Implement lifecycle",
        body="Connect boundaries.",
        created_at=datetime(2026, 9, 20, tzinfo=UTC),
        state="OPEN",
        labels=frozenset({"ready-for-agent"}),
        assignee_logins=(),
        blocked_by=0,
        author_login="reporter",
    )


def evidence(outcome: AttemptOutcome | None = None):
    return _evidence(outcome)


def _evidence(outcome: AttemptOutcome | None = None):
    from simple_coding_agent.lifecycle import AttemptEvidence

    return AttemptEvidence(
        decision=CompletionDecision(outcome, True, PublicationPath.COMPLETE, ("ready",)),
        check_command="pytest",
        check_exit_code=0,
        review_cycles=1,
        review_findings="all clear",
        details="Implemented the lifecycle.",
    )


def blocking_evidence(
    started: threading.Event, release: threading.Event, outcome: AttemptOutcome | None
):
    def run(received_claim: Claim, profile: object, prepared: object):
        started.set()
        assert release.wait(timeout=30)
        return _evidence(outcome)

    return run


class FakeTracker:
    def __init__(self, next_claim: Claim | None) -> None:
        self.next_claim = next_claim
        self.cleanup: list[tuple[int, str, str]] = []
        self.claimed: list[int] = []

    def claim_next(self) -> Claim | None:
        claim_value, self.next_claim = self.next_claim, None
        if claim_value is not None:
            self.claimed.append(claim_value.issue.number)
        return claim_value

    def recover_claim(self, issue_number: int) -> Claim | None:
        return Claim(issue(issue_number), Assignment(f"issue-{issue_number}", "agent-id"))

    def release_attempt(self, issue_number: int, label: str, assignee_id: str) -> None:
        self.cleanup.append((issue_number, label, assignee_id))

    def release_handoff(self, issue_number: int, assignee_id: str) -> None:
        self.cleanup.append((issue_number, "round-finished", assignee_id))


class FakeWorkspace:
    working_directory = Path("/repository")

    def __init__(self) -> None:
        self.cleanup_calls: list[tuple[str, bool]] = []
        self.dirty = False

    def is_clean(self) -> bool:
        return not self.dirty

    def prepare_for_profile_read(self, *, base_branch: str = "main") -> None:
        return None

    def prepare_attempt(self, *, base_branch: str, issue_number: int):
        return type("Prepared", (), {"branch": f"agent/issue-{issue_number}"})()

    def cleanup(self, *, base_branch: str, prepared: object, retain_branch: bool) -> None:
        self.cleanup_calls.append((base_branch, retain_branch))


class FakePublisher:
    def __init__(self) -> None:
        self.requests: list[object] = []

    def publish(self, request: object):
        self.requests.append(request)
        outcome = request.decision.outcome or AttemptOutcome.COMPLETE
        return type("Published", (), {"outcome": outcome, "branch_url": "https://example.test/x"})()


def make_lifecycle(tmp_path: Path, *, next_claim: Claim | None):
    from simple_coding_agent.lifecycle import AgentLifecycle

    tracker = FakeTracker(next_claim)
    workspace = FakeWorkspace()
    store = ControlStore(tmp_path)
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=lambda received_claim, profile, prepared: _evidence(None),
        control_store=store,
        sleeper=lambda seconds: None,
    )
    return lifecycle, tracker, store


def workspace_of(lifecycle: AgentLifecycle):
    return lifecycle._workspace  # noqa: SLF001 - test seam


# ---------------------------------------------------------------------------
# resume (issue #81)
# ---------------------------------------------------------------------------


def test_resume_from_stopped_permits_a_later_claim(tmp_path: Path) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, next_claim=claim(24))

    lifecycle.submit_stop("req-stop")
    assert store.intake_state() is IntakeState.STOPPED
    assert lifecycle.run_once().status is LifecycleStatus.IDLE
    assert tracker.claimed == []

    record = lifecycle.submit_resume("req-resume")

    assert record.acknowledgement is CommandAcknowledgement.COMPLETED
    assert store.intake_state() is IntakeState.RUNNING
    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.ATTEMPTED
    assert tracker.claimed == [24]


def test_resume_during_an_active_attempt_cancels_the_stop_plan(
    tmp_path: Path,
) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, next_claim=claim(24))
    started = threading.Event()
    release = threading.Event()
    lifecycle.set_attempt_runner(blocking_evidence(started, release, AttemptOutcome.COMPLETE))
    worker = threading.Thread(target=lifecycle.run_once)
    worker.start()
    assert started.wait(timeout=30)

    stop = lifecycle.submit_stop("req-stop")
    assert stop.acknowledgement is CommandAcknowledgement.ACCEPTED

    resume = lifecycle.submit_resume("req-resume")

    assert resume.acknowledgement is CommandAcknowledgement.COMPLETED
    assert "req-stop" in resume.detail
    assert store.intake_state() is IntakeState.RUNNING
    replaced = store.get_command("req-stop")
    assert replaced is not None
    assert replaced.acknowledgement is CommandAcknowledgement.SUPERSEDED
    assert "req-resume" in replaced.detail

    release.set()
    worker.join(timeout=30)
    assert not worker.is_alive()

    # The attempt keeps its ordinary outcome and finalizes without stopping.
    assert tracker.cleanup == [(24, "ready-for-agent", "agent-id")]
    assert store.intake_state() is IntakeState.RUNNING

    tracker.next_claim = claim(25)
    assert lifecycle.run_once().status is LifecycleStatus.ATTEMPTED
    assert tracker.claimed == [24, 25]


def test_resume_while_running_without_a_plan_is_idempotent(tmp_path: Path) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, next_claim=None)

    first = lifecycle.submit_resume("req-r1")
    second = lifecycle.submit_resume("req-r2")

    assert first.acknowledgement is CommandAcknowledgement.COMPLETED
    assert second.acknowledgement is CommandAcknowledgement.COMPLETED
    assert store.intake_state() is IntakeState.RUNNING
    assert tracker.claimed == []


def test_resume_rejected_while_retained_finalization_is_unresolved(
    tmp_path: Path,
) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, next_claim=claim(24))
    AttemptStateStore(tmp_path).start(issue_number=24, branch="agent/issue-24")

    with pytest.raises(ResumeBlockedError, match=r"issue #24"):
        lifecycle.submit_resume("req-resume")

    assert store.intake_state() is IntakeState.RUNNING
    assert store.get_command("req-resume") is None
    assert tracker.claimed == []


def test_resume_rejected_while_startup_reconciliation_runs(tmp_path: Path) -> None:
    state = AttemptStateStore(tmp_path)
    state.start(issue_number=24, branch="agent/issue-24")
    lifecycle, tracker, store = make_lifecycle(tmp_path, next_claim=None)
    entered = threading.Event()
    release = threading.Event()
    original_recover = tracker.recover_claim

    def blocking_recover(issue_number: int):
        entered.set()
        assert release.wait(timeout=30)
        return original_recover(issue_number)

    tracker.recover_claim = blocking_recover  # type: ignore[method-assign]
    worker = threading.Thread(target=lifecycle.run_once)
    worker.start()
    assert entered.wait(timeout=30)

    try:
        with pytest.raises(ResumeBlockedError, match="reconciliation"):
            lifecycle.submit_resume("req-resume")
    finally:
        release.set()
        worker.join(timeout=30)

    assert store.get_command("req-resume") is None


def test_resume_never_requeues_a_round_finished_issue(tmp_path: Path) -> None:
    from simple_coding_agent.lifecycle import AttemptEvidence

    lifecycle, tracker, store = make_lifecycle(tmp_path, next_claim=claim(24))

    def handoff_workflow(received_claim: Claim, profile: object, prepared: object):
        return AttemptEvidence(
            decision=CompletionDecision(
                AttemptOutcome.HANDOFF, False, PublicationPath.PARTIAL, ("handoff note",)
            ),
            check_command="pytest",
            check_exit_code=None,
            review_cycles=0,
            review_findings="not run",
            details="handoff note",
        )

    lifecycle.set_attempt_runner(handoff_workflow)
    assert lifecycle.run_once().status is LifecycleStatus.ATTEMPTED
    assert tracker.cleanup == [(24, "round-finished", "agent-id")]

    claimed_before = list(tracker.claimed)
    cleanup_before = list(tracker.cleanup)
    record = lifecycle.submit_resume("req-resume")

    assert record.acknowledgement is CommandAcknowledgement.COMPLETED
    # Resume performs no claim, release, or label change of its own.
    assert tracker.claimed == claimed_before
    assert tracker.cleanup == cleanup_before

    # The round-finished issue stays out of the runnable queue.
    tracker.next_claim = None
    assert lifecycle.run_once().status is LifecycleStatus.IDLE
    assert tracker.cleanup == cleanup_before


def test_restart_after_resume_reconciles_before_claim(tmp_path: Path) -> None:
    from simple_coding_agent.lifecycle import AgentLifecycle

    first, _, store = make_lifecycle(tmp_path, next_claim=claim(24))
    first.submit_stop("req-stop")
    first.submit_resume("req-resume")
    assert store.intake_state() is IntakeState.RUNNING

    tracker = FakeTracker(claim(25))
    workspace = FakeWorkspace()
    restarted = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=lambda received_claim, profile, prepared: _evidence(None),
        control_store=store,
        sleeper=lambda seconds: None,
    )
    result = restarted.run_once()

    assert result.status is LifecycleStatus.ATTEMPTED
    assert tracker.claimed == [25]


def test_resume_rejected_while_unexplained_dirt_has_no_attempt(
    tmp_path: Path,
) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, next_claim=claim(24))
    workspace_of(lifecycle).dirty = True

    with pytest.raises(ResumeBlockedError, match="dirty or untracked"):
        lifecycle.submit_resume("req-resume")

    assert store.intake_state() is IntakeState.RUNNING
    assert store.get_command("req-resume") is None
    assert tracker.claimed == []

    # After manual repair, resume is accepted again.
    workspace_of(lifecycle).dirty = False
    record = lifecycle.submit_resume("req-resume")
    assert record.acknowledgement is CommandAcknowledgement.COMPLETED


def test_resume_during_an_active_attempt_ignores_the_attempts_own_dirt(
    tmp_path: Path,
) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, next_claim=claim(24))
    started = threading.Event()
    release = threading.Event()
    lifecycle.set_attempt_runner(blocking_evidence(started, release, AttemptOutcome.COMPLETE))
    worker = threading.Thread(target=lifecycle.run_once)
    worker.start()
    assert started.wait(timeout=30)

    # Dirt appearing while the attempt runs is the attempt's own work.
    workspace_of(lifecycle).dirty = True
    try:
        resume = lifecycle.submit_resume("req-resume")
    finally:
        release.set()
        worker.join(timeout=30)

    assert resume.acknowledgement is CommandAcknowledgement.COMPLETED
    assert not worker.is_alive()
    assert tracker.cleanup == [(24, "ready-for-agent", "agent-id")]
    assert store.intake_state() is IntakeState.RUNNING
