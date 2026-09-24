"""Stop after N implementation attempts (issue #82).

An operator can set ``agentctl stop --after N`` on the selected instance. It
counts every fully finalized implementation attempt, including the one active
when the command is accepted, and stops intake after the Nth. Status displays
the requested and remaining count and the latest counted outcome.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import threading

import pytest

from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.completion import AttemptOutcome, CompletionDecision, PublicationPath
from simple_coding_agent.control import (
    CommandAcknowledgement,
    ControlStore,
    IntakeState,
    PayloadMismatchError,
    StopAfterRejectedError,
    parse_stop_after,
)
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.lifecycle import AgentLifecycle, LifecycleStatus


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [0, -1, -100, 3.5, 2.0, "abc", "", "2.5", None, True, False, [2], {"after": 2}])
def test_rejects_nonpositive_or_noninteger_without_state_change(
    tmp_path: Path, raw: object
) -> None:
    store = ControlStore(tmp_path)

    with pytest.raises(StopAfterRejectedError):
        store.submit_stop_after("req-bad", raw, has_active_attempt=False)

    assert store.intake_state() is IntakeState.RUNNING
    assert store.pending_command_id() is None
    assert store.stop_plan_snapshot() is None
    assert store.get_command("req-bad") is None
    assert store.recent_commands() == []


@pytest.mark.parametrize("raw", ["1", " 2 ", "+3", "007"])
def test_accepts_integer_valued_strings(tmp_path: Path, raw: str) -> None:
    store = ControlStore(tmp_path)

    record = store.submit_stop_after("req-ok", raw, has_active_attempt=False)

    assert record.acknowledgement is CommandAcknowledgement.ACCEPTED
    assert store.stop_plan_snapshot() is not None
    assert store.stop_plan_snapshot()["requested"] == int(raw.strip().lstrip("+"))


def test_parse_stop_after_rejects_bad_values() -> None:
    for raw in (0, -2, 1.5, "x", "", None, True):
        with pytest.raises(StopAfterRejectedError):
            parse_stop_after(raw)
    assert parse_stop_after(1) == 1
    assert parse_stop_after("4") == 4


def test_rejects_while_stopped_without_state_change(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop("req-stop", has_active_attempt=False)
    assert store.intake_state() is IntakeState.STOPPED

    with pytest.raises(StopAfterRejectedError, match="resume"):
        store.submit_stop_after("req-after", 2, has_active_attempt=False)

    assert store.intake_state() is IntakeState.STOPPED
    assert store.pending_command_id() is None
    assert store.stop_plan_snapshot() is None
    assert store.get_command("req-after") is None


def test_rejection_preserves_an_existing_plan(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    plan = store.submit_stop_after("req-plan", 3, has_active_attempt=False)

    with pytest.raises(StopAfterRejectedError):
        store.submit_stop_after("req-bad", 0, has_active_attempt=False)
    with pytest.raises(StopAfterRejectedError):
        store.submit_stop_after("req-bad-2", "many", has_active_attempt=False)

    assert store.pending_command_id() == "req-plan"
    snapshot = store.stop_plan_snapshot()
    assert snapshot is not None
    assert snapshot["requested"] == 3
    assert snapshot["remaining"] == 3
    assert store.intake_state() is IntakeState.RUNNING
    stored = store.get_command("req-plan")
    assert stored is not None
    assert stored.sequence == plan.sequence
    assert stored.acknowledgement is CommandAcknowledgement.ACCEPTED


# ---------------------------------------------------------------------------
# acceptance and counting
# ---------------------------------------------------------------------------


def test_active_attempt_is_count_one_and_n1_enters_stopping(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)

    record = store.submit_stop_after(
        "req-1", 1, has_active_attempt=True, active_attempt_id="attempt-a"
    )

    assert record.acknowledgement is CommandAcknowledgement.ACCEPTED
    assert record.kind == "stop_after"
    assert store.intake_state() is IntakeState.STOPPING
    assert store.pending_command_id() == "req-1"
    snapshot = store.stop_plan_snapshot()
    assert snapshot == {
        "request_id": "req-1",
        "kind": "stop_after",
        "requested": 1,
        "remaining": 1,
        "includes_active_attempt": True,
        "active_attempt_id": "attempt-a",
        "latest_counted_attempt_id": None,
        "latest_counted_outcome": None,
    }


def test_plan_above_one_stays_running_with_visible_count(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)

    store.submit_stop_after("req-1", 3, has_active_attempt=True, active_attempt_id="attempt-a")

    assert store.intake_state() is IntakeState.RUNNING
    snapshot = store.stop_plan_snapshot()
    assert snapshot is not None
    assert snapshot["requested"] == 3
    assert snapshot["remaining"] == 3
    assert snapshot["includes_active_attempt"] is True


def test_plan_without_active_attempt_waits_for_the_next_claim(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)

    record = store.submit_stop_after("req-1", 2, has_active_attempt=False)

    assert record.acknowledgement is CommandAcknowledgement.ACCEPTED
    assert store.intake_state() is IntakeState.RUNNING
    snapshot = store.stop_plan_snapshot()
    assert snapshot is not None
    assert snapshot["includes_active_attempt"] is False
    assert snapshot["active_attempt_id"] is None


@pytest.mark.parametrize(
    "outcome",
    [
        AttemptOutcome.COMPLETE,
        AttemptOutcome.INCOMPLETE,
        AttemptOutcome.INFRASTRUCTURE_ERROR,
        AttemptOutcome.NO_CHANGES,
        AttemptOutcome.HANDOFF,
    ],
)
def test_every_terminal_outcome_counts_exactly_once(
    tmp_path: Path, outcome: AttemptOutcome
) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop_after("req-1", 2, has_active_attempt=True, active_attempt_id="attempt-a")

    completed = store.record_attempt_finalized("attempt-a", outcome.value)

    assert completed == []
    assert store.intake_state() is IntakeState.RUNNING
    snapshot = store.stop_plan_snapshot()
    assert snapshot is not None
    assert snapshot["remaining"] == 1
    assert snapshot["latest_counted_attempt_id"] == "attempt-a"
    assert snapshot["latest_counted_outcome"] == outcome.value

    # Replaying the same attempt never decrements twice.
    assert store.record_attempt_finalized("attempt-a", outcome.value) == []
    assert store.stop_plan_snapshot()["remaining"] == 1  # type: ignore[index]


def test_final_counted_attempt_transitions_to_stopped_atomically(
    tmp_path: Path,
) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop_after("req-1", 1, has_active_attempt=False)
    assert store.intake_state() is IntakeState.RUNNING

    assert store.enter_stopping_for_final_claim() is True
    assert store.intake_state() is IntakeState.STOPPING
    assert store.pending_command_id() == "req-1"

    completed = store.record_attempt_finalized("attempt-z", AttemptOutcome.COMPLETE.value)

    assert [record.request_id for record in completed] == ["req-1"]
    assert completed[0].acknowledgement is CommandAcknowledgement.COMPLETED
    assert store.intake_state() is IntakeState.STOPPED
    assert store.pending_command_id() is None
    assert store.stop_plan_snapshot() is None
    # Replaying the final attempt after the transition cannot reopen accounting.
    assert store.record_attempt_finalized("attempt-z", AttemptOutcome.COMPLETE.value) == []
    assert store.intake_state() is IntakeState.STOPPED


def test_non_final_claim_leaves_running_intake(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop_after("req-1", 3, has_active_attempt=False)

    assert store.enter_stopping_for_final_claim() is False
    assert store.intake_state() is IntakeState.RUNNING
    assert store.pending_command_id() == "req-1"


def test_repeated_attempts_on_one_issue_count_separately(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop_after("req-1", 3, has_active_attempt=True, active_attempt_id="try-1")

    store.record_attempt_finalized("try-1", AttemptOutcome.INCOMPLETE.value)
    store.record_attempt_finalized("try-2", AttemptOutcome.INCOMPLETE.value)

    snapshot = store.stop_plan_snapshot()
    assert snapshot is not None
    assert snapshot["remaining"] == 1
    assert snapshot["latest_counted_attempt_id"] == "try-2"


# ---------------------------------------------------------------------------
# replacement
# ---------------------------------------------------------------------------


def test_replacement_starts_a_fresh_count_and_supersedes(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    # The first plan is replaced while its counted attempt is still active:
    # nothing has been finalized yet, so no count exists to carry forward.
    store.submit_stop_after("req-old", 3, has_active_attempt=True, active_attempt_id="attempt-a")

    record = store.submit_stop_after(
        "req-new", 2, has_active_attempt=True, active_attempt_id="attempt-a"
    )

    assert record.acknowledgement is CommandAcknowledgement.ACCEPTED
    snapshot = store.stop_plan_snapshot()
    assert snapshot is not None
    assert snapshot["request_id"] == "req-new"
    assert snapshot["requested"] == 2
    assert snapshot["remaining"] == 2
    # Fresh count: the previous plan's progress is not carried forward.
    assert snapshot["latest_counted_attempt_id"] is None
    assert snapshot["active_attempt_id"] == "attempt-a"
    old = store.get_command("req-old")
    assert old is not None
    assert old.acknowledgement is CommandAcknowledgement.SUPERSEDED
    assert "req-new" in old.detail
    assert "req-old" in record.detail

    # The shared active attempt decrements only the current plan, once.
    store.record_attempt_finalized("attempt-a", AttemptOutcome.COMPLETE.value)
    assert store.stop_plan_snapshot()["remaining"] == 1  # type: ignore[index]
    assert store.record_attempt_finalized("attempt-a", AttemptOutcome.COMPLETE.value) == []
    assert store.stop_plan_snapshot()["remaining"] == 1  # type: ignore[index]


def test_already_counted_attempt_never_decrements_a_replacement_plan(
    tmp_path: Path,
) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop_after("req-old", 3, has_active_attempt=True, active_attempt_id="attempt-a")
    store.record_attempt_finalized("attempt-a", AttemptOutcome.COMPLETE.value)
    assert store.stop_plan_snapshot()["remaining"] == 2  # type: ignore[index]

    # The old attempt already finalized: the replacement starts fresh and the
    # replayed attempt ID can never decrement it.
    store.submit_stop_after("req-new", 2, has_active_attempt=False)
    assert store.record_attempt_finalized("attempt-a", AttemptOutcome.COMPLETE.value) == []
    assert store.stop_plan_snapshot()["remaining"] == 2  # type: ignore[index]


def test_replacement_without_active_attempt_starts_with_next_attempt(
    tmp_path: Path,
) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop_after("req-old", 5, has_active_attempt=False)
    store.record_attempt_finalized("attempt-a", AttemptOutcome.COMPLETE.value)

    store.submit_stop_after("req-new", 1, has_active_attempt=False)

    snapshot = store.stop_plan_snapshot()
    assert snapshot is not None
    assert snapshot["requested"] == 1
    assert snapshot["remaining"] == 1
    assert snapshot["includes_active_attempt"] is False
    assert store.intake_state() is IntakeState.RUNNING


def test_plain_stop_supersedes_a_stop_after_plan(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop_after("req-after", 3, has_active_attempt=False)

    record = store.submit_stop("req-stop", has_active_attempt=False)

    assert record.acknowledgement is CommandAcknowledgement.COMPLETED
    assert store.intake_state() is IntakeState.STOPPED
    assert store.stop_plan_snapshot() is None
    old = store.get_command("req-after")
    assert old is not None
    assert old.acknowledgement is CommandAcknowledgement.SUPERSEDED
    assert "req-stop" in old.detail


def test_resume_clears_a_stop_after_plan(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop_after("req-after", 3, has_active_attempt=True, active_attempt_id="a")

    record = store.submit_resume("req-resume", has_active_attempt=True)

    assert record.acknowledgement is CommandAcknowledgement.COMPLETED
    assert store.intake_state() is IntakeState.RUNNING
    assert store.stop_plan_snapshot() is None
    old = store.get_command("req-after")
    assert old is not None
    assert old.acknowledgement is CommandAcknowledgement.SUPERSEDED
    assert "req-resume" in old.detail


def test_identical_retry_returns_the_same_record(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    first = store.submit_stop_after("req-1", 2, has_active_attempt=False)

    retry = ControlStore(tmp_path).submit_stop_after("req-1", 2, has_active_attempt=True)

    assert retry.sequence == first.sequence
    assert retry.acknowledgement is first.acknowledgement
    assert store.stop_plan_snapshot()["remaining"] == 2  # type: ignore[index]


def test_request_id_reused_with_different_payload_is_rejected(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop_after("req-1", 2, has_active_attempt=False)

    with pytest.raises(PayloadMismatchError):
        store.submit_stop_after("req-1", 3, has_active_attempt=False)
    with pytest.raises(PayloadMismatchError):
        store.submit_stop("req-1", has_active_attempt=False)

    assert store.stop_plan_snapshot()["remaining"] == 2  # type: ignore[index]


def test_plan_and_markers_survive_restart(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop_after("req-1", 2, has_active_attempt=True, active_attempt_id="attempt-a")
    store.record_attempt_finalized("attempt-a", AttemptOutcome.INCOMPLETE.value)

    restarted = ControlStore(tmp_path)

    assert restarted.intake_state() is IntakeState.RUNNING
    snapshot = restarted.stop_plan_snapshot()
    assert snapshot is not None
    assert snapshot["remaining"] == 1
    assert snapshot["latest_counted_attempt_id"] == "attempt-a"
    # The counted marker survived: a replay after restart still cannot double-count.
    assert restarted.record_attempt_finalized("attempt-a", AttemptOutcome.INCOMPLETE.value) == []
    assert restarted.stop_plan_snapshot()["remaining"] == 1  # type: ignore[index]


def test_migrates_a_pre_stop_after_database(tmp_path: Path) -> None:
    import sqlite3

    data_dir = tmp_path / "old"
    first = ControlStore(data_dir)
    first.submit_stop("req-stop", has_active_attempt=True)
    path = data_dir / "state" / "control.sqlite3"
    with sqlite3.connect(str(path)) as connection:
        connection.execute("ALTER TABLE control_state DROP COLUMN stop_after_requested")
        connection.execute("ALTER TABLE control_state DROP COLUMN stop_after_remaining")
        connection.execute(
            "ALTER TABLE control_state DROP COLUMN stop_after_active_attempt_id"
        )
        connection.execute("ALTER TABLE control_state DROP COLUMN latest_counted_attempt_id")
        connection.execute("ALTER TABLE control_state DROP COLUMN latest_counted_outcome")
        connection.execute("DROP TABLE counted_attempts")
        connection.commit()

    migrated = ControlStore(data_dir)

    assert migrated.intake_state() is IntakeState.STOPPING
    record = migrated.submit_stop_after("req-new", 2, has_active_attempt=True, active_attempt_id="a")
    assert record.acknowledgement is CommandAcknowledgement.ACCEPTED
    assert migrated.stop_plan_snapshot()["remaining"] == 2  # type: ignore[index]


# ---------------------------------------------------------------------------
# lifecycle integration
# ---------------------------------------------------------------------------


def claim(number: int) -> Claim:
    return Claim(issue(number), Assignment(f"issue-{number}", "agent-id"))


def issue(number: int) -> TrackerIssue:
    return TrackerIssue(
        id=f"issue-{number}",
        number=number,
        title="Implement feature",
        body="Do the work.",
        created_at=datetime(2026, 9, 20, tzinfo=UTC),
        state="OPEN",
        labels=frozenset({"ready-for-agent"}),
        assignee_logins=(),
        blocked_by=0,
        author_login="reporter",
    )


def lifecycle_evidence(outcome: AttemptOutcome | None):
    from simple_coding_agent.lifecycle import AttemptEvidence

    return AttemptEvidence(
        decision=CompletionDecision(outcome, True, PublicationPath.COMPLETE, ("ready",)),
        check_command="pytest",
        check_exit_code=0,
        review_cycles=1,
        review_findings="all clear",
        details="Implemented the change.",
    )


class FakeTracker:
    def __init__(self, numbers: list[int]) -> None:
        self._pending = [claim(number) for number in numbers]
        self.claimed: list[int] = []
        self.cleanup: list[tuple[int, str, str]] = []

    def claim_next(self) -> Claim | None:
        if not self._pending:
            return None
        value = self._pending.pop(0)
        self.claimed.append(value.issue.number)
        return value

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
        outcome = request.decision.outcome or AttemptOutcome.COMPLETE  # type: ignore[union-attr]
        return type("Published", (), {"outcome": outcome, "branch_url": "https://example.test/x"})()


def make_lifecycle(tmp_path: Path, numbers: list[int], outcomes: list[AttemptOutcome | None]):
    from simple_coding_agent.lifecycle import AgentLifecycle

    tracker = FakeTracker(numbers)
    workspace = FakeWorkspace()
    store = ControlStore(tmp_path)
    pending = list(outcomes)

    def attempt_runner(received_claim: Claim, profile: object, prepared: object):
        return lifecycle_evidence(pending.pop(0))

    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=attempt_runner,
        control_store=store,
        sleeper=lambda seconds: None,
    )
    return lifecycle, tracker, store


def test_attempt_completed_before_acceptance_is_not_counted(tmp_path: Path) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, [24, 25], [AttemptOutcome.COMPLETE] * 2)
    assert lifecycle.run_once().status is LifecycleStatus.ATTEMPTED
    assert tracker.claimed == [24]

    # The first attempt already finalized before the plan: counting starts next.
    record = lifecycle.submit_stop_after("req-after", 1)

    assert record.acknowledgement is CommandAcknowledgement.ACCEPTED
    snapshot = store.stop_plan_snapshot()
    assert snapshot is not None
    assert snapshot["includes_active_attempt"] is False
    assert snapshot["remaining"] == 1

    assert lifecycle.run_once().status is LifecycleStatus.ATTEMPTED
    assert tracker.claimed == [24, 25]
    assert store.intake_state() is IntakeState.STOPPED


def test_stop_after_two_counts_active_first_then_stops(tmp_path: Path) -> None:
    lifecycle, tracker, store = make_lifecycle(
        tmp_path, [24, 25, 26], [AttemptOutcome.COMPLETE, AttemptOutcome.INCOMPLETE]
    )
    started = threading.Event()
    release = threading.Event()
    original_runner = lifecycle._attempt_runner  # noqa: SLF001

    def blocking_first(received_claim: Claim, profile: object, prepared: object):
        started.set()
        assert release.wait(timeout=30)
        return lifecycle_evidence(AttemptOutcome.COMPLETE)

    lifecycle.set_attempt_runner(blocking_first)
    worker = threading.Thread(target=lifecycle.run_once)
    worker.start()
    assert started.wait(timeout=30)

    record = lifecycle.submit_stop_after("req-after", 2)

    assert record.acknowledgement is CommandAcknowledgement.ACCEPTED
    assert store.intake_state() is IntakeState.RUNNING
    release.set()
    worker.join(timeout=30)

    snapshot = store.stop_plan_snapshot()
    assert snapshot is not None
    assert snapshot["remaining"] == 1
    assert snapshot["latest_counted_outcome"] == AttemptOutcome.COMPLETE.value

    lifecycle.set_attempt_runner(original_runner)
    tracker._pending.extend([claim(26)])
    assert lifecycle.run_once().status is LifecycleStatus.ATTEMPTED
    assert tracker.claimed == [24, 25]
    # The final counted attempt entered stopping at claim and is now stopped.
    assert store.intake_state() is IntakeState.STOPPED
    assert store.stop_plan_snapshot() is None
    assert store.get_command("req-after").acknowledgement is CommandAcknowledgement.COMPLETED  # type: ignore[union-attr]

    # No further claim is possible after the boundary.
    assert lifecycle.run_once().status is LifecycleStatus.IDLE
    assert tracker.claimed == [24, 25]


def test_empty_queue_preserves_the_remaining_count(tmp_path: Path) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, [], [AttemptOutcome.COMPLETE])
    lifecycle.submit_stop_after("req-after", 2)

    assert lifecycle.run_once().status is LifecycleStatus.IDLE

    snapshot = store.stop_plan_snapshot()
    assert snapshot is not None
    assert snapshot["remaining"] == 2
    assert store.intake_state() is IntakeState.RUNNING


def test_all_five_outcomes_count_through_the_lifecycle(tmp_path: Path) -> None:
    outcomes = [
        AttemptOutcome.COMPLETE,
        AttemptOutcome.INCOMPLETE,
        AttemptOutcome.INFRASTRUCTURE_ERROR,
        AttemptOutcome.NO_CHANGES,
        AttemptOutcome.HANDOFF,
    ]
    lifecycle, tracker, store = make_lifecycle(tmp_path, [21, 22, 23, 24, 25], list(outcomes))
    lifecycle.submit_stop_after("req-after", 5)

    for _ in outcomes:
        assert lifecycle.run_once().status is LifecycleStatus.ATTEMPTED

    assert tracker.claimed == [21, 22, 23, 24, 25]
    assert store.intake_state() is IntakeState.STOPPED
    assert store.get_command("req-after").acknowledgement is CommandAcknowledgement.COMPLETED  # type: ignore[union-attr]
    assert lifecycle.run_once().status is LifecycleStatus.IDLE
    assert tracker.claimed == [21, 22, 23, 24, 25]


def test_held_finalization_does_not_consume_a_count(tmp_path: Path) -> None:
    from simple_coding_agent.lifecycle import AgentLifecycle

    tracker = FakeTracker([24])
    workspace = FakeWorkspace()
    store = ControlStore(tmp_path)

    class UnconfirmedPublisher(FakePublisher):
        def publish(self, request: object):
            self.requests.append(request)
            outcome = request.decision.outcome or AttemptOutcome.COMPLETE  # type: ignore[union-attr]
            return type(
                "Published",
                (),
                {"outcome": outcome, "branch_url": "https://example.test/x", "comment_posted": False},
            )()

    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=UnconfirmedPublisher(),
        attempt_runner=lambda c, p, r: lifecycle_evidence(AttemptOutcome.COMPLETE),
        control_store=store,
        sleeper=lambda seconds: None,
    )
    lifecycle.submit_stop_after("req-after", 2)
    assert lifecycle.run_once().status is LifecycleStatus.ATTEMPTED

    snapshot = store.stop_plan_snapshot()
    assert snapshot is not None
    assert snapshot["remaining"] == 2
    assert snapshot["latest_counted_attempt_id"] is None
    assert AttemptStateStore(tmp_path).read() is not None


def test_recovery_after_json_record_counts_once_without_double_publish(
    tmp_path: Path,
) -> None:
    from simple_coding_agent.finalization import AttemptCompletionStore
    from simple_coding_agent.lifecycle import AgentLifecycle

    state = AttemptStateStore(tmp_path)
    checkpoint = state.start(issue_number=24, branch="agent/issue-24")
    completion_store = AttemptCompletionStore(tmp_path)
    completion_store.record(
        attempt_id=checkpoint.started_at,
        issue_number=24,
        branch="agent/issue-24",
        outcome=AttemptOutcome.INCOMPLETE,
    )
    store = ControlStore(tmp_path)
    store.submit_stop_after(
        "req-after", 1, has_active_attempt=True, active_attempt_id=checkpoint.started_at
    )
    assert store.intake_state() is IntakeState.STOPPING

    tracker = FakeTracker([])
    restarted = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=FakeWorkspace(),
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=lambda c, p, r: lifecycle_evidence(AttemptOutcome.COMPLETE),
        control_store=store,
        completion_store=completion_store,
        sleeper=lambda seconds: None,
    )
    assert restarted.run_once().status is LifecycleStatus.IDLE

    assert store.intake_state() is IntakeState.STOPPED
    assert store.get_command("req-after").acknowledgement is CommandAcknowledgement.COMPLETED  # type: ignore[union-attr]
    assert tracker.claimed == []
    # A second restart replays nothing: the counted marker survives.
    assert restarted.run_once().status is LifecycleStatus.IDLE
    assert store.intake_state() is IntakeState.STOPPED


def test_recovery_with_marker_but_leftover_checkpoint_does_not_recount(
    tmp_path: Path,
) -> None:
    from simple_coding_agent.finalization import AttemptCompletionStore
    from simple_coding_agent.lifecycle import AgentLifecycle

    state = AttemptStateStore(tmp_path)
    checkpoint = state.start(issue_number=24, branch="agent/issue-24")
    completion_store = AttemptCompletionStore(tmp_path)
    completion_store.record(
        attempt_id=checkpoint.started_at,
        issue_number=24,
        branch="agent/issue-24",
        outcome=AttemptOutcome.INCOMPLETE,
    )
    store = ControlStore(tmp_path)
    store.submit_stop_after(
        "req-after", 3, has_active_attempt=True, active_attempt_id=checkpoint.started_at
    )
    # Crash between the accounting transaction and checkpoint removal.
    assert store.record_attempt_finalized(checkpoint.started_at, AttemptOutcome.INCOMPLETE.value) == []
    assert store.stop_plan_snapshot()["remaining"] == 2  # type: ignore[index]

    tracker = FakeTracker([])
    restarted = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=FakeWorkspace(),
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=lambda c, p, r: lifecycle_evidence(AttemptOutcome.COMPLETE),
        control_store=store,
        completion_store=completion_store,
        sleeper=lambda seconds: None,
    )
    assert restarted.run_once().status is LifecycleStatus.IDLE

    assert store.stop_plan_snapshot()["remaining"] == 2  # type: ignore[index]
    assert store.intake_state() is IntakeState.RUNNING


def test_status_reports_the_countdown_and_latest_outcome(tmp_path: Path) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, [], [AttemptOutcome.COMPLETE])
    AttemptStateStore(tmp_path).start(issue_number=24, branch="agent/issue-24")
    lifecycle.submit_stop_after("req-after", 3)

    snapshot = lifecycle.control_status("owner/repo")

    plan = snapshot["stop_plan"]
    assert plan["request_id"] == "req-after"
    assert plan["kind"] == "stop_after"
    assert plan["requested"] == 3
    assert plan["remaining"] == 3
    assert plan["includes_active_attempt"] is True
    assert plan["latest_counted_attempt_id"] is None


# ---------------------------------------------------------------------------
# socket and CLI
# ---------------------------------------------------------------------------


def start_server(tmp_path: Path, numbers: list[int]):
    from simple_coding_agent.control_server import ControlServer

    lifecycle, _, _ = make_lifecycle(tmp_path, numbers, [AttemptOutcome.COMPLETE])
    socket_path = tmp_path / "state" / "agentctl.sock"
    server = ControlServer(socket_path, lifecycle)
    server.start()
    return server, lifecycle, socket_path


def send(socket_path: Path, payload: dict) -> dict:
    import json
    import socket as stdlib_socket

    with stdlib_socket.socket(stdlib_socket.AF_UNIX, stdlib_socket.SOCK_STREAM) as client:
        client.settimeout(5)
        client.connect(str(socket_path))
        client.sendall((json.dumps(payload) + "\n").encode())
        data = b""
        while not data.endswith(b"\n"):
            chunk = client.recv(65536)
            if not chunk:
                break
            data += chunk
    import json as json_module

    return json_module.loads(data.decode())


def test_stop_after_over_the_socket(tmp_path: Path) -> None:
    server, _, socket_path = start_server(tmp_path, [])
    try:
        reply = send(socket_path, {"op": "stop_after", "request_id": "req-1", "after": 2})
        assert reply["ok"] is True
        assert reply["acknowledgement"] == "accepted"
        assert reply["kind"] == "stop_after"

        status = send(socket_path, {"op": "status"})
        assert status["ok"] is True
        plan = status["status"]["stop_plan"]
        assert plan["requested"] == 2
        assert plan["remaining"] == 2

        bad = send(socket_path, {"op": "stop_after", "request_id": "req-bad", "after": 0})
        assert bad["ok"] is False
        assert status["status"]["stop_plan"]["remaining"] == 2
    finally:
        server.stop()


def test_stop_after_rejected_while_stopped_over_the_socket(tmp_path: Path) -> None:
    server, _, socket_path = start_server(tmp_path, [])
    try:
        assert send(socket_path, {"op": "stop", "request_id": "req-stop"})["ok"] is True
        reply = send(socket_path, {"op": "stop_after", "request_id": "req-1", "after": 2})
        assert reply["ok"] is False
        assert "resume" in reply["error"]
    finally:
        server.stop()


def test_agentctl_stop_after_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from simple_coding_agent.agentctl import main as agentctl_main

    server, _, socket_path = start_server(tmp_path, [])
    try:
        code = agentctl_main(["--socket", str(socket_path), "stop", "--after", "2"])
        assert code == 0
        out = capsys.readouterr().out
        assert "accepted" in out

        code = agentctl_main(["--socket", str(socket_path), "status"])
        assert code == 0
        out = capsys.readouterr().out
        assert "requested 2" in out
        assert "remaining 2" in out
    finally:
        server.stop()


@pytest.mark.parametrize("raw", ["0", "-3", "many", "2.5"])
def test_agentctl_stop_after_rejects_bad_counts_without_a_control_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], raw: str
) -> None:
    from simple_coding_agent.agentctl import main as agentctl_main

    server, lifecycle, socket_path = start_server(tmp_path, [])
    try:
        code = agentctl_main(["--socket", str(socket_path), "stop", "--after", raw])
        assert code == 1
        err = capsys.readouterr().err
        assert "rejected" in err
        assert "No control change was accepted" in err
        assert lifecycle.control_status()["stop_plan"] is None
    finally:
        server.stop()
