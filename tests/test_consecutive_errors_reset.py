"""Consecutive-error guard visibility and reset (issue #106).

``agentctl status`` shows the guard counter and ``agentctl errors reset``
clears it through the live process.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

from simple_coding_agent.agentctl import format_status
from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.completion import AttemptOutcome, CompletionDecision, PublicationPath
from simple_coding_agent.control import CommandAcknowledgement, ControlStore, IntakeState
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.lifecycle import AgentLifecycle
from simple_coding_agent.operating import ConsecutiveErrorStore
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def claim(number: int) -> Claim:
    return Claim(_issue(number), Assignment(f"issue-{number}", "agent-id"))


def _issue(number: int) -> TrackerIssue:
    return TrackerIssue(
        id=f"issue-{number}",
        number=number,
        title="Implement feature",
        body="Connect boundaries.",
        created_at=datetime(2026, 9, 20, tzinfo=UTC),
        state="OPEN",
        labels=frozenset({"ready-for-agent"}),
        assignee_logins=(),
        blocked_by=0,
        author_login="reporter",
    )


def _evidence(outcome: AttemptOutcome | None = None):
    from simple_coding_agent.lifecycle import AttemptEvidence

    return AttemptEvidence(
        decision=CompletionDecision(outcome, True, PublicationPath.COMPLETE, ("ready",)),
        check_command="pytest",
        check_exit_code=0,
        review_cycles=1,
        review_findings="all clear",
        details="Implemented the feature.",
    )


def make_lifecycle(tmp_path: Path, *, next_claim: Claim | None = None):
    events: list[tuple[str, str]] = []
    tracker = FakeTracker(next_claim=next_claim)
    workspace = InMemoryWorkspace(tmp_path / "workspace")
    store = ControlStore(tmp_path)
    error_store = ConsecutiveErrorStore(tmp_path)
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=lambda received_claim, profile, prepared: _evidence(None),
        control_store=store,
        error_store=error_store,
        max_consecutive_errors=3,
        sleeper=lambda seconds: None,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append((event, detail)),
        repository="owner/repo",
    )
    return lifecycle, tracker, store, error_store, events


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_shows_the_guard_count_limit_and_timestamps(tmp_path: Path) -> None:
    lifecycle, _, _, error_store, _ = make_lifecycle(tmp_path)
    error_store.record(AttemptOutcome.INFRASTRUCTURE_ERROR, "attempt-a")

    snapshot = lifecycle.control_status("owner/repo")

    section = snapshot["consecutive_errors"]
    assert section["count"] == 1
    assert section["max_consecutive_errors"] == 3
    assert section["last_attempt_id"] == "attempt-a"
    assert section["last_success_at"] is None
    assert section["error"] is None

    text = format_status(snapshot)
    assert "consecutive errors: 1 / 3" in text
    assert "last attempt attempt-a" in text


def test_status_reports_last_success_timestamp(tmp_path: Path) -> None:
    lifecycle, _, _, error_store, _ = make_lifecycle(tmp_path)
    error_store.record(AttemptOutcome.INFRASTRUCTURE_ERROR, "attempt-a")
    error_store.record(AttemptOutcome.COMPLETE, "attempt-b")
    error_store.record(AttemptOutcome.INFRASTRUCTURE_ERROR, "attempt-c")

    snapshot = lifecycle.control_status("owner/repo")

    assert snapshot["consecutive_errors"]["count"] == 1
    assert snapshot["consecutive_errors"]["last_success_at"] is not None
    text = format_status(snapshot)
    assert "consecutive errors: 1 / 3" in text
    assert "last success" in text


def test_status_marks_an_unreadable_guard_store_without_failing(tmp_path: Path) -> None:
    lifecycle, _, _, _, _ = make_lifecycle(tmp_path)
    state_path = tmp_path / "state" / "consecutive_errors.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{not-json")

    snapshot = lifecycle.control_status("owner/repo")

    section = snapshot["consecutive_errors"]
    assert section["count"] is None
    assert section["error"]
    assert "Consecutive-error" in section["error"]
    # The rest of the snapshot is intact.
    assert snapshot["repository"] == "owner/repo"
    assert snapshot["intake"] == IntakeState.RUNNING.value

    text = format_status(snapshot)
    assert "consecutive errors: unavailable" in text


def test_format_status_without_a_guard_renders_none() -> None:
    text = format_status(
        {
            "repository": "owner/repo",
            "intake": "running",
            "active_attempt": None,
            "pending_command": None,
            "pending_next_issue": None,
            "stop_plan": None,
            "recovery": None,
            "handoff": None,
            "consecutive_errors": None,
            "commands": {},
        }
    )

    assert "consecutive errors: none" in text


# ---------------------------------------------------------------------------
# reset while idle
# ---------------------------------------------------------------------------


def test_reset_while_idle_clears_the_count_and_records_the_command(tmp_path: Path) -> None:
    lifecycle, _, store, error_store, events = make_lifecycle(tmp_path)
    error_store.record(AttemptOutcome.INFRASTRUCTURE_ERROR, "attempt-a")
    error_store.record(AttemptOutcome.INFRASTRUCTURE_ERROR, "attempt-b")
    assert error_store.read().count == 2

    record = lifecycle.submit_errors_reset("req-reset-1")

    assert record.request_id == "req-reset-1"
    assert record.kind == "errors_reset"
    assert record.acknowledgement is CommandAcknowledgement.COMPLETED
    assert "2" in record.detail
    assert error_store.read().count == 0
    # No success happened: the success timestamp is untouched.
    assert error_store.read().last_success_at is None
    # Intake is unchanged and no stop plan appears.
    assert store.intake_state() is IntakeState.RUNNING
    assert store.pending_command_id() is None
    stored = store.get_command("req-reset-1")
    assert stored is not None
    assert stored.acknowledgement is CommandAcknowledgement.COMPLETED
    assert any(event == "consecutive_errors_reset" for event, _ in events)
    reset_detail = next(detail for event, detail in events if event == "consecutive_errors_reset")
    assert "req-reset-1" in reset_detail
    assert "2" in reset_detail


def test_reset_keeps_the_last_success_timestamp(tmp_path: Path) -> None:
    lifecycle, _, _, error_store, _ = make_lifecycle(tmp_path)
    error_store.record(AttemptOutcome.COMPLETE, "attempt-ok")
    success_at = error_store.read().last_success_at
    assert success_at is not None
    error_store.record(AttemptOutcome.INFRASTRUCTURE_ERROR, "attempt-a")

    lifecycle.submit_errors_reset("req-reset-keep")

    state = error_store.read()
    assert state.count == 0
    assert state.last_success_at == success_at


def test_retry_with_the_same_request_id_returns_the_stored_record(tmp_path: Path) -> None:
    lifecycle, _, store, error_store, _ = make_lifecycle(tmp_path)
    error_store.record(AttemptOutcome.INFRASTRUCTURE_ERROR, "attempt-a")

    first = lifecycle.submit_errors_reset("req-reset-retry")
    retry = lifecycle.submit_errors_reset("req-reset-retry")

    assert retry.sequence == first.sequence
    assert retry.acknowledgement is first.acknowledgement
    assert retry.detail == first.detail
    assert error_store.read().count == 0
    assert len(store.recent_commands()) == 1


def test_retry_after_new_errors_leaves_the_new_count_untouched(tmp_path: Path) -> None:
    lifecycle, _, _, error_store, _ = make_lifecycle(tmp_path)
    error_store.record(AttemptOutcome.INFRASTRUCTURE_ERROR, "attempt-a")
    first = lifecycle.submit_errors_reset("req-reset-stale")
    error_store.record(AttemptOutcome.INFRASTRUCTURE_ERROR, "attempt-b")
    assert error_store.read().count == 1

    retry = lifecycle.submit_errors_reset("req-reset-stale")

    assert retry.sequence == first.sequence
    assert retry.detail == first.detail
    assert error_store.read().count == 1


def test_infrastructure_error_after_a_reset_counts_from_one(tmp_path: Path) -> None:
    lifecycle, _, _, error_store, _ = make_lifecycle(tmp_path)
    error_store.record(AttemptOutcome.INFRASTRUCTURE_ERROR, "attempt-a")
    error_store.record(AttemptOutcome.INFRASTRUCTURE_ERROR, "attempt-b")
    lifecycle.submit_errors_reset("req-reset-count")

    lifecycle._record_terminal_outcome(  # noqa: SLF001 - guard seam
        AttemptOutcome.INFRASTRUCTURE_ERROR, "attempt-c", issue_number=24
    )

    assert error_store.read().count == 1


# ---------------------------------------------------------------------------
# reset while an attempt is active
# ---------------------------------------------------------------------------


def test_reset_while_an_attempt_is_active_applies_immediately(tmp_path: Path) -> None:
    lifecycle, tracker, store, error_store, _ = make_lifecycle(
        tmp_path, next_claim=claim(24)
    )
    error_store.record(AttemptOutcome.INFRASTRUCTURE_ERROR, "attempt-old")
    assert error_store.read().count == 1
    started = threading.Event()
    release = threading.Event()

    def blocking_workflow(received_claim: Claim, profile: object, prepared: object):
        started.set()
        assert release.wait(timeout=30)
        return _evidence(AttemptOutcome.COMPLETE)

    lifecycle.set_attempt_runner(blocking_workflow)
    worker = threading.Thread(target=lifecycle.run_once)
    worker.start()
    try:
        assert started.wait(timeout=30)

        record = lifecycle.submit_errors_reset("req-reset-active")

        assert record.acknowledgement is CommandAcknowledgement.COMPLETED
        assert error_store.read().count == 0
        assert store.get_command("req-reset-active") is not None
    finally:
        release.set()
        worker.join(timeout=30)

    assert not worker.is_alive()
    # The active attempt still finishes with its ordinary outcome.
    assert tracker.cleanup == [(24, "ready-for-agent", "agent-id")]


# ---------------------------------------------------------------------------
# control store unit behaviour
# ---------------------------------------------------------------------------


def test_control_store_reset_is_completed_and_idempotent(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)

    first = store.submit_errors_reset("req-1", previous_count=2)

    assert first.kind == "errors_reset"
    assert first.acknowledgement is CommandAcknowledgement.COMPLETED
    assert "2" in first.detail
    assert store.intake_state() is IntakeState.RUNNING

    retry = store.submit_errors_reset("req-1")

    assert retry.sequence == first.sequence
    assert retry.detail == first.detail


def test_control_store_reset_rejects_a_reused_id_with_a_different_kind(
    tmp_path: Path,
) -> None:
    import pytest

    from simple_coding_agent.control import PayloadMismatchError

    store = ControlStore(tmp_path)
    store.submit_stop("req-1", has_active_attempt=False)

    with pytest.raises(PayloadMismatchError):
        store.submit_errors_reset("req-1")


# ---------------------------------------------------------------------------
# socket + CLI end to end
# ---------------------------------------------------------------------------


def test_errors_reset_over_the_socket_and_cli(tmp_path: Path, capsys) -> None:
    from simple_coding_agent.agentctl import main as agentctl_main
    from simple_coding_agent.completion import AttemptOutcome
    from simple_coding_agent.control_server import ControlServer

    lifecycle, _, _, error_store, _ = make_lifecycle(tmp_path)
    error_store.record(AttemptOutcome.INFRASTRUCTURE_ERROR, "attempt-a")
    error_store.record(AttemptOutcome.INFRASTRUCTURE_ERROR, "attempt-b")
    socket_path = tmp_path / "run" / "control.sock"
    server = ControlServer(socket_path, lifecycle)
    server.start()
    try:
        code = agentctl_main(
            ["--socket", str(socket_path), "errors", "reset", "--request-id", "req-sock-1"]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "req-sock-1" in out
        assert "completed" in out
        assert error_store.read().count == 0

        capsys.readouterr()
        lookup_code = agentctl_main(["--socket", str(socket_path), "command", "req-sock-1"])
        assert lookup_code == 0
        lookup_out = capsys.readouterr().out
        assert "consecutive errors reset" in lookup_out

        capsys.readouterr()
        assert agentctl_main(["--socket", str(socket_path), "status"]) == 0
        status_out = capsys.readouterr().out
        assert "consecutive errors: 0 / 3" in status_out
    finally:
        server.stop()


def test_errors_reset_alias_consecutive_errors(tmp_path: Path, capsys) -> None:
    from simple_coding_agent.agentctl import main as agentctl_main
    from simple_coding_agent.control_server import ControlServer

    lifecycle, _, _, error_store, _ = make_lifecycle(tmp_path)
    socket_path = tmp_path / "run" / "control.sock"
    server = ControlServer(socket_path, lifecycle)
    server.start()
    try:
        code = agentctl_main(
            ["--socket", str(socket_path), "consecutive-errors", "reset"]
        )
        assert code == 0
        assert error_store.read().count == 0
    finally:
        server.stop()
