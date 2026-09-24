"""Behaviour tests for the durable operator-control command record (issue #80).

The live agent alone owns a SQLite control store under ``$DATA_DIR/state``.
``stop`` is the only mutating command in this slice; the record and its
ordering rules already carry unique request IDs and durable sequences so
later commands (resume, stop-after, next-issue, handoff) can build on them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from simple_coding_agent.control import (
    CommandAcknowledgement,
    ControlStore,
    ControlStoreError,
    IntakeState,
    PayloadMismatchError,
    RecoveryHold,
    RequestIdError,
    ResumeBlockedError,
)


def test_idle_stop_completes_immediately_and_stops_intake(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)

    assert store.intake_state() is IntakeState.RUNNING
    record = store.submit_stop("req-1", has_active_attempt=False)

    assert record.request_id == "req-1"
    assert record.sequence == 1
    assert record.acknowledgement is CommandAcknowledgement.COMPLETED
    assert store.intake_state() is IntakeState.STOPPED


def test_active_stop_is_accepted_and_waits_for_the_attempt(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)

    record = store.submit_stop("req-1", has_active_attempt=True)

    assert record.acknowledgement is CommandAcknowledgement.ACCEPTED
    assert store.intake_state() is IntakeState.STOPPING
    assert store.pending_command_id() == "req-1"


def test_completing_the_pending_stop_enters_stopped(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop("req-1", has_active_attempt=True)

    completed = store.complete_pending_stop()

    assert [record.request_id for record in completed] == ["req-1"]
    assert completed[0].acknowledgement is CommandAcknowledgement.COMPLETED
    assert store.intake_state() is IntakeState.STOPPED
    assert store.pending_command_id() is None


def test_repeated_stop_in_stopped_completes_idempotently(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop("req-1", has_active_attempt=False)

    second = store.submit_stop("req-2", has_active_attempt=False)

    assert second.acknowledgement is CommandAcknowledgement.COMPLETED
    assert store.intake_state() is IntakeState.STOPPED


def test_identical_retry_returns_the_same_sequence_without_double_apply(
    tmp_path: Path,
) -> None:
    store = ControlStore(tmp_path)
    first = store.submit_stop("req-1", has_active_attempt=True)

    retry = store.submit_stop("req-1", has_active_attempt=True)

    assert retry.sequence == first.sequence
    assert retry.acknowledgement is first.acknowledgement
    assert store.recent_commands() == [first]


def test_identical_retry_after_restart_returns_the_existing_acknowledgement(
    tmp_path: Path,
) -> None:
    store = ControlStore(tmp_path)
    first = store.submit_stop("req-1", has_active_attempt=True)

    restarted = ControlStore(tmp_path)
    retry = restarted.submit_stop("req-1", has_active_attempt=False)

    assert retry.sequence == first.sequence
    # The retry must not reinterpret the command against the new liveness:
    # the stored acknowledgement is returned unchanged.
    assert retry.acknowledgement is CommandAcknowledgement.ACCEPTED
    assert restarted.intake_state() is IntakeState.STOPPING


def test_reusing_a_request_id_with_a_different_payload_is_rejected(
    tmp_path: Path,
) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop("req-1", has_active_attempt=False)

    with pytest.raises(PayloadMismatchError):
        store.submit_command("resume", "req-1", {})

    stored = store.get_command("req-1")
    assert stored is not None
    assert stored.kind == "stop"
    assert stored.acknowledgement is CommandAcknowledgement.COMPLETED
    assert store.intake_state() is IntakeState.STOPPED


def test_second_stop_while_stopping_waits_for_the_same_boundary(
    tmp_path: Path,
) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop("req-1", has_active_attempt=True)

    second = store.submit_stop("req-2", has_active_attempt=False)

    # Nothing is `completed` before the effect is durably reached: both
    # stops stay accepted against the one active attempt.
    assert second.acknowledgement is CommandAcknowledgement.ACCEPTED
    assert store.intake_state() is IntakeState.STOPPING
    assert store.pending_command_id() == "req-1"

    completed = store.complete_pending_stop()

    assert [record.request_id for record in completed] == ["req-1", "req-2"]
    assert all(
        record.acknowledgement is CommandAcknowledgement.COMPLETED for record in completed
    )
    assert store.intake_state() is IntakeState.STOPPED
    assert store.pending_command_id() is None


def test_sequences_are_monotonic_across_commands(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    first = store.submit_stop("req-1", has_active_attempt=False)
    second = store.submit_stop("req-2", has_active_attempt=False)

    assert (first.sequence, second.sequence) == (1, 2)


def test_blank_request_id_is_rejected_without_state_change(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)

    with pytest.raises(RequestIdError):
        store.submit_stop("  ", has_active_attempt=False)

    assert store.intake_state() is IntakeState.RUNNING
    assert store.recent_commands() == []


def test_unknown_request_id_lookup_returns_none(tmp_path: Path) -> None:
    assert ControlStore(tmp_path).get_command("req-missing") is None


def test_commands_and_intake_survive_process_restart(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop("req-1", has_active_attempt=True)

    restarted = ControlStore(tmp_path)

    assert restarted.intake_state() is IntakeState.STOPPING
    assert restarted.pending_command_id() == "req-1"
    record = restarted.get_command("req-1")
    assert record is not None
    assert record.sequence == 1
    assert record.acknowledgement is CommandAcknowledgement.ACCEPTED


def test_storage_failure_never_reports_an_accepted_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    store = ControlStore(tmp_path)
    store.submit_stop("req-1", has_active_attempt=False)

    def failing_connect(*args: object, **kwargs: object):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(sqlite3, "connect", failing_connect)
    with pytest.raises(ControlStoreError):
        store.submit_stop("req-2", has_active_attempt=False)

    monkeypatch.undo()
    assert store.get_command("req-2") is None
    assert store.intake_state() is IntakeState.STOPPED
    assert [record.request_id for record in store.recent_commands()] == ["req-1"]


# ---------------------------------------------------------------------------
# resume (issue #81)
# ---------------------------------------------------------------------------


def test_resume_from_stopped_enters_running(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop("req-stop", has_active_attempt=False)
    assert store.intake_state() is IntakeState.STOPPED

    record = store.submit_resume("req-resume", has_active_attempt=False)

    assert record.request_id == "req-resume"
    assert record.kind == "resume"
    assert record.acknowledgement is CommandAcknowledgement.COMPLETED
    assert store.intake_state() is IntakeState.RUNNING
    assert store.pending_command_id() is None


def test_resume_while_running_without_plan_completes_idempotently(
    tmp_path: Path,
) -> None:
    store = ControlStore(tmp_path)

    first = store.submit_resume("req-r1", has_active_attempt=False)
    second = store.submit_resume("req-r2", has_active_attempt=False)

    assert first.acknowledgement is CommandAcknowledgement.COMPLETED
    assert second.acknowledgement is CommandAcknowledgement.COMPLETED
    assert "already running" in second.detail
    assert store.intake_state() is IntakeState.RUNNING
    assert store.pending_command_id() is None
    assert (first.sequence, second.sequence) == (1, 2)


def test_resume_supersedes_pending_stop_and_names_the_replacement(
    tmp_path: Path,
) -> None:
    store = ControlStore(tmp_path)
    stop = store.submit_stop("req-stop", has_active_attempt=True)
    assert stop.acknowledgement is CommandAcknowledgement.ACCEPTED

    resume = store.submit_resume("req-resume", has_active_attempt=True)

    assert resume.acknowledgement is CommandAcknowledgement.COMPLETED
    assert "req-stop" in resume.detail
    assert store.intake_state() is IntakeState.RUNNING
    assert store.pending_command_id() is None

    replaced = store.get_command("req-stop")
    assert replaced is not None
    assert replaced.acknowledgement is CommandAcknowledgement.SUPERSEDED
    assert "req-resume" in replaced.detail

    # The superseded stop no longer completes at the attempt boundary.
    assert store.complete_pending_stop() == []
    assert store.intake_state() is IntakeState.RUNNING


def test_resume_identical_retry_returns_the_same_record_across_restart(
    tmp_path: Path,
) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop("req-stop", has_active_attempt=False)
    first = store.submit_resume("req-resume", has_active_attempt=False)

    restarted = ControlStore(tmp_path)
    retry = restarted.submit_resume("req-resume", has_active_attempt=True)

    assert retry.sequence == first.sequence
    assert retry.acknowledgement is first.acknowledgement
    assert restarted.intake_state() is IntakeState.RUNNING


def test_resume_request_id_reused_with_a_different_payload_is_rejected(
    tmp_path: Path,
) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop("req-1", has_active_attempt=False)

    with pytest.raises(PayloadMismatchError):
        store.submit_resume("req-1", has_active_attempt=False)

    stored = store.get_command("req-1")
    assert stored is not None
    assert stored.kind == "stop"
    assert store.intake_state() is IntakeState.STOPPED


def test_resume_rejected_while_finalization_is_unresolved(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop("req-stop", has_active_attempt=False)
    hold = RecoveryHold(
        issue_number=24,
        attempt_id="2026-09-24T00:00:00Z",
        branch="agent/issue-24",
        phase="publishing",
        reason="publication is unconfirmed; retained work is preserved for recovery",
    )

    with pytest.raises(ResumeBlockedError, match=r"issue #24"):
        store.submit_resume("req-resume", has_active_attempt=True, recovery_hold=hold)

    with pytest.raises(ResumeBlockedError, match="unconfirmed"):
        store.submit_resume("req-other", has_active_attempt=True, recovery_hold=hold)

    # The rejection names the stable attempt identity, not just the issue.
    with pytest.raises(ResumeBlockedError, match="2026-09-24T00:00:00Z"):
        store.submit_resume("req-attempt", has_active_attempt=True, recovery_hold=hold)

    assert store.intake_state() is IntakeState.STOPPED
    assert store.pending_command_id() is None
    assert store.get_command("req-resume") is None
    assert store.get_command("req-other") is None
    assert store.get_command("req-attempt") is None


def test_resume_rejected_while_startup_reconciliation_runs(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    hold = RecoveryHold(
        issue_number=None,
        attempt_id=None,
        branch=None,
        phase=None,
        reason="startup reconciliation is still running",
    )

    with pytest.raises(ResumeBlockedError, match="reconciliation"):
        store.submit_resume("req-resume", has_active_attempt=False, recovery_hold=hold)

    assert store.intake_state() is IntakeState.RUNNING
    assert store.get_command("req-resume") is None
