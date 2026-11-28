"""Operator handoff module tests: lifecycle, deadlines, verdict.

The durable ControlStore in a tmp dir is the persistence; a mutable current
time is the clock port. These tests pin the request lifecycle
(request → delivering → begun → complete / fail / expire), the single
deadline computation, and the verdict handed to the executor and lifecycle.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from simple_coding_agent.control import ControlStore
from simple_coding_agent.operator_handoff import (
    HANDOFF_ABSOLUTE_DEADLINE_SECONDS,
    HANDOFF_MODEL_DEADLINE_SECONDS,
    HANDOFF_PUBLICATION_ALLOWANCE_SECONDS,
    ControlOperatorHandoff,
    classify_verdict,
    format_control_timestamp,
    handoff_completion_cause,
    handoff_deadlines,
    handoff_owned_by_attempt,
    operator_due_action,
    operator_handoff_provider,
    operator_handoff_reporter,
    operator_stream_wait_seconds,
    parse_operator_accepted_at,
    publish_budget_for_snapshot,
    wire_operator_handoff,
)


ACCEPTED_AT = datetime(2026, 9, 22, 0, 0, 0, tzinfo=UTC)


def make_store(tmp_path: Path, now: list[datetime]) -> ControlStore:
    return ControlStore(tmp_path, clock=lambda: now[0])


def test_parse_round_trips_the_control_format() -> None:
    assert parse_operator_accepted_at("2026-09-22T00:00:00Z") == ACCEPTED_AT
    assert parse_operator_accepted_at("not a time") is None
    assert parse_operator_accepted_at(None) is None
    assert parse_operator_accepted_at("2026-09-22T00:00:00Z").tzinfo is not None


def test_deadlines_derive_from_acceptance() -> None:
    deadlines = handoff_deadlines("2026-09-22T00:00:00Z")

    assert deadlines == {
        "model_deadline_at": "2026-09-22T00:04:00Z",
        "publication_deadline_at": "2026-09-22T00:06:00Z",
    }
    assert handoff_deadlines("bogus") == {
        "model_deadline_at": None,
        "publication_deadline_at": None,
    }


def test_request_lifecycle_through_the_adapter(tmp_path: Path) -> None:
    now = [ACCEPTED_AT]
    store = make_store(tmp_path, now)
    events: list[tuple] = []
    control = ControlOperatorHandoff(
        store, lambda event, detail="", level="INFO", **_: events.append(event)
    )

    store.submit_handoff("req-1", has_active_attempt=True)
    snapshot = control.snapshot()
    assert snapshot is not None and snapshot["acknowledgement"] == "accepted"

    assert control.mark_delivering("req-1") is True
    assert control.mark_begun("req-1") is True
    assert control.snapshot()["begun"] is True

    control.complete("req-1", "handed off")
    assert control.snapshot()["acknowledgement"] == "completed"


def test_provider_serves_only_the_accepted_request(tmp_path: Path) -> None:
    now = [ACCEPTED_AT]
    store = make_store(tmp_path, now)
    control = ControlOperatorHandoff(store, lambda *a, **k: None)
    provider = operator_handoff_provider(control)

    assert provider() is None

    store.submit_handoff("req-1", has_active_attempt=True)
    served = provider()
    assert served is not None and served.request_id == "req-1"
    assert served.accepted_at == ACCEPTED_AT

    control.complete("req-1", "handed off")
    assert provider() is None


def test_reporter_marks_delivering_and_begun(tmp_path: Path) -> None:
    now = [ACCEPTED_AT]
    store = make_store(tmp_path, now)
    control = ControlOperatorHandoff(store, lambda *a, **k: None)
    reporter = operator_handoff_reporter(control, lambda *a, **k: None)

    store.submit_handoff("req-1", has_active_attempt=True)
    reporter("delivered", "req-1")
    reporter("begun", "req-1")

    assert store.handoff_snapshot()["begun"] is True


def test_wire_leaves_executors_without_seams_alone(tmp_path: Path) -> None:
    now = [ACCEPTED_AT]
    store = make_store(tmp_path, now)
    control = ControlOperatorHandoff(store, lambda *a, **k: None)

    wire_operator_handoff(object(), control)  # must not raise


def test_publish_budget_counts_down_from_acceptance() -> None:
    snapshot = {"acknowledgement": "accepted", "accepted_at": "2026-09-22T00:00:00Z"}

    assert (
        publish_budget_for_snapshot(snapshot, ACCEPTED_AT)
        == HANDOFF_PUBLICATION_ALLOWANCE_SECONDS
    )
    aged = ACCEPTED_AT + timedelta(seconds=300)
    assert publish_budget_for_snapshot(snapshot, aged) == 60.0
    expired = ACCEPTED_AT + timedelta(seconds=9999)
    assert publish_budget_for_snapshot(snapshot, expired) == 0.0
    assert publish_budget_for_snapshot({"acknowledgement": "completed"}, ACCEPTED_AT) is None


def test_due_action_falls_back_then_expires() -> None:
    assert (
        operator_due_action(
            accepted_at=ACCEPTED_AT,
            now=ACCEPTED_AT + timedelta(seconds=61),
            context_delivered=False,
            begun=False,
            fallback_used=False,
            deadline_expired=False,
        )
        == "fallback"
    )
    assert (
        operator_due_action(
            accepted_at=ACCEPTED_AT,
            now=ACCEPTED_AT
            + timedelta(seconds=HANDOFF_MODEL_DEADLINE_SECONDS + 1),
            context_delivered=False,
            begun=False,
            fallback_used=True,
            deadline_expired=False,
        )
        == "expire"
    )
    assert (
        operator_due_action(
            accepted_at=ACCEPTED_AT,
            now=ACCEPTED_AT + timedelta(seconds=61),
            context_delivered=True,
            begun=False,
            fallback_used=False,
            deadline_expired=False,
        )
        is None
    )


def test_stream_wait_ends_at_the_next_boundary() -> None:
    assert operator_stream_wait_seconds(
        accepted_at=None,
        now=ACCEPTED_AT,
        provider_present=False,
        context_delivered=False,
        begun=False,
        fallback_used=False,
    ) is None
    wait = operator_stream_wait_seconds(
        accepted_at=ACCEPTED_AT,
        now=ACCEPTED_AT,
        provider_present=True,
        context_delivered=False,
        begun=False,
        fallback_used=False,
    )
    assert wait == 60.0


def test_completion_cause_accepts_a_clean_handoff() -> None:
    snapshot = {"accepted_at": "2026-09-22T00:00:00Z", "begun": True}

    assert (
        handoff_completion_cause(
            snapshot,
            outcome_value="handoff",
            handoff_outcome_value="handoff",
            note_valid=True,
            note_why="valid",
            begun=True,
            publication_started_at=ACCEPTED_AT,
            now=ACCEPTED_AT + timedelta(seconds=10),
        )
        is None
    )


def test_completion_cause_rejects_an_unbegun_handoff() -> None:
    snapshot = {"accepted_at": "2026-09-22T00:00:00Z", "begun": False}

    cause = handoff_completion_cause(
        snapshot,
        outcome_value="handoff",
        handoff_outcome_value="handoff",
        note_valid=True,
        note_why="valid",
        begun=False,
        publication_started_at=None,
        now=ACCEPTED_AT,
    )

    assert "never began" in cause


def test_completion_cause_rejects_a_blown_absolute_deadline() -> None:
    snapshot = {"accepted_at": "2026-09-22T00:00:00Z", "begun": True}

    cause = handoff_completion_cause(
        snapshot,
        outcome_value="handoff",
        handoff_outcome_value="handoff",
        note_valid=True,
        note_why="valid",
        begun=True,
        publication_started_at=None,
        now=ACCEPTED_AT
        + timedelta(seconds=HANDOFF_ABSOLUTE_DEADLINE_SECONDS + 1),
    )

    assert "360-second" in cause


def test_verdict_distinguishes_hold_from_foreign_failure() -> None:
    assert (
        classify_verdict("accepted", completion_cause=None, owned_by_attempt=True)
        == "owned"
    )
    assert (
        classify_verdict(
            "accepted", completion_cause="too late", owned_by_attempt=True
        )
        == "hold"
    )
    assert (
        classify_verdict(
            "not fulfilled", completion_cause="too late", owned_by_attempt=True
        )
        == "hold"
    )
    assert (
        classify_verdict(
            "not fulfilled", completion_cause="too late", owned_by_attempt=False
        )
        == "failed"
    )
    assert (
        classify_verdict("completed", completion_cause="too late", owned_by_attempt=True)
        == "owned"
    )
    assert (
        classify_verdict(None, completion_cause=None, owned_by_attempt=False) is None
    )


def test_hold_ownership_is_anchored_on_the_attempt_clause() -> None:
    assert (
        handoff_owned_by_attempt("... retained for recovery of attempt abc.", "abc")
        is True
    )
    assert handoff_owned_by_attempt("... of attempt xyz.", "abc") is False
    assert handoff_owned_by_attempt(None, "abc") is False


def test_format_matches_the_store_clock(tmp_path: Path) -> None:
    now = [ACCEPTED_AT]
    store = make_store(tmp_path, now)
    store.submit_handoff("req-1", has_active_attempt=True)

    assert store.handoff_snapshot()["accepted_at"] == format_control_timestamp(
        ACCEPTED_AT
    )
