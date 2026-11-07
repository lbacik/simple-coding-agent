"""Contract ``status`` projection over the control store and telemetry reporter.

A dashboard renders a running instance from one ``agentctl --contract 1
status`` call. Every test drives the shared fakes (a real lifecycle, control
store, attempt checkpoint and telemetry reporter) through the real CLI entry
point in-process and asserts only on the exit code and the parsed stdout
envelope.
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Iterator
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from simple_coding_agent import agentctl
from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.control import ControlStore
from simple_coding_agent.control_server import ControlServer, send_request
from simple_coding_agent.telemetry import TelemetryReporter
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace

REPOSITORY = "owner/repo"

STATUS_RESULT_KEYS = {
    "current_attempt",
    "generated_at",
    "identity",
    "instance",
    "intake",
    "last_finished",
    "recent_commands",
    "recent_events",
    "truncated",
}

COMMAND_KEYS = {
    "acknowledgement",
    "arguments",
    "created_at",
    "detail",
    "kind",
    "repository",
    "request_id",
    "sequence",
    "updated_at",
}

FIGURE_BASES = {
    "measured",
    "estimated",
    "configured",
    "assumed",
    "unavailable",
    "not_applicable",
}


def scripted_clock(start: datetime = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)):
    """A clock that advances one second per call (deterministic timestamps)."""

    state = {"now": start}

    def tick() -> datetime:
        state["now"] = state["now"] + timedelta(seconds=1)
        return state["now"]

    return tick


def make_reporter(**overrides: object) -> TelemetryReporter:
    values: dict[str, object] = {
        "repository": REPOSITORY,
        "agent_version": "0.0-test",
        "sdk_version": "0.2.163-test",
        "cli_version": "2.1.286-test",
        "process_started_at": "2026-01-01T00:00:00Z",
        "hard_ceiling": 4_000_000,
        "turns_max": 60,
        "usd_ceiling": 20,
        "timeout_seconds": 3600,
        "continuation_max": 0,
        "clock": scripted_clock(),
    }
    values.update(overrides)
    return TelemetryReporter(**values)  # type: ignore[arg-type]


def claim_seed(**overrides: object) -> dict:
    seed: dict[str, object] = {
        "attempt_id": "2026-10-14T12:00:00Z",
        "issue_number": 42,
        "issue_title": "Add CSV export",
        "continuation_index": 0,
    }
    seed.update(overrides)
    return seed


def live_view(**overrides: object) -> dict:
    """One executor-shaped raw live view (pre-basis readings)."""

    view: dict[str, object] = {
        "budget_tokens": 640000.0,
        "measured_tokens": 512000.0,
        "estimated_tokens": 128000.0,
        "hard_ceiling_reached": False,
        "soft_threshold_value": 3200000,
        "soft_threshold_rule": "percentage",
        "soft_threshold_percentage": 0.2,
        "soft_threshold_reserve_turns": 6,
        "soft_threshold_crossed": False,
        "soft_threshold_crossed_at_tokens": None,
        "soft_threshold_crossed_at": None,
        "main_context_basis": "measured",
        "main_context_tokens": 400000,
        "main_peak_context_tokens": 450000.0,
        "main_peak_context_basis": "measured",
        "tokens_by_category": {
            "input_tokens": 120000,
            "cache_read_input_tokens": 350000,
            "cache_creation_input_tokens": 0,
            "output_tokens": 42000,
        },
        "turns": 18,
        "cache_hit_rate": 0.86,
        "cache_counted": 22,
        "elapsed_seconds": 2700,
        "context_window": 200000,
        "context_window_basis": "sdk_assumed",
        "auto_compact_threshold": 160000,
        "auto_compact_threshold_basis": "configured",
        "compaction_count": 0,
        "compaction_last": None,
        "subagents": [],
    }
    view.update(overrides)
    return view


def last_finished_summary(**overrides: object) -> dict:
    summary: dict[str, object] = {
        "attempt_id": "2026-10-14T11:00:00Z",
        "issue_number": 41,
        "issue_title": "Fix pagination",
        "outcome": "complete",
        "finished_at": "2026-10-14T11:44:52Z",
        "budget_used": 412530,
        "turns": 118,
        "pull_request": {
            "number": 77,
            "url": "https://github.com/owner/repo/pull/77",
        },
    }
    summary.update(overrides)
    return summary


def make_lifecycle(tmp_path: Path, *, reporter: TelemetryReporter | None = None):
    """Wire the real lifecycle against the shared fakes with control enabled."""

    from simple_coding_agent.lifecycle import AgentLifecycle

    tracker = FakeTracker()
    workspace = InMemoryWorkspace(tmp_path / "repo")
    store = ControlStore(tmp_path)
    state = AttemptStateStore(tmp_path)
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=lambda claim, profile, prepared: (_ for _ in ()).throw(
            AssertionError("attempt runner must not run during status")
        ),
        control_store=store,
        repository=REPOSITORY,
        telemetry=reporter if reporter is not None else make_reporter(),
        sleeper=lambda seconds: None,
    )
    return lifecycle, store, state


@pytest.fixture
def status_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[ControlServer, object, ControlStore, AttemptStateStore]]:
    """A fresh control socket serving the real lifecycle; yields server parts."""

    lifecycle, store, state = make_lifecycle(tmp_path)
    socket_path = tmp_path / "state" / "contract-status.sock"
    server = ControlServer(socket_path, lifecycle)  # type: ignore[arg-type]
    server.start()
    monkeypatch.setenv("AGENTCTL_SOCKET", str(socket_path))
    try:
        yield server, lifecycle, store, state
    finally:
        server.stop()


def _run(argv: list[str]) -> tuple[int, dict, str, str]:
    """Run the real CLI entry point; return (exit code, envelope, stdout, stderr)."""

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = agentctl.main(argv)
    assert isinstance(code, int)
    stdout = out.getvalue()
    assert stdout.endswith("\n"), "stdout must end with exactly one newline"
    assert stdout.count("\n") == 1, "stdout must hold exactly one envelope"
    return code, json.loads(stdout), stdout, err.getvalue()


def _status() -> tuple[int, dict]:
    code, envelope, _, err = _run(["--contract", "1", "status"])
    assert err == ""
    return code, envelope


def _result(envelope: dict) -> dict:
    assert envelope["contract"] == "1.0"
    assert envelope["ok"] is True
    return envelope["result"]


def _assert_figure(figure: object, *, unit: str) -> None:
    assert isinstance(figure, dict), figure
    assert set(figure) >= {"value", "unit", "basis"}, figure
    assert figure["unit"] == unit, figure
    assert figure["basis"] in FIGURE_BASES, figure
    if figure["basis"] in ("unavailable", "not_applicable"):
        assert figure["value"] is None, figure
    else:
        assert figure["value"] is not None, figure


# ---------------------------------------------------------------------------
# Instance and intake states
# ---------------------------------------------------------------------------


def test_idle_instance_with_open_intake_and_no_attempt(status_server) -> None:
    _, _, _, _ = status_server
    code, envelope = _status()
    assert code == 0
    result = _result(envelope)
    assert set(result) == STATUS_RESULT_KEYS
    assert result["instance"]["state"] == "idle"
    assert result["intake"]["state"] == "open"
    assert result["intake"]["pending_stop"] is None
    assert result["intake"]["stop_plan"] is None
    assert result["intake"]["priority"] is None
    assert result["current_attempt"] is None
    assert result["last_finished"] is None
    assert result["recent_commands"] == []
    assert result["recent_events"] == []
    assert result["truncated"] is False


def test_active_instance_while_an_attempt_runs(status_server) -> None:
    _, lifecycle, _, state = status_server
    state.start(issue_number=42, branch="agent/issue-42", issue_title="Add CSV export")
    state.transition(AttemptPhase.SETUP)
    state.transition(AttemptPhase.MODEL_RUNNING)
    lifecycle._telemetry.begin_attempt(claim_seed())  # noqa: SLF001

    code, envelope = _status()
    assert code == 0
    result = _result(envelope)
    assert result["instance"]["state"] == "active"
    assert result["intake"]["state"] == "open"
    current = result["current_attempt"]
    assert current["attempt_id"] == "2026-10-14T12:00:00Z"
    assert current["issue"] == {"number": 42, "title": "Add CSV export"}
    assert current["native_phase"] == "model_running"
    assert current["started_at"] == "2026-10-14T12:00:00Z"
    assert isinstance(current["last_activity_at"], str)
    assert isinstance(current["observed_at"], str)


def test_draining_instance_while_a_stop_waits_on_the_attempt(status_server) -> None:
    _, lifecycle, store, state = status_server
    checkpoint = state.start(
        issue_number=42, branch="agent/issue-42", issue_title="Add CSV export"
    )
    lifecycle._telemetry.begin_attempt(claim_seed())  # noqa: SLF001
    record = store.submit_stop("req-drain-1", has_active_attempt=True)
    assert checkpoint is not None

    code, envelope = _status()
    assert code == 0
    result = _result(envelope)
    assert result["instance"]["state"] == "draining"
    assert result["intake"]["state"] == "closed"
    assert result["intake"]["pending_stop"] == {
        "request_id": "req-drain-1",
        "sequence": record.sequence,
    }
    assert result["intake"]["stop_plan"] is None


def test_paused_instance_after_intake_stops_with_no_attempt(status_server) -> None:
    _, _, store, _ = status_server
    store.submit_stop("req-stop-1", has_active_attempt=False)

    code, envelope = _status()
    assert code == 0
    result = _result(envelope)
    assert result["instance"]["state"] == "paused"
    assert result["intake"]["state"] == "closed"
    assert result["current_attempt"] is None


def test_intake_stop_plan_and_priority_match_the_control_store(status_server) -> None:
    _, lifecycle, store, state = status_server
    checkpoint = state.start(
        issue_number=42, branch="agent/issue-42", issue_title="Add CSV export"
    )
    lifecycle._telemetry.begin_attempt(claim_seed())  # noqa: SLF001
    plan = store.submit_stop_after(
        "req-plan-1", 3, has_active_attempt=True, active_attempt_id=checkpoint.started_at
    )
    priority = store.submit_next_issue("req-prio-1", 43)

    code, envelope = _status()
    assert code == 0
    result = _result(envelope)
    assert result["intake"]["stop_plan"] == {
        "request_id": "req-plan-1",
        "remaining": 3,
        "counted": 0,
    }
    assert result["intake"]["priority"] == {
        "request_id": "req-prio-1",
        "issue": 43,
    }
    assert plan.sequence >= 1
    assert priority.sequence >= 1


# ---------------------------------------------------------------------------
# Budget and figures
# ---------------------------------------------------------------------------


def test_required_budget_figures_are_always_present_with_a_basis(
    status_server,
) -> None:
    _, lifecycle, _, state = status_server
    state.start(issue_number=42, branch="agent/issue-42", issue_title="Add CSV export")
    lifecycle._telemetry.begin_attempt(claim_seed())  # noqa: SLF001

    code, envelope = _status()
    assert code == 0
    budget = _result(envelope)["current_attempt"]["budget"]
    # No live view published yet: unknown, never a measured zero.
    assert budget["raw_tokens"] == {
        "value": None,
        "unit": "tokens",
        "basis": "unavailable",
        "reason": "not yet published",
    }
    assert budget["effective_tokens"]["unit"] == "tokens"
    assert budget["effective_tokens"]["basis"] == "unavailable"
    assert budget["effective_tokens"]["value"] is None
    assert budget["enforced_tokens"] == "raw"


def test_live_budget_and_limits_list_only_enforced_ceilings(status_server) -> None:
    _, lifecycle, _, state = status_server
    state.start(issue_number=42, branch="agent/issue-42", issue_title="Add CSV export")
    reporter = lifecycle._telemetry  # noqa: SLF001
    reporter.begin_attempt(claim_seed())
    reporter.publish_attempt_view(live_view())

    code, envelope = _status()
    assert code == 0
    budget = _result(envelope)["current_attempt"]["budget"]
    assert budget["raw_tokens"] == {
        "value": 640000,
        "unit": "tokens",
        "basis": "estimated",
    }
    assert budget["effective_tokens"] == {
        "value": 640000,
        "unit": "tokens",
        "basis": "estimated",
    }
    assert budget["enforced_tokens"] == "raw"
    assert [entry["name"] for entry in budget["limits"]] == [
        "raw_tokens",
        "cost_usd",
        "turns",
    ]
    by_name = {entry["name"]: entry for entry in budget["limits"]}
    assert by_name["raw_tokens"]["used"] == budget["raw_tokens"]
    assert by_name["raw_tokens"]["limit"] == {
        "value": 4000000,
        "unit": "tokens",
        "basis": "configured",
    }
    assert by_name["cost_usd"]["limit"] == {
        "value": 20,
        "unit": "usd",
        "basis": "configured",
    }
    assert by_name["turns"]["used"] == {
        "value": 18,
        "unit": "count",
        "basis": "measured",
    }
    assert by_name["turns"]["limit"] == {
        "value": 60,
        "unit": "count",
        "basis": "configured",
    }
    # The engine does not track live cost: unavailable, never zero.
    _assert_figure(by_name["cost_usd"]["used"], unit="usd")
    assert by_name["cost_usd"]["used"]["basis"] == "unavailable"


def test_sdk_assumed_basis_is_reported_as_assumed(status_server) -> None:
    _, lifecycle, _, state = status_server
    state.start(issue_number=42, branch="agent/issue-42", issue_title="Add CSV export")
    reporter = lifecycle._telemetry  # noqa: SLF001
    reporter.begin_attempt(claim_seed())
    reporter.publish_attempt_view(live_view())

    code, envelope = _status()
    assert code == 0
    figures = _result(envelope)["current_attempt"]["figures"]
    assert figures["context_window"] == {
        "value": 200000,
        "unit": "tokens",
        "basis": "assumed",
    }
    _assert_figure(figures["cache_hit_rate"], unit="ratio")
    assert figures["cache_hit_rate"]["value"] == 0.86
    _assert_figure(figures["context_used"], unit="tokens")
    _assert_figure(figures["context_peak"], unit="tokens")
    _assert_figure(figures["turns"], unit="count")
    assert figures["turns"]["value"] == 18
    _assert_figure(figures["elapsed_seconds"], unit="seconds")
    for figure in figures.values():
        _assert_figure(figure, unit=figure["unit"])


def test_every_figure_carries_a_unit_and_a_known_basis(status_server) -> None:
    _, lifecycle, _, state = status_server
    state.start(issue_number=42, branch="agent/issue-42", issue_title="Add CSV export")
    reporter = lifecycle._telemetry  # noqa: SLF001
    reporter.begin_attempt(claim_seed())
    reporter.publish_attempt_view(live_view())

    code, envelope = _status()
    assert code == 0
    current = _result(envelope)["current_attempt"]
    _assert_figure(current["budget"]["raw_tokens"], unit="tokens")
    _assert_figure(current["budget"]["effective_tokens"], unit="tokens")
    for entry in current["budget"]["limits"]:
        _assert_figure(entry["used"], unit=entry["used"]["unit"])
        _assert_figure(entry["limit"], unit=entry["limit"]["unit"])
    for figure in current["figures"].values():
        assert isinstance(figure, dict)
        _assert_figure(figure, unit=figure["unit"])


# ---------------------------------------------------------------------------
# Last finished attempt
# ---------------------------------------------------------------------------


def test_last_finished_attempt_maps_the_native_summary(status_server) -> None:
    _, lifecycle, _, _ = status_server
    lifecycle._telemetry.finish_attempt(last_finished_summary())  # noqa: SLF001

    code, envelope = _status()
    assert code == 0
    finished = _result(envelope)["last_finished"]
    assert finished["attempt_id"] == "2026-10-14T11:00:00Z"
    assert finished["issue"] == {"number": 41, "title": "Fix pagination"}
    assert finished["started_at"] == "2026-10-14T11:00:00Z"
    assert finished["finished_at"] == "2026-10-14T11:44:52Z"
    assert finished["lifecycle_state"] == "finished"
    assert finished["outcome"] == {
        "native": "complete",
        "reason": None,
        "label": "Complete",
    }
    assert finished["delivery"] == {
        "pull_request": {
            "number": 77,
            "url": "https://github.com/owner/repo/pull/77",
        }
    }
    assert finished["attention_required"] is False
    assert finished["sealed"] is True
    assert finished["sealed_at"] == "2026-10-14T11:44:52Z"


# ---------------------------------------------------------------------------
# Recent commands
# ---------------------------------------------------------------------------


def test_recent_commands_use_contract_acknowledgements_and_arguments(
    status_server,
) -> None:
    _, _, store, state = status_server
    checkpoint = state.start(
        issue_number=42, branch="agent/issue-42", issue_title="Add CSV export"
    )
    store.submit_stop_after(
        "req-plan-1", 2, has_active_attempt=True, active_attempt_id=checkpoint.started_at
    )
    store.submit_next_issue("req-prio-1", 43)
    store.submit_errors_reset("req-reset-1")
    store.submit_recovery_retry("req-retry-1", checkpoint.started_at)
    store.submit_recovery_release("req-release-1", checkpoint.started_at, "/tmp/kept")
    store.fail_next_issue("issue is no longer eligible")

    code, envelope = _status()
    assert code == 0
    commands = _result(envelope)["recent_commands"]
    assert len(commands) == 5
    assert [command["request_id"] for command in commands] == [
        "req-plan-1",
        "req-prio-1",
        "req-reset-1",
        "req-retry-1",
        "req-release-1",
    ]
    for command in commands:
        assert set(command) == COMMAND_KEYS, command
        assert command["repository"] == REPOSITORY
        assert "\n" not in command["detail"]
    by_id = {command["request_id"]: command for command in commands}
    assert by_id["req-plan-1"]["kind"] == "stop_after"
    assert by_id["req-plan-1"]["arguments"] == {"after": 2}
    assert by_id["req-plan-1"]["acknowledgement"] == "accepted"
    assert by_id["req-prio-1"]["kind"] == "next_issue"
    assert by_id["req-prio-1"]["arguments"] == {"issue": 43}
    # The failed priority reached its terminal acknowledgement, renamed.
    assert by_id["req-prio-1"]["acknowledgement"] == "not_fulfilled"
    assert by_id["req-reset-1"]["arguments"] == {}
    assert by_id["req-reset-1"]["acknowledgement"] == "completed"
    assert by_id["req-retry-1"]["arguments"] == {"attempt_id": checkpoint.started_at}
    assert by_id["req-release-1"]["arguments"] == {
        "attempt_id": checkpoint.started_at,
        "saved_at": "/tmp/kept",
    }


# ---------------------------------------------------------------------------
# Freshness and size
# ---------------------------------------------------------------------------


def test_sections_carry_observed_at_and_no_sequence(status_server) -> None:
    _, lifecycle, _, state = status_server
    state.start(issue_number=42, branch="agent/issue-42", issue_title="Add CSV export")
    lifecycle._telemetry.begin_attempt(claim_seed())  # noqa: SLF001

    code, envelope = _status()
    assert code == 0
    result = _result(envelope)
    assert isinstance(result["generated_at"], str)
    assert "sequence" not in result
    assert isinstance(result["instance"]["observed_at"], str)
    assert isinstance(result["intake"]["observed_at"], str)
    assert isinstance(result["current_attempt"]["observed_at"], str)


def test_oversize_snapshot_drops_figures_then_commands_and_marks_truncated(
    status_server,
) -> None:
    _, lifecycle, store, state = status_server
    state.start(issue_number=42, branch="agent/issue-42", issue_title="T" * 300_000)
    lifecycle._telemetry.begin_attempt(claim_seed(issue_title="T" * 300_000))  # noqa: SLF001
    store.submit_stop_after("req-plan-1", 2, has_active_attempt=False)
    store.submit_next_issue("req-prio-1", 43)

    code, envelope = _status()
    assert code == 0
    result = _result(envelope)
    assert result["truncated"] is True
    # Optional figures go before the oldest commands, newest last.
    assert result["current_attempt"]["figures"] == {}
    assert result["recent_commands"] == []
    # Required budget figures are always present, even when truncated.
    assert result["current_attempt"]["budget"]["raw_tokens"]["unit"] == "tokens"


def test_contract_status_answers_while_a_mutation_holds_the_control_lock(
    status_server,
) -> None:
    server, lifecycle, _, state = status_server
    state.start(issue_number=42, branch="agent/issue-42", issue_title="Add CSV export")
    lifecycle._telemetry.begin_attempt(claim_seed())  # noqa: SLF001
    # Prime the prepared snapshot so the bounded fallback has something to serve.
    code, _ = _status()
    assert code == 0

    lock = lifecycle._control_lock  # noqa: SLF001 - hold it as the intake claim does
    lock.acquire()
    try:
        start = time.monotonic()
        code, envelope = _status()
        elapsed = time.monotonic() - start
        assert code == 0
        assert elapsed < 3
        assert _result(envelope)["instance"]["state"] == "active"

        start = time.monotonic()
        identity_code, identity_envelope, _, _ = _run(["--contract", "1", "identity"])
        identity_elapsed = time.monotonic() - start
        assert identity_code == 0
        assert identity_envelope["ok"] is True
        assert identity_elapsed < 3
    finally:
        lock.release()


def test_contract_status_is_served_from_prepared_state_without_live_lookups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No tracker, publisher or workspace call runs on the status path."""

    lifecycle, _, state = make_lifecycle(tmp_path)

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("status must serve prepared state only")

    monkeypatch.setattr(lifecycle, "_recovery_snapshot_network_free", _forbidden)
    state.start(issue_number=42, branch="agent/issue-42", issue_title="Add CSV export")

    socket_path = tmp_path / "state" / "prepared.sock"
    server = ControlServer(socket_path, lifecycle)  # type: ignore[arg-type]
    server.start()
    monkeypatch.setenv("AGENTCTL_SOCKET", str(socket_path))
    try:
        reply = send_request(socket_path, {"op": "contract_status"})
    finally:
        server.stop()
    assert reply["ok"] is True
    assert set(reply["contract_status"]) == STATUS_RESULT_KEYS
