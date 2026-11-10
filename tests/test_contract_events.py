"""Operator-contract Contract Events produced by the reference backend (#343).

Every backend's history reads the same: the structured event logger maps
each redacted native record onto one Contract Event (spec section 10).
Native events with a Contract Event counterpart project to it; everything
else is exposed only as ``native.<event>``. The process-local ring of the
newest 50 feeds ``status.recent_events`` (redacted, text capped at 300
characters); each Attempt's Contract Events land in ``events.jsonl`` in its
archive directory (manifest kind ``events``). Stdout keeps the native shape.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from simple_coding_agent import __main__ as entrypoint
from simple_coding_agent.attempt_history import show_attempt
from simple_coding_agent.contract_events import (
    CONTRACT_EVENTS_LIMIT,
    ContractEventsBuffer,
    command_accepted_detail,
    command_resolved_detail,
    halted_detail,
    outcome_recorded_detail,
    parse_detail_fields,
    process_started_detail,
    startup_check_detail,
    to_contract_event,
)
from simple_coding_agent.control import CommandAcknowledgement
from simple_coding_agent.lifecycle import LifecycleStatus
from simple_coding_agent.observability import AttemptArchive, JsonEventLogger
from tests.contract_schema import validate_contract_event
from tests.fakes import (
    FakeGitHub,
    FakeModelExecutor,
    build_lifecycle,
    lifecycle_issue as issue,
    repository_with_main,
    write_and_commit,
)


def _native(
    event: str,
    detail: str = "",
    *,
    level: str = "INFO",
    issue_number: int | None = None,
) -> dict:
    return {
        "timestamp": "2026-11-07T00:00:00Z",
        "level": level,
        "event": event,
        "issue_number": issue_number,
        "phase": None,
        "detail": detail,
    }


# --- mapper unit tests -----------------------------------------------------


def test_claim_natives_map_to_claim_opened() -> None:
    for native in ("issue_claimed", "next_issue_claimed"):
        event = to_contract_event(
            _native(native, "title='x'", issue_number=7),
            attempt_id="attempt-1",
        )
        assert event["event"] == "claim_opened"
        assert event["attempt_id"] == "attempt-1"
        assert event["issue"] == 7


def test_phase_transition_maps_to_phase_changed() -> None:
    event = to_contract_event(
        _native("attempt_phase_transitioned", "phase=model_running", issue_number=7),
        attempt_id="attempt-1",
    )
    assert event["event"] == "phase_changed"
    assert event["phase"] == "model_running"
    assert event["attempt_id"] == "attempt-1"


def test_contract_natives_pass_through_with_extras() -> None:
    event = to_contract_event(_native("process_started", "agent_version=1.2.3"))
    assert event["event"] == "process_started"
    assert event["agent_version"] == "1.2.3"

    event = to_contract_event(_native("startup_check", "check=provenance; result=ok"))
    assert event["event"] == "startup_check"
    assert event["check"] == "provenance"
    assert event["result"] == "ok"

    assert to_contract_event(_native("ready", ""))["event"] == "ready"
    assert (
        to_contract_event(_native("halted", "reason=boom"))["reason"] == "boom"
    )
    finished = to_contract_event(
        _native("attempt_finished", "", issue_number=7), attempt_id="attempt-9"
    )
    assert finished["event"] == "attempt_finished"
    assert finished["attempt_id"] == "attempt-9"


def test_outcome_recorded_without_reason_projects_null_reason() -> None:
    event = to_contract_event(
        _native("outcome_recorded", outcome_recorded_detail("complete"), issue_number=7),
        attempt_id="attempt-1",
    )
    assert event["event"] == "outcome_recorded"
    assert event["attempt_id"] == "attempt-1"
    assert event["outcome"] == "complete"
    assert "reason" in event
    assert event["reason"] is None
    validate_contract_event(event)


def test_outcome_recorded_with_reason_projects_reason_text() -> None:
    event = to_contract_event(
        _native(
            "outcome_recorded",
            outcome_recorded_detail("incomplete", reason="tests failed"),
            issue_number=7,
        ),
        attempt_id="attempt-1",
    )
    assert event["outcome"] == "incomplete"
    assert event["reason"] == "tests failed"
    validate_contract_event(event)

    blank = to_contract_event(
        _native("outcome_recorded", "outcome=complete; reason="),
        attempt_id="attempt-1",
    )
    assert blank["reason"] is None


def test_optional_extras_stay_omitted_on_other_events() -> None:
    halted = to_contract_event(_native("halted", ""))
    assert "reason" not in halted
    started = to_contract_event(_native("process_started", ""))
    assert "agent_version" not in started
    resolved = to_contract_event(_native("command_resolved", "request_id=r1"))
    assert "acknowledgement" not in resolved


def test_non_contract_events_gain_native_prefix() -> None:
    event = to_contract_event(_native("workspace_prepared", "branch=x"))
    assert event["event"] == "native.workspace_prepared"
    assert "attempt_id" not in event


def test_level_mapping_defaults_to_info() -> None:
    assert to_contract_event(_native("x", "", level="DEBUG"))["level"] == "debug"
    assert to_contract_event(_native("x", "", level="WARNING"))["level"] == "warning"
    assert to_contract_event(_native("x", "", level="ERROR"))["level"] == "error"
    assert to_contract_event(_native("x", "", level="BOGUS"))["level"] == "info"


def test_text_capped_at_300_characters() -> None:
    event = to_contract_event(_native("ready", "t" * 500))
    assert len(event["text"]) == 300


def test_detail_builders_round_trip_through_mapper() -> None:
    assert parse_detail_fields(process_started_detail("9.9.9")) == {
        "agent_version": "9.9.9"
    }
    assert parse_detail_fields(startup_check_detail("provenance", "skipped")) == {
        "check": "provenance",
        "result": "skipped",
    }
    assert parse_detail_fields(halted_detail("boom")) == {"reason": "boom"}
    assert parse_detail_fields(outcome_recorded_detail("complete")) == {
        "outcome": "complete"
    }
    assert parse_detail_fields(outcome_recorded_detail("incomplete", reason="why")) == {
        "outcome": "incomplete",
        "reason": "why",
    }
    accepted = to_contract_event(
        _native("command_accepted", command_accepted_detail("r1", "stop", "accepted"))
    )
    assert accepted["request_id"] == "r1"
    assert accepted["kind"] == "stop"
    assert accepted["acknowledgement"] == "accepted"
    resolved = to_contract_event(
        _native("command_resolved", command_resolved_detail("r1", "completed"))
    )
    assert resolved["event"] == "command_resolved"
    assert resolved["acknowledgement"] == "completed"


def test_buffer_keeps_newest_50_and_dedups_claims() -> None:
    buffer = ContractEventsBuffer()
    for index in range(60):
        buffer.record(to_contract_event(_native(f"event_{index}", f"d={index}")))
    snapshot = buffer.snapshot()
    assert len(snapshot) == CONTRACT_EVENTS_LIMIT == 50
    assert snapshot[0]["event"] == "native.event_10"
    assert snapshot[-1]["event"] == "native.event_59"


def test_buffer_claim_opened_fires_once_per_attempt() -> None:
    buffer = ContractEventsBuffer()
    first = to_contract_event(
        _native("next_issue_claimed", "", issue_number=7), attempt_id="a1"
    )
    second = to_contract_event(
        _native("issue_claimed", "", issue_number=7), attempt_id="a1"
    )
    assert buffer.record(first) is True
    assert buffer.record(second) is False
    assert [entry["event"] for entry in buffer.snapshot()] == ["claim_opened"]


# --- full Attempt via the shared fakes --------------------------------------


def _wired_attempt(tmp_path: Path, *, secret: str = "s3cr3t-value"):
    """Build a fake-driven lifecycle whose logger feeds one Contract ring."""

    remote, _ = repository_with_main(tmp_path)
    data_dir = tmp_path / "data"
    stream = io.StringIO()
    buffer = ContractEventsBuffer()
    box: dict = {}

    def archive_factory(issue_number: int, started_at: str) -> AttemptArchive:
        return AttemptArchive(
            data_dir,
            issue_number=issue_number,
            started_at=started_at,
            redactions=[secret],
        )

    def attempt_sink(issue_number: int) -> AttemptArchive | None:
        lifecycle = box.get("lifecycle")
        if lifecycle is None:
            return None
        try:
            checkpoint = lifecycle._attempt_state.read()
        except Exception:
            return None
        if checkpoint is None or checkpoint.issue_number != issue_number:
            return None
        return archive_factory(issue_number, checkpoint.started_at)

    logger = JsonEventLogger(
        stream,
        redactions=[secret],
        attempt_sink=attempt_sink,
        contract_events=buffer,
    )

    def event_log(
        event: str,
        detail: str = "",
        level: str = "INFO",
        issue_number: int | None = None,
        phase: str | None = None,
    ) -> None:
        logger.emit(
            event,
            phase=phase,
            detail=detail,
            level=level,
            issue_number=issue_number,
        )

    github = FakeGitHub(issue(24, body="Add logging."))
    executor = FakeModelExecutor(
        actions=[
            lambda wd: write_and_commit(wd, "feature.py", "done", "Add feature")
        ]
    )
    lifecycle = build_lifecycle(
        remote,
        tmp_path / "clone",
        data_dir,
        github,
        executor,
        event_log=event_log,
        attempt_archive_factory=archive_factory,
        contract_events=buffer,
    )
    box["lifecycle"] = lifecycle
    return lifecycle, logger, buffer, stream, data_dir, secret


def _contract_names(buffer: ContractEventsBuffer) -> list[str]:
    return [entry["event"] for entry in buffer.snapshot()]


def test_full_attempt_emits_lifecycle_contract_events_in_order(
    tmp_path: Path,
) -> None:
    lifecycle, _, buffer, _, _, _ = _wired_attempt(tmp_path)

    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.ATTEMPTED
    names = _contract_names(buffer)
    assert "claim_opened" in names
    assert "outcome_recorded" in names
    assert "attempt_finished" in names
    phases = [
        entry.get("phase")
        for entry in buffer.snapshot()
        if entry["event"] == "phase_changed"
    ]
    assert phases == ["setup", "model_running", "pushing"]
    ordered = ["claim_opened", "phase_changed", "outcome_recorded", "attempt_finished"]
    positions = [names.index(name) for name in ordered]
    assert positions == sorted(positions)


def test_archive_events_jsonl_and_manifest_kind_events(tmp_path: Path) -> None:
    lifecycle, _, buffer, _, data_dir, _ = _wired_attempt(tmp_path)

    lifecycle.run_once()

    run_dirs = list((data_dir / "logs" / "24").iterdir())
    assert len(run_dirs) == 1
    events_path = run_dirs[0] / "events.jsonl"
    assert events_path.is_file()
    attempt_record = json.loads((run_dirs[0] / "attempt.json").read_text())
    attempt_id = attempt_record["started_at"]
    lines = [
        json.loads(line)
        for line in events_path.read_text().splitlines()
        if line.strip()
    ]
    assert lines, "expected Contract Events in events.jsonl"
    assert all("issue_number" not in entry for entry in lines)
    assert lines[0]["event"] == "claim_opened"
    assert lines[0]["attempt_id"] == attempt_id
    assert lines[-1]["event"] == "attempt_finished"
    manifest = show_attempt(data_dir, attempt_id)["manifest"]
    events_entries = [entry for entry in manifest if entry["kind"] == "events"]
    assert len(events_entries) == 1
    assert events_entries[0]["path"] == "events.jsonl"


def test_recent_events_redacted_capped_and_prefixed(tmp_path: Path) -> None:
    _, logger, buffer, _, _, secret = _wired_attempt(tmp_path)

    logger.emit(
        "workspace_prepared", detail=f"token={secret}; padding=" + "x" * 500
    )

    (entry,) = buffer.snapshot()
    assert entry["event"] == "native.workspace_prepared"
    assert secret not in entry["text"]
    assert "[REDACTED]" in entry["text"]
    assert len(entry["text"]) <= 300


def test_status_serves_recent_events(tmp_path: Path) -> None:
    lifecycle, _, buffer, _, _, _ = _wired_attempt(tmp_path)

    lifecycle.run_once()
    snapshot = lifecycle.contract_status_snapshot()

    assert snapshot["recent_events"] == buffer.snapshot()
    assert len(snapshot["recent_events"]) <= 50
    assert all(len(entry["text"]) <= 300 for entry in snapshot["recent_events"])


def test_stdout_event_lines_unchanged(tmp_path: Path) -> None:
    lifecycle, _, _, stream, _, _ = _wired_attempt(tmp_path)

    lifecycle.run_once()

    lines = stream.getvalue().splitlines()
    assert lines
    for line in lines:
        record = json.loads(line)
        assert set(record) == {
            "timestamp",
            "level",
            "event",
            "issue_number",
            "phase",
            "detail",
        }
    native_names = {json.loads(line)["event"] for line in lines}
    assert "issue_claimed" in native_names
    assert "attempt_phase_transitioned" in native_names
    assert "claim_opened" not in native_names
    assert "phase_changed" not in native_names


# --- startup, halt, and commands --------------------------------------------


def test_startup_emits_process_started_startup_check_and_ready(
    tmp_path: Path,
) -> None:
    lifecycle, _, buffer, _, _, _ = _wired_attempt(tmp_path)

    lifecycle.mark_starting()
    lifecycle.mark_started()
    lifecycle.mark_started()  # leaving twice never re-announces

    names = _contract_names(buffer)
    assert names.count("startup_check") == 1
    (check,) = [
        entry for entry in buffer.snapshot() if entry["event"] == "startup_check"
    ]
    assert check["check"] == "provenance"
    assert check["result"] == "started"
    assert names.count("ready") == 1


def test_entrypoint_helpers_emit_startup_trio() -> None:
    stream = io.StringIO()
    buffer = ContractEventsBuffer()
    logger = JsonEventLogger(stream, contract_events=buffer)

    entrypoint.emit_process_started(logger)
    entrypoint.emit_startup_check(logger, "provenance", "started")
    entrypoint.emit_startup_check(logger, "provenance", "ok")

    names = [entry["event"] for entry in buffer.snapshot()]
    assert names == ["process_started", "startup_check", "startup_check"]
    assert buffer.snapshot()[0]["agent_version"]
    assert buffer.snapshot()[-1]["result"] == "ok"


def test_entrypoint_halted_helper_and_engine_halt(tmp_path: Path) -> None:
    stream = io.StringIO()
    buffer = ContractEventsBuffer()
    logger = JsonEventLogger(stream, contract_events=buffer)
    entrypoint.emit_halted(logger, "consecutive_error_limit_reached")
    (halted,) = buffer.snapshot()
    assert halted["event"] == "halted"
    assert halted["reason"] == "consecutive_error_limit_reached"
    assert halted["level"] == "error"

    lifecycle, _, engine_buffer, _, _, _ = _wired_attempt(tmp_path)
    lifecycle._record_halt("consecutive_error_limit_reached", "detail")
    names = _contract_names(engine_buffer)
    assert names.count("halted") == 1


def test_command_accepted_then_resolved(tmp_path: Path) -> None:
    lifecycle, _, buffer, _, _, _ = _wired_attempt(tmp_path)

    record = lifecycle.submit_next_issue("cmd-1", 24)
    assert record.acknowledgement is CommandAcknowledgement.ACCEPTED
    lifecycle.run_once()

    names = _contract_names(buffer)
    assert "command_accepted" in names
    assert "command_resolved" in names
    assert names.index("command_accepted") < names.index("command_resolved")
    (accepted,) = [
        entry for entry in buffer.snapshot() if entry["event"] == "command_accepted"
    ]
    assert accepted["request_id"] == "cmd-1"
    assert accepted["kind"] == "next_issue"
    (resolved,) = [
        entry for entry in buffer.snapshot() if entry["event"] == "command_resolved"
    ]
    assert resolved["request_id"] == "cmd-1"
    assert resolved["acknowledgement"] == "completed"
