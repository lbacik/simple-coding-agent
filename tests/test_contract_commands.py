"""Operator contract ``stop``/``resume``/``command`` over the contract path.

Every test drives the real CLI entry point in-process with argv and asserts
only on the exit code and the parsed stdout envelope: stdout holds exactly
one JSON document with a trailing newline and stderr stays empty.
"""

from __future__ import annotations

import io
import json
import threading
import time
from collections.abc import Iterator
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

import pytest

from simple_coding_agent import agentctl
from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.completion import (
    AttemptOutcome,
    CompletionDecision,
    PublicationPath,
)
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.lifecycle import AttemptEvidence
from simple_coding_agent.control import ControlStore
from simple_coding_agent.control_server import ControlServer
from simple_coding_agent.telemetry import TelemetryReporter
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace

REPOSITORY = "owner/repo"

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


def make_reporter(**overrides: object) -> TelemetryReporter:
    from datetime import UTC, datetime, timedelta

    start = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
    state = {"now": start}

    def tick() -> datetime:
        state["now"] = state["now"] + timedelta(seconds=1)
        return state["now"]

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
        "clock": tick,
    }
    values.update(overrides)
    return TelemetryReporter(**values)  # type: ignore[arg-type]


def make_lifecycle(tmp_path: Path):
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
            AssertionError("attempt runner must not run during command tests")
        ),
        control_store=store,
        repository=REPOSITORY,
        telemetry=make_reporter(),
        sleeper=lambda seconds: None,
    )
    return lifecycle, store, state


@pytest.fixture
def command_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[ControlServer, object, ControlStore, AttemptStateStore]]:
    lifecycle, store, state = make_lifecycle(tmp_path)
    socket_path = tmp_path / "state" / "contract-commands.sock"
    server = ControlServer(socket_path, lifecycle)  # type: ignore[arg-type]
    server.start()
    monkeypatch.setenv("AGENTCTL_SOCKET", str(socket_path))
    try:
        yield server, lifecycle, store, state
    finally:
        server.stop()


def _run(argv: list[str]) -> tuple[int, dict, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = agentctl.main(argv)
    assert isinstance(code, int)
    stdout = out.getvalue()
    assert stdout.endswith("\n"), "stdout must end with exactly one newline"
    assert stdout.count("\n") == 1, "stdout must hold exactly one envelope"
    return code, json.loads(stdout), stdout, err.getvalue()


def _mutation(
    operation: list[str],
    request_id: str = "req-1a2b3c4d5e6f",
    repository: str = REPOSITORY,
) -> list[str]:
    return [
        "--contract",
        "1",
        *operation,
        "--request-id",
        request_id,
        "--expect-repo",
        repository,
    ]


def _result(envelope: dict) -> dict:
    assert envelope["contract"] == "1.0"
    assert envelope["ok"] is True
    return envelope["result"]


# ---------------------------------------------------------------------------
# Wrong repository stores nothing
# ---------------------------------------------------------------------------


def test_wrong_expect_repo_on_stop_answers_repository_mismatch(command_server) -> None:
    _, _, _, _ = command_server
    code, envelope, _, err = _run(
        _mutation(["stop"], request_id="req-aaaabbbbcccc", repository="owner/other")
    )
    assert err == ""
    assert code == 1
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "repository_mismatch"
    assert envelope["error"]["retryable"] is False
    assert envelope["error"]["request_id"] == "req-aaaabbbbcccc"
    assert envelope["error"]["details"] == {"repository": REPOSITORY}

    lookup_code, lookup_envelope, _, lookup_err = _run(
        ["--contract", "1", "command", "req-aaaabbbbcccc"]
    )
    assert lookup_err == ""
    assert lookup_code == 5
    assert lookup_envelope["error"]["code"] == "not_found"


def test_wrong_expect_repo_on_resume_answers_repository_mismatch(
    command_server,
) -> None:
    _, _, _, _ = command_server
    code, envelope, _, err = _run(
        _mutation(["resume"], request_id="req-ddddeeeeffff", repository="owner/other")
    )
    assert err == ""
    assert code == 1
    assert envelope["error"]["code"] == "repository_mismatch"
    assert envelope["error"]["request_id"] == "req-ddddeeeeffff"
    assert envelope["error"]["details"] == {"repository": REPOSITORY}

    lookup_code, lookup_envelope, _, _ = _run(
        ["--contract", "1", "command", "req-ddddeeeeffff"]
    )
    assert lookup_code == 5
    assert lookup_envelope["error"]["code"] == "not_found"


# ---------------------------------------------------------------------------
# Idempotency: replayed vs payload mismatch
# ---------------------------------------------------------------------------


def test_same_stop_payload_replays_and_kind_change_mismatches(command_server) -> None:
    _, _, _, _ = command_server
    first_code, first_envelope, _, first_err = _run(_mutation(["stop"]))
    assert first_err == ""
    assert first_code == 0
    first = _result(first_envelope)
    assert set(first["command"]) == COMMAND_KEYS
    assert first["command"]["kind"] == "stop"
    assert first["command"]["arguments"] == {}
    assert first["command"]["repository"] == REPOSITORY
    assert first["command"]["acknowledgement"] == "completed"
    assert first["replayed"] is False

    second_code, second_envelope, _, second_err = _run(_mutation(["stop"]))
    assert second_err == ""
    assert second_code == 0
    second = _result(second_envelope)
    assert second["replayed"] is True
    assert second["command"]["request_id"] == first["command"]["request_id"]
    assert second["command"]["sequence"] == first["command"]["sequence"]

    mismatch_code, mismatch_envelope, _, mismatch_err = _run(
        _mutation(["resume"], request_id="req-1a2b3c4d5e6f")
    )
    assert mismatch_err == ""
    assert mismatch_code == 1
    assert mismatch_envelope["error"]["code"] == "payload_mismatch"
    assert mismatch_envelope["error"]["retryable"] is False
    assert mismatch_envelope["error"]["request_id"] == "req-1a2b3c4d5e6f"


# ---------------------------------------------------------------------------
# Stop before the claim completes immediately; stop during an attempt waits
# ---------------------------------------------------------------------------


def test_stop_without_attempt_completes_and_status_is_paused(command_server) -> None:
    _, _, _, _ = command_server
    code, envelope, _, err = _run(
        _mutation(["stop"], request_id="req-stop-idle-1")
    )
    assert err == ""
    assert code == 0
    result = _result(envelope)
    assert result["command"]["acknowledgement"] == "completed"
    assert result["replayed"] is False

    status_code, status_envelope, _, status_err = _run(["--contract", "1", "status"])
    assert status_err == ""
    assert status_code == 0
    status = _result(status_envelope)
    assert status["instance"]["state"] == "paused"
    assert status["intake"]["state"] == "closed"
    assert status["current_attempt"] is None


def test_stop_during_attempt_is_accepted_then_completed(command_server) -> None:
    _, lifecycle, store, state = command_server
    state.start(
        issue_number=42, branch="agent/issue-42", issue_title="Add CSV export"
    )

    code, envelope, _, err = _run(
        _mutation(["stop"], request_id="req-stop-active-1")
    )
    assert err == ""
    assert code == 0
    result = _result(envelope)
    assert result["command"]["acknowledgement"] == "accepted"
    assert result["replayed"] is False

    status_code, status_envelope, _, _ = _run(["--contract", "1", "status"])
    assert status_code == 0
    assert _result(status_envelope)["instance"]["state"] == "draining"

    completed = store.complete_pending_stop()
    assert len(completed) >= 1

    lookup_code, lookup_envelope, _, lookup_err = _run(
        ["--contract", "1", "command", "req-stop-active-1"]
    )
    assert lookup_err == ""
    assert lookup_code == 0
    assert _result(lookup_envelope)["command"]["acknowledgement"] == "completed"
    # The lifecycle object is unused beyond the server; keep the binding.
    assert lifecycle is not None


def test_accepted_stop_survives_a_restart(tmp_path: Path) -> None:
    lifecycle, store, state = make_lifecycle(tmp_path)
    socket_path = tmp_path / "state" / "contract-restart.sock"
    server = ControlServer(socket_path, lifecycle)  # type: ignore[arg-type]
    server.start()
    import os

    old_socket = os.environ.get("AGENTCTL_SOCKET")
    os.environ["AGENTCTL_SOCKET"] = str(socket_path)
    try:
        state.start(
            issue_number=42, branch="agent/issue-42", issue_title="Add CSV export"
        )
        code, envelope, _, _ = _run(
            _mutation(["stop"], request_id="req-stop-restart-1")
        )
        assert code == 0
        assert _result(envelope)["command"]["acknowledgement"] == "accepted"
    finally:
        server.stop()

    restarted_lifecycle, _, _ = make_lifecycle(tmp_path)
    restarted_server = ControlServer(socket_path, restarted_lifecycle)  # type: ignore[arg-type]
    restarted_server.start()
    if old_socket is None:
        os.environ.pop("AGENTCTL_SOCKET", None)
    os.environ["AGENTCTL_SOCKET"] = str(socket_path)
    try:
        lookup_code, lookup_envelope, _, lookup_err = _run(
            ["--contract", "1", "command", "req-stop-restart-1"]
        )
        assert lookup_err == ""
        assert lookup_code == 0
        assert (
            _result(lookup_envelope)["command"]["acknowledgement"] == "accepted"
        )

        status_code, status_envelope, _, _ = _run(["--contract", "1", "status"])
        assert status_code == 0
        assert _result(status_envelope)["instance"]["state"] == "draining"
    finally:
        restarted_server.stop()
        if old_socket is None:
            os.environ.pop("AGENTCTL_SOCKET", None)
        else:
            os.environ["AGENTCTL_SOCKET"] = old_socket


# ---------------------------------------------------------------------------
# Resume supersedes a pending stop and reopens intake
# ---------------------------------------------------------------------------


def test_resume_supersedes_accepted_stop_and_reopens_intake(command_server) -> None:
    _, lifecycle, _, _ = command_server
    # A genuinely running attempt (mirroring the lifecycle-level resume test):
    # a bare checkpoint with nothing processing is a recovery hold and
    # resume must stay rejected there.
    tracker = lifecycle._tracker  # noqa: SLF001 - setup seam
    tracker.next_claim = Claim(
        TrackerIssue(
            id="issue-42",
            number=42,
            title="Add CSV export",
            body="Connect boundaries.",
            created_at=datetime(2026, 9, 20, tzinfo=UTC),
            state="OPEN",
            labels=frozenset({"ready-for-agent"}),
            assignee_logins=(),
            blocked_by=0,
            author_login="reporter",
        ),
        Assignment("issue-42", "agent-id"),
    )
    started = threading.Event()
    release = threading.Event()

    def blocking_workflow(received_claim, profile, prepared):
        started.set()
        assert release.wait(timeout=30)
        return AttemptEvidence(
            decision=CompletionDecision(
                AttemptOutcome.COMPLETE, True, PublicationPath.COMPLETE, ("ready",)
            ),
            check_command="pytest",
            check_exit_code=0,
            review_cycles=1,
            review_findings="all clear",
            details="Implemented the export.",
        )

    lifecycle.set_attempt_runner(blocking_workflow)
    worker = threading.Thread(target=lifecycle.run_once)
    worker.start()
    assert started.wait(timeout=30)
    try:
        stop_code, _, _, _ = _run(_mutation(["stop"], request_id="req-stop-sup-1"))
        assert stop_code == 0

        resume_code, resume_envelope, _, resume_err = _run(
            _mutation(["resume"], request_id="req-resume-sup-1")
        )
        assert resume_err == ""
        assert resume_code == 0
        resume = _result(resume_envelope)
        assert resume["command"]["kind"] == "resume"
        assert resume["command"]["acknowledgement"] == "completed"
        assert resume["replayed"] is False

        stop_lookup_code, stop_lookup_envelope, _, _ = _run(
            ["--contract", "1", "command", "req-stop-sup-1"]
        )
        assert stop_lookup_code == 0
        assert (
            _result(stop_lookup_envelope)["command"]["acknowledgement"]
            == "superseded"
        )

        status_code, status_envelope, _, _ = _run(["--contract", "1", "status"])
        assert status_code == 0
        status = _result(status_envelope)
        assert status["instance"]["state"] == "active"
        assert status["intake"]["state"] == "open"
    finally:
        release.set()
        worker.join(timeout=30)
        assert not worker.is_alive()


# ---------------------------------------------------------------------------
# Command lookup and responsiveness
# ---------------------------------------------------------------------------


def test_command_returns_record_and_unknown_is_not_found(command_server) -> None:
    _, _, _, _ = command_server
    stop_code, stop_envelope, _, _ = _run(
        _mutation(["stop"], request_id="req-cmd-found-1")
    )
    assert stop_code == 0
    stored = _result(stop_envelope)["command"]

    lookup_code, lookup_envelope, _, lookup_err = _run(
        ["--contract", "1", "command", "req-cmd-found-1"]
    )
    assert lookup_err == ""
    assert lookup_code == 0
    found = _result(lookup_envelope)["command"]
    assert found == stored

    missing_code, missing_envelope, _, missing_err = _run(
        ["--contract", "1", "command", "req-000000000000"]
    )
    assert missing_err == ""
    assert missing_code == 5
    assert missing_envelope["ok"] is False
    assert missing_envelope["error"]["code"] == "not_found"
    assert missing_envelope["error"]["retryable"] is False


def test_command_answers_while_a_mutation_holds_the_control_lock(
    command_server,
) -> None:
    _, lifecycle, _, state = command_server
    state.start(
        issue_number=42, branch="agent/issue-42", issue_title="Add CSV export"
    )
    stop_code, _, _, _ = _run(_mutation(["stop"], request_id="req-cmd-lock-1"))
    assert stop_code == 0

    lock = lifecycle._control_lock  # noqa: SLF001 - hold it as the intake claim does
    lock.acquire()
    try:
        start = time.monotonic()
        lookup_code, lookup_envelope, _, lookup_err = _run(
            ["--contract", "1", "command", "req-cmd-lock-1"]
        )
        elapsed = time.monotonic() - start
        assert lookup_err == ""
        assert lookup_code == 0
        assert elapsed < 3
        assert (
            _result(lookup_envelope)["command"]["request_id"] == "req-cmd-lock-1"
        )
    finally:
        lock.release()
