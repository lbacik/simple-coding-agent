"""Operator-contract coverage for the remaining mutations (issue #340).

Every test drives the real CLI entry point in-process with argv and asserts
only on the exit code and the parsed stdout envelope: stdout holds exactly
one JSON document with a trailing newline and stderr stays empty (except the
legacy retry-hint test, which asserts on the legacy stderr hint). The engine
is the real lifecycle behind a control socket, so the envelope, idempotency,
rejection mapping, and durability behaviour here are the production path,
not a stub.
"""

from __future__ import annotations

import io
import json
import threading
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
from simple_coding_agent.control import ControlStore
from simple_coding_agent.control_server import ControlServer
from simple_coding_agent.github_tracker import TrackerIssue
from simple_coding_agent.lifecycle import AttemptEvidence
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


def eligible_issue(number: int, **overrides: object) -> TrackerIssue:
    base: dict[str, object] = {
        "id": f"issue-{number}",
        "number": number,
        "title": f"Issue {number}",
        "body": f"Implement {number}.",
        "created_at": datetime(2026, 9, 20, 13, 0, tzinfo=UTC),
        "state": "OPEN",
        "labels": frozenset({"ready-for-agent"}),
        "assignee_logins": (),
        "blocked_by": 0,
        "author_login": "reporter",
    }
    base.update(overrides)
    return TrackerIssue(**base)  # type: ignore[arg-type]


def complete_evidence() -> AttemptEvidence:
    return AttemptEvidence(
        decision=CompletionDecision(
            AttemptOutcome.COMPLETE, True, PublicationPath.COMPLETE, ("ready",)
        ),
        check_command="pytest",
        check_exit_code=0,
        review_cycles=1,
        review_findings="all clear",
        details="Implemented the change.",
    )


def make_reporter(**overrides: object) -> TelemetryReporter:
    from datetime import timedelta

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


def make_lifecycle(
    tmp_path: Path,
    tracker: FakeTracker,
    runner=None,
    publisher=None,
):
    from simple_coding_agent.lifecycle import AgentLifecycle

    if runner is None:

        def runner(received_claim, profile, prepared):  # type: ignore[no-untyped-def]
            return complete_evidence()

    workspace = InMemoryWorkspace(tmp_path / "repo")
    store = ControlStore(tmp_path)
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=publisher if publisher is not None else FakePublisher(),
        attempt_runner=runner,
        control_store=store,
        repository=REPOSITORY,
        telemetry=make_reporter(),
        sleeper=lambda seconds: None,
    )
    return lifecycle, store


def start_server(
    tmp_path: Path,
    tracker: FakeTracker,
    monkeypatch: pytest.MonkeyPatch,
    runner=None,
    publisher=None,
):
    lifecycle, store = make_lifecycle(tmp_path, tracker, runner, publisher)
    socket_path = tmp_path / "state" / "contract-remaining.sock"
    server = ControlServer(socket_path, lifecycle)  # type: ignore[arg-type]
    server.start()
    monkeypatch.setenv("AGENTCTL_SOCKET", str(socket_path))
    return server, lifecycle, store


@pytest.fixture
def command_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[ControlServer, object, ControlStore]]:
    server, lifecycle, store = start_server(
        tmp_path, FakeTracker(), monkeypatch
    )
    try:
        yield server, lifecycle, store
    finally:
        server.stop()


def restart_server(
    tmp_path: Path,
    tracker: FakeTracker,
    monkeypatch: pytest.MonkeyPatch,
    runner=None,
):
    lifecycle, store = make_lifecycle(tmp_path, tracker, runner)
    socket_path = tmp_path / "state" / "contract-remaining.sock"
    server = ControlServer(socket_path, lifecycle)  # type: ignore[arg-type]
    server.start()
    monkeypatch.setenv("AGENTCTL_SOCKET", str(socket_path))
    return server, lifecycle, store


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


def _error(envelope: dict) -> dict:
    assert envelope["contract"] == "1.0"
    assert envelope["ok"] is False
    return envelope["error"]


# ---------------------------------------------------------------------------
# Wrong --expect-repo on every remaining mutation stores nothing
# ---------------------------------------------------------------------------


def _remaining_mutations() -> list[tuple[str, list[str]]]:
    return [
        ("stop-after", ["stop", "--after", "2"]),
        ("next-issue", ["next", "issue", "24"]),
        ("handoff", ["handoff", "now"]),
        ("recovery-retry", ["recovery", "retry", "attempt-1"]),
        (
            "recovery-release",
            ["recovery", "release", "attempt-1", "--saved-at", "s3://bucket/work"],
        ),
        ("errors-reset", ["errors", "reset"]),
    ]


_REMAINING_MUTATION_CASES = _remaining_mutations()


@pytest.mark.parametrize(
    "name,operation",
    _REMAINING_MUTATION_CASES,
    ids=[name for name, _ in _REMAINING_MUTATION_CASES],
)
def test_wrong_expect_repo_answers_repository_mismatch_and_stores_nothing(
    command_server, name: str, operation: list[str]
) -> None:
    _, _, _ = command_server
    request_id = f"req-wrong-repo-{name}"
    code, envelope, _, err = _run(
        _mutation(operation, request_id=request_id, repository="owner/other")
    )
    assert err == ""
    assert code == 1
    error = _error(envelope)
    assert error["code"] == "repository_mismatch"
    assert error["retryable"] is False
    assert error["request_id"] == request_id
    assert error["details"] == {"repository": REPOSITORY}

    lookup_code, lookup_envelope, _, lookup_err = _run(
        ["--contract", "1", "command", request_id]
    )
    assert lookup_err == ""
    assert lookup_code == 5
    assert lookup_envelope["error"]["code"] == "not_found"


# ---------------------------------------------------------------------------
# Value errors answer invalid_argument, distinct from usage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operation",
    [
        ["stop", "--after", "0"],
        ["next", "issue", "0"],
        ["next", "issue", "-4"],
        ["recovery", "retry", "   "],
        ["recovery", "release", "attempt-1", "--saved-at", "   "],
    ],
)
def test_value_errors_answer_invalid_argument(command_server, operation: list[str]) -> None:
    _, _, _ = command_server
    code, envelope, _, err = _run(_mutation(operation))
    assert err == ""
    assert code == 1
    error = _error(envelope)
    assert error["code"] == "invalid_argument"
    assert error["retryable"] is False
    assert error["request_id"] == "req-1a2b3c4d5e6f"

    lookup_code, lookup_envelope, _, _ = _run(
        ["--contract", "1", "command", "req-1a2b3c4d5e6f"]
    )
    assert lookup_code == 5
    assert lookup_envelope["error"]["code"] == "not_found"


# ---------------------------------------------------------------------------
# stop --after: replay, payload mismatch, restart durability, completion
# ---------------------------------------------------------------------------


def test_stop_after_replay_and_payload_mismatch(command_server) -> None:
    _, _, _ = command_server
    first_code, first_envelope, _, first_err = _run(
        _mutation(["stop", "--after", "2"])
    )
    assert first_err == ""
    assert first_code == 0
    first = _result(first_envelope)
    assert set(first["command"]) == COMMAND_KEYS
    assert first["command"]["kind"] == "stop_after"
    assert first["command"]["arguments"] == {"after": 2}
    assert first["command"]["acknowledgement"] == "accepted"
    assert first["replayed"] is False

    second_code, second_envelope, _, second_err = _run(
        _mutation(["stop", "--after", "2"])
    )
    assert second_err == ""
    assert second_code == 0
    second = _result(second_envelope)
    assert second["replayed"] is True
    assert second["command"]["sequence"] == first["command"]["sequence"]

    mismatch_code, mismatch_envelope, _, mismatch_err = _run(
        _mutation(["resume"], request_id="req-1a2b3c4d5e6f")
    )
    assert mismatch_err == ""
    assert mismatch_code == 1
    assert _error(mismatch_envelope)["code"] == "payload_mismatch"


def test_stop_after_count_survives_restart_and_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = FakeTracker(
        issues={24: eligible_issue(24), 25: eligible_issue(25)}, fifo=[24, 25]
    )
    server, _, _ = start_server(tmp_path, tracker, monkeypatch)
    try:
        code, envelope, _, err = _run(
            _mutation(["stop", "--after", "2"], request_id="req-after-restart")
        )
        assert err == ""
        assert code == 0
        submitted = _result(envelope)
        assert submitted["command"]["acknowledgement"] == "accepted"
        assert submitted["command"]["arguments"] == {"after": 2}
    finally:
        server.stop()

    tracker2 = FakeTracker(
        issues={24: eligible_issue(24), 25: eligible_issue(25)}, fifo=[24, 25]
    )
    server2, lifecycle2, _ = restart_server(tmp_path, tracker2, monkeypatch)
    try:
        lookup_code, lookup_envelope, _, lookup_err = _run(
            ["--contract", "1", "command", "req-after-restart"]
        )
        assert lookup_err == ""
        assert lookup_code == 0
        looked_up = _result(lookup_envelope)
        assert looked_up["command"]["acknowledgement"] == "accepted"
        assert looked_up["command"]["sequence"] == submitted["command"]["sequence"]

        assert lifecycle2.run_once().status.value == "attempted"
        assert lifecycle2.run_once().status.value == "attempted"

        final_code, final_envelope, _, final_err = _run(
            ["--contract", "1", "command", "req-after-restart"]
        )
        assert final_err == ""
        assert final_code == 0
        assert _result(final_envelope)["command"]["acknowledgement"] == "completed"

        status_code, status_envelope, _, status_err = _run(["--contract", "1", "status"])
        assert status_err == ""
        assert status_code == 0
        assert _result(status_envelope)["intake"]["state"] == "closed"
    finally:
        server2.stop()


# ---------------------------------------------------------------------------
# next issue: restart durability and claim-boundary revalidation
# ---------------------------------------------------------------------------


def test_next_issue_survives_restart_and_fails_when_ineligible_at_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = FakeTracker(
        issues={24: eligible_issue(24), 25: eligible_issue(25)}, fifo=[25]
    )
    server, _, _ = start_server(tmp_path, tracker, monkeypatch)
    try:
        code, envelope, _, err = _run(
            _mutation(["next", "issue", "24"], request_id="req-next-restart")
        )
        assert err == ""
        assert code == 0
        submitted = _result(envelope)
        assert submitted["command"]["kind"] == "next_issue"
        assert submitted["command"]["arguments"] == {"issue": 24}
        assert submitted["command"]["acknowledgement"] == "accepted"
        assert submitted["replayed"] is False
    finally:
        server.stop()

    tracker2 = FakeTracker(
        issues={24: eligible_issue(24), 25: eligible_issue(25)}, fifo=[25]
    )
    server2, lifecycle2, _ = restart_server(tmp_path, tracker2, monkeypatch)
    try:
        lookup_code, lookup_envelope, _, lookup_err = _run(
            ["--contract", "1", "command", "req-next-restart"]
        )
        assert lookup_err == ""
        assert lookup_code == 0
        assert _result(lookup_envelope)["command"]["acknowledgement"] == "accepted"

        # The issue loses eligibility before the claim boundary.
        tracker2.issues[24] = eligible_issue(24, assignee_logins=("human",))
        assert lifecycle2.run_once().status.value == "attempted"

        failed_code, failed_envelope, _, failed_err = _run(
            ["--contract", "1", "command", "req-next-restart"]
        )
        assert failed_err == ""
        assert failed_code == 0
        failed = _result(failed_envelope)["command"]
        assert failed["acknowledgement"] == "not_fulfilled"
        assert "24" in failed["detail"]

        status_code, status_envelope, _, status_err = _run(["--contract", "1", "status"])
        assert status_err == ""
        assert status_code == 0
        status = _result(status_envelope)
        assert status["intake"]["state"] == "open"
        assert status["intake"]["priority"] is None
        assert status["last_finished"]["issue"]["number"] == 25
    finally:
        server2.stop()


def test_next_issue_replay_and_payload_mismatch(command_server) -> None:
    _, lifecycle, _ = command_server
    tracker = lifecycle._tracker  # noqa: SLF001 - seed an eligible target
    tracker.issues[24] = eligible_issue(24)

    first_code, first_envelope, _, first_err = _run(_mutation(["next", "issue", "24"]))
    assert first_err == ""
    assert first_code == 0
    first = _result(first_envelope)
    assert first["command"]["acknowledgement"] == "accepted"
    assert first["replayed"] is False

    second_code, second_envelope, _, second_err = _run(
        _mutation(["next", "issue", "24"])
    )
    assert second_err == ""
    assert second_code == 0
    second = _result(second_envelope)
    assert second["replayed"] is True
    assert second["command"]["sequence"] == first["command"]["sequence"]

    mismatch_code, mismatch_envelope, _, mismatch_err = _run(
        _mutation(["next", "issue", "25"], request_id="req-1a2b3c4d5e6f")
    )
    assert mismatch_err == ""
    assert mismatch_code == 1
    assert _error(mismatch_envelope)["code"] == "payload_mismatch"

    ineligible_code, ineligible_envelope, _, ineligible_err = _run(
        _mutation(["next", "issue", "99"], request_id="req-next-missing")
    )
    assert ineligible_err == ""
    assert ineligible_code == 1
    assert _error(ineligible_envelope)["code"] == "rejected"


# ---------------------------------------------------------------------------
# handoff now: accepted while active, not_fulfilled with work retained
# ---------------------------------------------------------------------------


def test_handoff_now_not_fulfilled_with_work_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = FakeTracker(issues={24: eligible_issue(24)}, fifo=[24])
    started = threading.Event()
    release = threading.Event()

    def blocking_workflow(received_claim, profile, prepared):  # type: ignore[no-untyped-def]
        started.set()
        assert release.wait(timeout=30)
        return complete_evidence()

    publisher = FakePublisher()
    server, lifecycle, _ = start_server(
        tmp_path, tracker, monkeypatch, blocking_workflow, publisher
    )
    worker = threading.Thread(target=lifecycle.run_once)
    try:
        worker.start()
        assert started.wait(timeout=30)

        code, envelope, _, err = _run(
            _mutation(["handoff", "now"], request_id="req-handoff-contract")
        )
        assert err == ""
        assert code == 0
        submitted = _result(envelope)
        assert submitted["command"]["kind"] == "handoff"
        assert submitted["command"]["arguments"] == {}
        assert submitted["command"]["acknowledgement"] == "accepted"
        assert submitted["replayed"] is False
    finally:
        release.set()
        worker.join(timeout=30)
        server.stop()

    # The failed handoff discards nothing: the attempt published its
    # COMPLETE outcome and that work stands.
    assert publisher.outcomes == [AttemptOutcome.COMPLETE]

    server2, _, _ = restart_server(
        tmp_path, FakeTracker(issues={24: eligible_issue(24)}), monkeypatch
    )
    try:
        lookup_code, lookup_envelope, _, lookup_err = _run(
            ["--contract", "1", "command", "req-handoff-contract"]
        )
        assert lookup_err == ""
        assert lookup_code == 0
        looked_up = _result(lookup_envelope)["command"]
        # A handoff interrupted short of publication never lingers as
        # accepted: the finalization boundary settles it terminally.
        assert looked_up["acknowledgement"] == "not_fulfilled"
        assert looked_up["sequence"] == submitted["command"]["sequence"]

        status_code, status_envelope, _, status_err = _run(["--contract", "1", "status"])
        assert status_err == ""
        assert status_code == 0
        assert _result(status_envelope)["intake"]["state"] == "closed"
    finally:
        server2.stop()


# ---------------------------------------------------------------------------
# recovery retry / release and errors reset over the contract path
# ---------------------------------------------------------------------------


def test_recovery_retry_unknown_attempt_is_rejected_without_storage(
    command_server,
) -> None:
    _, _, _ = command_server
    code, envelope, _, err = _run(
        _mutation(["recovery", "retry", "no-such-attempt"], request_id="req-rec-unknown")
    )
    assert err == ""
    assert code == 1
    error = _error(envelope)
    assert error["code"] == "rejected"
    assert error["retryable"] is False

    lookup_code, lookup_envelope, _, _ = _run(
        ["--contract", "1", "command", "req-rec-unknown"]
    )
    assert lookup_code == 5
    assert lookup_envelope["error"]["code"] == "not_found"


def test_recovery_retry_not_fulfilled_is_stored_replayable_and_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, lifecycle, _ = start_server(tmp_path, FakeTracker(), monkeypatch)
    try:
        checkpoint = lifecycle._attempt_state.read()  # noqa: SLF001
        assert checkpoint is None
        state = AttemptStateStore(tmp_path)
        state.start(issue_number=24, branch="agent/issue-24")
        attempt_id = state.read().started_at  # type: ignore[union-attr]
        assert isinstance(attempt_id, str)
    finally:
        server.stop()

    server2, _, _ = restart_server(tmp_path, FakeTracker(), monkeypatch)
    try:
        code, envelope, _, err = _run(
            _mutation(
                ["recovery", "retry", attempt_id], request_id="req-rec-retry-held"
            )
        )
        assert err == "", err
        assert code == 0
        submitted = _result(envelope)
        assert submitted["command"]["kind"] == "recovery_retry"
        assert submitted["command"]["arguments"] == {"attempt_id": attempt_id}
        assert submitted["command"]["acknowledgement"] == "not_fulfilled"
        assert submitted["replayed"] is False

        replay_code, replay_envelope, _, replay_err = _run(
            _mutation(
                ["recovery", "retry", attempt_id], request_id="req-rec-retry-held"
            )
        )
        assert replay_err == ""
        assert replay_code == 0
        replayed = _result(replay_envelope)
        assert replayed["replayed"] is True
        assert replayed["command"]["sequence"] == submitted["command"]["sequence"]
    finally:
        server2.stop()

    server3, _, _ = restart_server(tmp_path, FakeTracker(), monkeypatch)
    try:
        lookup_code, lookup_envelope, _, lookup_err = _run(
            ["--contract", "1", "command", "req-rec-retry-held"]
        )
        assert lookup_err == ""
        assert lookup_code == 0
        assert _result(lookup_envelope)["command"]["acknowledgement"] == "not_fulfilled"
    finally:
        server3.stop()


def test_recovery_release_rejected_for_unknown_attempt(command_server) -> None:
    _, _, _ = command_server
    code, envelope, _, err = _run(
        _mutation(
            ["recovery", "release", "no-such-attempt", "--saved-at", "s3://bucket/work"],
            request_id="req-rel-unknown",
        )
    )
    assert err == ""
    assert code == 1
    assert _error(envelope)["code"] == "rejected"


def test_recovery_release_stores_saved_at_echoes_it_and_survives_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, lifecycle, _ = start_server(tmp_path, FakeTracker(), monkeypatch)
    try:
        checkpoint = lifecycle._attempt_state.read()  # noqa: SLF001
        assert checkpoint is None
        state = AttemptStateStore(tmp_path)
        state.start(issue_number=24, branch="agent/issue-24")
        attempt_id = state.read().started_at  # type: ignore[union-attr]
        assert isinstance(attempt_id, str)
    finally:
        server.stop()

    server2, _, _ = restart_server(tmp_path, FakeTracker(), monkeypatch)
    try:
        code, envelope, _, err = _run(
            _mutation(
                ["recovery", "release", attempt_id, "--saved-at", "s3://bucket/work"],
                request_id="req-rec-release-held",
            )
        )
        assert err == "", err
        assert code == 0
        submitted = _result(envelope)
        assert submitted["command"]["kind"] == "recovery_release"
        assert submitted["command"]["repository"] == REPOSITORY
        # --saved-at is stored and echoed as a location only.
        assert submitted["command"]["arguments"] == {
            "attempt_id": attempt_id,
            "saved_at": "s3://bucket/work",
        }
        terminal = submitted["command"]["acknowledgement"]
        assert terminal in ("completed", "not_fulfilled")
        assert submitted["replayed"] is False

        replay_code, replay_envelope, _, replay_err = _run(
            _mutation(
                ["recovery", "release", attempt_id, "--saved-at", "s3://bucket/work"],
                request_id="req-rec-release-held",
            )
        )
        assert replay_err == ""
        assert replay_code == 0
        replayed = _result(replay_envelope)
        assert replayed["replayed"] is True
        assert replayed["command"]["sequence"] == submitted["command"]["sequence"]
        assert replayed["command"]["arguments"] == {
            "attempt_id": attempt_id,
            "saved_at": "s3://bucket/work",
        }
    finally:
        server2.stop()

    server3, _, _ = restart_server(tmp_path, FakeTracker(), monkeypatch)
    try:
        lookup_code, lookup_envelope, _, lookup_err = _run(
            ["--contract", "1", "command", "req-rec-release-held"]
        )
        assert lookup_err == ""
        assert lookup_code == 0
        looked_up = _result(lookup_envelope)["command"]
        assert looked_up["acknowledgement"] == terminal
        assert looked_up["arguments"] == {
            "attempt_id": attempt_id,
            "saved_at": "s3://bucket/work",
        }
    finally:
        server3.stop()


def test_errors_reset_completes_and_replays(command_server) -> None:
    _, _, _ = command_server
    code, envelope, _, err = _run(
        _mutation(["errors", "reset"], request_id="req-errors-reset-1")
    )
    assert err == ""
    assert code == 0
    first = _result(envelope)
    assert first["command"]["kind"] == "errors_reset"
    assert first["command"]["arguments"] == {}
    assert first["command"]["acknowledgement"] == "completed"
    assert first["replayed"] is False

    replay_code, replay_envelope, _, replay_err = _run(
        _mutation(["errors", "reset"], request_id="req-errors-reset-1")
    )
    assert replay_err == ""
    assert replay_code == 0
    replayed = _result(replay_envelope)
    assert replayed["replayed"] is True
    assert replayed["command"]["sequence"] == first["command"]["sequence"]


# ---------------------------------------------------------------------------
# Legacy retry hint for recovery release carries --saved-at
# ---------------------------------------------------------------------------


def test_legacy_recovery_release_hint_contains_saved_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTCTL_SOCKET", str(tmp_path / "missing.sock"))
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = agentctl.main(
            ["recovery", "release", "attempt-1", "--saved-at", "s3://bucket/work"]
        )
    assert code == 2
    assert out.getvalue() == ""
    hint = err.getvalue()
    assert "recovery release attempt-1 --saved-at s3://bucket/work" in hint
    assert "--request-id" in hint
