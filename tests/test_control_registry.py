"""Command registry coverage: one entry per operator command.

Parametrised round-trip (CLI -> socket -> handler -> store) plus
per-command handler tests with no socket.
"""

from __future__ import annotations

import io
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
from simple_coding_agent.control_registry import (
    COMMAND_REGISTRY,
    REGISTERED_OPS,
    build_socket_request,
    cli_label,
    cli_route_to_op,
    get_spec,
    parse_cli_payload,
    parse_payload,
    rejection_errors_for,
)
from simple_coding_agent.control_server import ControlServer
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.lifecycle import AgentLifecycle
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace


EXPECTED_OPS = {
    "stop",
    "stop_after",
    "resume",
    "handoff",
    "next_issue",
    "recovery_retry",
    "recovery_release",
    "errors_reset",
}


def _issue(number: int) -> TrackerIssue:
    return TrackerIssue(
        id=f"issue-{number}",
        number=number,
        title="Implement registry",
        body="Connect boundaries.",
        created_at=datetime(2026, 9, 20, tzinfo=UTC),
        state="OPEN",
        labels=frozenset({"ready-for-agent"}),
        assignee_logins=(),
        blocked_by=0,
        author_login="reporter",
    )


def _claim(number: int) -> Claim:
    return Claim(_issue(number), Assignment(f"issue-{number}", "agent-id"))


def _evidence():
    from simple_coding_agent.lifecycle import AttemptEvidence

    return AttemptEvidence(
        decision=CompletionDecision(
            AttemptOutcome.COMPLETE, True, PublicationPath.COMPLETE, ("ready",)
        ),
        check_command="pytest",
        check_exit_code=0,
        review_cycles=1,
        review_findings="all clear",
        details="Implemented the registry.",
    )


def _make_lifecycle(tmp_path: Path, *, next_claim: Claim | None = None) -> tuple[AgentLifecycle, ControlStore]:
    tracker = FakeTracker(next_claim=next_claim)
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
    store = ControlStore(tmp_path)
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=lambda received_claim, profile, prepared: _evidence(),
        control_store=store,
        sleeper=lambda seconds: None,
        repository="owner/repo",
    )
    return lifecycle, store


def _cli_argv(op: str, request_id: str) -> list[str]:
    if op == "stop":
        return ["stop", "--request-id", request_id]
    if op == "stop_after":
        return ["stop", "--after", "2", "--request-id", request_id]
    if op == "resume":
        return ["resume", "--request-id", request_id]
    if op == "handoff":
        return ["handoff", "now", "--request-id", request_id]
    if op == "next_issue":
        return ["next", "issue", "7", "--request-id", request_id]
    if op == "recovery_retry":
        return ["recovery", "retry", "attempt-1", "--request-id", request_id]
    if op == "recovery_release":
        return [
            "recovery",
            "release",
            "attempt-1",
            "--saved-at",
            "/tmp/kept",
            "--request-id",
            request_id,
        ]
    if op == "errors_reset":
        return ["errors", "reset", "--request-id", request_id]
    raise AssertionError(op)  # pragma: no cover


# Ops that accept on a fresh idle lifecycle with the fixtures above.
ACCEPTED_ON_FRESH = {"stop", "stop_after", "resume", "next_issue", "errors_reset"}
REJECTED_ON_FRESH = {"handoff", "recovery_retry", "recovery_release"}


def test_registry_has_one_entry_per_command() -> None:
    assert set(COMMAND_REGISTRY.keys()) == EXPECTED_OPS
    assert set(REGISTERED_OPS) == EXPECTED_OPS
    for op in EXPECTED_OPS:
        spec = get_spec(op)
        assert spec is not None
        assert spec.op == op
        assert spec.kind
        assert spec.summary
        assert spec.parse_request is not None
        assert spec.invoke is not None
        assert spec.cli_route
        assert isinstance(spec.supports_expect_repo, bool)
        assert isinstance(spec.requires_active_attempt, bool)
        assert isinstance(spec.needs_lifecycle, bool)
        rejected = rejection_errors_for(op)
        assert len(rejected) >= 2
        names = {t.__name__ for t in rejected}
        assert "RequestIdError" in names
        assert "PayloadMismatchError" in names


def test_only_handoff_requires_active_attempt() -> None:
    for op in EXPECTED_OPS:
        spec = get_spec(op)
        assert spec is not None
        if op == "handoff":
            assert spec.requires_active_attempt is True
        else:
            assert spec.requires_active_attempt is False


def test_cli_parser_generated_covers_every_command() -> None:
    parser = agentctl.build_parser()
    cases = [
        (["stop"], "stop"),
        (["stop", "--after", "2"], "stop_after"),
        (["resume"], "resume"),
        (["handoff", "now"], "handoff"),
        (["next", "issue", "7"], "next_issue"),
        (["recovery", "retry", "a"], "recovery_retry"),
        (["recovery", "release", "a", "--saved-at", "/tmp/x"], "recovery_release"),
        (["errors", "reset"], "errors_reset"),
        (["consecutive-errors", "reset"], "errors_reset"),
    ]
    for argv, expected_op in cases:
        args = parser.parse_args(argv)
        assert cli_route_to_op(args) == expected_op
        payload = parse_cli_payload(expected_op, args)
        assert isinstance(payload, dict)
        request = build_socket_request(expected_op, "req-x", payload)
        assert request["op"] in EXPECTED_OPS
        assert request["request_id"] == "req-x"
        label = cli_label(expected_op, payload)
        assert isinstance(label, str) and label


def test_socket_payload_parsers_match_cli() -> None:
    assert parse_payload("stop", {"op": "stop"}) == {}
    assert parse_payload("stop_after", {"after": "2"}) == {"after": 2}
    assert parse_payload("stop_after", {"after": 2}) == {"after": 2}
    assert parse_payload("next_issue", {"issue": "7"}) == {"issue": 7}
    assert parse_payload("recovery_retry", {"attempt_id": "  attempt-1 "}) == {
        "attempt_id": "attempt-1"
    }
    assert parse_payload(
        "recovery_release", {"attempt_id": "a", "saved_at": "/tmp/kept"}
    ) == {"attempt_id": "a", "saved_at": "/tmp/kept"}
    parser = agentctl.build_parser()
    for op, argv in [
        ("stop", ["stop"]),
        ("stop_after", ["stop", "--after", "2"]),
        ("next_issue", ["next", "issue", "7"]),
    ]:
        args = parser.parse_args(argv)
        cli_payload = parse_cli_payload(op, args)
        # Strip the CLI-only expect_repo guard before comparing to socket payload.
        cli_payload = {k: v for k, v in cli_payload.items() if k != "expect_repo"}
        socket_payload = parse_payload(op, build_socket_request(op, "r", cli_payload))
        assert socket_payload == cli_payload


@pytest.mark.parametrize("op", sorted(EXPECTED_OPS))
def test_roundtrip_cli_socket_handler_store(tmp_path: Path, op: str) -> None:
    """CLI -> socket -> registry -> lifecycle handler -> store, per command."""

    lifecycle, store = _make_lifecycle(tmp_path, next_claim=_claim(7))
    socket_path = tmp_path / "agentctl.sock"
    server = ControlServer(socket_path, lifecycle)
    server.start()
    try:
        request_id = f"req-{op}-1"
        argv = ["--socket", str(socket_path), *_cli_argv(op, request_id)]
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = agentctl.main(argv)
        if op in ACCEPTED_ON_FRESH:
            assert code == 0, f"{op} should accept on fresh idle: {err.getvalue()}"
            record = store.get_command(request_id)
            assert record is not None
            assert record.request_id == request_id
            # Idempotent retry returns the stored acknowledgement.
            out2, err2 = io.StringIO(), io.StringIO()
            with redirect_stdout(out2), redirect_stderr(err2):
                retry_code = agentctl.main(argv)
            assert retry_code == 0
            assert store.get_command(request_id) is not None
        else:
            assert code == 1, f"{op} should reject on fresh idle: {out.getvalue()}"
            assert store.get_command(request_id) is None
    finally:
        server.stop()


@pytest.mark.parametrize("op", sorted(EXPECTED_OPS))
def test_lifecycle_submit_registered_no_socket(tmp_path: Path, op: str) -> None:
    """Per-command handler test with no socket via submit_registered_command."""

    lifecycle, store = _make_lifecycle(tmp_path, next_claim=_claim(7))
    request_id = f"req-direct-{op}"
    if op == "stop":
        record = lifecycle.submit_registered_command(op, request_id, {})
        assert record.request_id == request_id
    elif op == "stop_after":
        record = lifecycle.submit_registered_command(op, request_id, {"after": 2})
        assert record.request_id == request_id
    elif op == "resume":
        record = lifecycle.submit_registered_command(op, request_id, {})
        assert record.request_id == request_id
    elif op == "handoff":
        from simple_coding_agent.control import HandoffRejectedError

        with pytest.raises(HandoffRejectedError):
            lifecycle.submit_registered_command(op, request_id, {})
        assert store.get_command(request_id) is None
    elif op == "next_issue":
        record = lifecycle.submit_registered_command(op, request_id, {"issue": 7})
        assert record.request_id == request_id
    elif op in ("recovery_retry", "recovery_release"):
        from simple_coding_agent.recovery import RecoveryRejectedError

        payload = (
            {"attempt_id": "unknown-attempt"}
            if op == "recovery_retry"
            else {"attempt_id": "unknown-attempt", "saved_at": "/tmp/kept"}
        )
        with pytest.raises(RecoveryRejectedError):
            lifecycle.submit_registered_command(op, request_id, payload)
        assert store.get_command(request_id) is None
    elif op == "errors_reset":
        record = lifecycle.submit_registered_command(op, request_id, {})
        assert record.request_id == request_id
        assert store.get_command(request_id) is not None
    else:  # pragma: no cover
        raise AssertionError(op)
