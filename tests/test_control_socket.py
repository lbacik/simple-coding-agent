"""Socket delivery and agentctl client behaviour (issue #80)."""

from __future__ import annotations

import json
import os
import socket
import stat
import threading
from pathlib import Path

import pytest

from simple_coding_agent.control import ControlStore, IntakeState
from simple_coding_agent.control_server import ControlServer, ControlUnavailableError


class StubControl:
    """Minimal lifecycle-facing control surface for socket tests."""

    def __init__(self, data_dir: Path, repository: str = "owner/repo") -> None:
        self._store = ControlStore(data_dir)
        self._repository = repository

    def submit_stop(self, request_id: str, *, has_active_attempt: bool = False):
        return self._store.submit_stop(request_id, has_active_attempt=has_active_attempt)

    def submit_resume(self, request_id: str):
        return self._store.submit_resume(request_id, has_active_attempt=False)

    def get_command(self, request_id: str):
        return self._store.get_command(request_id)

    def control_status(self, repository: str | None = None):
        from simple_coding_agent.control import build_status

        return build_status(
            repository=repository or self._repository,
            intake=self._store.intake_state(),
            recovering=False,
            active_attempt=None,
            pending_command_id=self._store.pending_command_id(),
            get_command=self._store.get_command,
            recent_commands=self._store.recent_commands,
        )


def start_server(tmp_path: Path) -> tuple[ControlServer, StubControl, Path]:
    control = StubControl(tmp_path)
    socket_path = tmp_path / "state" / "agentctl.sock"
    server = ControlServer(socket_path, control)
    server.start()
    return server, control, socket_path


def send(socket_path: Path, payload: dict) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(5)
        client.connect(str(socket_path))
        client.sendall((json.dumps(payload) + "\n").encode())
        data = b""
        while not data.endswith(b"\n"):
            chunk = client.recv(65536)
            if not chunk:
                break
            data += chunk
    return json.loads(data.decode())


def test_stop_status_and_command_roundtrip(tmp_path: Path) -> None:
    server, _, socket_path = start_server(tmp_path)
    try:
        reply = send(socket_path, {"op": "stop", "request_id": "req-1"})
        assert reply["ok"] is True
        assert reply["request_id"] == "req-1"
        assert reply["sequence"] == 1
        assert reply["acknowledgement"] == "completed"

        status = send(socket_path, {"op": "status"})
        assert status["ok"] is True
        assert status["status"]["repository"] == "owner/repo"
        assert status["status"]["intake"] == IntakeState.STOPPED.value

        lookup = send(socket_path, {"op": "command", "request_id": "req-1"})
        assert lookup["ok"] is True
        assert lookup["command"]["acknowledgement"] == "completed"
    finally:
        server.stop()


def test_identical_retry_returns_the_same_sequence(tmp_path: Path) -> None:
    server, _, socket_path = start_server(tmp_path)
    try:
        first = send(socket_path, {"op": "stop", "request_id": "req-1"})
        second = send(socket_path, {"op": "stop", "request_id": "req-1"})
        assert second["sequence"] == first["sequence"]
        assert second["acknowledgement"] == first["acknowledgement"]
    finally:
        server.stop()


def test_concurrent_submits_are_serialized_with_unique_sequences(
    tmp_path: Path,
) -> None:
    server, _, socket_path = start_server(tmp_path)
    try:
        results: list[dict] = []
        errors: list[Exception] = []

        def submit(index: int) -> None:
            try:
                results.append(send(socket_path, {"op": "stop", "request_id": f"req-{index}"}))
            except Exception as error:  # pragma: no cover - diagnostic
                errors.append(error)

        threads = [threading.Thread(target=submit, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert not errors
        sequences = sorted(reply["sequence"] for reply in results)
        assert sequences == list(range(1, 9))
    finally:
        server.stop()


def test_unknown_command_lookup_returns_null(tmp_path: Path) -> None:
    server, _, socket_path = start_server(tmp_path)
    try:
        reply = send(socket_path, {"op": "command", "request_id": "req-missing"})
        assert reply["ok"] is True
        assert reply["command"] is None
    finally:
        server.stop()


def test_server_recreates_a_stale_socket_endpoint(tmp_path: Path) -> None:
    socket_path = tmp_path / "state" / "agentctl.sock"
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.write_bytes(b"stale")
    control = StubControl(tmp_path)

    server = ControlServer(socket_path, control)
    server.start()
    try:
        reply = send(socket_path, {"op": "status"})
        assert reply["ok"] is True
    finally:
        server.stop()


def test_socket_endpoint_is_private_to_the_owner(tmp_path: Path) -> None:
    server, _, socket_path = start_server(tmp_path)
    try:
        mode = stat.S_IMODE(os.stat(socket_path).st_mode)
        assert mode & 0o077 == 0, f"socket must not be group/other accessible: {oct(mode)}"
    finally:
        server.stop()


def test_client_reports_a_connection_error_when_the_process_is_unavailable(
    tmp_path: Path,
) -> None:
    from simple_coding_agent.agentctl import run_command_lookup, run_status

    missing = tmp_path / "state" / "agentctl.sock"
    with pytest.raises(ControlUnavailableError):
        run_status(missing)
    with pytest.raises(ControlUnavailableError):
        run_command_lookup(missing, "req-1")


def test_agentctl_cli_generates_a_request_id_and_prints_the_ack(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    server, _, socket_path = start_server(tmp_path)
    try:
        from simple_coding_agent.agentctl import main as agentctl_main

        code = agentctl_main(["--socket", str(socket_path), "stop"])
        assert code == 0
        out = capsys.readouterr().out
        assert "accepted" in out or "completed" in out
        assert "sequence" in out
    finally:
        server.stop()


def test_failed_submission_keeps_its_request_id_for_retry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from simple_coding_agent.agentctl import main as agentctl_main

    missing = tmp_path / "state" / "agentctl.sock"
    code = agentctl_main(["--socket", str(missing), "stop", "--request-id", "req-retry-1"])

    assert code == 2
    err = capsys.readouterr().err
    assert "req-retry-1" in err
    assert "--request-id req-retry-1" in err


# ---------------------------------------------------------------------------
# resume (issue #81)
# ---------------------------------------------------------------------------


def test_resume_replaces_a_pending_stop_over_the_socket(tmp_path: Path) -> None:
    from simple_coding_agent.control import CommandAcknowledgement

    server, control, socket_path = start_server(tmp_path)
    try:
        stopped = control.submit_stop("req-stop", has_active_attempt=True)
        assert stopped.acknowledgement == CommandAcknowledgement.ACCEPTED

        reply = send(socket_path, {"op": "resume", "request_id": "req-resume"})
        assert reply["ok"] is True
        assert reply["request_id"] == "req-resume"
        assert reply["kind"] == "resume"
        assert reply["acknowledgement"] == "completed"
        assert "req-stop" in reply["detail"]

        status = send(socket_path, {"op": "status"})
        assert status["ok"] is True
        assert status["status"]["intake"] == "running"
        assert status["status"]["pending_command"] is None

        lookup = send(socket_path, {"op": "command", "request_id": "req-stop"})
        assert lookup["ok"] is True
        assert lookup["command"]["acknowledgement"] == "superseded"
        assert "req-resume" in lookup["command"]["detail"]
    finally:
        server.stop()


def test_resume_with_a_blank_request_id_is_rejected(tmp_path: Path) -> None:
    server, _, socket_path = start_server(tmp_path)
    try:
        reply = send(socket_path, {"op": "resume", "request_id": "  "})
        assert reply["ok"] is False
        assert "request_id" in reply
    finally:
        server.stop()


def test_agentctl_resume_cli_prints_the_ack(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    server, _, socket_path = start_server(tmp_path)
    try:
        from simple_coding_agent.agentctl import main as agentctl_main

        code = agentctl_main(["--socket", str(socket_path), "resume"])
        assert code == 0
        out = capsys.readouterr().out
        assert "completed" in out
        assert "sequence" in out
    finally:
        server.stop()


def test_agentctl_resume_reports_rejection_without_a_control_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from simple_coding_agent.agentctl import main as agentctl_main

    server, control, socket_path = start_server(tmp_path)
    try:
        assert agentctl_main(["--socket", str(socket_path), "stop"]) == 0
        capsys.readouterr()

        # Reusing the stop's request ID with the resume payload is rejected:
        # the ID is taken, so look it up from the socket state instead.
        status_reply = send(socket_path, {"op": "status"})
        stop_id = next(
            request_id
            for request_id, entry in status_reply["status"]["commands"].items()
            if entry["kind"] == "stop"
        )
        code = agentctl_main(
            ["--socket", str(socket_path), "resume", "--request-id", stop_id]
        )

        assert code == 1
        err = capsys.readouterr().err
        assert stop_id in err
        assert "rejected" in err
        assert "No control change was accepted" in err
        stored = control.get_command(stop_id)
        assert stored is not None
        assert stored.kind == "stop"
    finally:
        server.stop()


def test_failed_resume_keeps_its_request_id_for_retry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from simple_coding_agent.agentctl import main as agentctl_main

    missing = tmp_path / "state" / "agentctl.sock"
    code = agentctl_main(["--socket", str(missing), "resume", "--request-id", "req-retry-r"])

    assert code == 2
    err = capsys.readouterr().err
    assert "req-retry-r" in err
    assert "--request-id req-retry-r" in err
