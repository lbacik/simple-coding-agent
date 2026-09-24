"""Per-instance Compose access to the control CLI (issue #87).

The operator opens a shell in the intended running container and invokes
``/usr/local/bin/agentctl`` there. The socket is container-local runtime
state at ``/run/simple-coding-agent/control.sock`` (never a shared volume or
a network listener) while accepted commands persist on that instance's
``$DATA_DIR`` volume. Every CLI response displays the configured
``TARGET_REPO`` so the operator can verify the selected instance.
"""

from __future__ import annotations

import json
import os
import socket
import stat
from pathlib import Path

import pytest

from simple_coding_agent.control import ControlStore
from simple_coding_agent.control_server import ControlServer


RUNTIME_SOCKET = Path("/run/simple-coding-agent/control.sock")
RUNTIME_DIR = Path("/run/simple-coding-agent")
REPOSITORY = "owner/repo"


class InstanceControl:
    """One agent instance: its own SQLite record plus a private socket server."""

    def __init__(self, data_dir: Path, socket_path: Path, repository: str = REPOSITORY) -> None:
        self._store = ControlStore(data_dir)
        self._repository = repository
        self.has_active_attempt = False
        self._server = ControlServer(socket_path, self)
        self.socket_path = socket_path

    @property
    def repository(self) -> str:
        return self._repository

    def submit_stop(self, request_id: str):
        return self._store.submit_stop(
            request_id, has_active_attempt=self.has_active_attempt
        )

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

    def start(self) -> InstanceControl:
        self._server.start()
        return self

    def stop(self) -> None:
        self._server.stop()


def send_raw_request(socket_path: Path, payload: dict) -> dict:
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


# ---------------------------------------------------------------------------
# Endpoint location
# ---------------------------------------------------------------------------


def test_default_control_socket_is_the_container_local_runtime_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from simple_coding_agent import agentctl
    from simple_coding_agent.control_server import DEFAULT_SOCKET_PATH

    monkeypatch.delenv("AGENTCTL_SOCKET", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)

    assert DEFAULT_SOCKET_PATH == RUNTIME_SOCKET
    assert agentctl.default_socket_path() == RUNTIME_SOCKET


def test_socket_path_for_defaults_to_the_container_local_runtime_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from simple_coding_agent.control_server import socket_path_for

    monkeypatch.delenv("AGENTCTL_SOCKET", raising=False)

    assert socket_path_for(tmp_path) == RUNTIME_SOCKET


def test_socket_path_override_selects_a_test_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from simple_coding_agent import agentctl
    from simple_coding_agent.control_server import socket_path_for

    override = tmp_path / "isolated" / "control.sock"
    monkeypatch.setenv("AGENTCTL_SOCKET", str(override))

    assert socket_path_for(tmp_path) == override
    assert agentctl.default_socket_path() == override


# ---------------------------------------------------------------------------
# TARGET_REPO identity on every CLI response
# ---------------------------------------------------------------------------


def test_stop_acceptance_displays_the_configured_repository(tmp_path: Path) -> None:
    from simple_coding_agent import agentctl

    instance = InstanceControl(tmp_path, tmp_path / "run" / "control.sock").start()
    try:
        reply = agentctl.run_stop(instance.socket_path, "req-accept-1")

        assert reply["ok"] is True
        assert reply["repository"] == REPOSITORY
        rendered = agentctl.format_command(reply)
        assert f"repository: {REPOSITORY}" in rendered
        assert "req-accept-1" in rendered
    finally:
        instance.stop()


def test_command_lookup_displays_the_configured_repository(tmp_path: Path) -> None:
    from simple_coding_agent import agentctl

    instance = InstanceControl(tmp_path, tmp_path / "run" / "control.sock").start()
    try:
        agentctl.run_stop(instance.socket_path, "req-lookup-1")
        record = agentctl.run_command_lookup(instance.socket_path, "req-lookup-1")

        assert record is not None
        assert record["repository"] == REPOSITORY
        assert agentctl.format_command(record).splitlines()[0] == f"repository: {REPOSITORY}"

        raw = send_raw_request(instance.socket_path, {"op": "command", "request_id": "req-lookup-1"})
        assert raw["repository"] == REPOSITORY
    finally:
        instance.stop()


def test_status_snapshot_displays_the_configured_repository(tmp_path: Path) -> None:
    from simple_coding_agent import agentctl

    instance = InstanceControl(tmp_path, tmp_path / "run" / "control.sock").start()
    try:
        status = agentctl.run_status(instance.socket_path)

        assert status["repository"] == REPOSITORY
        assert f"repository: {REPOSITORY}" in agentctl.format_status(status).splitlines()[0]
    finally:
        instance.stop()


def test_error_replies_still_identify_the_answering_instance(tmp_path: Path) -> None:
    instance = InstanceControl(tmp_path, tmp_path / "run" / "control.sock").start()
    try:
        unknown = send_raw_request(instance.socket_path, {"op": "explode"})
        assert unknown["ok"] is False
        assert unknown["repository"] == REPOSITORY

        missing_id = send_raw_request(instance.socket_path, {"op": "command"})
        assert missing_id["ok"] is False
        assert missing_id["repository"] == REPOSITORY
    finally:
        instance.stop()


# ---------------------------------------------------------------------------
# Instance isolation and container replacement
# ---------------------------------------------------------------------------


def test_selected_instance_command_leaves_the_other_instance_unchanged(
    tmp_path: Path,
) -> None:
    from simple_coding_agent import agentctl
    from simple_coding_agent.control import IntakeState

    first = InstanceControl(
        tmp_path / "first", tmp_path / "first-run" / "control.sock", repository="owner/first"
    ).start()
    second = InstanceControl(
        tmp_path / "second", tmp_path / "second-run" / "control.sock", repository="owner/second"
    ).start()
    try:
        reply = agentctl.run_stop(first.socket_path, "req-selected-1")

        assert reply["ok"] is True
        assert reply["repository"] == "owner/first"

        lookup = agentctl.run_command_lookup(second.socket_path, "req-selected-1")
        assert lookup is None
        other_status = agentctl.run_status(second.socket_path)
        assert other_status["repository"] == "owner/second"
        assert other_status["intake"] == IntakeState.RUNNING.value
        assert other_status["commands"] == {}
    finally:
        first.stop()
        second.stop()


def test_container_replacement_recreates_the_socket_and_preserves_commands(
    tmp_path: Path,
) -> None:
    from simple_coding_agent import agentctl

    data_dir = tmp_path / "data"
    socket_path = tmp_path / "run" / "control.sock"

    instance = InstanceControl(data_dir, socket_path)
    instance.has_active_attempt = True
    instance.start()
    try:
        first = agentctl.run_stop(instance.socket_path, "req-persist-1")
        assert first["acknowledgement"] == "accepted"
    finally:
        instance.stop()
    assert not socket_path.exists()

    replacement = InstanceControl(data_dir, socket_path)
    replacement.start()
    try:
        assert socket_path.exists()
        record = agentctl.run_command_lookup(replacement.socket_path, "req-persist-1")

        assert record is not None
        assert record["sequence"] == first["sequence"]
        assert record["acknowledgement"] == "accepted"
        assert record["repository"] == REPOSITORY

        status = agentctl.run_status(replacement.socket_path)
        assert status["commands"]["req-persist-1"]["acknowledgement"] == "accepted"
    finally:
        replacement.stop()


def test_runtime_directory_and_socket_are_private_to_the_owner(tmp_path: Path) -> None:
    from simple_coding_agent.control_server import check_socket_private

    instance = InstanceControl(tmp_path, tmp_path / "run" / "control.sock").start()
    try:
        dir_mode = stat.S_IMODE(os.stat(instance.socket_path.parent).st_mode)
        socket_mode = stat.S_IMODE(os.stat(instance.socket_path).st_mode)

        assert dir_mode == 0o700, f"runtime dir must be 0700, got {oct(dir_mode)}"
        assert socket_mode == 0o600, f"socket must be 0600, got {oct(socket_mode)}"
        assert check_socket_private(instance.socket_path)
    finally:
        instance.stop()


# ---------------------------------------------------------------------------
# Packaging: no shared socket volume, no network listener, installed CLI
# ---------------------------------------------------------------------------


def test_control_channel_has_no_network_listener() -> None:
    source = (Path(__file__).parent.parent / "simple_coding_agent" / "control_server.py").read_text()

    assert "AF_UNIX" in source
    assert "AF_INET" not in source
    assert "SOCK_STREAM" in source


def test_compose_exposes_no_shared_socket_volume_or_control_port() -> None:
    import re

    compose = (Path(__file__).parent.parent / "docker-compose.yml").read_text()
    mount_lines = [
        line.strip() for line in compose.splitlines() if line.strip().startswith("- ")
    ]

    assert not any(".sock" in line for line in mount_lines)
    assert not any("/run/" in line for line in mount_lines)
    assert re.search(r"(?m)^\s*ports\s*:", compose) is None


def test_dockerfile_installs_agentctl_at_its_absolute_path() -> None:
    dockerfile = (Path(__file__).parent.parent / "Dockerfile").read_text()

    assert "/usr/local/bin/agentctl" in dockerfile


def test_dockerfile_creates_the_private_runtime_directory() -> None:
    dockerfile = (Path(__file__).parent.parent / "Dockerfile").read_text()

    assert "/run/simple-coding-agent" in dockerfile
    assert "0700" in dockerfile
    assert "chown agent:agent /run/simple-coding-agent" in dockerfile


# ---------------------------------------------------------------------------
# Operator documentation
# ---------------------------------------------------------------------------


def _onboarding() -> str:
    return (
        Path(__file__).parent.parent / "docs" / "agents" / "operator-onboarding.md"
    ).read_text()


def test_docs_tell_the_operator_to_enter_the_running_container() -> None:
    content = _onboarding()

    assert "/usr/local/bin/agentctl status" in content
    assert "TARGET_REPO" in content


def test_docs_warn_against_launching_a_second_agent_process() -> None:
    content = _onboarding().lower()

    assert "second agent process" in content
    assert "one-shot" in content
