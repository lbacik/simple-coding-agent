"""Contract adapter tests: grammar, envelope, exit codes and identity.

Every test calls the CLI entry point in-process with argv and asserts
only on the exit code and the parsed stdout envelope: stdout always
holds exactly one JSON document with a trailing newline, stderr stays
empty, and the legacy path is untouched (see
``test_agentctl_command_replies_examples`` and ``test_telemetry_examples``).
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterator
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from simple_coding_agent import agentctl, contract_adapter
from simple_coding_agent.agent_version import agent_version
from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.contract_adapter import advertised_capabilities
from simple_coding_agent.control import CommandRecord, ControlStore
from simple_coding_agent.control_server import ControlServer

REPOSITORY = "example-owner/example-repo"


class _ContractControl:
    """Minimal control surface answering the contract ``identity`` op."""

    def __init__(
        self,
        data_dir: Path,
        *,
        repository: str = REPOSITORY,
        worker_id: str = "",
        capabilities: list[str] | None = None,
        fail_identity: bool = False,
        malformed_identity: bool = False,
        not_ok_identity: bool = False,
    ) -> None:
        self._store = ControlStore(data_dir)
        self._repository = repository
        self._worker_id = worker_id
        self._capabilities = capabilities
        self._fail_identity = fail_identity
        self._malformed_identity = malformed_identity
        self._not_ok_identity = not_ok_identity

    @property
    def repository(self) -> str:
        return self._repository

    def control_identity(self) -> dict:
        if self._fail_identity:
            raise RuntimeError("identity store exploded")
        if self._malformed_identity:
            return {"backend": "simple-coding-agent"}  # type: ignore[return-value]
        capabilities = (
            self._capabilities
            if self._capabilities is not None
            else advertised_capabilities(self._worker_id)
        )
        return {
            "backend": "simple-coding-agent",
            "agent_version": agent_version(),
            "contract": {"major": 1, "minor": 0},
            "repository": self._repository,
            "mode": "online",
            "process_started_at": "2026-11-06T08:00:00Z",
            "capabilities": capabilities,
        }

    def get_command(self, request_id: str) -> CommandRecord | None:
        return self._store.get_command(request_id)


@pytest.fixture
def contract_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., tuple[ControlServer, _ContractControl]]]:
    """A fresh control socket; tests set AGENTCTL_SOCKET at the server path."""

    servers: list[ControlServer] = []

    def _start(
        name: str = "contract", **kwargs: object
    ) -> tuple[ControlServer, _ContractControl]:
        data_dir = tmp_path / name
        data_dir.mkdir(parents=True, exist_ok=True)
        socket_path = tmp_path / f"{name}.sock"
        control = _ContractControl(data_dir, **kwargs)
        server = ControlServer(socket_path, control)
        server.start()
        servers.append(server)
        monkeypatch.setenv("AGENTCTL_SOCKET", str(socket_path))
        return server, control

    yield _start

    for server in servers:
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


# ---------------------------------------------------------------------------
# Contract major
# ---------------------------------------------------------------------------


def test_contract_mismatch_before_anything_executes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTCTL_SOCKET", str(tmp_path / "missing.sock"))
    code, envelope, _, err = _run(["--contract", "2", "status"])
    assert code == 4
    assert envelope == {
        "contract": "1.0",
        "error": {
            "code": "contract_mismatch",
            "details": {"supported": [1]},
            "message": "this instance speaks contract major 1",
            "retryable": False,
        },
        "ok": False,
    }
    assert err == ""


def test_contract_mismatch_with_garbage_operation() -> None:
    code, envelope, _, err = _run(["--contract", "2", "frobnicate", "--bogus"])
    assert code == 4
    assert envelope["error"]["code"] == "contract_mismatch"
    assert envelope["error"]["details"] == {"supported": [1]}
    assert err == ""


def test_missing_major_is_usage() -> None:
    code, envelope, _, err = _run(["--contract"])
    assert code == 2
    assert envelope["contract"] == "1.0"
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "usage"
    assert envelope["error"]["retryable"] is False
    assert err == ""


# ---------------------------------------------------------------------------
# Grammar (usage, exit 2) needs no engine
# ---------------------------------------------------------------------------


def test_unknown_operation_is_usage_without_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTCTL_SOCKET", str(tmp_path / "missing.sock"))
    code, envelope, _, err = _run(["--contract", "1", "frobnicate"])
    assert code == 2
    assert envelope["error"]["code"] == "usage"
    assert err == ""


def test_missing_operation_is_usage() -> None:
    code, envelope, _, _ = _run(["--contract", "1"])
    assert code == 2
    assert envelope["error"]["code"] == "usage"


def test_offline_without_operation_is_usage() -> None:
    code, envelope, _, _ = _run(["--contract", "1", "--offline"])
    assert code == 2
    assert envelope["error"]["code"] == "usage"


@pytest.mark.parametrize(
    "argv",
    [
        ["--contract", "1", "identity", "extra"],
        ["--contract", "1", "status", "--json"],
        ["--contract", "1", "command"],
        ["--contract", "1", "command", "a", "b"],
        ["--contract", "1", "attempts"],
        ["--contract", "1", "attempts", "delete"],
        ["--contract", "1", "attempts", "show"],
        ["--contract", "1", "attempts", "export"],
        ["--contract", "1", "stop", "--after"],
        ["--contract", "1", "stop", "--bogus", "x"],
        ["--contract", "1", "stop"],
        ["--contract", "1", "stop", "--request-id", "req-1"],
        ["--contract", "1", "next"],
        ["--contract", "1", "next", "issue"],
        ["--contract", "1", "handoff"],
        ["--contract", "1", "handoff", "soon"],
        ["--contract", "1", "recovery"],
        ["--contract", "1", "recovery", "retry"],
        ["--contract", "1", "recovery", "release", "a1"],
        ["--contract", "1", "errors"],
        ["--contract", "1", "errors", "clear"],
        ["--contract", "1", "status", "--offline"],
        ["--contract", "1", "--socket", "x", "status"],
    ],
)
def test_malformed_argv_is_usage(argv: list[str]) -> None:
    code, envelope, _, err = _run(argv)
    assert code == 2, argv
    assert envelope["contract"] == "1.0", argv
    assert envelope["ok"] is False, argv
    assert envelope["error"]["code"] == "usage", argv
    assert envelope["error"]["retryable"] is False, argv
    assert err == "", argv


def test_usage_carries_request_id_when_a_mutation_carried_one() -> None:
    code, envelope, _, _ = _run(
        ["--contract", "1", "stop", "--request-id", "req-abc", "--bogus"]
    )
    assert code == 2
    assert envelope["error"]["request_id"] == "req-abc"


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_identity_reports_backend_version_contract_repository_and_mode(
    contract_server,
) -> None:
    contract_server()
    code, envelope, _, err = _run(["--contract", "1", "identity"])
    assert code == 0
    assert envelope == {
        "contract": "1.0",
        "ok": True,
        "result": {
            "backend": "simple-coding-agent",
            "agent_version": agent_version(),
            "capabilities": [
                "errors-reset",
                "next-issue",
                "operator-handoff",
                "recovery-release",
                "recovery-retry",
                "stop-after",
            ],
            "contract": {"major": 1, "minor": 0},
            "mode": "online",
            "process_started_at": "2026-11-06T08:00:00Z",
            "repository": REPOSITORY,
        },
    }
    assert err == ""


def test_identity_advertises_pool_only_with_a_worker_id(contract_server) -> None:
    contract_server("pooled", worker_id="worker-1")
    _, envelope, _, _ = _run(["--contract", "1", "identity"])
    assert "pool" in envelope["result"]["capabilities"]
    assert envelope["result"]["capabilities"] == sorted(
        envelope["result"]["capabilities"]
    )


def test_identity_without_worker_id_has_no_pool(contract_server) -> None:
    contract_server()
    _, envelope, _, _ = _run(["--contract", "1", "identity"])
    assert "pool" not in envelope["result"]["capabilities"]


def test_identity_matches_lifecycle_control_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real engine serves identity with its version, repo and pool flag."""

    from tests.fakes import FakeTracker

    from simple_coding_agent.lifecycle import AgentLifecycle

    data_dir = tmp_path / "engine"
    data_dir.mkdir(parents=True, exist_ok=True)
    socket_path = tmp_path / "engine.sock"
    lifecycle = AgentLifecycle(
        tracker=FakeTracker(worker_id="worker-9"),
        attempt_state=AttemptStateStore(data_dir),
        workspace=None,  # type: ignore[arg-type]
        profile_loader=lambda _directory: None,  # type: ignore[return-value]
        publisher=None,  # type: ignore[arg-type]
        repository=REPOSITORY,
        data_dir=data_dir,
    )
    server = ControlServer(socket_path, lifecycle)
    server.start()
    monkeypatch.setenv("AGENTCTL_SOCKET", str(socket_path))
    try:
        code, envelope, _, _ = _run(["--contract", "1", "identity"])
    finally:
        server.stop()
    assert code == 0
    result = envelope["result"]
    assert result["backend"] == "simple-coding-agent"
    assert result["agent_version"] == agent_version()
    assert result["contract"] == {"major": 1, "minor": 0}
    assert result["repository"] == REPOSITORY
    assert result["mode"] == "online"
    assert isinstance(result["process_started_at"], str)
    assert "pool" in result["capabilities"]


# ---------------------------------------------------------------------------
# Unreachable engine and engine failure
# ---------------------------------------------------------------------------


def test_unreachable_engine_answers_agent_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTCTL_SOCKET", str(tmp_path / "missing.sock"))
    code, envelope, _, err = _run(["--contract", "1", "identity"])
    assert code == 6
    assert envelope["contract"] == "1.0"
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "agent_unreachable"
    assert envelope["error"]["retryable"] is True
    assert "request_id" not in envelope["error"]
    assert err == ""


def test_unreachable_mutation_echoes_the_request_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTCTL_SOCKET", str(tmp_path / "missing.sock"))
    code, envelope, _, _ = _run(_mutation(["stop"]))
    assert code == 6
    assert envelope["error"]["code"] == "agent_unreachable"
    assert envelope["error"]["retryable"] is True
    assert envelope["error"]["request_id"] == "req-1a2b3c4d5e6f"


def test_connection_refused_answers_agent_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "blocker.sock"
    blocker.write_text("not a socket")
    monkeypatch.setenv("AGENTCTL_SOCKET", str(blocker))
    code, envelope, _, _ = _run(["--contract", "1", "identity"])
    assert code == 6
    assert envelope["error"]["code"] == "agent_unreachable"


def test_engine_failure_answers_internal(contract_server) -> None:
    contract_server("failing", fail_identity=True)
    code, envelope, _, err = _run(["--contract", "1", "identity"])
    assert code == 7
    assert envelope["error"]["code"] == "internal"
    assert envelope["error"]["retryable"] is True
    assert envelope["error"]["details"] == {"stored": False}
    assert err == ""


def test_malformed_engine_identity_answers_internal(contract_server) -> None:
    contract_server("malformed", malformed_identity=True)
    code, envelope, _, _ = _run(["--contract", "1", "identity"])
    assert code == 7
    assert envelope["error"]["code"] == "internal"


# ---------------------------------------------------------------------------
# Capability gating (unsupported, exit 3) before argument values
# ---------------------------------------------------------------------------


def test_unadvertised_capability_is_unsupported(contract_server) -> None:
    _, control = contract_server("reduced", capabilities=["next-issue"])
    code, envelope, _, err = _run(
        _mutation(["handoff", "now"], request_id="req-5e6f7a8b9c0d")
    )
    assert code == 3
    assert envelope == {
        "contract": "1.0",
        "error": {
            "code": "unsupported",
            "details": {"capability": "operator-handoff"},
            "message": "this instance does not advertise operator-handoff",
            "request_id": "req-5e6f7a8b9c0d",
            "retryable": False,
        },
        "ok": False,
    }
    assert err == ""
    assert control.get_command("req-5e6f7a8b9c0d") is None


def test_capability_gating_comes_before_argument_values(contract_server) -> None:
    contract_server("reduced-values", capabilities=["next-issue"])
    code, envelope, _, _ = _run(
        _mutation(["handoff", "now"], request_id="not a valid id!")
    )
    assert code == 3
    assert envelope["error"]["code"] == "unsupported"


def test_advertised_capability_passes_gating_to_values(contract_server) -> None:
    contract_server()
    code, envelope, _, _ = _run(
        _mutation(["handoff", "now"], request_id="not a valid id!")
    )
    assert code == 1
    assert envelope["error"]["code"] == "invalid_argument"


def test_offline_mutation_is_unsupported_without_engine_contact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTCTL_SOCKET", str(tmp_path / "missing.sock"))
    code, envelope, _, err = _run(
        ["--contract", "1", "--offline", *_mutation(["stop"])[2:]]
    )
    assert code == 3
    assert envelope["error"]["code"] == "unsupported"
    assert envelope["error"]["request_id"] == "req-1a2b3c4d5e6f"
    assert err == ""


def test_offline_read_is_unsupported_until_offline_read_lands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTCTL_SOCKET", str(tmp_path / "missing.sock"))
    code, envelope, _, _ = _run(["--contract", "1", "--offline", "status"])
    assert code == 3
    assert envelope["error"]["code"] == "unsupported"


def test_offline_unsupported_comes_before_argument_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTCTL_SOCKET", str(tmp_path / "missing.sock"))
    code, envelope, _, _ = _run(
        [
            "--contract",
            "1",
            "--offline",
            "handoff",
            "now",
            "--request-id",
            "bad id!",
            "--expect-repo",
            REPOSITORY,
        ]
    )
    assert code == 3
    assert envelope["error"]["code"] == "unsupported"


# ---------------------------------------------------------------------------
# Argument values (invalid_argument, exit 1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        _mutation(["stop"], request_id="bad id!"),
        _mutation(["stop"], request_id=""),
        _mutation(["resume"], request_id="-lead"),
        _mutation(["stop", "--after", "0"]),
        _mutation(["stop", "--after", "1000"]),
        _mutation(["stop", "--after", "many"]),
        _mutation(["stop", "--after", "-3"]),
        _mutation(["next", "issue", "0"]),
        _mutation(["next", "issue", "-2"]),
        _mutation(["next", "issue", "many"]),
        _mutation(["handoff", "now"], repository=""),
        _mutation(["recovery", "release", "a1", "--saved-at", ""]),
        ["--contract", "1", "attempts", "list", "--limit", "0"],
        ["--contract", "1", "attempts", "list", "--limit", "101"],
        ["--contract", "1", "attempts", "list", "--limit", "many"],
        ["--contract", "1", "attempts", "list", "--before", ""],
        ["--contract", "1", "command", ""],
    ],
)
def test_invalid_values_answer_invalid_argument(
    contract_server, argv: list[str]
) -> None:
    contract_server()
    code, envelope, _, err = _run(argv)
    assert code == 1, argv
    assert envelope["error"]["code"] == "invalid_argument", argv
    assert envelope["error"]["retryable"] is False, argv
    assert err == "", argv


def test_invalid_argument_echoes_the_request_id(contract_server) -> None:
    contract_server()
    code, envelope, _, _ = _run(_mutation(["stop", "--after", "0"]))
    assert code == 1
    assert envelope["error"]["request_id"] == "req-1a2b3c4d5e6f"


# ---------------------------------------------------------------------------
# Later tickets own every other operation for now
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["--contract", "1", "attempts", "export", "#41/1"],
        _mutation(["stop", "--after", "2"]),
        _mutation(["next", "issue", "43"]),
        _mutation(["handoff", "now"]),
        _mutation(["recovery", "retry", "attempt-1"]),
        _mutation(["recovery", "release", "attempt-1", "--saved-at", "somewhere"]),
        _mutation(["errors", "reset"]),
    ],
)
def test_unimplemented_operations_answer_unsupported_when_reachable(
    contract_server, argv: list[str]
) -> None:
    contract_server()
    code, envelope, _, err = _run(argv)
    assert code == 3, argv
    assert envelope["contract"] == "1.0", argv
    assert envelope["ok"] is False, argv
    assert envelope["error"]["code"] == "unsupported", argv
    assert envelope["error"]["retryable"] is False, argv
    assert err == "", argv


def test_unsupported_mutation_does_not_consume_the_request_id(
    contract_server,
) -> None:
    _, control = contract_server()
    code, _, _, _ = _run(_mutation(["handoff", "now"], request_id="req-reusable-1"))
    assert code == 3
    assert control.get_command("req-reusable-1") is None


# ---------------------------------------------------------------------------
# Adapter seams
# ---------------------------------------------------------------------------


def test_capability_sets() -> None:
    assert advertised_capabilities("") == [
        "errors-reset",
        "next-issue",
        "operator-handoff",
        "recovery-release",
        "recovery-retry",
        "stop-after",
    ]
    assert "pool" in advertised_capabilities("worker-1")
    assert advertised_capabilities("worker-1") == sorted(
        advertised_capabilities("worker-1")
    )


def test_socket_timeout_is_a_short_named_constant() -> None:
    assert contract_adapter.SOCKET_TIMEOUT_SECONDS == 10.0
