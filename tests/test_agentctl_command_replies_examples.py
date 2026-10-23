"""Conformance tests for the synthetic agentctl command-reply examples.

Each file under ``examples/agentctl/`` pins the real CLI output for one
contract case (``{"argv", "exit_code", "stdout", "stderr"}``). Every test
here reproduces its example from the real ``agentctl`` CLI against a
fresh test control socket, so the pinned files fail the suite when the
CLI output drifts. The repository is invented (``example-owner/...``)
and every request ID is fixed, so committed commands always carry
sequence 1.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from simple_coding_agent import agentctl
from simple_coding_agent.control import ControlStore, build_status
from simple_coding_agent.control_server import ControlServer

ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = ROOT / "examples" / "agentctl"

REPOSITORY = "example-owner/example-repo"

EXPECTED_EXAMPLES = {
    "accepted.json",
    "completed.json",
    "rejected.json",
    "not-submitted.json",
    "usage-error.json",
    "command-found.json",
    "command-not-found.json",
}


class _ExampleControl:
    """Minimal lifecycle-facing control surface backed by a fresh store."""

    def __init__(
        self,
        data_dir: Path,
        repository: str = REPOSITORY,
        has_active_attempt: bool = False,
    ) -> None:
        self._store = ControlStore(data_dir)
        self._repository = repository
        self.has_active_attempt = has_active_attempt

    @property
    def repository(self) -> str:
        return self._repository

    def submit_stop(self, request_id: str):
        return self._store.submit_stop(
            request_id, has_active_attempt=self.has_active_attempt
        )

    def submit_stop_after(self, request_id: str, after: object):
        return self._store.submit_stop_after(
            request_id, after, has_active_attempt=self.has_active_attempt
        )

    def get_command(self, request_id: str):
        return self._store.get_command(request_id)

    def control_status(self, repository: str | None = None):
        return build_status(
            repository=repository or self._repository,
            intake=self._store.intake_state(),
            recovering=False,
            active_attempt=None,
            pending_command_id=self._store.pending_command_id(),
            get_command=self._store.get_command,
            recent_commands=self._store.recent_commands,
        )


def _load(name: str) -> dict:
    path = EXAMPLES_DIR / name
    assert path.is_file(), f"agentctl example {name} must exist"
    return json.loads(path.read_text())


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    """Run the real CLI, capturing both streams; usage errors exit via SystemExit."""

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = agentctl.main(argv)
        except SystemExit as exit_info:
            assert isinstance(exit_info.code, int)
            code = exit_info.code
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def example_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[object]:
    """A fresh test control socket answering as the example repository."""

    monkeypatch.setenv("COLUMNS", "80")
    servers: list[ControlServer] = []

    def _start(name: str, **kwargs) -> tuple[ControlServer, _ExampleControl, Path]:
        data_dir = tmp_path / name
        data_dir.mkdir(parents=True, exist_ok=True)
        socket_path = tmp_path / f"{name}.sock"
        control = _ExampleControl(data_dir, **kwargs)
        server = ControlServer(socket_path, control)
        server.start()
        servers.append(server)
        return server, control, socket_path

    yield _start  # type: ignore[misc]

    for server in servers:
        server.stop()


def _replay(
    example_server,  # noqa: ANN001, ANN202
    name: str,
    *,
    pre_stop: bool = False,
    has_active_attempt: bool = False,
) -> None:
    expected = _load(name)
    argv_template: list[str] = expected["argv"]
    if "{{SOCKET}}" in argv_template:
        _, _, socket_path = example_server(
            name, has_active_attempt=has_active_attempt
        )
        if pre_stop:
            first_id = f"req-example-first-{name}"
            assert agentctl.run_stop(socket_path, first_id)["ok"] is True
        argv = [
            str(socket_path) if token == "{{SOCKET}}" else token
            for token in argv_template
        ]
    else:
        argv = list(argv_template)
        if name == "not-submitted.json":
            missing = Path(argv[argv.index("--socket") + 1])
            missing.unlink(missing_ok=True)
    code, out, err = _run_cli(argv)
    assert code == expected["exit_code"], f"{name}: exit code drifted"
    assert out == expected["stdout"], f"{name}: stdout drifted"
    assert err == expected["stderr"], f"{name}: stderr drifted"


def test_expected_examples_exist() -> None:
    found = {path.name for path in EXAMPLES_DIR.glob("*.json")}
    assert EXPECTED_EXAMPLES <= found, (
        f"missing agentctl examples: {sorted(EXPECTED_EXAMPLES - found)}"
    )


def test_accepted_example_matches_the_real_cli(example_server) -> None:  # noqa: ANN001
    _replay(example_server, "accepted.json", has_active_attempt=True)


def test_completed_example_matches_the_real_cli(example_server) -> None:  # noqa: ANN001
    _replay(example_server, "completed.json")


def test_rejected_example_matches_the_real_cli(example_server) -> None:  # noqa: ANN001
    _replay(example_server, "rejected.json", pre_stop=True)


def test_not_submitted_example_matches_the_real_cli(example_server) -> None:  # noqa: ANN001
    _replay(example_server, "not-submitted.json")


def test_usage_error_example_matches_the_real_cli(example_server) -> None:  # noqa: ANN001
    _replay(example_server, "usage-error.json")


def test_command_found_example_matches_the_real_cli(example_server) -> None:  # noqa: ANN001
    _, control, socket_path = example_server("command-found.json")
    assert agentctl.run_stop(socket_path, "req-example-record")["ok"] is True
    expected = _load("command-found.json")
    argv = [
        str(socket_path) if token == "{{SOCKET}}" else token
        for token in expected["argv"]
    ]
    code, out, err = _run_cli(argv)
    assert code == expected["exit_code"]
    assert out == expected["stdout"]
    assert err == expected["stderr"]
    assert control.get_command("req-example-record") is not None


def test_command_not_found_example_matches_the_real_cli(example_server) -> None:  # noqa: ANN001
    _replay(example_server, "command-not-found.json")


# ---------------------------------------------------------------------------
# Contract unit tests: single-line values, repository line, one suffix
# ---------------------------------------------------------------------------


def test_effect_with_an_embedded_newline_renders_as_one_line() -> None:
    reply = {
        "repository": REPOSITORY,
        "request_id": "req-multiline",
        "sequence": 3,
        "acknowledgement": "accepted",
        "detail": "first line\nsecond line\r\nthird line",
    }
    rendered = agentctl.format_command(reply)
    lines = rendered.splitlines()
    assert len(lines) == 5
    assert lines[4] == "effect: first line second line third line"
    assert "\n" not in lines[4]


def test_every_stdout_value_is_guaranteed_single_line() -> None:
    reply = {
        "repository": "example-owner/line\nbreak",
        "request_id": "req-multi\nline",
        "sequence": "1\n2",
        "acknowledgement": "accept\red",
        "detail": "a\nb\rc\nd",
    }
    for line in agentctl.format_command(reply).splitlines():
        assert "\n" not in line and "\r" not in line
    assert len(agentctl.format_command(reply).splitlines()) == 5


def test_rejection_names_the_repository_first_and_the_suffix_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def submit(_socket_path: Path, _request_id: str) -> dict:
        return {
            "ok": False,
            "repository": REPOSITORY,
            "error": "stop is rejected without an active attempt. No control change was accepted.",
            "request_id": "req-dupe-1",
        }

    code = agentctl.submit_mutating(tmp_path / "control.sock", "stop", "req-dupe-1", submit)

    assert code == 1
    err = capsys.readouterr().err
    lines = err.splitlines()
    assert lines[0] == f"repository: {REPOSITORY}"
    assert "req-dupe-1 rejected" in lines[1]
    assert err.count("No control change was accepted.") == 1


def test_rejection_without_a_stored_suffix_gains_exactly_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def submit(_socket_path: Path, _request_id: str) -> dict:
        return {"ok": False, "repository": REPOSITORY, "error": "boom"}

    code = agentctl.submit_mutating(tmp_path / "control.sock", "stop", "req-bare-1", submit)

    assert code == 1
    err = capsys.readouterr().err
    assert err.splitlines()[0] == f"repository: {REPOSITORY}"
    assert err.count("No control change was accepted.") == 1


def test_rejection_without_a_reply_prints_no_repository_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def submit(_socket_path: Path, _request_id: str) -> dict:
        return {"ok": False, "error": "boom"}

    code = agentctl.submit_mutating(tmp_path / "control.sock", "stop", "req-norepo-1", submit)

    assert code == 1
    err = capsys.readouterr().err
    assert "repository:" not in err
    assert "req-norepo-1 rejected" in err


def test_local_stop_after_validation_names_the_request_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from simple_coding_agent.control_server import ControlServer

    control = _ExampleControl(tmp_path / "data")
    socket_path = tmp_path / "control.sock"
    server = ControlServer(socket_path, control)
    server.start()
    try:
        code = agentctl.main(
            [
                "--socket",
                str(socket_path),
                "stop",
                "--after",
                "many",
                "--request-id",
                "req-local-1",
            ]
        )
    finally:
        server.stop()

    assert code == 1
    err = capsys.readouterr().err
    assert "req-local-1 rejected" in err
    assert "repository:" not in err
