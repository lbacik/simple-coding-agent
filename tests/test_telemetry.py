"""Tracer bullet for agent telemetry (schema v1) over the control socket.

``agentctl telemetry --json`` returns a schema-v1 snapshot built without
network or git I/O and without holding up other control commands. Identity
and control are real; ``current_attempt`` and ``last_finished`` stay ``None``
and ``recent_events`` stays empty until later tickets fill them.
"""

from __future__ import annotations

import inspect
import json
import socket
import threading
import time
from pathlib import Path

import pytest

from simple_coding_agent.agentctl import main as agentctl_main
from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.control import ControlStore
from simple_coding_agent.control_server import ControlServer, send_request
from simple_coding_agent.lifecycle import AgentLifecycle
from simple_coding_agent.telemetry import (
    SNAPSHOT_SIZE_LIMIT_BYTES,
    SnapshotFailedError,
    TelemetryReporter,
    apply_size_cap,
)
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_reporter(**overrides: object) -> TelemetryReporter:
    values: dict[str, object] = {
        "repository": "owner/repo",
        "agent_version": "0.0-test",
        "sdk_version": "0.2.163-test",
        "cli_version": "2.1.286-test",
        "process_started_at": "2026-01-01T00:00:00Z",
    }
    values.update(overrides)
    return TelemetryReporter(**values)  # type: ignore[arg-type]


def make_lifecycle(
    tmp_path: Path, *, reporter: TelemetryReporter | None = None
) -> tuple[AgentLifecycle, FakeTracker, ControlStore]:
    tracker = FakeTracker()
    workspace = InMemoryWorkspace(tmp_path / "repo")
    store = ControlStore(tmp_path)
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=lambda received_claim, profile, prepared: (_ for _ in ()).throw(
            AssertionError("attempt runner must not run during telemetry")
        ),
        control_store=store,
        repository="owner/repo",
        telemetry=reporter if reporter is not None else make_reporter(),
        sleeper=lambda seconds: None,
    )
    return lifecycle, tracker, store


def start_telemetry_server(
    tmp_path: Path, socket_name: str = "telemetry.sock"
) -> tuple[ControlServer, AgentLifecycle, ControlStore, Path]:
    lifecycle, _, store = make_lifecycle(tmp_path)
    socket_path = tmp_path / "state" / socket_name
    server = ControlServer(socket_path, lifecycle)
    server.start()
    return server, lifecycle, store, socket_path


def send(socket_path: Path, payload: dict) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(10)
        client.connect(str(socket_path))
        client.sendall((json.dumps(payload) + "\n").encode())
        data = b""
        while not data.endswith(b"\n"):
            chunk = client.recv(65536)
            if not chunk:
                break
            data += chunk
    return json.loads(data.decode())


class ScriptedSocketServer:
    """Answer every connection with the same scripted bytes (or behaviour)."""

    def __init__(self, socket_path: Path, handler) -> None:
        self._socket_path = socket_path
        self._handler = handler
        self._stopped = threading.Event()
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> Path:
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._socket_path.unlink()
        except OSError:
            pass
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self._socket_path))
        listener.listen(8)
        listener.settimeout(0.2)
        self._listener = listener
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self._socket_path

    def _serve(self) -> None:
        assert self._listener is not None
        while not self._stopped.is_set():
            try:
                connection, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                with connection:
                    connection.settimeout(10)
                    self._handler(connection)
            except OSError:
                pass

    def stop(self) -> None:
        self._stopped.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=10)


def read_request_then_reply(connection: socket.socket, payload: bytes) -> None:
    data = b""
    while not data.endswith(b"\n"):
        chunk = connection.recv(65536)
        if not chunk:
            break
        data += chunk
    connection.sendall(payload)


# ---------------------------------------------------------------------------
# socket operation and snapshot shape
# ---------------------------------------------------------------------------


def test_telemetry_socket_serves_a_schema_v1_snapshot(tmp_path: Path) -> None:
    server, _, _, socket_path = start_telemetry_server(tmp_path)
    try:
        reply = send(socket_path, {"op": "telemetry"})
    finally:
        server.stop()

    assert reply["ok"] is True
    snapshot = reply["telemetry"]
    assert snapshot["schema_version"] == 1
    assert isinstance(snapshot["generated_at"], str) and snapshot["generated_at"]
    assert snapshot["sequence"] == 1
    assert snapshot["identity"] == {
        "repository": "owner/repo",
        "agent_version": "0.0-test",
        "sdk_version": "0.2.163-test",
        "cli_version": "2.1.286-test",
        "process_started_at": "2026-01-01T00:00:00Z",
    }
    assert snapshot["current_attempt"] is None
    assert snapshot["last_finished"] is None
    assert snapshot["recent_events"] == []
    assert snapshot["truncated"] is False

    control = snapshot["control"]
    assert control["repository"] == "owner/repo"
    assert control["agent_version"] == snapshot["identity"]["agent_version"] or isinstance(
        control["agent_version"], str
    )
    assert isinstance(control["control_observed_at"], str) and control["control_observed_at"]
    assert control["active_attempt"] is None
    assert control["recovery"] is None


def test_telemetry_sequence_increments_per_snapshot(tmp_path: Path) -> None:
    server, _, _, socket_path = start_telemetry_server(tmp_path)
    try:
        first = send(socket_path, {"op": "telemetry"})
        second = send(socket_path, {"op": "telemetry"})
    finally:
        server.stop()

    assert first["telemetry"]["sequence"] == 1
    assert second["telemetry"]["sequence"] == 2


def test_telemetry_failed_build_does_not_consume_sequence() -> None:
    attempts = {"count": 0}

    class FlakyControl:
        def telemetry_control_snapshot(self) -> dict:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise SnapshotFailedError("no control section has been built yet")
            return {"control_observed_at": "2026-01-01T00:00:00Z"}

    reporter = make_reporter()
    with pytest.raises(SnapshotFailedError):
        reporter.snapshot(FlakyControl())
    snapshot = reporter.snapshot(FlakyControl())

    assert snapshot["sequence"] == 1


def test_telemetry_snapshot_matches_the_idle_example_shape(tmp_path: Path) -> None:
    example = json.loads(
        (Path(__file__).parent.parent / "examples" / "telemetry"
         / "snapshot-idle-with-last-finished.json").read_text()
    )
    server, _, _, socket_path = start_telemetry_server(tmp_path)
    try:
        reply = send(socket_path, {"op": "telemetry"})
    finally:
        server.stop()

    snapshot = reply["telemetry"]
    assert set(snapshot) == set(example)
    assert set(snapshot["identity"]) == set(example["identity"])
    assert set(snapshot["control"]) == set(example["control"])
    # Tracer values diverge from the populated example on purpose: the writer
    # side that fills these lands in later tickets.
    assert snapshot["current_attempt"] is None
    assert snapshot["last_finished"] is None
    assert snapshot["recent_events"] == []
    assert snapshot["truncated"] is False


def test_telemetry_serves_cached_control_while_the_lock_is_busy(tmp_path: Path) -> None:
    server, _, store, socket_path = start_telemetry_server(tmp_path)
    try:
        first = send(socket_path, {"op": "telemetry"})
        observed = first["telemetry"]["control"]["control_observed_at"]

        # Hold the control lock on this thread past the 1 s telemetry wait so
        # the listener thread must reuse the last built control section.
        store.lock.acquire()
        results: list[dict] = []
        worker = threading.Thread(
            target=lambda: results.append(send(socket_path, {"op": "telemetry"}))
        )
        try:
            worker.start()
            time.sleep(1.5)
        finally:
            store.lock.release()
        worker.join(timeout=30)
        assert not worker.is_alive()
    finally:
        server.stop()

    assert len(results) == 1
    assert results[0]["ok"] is True
    assert results[0]["telemetry"]["control"]["control_observed_at"] == observed


def test_telemetry_without_a_cached_section_fails_while_the_lock_is_busy(
    tmp_path: Path,
) -> None:
    server, _, store, socket_path = start_telemetry_server(tmp_path)
    try:
        store.lock.acquire()
        results: list[dict] = []
        worker = threading.Thread(
            target=lambda: results.append(send(socket_path, {"op": "telemetry"}))
        )
        try:
            worker.start()
            time.sleep(1.5)
        finally:
            store.lock.release()
        worker.join(timeout=30)
        assert not worker.is_alive()
    finally:
        server.stop()

    assert len(results) == 1
    assert results[0]["ok"] is False
    assert "snapshot failed" in results[0]["error"].lower()


# ---------------------------------------------------------------------------
# no tracker or git I/O on the telemetry path
# ---------------------------------------------------------------------------


class _ExplodingAttribute:
    """Fail any method call; plain attribute reads still work."""

    def __getattr__(self, name: str):
        raise AssertionError(f"{name} must not be called on the telemetry path")


class ExplodingTracker(_ExplodingAttribute):
    pass


class ExplodingWorkspace(_ExplodingAttribute):
    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.working_directory = root


class ExplodingPublisher(_ExplodingAttribute):
    pass


def make_exploding_lifecycle(tmp_path: Path) -> AgentLifecycle:
    store = ControlStore(tmp_path)
    return AgentLifecycle(
        tracker=ExplodingTracker(),  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(tmp_path),
        workspace=ExplodingWorkspace(tmp_path / "repo"),  # type: ignore[arg-type]
        profile_loader=lambda _: (_ for _ in ()).throw(
            AssertionError("profile loader must not run during telemetry")
        ),
        publisher=ExplodingPublisher(),  # type: ignore[arg-type]
        control_store=store,
        repository="owner/repo",
        telemetry=make_reporter(),
        sleeper=lambda seconds: None,
    )


def test_telemetry_performs_no_tracker_or_workspace_calls_when_idle(tmp_path: Path) -> None:
    lifecycle = make_exploding_lifecycle(tmp_path)

    snapshot = lifecycle.telemetry_snapshot()

    assert snapshot["schema_version"] == 1
    assert snapshot["control"]["recovery"] is None
    assert snapshot["control"]["active_attempt"] is None


def test_telemetry_reports_a_retained_hold_without_lookups(tmp_path: Path) -> None:
    lifecycle = make_exploding_lifecycle(tmp_path)
    state = AttemptStateStore(tmp_path)
    checkpoint = state.start(issue_number=24, branch="agent/issue-24")

    snapshot = lifecycle.telemetry_snapshot()

    control = snapshot["control"]
    assert control["active_attempt"]["issue_number"] == 24
    recovery = control["recovery"]
    assert recovery is not None
    assert recovery["attempt_id"] == checkpoint.started_at
    assert recovery["issue_number"] == 24
    assert recovery["checkpoint"] is True
    # Publication fields that would need a GitHub lookup stay unknown.
    assert recovery["publication"] == {
        "phase": checkpoint.phase.value,
        "comment_confirmed": None,
        "pull_request": None,
    }
    assert "issue #24" in recovery["hold_reason"]
    assert isinstance(recovery["next_action"], str) and recovery["next_action"]


# ---------------------------------------------------------------------------
# agentctl telemetry command
# ---------------------------------------------------------------------------


def test_agentctl_telemetry_prints_the_bare_snapshot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    server, _, _, socket_path = start_telemetry_server(tmp_path)
    try:
        code = agentctl_main(["--socket", str(socket_path), "telemetry", "--json"])
    finally:
        server.stop()

    assert code == 0
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err
    snapshot = json.loads(captured.out)
    assert snapshot["schema_version"] == 1
    assert snapshot["control"]["repository"] == "owner/repo"
    assert snapshot["current_attempt"] is None


def test_agentctl_telemetry_requires_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        agentctl_main(["--socket", str(tmp_path / "missing.sock"), "telemetry"])

    assert excinfo.value.code == 2
    assert capsys.readouterr().out == ""


def test_agentctl_telemetry_reports_agent_unreachable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = agentctl_main(
        ["--socket", str(tmp_path / "missing.sock"), "telemetry", "--json"]
    )

    assert code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err
    reply = json.loads(captured.out)
    assert reply == {
        "ok": False,
        "error": {"code": "agent_unreachable", "message": reply["error"]["message"]},
    }
    assert reply["error"]["message"]


def test_agentctl_telemetry_reports_unsupported_operation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class OldControl:
        repository = "owner/repo"

    socket_path = tmp_path / "state" / "old.sock"
    server = ControlServer(socket_path, OldControl())  # type: ignore[arg-type]
    server.start()
    try:
        code = agentctl_main(["--socket", str(socket_path), "telemetry", "--json"])
    finally:
        server.stop()

    assert code == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err
    reply = json.loads(captured.out)
    assert reply["ok"] is False
    assert reply["error"]["code"] == "unsupported_operation"
    assert reply["error"]["message"]


def test_agentctl_telemetry_reports_snapshot_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class FailingControl:
        repository = "owner/repo"

        def __init__(self) -> None:
            self.events: list[tuple] = []

        def telemetry_snapshot(self) -> dict:
            raise RuntimeError("boom-build")

        def _log(self, event: str, detail: str = "", level: str = "INFO") -> None:
            self.events.append((event, detail, level))

    control = FailingControl()
    socket_path = tmp_path / "state" / "failing.sock"
    server = ControlServer(socket_path, control, event_log=control._log)  # noqa: SLF001
    server.start()
    try:
        code = agentctl_main(["--socket", str(socket_path), "telemetry", "--json"])
        first = capsys.readouterr()
        first_events = list(control.events)
        # A second immediate failure is rate-limited to one telemetry_failed event.
        second = agentctl_main(["--socket", str(socket_path), "telemetry", "--json"])
        second_events = list(control.events)
        assert server._thread is not None and server._thread.is_alive()  # noqa: SLF001
    finally:
        server.stop()

    assert code == 1
    assert second == 1
    assert "Traceback" not in first.out + first.err
    reply = json.loads(first.out)
    assert reply["ok"] is False
    assert reply["error"]["code"] == "snapshot_failed"
    assert [event for event, _, _ in first_events] == ["telemetry_failed"]
    assert second_events == first_events


def test_telemetry_non_serialisable_snapshot_answers_snapshot_failed(tmp_path: Path) -> None:
    class UnserialisableControl:
        repository = "owner/repo"

        def __init__(self) -> None:
            self.events: list[tuple] = []

        def telemetry_snapshot(self) -> dict:
            return {"not": "a snapshot", "unserialisable": object()}

        def _log(self, event: str, detail: str = "", level: str = "INFO") -> None:
            self.events.append((event, detail, level))

    control = UnserialisableControl()
    socket_path = tmp_path / "state" / "unserialisable.sock"
    server = ControlServer(socket_path, control, event_log=control._log)  # noqa: SLF001
    server.start()
    try:
        reply = send(socket_path, {"op": "telemetry"})
        assert server._thread is not None and server._thread.is_alive()  # noqa: SLF001
    finally:
        server.stop()

    assert reply["ok"] is False
    assert "snapshot failed" in reply["error"].lower()
    assert [event for event, _, _ in control.events] == ["telemetry_failed"]


@pytest.mark.parametrize(
    "payload",
    [
        b"not json\n",
        b'{"ok": true}\n',
        b'{"ok": true, "telemetry": [1, 2]}\n',
        b'{"ok": false, "repository": "owner/repo",'
        b' "error": "Status is unavailable: boom"}\n',
    ],
    ids=["unparsable", "missing-snapshot", "malformed-snapshot", "agent-error"],
)
def test_agentctl_telemetry_maps_bad_replies_to_snapshot_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], payload: bytes
) -> None:
    socket_path = tmp_path / "state" / "scripted.sock"
    server = ScriptedSocketServer(
        socket_path, lambda connection: read_request_then_reply(connection, payload)
    )
    server.start()
    try:
        code = agentctl_main(["--socket", str(socket_path), "telemetry", "--json"])
    finally:
        server.stop()

    assert code == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err
    reply = json.loads(captured.out)
    assert reply["ok"] is False
    assert reply["error"]["code"] == "snapshot_failed"


def test_agentctl_telemetry_maps_an_oversized_reply_to_snapshot_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = b'{"ok": true, "pad": "' + b"x" * (2**20) + b'"}\n'
    socket_path = tmp_path / "state" / "oversized.sock"
    server = ScriptedSocketServer(
        socket_path, lambda connection: read_request_then_reply(connection, payload)
    )
    server.start()
    try:
        code = agentctl_main(["--socket", str(socket_path), "telemetry", "--json"])
    finally:
        server.stop()

    assert code == 1
    reply = json.loads(capsys.readouterr().out)
    assert reply["error"]["code"] == "snapshot_failed"


def test_agentctl_telemetry_times_out_after_three_seconds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def black_hole(connection: socket.socket) -> None:
        data = b""
        while not data.endswith(b"\n"):
            chunk = connection.recv(65536)
            if not chunk:
                return
            data += chunk
        time.sleep(30)

    socket_path = tmp_path / "state" / "blackhole.sock"
    server = ScriptedSocketServer(socket_path, black_hole)
    server.start()
    try:
        started = time.monotonic()
        code = agentctl_main(["--socket", str(socket_path), "telemetry", "--json"])
        elapsed = time.monotonic() - started
    finally:
        server.stop()

    assert code == 2
    reply = json.loads(capsys.readouterr().out)
    assert reply["error"]["code"] == "agent_unreachable"
    assert elapsed < 8


def test_telemetry_uses_a_short_timeout_while_other_commands_keep_theirs() -> None:
    import simple_coding_agent.telemetry as telemetry_module

    assert telemetry_module.TELEMETRY_TIMEOUT_SECONDS == 3.0
    assert (
        inspect.signature(send_request).parameters["timeout"].default == 10.0
    )


# ---------------------------------------------------------------------------
# size bound
# ---------------------------------------------------------------------------


def test_size_cap_leaves_small_snapshots_untouched() -> None:
    snapshot = {
        "recent_events": [{"text": "small"}],
        "current_attempt": None,
        "truncated": False,
    }

    assert apply_size_cap(snapshot) is snapshot
    assert snapshot["truncated"] is False
    assert snapshot["recent_events"] == [{"text": "small"}]


def test_size_cap_drops_oldest_events_first() -> None:
    events = [
        {"timestamp": f"2026-01-01T00:00:{index:02d}Z", "text": f"event-{index:04d}-" + "x" * 1000}
        for index in range(300)
    ]
    snapshot: dict = {
        "recent_events": events,
        "current_attempt": None,
        "truncated": False,
    }

    trimmed = apply_size_cap(snapshot)

    assert trimmed["truncated"] is True
    assert len(json.dumps(trimmed).encode("utf-8")) <= SNAPSHOT_SIZE_LIMIT_BYTES
    assert trimmed["recent_events"]
    first_index = int(trimmed["recent_events"][0]["text"].split("-")[1])
    assert first_index > 0
    kept = [int(event["text"].split("-")[1]) for event in trimmed["recent_events"]]
    assert kept == sorted(kept)
    assert kept[-1] == 299


def test_size_cap_drops_subagents_after_events_are_empty() -> None:
    subagents = [
        {"id": f"agent-{index:03d}", "pad": "y" * 5000} for index in range(60)
    ]
    snapshot: dict = {
        "recent_events": [],
        "current_attempt": {"context": {"subagents": subagents}},
        "truncated": False,
    }

    trimmed = apply_size_cap(snapshot)

    assert trimmed["truncated"] is True
    assert len(json.dumps(trimmed).encode("utf-8")) <= SNAPSHOT_SIZE_LIMIT_BYTES
    remaining = trimmed["current_attempt"]["context"]["subagents"]
    assert remaining
    assert remaining[0]["id"] != "agent-000"
    assert remaining[-1]["id"] == "agent-059"
