"""Control socket stays responsive while a submit waits on the control lock."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.control import ControlStore
from simple_coding_agent.control_server import ControlServer, send_request
from simple_coding_agent.telemetry import TELEMETRY_TIMEOUT_SECONDS
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace


class _StubTelemetry:
    """Minimal reporter exercising the bounded control-lock path."""

    def snapshot(self, control):  # type: ignore[no-untyped-def]
        return {"control": control.telemetry_control_snapshot()}


def _make_lifecycle(tmp_path: Path, **kwargs):  # type: ignore[no-untyped-def]
    from simple_coding_agent.lifecycle import AgentLifecycle

    tracker = FakeTracker(next_claim=None)
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
    store = ControlStore(tmp_path)
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=lambda claim, profile, prepared: (_ for _ in ()).throw(
            AssertionError("no attempt expected")
        ),
        control_store=store,
        repository="owner/repo",
        telemetry=_StubTelemetry(),  # type: ignore[arg-type]
        sleeper=lambda seconds: None,
        **kwargs,
    )
    return lifecycle, store


def test_telemetry_and_command_answer_while_submit_waits_on_control_lock(
    tmp_path: Path,
) -> None:
    lifecycle, _ = _make_lifecycle(tmp_path)
    socket_path = tmp_path / "state" / "agentctl.sock"
    server = ControlServer(socket_path, lifecycle)  # type: ignore[arg-type]
    server.start()
    try:
        # Prime the cached telemetry control section so the bounded
        # fallback has something to serve while the lock is held.
        primed = send_request(
            socket_path, {"op": "telemetry"}, timeout=TELEMETRY_TIMEOUT_SECONDS
        )
        assert primed["ok"] is True

        lock = lifecycle._control_lock  # noqa: SLF001 - hold it as the intake claim does
        lock.acquire()
        try:
            submit_result: dict = {}
            submit_error: list[Exception] = []

            def do_submit() -> None:
                try:
                    submit_result.update(
                        send_request(
                            socket_path,
                            {"op": "stop", "request_id": "req-waiting"},
                            timeout=30,
                        )
                    )
                except Exception as error:  # pragma: no cover - diagnostic
                    submit_error.append(error)

            submitter = threading.Thread(target=do_submit, daemon=True)
            submitter.start()
            # Give the submit a moment to arrive and block on the lock.
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                # The listener must have accepted the connection even
                # though the submit is still waiting: it answers a
                # second connection instead of queueing behind the first.
                probe = threading.Thread(
                    target=lambda: send_request(
                        socket_path,
                        {"op": "command", "request_id": "req-probe"},
                        timeout=5,
                    ),
                    daemon=True,
                )
                probe.start()
                probe.join(timeout=5)
                if not probe.is_alive():
                    break
                time.sleep(0.05)
            assert submitter.is_alive(), "submit should still be waiting on the lock"

            start = time.monotonic()
            telemetry = send_request(
                socket_path, {"op": "telemetry"}, timeout=TELEMETRY_TIMEOUT_SECONDS
            )
            telemetry_elapsed = time.monotonic() - start
            assert telemetry["ok"] is True
            assert telemetry_elapsed < TELEMETRY_TIMEOUT_SECONDS

            start = time.monotonic()
            lookup = send_request(
                socket_path, {"op": "command", "request_id": "req-waiting"}
            )
            lookup_elapsed = time.monotonic() - start
            assert lookup["ok"] is True
            # The submit has not committed yet: the last committed state
            # correctly reports it as not recorded, without blocking.
            assert lookup["command"] is None
            assert lookup_elapsed < 10
            assert submitter.is_alive()
        finally:
            lock.release()

        submitter.join(timeout=30)
        assert not submit_error
        assert submit_result.get("ok") is True
        assert submit_result.get("request_id") == "req-waiting"

        committed = send_request(
            socket_path, {"op": "command", "request_id": "req-waiting"}
        )
        assert committed["ok"] is True
        assert committed["command"] is not None
        assert committed["command"]["request_id"] == "req-waiting"
    finally:
        server.stop()


def test_stop_returns_promptly_while_a_connection_is_in_flight(
    tmp_path: Path,
) -> None:
    from simple_coding_agent.control_server import ControlServer

    entered = threading.Event()
    release = threading.Event()

    class SlowControl:
        repository = "owner/repo"

        def submit_stop(self, request_id: str, **kwargs):  # type: ignore[no-untyped-def]
            entered.set()
            assert release.wait(timeout=30)
            return {"request_id": request_id}

        def get_command(self, request_id: str):  # type: ignore[no-untyped-def]
            return None

        def control_status(self, repository: str | None = None):  # type: ignore[no-untyped-def]
            return {}

        def telemetry_snapshot(self):  # type: ignore[no-untyped-def]
            return {}

    socket_path = tmp_path / "state" / "agentctl.sock"
    server = ControlServer(socket_path, SlowControl())  # type: ignore[arg-type]
    server.start()
    try:
        worker = threading.Thread(
            target=lambda: send_request(
                socket_path, {"op": "stop", "request_id": "req-slow"}, timeout=30
            ),
            daemon=True,
        )
        worker.start()
        assert entered.wait(timeout=30)

        start = time.monotonic()
        server.stop()
        elapsed = time.monotonic() - start
        assert elapsed < 10
        assert not socket_path.exists()
    finally:
        release.set()
        try:
            server.stop()
        except Exception:
            pass
