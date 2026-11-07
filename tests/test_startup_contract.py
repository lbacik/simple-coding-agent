"""Operator contract startup: volume lock, starting/recovering and Halt.

An operator can watch an instance go through startup and recovery, and
learn why it stopped without reading its logs. Every test drives the
real lifecycle (shared fakes) and asserts at the CLI seam (exit code
and parsed envelope) or on the durable state directory; where the CLI
seam cannot reach (gated-claim gating, reconciliation flags, halt
writes) tests use the documented startup seams (``mark_starting`` /
``mark_started``) with the same ``noqa: SLF001`` private access the
contract-status tests already use.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

import pytest

from simple_coding_agent import agentctl
from simple_coding_agent import __main__ as entrypoint
from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.config import ConfigurationError
from simple_coding_agent.control import ControlStore, IntakeState
from simple_coding_agent.control_server import ControlServer
from simple_coding_agent.github_tracker import TrackerIssue
from simple_coding_agent.halt import (
    HALT_CONSECUTIVE_ERROR_LIMIT,
    HALT_CONTROL_ENDPOINT_UNAVAILABLE,
    HALT_PROVENANCE_FAILED,
    HaltStore,
)
from simple_coding_agent.lifecycle import AgentLifecycle
from simple_coding_agent.observability import JsonEventLogger
from simple_coding_agent.volume_lock import VolumeLock, VolumeLockError
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace

REPOSITORY = "owner/repo"


def _issue(number: int) -> TrackerIssue:
    return TrackerIssue(
        id=f"issue-{number}",
        number=number,
        title="Implement lifecycle",
        body="Connect boundaries.",
        created_at=datetime(2026, 9, 20, tzinfo=UTC),
        state="OPEN",
        labels=frozenset({"ready-for-agent"}),
        assignee_logins=(),
        blocked_by=0,
        author_login="reporter",
    )


def make_lifecycle(
    tmp_path: Path,
    tracker: FakeTracker | None = None,
    *,
    starting: bool = False,
    startup_check: str | None = None,
    halt_store: HaltStore | None = None,
) -> tuple[AgentLifecycle, ControlStore, AttemptStateStore]:
    """Wire the real lifecycle against the shared fakes with control enabled."""

    store = ControlStore(tmp_path)
    state = AttemptStateStore(tmp_path)
    lifecycle = AgentLifecycle(
        tracker=tracker if tracker is not None else FakeTracker(),
        attempt_state=state,
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=lambda claim, profile, prepared: (_ for _ in ()).throw(
            AssertionError("attempt runner must not run during startup tests")
        ),
        control_store=store,
        repository=REPOSITORY,
        telemetry=None,
        sleeper=lambda seconds: None,
        starting=starting,
        startup_check=startup_check,
        halt_store=halt_store,
    )
    return lifecycle, store, state


@pytest.fixture
def startup_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[ControlServer, AgentLifecycle, ControlStore, AttemptStateStore]]:
    """A control socket serving a lifecycle stuck in ``starting``."""

    lifecycle, store, state = make_lifecycle(tmp_path, starting=True)
    socket_path = tmp_path / "state" / "startup.sock"
    server = ControlServer(socket_path, lifecycle)  # type: ignore[arg-type]
    server.start()
    monkeypatch.setenv("AGENTCTL_SOCKET", str(socket_path))
    try:
        yield server, lifecycle, store, state
    finally:
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


def _status() -> tuple[int, dict]:
    code, envelope, _, err = _run(["--contract", "1", "status"])
    assert err == ""
    return code, envelope


def _result(envelope: dict) -> dict:
    assert envelope["contract"] == "1.0"
    assert envelope["ok"] is True
    return envelope["result"]


# ---------------------------------------------------------------------------
# Volume lock: a second engine on the same state directory exits before
# serving or claiming.
# ---------------------------------------------------------------------------


def test_second_engine_fails_to_acquire_the_lock(tmp_path: Path) -> None:
    first = VolumeLock(tmp_path)
    first.acquire()
    try:
        with pytest.raises(VolumeLockError):
            VolumeLock(tmp_path).acquire()
    finally:
        first.release()


def test_volume_lock_release_allows_reacquire(tmp_path: Path) -> None:
    first = VolumeLock(tmp_path)
    first.acquire()
    first.release()
    second = VolumeLock(tmp_path)
    second.acquire()
    try:
        assert second.held
    finally:
        second.release()


def test_acquire_volume_lock_exits_nonzero_before_serving(tmp_path: Path) -> None:
    first = VolumeLock(tmp_path)
    first.acquire()
    try:
        with pytest.raises(SystemExit) as exc_info:
            entrypoint._acquire_volume_lock(tmp_path)
        assert exc_info.value.code != 0
    finally:
        first.release()


def test_lock_file_lives_in_the_state_directory(tmp_path: Path) -> None:
    lock = VolumeLock(tmp_path)
    assert lock.path.parent == tmp_path / "state"
    assert lock.path.name == "volume.lock"


# ---------------------------------------------------------------------------
# Starting: status answers ``starting`` with ``startup_check`` naming
# provenance verification while it is held.
# ---------------------------------------------------------------------------


def test_status_answers_starting_with_provenance_check(startup_server) -> None:
    _, _, _, _ = startup_server
    code, envelope = _status()
    assert code == 0
    result = _result(envelope)
    assert result["instance"]["state"] == "starting"
    assert result["instance"]["startup_check"] == "provenance"


def test_mark_started_clears_starting_and_startup_check(startup_server) -> None:
    _, lifecycle, _, _ = startup_server
    lifecycle.mark_started()
    code, envelope = _status()
    assert code == 0
    result = _result(envelope)
    assert result["instance"]["state"] == "idle"
    assert result["instance"]["startup_check"] is None


def test_starting_wins_over_recovering_and_active(tmp_path: Path) -> None:
    lifecycle, _, state = make_lifecycle(tmp_path, starting=True)
    state.start(issue_number=42, branch="agent/issue-42", issue_title="Add CSV export")
    lifecycle._recovering = True  # noqa: SLF001 - simulate reconciliation in flight
    snapshot = lifecycle.contract_status_snapshot()
    assert snapshot["instance"]["state"] == "starting"
    assert snapshot["instance"]["startup_check"] == "provenance"


# ---------------------------------------------------------------------------
# Recovering: status answers ``recovering`` during startup reconciliation
# and while intake is held for recovery.
# ---------------------------------------------------------------------------


def test_status_answers_recovering_while_reconciliation_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifecycle, _, _ = make_lifecycle(tmp_path)
    lifecycle._recovering = True  # noqa: SLF001 - startup reconciliation in flight
    socket_path = tmp_path / "state" / "recovering.sock"
    server = ControlServer(socket_path, lifecycle)  # type: ignore[arg-type]
    server.start()
    monkeypatch.setenv("AGENTCTL_SOCKET", str(socket_path))
    try:
        code, envelope = _status()
    finally:
        server.stop()
    assert code == 0
    assert _result(envelope)["instance"]["state"] == "recovering"


def test_status_answers_recovering_for_retained_work_after_reconciliation(
    tmp_path: Path,
) -> None:
    lifecycle, _, state = make_lifecycle(tmp_path)
    # A retained checkpoint after reconciliation: recorded effects of an
    # earlier attempt awaiting a recovery command.
    state.start(issue_number=42, branch="agent/issue-42", issue_title="Add CSV export")
    lifecycle._startup_reconciled = True  # noqa: SLF001 - reconciliation done
    snapshot = lifecycle.contract_status_snapshot()
    assert snapshot["instance"]["state"] == "recovering"


def test_live_claim_is_not_a_recovering_hold(tmp_path: Path) -> None:
    """A checkpoint claimed by this process (before reconciliation in tests)
    is not a recovery hold: the instance stays active."""

    lifecycle, _, state = make_lifecycle(tmp_path)
    state.start(issue_number=42, branch="agent/issue-42", issue_title="Add CSV export")
    state.transition(AttemptPhase.SETUP)
    state.transition(AttemptPhase.MODEL_RUNNING)
    snapshot = lifecycle.contract_status_snapshot()
    assert snapshot["instance"]["state"] == "active"


def test_status_answers_recovering_while_a_startup_hold_is_active(
    tmp_path: Path,
) -> None:
    lifecycle, _, _ = make_lifecycle(tmp_path)
    lifecycle._startup_hold_detail = (  # noqa: SLF001 - unreadable checkpoint ledger
        "attempt checkpoint for issue #42 could not be read; "
        "repair the state directory, then resume"
    )
    snapshot = lifecycle.contract_status_snapshot()
    assert snapshot["instance"]["state"] == "recovering"


# ---------------------------------------------------------------------------
# Commands committed during ``starting`` apply at the first claim boundary:
# a stop then prevents the first claim.
# ---------------------------------------------------------------------------


def test_gated_claim_blocks_while_starting_without_consuming_tracker(
    tmp_path: Path,
) -> None:
    tracker = FakeTracker(issues={42: _issue(42)}, fifo=[42])
    lifecycle, store, _ = make_lifecycle(tmp_path, tracker, starting=True)
    assert store.intake_state() is IntakeState.RUNNING

    claim, checkpoint = lifecycle._gated_claim()  # noqa: SLF001
    assert (claim, checkpoint) == (None, None)
    assert tracker.claimed == []

    lifecycle.mark_started()
    claim, checkpoint = lifecycle._gated_claim()  # noqa: SLF001
    assert claim is not None
    assert checkpoint is not None
    assert claim.issue.number == 42


def test_stop_committed_during_starting_prevents_the_first_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = FakeTracker(issues={42: _issue(42)}, fifo=[42])
    lifecycle, store, _ = make_lifecycle(tmp_path, tracker, starting=True)
    socket_path = tmp_path / "state" / "stop-during-start.sock"
    server = ControlServer(socket_path, lifecycle)  # type: ignore[arg-type]
    server.start()
    monkeypatch.setenv("AGENTCTL_SOCKET", str(socket_path))
    try:
        code, envelope, _, _ = _run(
            ["--contract", "1", "stop", "--request-id", "req-stop-1",
             "--expect-repo", REPOSITORY]
        )
    finally:
        server.stop()
    assert code == 0
    assert envelope["ok"] is True

    lifecycle.mark_started()
    claim, checkpoint = lifecycle._gated_claim()  # noqa: SLF001
    assert (claim, checkpoint) == (None, None)
    assert tracker.claimed == []
    assert store.intake_state() is not IntakeState.RUNNING


# ---------------------------------------------------------------------------
# Halt: each engine-decided exit records a Halt before the process exits;
# the next start reports it in ``last_halt``; a later successful start
# does not erase it.
# ---------------------------------------------------------------------------


def test_halt_record_roundtrip(tmp_path: Path) -> None:
    store = HaltStore(tmp_path)
    assert store.read() is None
    halt = store.record("consecutive_error_limit_reached", "3 errors")
    assert halt.reason == "consecutive_error_limit_reached"
    assert halt.detail == "3 errors"
    assert isinstance(halt.at, str) and halt.at
    reread = store.read()
    assert reread == halt
    assert store.path.parent == tmp_path / "state"


def test_consecutive_error_limit_records_a_halt_before_exiting(
    tmp_path: Path,
) -> None:
    from simple_coding_agent.operating import ConsecutiveErrorStore

    halt_store = HaltStore(tmp_path)
    lifecycle, _, _ = make_lifecycle(
        tmp_path, halt_store=halt_store, starting=False
    )
    lifecycle._error_store = ConsecutiveErrorStore(tmp_path)  # noqa: SLF001
    lifecycle._max_consecutive_errors = 1  # noqa: SLF001

    from simple_coding_agent.lifecycle import AttemptOutcome

    with pytest.raises(SystemExit) as exc_info:
        lifecycle._record_terminal_outcome(  # noqa: SLF001
            AttemptOutcome.INFRASTRUCTURE_ERROR, "attempt-1", issue_number=42
        )
    assert exc_info.value.code == 1
    halt = halt_store.read()
    assert halt is not None
    assert halt.reason == HALT_CONSECUTIVE_ERROR_LIMIT
    assert "attempt-1" in halt.detail


def test_next_start_reports_last_halt_and_successful_start_keeps_it(
    tmp_path: Path,
) -> None:
    halt_store = HaltStore(tmp_path)
    halt_store.record(HALT_PROVENANCE_FAILED, "signer key missing")

    lifecycle, _, _ = make_lifecycle(tmp_path, halt_store=halt_store)
    snapshot = lifecycle.contract_status_snapshot()
    assert snapshot["instance"]["last_halt"] == {
        "reason": HALT_PROVENANCE_FAILED,
        "detail": "signer key missing",
        "at": halt_store.read().at,  # type: ignore[union-attr]
    }

    # A later successful start (no new halt recorded) still reports it.
    relaunched, _, _ = make_lifecycle(tmp_path, halt_store=HaltStore(tmp_path))
    assert relaunched.contract_status_snapshot()["instance"][
        "last_halt"
    ] == snapshot["instance"]["last_halt"]


def test_newer_halt_replaces_last_halt(tmp_path: Path) -> None:
    halt_store = HaltStore(tmp_path)
    halt_store.record(HALT_PROVENANCE_FAILED, "first")
    halt_store.record(HALT_CONSECUTIVE_ERROR_LIMIT, "second")
    lifecycle, _, _ = make_lifecycle(tmp_path, halt_store=HaltStore(tmp_path))
    assert lifecycle.contract_status_snapshot()["instance"]["last_halt"][
        "reason"
    ] == HALT_CONSECUTIVE_ERROR_LIMIT


def test_control_endpoint_failure_records_a_halt(tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = JsonEventLogger(stream)
    halt_store = HaltStore(tmp_path)

    class _FailingServer:
        def start(self) -> None:
            raise OSError(30, "Read-only file system")

    with pytest.raises(SystemExit) as exc_info:
        entrypoint._start_control_endpoint(logger, _FailingServer(), halt_store)  # type: ignore[arg-type]
    assert exc_info.value.code == 1
    halt = halt_store.read()
    assert halt is not None
    assert halt.reason == HALT_CONTROL_ENDPOINT_UNAVAILABLE


def test_record_startup_halt_never_raises(tmp_path: Path) -> None:
    entrypoint._record_startup_halt(None, HALT_PROVENANCE_FAILED, "detail")

    class _FailingStore:
        def record(self, *args: object, **kwargs: object) -> object:
            raise OSError("disk gone")

    entrypoint._record_startup_halt(_FailingStore(), HALT_PROVENANCE_FAILED, "detail")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Configuration error: exits non-zero with a stderr diagnostic and records
# nothing (a documented limitation: no halt is possible before the lock).
# ---------------------------------------------------------------------------


def test_configuration_error_exits_nonzero_with_stderr_and_records_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    def _fail() -> object:
        raise ConfigurationError("AGENT_GITHUB_TOKEN is not set")

    monkeypatch.setattr(entrypoint, "load_runtime_config", _fail)
    with pytest.raises(SystemExit) as exc_info:
        entrypoint._load_config()
    assert exc_info.value.code != 0
    _, err = capsys.readouterr()
    assert "configuration error" in err
    assert "AGENT_GITHUB_TOKEN" in err
    # Nothing recorded: no halt file appears in the state directory.
    assert not (tmp_path / "state" / "halt.json").exists()


def test_unexpected_config_failure_also_exits_with_stderr_not_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Any config-load failure (not just ConfigurationError) exits 2 cleanly."""

    def _fail() -> object:
        raise OSError("cannot read environment")

    monkeypatch.setattr(entrypoint, "load_runtime_config", _fail)
    with pytest.raises(SystemExit) as exc_info:
        entrypoint._load_config()
    assert exc_info.value.code != 0
    _, err = capsys.readouterr()
    assert "configuration error" in err
    assert not (tmp_path / "state" / "halt.json").exists()
