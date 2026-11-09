"""Offline operator-contract reads of a stopped instance (issue #345).

An operator observes a stopped instance without starting its engine:
every ``agentctl --contract 1 --offline <read>`` is served from the
volume alone. The engine is built from the shared fakes, stopped
cleanly, and the volume is made read-only; the offline replies must
match the volume's committed state without the lock and without writing.
"""

from __future__ import annotations

import gc
import io
import json
import os
import sqlite3
import sys
import tarfile
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.contract_schema import validate_export_stream, validate_producer_envelope

from simple_coding_agent import agentctl
from simple_coding_agent.agent_version import agent_version
from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.completion import (
    AttemptOutcome,
    CompletionDecision,
    PublicationPath,
)
from simple_coding_agent.contract_adapter import advertised_capabilities
from simple_coding_agent.control import ControlStore
from simple_coding_agent.halt import HaltStore
from simple_coding_agent.lifecycle import AgentLifecycle, AttemptEvidence
from simple_coding_agent.observability import AttemptArchive
from simple_coding_agent.publication import PullRequest
from simple_coding_agent.telemetry import TelemetryReporter
from simple_coding_agent.volume_lock import VolumeLock
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace

REPOSITORY = "example-owner/example-repo"
STOP_REQUEST_ID = "req-stop-01"
HALT_REASON = "consecutive_error_limit_reached"
HALT_DETAIL = "3 consecutive infrastructure errors reached the limit of 3"


def _tracked_issue(number: int):
    from simple_coding_agent.github_tracker import TrackerIssue

    return TrackerIssue(
        id=f"issue-{number}",
        number=number,
        title=f"Issue {number}",
        body=f"Implement {number}.",
        created_at=datetime(2026, 11, 6, 8, 0, tzinfo=UTC),
        state="OPEN",
        labels=frozenset({"ready-for-agent"}),
        assignee_logins=(),
        blocked_by=0,
        author_login="reporter",
    )


def _complete_evidence() -> AttemptEvidence:
    return AttemptEvidence(
        decision=CompletionDecision(
            AttemptOutcome.COMPLETE, True, PublicationPath.COMPLETE, ("ready",)
        ),
        check_command="pytest",
        check_exit_code=0,
        review_cycles=1,
        review_findings="all clear",
        details="Implemented.",
    )


def _reporter() -> TelemetryReporter:
    return TelemetryReporter(
        repository=REPOSITORY,
        agent_version="0.0-test",
        sdk_version="0.2.163-test",
        cli_version="2.1.286-test",
        process_started_at="2026-01-01T00:00:00Z",
        hard_ceiling=1000,
        turns_max=50,
        timeout_seconds=600,
        continuation_max=2,
    )


def _build_stopped_volume(data_dir: Path, *, record_halt: bool = True) -> dict:
    """Run the faked engine once, stop it cleanly, and checkpoint the WAL."""

    from simple_coding_agent.config import RepositoryProfile

    store = ControlStore(data_dir)
    lifecycle = AgentLifecycle(
        tracker=FakeTracker({7: _tracked_issue(7)}, fifo=[7]),  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(data_dir),
        workspace=InMemoryWorkspace(data_dir / "repo"),
        profile_loader=lambda _: RepositoryProfile(
            setup=("setup.sh",),
            check=("pytest",),
            base_branch="main",
            timeout=300,
            setup_timeout=120,
            env={},
        ),
        publisher=FakePublisher(
            pull_request=PullRequest(number=11, url="https://example.test/11")
        ),
        attempt_runner=lambda claim, profile, prepared: _complete_evidence(),
        control_store=store,
        telemetry=_reporter(),
        repository=REPOSITORY,
        data_dir=data_dir,
        halt_store=HaltStore(data_dir),
        sleeper=lambda seconds: None,
        event_log=lambda *args, **kwargs: None,
        attempt_archive_factory=lambda issue_number, started_at: AttemptArchive(
            data_dir, issue_number=issue_number, started_at=started_at
        ),
    )
    assert lifecycle.run_once().status.value == "attempted"
    stop_record = store.submit_stop(STOP_REQUEST_ID, has_active_attempt=False)
    halt = None
    if record_halt:
        # A previous engine-decided exit: a successful run never erases it,
        # so a stopped volume still reports the last halt.
        halt = HaltStore(data_dir).record(HALT_REASON, HALT_DETAIL)
    store.checkpoint()
    gc.collect()
    return {
        "attempt_id": _finished_attempt_id(data_dir),
        "stop_record": stop_record,
        "halt": halt,
    }


def _finished_attempt_id(data_dir: Path) -> str:
    """Discover the completed attempt the faked engine just finalized."""

    for issue_dir in sorted((data_dir / "logs").iterdir()):
        if not issue_dir.is_dir():
            continue
        for run_dir in sorted(issue_dir.iterdir()):
            record_path = run_dir / "attempt.json"
            if not record_path.is_file():
                continue
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if isinstance(record, dict) and record.get("outcome"):
                return str(record["started_at"])
    raise AssertionError("the faked engine left no completed attempt")


def _make_read_only(data_dir: Path) -> None:
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and not path.is_symlink():
            os.chmod(path, 0o444)
    for path in sorted(data_dir.rglob("*"), reverse=True):
        if path.is_dir() and not path.is_symlink():
            os.chmod(path, 0o555)
    os.chmod(data_dir, 0o555)


def _make_writable(data_dir: Path) -> None:
    os.chmod(data_dir, 0o755)
    for path in sorted(data_dir.rglob("*")):
        if path.is_symlink():
            continue
        if path.is_dir():
            os.chmod(path, 0o755)
        else:
            os.chmod(path, 0o644)


def _volume_bytes(data_dir: Path) -> dict[str, bytes]:
    """Map every volume file to its bytes; the reads must not move any."""

    snapshot: dict[str, bytes] = {}
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and not path.is_symlink():
            snapshot[str(path.relative_to(data_dir))] = path.read_bytes()
    return snapshot


def _stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, record_halt: bool = True
) -> dict:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AGENTCTL_SOCKET", str(tmp_path / "missing.sock"))
    return _build_stopped_volume(tmp_path, record_halt=record_halt)


def _run(argv: list[str]) -> tuple[int, dict, str, str]:
    """Run the CLI entry point; return (exit code, envelope, stdout, stderr)."""

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = agentctl.main(argv)
    assert isinstance(code, int)
    stdout = out.getvalue()
    assert stdout.endswith("\n"), "stdout must end with exactly one newline"
    assert stdout.count("\n") == 1, "stdout must hold exactly one envelope"
    envelope = json.loads(stdout)
    validate_producer_envelope(envelope, argv, code)
    return code, envelope, stdout, err.getvalue()


def _run_export_raw(argv: list[str]) -> tuple[int, bytes, str]:
    """Run the CLI with a binary stdout; return (exit code, raw bytes, stderr)."""

    raw = io.BytesIO()
    wrapper = io.TextIOWrapper(raw, encoding="utf-8", newline="\n")
    previous = sys.stdout
    sys.stdout = wrapper  # type: ignore[assignment]
    err = io.StringIO()
    try:
        with redirect_stderr(err):
            code = agentctl.main(argv)
    finally:
        wrapper.flush()
        sys.stdout = previous
    assert isinstance(code, int)
    payload = raw.getvalue()
    if code == 0:
        validate_export_stream(payload)
    return code, payload, err.getvalue()


def _tar_members(payload: bytes) -> tuple[list[str], dict]:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r|") as archive:
        names: list[str] = []
        manifest: dict | None = None
        for member in archive:
            assert member.name is not None
            names.append(member.name)
            if member.name == "manifest.json":
                handle = archive.extractfile(member)
                assert handle is not None
                manifest = json.loads(handle.read().decode("utf-8"))
    assert manifest is not None
    return names, manifest


# ---------------------------------------------------------------------------
# Reads from a read-only stopped volume
# ---------------------------------------------------------------------------


def test_offline_reads_from_read_only_stopped_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every offline read is served from the volume alone, nothing written."""

    info = _stopped(tmp_path, monkeypatch)
    attempt_id: str = info["attempt_id"]
    before = _volume_bytes(tmp_path)
    _make_read_only(tmp_path)
    try:
        code, envelope, _, err = _run(["--contract", "1", "--offline", "identity"])
        assert code == 0, envelope
        assert envelope["result"] == {
            "backend": "simple-coding-agent",
            "agent_version": agent_version(),
            "capabilities": advertised_capabilities(""),
            "contract": {"major": 1, "minor": 0},
            "mode": "offline",
            "process_started_at": None,
            "repository": REPOSITORY,
        }
        assert err == ""

        code, envelope, _, err = _run(["--contract", "1", "--offline", "status"])
        assert code == 0, envelope
        status = envelope["result"]
        assert status["instance"]["state"] == "paused"
        assert status["intake"]["state"] == "closed"
        assert status["instance"]["last_halt"] == {
            "reason": HALT_REASON,
            "detail": HALT_DETAIL,
            "at": info["halt"].at,
        }
        assert status["current_attempt"] is None
        assert status["last_finished"]["attempt_id"] == attempt_id
        assert status["recent_events"] == []
        assert any(
            entry["request_id"] == STOP_REQUEST_ID and entry["kind"] == "stop"
            for entry in status["recent_commands"]
        )
        assert err == ""

        code, envelope, _, err = _run(
            ["--contract", "1", "--offline", "command", STOP_REQUEST_ID]
        )
        assert code == 0, envelope
        command = envelope["result"]["command"]
        assert command["request_id"] == STOP_REQUEST_ID
        assert command["kind"] == "stop"
        assert command["repository"] == REPOSITORY
        assert command["acknowledgement"] == "completed"
        assert err == ""

        code, envelope, _, _ = _run(
            ["--contract", "1", "--offline", "attempts", "list"]
        )
        assert code == 0, envelope
        assert attempt_id in [
            entry["attempt_id"] for entry in envelope["result"]["attempts"]
        ]

        code, envelope, _, _ = _run(
            ["--contract", "1", "--offline", "attempts", "show", attempt_id]
        )
        assert code == 0, envelope
        assert envelope["result"]["attempt"]["attempt_id"] == attempt_id
        assert "attempt.json" in [
            entry["path"] for entry in envelope["result"]["manifest"]
        ]

        code, payload, err = _run_export_raw(
            ["--contract", "1", "--offline", "attempts", "export", attempt_id]
        )
        assert code == 0
        names, manifest = _tar_members(payload)
        assert names[0] == "manifest.json"
        assert names[1:] == [entry["path"] for entry in manifest["files"]]
        assert "attempt.json" in names
        assert err == ""
    finally:
        _make_writable(tmp_path)
    assert _volume_bytes(tmp_path) == before


def test_offline_status_reports_stopped_and_last_halt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stopped volume reports no live attempt and the durable last halt."""

    info = _stopped(tmp_path, monkeypatch)
    code, envelope, _, _ = _run(["--contract", "1", "--offline", "status"])
    assert code == 0, envelope
    status = envelope["result"]
    assert status["instance"]["state"] == "paused"
    assert status["current_attempt"] is None
    assert status["instance"]["last_halt"]["reason"] == HALT_REASON
    assert status["instance"]["last_halt"]["at"] == info["halt"].at


def test_offline_status_without_halt_reports_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean stop with no engine-decided exit reports no last halt."""

    _stopped(tmp_path, monkeypatch, record_halt=False)
    code, envelope, _, _ = _run(["--contract", "1", "--offline", "status"])
    assert code == 0, envelope
    status = envelope["result"]
    assert status["instance"]["state"] == "paused"
    assert status["instance"]["last_halt"] is None


def test_offline_reads_rejected_while_lock_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A volume a live engine still holds answers ``instance_running``."""

    info = _stopped(tmp_path, monkeypatch)
    attempt_id: str = info["attempt_id"]
    lock = VolumeLock(tmp_path)
    lock.acquire()
    try:
        reads = [
            ["--contract", "1", "--offline", "identity"],
            ["--contract", "1", "--offline", "status"],
            ["--contract", "1", "--offline", "command", STOP_REQUEST_ID],
            ["--contract", "1", "--offline", "attempts", "list"],
            ["--contract", "1", "--offline", "attempts", "show", attempt_id],
        ]
        for argv in reads:
            code, envelope, _, err = _run(argv)
            assert code == 1, argv
            assert envelope["error"]["code"] == "rejected", argv
            assert envelope["error"]["details"] == {
                "reason": "instance_running"
            }, argv
            assert envelope["error"]["retryable"] is False, argv
            assert "request_id" not in envelope["error"], argv
            assert err == "", argv
        code, payload, err = _run_export_raw(
            ["--contract", "1", "--offline", "attempts", "export", attempt_id]
        )
        assert code == 1
        envelope = json.loads(payload.decode("utf-8"))
        assert envelope["error"]["code"] == "rejected"
        assert envelope["error"]["details"] == {"reason": "instance_running"}
        assert envelope["error"]["retryable"] is False
        assert err == ""
    finally:
        lock.release()


def test_offline_reads_after_unclean_stop_are_correct_or_internal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-empty WAL never yields stale data presented as current."""

    info = _stopped(tmp_path, monkeypatch)
    attempt_id: str = info["attempt_id"]
    db_path = tmp_path / "state" / "control.sqlite3"
    killer = sqlite3.connect(str(db_path))
    killer.execute("PRAGMA journal_mode=WAL")
    row = killer.execute("SELECT COALESCE(MAX(sequence), 0) FROM commands").fetchone()
    killer.execute(
        "INSERT INTO commands (request_id, sequence, kind, payload,"
        " acknowledgement, detail, created_at, updated_at)"
        " VALUES ('req-killed', ?, 'resume', '{}', 'completed',"
        " 'killed engine row', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
        (int(row[0]) + 1,),
    )
    killer.commit()
    assert db_path.with_name("control.sqlite3-wal").stat().st_size > 0
    try:
        code, envelope, _, _ = _run(["--contract", "1", "--offline", "identity"])
        assert code in (0, 7), envelope
        if code == 0:
            assert envelope["result"]["repository"] == REPOSITORY

        code, envelope, _, _ = _run(["--contract", "1", "--offline", "status"])
        assert code in (0, 7), envelope
        if code == 0:
            # The committed WAL row is current data, never stale: a
            # successful read surfaces it instead of the older snapshot.
            status = envelope["result"]
            assert status["instance"]["state"] == "paused"
            assert "req-killed" in [
                entry["request_id"] for entry in status["recent_commands"]
            ]

        code, envelope, _, _ = _run(
            ["--contract", "1", "--offline", "command", STOP_REQUEST_ID]
        )
        assert code in (0, 7), envelope
        if code == 0:
            assert envelope["result"]["command"]["request_id"] == STOP_REQUEST_ID

        code, envelope, _, _ = _run(
            ["--contract", "1", "--offline", "attempts", "list"]
        )
        assert code in (0, 7), envelope
        if code == 0:
            assert attempt_id in [
                entry["attempt_id"] for entry in envelope["result"]["attempts"]
            ]

        code, envelope, _, _ = _run(
            ["--contract", "1", "--offline", "attempts", "show", attempt_id]
        )
        assert code in (0, 7), envelope
        if code == 0:
            assert envelope["result"]["attempt"]["attempt_id"] == attempt_id

        code, payload, _ = _run_export_raw(
            ["--contract", "1", "--offline", "attempts", "export", attempt_id]
        )
        assert code in (0, 7), payload
        if code == 0:
            names, _ = _tar_members(payload)
            assert names[0] == "manifest.json"
    finally:
        killer.close()


def test_offline_mutations_are_unsupported_and_leave_volume_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every mutation answers ``unsupported`` offline, storing nothing."""

    info = _stopped(tmp_path, monkeypatch)
    attempt_id: str = info["attempt_id"]
    before = _volume_bytes(tmp_path)
    request_id = "req-1a2b3c4d5e6f"
    mutations = [
        ["stop", "--request-id", request_id, "--expect-repo", REPOSITORY],
        ["resume", "--request-id", request_id, "--expect-repo", REPOSITORY],
        [
            "stop",
            "--after",
            "2",
            "--request-id",
            request_id,
            "--expect-repo",
            REPOSITORY,
        ],
        ["next", "issue", "7", "--request-id", request_id, "--expect-repo", REPOSITORY],
        ["handoff", "now", "--request-id", request_id, "--expect-repo", REPOSITORY],
        [
            "recovery",
            "retry",
            attempt_id,
            "--request-id",
            request_id,
            "--expect-repo",
            REPOSITORY,
        ],
        [
            "recovery",
            "release",
            attempt_id,
            "--saved-at",
            "2026-01-01T00:00:00Z",
            "--request-id",
            request_id,
            "--expect-repo",
            REPOSITORY,
        ],
        ["errors", "reset", "--request-id", request_id, "--expect-repo", REPOSITORY],
    ]
    for operation in mutations:
        code, envelope, _, err = _run(["--contract", "1", "--offline", *operation])
        assert code == 3, operation
        assert envelope["error"]["code"] == "unsupported", operation
        assert envelope["error"]["request_id"] == request_id, operation
        assert err == "", operation
    assert _volume_bytes(tmp_path) == before


def test_offline_unknown_command_and_paths_are_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown commands, attempts and archive paths answer ``not_found``."""

    info = _stopped(tmp_path, monkeypatch)
    attempt_id: str = info["attempt_id"]

    code, envelope, _, _ = _run(
        ["--contract", "1", "--offline", "command", "req-never-submitted"]
    )
    assert code == 5
    assert envelope["error"]["code"] == "not_found"

    code, envelope, _, _ = _run(
        ["--contract", "1", "--offline", "attempts", "show", "never-an-attempt"]
    )
    assert code == 5
    assert envelope["error"]["code"] == "not_found"

    code, envelope, _, _ = _run(
        ["--contract", "1", "--offline", "attempts", "export", "never-an-attempt"]
    )
    assert code == 5
    assert envelope["error"]["code"] == "not_found"

    code, payload, _ = _run_export_raw(
        [
            "--contract",
            "1",
            "--offline",
            "attempts",
            "export",
            attempt_id,
            "--path",
            "no-such-file.txt",
        ]
    )
    assert code == 5
    assert json.loads(payload.decode("utf-8"))["error"]["code"] == "not_found"


def test_offline_invalid_values_are_invalid_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Value checks run before the volume is touched, like the online path."""

    _stopped(tmp_path, monkeypatch)

    code, envelope, _, _ = _run(
        ["--contract", "1", "--offline", "attempts", "list", "--limit", "0"]
    )
    assert code == 1
    assert envelope["error"]["code"] == "invalid_argument"

    code, envelope, _, _ = _run(
        ["--contract", "1", "--offline", "attempts", "list", "--before", ""]
    )
    assert code == 1
    assert envelope["error"]["code"] == "invalid_argument"

    code, envelope, _, _ = _run(["--contract", "1", "--offline", "command", ""])
    assert code == 1
    assert envelope["error"]["code"] == "invalid_argument"
