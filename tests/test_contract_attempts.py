"""Contract ``attempts list`` / ``attempts show`` over the live engine.

A dashboard pages an instance's Attempt history through the contract
path. Every test drives the real lifecycle, control socket and CLI entry
point in-process and asserts only on the exit code and the parsed stdout
envelope: stdout always holds exactly one JSON document with a trailing
newline and stderr stays empty.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterator
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

import pytest

from simple_coding_agent import agentctl
from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.control import ControlStore
from simple_coding_agent.control_server import ControlServer, send_request
from simple_coding_agent.observability import AttemptArchive

REPOSITORY = "example-owner/example-repo"


def _tracker_issue(number: int, **overrides: object):
    from simple_coding_agent.github_tracker import TrackerIssue

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


def make_lifecycle(tmp_path: Path, *, issues: list[int] | None = None):
    """Wire the real lifecycle with control and attempt archives enabled."""

    from simple_coding_agent.config import RepositoryProfile
    from simple_coding_agent.lifecycle import AgentLifecycle
    from tests.fakes import (
        FakeCommandRunner,
        FakeConflictExecutor,
        FakePublisher,
        FakeTracker,
        InMemoryWorkspace,
    )

    numbers = issues or []
    tracker = FakeTracker(
        {number: _tracker_issue(number) for number in numbers},
        fifo=list(numbers),
    )
    lifecycle = AgentLifecycle(
        tracker=tracker,  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: RepositoryProfile(
            setup=("setup.sh",),
            check=("pytest",),
            base_branch="main",
            timeout=300,
            setup_timeout=120,
            env={},
        ),
        publisher=FakePublisher(),
        attempt_runner=lambda claim, profile, prepared: (_ for _ in ()).throw(
            AssertionError("claims in this test never run the model")
        ),
        control_store=ControlStore(tmp_path),
        repository=REPOSITORY,
        sleeper=lambda seconds: None,
        event_log=lambda event, detail="", level="INFO", issue_number=None: None,
        command_runner=FakeCommandRunner([]),
        conflict_model_executor=FakeConflictExecutor(),
        conflict_issue_comments=lambda _: (),
        attempt_archive_factory=lambda issue_number, started_at: AttemptArchive(
            tmp_path, issue_number=issue_number, started_at=started_at
        ),
        data_dir=tmp_path,
    )
    return lifecycle


@pytest.fixture
def attempts_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[ControlServer, object, Path]]:
    """A fresh control socket serving the real lifecycle with archives."""

    lifecycle = make_lifecycle(tmp_path)
    socket_path = tmp_path / "attempts.sock"
    server = ControlServer(socket_path, lifecycle)  # type: ignore[arg-type]
    server.start()
    monkeypatch.setenv("AGENTCTL_SOCKET", str(socket_path))
    try:
        yield server, lifecycle, socket_path
    finally:
        server.stop()


@pytest.fixture
def claim_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[ControlServer, object, Path]]:
    """A control socket whose tracker offers issue 7 for claiming."""

    lifecycle = make_lifecycle(tmp_path, issues=[7])
    socket_path = tmp_path / "attempts-claim.sock"
    server = ControlServer(socket_path, lifecycle)  # type: ignore[arg-type]
    server.start()
    monkeypatch.setenv("AGENTCTL_SOCKET", str(socket_path))
    try:
        yield server, lifecycle, socket_path
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


def _result(envelope: dict) -> dict:
    assert envelope["contract"] == "1.0"
    assert envelope["ok"] is True
    return envelope["result"]


def _write_record(
    data_dir: Path, issue_number: int, started_at: str, **fields: object
) -> AttemptArchive:
    archive = AttemptArchive(
        data_dir, issue_number=issue_number, started_at=started_at
    )
    record = {"started_at": started_at, "issue_number": issue_number}
    record.update(fields)
    archive.write_attempt(record)
    return archive


# ---------------------------------------------------------------------------
# attempts list
# ---------------------------------------------------------------------------


def test_list_empty_when_no_attempts(attempts_server) -> None:
    _, _, _ = attempts_server

    code, envelope, _, err = _run(["--contract", "1", "attempts", "list"])

    assert err == ""
    assert code == 0
    assert _result(envelope) == {"attempts": [], "next_cursor": None}


def test_list_reports_archived_attempts_newest_first(attempts_server) -> None:
    _, _, _ = attempts_server
    data_dir = _data_dir(attempts_server)
    _write_record(
        data_dir,
        41,
        "2026-11-06T08:01:10Z",
        issue_title="Fix pagination",
        outcome="complete",
        completed_at="2026-11-06T08:44:52Z",
        pull_request={
            "number": 77,
            "url": "https://github.com/example-owner/example-repo/pull/77",
        },
    )
    _write_record(data_dir, 7, "2026-11-06T09:01:10Z", issue_title="Later work")

    code, envelope, _, err = _run(["--contract", "1", "attempts", "list"])

    assert err == ""
    assert code == 0
    result = _result(envelope)
    assert [entry["attempt_id"] for entry in result["attempts"]] == [
        "2026-11-06T09:01:10Z",
        "2026-11-06T08:01:10Z",
    ]
    assert result["next_cursor"] is None
    finished = result["attempts"][1]
    assert finished["issue"] == {"number": 41, "title": "Fix pagination"}
    assert finished["started_at"] == "2026-11-06T08:01:10Z"
    assert finished["finished_at"] == "2026-11-06T08:44:52Z"
    assert finished["lifecycle_state"] == "finished"
    assert finished["outcome"] == {
        "native": "complete",
        "reason": None,
        "label": "Complete",
    }
    assert finished["delivery"] == {
        "pull_request": {
            "number": 77,
            "url": "https://github.com/example-owner/example-repo/pull/77",
        }
    }
    assert finished["attention_required"] is False
    assert finished["sealed"] is False
    assert finished["sealed_at"] is None


def test_active_attempt_listed_as_soon_as_claim_opens(claim_server) -> None:
    _, lifecycle, _ = claim_server

    _, checkpoint = lifecycle._gated_claim()

    assert checkpoint is not None
    code, envelope, _, err = _run(["--contract", "1", "attempts", "list"])

    assert err == ""
    assert code == 0
    (entry,) = _result(envelope)["attempts"]
    assert entry["attempt_id"] == checkpoint.started_at
    assert entry["lifecycle_state"] == "claimed"
    assert entry["issue"] == {"number": 7, "title": "Issue 7"}
    assert entry["outcome"] is None
    assert entry["finished_at"] is None
    assert entry["attention_required"] is False


def test_held_attempt_listed_with_attention_required(claim_server) -> None:
    _, lifecycle, _ = claim_server

    _, checkpoint = lifecycle._gated_claim()
    assert checkpoint is not None
    # A retained checkpoint with no work in flight is held for recovery.
    lifecycle._attempt_processing = False

    code, envelope, _, err = _run(["--contract", "1", "attempts", "list"])

    assert err == ""
    assert code == 0
    (entry,) = _result(envelope)["attempts"]
    assert entry["attempt_id"] == checkpoint.started_at
    assert entry["attention_required"] is True


def test_list_pages_newest_first_with_cursor(attempts_server) -> None:
    _, _, _ = attempts_server
    data_dir = _data_dir(attempts_server)
    for index in range(5):
        _write_record(data_dir, 7, f"2026-11-06T08:{index:02d}:10Z")

    code, first, _, _ = _run(["--contract", "1", "attempts", "list", "--limit", "2"])
    assert code == 0
    first_result = _result(first)
    assert [entry["attempt_id"] for entry in first_result["attempts"]] == [
        "2026-11-06T08:04:10Z",
        "2026-11-06T08:03:10Z",
    ]
    assert first_result["next_cursor"] is not None

    code, second, _, _ = _run(
        ["--contract", "1", "attempts", "list", "--limit", "2",
         "--before", first_result["next_cursor"]]
    )
    assert code == 0
    second_result = _result(second)
    assert [entry["attempt_id"] for entry in second_result["attempts"]] == [
        "2026-11-06T08:02:10Z",
        "2026-11-06T08:01:10Z",
    ]

    code, third, _, _ = _run(
        ["--contract", "1", "attempts", "list", "--limit", "2",
         "--before", second_result["next_cursor"]]
    )
    assert code == 0
    third_result = _result(third)
    assert [entry["attempt_id"] for entry in third_result["attempts"]] == [
        "2026-11-06T08:00:10Z",
    ]
    assert third_result["next_cursor"] is None


def test_list_default_limit_is_twenty(attempts_server) -> None:
    _, _, _ = attempts_server
    data_dir = _data_dir(attempts_server)
    for index in range(25):
        _write_record(data_dir, 7, f"2026-11-06T08:{index:02d}:10Z")

    code, envelope, _, err = _run(["--contract", "1", "attempts", "list"])

    assert err == ""
    assert code == 0
    result = _result(envelope)
    assert len(result["attempts"]) == 20
    assert result["next_cursor"] is not None


def test_list_limit_above_maximum_is_rejected(attempts_server) -> None:
    _, _, _ = attempts_server

    code, envelope, _, err = _run(
        ["--contract", "1", "attempts", "list", "--limit", "101"]
    )

    assert err == ""
    assert code == 1
    assert envelope["error"]["code"] == "invalid_argument"
    assert envelope["error"]["retryable"] is False


def test_list_malformed_cursor_is_rejected(attempts_server) -> None:
    _, _, _ = attempts_server

    code, envelope, _, err = _run(
        ["--contract", "1", "attempts", "list", "--before", "not a cursor!!!"]
    )

    assert err == ""
    assert code == 1
    assert envelope["error"]["code"] == "invalid_argument"
    assert envelope["error"]["retryable"] is False


# ---------------------------------------------------------------------------
# attempts show
# ---------------------------------------------------------------------------


def test_show_returns_summary_and_manifest(attempts_server) -> None:
    _, _, _ = attempts_server
    data_dir = _data_dir(attempts_server)
    archive = _write_record(
        data_dir,
        7,
        "2026-11-06T08:01:10Z",
        issue_title="Fix pagination",
        outcome="complete",
        completed_at="2026-11-06T08:44:52Z",
    )
    archive.write_text("0001_tool_call.json", '{"tool": "Bash"}')
    archive.write_text("0002_tool_result.json", '{"ok": true}')
    archive.write_text("0003_model_response.txt", "done")
    archive.write_text("setup_stdout.log", "setup ok\n")
    (archive.directory / "transcripts").mkdir(parents=True, exist_ok=True)
    (archive.directory / "transcripts" / "sess.jsonl").write_text(
        '{"role": "user"}\n', encoding="utf-8"
    )
    archive.append_event(
        {
            "timestamp": "2026-11-06T08:01:11Z",
            "level": "INFO",
            "event": "claim_opened",
            "issue_number": 7,
            "phase": "claimed",
            "detail": "",
        }
    )

    code, envelope, _, err = _run(
        ["--contract", "1", "attempts", "show", "2026-11-06T08:01:10Z"]
    )

    assert err == ""
    assert code == 0
    result = _result(envelope)
    assert result["attempt"]["attempt_id"] == "2026-11-06T08:01:10Z"
    assert result["attempt"]["outcome"] == {
        "native": "complete",
        "reason": None,
        "label": "Complete",
    }
    by_path = {entry["path"]: entry for entry in result["manifest"]}
    assert by_path["attempt.json"]["kind"] == "record"
    assert by_path["attempt.json"]["media_type"] == "application/json"
    assert by_path["agent_output.json"]["kind"] == "record"
    assert by_path["agent_output.json"]["media_type"] == "application/x-ndjson"
    assert by_path["0001_tool_call.json"]["kind"] == "tool_call"
    assert by_path["0002_tool_result.json"]["kind"] == "tool_result"
    assert by_path["0003_model_response.txt"]["kind"] == "model_output"
    assert by_path["0003_model_response.txt"]["media_type"] == "text/plain"
    assert by_path["setup_stdout.log"]["kind"] == "command_output"
    assert by_path["transcripts/sess.jsonl"]["kind"] == "transcript"
    assert by_path["transcripts/sess.jsonl"]["media_type"] == "application/x-ndjson"
    assert [entry["path"] for entry in result["manifest"]] == sorted(by_path)
    for path, entry in by_path.items():
        raw = (archive.directory / path).read_bytes()
        assert entry["size"] == len(raw)
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()


def test_show_unknown_attempt_is_not_found(attempts_server) -> None:
    _, _, _ = attempts_server
    data_dir = _data_dir(attempts_server)
    _write_record(data_dir, 7, "2026-11-06T08:01:10Z")

    code, envelope, _, err = _run(
        ["--contract", "1", "attempts", "show", "2026-11-06T09:00:00Z"]
    )

    assert err == ""
    assert code == 5
    assert envelope["contract"] == "1.0"
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "not_found"
    assert envelope["error"]["retryable"] is False


def test_show_manifest_never_includes_private_engine_state(
    attempts_server,
) -> None:
    _, _, _ = attempts_server
    data_dir = _data_dir(attempts_server)
    archive = _write_record(data_dir, 7, "2026-11-06T08:01:10Z")
    archive.write_text("0001_tool_call.json", "{}")
    (archive.directory / ".attempt-001.tmp").write_text("partial", encoding="utf-8")
    (archive.directory / "control.sqlite3").write_bytes(b"sqlite")
    (archive.directory / "service-credentials.json").write_text("{}", encoding="utf-8")
    (archive.directory / "config-with-secret.json").write_text("{}", encoding="utf-8")
    (archive.directory / "agent_output.json.lnk").symlink_to("agent_output.json")

    code, envelope, _, err = _run(
        ["--contract", "1", "attempts", "show", "2026-11-06T08:01:10Z"]
    )

    assert err == ""
    assert code == 0
    paths = [entry["path"] for entry in _result(envelope)["manifest"]]
    assert "0001_tool_call.json" in paths
    assert ".attempt-001.tmp" not in paths
    assert "control.sqlite3" not in paths
    assert "service-credentials.json" not in paths
    assert "config-with-secret.json" not in paths
    assert "agent_output.json.lnk" not in paths


def test_show_live_attempt_merges_checkpoint_summary_with_archive_manifest(
    claim_server,
) -> None:
    _, lifecycle, _ = claim_server

    _, checkpoint = lifecycle._gated_claim()
    assert checkpoint is not None
    AttemptArchive(
        _data_dir(claim_server), issue_number=7, started_at=checkpoint.started_at
    ).write_text("0001_tool_call.json", "{}")

    code, envelope, _, err = _run(
        ["--contract", "1", "attempts", "show", checkpoint.started_at]
    )

    assert err == ""
    assert code == 0
    result = _result(envelope)
    assert result["attempt"]["lifecycle_state"] == "claimed"
    assert result["attempt"]["outcome"] is None
    assert result["attempt"]["attention_required"] is False
    assert {entry["path"] for entry in result["manifest"]} >= {
        "attempt.json",
        "0001_tool_call.json",
    }


# ---------------------------------------------------------------------------
# engine-side argument values (socket level)
# ---------------------------------------------------------------------------


def test_engine_rejects_out_of_range_limit(attempts_server) -> None:
    _, _, socket_path = attempts_server

    reply = send_request(socket_path, {"op": "attempts_list", "limit": 101})

    assert reply["ok"] is False
    assert reply["code"] == "invalid_argument"


def test_engine_rejects_empty_attempt_id(attempts_server) -> None:
    _, _, socket_path = attempts_server

    reply = send_request(socket_path, {"op": "attempts_show", "attempt_id": ""})

    assert reply["ok"] is False
    assert reply["code"] == "invalid_argument"


def _data_dir(server_fixture: tuple[ControlServer, object, Path]) -> Path:
    _, lifecycle, _ = server_fixture
    data_dir = lifecycle._data_dir  # type: ignore[union-attr]
    assert isinstance(data_dir, Path)
    return data_dir
