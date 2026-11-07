"""Attempt history projection: `attempts list` over archive directories.

The dashboard pages an instance's Attempt history through the contract
path. Listing reads the archive directories plus the live checkpoint, so
an Attempt is listed as soon as its Claim opens; the Attempt ID is the
checkpoint ``started_at`` carried by the recorded ``started_at``, never
by string reversal of the directory name.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from simple_coding_agent.attempt_history import ActiveAttempt, list_attempts
from simple_coding_agent.observability import AttemptArchive


def _write_record(
    data_dir: Path, issue_number: int, started_at: str, **fields: object
) -> Path:
    archive = AttemptArchive(
        data_dir, issue_number=issue_number, started_at=started_at
    )
    record = {"started_at": started_at, "issue_number": issue_number}
    record.update(fields)
    archive.write_attempt(record)
    return archive.directory


def test_empty_volume_lists_no_attempts(tmp_path: Path) -> None:
    assert list_attempts(tmp_path) == {"attempts": [], "next_cursor": None}


def test_archived_attempts_list_newest_first(tmp_path: Path) -> None:
    _write_record(tmp_path, 7, "2026-11-06T08:01:10Z", issue_title="Fix pagination")
    _write_record(tmp_path, 7, "2026-11-06T09:01:10Z", issue_title="Fix pagination")

    result = list_attempts(tmp_path)

    assert [entry["attempt_id"] for entry in result["attempts"]] == [
        "2026-11-06T09:01:10Z",
        "2026-11-06T08:01:10Z",
    ]
    assert result["next_cursor"] is None


def test_active_attempt_is_listed_as_soon_as_its_claim_opens(
    tmp_path: Path,
) -> None:
    _write_record(tmp_path, 7, "2026-11-06T08:01:10Z")
    active = ActiveAttempt(
        attempt_id="2026-11-06T09:00:00Z",
        issue_number=7,
        issue_title="Fix pagination",
        phase="claimed",
    )

    result = list_attempts(tmp_path, active=active)

    assert [entry["attempt_id"] for entry in result["attempts"]] == [
        "2026-11-06T09:00:00Z",
        "2026-11-06T08:01:10Z",
    ]
    first = result["attempts"][0]
    assert first["lifecycle_state"] == "claimed"
    assert first["issue"] == {"number": 7, "title": "Fix pagination"}
    assert first["outcome"] is None
    assert first["finished_at"] is None


def test_finished_archive_reports_outcome_pull_request_and_state(
    tmp_path: Path,
) -> None:
    _write_record(
        tmp_path,
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

    (entry,) = list_attempts(tmp_path)["attempts"]

    assert entry["attempt_id"] == "2026-11-06T08:01:10Z"
    assert entry["issue"] == {"number": 41, "title": "Fix pagination"}
    assert entry["started_at"] == "2026-11-06T08:01:10Z"
    assert entry["finished_at"] == "2026-11-06T08:44:52Z"
    assert entry["lifecycle_state"] == "finished"
    assert entry["outcome"] == {
        "native": "complete",
        "reason": None,
        "label": "Complete",
    }
    assert entry["delivery"] == {
        "pull_request": {
            "number": 77,
            "url": "https://github.com/example-owner/example-repo/pull/77",
        }
    }
    assert entry["attention_required"] is False
    assert entry["sealed"] is False
    assert entry["sealed_at"] is None


def test_held_attempt_is_listed_with_attention_required(tmp_path: Path) -> None:
    _write_record(tmp_path, 7, "2026-11-06T08:01:10Z")
    active = ActiveAttempt(
        attempt_id="2026-11-06T08:01:10Z",
        issue_number=7,
        issue_title=None,
        phase="model_running",
    )

    result = list_attempts(
        tmp_path, active=active, held={"2026-11-06T08:01:10Z"}
    )

    (entry,) = result["attempts"]
    assert entry["attempt_id"] == "2026-11-06T08:01:10Z"
    assert entry["attention_required"] is True


def test_default_limit_is_twenty(tmp_path: Path) -> None:
    for index in range(25):
        _write_record(tmp_path, 7, f"2026-11-06T08:{index:02d}:10Z")

    result = list_attempts(tmp_path)

    assert len(result["attempts"]) == 20
    assert result["next_cursor"] is not None


def test_limit_caps_the_page(tmp_path: Path) -> None:
    for index in range(5):
        _write_record(tmp_path, 7, f"2026-11-06T08:{index:02d}:10Z")

    result = list_attempts(tmp_path, limit=2)

    assert [entry["attempt_id"] for entry in result["attempts"]] == [
        "2026-11-06T08:04:10Z",
        "2026-11-06T08:03:10Z",
    ]
    assert result["next_cursor"] is not None


def test_cursor_pages_through_history(tmp_path: Path) -> None:
    for index in range(5):
        _write_record(tmp_path, 7, f"2026-11-06T08:{index:02d}:10Z")

    first = list_attempts(tmp_path, limit=2)
    second = list_attempts(tmp_path, limit=2, before=first["next_cursor"])
    third = list_attempts(tmp_path, limit=2, before=second["next_cursor"])

    assert [entry["attempt_id"] for entry in first["attempts"]] == [
        "2026-11-06T08:04:10Z",
        "2026-11-06T08:03:10Z",
    ]
    assert [entry["attempt_id"] for entry in second["attempts"]] == [
        "2026-11-06T08:02:10Z",
        "2026-11-06T08:01:10Z",
    ]
    assert [entry["attempt_id"] for entry in third["attempts"]] == [
        "2026-11-06T08:00:10Z",
    ]
    assert third["next_cursor"] is None


def test_zero_limit_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="limit"):
        list_attempts(tmp_path, limit=0)


def test_limit_above_maximum_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="limit"):
        list_attempts(tmp_path, limit=101)


def test_malformed_cursor_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="[Cc]ursor"):
        list_attempts(tmp_path, before="not a cursor!!!")


def test_directory_without_a_recorded_started_at_is_skipped(
    tmp_path: Path,
) -> None:
    archive = AttemptArchive(tmp_path, issue_number=7, started_at="orphan")
    archive.write_text("agent_output.json", '{"event": "x"}\n')
    _write_record(tmp_path, 7, "2026-11-06T08:01:10Z")

    result = list_attempts(tmp_path)

    assert [entry["attempt_id"] for entry in result["attempts"]] == [
        "2026-11-06T08:01:10Z"
    ]


def test_show_returns_summary_and_manifest(tmp_path: Path) -> None:
    from simple_coding_agent.attempt_history import show_attempt

    archive = AttemptArchive(
        tmp_path, issue_number=7, started_at="2026-11-06T08:01:10Z"
    )
    archive.write_attempt(
        {
            "started_at": "2026-11-06T08:01:10Z",
            "issue_number": 7,
            "issue_title": "Fix pagination",
            "outcome": "complete",
            "completed_at": "2026-11-06T08:44:52Z",
        }
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

    result = show_attempt(tmp_path, "2026-11-06T08:01:10Z")

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


def test_show_manifest_covers_every_mapped_kind(tmp_path: Path) -> None:
    from simple_coding_agent.attempt_history import show_attempt

    archive = AttemptArchive(
        tmp_path, issue_number=7, started_at="2026-11-06T08:01:10Z"
    )
    archive.write_attempt({"started_at": "2026-11-06T08:01:10Z"})
    for name in (
        "0001_tool_call.json",
        "0002_tool_result.json",
        "0003_model_response.txt",
        "setup_stdout.log",
        "setup_stderr.log",
        "baseline_check_stdout.log",
        "baseline_check_stderr.log",
        "check_stdout.log",
        "check_stderr.log",
    ):
        archive.write_text(name, "evidence\n")
    (archive.directory / "transcripts").mkdir(parents=True, exist_ok=True)
    (archive.directory / "transcripts" / "sess.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )

    by_path = {
        entry["path"]: entry
        for entry in show_attempt(tmp_path, "2026-11-06T08:01:10Z")["manifest"]
    }

    assert by_path["attempt.json"]["kind"] == "record"
    assert by_path["0001_tool_call.json"]["kind"] == "tool_call"
    assert by_path["0001_tool_call.json"]["media_type"] == "application/json"
    assert by_path["0002_tool_result.json"]["kind"] == "tool_result"
    assert by_path["0003_model_response.txt"]["kind"] == "model_output"
    for name in (
        "setup_stdout.log",
        "setup_stderr.log",
        "baseline_check_stdout.log",
        "baseline_check_stderr.log",
        "check_stdout.log",
        "check_stderr.log",
    ):
        assert by_path[name]["kind"] == "command_output", name
        assert by_path[name]["media_type"] == "text/plain", name
    assert by_path["transcripts/sess.jsonl"]["kind"] == "transcript"


def test_show_manifest_never_includes_private_engine_state(
    tmp_path: Path,
) -> None:
    from simple_coding_agent.attempt_history import show_attempt

    archive = AttemptArchive(
        tmp_path, issue_number=7, started_at="2026-11-06T08:01:10Z"
    )
    archive.write_attempt({"started_at": "2026-11-06T08:01:10Z"})
    archive.write_text("0001_tool_call.json", "{}")
    (archive.directory / ".attempt-001.tmp").write_text("partial", encoding="utf-8")
    (archive.directory / "control.sqlite3").write_bytes(b"sqlite")
    (archive.directory / "service-credentials.json").write_text("{}", encoding="utf-8")
    (archive.directory / "config-with-secret.json").write_text("{}", encoding="utf-8")
    (archive.directory / "agent_output.json.lnk").symlink_to("agent_output.json")

    paths = [
        entry["path"]
        for entry in show_attempt(tmp_path, "2026-11-06T08:01:10Z")["manifest"]
    ]

    assert "0001_tool_call.json" in paths
    assert ".attempt-001.tmp" not in paths
    assert "control.sqlite3" not in paths
    assert "service-credentials.json" not in paths
    assert "config-with-secret.json" not in paths
    assert "agent_output.json.lnk" not in paths


def test_show_unknown_attempt_is_not_found(tmp_path: Path) -> None:
    from simple_coding_agent.attempt_history import (
        AttemptHistoryError,
        show_attempt,
    )

    _write_record(tmp_path, 7, "2026-11-06T08:01:10Z")

    with pytest.raises(AttemptHistoryError) as excinfo:
        show_attempt(tmp_path, "2026-11-06T09:00:00Z")

    assert excinfo.value.code == "not_found"


def test_show_live_attempt_merges_checkpoint_summary_with_archive_manifest(
    tmp_path: Path,
) -> None:
    from simple_coding_agent.attempt_history import show_attempt

    archive = AttemptArchive(
        tmp_path, issue_number=7, started_at="2026-11-06T08:01:10Z"
    )
    archive.write_attempt(
        {
            "started_at": "2026-11-06T08:01:10Z",
            "issue_number": 7,
            "issue_title": "Fix pagination",
        }
    )
    archive.write_text("0001_tool_call.json", "{}")
    active = ActiveAttempt(
        attempt_id="2026-11-06T08:01:10Z",
        issue_number=7,
        issue_title="Fix pagination",
        phase="model_running",
    )

    result = show_attempt(
        tmp_path, "2026-11-06T08:01:10Z", active=active, held=set()
    )

    assert result["attempt"]["lifecycle_state"] == "model_running"
    assert result["attempt"]["outcome"] is None
    assert result["attempt"]["attention_required"] is False
    assert {entry["path"] for entry in result["manifest"]} >= {
        "attempt.json",
        "0001_tool_call.json",
    }


# ---------------------------------------------------------------------------
# claim: the archive maps back through the recorded started_at from Claim open
# ---------------------------------------------------------------------------


def _claim_lifecycle(tmp_path: Path, data_dir: Path, **issue_overrides: object):
    from datetime import UTC, datetime

    from simple_coding_agent.attempt_state import AttemptStateStore
    from simple_coding_agent.config import RepositoryProfile
    from simple_coding_agent.github_tracker import TrackerIssue
    from simple_coding_agent.lifecycle import AgentLifecycle
    from tests.fakes import (
        FakeCommandRunner,
        FakeConflictExecutor,
        FakePublisher,
        FakeTracker,
        InMemoryWorkspace,
    )

    created = datetime(2026, 9, 20, 13, 0, tzinfo=UTC)
    base: dict[str, object] = {
        "id": "issue-7",
        "number": 7,
        "title": "Issue 7",
        "body": "Implement 7.",
        "created_at": created,
        "state": "OPEN",
        "labels": frozenset({"ready-for-agent"}),
        "assignee_logins": (),
        "blocked_by": 0,
        "author_login": "reporter",
    }
    base.update(issue_overrides)
    tracker = FakeTracker({7: TrackerIssue(**base)}, fifo=[7])  # type: ignore[arg-type]
    lifecycle = AgentLifecycle(
        tracker=tracker,  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(tmp_path / "store"),
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
        control_store=None,
        sleeper=lambda seconds: None,
        event_log=lambda event, detail="", level="INFO", issue_number=None: None,
        max_conflict_attempts=2,
        command_runner=FakeCommandRunner([]),
        conflict_model_executor=FakeConflictExecutor(),
        conflict_issue_comments=lambda _: (),
        attempt_archive_factory=lambda issue_number, started_at: AttemptArchive(
            data_dir, issue_number=issue_number, started_at=started_at
        ),
    )
    return lifecycle


def test_ordinary_claim_records_attempt_identity_in_the_archive(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    lifecycle = _claim_lifecycle(tmp_path, data_dir)

    claim, checkpoint = lifecycle._gated_claim()

    assert claim is not None and checkpoint is not None
    record = AttemptArchive(
        data_dir, issue_number=7, started_at=checkpoint.started_at
    ).read_attempt()
    assert record.get("started_at") == checkpoint.started_at
    assert record.get("issue_number") == 7
    assert record.get("issue_title") == "Issue 7"


def test_conflict_claim_records_attempt_identity_in_the_archive(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    lifecycle = _claim_lifecycle(
        tmp_path, data_dir, labels=frozenset({"merge-conflict"})
    )

    claim, checkpoint = lifecycle._gated_claim()

    assert claim is not None and checkpoint is not None
    record = AttemptArchive(
        data_dir, issue_number=7, started_at=checkpoint.started_at
    ).read_attempt()
    assert record.get("started_at") == checkpoint.started_at
    assert record.get("issue_number") == 7
    assert record.get("attempt_kind") == "conflict"
