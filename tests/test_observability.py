"""Behaviour tests for structured, credential-safe attempt observability."""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path

from simple_coding_agent.observability import AttemptArchive, JsonEventLogger


def test_emits_required_json_fields_and_redacts_credentials() -> None:
    stream = io.StringIO()
    logger = JsonEventLogger(stream, redactions=("github-secret",))

    logger.emit("attempt_finished", issue_number=26, phase="publishing", detail="github-secret failed")

    event = json.loads(stream.getvalue())
    assert set(event) == {"timestamp", "level", "event", "issue_number", "phase", "detail"}
    assert event["detail"] == "[REDACTED] failed"


def test_archives_attempt_metadata_without_credentials(tmp_path: Path) -> None:
    archive = AttemptArchive(
        tmp_path,
        issue_number=26,
        started_at="2026-09-20T13:00:00Z",
        redactions=("meta-secret",),
    )

    archive.write_attempt({"outcome": "infrastructure_error", "token_estimate_usd": "meta-secret"})
    archive.write_text("setup_stdout.log", "meta-secret")

    directory = tmp_path / "logs" / "26" / "2026-09-20T13-00-00Z"
    metadata = json.loads((directory / "attempt.json").read_text())
    assert metadata["token_estimate_usd"] == "[REDACTED]"
    assert metadata["token_estimate_authority"] == "non-authoritative"
    assert (directory / "setup_stdout.log").read_text() == "[REDACTED]"


def test_mirrors_emitted_events_into_the_attempt_output_file(tmp_path: Path) -> None:
    archive = AttemptArchive(
        tmp_path, issue_number=26, started_at="2026-09-20T13:00:00Z", redactions=("meta-secret",)
    )
    stream = io.StringIO()
    logger = JsonEventLogger(
        stream, redactions=("meta-secret",), attempt_sink=lambda issue_number: archive
    )

    logger.emit("tool_call", issue_number=26, phase="model_execution", detail="Bash -> meta-secret")
    logger.emit("provenance_verified", phase="startup", detail="sdk=1.0")

    directory = tmp_path / "logs" / "26" / "2026-09-20T13-00-00Z"
    lines = (directory / "agent_output.json").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "tool_call"
    assert record["detail"] == "Bash -> [REDACTED]"


def test_append_event_writes_one_json_line_per_call(tmp_path: Path) -> None:
    archive = AttemptArchive(
        tmp_path, issue_number=26, started_at="2026-09-20T13:00:00Z", redactions=("meta-secret",)
    )

    archive.append_event({"event": "a", "detail": "meta-secret"})
    archive.append_event({"event": "b", "detail": "fine"})

    directory = tmp_path / "logs" / "26" / "2026-09-20T13-00-00Z"
    records = [json.loads(line) for line in (directory / "agent_output.json").read_text().splitlines()]
    assert [record["event"] for record in records] == ["a", "b"]
    assert records[0]["detail"] == "[REDACTED]"


def _write_transcript(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _recording_event_log() -> tuple[list[tuple[str, str, str]], Callable[..., None]]:
    events: list[tuple[str, str, str]] = []

    def log(event: str, detail: str = "", level: str = "INFO", issue_number: int | None = None) -> None:
        events.append((level, event, detail))

    return events, log


def test_copies_main_and_subagent_transcripts_of_the_recorded_session(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    projects = home / ".claude" / "projects"
    session = "6618f0cd-1d98-4d0b-a985-1f2b34dae69b"
    main_content = '{"type":"user","sessionId":"%s"}\n{"type":"assistant"}\n' % session
    subagent_content = '{"type":"assistant","agent":"abc"}\n'
    _write_transcript(projects / "projA" / f"{session}.jsonl", main_content)
    _write_transcript(projects / "projA" / session / "subagents" / "agent-abc.jsonl", subagent_content)
    _write_transcript(projects / "projA" / session / "tool-results" / "b9l3b201t.txt", "sidecar\n")
    _write_transcript(projects / "projA" / "other-session.jsonl", '{"type":"user"}\n')
    _write_transcript(projects / "projB" / "other-session.jsonl", '{"type":"user"}\n')
    monkeypatch.setenv("HOME", str(home))
    archive = AttemptArchive(tmp_path / "data", issue_number=26, started_at="2026-09-20T13:00:00Z")
    events, log = _recording_event_log()

    count, total = archive.copy_cli_transcripts(session, issue_number=26, event_log=log)

    transcripts = tmp_path / "data" / "logs" / "26" / "2026-09-20T13-00-00Z" / "transcripts"
    assert (transcripts / f"{session}.jsonl").read_text() == main_content
    assert (transcripts / session / "subagents" / "agent-abc.jsonl").read_text() == subagent_content
    assert not (transcripts / session / "tool-results" / "b9l3b201t.txt").exists()
    assert not (transcripts / "other-session.jsonl").exists()
    assert count == 2
    assert total == len((main_content + subagent_content).encode("utf-8"))
    assert ("INFO", "transcripts_copied", f"session_id={session}; files=2; bytes={total}") in events


def test_copied_transcripts_redact_configured_secrets(tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    session = "session-1"
    _write_transcript(projects / "projA" / f"{session}.jsonl", '{"token":"meta-secret"}\nplain\n')
    _write_transcript(
        projects / "projA" / session / "subagents" / "agent-1.jsonl", "uses meta-secret here\n"
    )
    archive = AttemptArchive(
        tmp_path / "data",
        issue_number=26,
        started_at="2026-09-20T13:00:00Z",
        redactions=("meta-secret",),
    )
    events, log = _recording_event_log()

    count, _ = archive.copy_cli_transcripts(session, issue_number=26, event_log=log, projects_dir=projects)

    transcripts = tmp_path / "data" / "logs" / "26" / "2026-09-20T13-00-00Z" / "transcripts"
    main = (transcripts / f"{session}.jsonl").read_text()
    subagent = (transcripts / session / "subagents" / "agent-1.jsonl").read_text()
    assert count == 2
    assert "meta-secret" not in main
    assert "meta-secret" not in subagent
    assert "[REDACTED]" in main
    assert "[REDACTED]" in subagent
    assert any(event == "transcripts_copied" for _, event, _ in events)


def test_session_id_is_matched_literally_not_as_a_glob(tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    session = "sess[abc]"
    _write_transcript(projects / "projA" / f"{session}.jsonl", '{"exact":true}\n')
    _write_transcript(projects / "projA" / "sessa.jsonl", '{"decoy":true}\n')
    archive = AttemptArchive(tmp_path / "data", issue_number=26, started_at="2026-09-20T13:00:00Z")
    events, log = _recording_event_log()

    count, _ = archive.copy_cli_transcripts(session, issue_number=26, event_log=log, projects_dir=projects)

    transcripts = tmp_path / "data" / "logs" / "26" / "2026-09-20T13-00-00Z" / "transcripts"
    assert count == 1
    assert (transcripts / f"{session}.jsonl").read_text() == '{"exact":true}\n'
    assert not (transcripts / "sessa.jsonl").exists()
    assert any(event == "transcripts_copied" for _, event, _ in events)


def test_missing_transcripts_directory_logs_failure_without_raising(tmp_path: Path) -> None:
    archive = AttemptArchive(tmp_path / "data", issue_number=26, started_at="2026-09-20T13:00:00Z")
    events, log = _recording_event_log()

    count, total = archive.copy_cli_transcripts(
        "session-1", issue_number=26, event_log=log, projects_dir=tmp_path / "missing"
    )

    assert (count, total) == (0, 0)
    assert len(events) == 1
    level, event, detail = events[0]
    assert (level, event) == ("WARNING", "transcripts_copy_failed")
    assert "session-1" in detail
    assert not (tmp_path / "data" / "logs" / "26" / "2026-09-20T13-00-00Z" / "transcripts").exists()


def test_unknown_session_logs_failure_without_raising(tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    _write_transcript(projects / "projA" / "other-session.jsonl", '{"type":"user"}\n')
    archive = AttemptArchive(tmp_path / "data", issue_number=26, started_at="2026-09-20T13:00:00Z")
    events, log = _recording_event_log()

    count, total = archive.copy_cli_transcripts(
        "missing-session", issue_number=26, event_log=log, projects_dir=projects
    )

    assert (count, total) == (0, 0)
    assert len(events) == 1
    level, event, detail = events[0]
    assert (level, event) == ("WARNING", "transcripts_copy_failed")
    assert "missing-session" in detail


def test_missing_session_id_logs_failure_without_raising(tmp_path: Path) -> None:
    archive = AttemptArchive(tmp_path / "data", issue_number=26, started_at="2026-09-20T13:00:00Z")
    events, log = _recording_event_log()

    count, total = archive.copy_cli_transcripts(None, issue_number=26, event_log=log)

    assert (count, total) == (0, 0)
    assert len(events) == 1
    assert (events[0][0], events[0][1]) == ("WARNING", "transcripts_copy_failed")
