"""Handoff note module tests: read, validate, emergency, recovered.

Every rule about the note file at ``.agent/handoff/<issue-number>.md`` lives
in ``simple_coding_agent.handoff_note``; these tests pin that module's
behavior through its public interface with plain temp-dir fixtures.
"""

from __future__ import annotations

from pathlib import Path

from simple_coding_agent.git_workspace import AttemptCommit
from simple_coding_agent.handoff_note import (
    build_emergency_handoff_note,
    classify_handoff_reason,
    extract_recovered_handoff_note,
    is_complete_handoff_note,
    is_operator_request_note,
    parse_handoff_note_fields,
    read_handoff_note,
    validate_handoff_note,
)


ISSUE_NUMBER = 24
WORK_REV = "c" * 40
NOTE_REV = "a" * 40
BASE_REV = "b" * 40


def write_note(directory: Path, body: str, *, issue_number: int = ISSUE_NUMBER) -> Path:
    note_path = directory / ".agent" / "handoff" / f"{issue_number}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(body)
    return note_path


def note_body(
    *,
    issue_number: int = ISSUE_NUMBER,
    started_at: str = "2026-09-22T00:00:00Z",
    reason: str = "token_soft_threshold",
    last_work_commit: str = WORK_REV,
) -> str:
    return (
        f"# Handoff note: issue #{issue_number}\n"
        f"\n"
        f"- issue: {issue_number}\n"
        f"- started_at: {started_at}\n"
        f"- reason: {reason}\n"
        f"- last_work_commit: {last_work_commit}\n"
        f"\n"
        f"## Summary\n"
        f"\n"
        f"Half done.\n"
    )


def test_read_returns_the_note_verbatim(tmp_path: Path) -> None:
    body = note_body()
    write_note(tmp_path, body)

    assert read_handoff_note(tmp_path, ISSUE_NUMBER) == body


def test_read_reports_a_missing_note(tmp_path: Path) -> None:
    assert "could not be read" in read_handoff_note(tmp_path, ISSUE_NUMBER)


def test_parse_reads_the_title_and_header_fields() -> None:
    fields = parse_handoff_note_fields(note_body(reason="operator_request"))

    assert fields["title_issue"] == str(ISSUE_NUMBER)
    assert fields["issue"] == str(ISSUE_NUMBER)
    assert fields["reason"] == "operator_request"
    assert fields["last_work_commit"] == WORK_REV


def test_valid_note_passes_validation(tmp_path: Path) -> None:
    write_note(tmp_path, note_body())
    note_commit = AttemptCommit(NOTE_REV, "Handoff note")
    commits = [note_commit, AttemptCommit(WORK_REV, "Half-finish the parser")]

    validation = validate_handoff_note(
        note_commit, commits, ISSUE_NUMBER, tmp_path, BASE_REV
    )

    assert validation.valid is True


def test_missing_note_commit_is_invalid(tmp_path: Path) -> None:
    validation = validate_handoff_note(None, [], ISSUE_NUMBER, tmp_path, BASE_REV)

    assert validation.valid is False
    assert "no handoff note commit" in validation.reason


def test_note_for_another_issue_is_invalid(tmp_path: Path) -> None:
    write_note(tmp_path, note_body(issue_number=99))
    note_commit = AttemptCommit(NOTE_REV, "Handoff note")

    validation = validate_handoff_note(
        note_commit, [note_commit], ISSUE_NUMBER, tmp_path, BASE_REV
    )

    assert validation.valid is False


def test_note_naming_stale_work_is_invalid(tmp_path: Path) -> None:
    write_note(tmp_path, note_body(last_work_commit="d" * 40))
    note_commit = AttemptCommit(NOTE_REV, "Handoff note")
    commits = [note_commit, AttemptCommit(WORK_REV, "Half-finish the parser")]

    validation = validate_handoff_note(
        note_commit, commits, ISSUE_NUMBER, tmp_path, BASE_REV
    )

    assert validation.valid is False
    assert "last_work_commit" in validation.reason


def test_operator_request_reason_is_recognised() -> None:
    assert is_operator_request_note(note_body(reason="operator_request")) is True
    assert is_operator_request_note(note_body(reason="token_soft_threshold")) is False


def test_emergency_note_names_the_limit_and_names_itself_runtime_written() -> None:
    note = build_emergency_handoff_note(
        issue_number=ISSUE_NUMBER,
        started_at="2026-09-22T00:00:00Z",
        reason="cost_hard_limit",
        last_work_commit=WORK_REV,
        explanation="budget exhausted",
    )

    fields = parse_handoff_note_fields(note)
    assert fields["reason"] == "cost_hard_limit"
    assert fields["last_work_commit"] == WORK_REV
    assert "written by the runtime" in note
    assert is_complete_handoff_note(note, ISSUE_NUMBER) is False


def test_emergency_note_carries_no_transcribable_reason() -> None:
    note = build_emergency_handoff_note(
        issue_number=ISSUE_NUMBER,
        started_at="2026-09-22T00:00:00Z",
        reason="operator_request",
        last_work_commit=WORK_REV,
        explanation="operator asked",
    )

    assert is_operator_request_note(note) is True


def test_recovered_note_comes_from_the_models_fenced_block() -> None:
    body = note_body(reason="operator_request")
    text = f"Some prose\n```\n{body}\n```\nTrailing prose."

    assert (
        extract_recovered_handoff_note(text, ISSUE_NUMBER, "2026-09-22T00:00:00Z")
        == body.strip()
    )


def test_recovered_note_rejects_another_issues_note() -> None:
    body = note_body(issue_number=99, reason="operator_request")
    text = f"```\n{body}\n```"

    assert (
        extract_recovered_handoff_note(text, ISSUE_NUMBER, "2026-09-22T00:00:00Z")
        is None
    )


def test_recovered_note_rejects_a_disallowed_reason() -> None:
    body = note_body(reason="bogus")
    text = f"```\n{body}\n```"

    assert (
        extract_recovered_handoff_note(text, ISSUE_NUMBER, "2026-09-22T00:00:00Z")
        is None
    )


def test_recovered_note_rejects_prose_without_a_fence() -> None:
    assert (
        extract_recovered_handoff_note(
            "no fences here", ISSUE_NUMBER, "2026-09-22T00:00:00Z"
        )
        is None
    )


def test_complete_note_is_a_model_note_for_this_issue() -> None:
    assert is_complete_handoff_note(note_body(), ISSUE_NUMBER) is True
    assert is_complete_handoff_note(note_body(issue_number=99), ISSUE_NUMBER) is False


def test_classify_prefers_the_token_hard_ceiling_signal() -> None:
    from types import SimpleNamespace

    execution = SimpleNamespace(
        token_hard_ceiling_reached=True, stop_reason="end_turn", terminal_reason=""
    )

    assert classify_handoff_reason(execution) == "token_hard_ceiling"


def test_git_backed_note_naming_stale_work_is_invalid(tmp_path: Path) -> None:
    """In a real git dir, a note naming an older real commit stays rejected."""

    import subprocess

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=tmp_path,
            check=True,
            text=True,
            capture_output=True,
        )
        return completed.stdout.strip()

    git("init")
    (tmp_path / "work.txt").write_text("v1")
    git("add", "work.txt")
    git("commit", "-m", "old work")
    old_rev = git("rev-parse", "HEAD")
    (tmp_path / "work.txt").write_text("v2")
    git("commit", "-am", "new work")
    head_rev = git("rev-parse", "HEAD")

    stale_body = note_body(last_work_commit=old_rev)
    write_note(tmp_path, stale_body)
    stale_commit = AttemptCommit("f" * 40, "Handoff note")
    stale_commits = [stale_commit, AttemptCommit(head_rev, "new work")]

    stale = validate_handoff_note(
        stale_commit, stale_commits, ISSUE_NUMBER, tmp_path, old_rev
    )
    assert stale.valid is False

    fresh_body = note_body(last_work_commit=head_rev)
    write_note(tmp_path, fresh_body)
    fresh = validate_handoff_note(
        stale_commit, stale_commits, ISSUE_NUMBER, tmp_path, old_rev
    )
    assert fresh.valid is True
