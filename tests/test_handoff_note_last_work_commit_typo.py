"""A mistyped ``last_work_commit`` warns instead of rejecting the handoff.

Regression tests for the run where a model dropped one character copying the
pre-filled 40-character SHA and the whole handoff was downgraded to
``incomplete``. A near-copy of the expected revision that names no commit is
evidence the model copied the right value, so it validates with a
``handoff_note_last_work_commit_mistyped`` warning; a value naming a real
commit (the stale-HEAD case, #323), or anything further away, is still
rejected.
"""

from __future__ import annotations

import os as _os
import subprocess as _subprocess
from pathlib import Path

from simple_coding_agent.git_workspace import AttemptCommit
from simple_coding_agent.lifecycle import (
    _emit_handoff_note_events,
    _validate_handoff_note,
)

ISSUE_NUMBER = 164
ATTEMPT_ID = "2026-11-04T19:37:32.830369Z"


def _git(cwd: Path, *arguments: str) -> str:
    env = dict(_os.environ, GIT_EDITOR="true")
    return _subprocess.run(
        ("git", *arguments), cwd=cwd, env=env, check=True, text=True, capture_output=True
    ).stdout.strip()


def _repo_with_work(tmp_path: Path) -> tuple[Path, str, str]:
    """A git repo with a base commit and one work commit on top."""

    repo = tmp_path / "repo"
    _git(tmp_path, "init", str(repo))
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "checkout", "-b", "main")
    (repo / "base.txt").write_text("base\n")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "Initial commit")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-b", "agent/issue-164")
    (repo / "work.txt").write_text("half done\n")
    _git(repo, "add", "work.txt")
    _git(repo, "commit", "-m", "Half-finish the work")
    work = _git(repo, "rev-parse", "HEAD")
    return repo, base, work


def _write_note(repo: Path, *, last_work_commit: str | None) -> None:
    lines = [
        f"# Handoff note: issue #{ISSUE_NUMBER}",
        "",
        f"- issue: {ISSUE_NUMBER}",
        f"- started_at: {ATTEMPT_ID}",
        "- reason: token_soft_threshold",
    ]
    if last_work_commit is not None:
        lines.append(f"- last_work_commit: {last_work_commit}")
    lines += ["", "Halfway through the migration.", ""]
    note_dir = repo / ".agent" / "handoff"
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / f"{ISSUE_NUMBER}.md").write_text("\n".join(lines))


def _commit_note(repo: Path) -> str:
    _git(repo, "add", f".agent/handoff/{ISSUE_NUMBER}.md")
    _git(repo, "commit", "-m", f"Handoff note: issue #{ISSUE_NUMBER}")
    return _git(repo, "rev-parse", "HEAD")


def _validate(repo: Path, base: str, work: str, note: str):
    return _validate_handoff_note(
        AttemptCommit(revision=note, subject=f"Handoff note: issue #{ISSUE_NUMBER}"),
        [
            AttemptCommit(revision=note, subject="Handoff note"),
            AttemptCommit(revision=work, subject="Model work"),
        ],
        ISSUE_NUMBER,
        repo,
        base,
    )


class _EventRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, str | None, int | None]] = []

    def __call__(
        self,
        event: str,
        detail: str,
        *,
        level: str | None = None,
        issue_number: int | None = None,
    ) -> None:
        self.events.append((event, detail, level, issue_number))


def _flip_one_hex_char(value: str, index: int) -> str:
    flipped = "0" if value[index] != "0" else "1"
    return value[:index] + flipped + value[index + 1 :]


# ---------------------------------------------------------------------------
# Transcription slips validate with a warning
# ---------------------------------------------------------------------------


def test_one_dropped_character_is_accepted_with_warning(tmp_path: Path) -> None:
    repo, base, work = _repo_with_work(tmp_path)
    # The production shape: `...77186b...` became `...7716b...`.
    mistyped = work[:14] + work[15:]
    assert len(mistyped) == 39
    _write_note(repo, last_work_commit=mistyped)
    note = _commit_note(repo)

    validation = _validate(repo, base, work, note)

    assert validation.valid
    assert validation.last_work_commit_warning is not None
    assert mistyped in validation.last_work_commit_warning
    assert work in validation.last_work_commit_warning


def test_one_changed_character_is_accepted_with_warning(tmp_path: Path) -> None:
    repo, base, work = _repo_with_work(tmp_path)
    mistyped = _flip_one_hex_char(work, 10)
    _write_note(repo, last_work_commit=mistyped)
    note = _commit_note(repo)

    validation = _validate(repo, base, work, note)

    assert validation.valid
    assert validation.last_work_commit_warning is not None
    assert mistyped in validation.last_work_commit_warning
    assert work in validation.last_work_commit_warning


def test_one_inserted_character_is_accepted_with_warning(tmp_path: Path) -> None:
    repo, base, work = _repo_with_work(tmp_path)
    mistyped = work[:20] + "a" + work[20:]
    assert len(mistyped) == 41
    _write_note(repo, last_work_commit=mistyped)
    note = _commit_note(repo)

    validation = _validate(repo, base, work, note)

    assert validation.valid
    assert validation.last_work_commit_warning is not None


def test_two_edits_are_accepted_with_warning(tmp_path: Path) -> None:
    repo, base, work = _repo_with_work(tmp_path)
    mistyped = _flip_one_hex_char(_flip_one_hex_char(work, 5), 30)
    _write_note(repo, last_work_commit=mistyped)
    note = _commit_note(repo)

    validation = _validate(repo, base, work, note)

    assert validation.valid
    assert validation.last_work_commit_warning is not None


def test_changed_character_in_short_prefix_is_accepted(tmp_path: Path) -> None:
    repo, base, work = _repo_with_work(tmp_path)
    short = work[:12]
    mistyped = _flip_one_hex_char(short, 4)
    assert len(mistyped) == 12
    _write_note(repo, last_work_commit=mistyped)
    note = _commit_note(repo)

    validation = _validate(repo, base, work, note)

    assert validation.valid
    assert validation.last_work_commit_warning is not None
    assert mistyped in validation.last_work_commit_warning
    assert work in validation.last_work_commit_warning


def test_mistyped_note_warns_alongside_started_at_warning(tmp_path: Path) -> None:
    repo, base, work = _repo_with_work(tmp_path)
    mistyped = _flip_one_hex_char(work, 10)
    lines = [
        f"# Handoff note: issue #{ISSUE_NUMBER}",
        "",
        f"- issue: {ISSUE_NUMBER}",
        "- reason: token_soft_threshold",
        f"- last_work_commit: {mistyped}",
        "",
        "Halfway through the migration.",
        "",
    ]
    note_dir = repo / ".agent" / "handoff"
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / f"{ISSUE_NUMBER}.md").write_text("\n".join(lines))
    note = _commit_note(repo)

    validation = _validate(repo, base, work, note)

    assert validation.valid
    assert validation.started_at_warning is not None
    assert validation.last_work_commit_warning is not None


# ---------------------------------------------------------------------------
# Anything else is still rejected
# ---------------------------------------------------------------------------


def test_stale_commit_resolving_to_real_commit_is_rejected(tmp_path: Path) -> None:
    """The #323 shape: the note names an older real commit, not a typo."""

    repo, base, work = _repo_with_work(tmp_path)
    (repo / "more.txt").write_text("more work\n")
    _git(repo, "add", "more.txt")
    _git(repo, "commit", "-m", "More work after the note header was read")
    newer = _git(repo, "rev-parse", "HEAD")
    assert newer != work
    # The note names the older work commit while HEAD has moved on.
    _write_note(repo, last_work_commit=work)
    note = _commit_note(repo)

    validation = _validate_handoff_note(
        AttemptCommit(revision=note, subject=f"Handoff note: issue #{ISSUE_NUMBER}"),
        [
            AttemptCommit(revision=note, subject="Handoff note"),
            AttemptCommit(revision=newer, subject="More work"),
            AttemptCommit(revision=work, subject="Model work"),
        ],
        ISSUE_NUMBER,
        repo,
        base,
    )

    assert not validation.valid
    assert "last_work_commit" in validation.reason
    assert validation.last_work_commit_warning is None


def test_three_edits_away_is_rejected(tmp_path: Path) -> None:
    repo, base, work = _repo_with_work(tmp_path)
    mistyped = _flip_one_hex_char(_flip_one_hex_char(_flip_one_hex_char(work, 5), 15), 25)
    _write_note(repo, last_work_commit=mistyped)
    note = _commit_note(repo)

    validation = _validate(repo, base, work, note)

    assert not validation.valid
    assert "last_work_commit" in validation.reason
    assert validation.last_work_commit_warning is None


def test_unrelated_hex_is_rejected(tmp_path: Path) -> None:
    repo, base, work = _repo_with_work(tmp_path)
    _write_note(repo, last_work_commit="deadbeef" * 5)
    note = _commit_note(repo)

    validation = _validate(repo, base, work, note)

    assert not validation.valid
    assert "last_work_commit" in validation.reason


def test_non_hex_value_is_rejected(tmp_path: Path) -> None:
    repo, base, work = _repo_with_work(tmp_path)
    _write_note(repo, last_work_commit="not-a-sha-at-all")
    note = _commit_note(repo)

    validation = _validate(repo, base, work, note)

    assert not validation.valid
    assert "last_work_commit" in validation.reason
    assert validation.last_work_commit_warning is None


def test_missing_and_short_values_are_rejected(tmp_path: Path) -> None:
    repo, base, work = _repo_with_work(tmp_path)

    _write_note(repo, last_work_commit=None)
    note = _commit_note(repo)
    missing = _validate(repo, base, work, note)
    assert not missing.valid
    assert "last_work_commit" in missing.reason

    _write_note(repo, last_work_commit=work[:6])
    note = _commit_note(repo)
    short = _validate(repo, base, work, note)
    assert not short.valid
    assert "last_work_commit" in short.reason


def test_exact_prefixes_are_accepted_without_warning(tmp_path: Path) -> None:
    repo, base, work = _repo_with_work(tmp_path)
    for value in (work, work[:12], work[:7]):
        _write_note(repo, last_work_commit=value)
        note = _commit_note(repo)

        validation = _validate(repo, base, work, note)

        assert validation.valid, value
        assert validation.last_work_commit_warning is None, value


# ---------------------------------------------------------------------------
# Events and the prefill
# ---------------------------------------------------------------------------


def test_mistyped_note_emits_mistyped_event(tmp_path: Path) -> None:
    repo, base, work = _repo_with_work(tmp_path)
    mistyped = work[:14] + work[15:]
    _write_note(repo, last_work_commit=mistyped)
    note = _commit_note(repo)
    recorder = _EventRecorder()

    validation = _validate(repo, base, work, note)
    _emit_handoff_note_events(recorder, ISSUE_NUMBER, validation)

    assert recorder.events == [
        (
            "handoff_note_last_work_commit_mistyped",
            validation.last_work_commit_warning,
            "WARNING",
            ISSUE_NUMBER,
        )
    ]
    assert mistyped in recorder.events[0][1]
    assert work in recorder.events[0][1]


def test_stale_note_emits_only_rejected_event(tmp_path: Path) -> None:
    repo, base, work = _repo_with_work(tmp_path)
    _write_note(repo, last_work_commit="deadbeef" * 5)
    note = _commit_note(repo)
    recorder = _EventRecorder()

    validation = _validate(repo, base, work, note)
    _emit_handoff_note_events(recorder, ISSUE_NUMBER, validation)

    assert [event for event, _, _, _ in recorder.events] == ["handoff_note_rejected"]


def test_clean_note_emits_no_mistyped_event(tmp_path: Path) -> None:
    repo, base, work = _repo_with_work(tmp_path)
    _write_note(repo, last_work_commit=work)
    note = _commit_note(repo)
    recorder = _EventRecorder()

    _emit_handoff_note_events(recorder, ISSUE_NUMBER, _validate(repo, base, work, note))

    assert recorder.events == []


def test_prefill_carries_twelve_character_sha_that_validates(tmp_path: Path) -> None:
    from simple_coding_agent.model_execution import _token_handoff_prefill

    repo, base, work = _repo_with_work(tmp_path)

    header = _token_handoff_prefill(ISSUE_NUMBER, ATTEMPT_ID, work)
    prefilled = next(
        line.split(":", 1)[1].strip()
        for line in header.splitlines()
        if line.startswith("- last_work_commit:")
    )

    assert len(prefilled) == 12
    assert prefilled == work[:12]

    _write_note(repo, last_work_commit=prefilled)
    before = (repo / ".agent" / "handoff" / f"{ISSUE_NUMBER}.md").read_text()
    note = _commit_note(repo)

    validation = _validate(repo, base, work, note)

    assert validation.valid
    assert validation.last_work_commit_warning is None
    # The runtime never rewrites the note's `last_work_commit`.
    assert (repo / ".agent" / "handoff" / f"{ISSUE_NUMBER}.md").read_text() == before


def test_prefill_unknown_fallback_is_unchanged() -> None:
    from simple_coding_agent.model_execution import _token_handoff_prefill

    header = _token_handoff_prefill(ISSUE_NUMBER, ATTEMPT_ID, None)

    assert "- last_work_commit: (unknown)" in header
