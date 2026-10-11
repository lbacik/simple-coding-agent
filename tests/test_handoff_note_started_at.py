"""``started_at`` is informational, never a handoff gate.

Regression tests for the production incident where a model-written handoff
note carried ``started_at: 2026-10-07 (UTC; exact time unavailable in
handoff mode)`` and the whole handoff was downgraded to ``incomplete``.
``started_at`` is the attempt id owned by the driving process: a missing
or malformed value warns instead of rejecting, every rejection is logged,
and every handoff prompt given to the model carries the attempt id.
"""

from __future__ import annotations

from pathlib import Path

from simple_coding_agent.git_workspace import AttemptCommit
from simple_coding_agent.lifecycle import (
    _emit_handoff_note_events,
    _validate_handoff_note,
)
from simple_coding_agent.model_execution import (
    _cost_handoff_followup_prompt,
    _cost_handoff_instruction,
    _limit_handoff_followup_prompt,
    _operator_handoff_followup_prompt,
    _operator_handoff_instruction,
)

ISSUE_NUMBER = 174
ATTEMPT_ID = "2026-10-07T15:20:33.118250Z"
# The exact malformed value the model wrote in the production incident.
MALFORMED_STARTED_AT = "2026-10-07 (UTC; exact time unavailable in handoff mode)"

WORK_COMMIT = "c5a2784aaaabbbbccccddddeeeeffff000011112222"
NOTE_COMMIT = "2452078aaaabbbbccccddddeeeeffff000011112222"


def _write_note(
    working_directory: Path,
    *,
    started_at: str | None,
    issue: int = ISSUE_NUMBER,
    reason: str = "token_soft_threshold",
    last_work_commit: str = WORK_COMMIT,
    title_issue: int = ISSUE_NUMBER,
) -> None:
    lines = [
        f"# Handoff note: issue #{title_issue}",
        "",
        f"- issue: {issue}",
    ]
    if started_at is not None:
        lines.append(f"- started_at: {started_at}")
    lines += [
        f"- reason: {reason}",
        f"- last_work_commit: {last_work_commit}",
        "",
        "Halfway through the migration.",
        "",
    ]
    note_dir = working_directory / ".agent" / "handoff"
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / f"{ISSUE_NUMBER}.md").write_text("\n".join(lines))


def _validate(
    working_directory: Path,
    *,
    note_commit_revision: str | None = NOTE_COMMIT,
):
    note_commit = (
        AttemptCommit(
            revision=note_commit_revision,
            subject=f"Handoff note: issue #{ISSUE_NUMBER}",
        )
        if note_commit_revision is not None
        else None
    )
    commits = (
        [
            AttemptCommit(revision=NOTE_COMMIT, subject="Handoff note"),
            AttemptCommit(revision=WORK_COMMIT, subject="Model work"),
        ]
        if note_commit is not None
        else []
    )
    return _validate_handoff_note(
        note_commit,
        commits,
        ISSUE_NUMBER,
        working_directory,
        "base-revision-0",
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


# ---------------------------------------------------------------------------
# Validator: ``started_at`` never rejects
# ---------------------------------------------------------------------------


def test_malformed_started_at_still_valid_with_warning(tmp_path: Path) -> None:
    _write_note(tmp_path, started_at=MALFORMED_STARTED_AT)

    validation = _validate(tmp_path)

    assert validation.valid
    assert validation.started_at_warning is not None
    assert MALFORMED_STARTED_AT in validation.started_at_warning
    assert str(ISSUE_NUMBER) in validation.started_at_warning


def test_missing_started_at_still_valid_with_warning(tmp_path: Path) -> None:
    _write_note(tmp_path, started_at=None)

    validation = _validate(tmp_path)

    assert validation.valid
    assert validation.started_at_warning is not None
    assert "missing" in validation.started_at_warning


def test_well_formed_started_at_emits_no_warning(tmp_path: Path) -> None:
    _write_note(tmp_path, started_at=ATTEMPT_ID)

    validation = _validate(tmp_path)

    assert validation.valid
    assert validation.started_at_warning is None


# ---------------------------------------------------------------------------
# Validator: the other checks stay strict
# ---------------------------------------------------------------------------


def test_wrong_issue_is_rejected(tmp_path: Path) -> None:
    _write_note(tmp_path, started_at=ATTEMPT_ID, issue=999)

    validation = _validate(tmp_path)

    assert not validation.valid
    assert validation.reason == "Handoff note is missing a matching `issue` field."


def test_invalid_reason_is_rejected(tmp_path: Path) -> None:
    _write_note(tmp_path, started_at=ATTEMPT_ID, reason="gave_up")

    validation = _validate(tmp_path)

    assert not validation.valid
    assert validation.reason == "Handoff note is missing a valid `reason` field."


def test_mismatched_last_work_commit_is_rejected(tmp_path: Path) -> None:
    _write_note(tmp_path, started_at=ATTEMPT_ID, last_work_commit="deadbee")

    validation = _validate(tmp_path)

    assert not validation.valid
    assert "last_work_commit" in validation.reason


def test_rejected_note_keeps_started_at_warning_alongside(tmp_path: Path) -> None:
    _write_note(tmp_path, started_at=None, reason="gave_up")

    validation = _validate(tmp_path)

    assert not validation.valid
    assert validation.started_at_warning is not None


def test_missing_note_commit_is_rejected_without_started_at_warning(
    tmp_path: Path,
) -> None:
    validation = _validate(tmp_path, note_commit_revision=None)

    assert not validation.valid
    assert "no handoff note commit" in validation.reason
    assert validation.started_at_warning is None


# ---------------------------------------------------------------------------
# Events: every verdict is visible in the event stream
# ---------------------------------------------------------------------------


def test_started_at_warning_emits_warning_event(tmp_path: Path) -> None:
    _write_note(tmp_path, started_at=MALFORMED_STARTED_AT)
    recorder = _EventRecorder()

    _emit_handoff_note_events(recorder, ISSUE_NUMBER, _validate(tmp_path))

    assert recorder.events == [
        (
            "handoff_note_started_at_invalid",
            _validate(tmp_path).started_at_warning,
            "WARNING",
            ISSUE_NUMBER,
        )
    ]


def test_clean_note_emits_no_events(tmp_path: Path) -> None:
    _write_note(tmp_path, started_at=ATTEMPT_ID)
    recorder = _EventRecorder()

    _emit_handoff_note_events(recorder, ISSUE_NUMBER, _validate(tmp_path))

    assert recorder.events == []


def test_rejected_note_emits_rejected_event_with_reason(tmp_path: Path) -> None:
    _write_note(tmp_path, started_at=ATTEMPT_ID, reason="gave_up")
    recorder = _EventRecorder()

    validation = _validate(tmp_path)
    _emit_handoff_note_events(recorder, ISSUE_NUMBER, validation)

    assert recorder.events == [
        ("handoff_note_rejected", validation.reason, "WARNING", ISSUE_NUMBER)
    ]


def test_rejected_note_with_bad_started_at_emits_both_events(
    tmp_path: Path,
) -> None:
    _write_note(tmp_path, started_at=None, reason="gave_up")
    recorder = _EventRecorder()

    validation = _validate(tmp_path)
    _emit_handoff_note_events(recorder, ISSUE_NUMBER, validation)

    assert [event for event, _, _, _ in recorder.events] == [
        "handoff_note_started_at_invalid",
        "handoff_note_rejected",
    ]
    assert all(level == "WARNING" for _, _, level, _ in recorder.events)
    assert all(number == ISSUE_NUMBER for _, _, _, number in recorder.events)


# ---------------------------------------------------------------------------
# Prompts: the model is given the attempt id
# ---------------------------------------------------------------------------


def test_soft_threshold_instruction_and_followup_carry_attempt_id() -> None:
    assert ATTEMPT_ID in _cost_handoff_instruction(ISSUE_NUMBER, ATTEMPT_ID)
    assert ATTEMPT_ID in _cost_handoff_followup_prompt(ISSUE_NUMBER, ATTEMPT_ID)


def test_operator_instruction_and_followup_carry_attempt_id() -> None:
    assert ATTEMPT_ID in _operator_handoff_instruction(ISSUE_NUMBER, ATTEMPT_ID)
    assert ATTEMPT_ID in _operator_handoff_followup_prompt(ISSUE_NUMBER, ATTEMPT_ID)


def test_limit_followup_carries_attempt_id() -> None:
    assert ATTEMPT_ID in _limit_handoff_followup_prompt(ISSUE_NUMBER, ATTEMPT_ID)


def test_prompts_name_started_at_as_the_verbatim_value() -> None:
    for prompt in (
        _cost_handoff_instruction(ISSUE_NUMBER, ATTEMPT_ID),
        _cost_handoff_followup_prompt(ISSUE_NUMBER, ATTEMPT_ID),
        _operator_handoff_instruction(ISSUE_NUMBER, ATTEMPT_ID),
        _limit_handoff_followup_prompt(ISSUE_NUMBER, ATTEMPT_ID),
        _operator_handoff_followup_prompt(ISSUE_NUMBER, ATTEMPT_ID),
    ):
        assert "started_at" in prompt


def test_prompts_without_attempt_id_say_to_leave_started_at_out() -> None:
    for prompt in (
        _cost_handoff_instruction(ISSUE_NUMBER),
        _cost_handoff_followup_prompt(ISSUE_NUMBER),
        _operator_handoff_instruction(ISSUE_NUMBER),
        _limit_handoff_followup_prompt(ISSUE_NUMBER),
        _operator_handoff_followup_prompt(ISSUE_NUMBER),
    ):
        assert "started_at" in prompt
        assert ATTEMPT_ID not in prompt
