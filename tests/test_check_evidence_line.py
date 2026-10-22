"""Outcome-only Evidence check line: no joined profile check command.

The attempt-result comment, the conflict attempt result comment, and the
pull-request body report only the check outcome (pass/fail/not-run plus,
for a failure, the single failing step with its position). The full
multi-step script lives in the profile configuration, so repeating it in
every comment only bloats the thread and breaks the inline-code markdown.
"""

from __future__ import annotations

from simple_coding_agent.publication import _check_evidence_line, _collapse_check_step


def test_passed_check_reports_only_the_outcome() -> None:
    line = _check_evidence_line(
        check_command="pytest && ruff check .",
        check_exit_code=0,
        check_steps=("pytest", "ruff check ."),
    )

    assert line == "- Check: passed (exit 0)"
    assert "pytest && ruff" not in line


def test_failed_check_names_only_the_failing_step_with_its_position() -> None:
    line = _check_evidence_line(
        check_command="pytest && ruff check . && mypy",
        check_exit_code=2,
        check_steps=("pytest", "ruff check .", "mypy"),
        check_failed_step="ruff check .",
        check_failed_index=2,
    )

    assert line == "- Check: failed (exit 2) at step 2/3: `ruff check .`"
    assert "pytest && ruff" not in line
    assert "mypy" not in line.replace("`ruff check .`", "")


def test_failed_step_is_collapsed_to_a_single_line() -> None:
    line = _check_evidence_line(
        check_exit_code=1,
        check_steps=("pytest", "make lint"),
        check_failed_step="make lint\n  --strict\n\t--fix",
        check_failed_index=2,
    )

    assert line == "- Check: failed (exit 1) at step 2/2: `make lint --strict --fix`"
    assert "\n" not in line
    assert "\t" not in line


def test_failed_step_is_truncated_when_long() -> None:
    long_step = "pytest " + "very-long-argument " * 40

    line = _check_evidence_line(
        check_exit_code=1,
        check_steps=("setup", long_step),
        check_failed_step=long_step,
        check_failed_index=2,
    )

    assert line.startswith("- Check: failed (exit 1) at step 2/2: `")
    assert line.endswith("…`")
    assert "\n" not in line
    assert len(line) < len(long_step)


def test_failed_step_backticks_cannot_break_the_inline_code() -> None:
    line = _check_evidence_line(
        check_exit_code=1,
        check_steps=("echo `date`",),
        check_failed_step="echo `date`",
        check_failed_index=1,
    )

    assert line == "- Check: failed (exit 1) at step 1/1: `echo 'date'`"


def test_not_run_never_prints_the_joined_command_or_exit_none() -> None:
    line = _check_evidence_line(
        check_command="pytest && ruff check .",
        check_exit_code=None,
        check_steps=("pytest", "ruff check ."),
    )

    assert line == "- Check: not run"
    assert "pytest" not in line
    assert "None" not in line


def test_not_run_keeps_the_reason_where_there_is_one() -> None:
    line = _check_evidence_line(
        check_command="not run",
        check_exit_code=None,
        check_not_run_reason="not rerun during recovery retry",
    )

    assert line == "- Check: not run (not rerun during recovery retry)"
    assert "None" not in line


def test_not_run_keeps_a_legacy_reason_carried_in_check_command() -> None:
    line = _check_evidence_line(
        check_command="not rerun during recovery retry",
        check_exit_code=None,
    )

    assert line == "- Check: not run (not rerun during recovery retry)"


def test_timed_out_check_reports_the_step_without_an_exit_code() -> None:
    line = _check_evidence_line(
        check_exit_code=None,
        check_steps=("pytest", "ruff check ."),
        check_failed_step="pytest",
        check_failed_index=1,
    )

    assert line == "- Check: failed (timed out) at step 1/2: `pytest`"
    assert "None" not in line


def test_collapse_check_step_handles_empty_input() -> None:
    assert _collapse_check_step("   \n  ") == "(empty step)"


def test_attempt_evidence_names_the_failing_step_with_its_position() -> None:
    from simple_coding_agent.command_runner import (
        CommandExecution,
        CommandResult,
        CommandStatus,
    )
    from simple_coding_agent.config import RepositoryProfile
    from simple_coding_agent.lifecycle import _attempt_evidence

    profile = RepositoryProfile(("setup",), ("pytest", "ruff check .", "mypy"), "main", 30, 20, {})
    final_check = CommandResult(
        CommandStatus.FAILED,
        (
            CommandExecution("pytest", 0, "", "", False),
            CommandExecution("ruff check .", 2, "", "boom", False),
        ),
        0.5,
    )

    evidence = _attempt_evidence(
        object(),  # type: ignore[arg-type]
        profile,
        final_check,
        0,
        "not collected",
    )

    assert evidence.check_exit_code == 2
    assert evidence.check_failed_step == "ruff check ."
    assert evidence.check_failed_index == 2
    assert evidence.check_steps == ("pytest", "ruff check .", "mypy")


def test_attempt_evidence_without_a_check_reports_not_run() -> None:
    from simple_coding_agent.config import RepositoryProfile
    from simple_coding_agent.lifecycle import _attempt_evidence

    profile = RepositoryProfile(("setup",), ("pytest", "ruff check ."), "main", 30, 20, {})

    evidence = _attempt_evidence(
        object(),  # type: ignore[arg-type]
        profile,
        None,
        0,
        "not run",
    )

    assert evidence.check_exit_code is None
    assert evidence.check_failed_step is None
    assert evidence.check_not_run_reason is None


def test_result_comment_and_pr_body_share_the_outcome_wording(tmp_path) -> None:
    from simple_coding_agent.completion import (
        AttemptOutcome,
        CompletionDecision,
        PublicationPath,
    )
    from simple_coding_agent.publication import PublicationRequest, _pull_request_body, _result_comment

    request = PublicationRequest(
        23,
        "Implement publication",
        "agent/issue-23",
        "2026-09-20T13:00:00Z",
        CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)),
        "pytest && ruff check .",
        2,
        0,
        "all clear",
        "Implemented the boundary.",
        "main",
        check_steps=("pytest", "ruff check ."),
        check_failed_step="ruff check .",
        check_failed_index=2,
    )

    comment = _result_comment(
        request,
        AttemptOutcome.COMPLETE,
        "https://example.test/tree/agent/issue-23",
        None,
        "Implemented the boundary.",
        "<!-- agent-attempt: 2026-09-20T13:00:00Z -->",
    )
    body = _pull_request_body(request)

    assert "- Check: failed (exit 2) at step 2/2: `ruff check .`" in comment
    assert "- Check: failed (exit 2) at step 2/2: `ruff check .`" in body
    assert "pytest && ruff" not in comment
    assert "pytest && ruff" not in body
    assert "exit code None" not in comment


def test_conflict_result_comment_reports_only_the_failing_step() -> None:
    from simple_coding_agent.completion import AttemptOutcome
    from simple_coding_agent.publication import ConflictPublicationRequest, _conflict_result_comment

    request = ConflictPublicationRequest(
        7,
        "Issue 7",
        "agent/issue-7",
        "2026-10-11T10:00:00Z",
        "main",
        AttemptOutcome.INCOMPLETE,
        "check_failed",
        "",
        "The profile check did not pass.",
        False,
        None,
        check_command="pytest && ruff check .",
        check_exit_code=1,
        check_steps=("pytest", "ruff check ."),
        check_failed_step="pytest",
        check_failed_index=1,
    )

    body = _conflict_result_comment(
        request,
        AttemptOutcome.INCOMPLETE,
        "check_failed",
        None,
        None,
        "The profile check did not pass.",
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->",
        None,
        None,
    )

    assert "- Check: failed (exit 1) at step 1/2: `pytest`" in body
    assert "pytest && ruff" not in body
