"""Behaviour tests for the model attempt runner's starting-prompt assembly."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.completion import (
    AttemptOutcome,
    CompletionDecision,
    CompletionEvaluator,
    PublicationPath,
)
from simple_coding_agent.config import RepositoryProfile
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.git_workspace import PreparedAttempt, RebaseConflictError
from simple_coding_agent.lifecycle import ModelAttemptRunner, _build_starting_prompt
from simple_coding_agent.model_execution import (
    ModelExecution,
    ModelExecutionStatus,
    SkillEvent,
)
from tests.fakes import InMemoryWorkspace


def seeded_workspace(root: Path, *subjects: str) -> InMemoryWorkspace:
    """Build the shared workspace double with retained-branch model commits."""

    workspace = InMemoryWorkspace(root)
    for subject in subjects:
        workspace.model_commit(subject, files=(subject,))
    return workspace


def test_prompt_appends_trusted_comments_in_order(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    comments = ("please also handle timeouts", "keep the old retry behaviour")
    runner = build_runner(tmp_path, model_executor=executor, issue_comments=lambda issue: comments)

    runner(claim(issue_body="Fix the parser."), profile(), FakePrepared())

    assert executor.captured_prompt is not None
    assert executor.captured_prompt.startswith(
        "Fix the parser.\n\nplease also handle timeouts\n\nkeep the old retry behaviour"
    )
    first = executor.captured_prompt.index("please also handle timeouts")
    second = executor.captured_prompt.index("keep the old retry behaviour")
    assert first < second
    assert "## Completion gate" in executor.captured_prompt
    assert "1. pytest" in executor.captured_prompt


def test_prompt_degrades_to_the_bare_issue_body_with_no_comments_or_continuation(
    tmp_path: Path,
) -> None:
    executor = FakeModelExecutor()
    runner = build_runner(tmp_path, model_executor=executor)

    runner(claim(issue_body="Fix the parser."), profile(), FakePrepared())

    assert executor.captured_prompt is not None
    assert executor.captured_prompt.startswith("Fix the parser.")
    assert "## Completion gate" in executor.captured_prompt
    assert "1. pytest" in executor.captured_prompt


def test_continuation_adds_only_a_pointer_to_the_handoff_note(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    workspace = seeded_workspace(tmp_path / "repo", "Previous attempt work")
    runner = build_runner(tmp_path, model_executor=executor, workspace=workspace, issue_number=24)

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    assert executor.captured_prompt is not None
    assert (
        "Fix the parser.\n\n"
        "This is a continued attempt. Read the current Handoff note at "
        ".agent/handoff/24.md before implementing: it is the authoritative, "
        "self-contained record of the previous attempt's state, remaining "
        "work, and evidence, and it supersedes every historical handoff "
        "result comment."
    ) in executor.captured_prompt
    assert "## Completion gate" in executor.captured_prompt
    pointer = executor.captured_prompt.index(".agent/handoff/24.md")
    gate = executor.captured_prompt.index("## Completion gate")
    assert pointer < gate


def test_missing_continuation_branch_skips_the_model_when_label_present(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    runner = build_runner(tmp_path, model_executor=executor)

    evidence = runner(
        claim(issue_number=24, labels=frozenset({"ready-for-agent", "round-finished"})),
        profile(),
        FakePrepared(restored_from_remote=False),
    )

    assert executor.captured_prompt is None
    assert evidence.decision.outcome is AttemptOutcome.INCOMPLETE
    assert not evidence.decision.publication_eligible
    assert "agent/issue-24" in evidence.details
    assert "not found on origin" in evidence.details


def test_missing_continuation_branch_skips_the_model_with_a_prior_handoff_comment(
    tmp_path: Path,
) -> None:
    executor = FakeModelExecutor()
    comments = ("## Agent Attempt Result: incomplete\n<!-- agent-attempt: earlier -->",)
    runner = build_runner(tmp_path, model_executor=executor, issue_comments=lambda issue: comments)

    evidence = runner(claim(issue_number=24), profile(), FakePrepared(restored_from_remote=False))

    assert executor.captured_prompt is None
    assert evidence.decision.outcome is AttemptOutcome.INCOMPLETE


def test_incomplete_without_a_published_branch_runs_the_model_normally(tmp_path: Path) -> None:
    """An incomplete attempt that never published a branch is not a handoff (issue #84).

    A baseline or setup failure posts ``no branch created``; the next attempt
    must start fresh from the base branch instead of short-circuiting with
    ``continuation_branch_missing``.
    """

    executor = FakeModelExecutor()
    comments = (
        "## Agent Attempt Result: incomplete\n\n"
        "**Branch**: no branch created\n"
        "**PR**: none\n\n"
        "### Details\n"
        "Baseline check failed before model execution.\n\n"
        "<!-- agent-attempt: earlier -->",
    )
    runner = build_runner(tmp_path, model_executor=executor, issue_comments=lambda issue: comments)

    runner(claim(issue_number=24), profile(), FakePrepared(restored_from_remote=False))

    assert executor.captured_prompt is not None


def test_missing_continuation_branch_skips_the_model_with_a_prior_published_handoff(
    tmp_path: Path,
) -> None:
    """A prior handoff that published a branch still expects a continuation (#48)."""

    executor = FakeModelExecutor()
    comments = (
        "## Agent Attempt Result: handoff\n\n"
        "**Branch**: [agent/issue-24](https://github.com/o/r/tree/agent/issue-24)\n"
        "**PR**: none\n\n"
        "### Details\n"
        "Cost threshold reached; handed off.\n\n"
        "<!-- agent-attempt: earlier -->",
    )
    runner = build_runner(tmp_path, model_executor=executor, issue_comments=lambda issue: comments)

    evidence = runner(claim(issue_number=24), profile(), FakePrepared(restored_from_remote=False))

    assert executor.captured_prompt is None
    assert evidence.decision.outcome is AttemptOutcome.INCOMPLETE


def test_incomplete_with_a_published_branch_still_expects_continuation(
    tmp_path: Path,
) -> None:
    """An incomplete attempt that pushed partial work still promises a branch.

    Only ``no branch created`` comments are excluded; a published branch link
    keeps the #48 missing-branch short-circuit so deleted work is not silently
    abandoned.
    """

    executor = FakeModelExecutor()
    comments = (
        "## Agent Attempt Result: incomplete\n\n"
        "**Branch**: [agent/issue-24](https://github.com/o/r/tree/agent/issue-24)\n"
        "**PR**: none\n\n"
        "### Details\n"
        "Final check failed with work preserved on the branch.\n\n"
        "<!-- agent-attempt: earlier -->",
    )
    runner = build_runner(tmp_path, model_executor=executor, issue_comments=lambda issue: comments)

    evidence = runner(claim(issue_number=24), profile(), FakePrepared(restored_from_remote=False))

    assert executor.captured_prompt is None
    assert evidence.decision.outcome is AttemptOutcome.INCOMPLETE


def test_continuation_missing_comment_does_not_trigger_another_continuation_missing(
    tmp_path: Path,
) -> None:
    """The ``continuation_branch_missing`` rejection must never feed itself (issue #84)."""

    executor = FakeModelExecutor()
    comments = (
        "## Agent Attempt Result: incomplete\n\n"
        "**Branch**: no branch created\n"
        "**PR**: none\n\n"
        "### Details\n"
        "Expected continuation branch `agent/issue-24` was not found on origin, so it "
        "could not be restored. A human should inspect this issue and decide next steps.\n\n"
        "<!-- agent-attempt: earlier -->",
    )
    runner = build_runner(tmp_path, model_executor=executor, issue_comments=lambda issue: comments)

    runner(claim(issue_number=24), profile(), FakePrepared(restored_from_remote=False))

    assert executor.captured_prompt is not None


def test_restored_continuation_branch_runs_the_model_normally(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    runner = build_runner(tmp_path, model_executor=executor)

    runner(
        claim(issue_number=24, labels=frozenset({"ready-for-agent", "round-finished"})),
        profile(),
        FakePrepared(restored_from_remote=True),
    )

    assert executor.captured_prompt is not None


def test_normal_claim_with_no_continuation_signal_runs_the_model_normally(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    runner = build_runner(tmp_path, model_executor=executor)

    runner(claim(issue_number=24), profile(), FakePrepared(restored_from_remote=False))

    assert executor.captured_prompt is not None


def test_defers_setup_until_after_the_model_resolves_conflicts(tmp_path: Path) -> None:
    """Setup commands must never observe a half-rebased tree.

    A retained branch rebased onto a new base can be left with conflict
    markers; running setup (which parses every source file) against that
    tree fails spuriously, e.g. a PHP ParseError on `<<<<<<<`. The model
    must run first to resolve the conflicts, and setup runs after.
    """

    calls: list[str] = []
    executor = FakeModelExecutor(status=ModelExecutionStatus.SUCCEEDED, calls=calls)
    workspace = seeded_workspace(tmp_path / "repo", "Retained work")
    workspace.leave_conflict(files=("SubscriptionController.php",))
    executor.after_execute = workspace.note_model_finished
    verifier = FakeVerifier(calls=calls)
    events: list[tuple[str, str]] = []
    runner = build_runner(
        tmp_path,
        model_executor=executor,
        workspace=workspace,
        verifier=verifier,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, level)
        ),
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    assert calls == ["execute", "prepare"]
    assert ("setup_deferred_unresolved_conflicts", "WARNING") in events
    assert "unresolved merge conflicts" in executor.captured_prompt
    assert "setup_succeeded" in [event for event, _ in events]


def test_successful_conflict_resolution_logs_started_and_succeeded_with_files(
    tmp_path: Path,
) -> None:
    """The attempt record shows the model resolved a rebase conflict."""

    executor = FakeModelExecutor(status=ModelExecutionStatus.SUCCEEDED)
    workspace = seeded_workspace(tmp_path / "repo", "Retained work")
    workspace.leave_conflict(files=("SubscriptionController.php",))
    executor.after_execute = workspace.note_model_finished
    events: list[tuple[str, str, str]] = []
    runner = build_runner(
        tmp_path,
        model_executor=executor,
        workspace=workspace,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level)
        ),
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    started = [detail for event, detail, _ in events if event == "rebase_resolution_started"]
    succeeded = [detail for event, detail, _ in events if event == "rebase_resolution_succeeded"]
    assert len(started) == 1
    assert "SubscriptionController.php" in started[0]
    assert len(succeeded) == 1
    assert "SubscriptionController.php" in succeeded[0]
    assert workspace.abort_calls == []
    assert "git rebase --continue" in (executor.captured_prompt or "")


def test_unresolved_conflicts_after_the_model_abort_restore_and_raise(
    tmp_path: Path,
) -> None:
    """The fallback matches the old abort-and-report behavior exactly.

    When the model leaves any unresolved state behind, the rebase is
    aborted, the branch restored, and the attempt reports
    infrastructure_error with a rebase_conflict cause — without setup ever
    running against the half-merged tree.
    """

    calls: list[str] = []
    executor = FakeModelExecutor(status=ModelExecutionStatus.SUCCEEDED, calls=calls)
    workspace = InMemoryWorkspace(tmp_path / "repo", resolve_after_model=False)
    workspace.model_commit("Retained work", files=("Retained work",))
    workspace.leave_conflict(files=("SubscriptionController.php",))
    executor.after_execute = workspace.note_model_finished
    verifier = FakeVerifier(calls=calls)
    events: list[tuple[str, str, str]] = []
    runner = build_runner(
        tmp_path,
        model_executor=executor,
        workspace=workspace,
        verifier=verifier,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level)
        ),
    )
    prepared = FakePrepared()

    with pytest.raises(RebaseConflictError) as caught:
        runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), prepared)

    assert calls == ["execute"]
    assert len(workspace.abort_calls) == 1
    assert workspace.abort_calls[0] is prepared
    assert any(event == "rebase_resolution_started" for event, _, _ in events)
    failed = [detail for event, detail, _ in events if event == "rebase_resolution_failed"]
    assert len(failed) == 1
    assert "SubscriptionController.php" in failed[0]
    assert "rebase_conflict" in str(caught.value)
    assert "setup was not started" in str(caught.value).lower() or "Setup was not started" in str(
        caught.value
    )
    assert caught.value.branch == "agent/issue-24"
    assert caught.value.base_branch == "main"
    assert caught.value.conflicted_files == ("SubscriptionController.php",)


def test_fallback_falls_back_to_the_prepare_snapshot_for_the_file_list(
    tmp_path: Path,
) -> None:
    """A conflict state without live file names still reports the file list."""

    executor = FakeModelExecutor(status=ModelExecutionStatus.SUCCEEDED)
    workspace = seeded_workspace(tmp_path / "repo", "Retained work")
    workspace.leave_conflict(files=())
    events: list[tuple[str, str, str]] = []
    runner = build_runner(
        tmp_path,
        model_executor=executor,
        workspace=workspace,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level)
        ),
    )
    prepared = FakePrepared(rebase_conflicts=("SubscriptionController.php",))

    with pytest.raises(RebaseConflictError) as caught:
        runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), prepared)

    assert caught.value.conflicted_files == ("SubscriptionController.php",)
    started = [detail for event, detail, _ in events if event == "rebase_resolution_started"]
    assert len(started) == 1
    assert "SubscriptionController.php" in started[0]
    assert workspace.abort_calls == [prepared]


def test_unreadable_verification_state_fails_safe_without_running_setup(
    tmp_path: Path,
) -> None:
    """A verification problem with no live file names still aborts and restores.

    The fail-safe rule lives in the adapter's ``conflict_state``: an
    unverifiable tree reports ``in_progress`` with a verification problem,
    so the runner aborts the rebase and restores the branch instead of
    running setup against an unverified tree.
    """

    calls: list[str] = []
    executor = FakeModelExecutor(status=ModelExecutionStatus.SUCCEEDED, calls=calls)
    workspace = seeded_workspace(tmp_path / "repo", "Retained work")
    workspace.leave_conflict(
        files=(),
        problems=("Conflict-resolution verification failed; treating the tree as conflicted.",),
    )
    verifier = FakeVerifier(calls=calls)
    events: list[tuple[str, str, str]] = []
    runner = build_runner(
        tmp_path,
        model_executor=executor,
        workspace=workspace,
        verifier=verifier,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level)
        ),
    )
    prepared = FakePrepared(rebase_conflicts=("SubscriptionController.php",))

    with pytest.raises(RebaseConflictError):
        runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), prepared)

    assert calls == ["execute"]
    assert workspace.abort_calls == [prepared]
    assert any(event == "rebase_resolution_started" for event, _, _ in events)
    assert any(event == "rebase_resolution_failed" for event, _, _ in events)


def test_deferred_setup_failure_still_reports_without_a_final_check(tmp_path: Path) -> None:
    calls: list[str] = []
    executor = FakeModelExecutor(status=ModelExecutionStatus.SUCCEEDED, calls=calls)
    workspace = seeded_workspace(tmp_path / "repo", "Retained work")
    workspace.leave_conflict(files=("SubscriptionController.php",))
    executor.after_execute = workspace.note_model_finished
    verifier = FakeVerifier(calls=calls, setup_succeeded=False)
    events: list[str] = []
    runner = build_runner(
        tmp_path,
        model_executor=executor,
        workspace=workspace,
        verifier=verifier,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(event),
    )

    evidence = runner(
        claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared()
    )

    assert calls == ["execute", "prepare"]
    assert "setup_failed" in events
    assert "final_check_started" not in events
    assert evidence.check_exit_code is None


def test_logs_final_check_start_and_success(tmp_path: Path) -> None:
    executor = FakeModelExecutor(status=ModelExecutionStatus.SUCCEEDED)
    workspace = seeded_workspace(tmp_path / "repo", "Implemented")
    events: list[str] = []
    runner = build_runner(
        tmp_path,
        model_executor=executor,
        workspace=workspace,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(event),
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    assert events[-3:] == ["final_check_started", "final_check_succeeded", "completion_decided"]


def test_writes_final_check_markers_to_the_archive(tmp_path: Path) -> None:
    executor = FakeModelExecutor(status=ModelExecutionStatus.SUCCEEDED)
    workspace = seeded_workspace(tmp_path / "repo", "Implemented")
    archive = FakeArchive()
    runner = build_runner(tmp_path, model_executor=executor, workspace=workspace)
    runner._attempt_archive_factory = lambda issue_number, started_at: archive

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    assert "final_check_started" in archive.merged
    assert "final_check_finished" in archive.merged


def build_runner(
    tmp_path: Path,
    *,
    model_executor: "FakeModelExecutor",
    workspace: "InMemoryWorkspace | None" = None,
    verifier: "FakeVerifier | None" = None,
    issue_comments=lambda issue: (),
    issue_number: int = 24,
    event_log=None,
    prompt_redactions: tuple[str, ...] = (),
) -> ModelAttemptRunner:
    attempt_state = AttemptStateStore(tmp_path)
    attempt_state.start(issue_number=issue_number, branch=f"agent/issue-{issue_number}")
    attempt_state.transition(AttemptPhase.SETUP)
    kwargs = {} if event_log is None else {"event_log": event_log}
    return ModelAttemptRunner(
        attempt_state=attempt_state,
        workspace=workspace or InMemoryWorkspace(tmp_path / "repo"),
        verifier=verifier or FakeVerifier(),
        evaluator=FakeEvaluator(),
        model_executor=model_executor,
        issue_comments=issue_comments,
        prompt_redactions=prompt_redactions,
        **kwargs,
    )


def test_logs_setup_baseline_and_model_dispatch_stages(tmp_path: Path) -> None:
    """Setup, baseline check, and model dispatch previously ran silently on

    success; only failures were logged. An operator watching the log stream
    couldn't tell an attempt was progressing normally versus stuck.
    """

    executor = FakeModelExecutor()
    events: list[tuple[str, int | None]] = []
    runner = build_runner(
        tmp_path,
        model_executor=executor,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, issue_number)
        ),
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    assert [event for event, _ in events] == [
        "setup_started",
        "setup_succeeded",
        "baseline_check_succeeded",
        "model_dispatch_starting",
        "completion_decided",
    ]
    assert all(issue_number == 24 for _, issue_number in events)


def test_baseline_failure_tolerated_on_continuation_and_logs_warning(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    events: list[tuple[str, str, int | None]] = []
    workspace = seeded_workspace(tmp_path / "repo", "Previous work")
    verifier = FakeVerifier(baseline_succeeded=False)
    runner = build_runner(
        tmp_path,
        model_executor=executor,
        workspace=workspace,
        verifier=verifier,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, level, issue_number)
        ),
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    assert [(event, level) for event, level, _ in events] == [
        ("setup_started", "INFO"),
        ("setup_succeeded", "INFO"),
        ("baseline_check_failed", "WARNING"),
        ("model_dispatch_starting", "INFO"),
        ("completion_decided", "WARNING"),
    ]
    assert executor.captured_prompt is not None


def test_baseline_failure_aborts_on_fresh_attempt_and_logs_error(
    tmp_path: Path, workspace: InMemoryWorkspace
) -> None:
    executor = FakeModelExecutor()
    events: list[tuple[str, str, int | None]] = []
    verifier = FakeVerifier(baseline_succeeded=False)
    runner = build_runner(
        tmp_path,
        model_executor=executor,
        workspace=workspace,
        verifier=verifier,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, level, issue_number)
        ),
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    assert [(event, level) for event, level, _ in events] == [
        ("setup_started", "INFO"),
        ("setup_succeeded", "INFO"),
        ("baseline_check_failed", "ERROR"),
        ("completion_decided", "WARNING"),
    ]
    assert executor.captured_prompt is None


def failed_baseline(
    *,
    command: str = "vendor/bin/php-cs-fixer fix --dry-run --diff",
    exit_code: int | None = 8,
    stdout: str = "checked files",
    stderr: str = "style violations",
    timed_out: bool = False,
    succeeded: bool = False,
):
    """Build a baseline-shaped stub carrying one failing command execution."""

    return SimpleNamespace(
        succeeded=succeeded,
        commands=(
            SimpleNamespace(
                command=command,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                timed_out=timed_out,
            ),
        ),
    )


def test_starting_prompt_with_failed_baseline_names_command_and_both_streams() -> None:
    """A red continuation baseline is quoted into the prompt with both streams.

    Linters such as ``php-cs-fixer --dry-run --diff`` report on stdout, so
    stderr alone (what the event detail carries) is not enough for the model
    to diagnose the failure.
    """

    prompt = _build_starting_prompt(
        "Fix the parser.",
        (),
        continuation=True,
        issue_number=24,
        profile=profile(),
        baseline=failed_baseline(stdout="1) src/Foo.php", stderr="found violations"),
    )

    assert "vendor/bin/php-cs-fixer fix --dry-run --diff" in prompt
    assert "exit 8" in prompt
    assert "1) src/Foo.php" in prompt
    assert "found violations" in prompt
    assert "must pass this check before the work is considered done" in prompt


def test_starting_prompt_with_failed_baseline_keeps_issue_body_and_handoff_pointer() -> None:
    prompt = _build_starting_prompt(
        "Fix the parser.",
        ("keep the old retry behaviour",),
        continuation=True,
        issue_number=24,
        profile=profile(),
        baseline=failed_baseline(),
    )

    assert prompt.startswith("Fix the parser.\n\nkeep the old retry behaviour\n\n")
    assert ".agent/handoff/24.md" in prompt


def test_starting_prompt_truncates_long_baseline_output_from_the_front() -> None:
    stdout = "old line\n" * 2000 + "final stdout line"
    stderr = "old err\n" * 2000 + "final stderr line"

    prompt = _build_starting_prompt(
        "Fix the parser.",
        (),
        continuation=True,
        issue_number=24,
        profile=profile(),
        baseline=failed_baseline(stdout=stdout, stderr=stderr),
    )

    assert "final stdout line" in prompt
    assert "final stderr line" in prompt
    # Cut from the front: most of the repeated head is gone from each stream.
    assert prompt.count("old line") < 2000
    assert prompt.count("old err") < 2000
    assert "truncat" in prompt.lower()


def test_starting_prompt_with_passing_baseline_is_unchanged() -> None:
    baseline = failed_baseline(succeeded=True)

    prompt = _build_starting_prompt(
        "Fix the parser.",
        ("keep the old retry behaviour",),
        continuation=True,
        issue_number=24,
        profile=profile(),
        baseline=baseline,
    )

    assert prompt == _build_starting_prompt(
        "Fix the parser.",
        ("keep the old retry behaviour",),
        continuation=True,
        issue_number=24,
        profile=profile(),
    )
    assert "Baseline check failure" not in prompt


def test_starting_prompt_without_baseline_is_unchanged() -> None:
    prompt = _build_starting_prompt(
        "Fix the parser.", (), continuation=False, issue_number=24, profile=profile()
    )

    assert prompt.startswith("Fix the parser.\n\n---\n## Completion gate")
    assert "Baseline check failure" not in prompt


def test_continuation_with_red_baseline_dispatches_prompt_with_failure(
    tmp_path: Path,
) -> None:
    """Lifecycle: a tolerated red baseline on a continuation reaches the model."""

    executor = FakeModelExecutor()
    workspace = seeded_workspace(tmp_path / "repo", "Previous work")
    verifier = FakeVerifier(
        baseline_succeeded=False,
        baseline_commands=(
            SimpleNamespace(
                command="vendor/bin/php-cs-fixer fix --dry-run --diff",
                exit_code=8,
                stdout="1) src/Foo.php",
                stderr="found violations",
                timed_out=False,
            ),
        ),
    )
    runner = build_runner(
        tmp_path, model_executor=executor, workspace=workspace, verifier=verifier
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    assert executor.captured_prompt is not None
    assert "vendor/bin/php-cs-fixer fix --dry-run --diff" in executor.captured_prompt
    assert "exit 8" in executor.captured_prompt
    assert "1) src/Foo.php" in executor.captured_prompt
    assert "found violations" in executor.captured_prompt
    assert "must pass this check before the work is considered done" in (
        executor.captured_prompt
    )
    # The gate stays last: the baseline failure comes before it.
    assert executor.captured_prompt.index("Baseline check failure") < (
        executor.captured_prompt.index("## Completion gate")
    )


def test_continuation_with_green_baseline_dispatches_unchanged_prompt(
    tmp_path: Path,
) -> None:
    executor = FakeModelExecutor()
    workspace = seeded_workspace(tmp_path / "repo", "Previous work")
    runner = build_runner(
        tmp_path, model_executor=executor, workspace=workspace
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    assert executor.captured_prompt == _build_starting_prompt(
        "Fix the parser.", (), continuation=True, issue_number=24, profile=profile()
    )
    assert "Baseline check failure" not in executor.captured_prompt


class FakeModelExecutor:
    def __init__(
        self,
        *,
        status: ModelExecutionStatus = ModelExecutionStatus.MODEL_LIMIT_REACHED,
        calls: "list[str] | None" = None,
        skill_events: tuple = (),
    ) -> None:
        self.captured_prompt: str | None = None
        self._status = status
        self._calls = calls
        self._skill_events = skill_events
        self.after_execute = None
        self.new_commits_provider = None

    def set_new_commits_provider(self, provider) -> None:
        self.new_commits_provider = provider

    async def execute(
        self, *, issue_body: str, working_directory: Path, issue_number: int | None = None, archive=None
    ) -> ModelExecution:
        if self._calls is not None:
            self._calls.append("execute")
        self.captured_prompt = issue_body
        if self.after_execute is not None:
            self.after_execute()
        return ModelExecution(
            status=self._status,
            explanation="stub",
            stop_reason="end_turn",
            model_usage=None,
            observed_models=(),
            skill_events=self._skill_events,
        )


class FakeVerifier:
    def __init__(
        self,
        *,
        baseline_succeeded: bool = True,
        setup_succeeded: bool = True,
        calls: "list[str] | None" = None,
        baseline_commands: tuple[object, ...] | None = None,
    ) -> None:
        self._baseline_succeeded = baseline_succeeded
        self._setup_succeeded = setup_succeeded
        self._calls = calls
        self._baseline_commands = baseline_commands

    def prepare(self, profile: object, working_directory: Path, **kwargs: object):
        if self._calls is not None:
            self._calls.append("prepare")
        setup = SimpleNamespace(
            succeeded=self._setup_succeeded,
            commands=(
                SimpleNamespace(command="setup", exit_code=0 if self._setup_succeeded else 1, stdout="", stderr="broken" if not self._setup_succeeded else ""),
            ),
        )
        baseline = SimpleNamespace(
            succeeded=self._baseline_succeeded,
            commands=self._baseline_commands if self._baseline_commands is not None else (),
            exit_code=0 if self._baseline_succeeded else 1,
        )
        return SimpleNamespace(setup=setup, baseline=baseline)

    def mark_review_complete(self, **kwargs: object) -> None:
        return None

    def final_check(self, profile: object, working_directory: Path):
        return SimpleNamespace(succeeded=True, commands=(), exit_code=0)


class FakeArchive:
    def __init__(self) -> None:
        self.merged: dict[str, object] = {}
        self.texts: dict[str, str] = {}

    def write_attempt(self, metadata: dict[str, object]) -> None:
        self.merged.update(metadata)

    def write_text(self, name: str, content: str) -> Path:
        self.texts[name] = content
        return Path(name)


class FakeEvaluator:
    def evaluate(self, **kwargs: object) -> CompletionDecision:
        return CompletionDecision(AttemptOutcome.INCOMPLETE, False, PublicationPath.NONE, ("stub",))


def FakePrepared(
    *,
    branch: str = "agent/issue-24",
    base_revision: str = "base-revision-0",
    restored_from_remote: bool = True,
    rebase_conflicts: tuple[str, ...] = (),
    pre_rebase_revision: str | None = None,
) -> PreparedAttempt:
    """Build a real ``PreparedAttempt`` with test-friendly defaults."""

    return PreparedAttempt(
        branch=branch,
        base_revision=base_revision,
        restored_from_remote=restored_from_remote,
        rebase_conflicts=rebase_conflicts,
        pre_rebase_revision=pre_rebase_revision,
    )


def claim(
    *,
    issue_number: int = 24,
    issue_body: str = "Fix the parser.",
    labels: frozenset[str] = frozenset({"ready-for-agent"}),
) -> Claim:
    return Claim(
        TrackerIssue(
            id=f"issue-{issue_number}",
            number=issue_number,
            title="Implement lifecycle",
            body=issue_body,
            created_at=datetime(2026, 9, 20, tzinfo=UTC),
            state="OPEN",
            labels=labels,
            assignee_logins=(),
            blocked_by=0,
            author_login="reporter",
        ),
        Assignment(f"issue-{issue_number}", "agent-id"),
    )


def profile(**overrides: object) -> RepositoryProfile:
    fields: dict[str, object] = {
        "setup": (),
        "check": ("pytest",),
        "base_branch": "main",
        "timeout": 60,
        "setup_timeout": 60,
        "env": {},
        "model_instructions": "",
    }
    fields.update(overrides)
    return RepositoryProfile(**fields)  # type: ignore[arg-type]


def test_infrastructure_error_seals_dirty_work(
    tmp_path: Path, workspace: InMemoryWorkspace
) -> None:
    """An SDK stream that dies without a terminal message still preserves edits.

    Regression test: ``INFRASTRUCTURE_ERROR`` used to skip the preservation
    path, so uncommitted work rode along when ``cleanup`` checked out the
    base branch and wedged the next poll cycle with ``working_tree_dirty``.
    The runner now seals unconditionally, so the work lands in a
    provenance-carrying seal commit on the attempt branch.
    """

    executor = FakeModelExecutor(status=ModelExecutionStatus.INFRASTRUCTURE_ERROR)
    workspace.make_dirty("work.txt", "model edits")
    runner = build_runner(tmp_path, model_executor=executor, workspace=workspace)
    prepared = FakePrepared()

    runner(
        claim(issue_number=24, issue_body="Fix the parser."), profile(), prepared
    )

    assert workspace.is_clean()
    commits = workspace.commits_added(prepared)
    assert [commit.subject for commit in commits] == [
        "Seal attempt work for #24 (infrastructure_error)"
    ]


def test_dirty_work_is_sealed_for_every_model_status(tmp_path: Path) -> None:
    """Sealing covers the whole ``ModelExecutionStatus`` enum.

    Guards against a future status being added to the enum without being
    sealed by ``ModelAttemptRunner``: the new member is picked up
    automatically because the test iterates the enum.
    """

    for status in ModelExecutionStatus:
        state_dir = tmp_path / status.value
        state_dir.mkdir(exist_ok=True)
        executor = FakeModelExecutor(status=status)
        workspace = InMemoryWorkspace(state_dir / "repo")
        workspace.make_dirty("work.txt", "model edits")
        runner = build_runner(state_dir, model_executor=executor, workspace=workspace)
        prepared = FakePrepared()

        runner(
            claim(issue_number=24, issue_body="Fix the parser."),
            profile(),
            prepared,
        )

        assert workspace.is_clean(), f"dirty work left behind for {status}"
        assert len(workspace.seal_calls) == 1, f"seal not called for {status}"
        assert workspace.seal_calls[0][2] is status
        subjects = [commit.subject for commit in workspace.commits_added(prepared)]
        assert f"Seal attempt work for #24 ({status.value})" in subjects, (
            f"no seal commit for {status}"
        )


def test_model_execution_failure_still_seals_dirty_work(
    tmp_path: Path, workspace: InMemoryWorkspace
) -> None:
    """When execution itself raises, the runner seals before propagating.

    The attempt finalizes as an infrastructure error with the model's work
    preserved on the branch instead of left dirty in the tree.
    """

    class ExplodingExecutor(FakeModelExecutor):
        async def execute(self, **kwargs: object) -> ModelExecution:
            raise RuntimeError("SDK stream died")

    workspace.make_dirty("work.txt", "model edits")
    runner = build_runner(
        tmp_path, model_executor=ExplodingExecutor(), workspace=workspace
    )
    prepared = FakePrepared()

    with pytest.raises(RuntimeError, match="SDK stream died"):
        runner(
            claim(issue_number=24, issue_body="Fix the parser."), profile(), prepared
        )

    assert workspace.is_clean()
    assert [commit.subject for commit in workspace.commits_added(prepared)] == [
        "Seal attempt work for #24 (infrastructure_error)"
    ]


def test_model_limit_path_seals_work_then_commits_note_only(
    tmp_path: Path, workspace: InMemoryWorkspace
) -> None:
    """On model limit the seal commit lands first and the note commit second.

    The model's dirty work is sealed with provenance, then the emergency
    handoff note is committed on top touching only the note file — the note
    commit must not sweep up unrelated work.
    """

    executor = FakeModelExecutor(status=ModelExecutionStatus.MODEL_LIMIT_REACHED)
    workspace.make_dirty("work.txt", "model edits")
    runner = build_runner(tmp_path, model_executor=executor, workspace=workspace)
    prepared = FakePrepared()

    runner(
        claim(issue_number=24, issue_body="Fix the parser."), profile(), prepared
    )

    assert workspace.is_clean()
    assert len(workspace.seal_calls) == 1
    assert len(workspace.note_calls) == 1
    commits = workspace.commits_added(prepared)
    assert [commit.subject for commit in commits] == [
        "Handoff note: issue #24",
        "Seal attempt work for #24 (model_limit_reached)",
    ]
    note_commit = workspace.handoff_note_commit(prepared, 24, attempt_id="unused")
    assert note_commit is not None
    assert note_commit.revision == commits[0].revision
    assert workspace.commit_files()[0] == frozenset({".agent/handoff/24.md"})


def test_model_limit_on_a_continuation_ignores_an_earlier_rounds_note(
    tmp_path: Path,
    workspace: InMemoryWorkspace,
) -> None:
    """A note left by an earlier round must not stand in for this attempt's.

    The continued branch already carries a committed note; on model limit
    the runner still writes a fresh emergency note on top of the sealed work.
    """

    executor = FakeModelExecutor(status=ModelExecutionStatus.MODEL_LIMIT_REACHED)
    workspace.prior_round_commit("Handoff note: issue #24", (".agent/handoff/24.md",))
    workspace.make_dirty("work.txt", "model edits")
    runner = build_runner(tmp_path, model_executor=executor, workspace=workspace)
    prepared = FakePrepared()

    runner(
        claim(issue_number=24, issue_body="Fix the parser."), profile(), prepared
    )

    assert len(workspace.note_calls) == 1
    subjects = [commit.subject for commit in workspace.commits_added(prepared)]
    assert subjects == [
        "Handoff note: issue #24",
        "Seal attempt work for #24 (model_limit_reached)",
        "Handoff note: issue #24",
    ]


def test_missing_transcript_source_logs_failure_and_leaves_outcome_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    """A missing CLI transcripts directory must not change the attempt outcome."""

    from simple_coding_agent.observability import AttemptArchive

    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))

    class SessionedExecutor(FakeModelExecutor):
        cli_session_id = "no-such-session"

    executor = SessionedExecutor(status=ModelExecutionStatus.SUCCEEDED)
    workspace = seeded_workspace(tmp_path / "repo", "Implemented")
    events: list[tuple[str, str, str]] = []
    runner = build_runner(
        tmp_path,
        model_executor=executor,
        workspace=workspace,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, level, detail)
        ),
    )
    runner._attempt_archive_factory = lambda issue_number, started_at: AttemptArchive(
        tmp_path / "data", issue_number=issue_number, started_at=started_at
    )

    evidence = runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    assert evidence.decision.outcome is AttemptOutcome.INCOMPLETE
    failures = [detail for event, level, detail in events if event == "transcripts_copy_failed"]
    assert len(failures) == 1
    assert "no-such-session" in failures[0]
    assert any(level == "WARNING" for event, level, _ in events if event == "transcripts_copy_failed")


def test_token_hard_ceiling_emergency_note_carries_the_token_hard_ceiling_reason(
    tmp_path: Path,
) -> None:
    """The latched hard-ceiling flag must survive into the emergency note."""

    from tests.fakes import InMemoryWorkspace

    class CeilingExecutor(FakeModelExecutor):
        async def execute(self, **kwargs: object) -> ModelExecution:
            if self.after_execute is not None:
                self.after_execute()
            return ModelExecution(
                status=ModelExecutionStatus.MODEL_LIMIT_REACHED,
                explanation=(
                    "Token budget reached the hard ceiling (budget_tokens=1100000; "
                    "max_budget_tokens=1000000)."
                ),
                stop_reason="end_turn",
                model_usage=None,
                observed_models=(),
                skill_events=(),
                token_hard_ceiling_reached=True,
            )

    workspace = InMemoryWorkspace(tmp_path / "repo")
    workspace.make_dirty("work.txt", "model edits")
    runner = build_runner(tmp_path, model_executor=CeilingExecutor(), workspace=workspace)

    runner(
        claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared()
    )

    assert len(workspace.note_calls) == 1
    assert "- reason: token_hard_ceiling" in workspace.note_calls[0][1]


def test_gate_section_lists_check_commands_in_order(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    comments = ("trusted note",)
    runner = build_runner(
        tmp_path, model_executor=executor, issue_comments=lambda issue: comments
    )

    runner(
        claim(issue_body="Fix it."),
        profile(check=("pytest", "ruff check .")),
        FakePrepared(),
    )

    assert executor.captured_prompt is not None
    prompt = executor.captured_prompt
    assert prompt.startswith("Fix it.\n\ntrusted note")
    assert "## Completion gate" in prompt
    first = prompt.index("1. pytest")
    second = prompt.index("2. ruff check .")
    assert first < second
    assert "in this order" in prompt
    assert "first failure fails the gate" in prompt
    assert "final check after the model finishes" in prompt


def test_gate_section_present_on_continuation(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    workspace = seeded_workspace(tmp_path / "repo", "Previous attempt work")
    runner = build_runner(tmp_path, model_executor=executor, workspace=workspace, issue_number=24)

    runner(
        claim(issue_number=24, issue_body="Fix it."),
        profile(check=("pytest", "ruff check .")),
        FakePrepared(),
    )

    assert executor.captured_prompt is not None
    assert ".agent/handoff/24.md" in executor.captured_prompt
    assert "## Completion gate" in executor.captured_prompt
    assert "1. pytest" in executor.captured_prompt
    assert "2. ruff check ." in executor.captured_prompt


def test_gate_section_present_on_setup_deferred_attempt(tmp_path: Path) -> None:
    executor = FakeModelExecutor(status=ModelExecutionStatus.SUCCEEDED)
    workspace = seeded_workspace(tmp_path / "repo", "Retained work")
    workspace.leave_conflict(files=("SubscriptionController.php",))
    executor.after_execute = workspace.note_model_finished
    runner = build_runner(tmp_path, model_executor=executor, workspace=workspace)

    runner(
        claim(issue_number=24, issue_body="Fix it."),
        profile(check=("pytest", "ruff check .")),
        FakePrepared(),
    )

    assert executor.captured_prompt is not None
    assert "git rebase --continue" in executor.captured_prompt
    assert "## Completion gate" in executor.captured_prompt
    assert "1. pytest" in executor.captured_prompt
    assert "2. ruff check ." in executor.captured_prompt


def test_gate_redacts_operator_credential_and_env_values(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    runner = build_runner(
        tmp_path,
        model_executor=executor,
        prompt_redactions=("operator-secret",),
    )

    runner(
        claim(issue_body="Fix it."),
        profile(
            check=("pytest --token operator-secret", "pytest --db-pass s3cr3t"),
            env={"DB_PASS": "s3cr3t"},
        ),
        FakePrepared(),
    )

    assert executor.captured_prompt is not None
    assert "operator-secret" not in executor.captured_prompt
    assert "s3cr3t" not in executor.captured_prompt
    assert "[REDACTED]" in executor.captured_prompt


def test_gate_lists_env_names_without_values(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    runner = build_runner(tmp_path, model_executor=executor)

    runner(
        claim(issue_body="Fix it."),
        profile(check=("pytest",), env={"APP_ENV": "test", "DB_PASS": "s3cr3t"}),
        FakePrepared(),
    )

    assert executor.captured_prompt is not None
    assert "APP_ENV" in executor.captured_prompt
    assert "DB_PASS" in executor.captured_prompt
    assert "s3cr3t" not in executor.captured_prompt


def test_gate_includes_model_instructions_when_present(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    runner = build_runner(tmp_path, model_executor=executor)

    runner(
        claim(issue_body="Fix it."),
        profile(
            check=("pytest",),
            model_instructions="Run php-cs-fixer fix before committing.",
        ),
        FakePrepared(),
    )

    assert executor.captured_prompt is not None
    assert "Run php-cs-fixer fix before committing." in executor.captured_prompt
    assert "Repository guidance" in executor.captured_prompt


def test_gate_omits_instructions_heading_when_absent(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    runner = build_runner(tmp_path, model_executor=executor)

    runner(claim(issue_body="Fix it."), profile(check=("pytest",)), FakePrepared())

    assert executor.captured_prompt is not None
    assert "Repository guidance" not in executor.captured_prompt


def test_gate_redacts_model_instructions(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    runner = build_runner(
        tmp_path,
        model_executor=executor,
        prompt_redactions=("operator-secret",),
    )

    runner(
        claim(issue_body="Fix it."),
        profile(
            check=("pytest",),
            env={"DB_PASS": "s3cr3t"},
            model_instructions="Use operator-secret and s3cr3t to connect.",
        ),
        FakePrepared(),
    )

    assert executor.captured_prompt is not None
    assert "operator-secret" not in executor.captured_prompt
    assert "s3cr3t" not in executor.captured_prompt
    assert "[REDACTED]" in executor.captured_prompt


# --- Missing code-review is visible instead of silent (issue #178) --------------


def real_evaluator() -> CompletionEvaluator:
    return CompletionEvaluator(frozenset({"must-fix"}))


def build_real_runner(tmp_path: Path, **kwargs: object) -> ModelAttemptRunner:
    """Build a runner with the real completion evaluator instead of the stub."""

    attempt_state = AttemptStateStore(tmp_path)
    attempt_state.start(issue_number=24, branch="agent/issue-24")
    attempt_state.transition(AttemptPhase.SETUP)
    workspace = kwargs.pop("workspace", None) or InMemoryWorkspace(tmp_path / "repo")
    executor = kwargs.pop("model_executor", None) or FakeModelExecutor()
    verifier = kwargs.pop("verifier", None) or FakeVerifier()
    event_log = kwargs.pop("event_log", None)
    runner_kwargs = {} if event_log is None else {"event_log": event_log}
    return ModelAttemptRunner(
        attempt_state=attempt_state,
        workspace=workspace,
        verifier=verifier,
        evaluator=real_evaluator(),
        model_executor=executor,
        **runner_kwargs,
    )


def code_review_event() -> SkillEvent:
    return SkillEvent(
        phase="PreToolUse",
        name="code-review",
        agent_id=None,
        timestamp="2026-10-11T00:00:00Z",
    )


def test_starting_prompt_requires_code_review_before_finishing(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    runner = build_runner(tmp_path, model_executor=executor)

    runner(claim(issue_body="Fix it."), profile(), FakePrepared())

    assert executor.captured_prompt is not None
    assert "code-review" in executor.captured_prompt
    assert "incomplete" in executor.captured_prompt.lower()


def test_continuation_prompt_requires_code_review_before_finishing(
    tmp_path: Path,
) -> None:
    executor = FakeModelExecutor()
    workspace = seeded_workspace(tmp_path / "repo", "Previous attempt work")
    runner = build_runner(tmp_path, model_executor=executor, workspace=workspace, issue_number=24)

    runner(claim(issue_number=24, issue_body="Fix it."), profile(), FakePrepared())

    assert executor.captured_prompt is not None
    assert ".agent/handoff/24.md" in executor.captured_prompt
    assert "code-review" in executor.captured_prompt


def test_missing_review_ends_incomplete_with_reason_and_event(
    tmp_path: Path,
) -> None:
    executor = FakeModelExecutor(status=ModelExecutionStatus.SUCCEEDED)
    workspace = seeded_workspace(tmp_path / "repo", "Implemented")
    events: list[tuple[str, str, str]] = []
    runner = build_real_runner(
        tmp_path,
        model_executor=executor,
        workspace=workspace,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, level, detail)
        ),
    )

    evidence = runner(claim(issue_number=24, issue_body="Fix it."), profile(), FakePrepared())

    assert evidence.decision.outcome is AttemptOutcome.INCOMPLETE
    assert evidence.reviews_performed == 0
    assert evidence.review_cycles == 0
    assert evidence.review_findings == "not run"
    assert "Acceptance criteria are not satisfied." in evidence.details
    decided = [detail for event, _, detail in events if event == "completion_decided"]
    assert len(decided) == 1
    assert "outcome=incomplete" in decided[0]
    assert "Acceptance criteria are not satisfied." in decided[0]


def test_reviewed_run_does_not_end_incomplete_for_a_missing_review(
    tmp_path: Path,
) -> None:
    executor = FakeModelExecutor(
        status=ModelExecutionStatus.SUCCEEDED, skill_events=(code_review_event(),)
    )
    workspace = seeded_workspace(tmp_path / "repo", "Implemented")
    events: list[tuple[str, str, str]] = []
    runner = build_real_runner(
        tmp_path,
        model_executor=executor,
        workspace=workspace,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, level, detail)
        ),
    )

    evidence = runner(claim(issue_number=24, issue_body="Fix it."), profile(), FakePrepared())

    assert evidence.decision.outcome is None
    assert evidence.decision.publication_eligible
    assert evidence.reviews_performed == 1
    assert evidence.review_cycles == 0
    decided = [detail for event, _, detail in events if event == "completion_decided"]
    assert len(decided) == 1
    assert "outcome=none" in decided[0]


def test_runner_wires_a_new_commits_provider_into_the_executor(
    tmp_path: Path,
) -> None:
    executor = FakeModelExecutor(status=ModelExecutionStatus.SUCCEEDED)
    workspace = seeded_workspace(tmp_path / "repo", "Implemented")
    executor.after_execute = lambda: workspace.model_commit("Format fix")
    runner = build_real_runner(tmp_path, model_executor=executor, workspace=workspace)

    runner(claim(issue_number=24, issue_body="Fix it."), profile(), FakePrepared())

    assert executor.new_commits_provider is not None
    assert executor.new_commits_provider() == 1


def test_setup_failure_emits_completion_decided(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    events: list[tuple[str, str, str]] = []
    runner = build_real_runner(
        tmp_path,
        model_executor=executor,
        verifier=FakeVerifier(setup_succeeded=False),
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, level, detail)
        ),
    )

    evidence = runner(claim(issue_number=24, issue_body="Fix it."), profile(), FakePrepared())

    assert evidence.decision.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    decided = [detail for event, _, detail in events if event == "completion_decided"]
    assert len(decided) == 1
    assert "outcome=infrastructure_error" in decided[0]
