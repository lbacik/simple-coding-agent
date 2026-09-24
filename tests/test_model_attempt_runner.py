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
    PublicationPath,
)
from simple_coding_agent.config import RepositoryProfile
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.git_workspace import RebaseConflictError
from simple_coding_agent.lifecycle import ModelAttemptRunner
from simple_coding_agent.model_execution import ModelExecution, ModelExecutionStatus


def test_prompt_appends_trusted_comments_in_order(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    comments = ("please also handle timeouts", "keep the old retry behaviour")
    runner = build_runner(tmp_path, model_executor=executor, issue_comments=lambda issue: comments)

    runner(claim(issue_body="Fix the parser."), profile(), FakePrepared())

    assert executor.captured_prompt == (
        "Fix the parser.\n\nplease also handle timeouts\n\nkeep the old retry behaviour"
    )


def test_prompt_degrades_to_the_bare_issue_body_with_no_comments_or_continuation(
    tmp_path: Path,
) -> None:
    executor = FakeModelExecutor()
    runner = build_runner(tmp_path, model_executor=executor)

    runner(claim(issue_body="Fix the parser."), profile(), FakePrepared())

    assert executor.captured_prompt == "Fix the parser."


def test_continuation_adds_only_a_pointer_to_the_handoff_note(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    workspace = FakeWorkspace(commits=("Previous attempt work",))
    runner = build_runner(tmp_path, model_executor=executor, workspace=workspace, issue_number=24)

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    assert executor.captured_prompt == (
        "Fix the parser.\n\n"
        "This is a continued attempt; if it exists, the handoff note from the "
        "previous attempt is at .agent/handoff/24.md."
    )


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
    workspace = FakeWorkspace(commits=("Retained work",), conflicts=True)
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
    workspace = FakeWorkspace(commits=("Retained work",), conflicts=True)
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
    workspace = FakeWorkspace(
        commits=("Retained work",), conflicts=True, resolve_after_model=False
    )
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
    """Workspaces without live conflict probes still report the file list."""

    executor = FakeModelExecutor(status=ModelExecutionStatus.SUCCEEDED)
    workspace = MinimalConflictingWorkspace()
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


def test_deferred_setup_failure_still_reports_without_a_final_check(tmp_path: Path) -> None:
    calls: list[str] = []
    executor = FakeModelExecutor(status=ModelExecutionStatus.SUCCEEDED, calls=calls)
    workspace = FakeWorkspace(commits=("Retained work",), conflicts=True)
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
    workspace = FakeWorkspace(commits=("Implemented",))
    events: list[str] = []
    runner = build_runner(
        tmp_path,
        model_executor=executor,
        workspace=workspace,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(event),
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    assert events[-2:] == ["final_check_started", "final_check_succeeded"]


def test_writes_final_check_markers_to_the_archive(tmp_path: Path) -> None:
    executor = FakeModelExecutor(status=ModelExecutionStatus.SUCCEEDED)
    workspace = FakeWorkspace(commits=("Implemented",))
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
    workspace: "FakeWorkspace | None" = None,
    verifier: "FakeVerifier | None" = None,
    issue_comments=lambda issue: (),
    issue_number: int = 24,
    event_log=None,
) -> ModelAttemptRunner:
    attempt_state = AttemptStateStore(tmp_path)
    attempt_state.start(issue_number=issue_number, branch=f"agent/issue-{issue_number}")
    attempt_state.transition(AttemptPhase.SETUP)
    kwargs = {} if event_log is None else {"event_log": event_log}
    return ModelAttemptRunner(
        attempt_state=attempt_state,
        workspace=workspace or FakeWorkspace(),
        verifier=verifier or FakeVerifier(),
        evaluator=FakeEvaluator(),
        model_executor=model_executor,
        issue_comments=issue_comments,
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
    ]
    assert all(issue_number == 24 for _, issue_number in events)


def test_baseline_failure_tolerated_on_continuation_and_logs_warning(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    events: list[tuple[str, str, int | None]] = []
    workspace = FakeWorkspace(commits=("Previous work",))
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
    ]
    assert executor.captured_prompt is not None


def test_baseline_failure_aborts_on_fresh_attempt_and_logs_error(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    events: list[tuple[str, str, int | None]] = []
    workspace = FakeWorkspace(commits=())
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
    ]
    assert executor.captured_prompt is None


class FakeModelExecutor:
    def __init__(
        self,
        *,
        status: ModelExecutionStatus = ModelExecutionStatus.MODEL_LIMIT_REACHED,
        calls: "list[str] | None" = None,
    ) -> None:
        self.captured_prompt: str | None = None
        self._status = status
        self._calls = calls
        self.after_execute = None

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
            skill_events=(),
        )


class FakeVerifier:
    def __init__(
        self,
        *,
        baseline_succeeded: bool = True,
        setup_succeeded: bool = True,
        calls: "list[str] | None" = None,
    ) -> None:
        self._baseline_succeeded = baseline_succeeded
        self._setup_succeeded = setup_succeeded
        self._calls = calls

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
            commands=(),
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


class FakeWorkspace:
    working_directory = Path("/repository")

    def __init__(
        self,
        *,
        commits: tuple[str, ...] = (),
        conflicts: bool = False,
        resolve_after_model: bool = True,
    ) -> None:
        self._commits = commits
        self._conflicts = conflicts
        self._resolve_after_model = resolve_after_model
        self._model_finished = False
        self.abort_calls: list[object] = []

    def note_model_finished(self) -> None:
        """Simulate the model's turn ending so resolution can take effect."""

        self._model_finished = True

    def _resolved(self) -> bool:
        return self._model_finished and self._resolve_after_model

    def commits_added(self, prepared: object) -> tuple[str, ...]:
        return self._commits

    def has_unresolved_conflicts(self) -> bool:
        return self._conflicts and not self._resolved()

    def conflicted_files(self) -> tuple[str, ...]:
        if self.has_unresolved_conflicts():
            return ("SubscriptionController.php",)
        return ()

    def rebase_resolution_problems(self) -> tuple[str, ...]:
        if self.has_unresolved_conflicts():
            return (
                "A rebase or merge is still in progress."
                " Conflicting files: SubscriptionController.php.",
            )
        return ()

    def abort_unresolved_rebase(self, prepared: object) -> None:
        self.abort_calls.append(prepared)


class MinimalConflictingWorkspace:
    """A workspace with conflicts but without the newer conflict probes.

    Pins the runner's fallback chain: the file list comes from the prepared
    attempt snapshot when live probes are unavailable.
    """

    working_directory = Path("/repository")

    def __init__(self) -> None:
        self.abort_calls: list[object] = []

    def commits_added(self, prepared: object) -> tuple[str, ...]:
        return ("Retained work",)

    def has_unresolved_conflicts(self) -> bool:
        return True

    def abort_unresolved_rebase(self, prepared: object) -> None:
        self.abort_calls.append(prepared)


class FakePrepared:
    def __init__(
        self,
        *,
        branch: str = "agent/issue-24",
        restored_from_remote: bool = True,
        rebase_conflicts: tuple[str, ...] = (),
        pre_rebase_revision: str | None = None,
    ) -> None:
        self.branch = branch
        self.restored_from_remote = restored_from_remote
        self.rebase_conflicts = rebase_conflicts
        self.pre_rebase_revision = pre_rebase_revision


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


def profile() -> RepositoryProfile:
    return RepositoryProfile(
        setup=(), check=("pytest",), base_branch="main", timeout=60, setup_timeout=60, env={}
    )
