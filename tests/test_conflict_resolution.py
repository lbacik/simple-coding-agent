"""One narrow model resolution session for conflict attempts (issue #271).

A conflict attempt that hits a real rebase conflict, or a red profile check
after a clean rebase, brings the pull request up to date with at most one
narrow model session under the ordinary model limits. The session preserves
the pull request's commit count (fixes land only as commits on top),
publishes only after the rebase-resolution check, setup, and one profile
check pass, and never invokes ``code-review``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.command_runner import CommandResult
from simple_coding_agent.completion import AttemptOutcome
from simple_coding_agent.conflict_attempt import (
    base_changes_text,
    build_resolution_prompt,
    files_outside_conflict,
    fix_commit_count,
    format_check_output,
)
from simple_coding_agent.config import RepositoryProfile
from simple_coding_agent.github_tracker import TrackerIssue
from simple_coding_agent.lifecycle import AgentLifecycle, _gate_section
from simple_coding_agent.model_execution import ModelExecutionStatus
from simple_coding_agent.publication import (
    ConflictPublicationRequest,
    PullRequest,
    _conflict_result_comment,
)
from simple_coding_agent.recovery import (
    CONFLICT_OUTCOME_CHECK_FAILED,
    CONFLICT_OUTCOME_ERROR,
    CONFLICT_OUTCOME_MODEL_LIMIT,
    CONFLICT_OUTCOME_RESOLUTION_FAILED,
    CONFLICT_OUTCOME_SETUP_FAILED,
    CONFLICT_OUTCOME_SUCCESS,
    parse_conflict_attempt,
)
from tests.fakes import (
    FakeCommandRunner,
    FakeConflictExecutor,
    FakePublisher,
    FakeTracker,
    InMemoryWorkspace,
    failed_result,
    ok_result,
)

CONFLICT = frozenset({"merge-conflict"})

_MANDATORY_REVIEW_SENTENCE = "an attempt with commits but no `code-review` invocation"


def issue(number: int = 7) -> TrackerIssue:
    return TrackerIssue(
        id=f"issue-{number}",
        number=number,
        title=f"Issue {number}",
        body=f"Implement {number}.",
        created_at="2026-10-11T10:00:00Z",  # type: ignore[arg-type]
        state="OPEN",
        labels=CONFLICT,
        assignee_logins=(),
        blocked_by=0,
        author_login="reporter",
    )


def pull_request() -> PullRequest:
    return PullRequest(
        number=7,
        url="https://example.test/pull/7",
        node_id="pr-7",
        draft=False,
        mergeable="CONFLICTING",
        base_ref="main",
    )


def profile() -> RepositoryProfile:
    return RepositoryProfile(
        setup=("setup.sh",),
        check=("pytest",),
        base_branch="main",
        timeout=300,
        setup_timeout=120,
        env={},
    )


def raising_runner(claim: object, profile: object, prepared: object) -> object:
    raise AssertionError("a conflict attempt must never call the ordinary model path")


def run_resolution(
    tmp_path: Path,
    *,
    workspace: InMemoryWorkspace | None = None,
    executor: FakeConflictExecutor | None = None,
    command_results: list[CommandResult] | None = None,
    comments: tuple[str, ...] = (),
) -> tuple[object, ...]:
    """Claim one conflict issue through ``run_once`` and return the harness."""
    tracker = FakeTracker({7: issue()}, fifo=[7])
    active_workspace = workspace or InMemoryWorkspace(tmp_path / "repo")
    publisher = FakePublisher(pull_requests={"agent/issue-7": pull_request()})
    active_runner = FakeCommandRunner(list(command_results or []))
    events: list[tuple[str, str, str, int | None]] = []

    def event_log(
        event: str,
        detail: str = "",
        level: str = "INFO",
        issue_number: int | None = None,
    ) -> None:
        events.append((event, detail, level, issue_number))

    lifecycle = AgentLifecycle(
        tracker=tracker,  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(tmp_path),
        workspace=active_workspace,
        profile_loader=lambda _: profile(),
        publisher=publisher,
        attempt_runner=raising_runner,  # type: ignore[arg-type]
        control_store=None,
        sleeper=lambda seconds: None,
        event_log=event_log,
        max_conflict_attempts=1,
        command_runner=active_runner,  # type: ignore[arg-type]
        conflict_model_executor=executor,
        conflict_issue_comments=(lambda _: comments),
    )
    result = lifecycle.run_once()
    return result, tracker, publisher, active_workspace, active_runner, events, executor


def resolving_workspace(
    tmp_path: Path, files: tuple[str, ...] = ("a.txt",)
) -> InMemoryWorkspace:
    """A workspace whose rebase conflict the fake model session resolves."""

    workspace = InMemoryWorkspace(tmp_path / "repo")
    workspace.prepare_conflicts = files
    return workspace


# ---------------------------------------------------------------------------
# prompt helpers
# ---------------------------------------------------------------------------


def test_resolution_prompt_marks_issue_context_as_do_not_implement() -> None:
    prompt = build_resolution_prompt(
        issue_body="Implement the widget.",
        comments=("A trusted comment.",),
        conflicted_files=("a.txt",),
        base_changes="abc123 fix base",
        check_output=None,
        gate_section="## Completion gate",
    )

    assert "Context only" in prompt
    assert "Do NOT implement the issue" in prompt
    assert "Implement the widget." in prompt
    assert "A trusted comment." in prompt
    assert "a.txt" in prompt
    assert "abc123 fix base" in prompt
    assert "git rebase --continue" in prompt
    assert "do not squash" in prompt
    # The full-implementation continuation and the mandatory review gate are
    # absent (the prompt names code-review only to forbid invoking it).
    assert "then continue the implementation" not in prompt
    assert _MANDATORY_REVIEW_SENTENCE not in prompt


def test_resolution_prompt_quotes_red_check_output() -> None:
    prompt = build_resolution_prompt(
        issue_body="Body.",
        comments=(),
        conflicted_files=(),
        base_changes="(unavailable)",
        check_output="Failing command: `pytest` (exit 1).",
        gate_section="## Completion gate",
    )

    assert "Failing check output" in prompt
    assert "Failing command: `pytest` (exit 1)." in prompt


def test_review_free_gate_section_has_no_review_instruction() -> None:
    section = _gate_section(profile(), (), include_review_instruction=False)

    assert "pytest" in section
    assert _MANDATORY_REVIEW_SENTENCE not in section
    assert "code-review" not in section


def test_review_gate_section_keeps_review_instruction_for_ordinary_work() -> None:
    section = _gate_section(profile(), ())

    assert _MANDATORY_REVIEW_SENTENCE in section


def test_base_changes_degrade_when_git_is_unavailable(tmp_path: Path) -> None:
    text = base_changes_text(tmp_path, "main", ("a.txt",))

    assert "unavailable" in text


def test_fix_commit_count_counts_fixes_on_top() -> None:
    assert fix_commit_count(("one", "two"), ("one", "two", "seal")) == 1
    assert fix_commit_count(("one",), ("one",)) == 0
    assert fix_commit_count((), ("seal",)) == 1


def test_fix_commit_count_rejects_rewritten_history() -> None:
    assert fix_commit_count(("one", "two"), ("one", "seal")) is None
    assert fix_commit_count(("one", "two"), ("two", "seal")) is None
    assert fix_commit_count(("one", "two"), ("one",)) is None


def test_files_outside_conflict_are_sorted_and_deduplicated() -> None:
    assert files_outside_conflict(
        ("b.txt", "a.txt", "b.txt", "extra.txt"), ("a.txt", "b.txt")
    ) == ("extra.txt",)
    assert files_outside_conflict((), ("a.txt",)) == ()


def test_format_check_output_names_the_failing_command() -> None:
    text = format_check_output(failed_result())

    assert "`pytest` (exit 1)" in text
    assert "boom" in text


# ---------------------------------------------------------------------------
# one session per attempt, then verify and publish
# ---------------------------------------------------------------------------


def test_rebase_conflict_resolves_pushes_and_reenables_nothing_extra(
    tmp_path: Path,
) -> None:
    workspace = resolving_workspace(tmp_path)
    calls: list[str] = []
    executor = FakeConflictExecutor(calls=calls)

    def after() -> None:
        workspace.make_dirty("a.txt", "resolved")
        workspace.make_dirty("extra.txt", "fixup")
        workspace.note_model_finished()

    executor.after_execute = after
    result, tracker, publisher, active, runner, events, _ = run_resolution(
        tmp_path,
        workspace=workspace,
        executor=executor,
        command_results=[ok_result("setup.sh"), ok_result()],
        comments=("A trusted comment.",),
    )

    assert result.outcome is AttemptOutcome.COMPLETE
    assert calls == ["execute"]
    request = publisher.conflict_requests[0]
    assert request.conflict_outcome == CONFLICT_OUTCOME_SUCCESS
    assert request.push_branch is True
    assert request.conflicted_files == ("a.txt",)
    assert request.extra_files == ("extra.txt",)
    assert len(request.fix_commits) == 1
    assert request.check_command == "pytest"
    assert request.check_exit_code == 0
    # The narrow prompt ran: context-only issue material, conflicted files,
    # no continuation, no mandatory review.
    assert executor.captured_prompt is not None
    assert "A trusted comment." in executor.captured_prompt
    assert "Do NOT implement the issue" in executor.captured_prompt
    assert "then continue the implementation" not in executor.captured_prompt
    assert _MANDATORY_REVIEW_SENTENCE not in executor.captured_prompt
    # Verification reran setup plus one check; the seal holds the fix work
    # and no handoff note was written.
    assert [call[0] for call in runner.calls] == [("setup.sh",), ("pytest",)]
    assert active.seal_calls != []
    assert active.note_calls == []
    assert active.abort_calls == []
    assert tracker.cleanup == [(7, "merge-conflict", "agent-id")]


def test_red_check_after_clean_rebase_runs_the_same_session(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo")
    calls: list[str] = []
    executor = FakeConflictExecutor(calls=calls)

    def after() -> None:
        workspace.make_dirty("fixed.txt", "fixup")

    executor.after_execute = after
    result, tracker, publisher, active, runner, events, _ = run_resolution(
        tmp_path,
        workspace=workspace,
        executor=executor,
        command_results=[
            ok_result("setup.sh"),
            failed_result(),
            ok_result("setup.sh"),
            ok_result(),
        ],
    )

    assert result.outcome is AttemptOutcome.COMPLETE
    assert calls == ["execute"]
    request = publisher.conflict_requests[0]
    assert request.conflict_outcome == CONFLICT_OUTCOME_SUCCESS
    assert request.push_branch is True
    assert request.conflicted_files == ()
    assert executor.captured_prompt is not None
    assert "Failing check output" in executor.captured_prompt
    assert _MANDATORY_REVIEW_SENTENCE not in executor.captured_prompt


def test_changed_commit_count_pushes_nothing(tmp_path: Path) -> None:
    workspace = resolving_workspace(tmp_path)
    workspace.prior_round_commit("PR work", ("a.txt",))
    calls: list[str] = []
    executor = FakeConflictExecutor(calls=calls)

    def after() -> None:
        # Simulate a squash: the pull request's own commit is dropped.
        workspace.drop_newest_commits(1)
        workspace.make_dirty("a.txt", "resolved")
        workspace.note_model_finished()

    executor.after_execute = after
    result, tracker, publisher, active, runner, events, _ = run_resolution(
        tmp_path,
        workspace=workspace,
        executor=executor,
        command_results=[ok_result("setup.sh"), ok_result()],
    )

    assert result.outcome is AttemptOutcome.INCOMPLETE
    assert calls == ["execute"]
    request = publisher.conflict_requests[0]
    assert request.conflict_outcome == CONFLICT_OUTCOME_RESOLUTION_FAILED
    assert request.push_branch is False
    assert request.conflicted_files == ("a.txt",)
    # Setup and the check never ran after the count violation.
    assert runner.calls == []
    assert tracker.cleanup == [(7, "merge-conflict", "agent-id")]


def test_unresolved_conflict_aborts_and_pushes_nothing(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo", resolve_after_model=False)
    workspace.prepare_conflicts = ("a.txt",)
    calls: list[str] = []
    executor = FakeConflictExecutor(calls=calls)
    executor.after_execute = workspace.note_model_finished
    result, tracker, publisher, active, runner, events, _ = run_resolution(
        tmp_path,
        workspace=workspace,
        executor=executor,
    )

    assert result.outcome is AttemptOutcome.INCOMPLETE
    assert calls == ["execute"]
    request = publisher.conflict_requests[0]
    assert request.conflict_outcome == "rebase_conflict"
    assert request.push_branch is False
    assert request.conflicted_files == ("a.txt",)
    assert active.abort_calls != []
    assert runner.calls == []


def test_red_post_model_check_pushes_nothing(tmp_path: Path) -> None:
    workspace = resolving_workspace(tmp_path)
    executor = FakeConflictExecutor()

    def after() -> None:
        workspace.make_dirty("a.txt", "resolved")
        workspace.note_model_finished()

    executor.after_execute = after
    result, tracker, publisher, active, runner, events, _ = run_resolution(
        tmp_path,
        workspace=workspace,
        executor=executor,
        command_results=[ok_result("setup.sh"), failed_result()],
    )

    assert result.outcome is AttemptOutcome.INCOMPLETE
    request = publisher.conflict_requests[0]
    assert request.conflict_outcome == CONFLICT_OUTCOME_CHECK_FAILED
    assert request.push_branch is False
    assert request.check_exit_code == 1


def test_red_post_model_setup_is_an_infrastructure_error(tmp_path: Path) -> None:
    workspace = resolving_workspace(tmp_path)
    executor = FakeConflictExecutor()

    def after() -> None:
        workspace.make_dirty("a.txt", "resolved")
        workspace.note_model_finished()

    executor.after_execute = after
    result, tracker, publisher, active, runner, events, _ = run_resolution(
        tmp_path,
        workspace=workspace,
        executor=executor,
        command_results=[failed_result("setup.sh")],
    )

    assert result.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    request = publisher.conflict_requests[0]
    assert request.conflict_outcome == CONFLICT_OUTCOME_SETUP_FAILED
    assert request.push_branch is False
    assert len(runner.calls) == 1


def test_model_limit_is_a_failure_without_push_or_note(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo", resolve_after_model=False)
    workspace.prepare_conflicts = ("a.txt",)
    calls: list[str] = []
    executor = FakeConflictExecutor(
        status=ModelExecutionStatus.MODEL_LIMIT_REACHED, calls=calls
    )
    executor.after_execute = workspace.note_model_finished
    result, tracker, publisher, active, runner, events, _ = run_resolution(
        tmp_path,
        workspace=workspace,
        executor=executor,
    )

    assert result.outcome is AttemptOutcome.INCOMPLETE
    assert calls == ["execute"]
    request = publisher.conflict_requests[0]
    assert request.conflict_outcome == CONFLICT_OUTCOME_MODEL_LIMIT
    assert request.push_branch is False
    assert request.conflicted_files == ("a.txt",)
    # No handoff note, no round-finished: the release only drops the
    # conflict signal and the assignee.
    assert active.note_calls == []
    assert active.abort_calls != []
    assert tracker.cleanup == [(7, "merge-conflict", "agent-id")]
    names = [event[0] for event in events]
    assert "conflict_model_started" in names
    assert "conflict_model_finished" in names


def test_errored_session_pushes_nothing(tmp_path: Path) -> None:
    workspace = resolving_workspace(tmp_path)
    executor = FakeConflictExecutor(status=ModelExecutionStatus.INFRASTRUCTURE_ERROR)
    executor.after_execute = workspace.note_model_finished
    result, tracker, publisher, active, runner, events, _ = run_resolution(
        tmp_path,
        workspace=workspace,
        executor=executor,
    )

    assert result.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    request = publisher.conflict_requests[0]
    assert request.conflict_outcome == CONFLICT_OUTCOME_ERROR
    assert request.push_branch is False
    assert runner.calls == []


def test_missing_executor_is_an_error_not_a_crash(tmp_path: Path) -> None:
    workspace = resolving_workspace(tmp_path)
    result, tracker, publisher, active, runner, events, _ = run_resolution(
        tmp_path,
        workspace=workspace,
        executor=None,
    )

    assert result.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    request = publisher.conflict_requests[0]
    assert request.conflict_outcome == CONFLICT_OUTCOME_ERROR
    assert request.push_branch is False
    assert runner.calls == []


# ---------------------------------------------------------------------------
# result comment: conflicted files, extra files, fix commits
# ---------------------------------------------------------------------------


def _comment_request(**overrides: object) -> ConflictPublicationRequest:
    base: dict[str, object] = {
        "issue_number": 7,
        "issue_title": "Issue 7",
        "branch": "agent/issue-7",
        "started_at": "2026-10-11T10:00:00Z",
        "base_branch": "main",
        "outcome": AttemptOutcome.COMPLETE,
        "conflict_outcome": CONFLICT_OUTCOME_SUCCESS,
        "reason": "",
        "details": "Resolved.",
        "push_branch": True,
        "pull_request": pull_request(),
        "check_command": "pytest",
        "check_exit_code": 0,
        "conflicted_files": ("a.txt",),
        "auto_merge_opt_in": False,
        "extra_files": ("extra.txt",),
        "fix_commits": ("deadbeefcafe",),
    }
    base.update(overrides)
    return ConflictPublicationRequest(**base)  # type: ignore[arg-type]


def test_success_comment_lists_files_and_fix_commits() -> None:
    request = _comment_request()
    body = _conflict_result_comment(
        request,
        AttemptOutcome.COMPLETE,
        CONFLICT_OUTCOME_SUCCESS,
        "https://example.test/tree/agent/issue-7",
        pull_request(),
        "Resolved.",
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->",
        None,
        "deadbeefcafe",
    )

    assert "## Conflict Attempt Result: success" in body
    assert "- Conflicted files: a.txt" in body
    assert "- Files changed outside the conflicted set: extra.txt" in body
    assert "- Fix commits: deadbeefcafe" in body
    parsed = parse_conflict_attempt(body)
    assert parsed is not None
    assert parsed["files"] == "a.txt"
    assert parsed["extra_files"] == "extra.txt"
    assert parsed["fix_commits"] == "deadbeefcafe"


def test_model_limit_comment_names_the_outcome() -> None:
    request = _comment_request(
        outcome=AttemptOutcome.INCOMPLETE,
        conflict_outcome=CONFLICT_OUTCOME_MODEL_LIMIT,
        details="Limited.",
        push_branch=False,
        extra_files=(),
        fix_commits=(),
    )
    body = _conflict_result_comment(
        request,
        AttemptOutcome.INCOMPLETE,
        CONFLICT_OUTCOME_MODEL_LIMIT,
        None,
        pull_request(),
        "Limited.",
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->",
        None,
        None,
    )

    assert "## Conflict Attempt Result: model_limit" in body
    assert parse_conflict_attempt(body)["outcome"] == "model_limit"
    assert "`agent/issue-7` (not pushed)" in body


def test_format_check_output_handles_a_missing_command() -> None:
    empty = SimpleNamespace(commands=(), exit_code=None)

    assert "before any command ran" in format_check_output(empty)
