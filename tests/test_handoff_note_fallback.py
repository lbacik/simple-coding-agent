"""Runtime-supplied handoff notes for a requested handoff without a note commit.

When model execution ends ``HANDOFF_REQUESTED`` but no ``.agent/handoff/<N>.md``
commit exists for the attempt, the runtime supplies the note so the attempt
still ends as a handoff: the model's own note recovered from its final
response when usable, otherwise a runtime-written emergency note.
"""

from __future__ import annotations

from pathlib import Path

from simple_coding_agent.completion import AttemptOutcome
from simple_coding_agent.lifecycle import (
    _build_emergency_handoff_note,
    _extract_recovered_handoff_note,
    _is_complete_handoff_note,
    _recovered_note_names_expected_work,
    _validate_handoff_note,
)
from simple_coding_agent.model_execution import (
    ModelExecution,
    ModelExecutionStatus,
    SkillEvent,
)
from tests.fakes import (
    FakeGitHub,
    FakeModelExecutor,
    InMemoryWorkspace,
    build_lifecycle,
    git,
    lifecycle_issue as issue,
    remote_branch_subjects,
    remote_has_branch,
    repository_with_main,
    write_and_commit,
)
from tests.test_model_attempt_runner import (
    FakePrepared,
    build_runner,
    claim,
    profile,
)


def _note_body(
    *,
    issue_number: int = 24,
    started_at: str,
    last_work_commit: str,
    reason: str = "token_soft_threshold",
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
        f"\n"
        f"## Remaining work\n"
        f"\n"
        f"Finish the parser.\n"
        f"\n"
        f"## Evidence\n"
        f"\n"
        f"parser.py.\n"
    )


class _HandoffExecutor:
    """A model execution that ends with a scripted terminal status and final text."""

    def __init__(
        self,
        text: object,
        action=None,
        status: ModelExecutionStatus = ModelExecutionStatus.HANDOFF_REQUESTED,
    ) -> None:
        self._text = text
        self._action = action
        self._status = status
        self.after_execute = None

    async def execute(
        self,
        *,
        issue_body: str,
        working_directory: Path,
        issue_number: int | None = None,
        archive=None,
        attempt_id: str | None = None,
        branch: str | None = None,
        base_revision: str | None = None,
    ) -> ModelExecution:
        if self._action is not None:
            self._action(working_directory)
        if self.after_execute is not None:
            self.after_execute()
        text = self._text(attempt_id) if callable(self._text) else self._text
        assert isinstance(text, str) or text is None
        return ModelExecution(
            status=self._status,
            explanation="scripted handoff",
            stop_reason="end_turn",
            model_usage=None,
            observed_models=(),
            skill_events=(
                SkillEvent(
                    phase="PreToolUse",
                    name="handoff",
                    agent_id=None,
                    timestamp="2026-09-22T00:00:00Z",
                ),
            ),
            last_assistant_text=text,
        )


def _capture_events():
    captured: list[tuple] = []
    return captured, lambda event, detail="", level="INFO", issue_number=None: captured.append(
        (event, detail, level, issue_number)
    )


def _validate_workspace_note(
    workspace: InMemoryWorkspace, prepared, issue_number: int = 24
):
    commits = workspace.commits_added(prepared)
    note_commit = workspace.handoff_note_commit(
        prepared, issue_number, attempt_id="unused"
    )
    return _validate_handoff_note(
        note_commit,
        commits,
        issue_number,
        workspace.working_directory,
        prepared.base_revision,
    )


# ---------------------------------------------------------------------------
# Runner-level: emergency fallback
# ---------------------------------------------------------------------------


def test_handoff_without_note_writes_an_emergency_note(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo")
    workspace.make_dirty("work.txt", "model edits")
    events, event_log = _capture_events()
    runner = build_runner(
        tmp_path,
        model_executor=_HandoffExecutor(None),
        workspace=workspace,
        event_log=event_log,
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    assert len(workspace.note_calls) == 1
    _, content = workspace.note_calls[0]
    assert "- reason: token_soft_threshold" in content
    assert "written by the runtime" in content
    assert _validate_workspace_note(workspace, FakePrepared()).valid
    kinds = [event for event, _, _, _ in events]
    assert "handoff_note_emergency_written" in kinds
    assert "handoff_note_recovered" not in kinds


def test_limit_stop_emergency_note_now_announces_itself(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo")
    workspace.make_dirty("work.txt", "model edits")
    events, event_log = _capture_events()
    runner = build_runner(
        tmp_path,
        model_executor=_HandoffExecutor(None, status=ModelExecutionStatus.MODEL_LIMIT_REACHED),
        workspace=workspace,
        event_log=event_log,
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    kinds = [event for event, _, _, _ in events]
    assert "handoff_note_emergency_written" in kinds
    assert "handoff_note_recovered" not in kinds


# ---------------------------------------------------------------------------
# Runner-level: recovery from the final response
# ---------------------------------------------------------------------------


def test_handoff_recovers_note_from_the_final_response(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo")
    work = workspace.model_commit("Half-finish the parser", files=("parser.py",))
    workspace.make_dirty("stray.tmp", "noise")
    executor = _HandoffExecutor(
        lambda attempt_id: (
            "I cannot write the handoff note file, so here it is:\n\n"
            "```markdown\n"
            f"{_note_body(started_at=attempt_id, last_work_commit=work.revision)}"
            "```\n"
        )
    )
    events, event_log = _capture_events()
    runner = build_runner(
        tmp_path, model_executor=executor, workspace=workspace, event_log=event_log
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    assert len(workspace.note_calls) == 1
    _, content = workspace.note_calls[0]
    assert "written by the runtime" not in content
    assert "Recovered by the runtime from the model's final response" in content
    assert "Half done." in content
    # The recovered note sits below the seal commit, like an ordinary handoff.
    subjects = [commit.subject for commit in workspace.commits_added(FakePrepared())]
    assert subjects == [
        "Seal attempt work for #24 (handoff_requested)",
        "Handoff note: issue #24",
        "Half-finish the parser",
    ]
    assert _validate_workspace_note(workspace, FakePrepared()).valid
    kinds = [event for event, _, _, _ in events]
    assert "handoff_note_recovered" in kinds
    assert "handoff_note_emergency_written" not in kinds


def test_recovered_note_counts_as_complete_while_emergency_does_not() -> None:
    recovered = (
        _note_body(started_at="2026-11-21T09:55:06Z", last_work_commit="abc1234")
        + "\nRecovered by the runtime from the model's final response: the model "
        "wrote this note but did not commit it.\n"
    )
    assert _is_complete_handoff_note(recovered, 24) is True

    emergency = _build_emergency_handoff_note(
        issue_number=24,
        started_at="2026-11-21T09:55:06Z",
        reason="token_soft_threshold",
        last_work_commit="abc1234",
        explanation="stopped",
    )
    assert _is_complete_handoff_note(emergency, 24) is False


def test_recovery_skips_blocks_that_do_not_belong_to_this_attempt(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo")
    work = workspace.model_commit("Half-finish the parser", files=("parser.py",))
    workspace.make_dirty("work.txt", "model edits")

    def text(attempt_id: str) -> str:
        other_issue = _note_body(
            issue_number=25, started_at=attempt_id, last_work_commit=work.revision
        )
        wrong_attempt = _note_body(
            started_at="2020-01-01T00:00:00Z", last_work_commit=work.revision
        )
        bad_work = _note_body(
            started_at=attempt_id, last_work_commit="deadbeef1234abcd"
        )
        return (
            f"```markdown\n{other_issue}```\n\n"
            f"```markdown\n{wrong_attempt}```\n\n"
            f"```markdown\n{bad_work}```\n"
        )

    events, event_log = _capture_events()
    runner = build_runner(
        tmp_path,
        model_executor=_HandoffExecutor(text),
        workspace=workspace,
        event_log=event_log,
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    # No usable block: exactly one note commit (the emergency one), no leftover.
    assert len(workspace.note_calls) == 1
    _, content = workspace.note_calls[0]
    assert "- reason: token_soft_threshold" in content
    assert "written by the runtime" in content
    assert _validate_workspace_note(workspace, FakePrepared()).valid
    assert workspace.handoff_note_commits(FakePrepared(), 24, attempt_id="unused") is not None
    kinds = [event for event, _, _, _ in events]
    assert "handoff_note_recovered" not in kinds
    assert "handoff_note_emergency_written" in kinds


def test_model_committed_but_invalid_note_never_triggers_either_fallback(
    tmp_path: Path,
) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo")
    note_path = workspace.working_directory / ".agent" / "handoff" / "24.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        _note_body(started_at="2026-09-22T00:00:00Z", last_work_commit="0" * 40)
    )
    workspace.model_commit(
        "Handoff note: issue #24", files=(".agent/handoff/24.md",)
    )
    events, event_log = _capture_events()
    runner = build_runner(
        tmp_path,
        model_executor=_HandoffExecutor(None),
        workspace=workspace,
        event_log=event_log,
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    assert workspace.note_calls == []
    kinds = [event for event, _, _, _ in events]
    assert "handoff_note_recovered" not in kinds
    assert "handoff_note_emergency_written" not in kinds


# ---------------------------------------------------------------------------
# Extractor unit tests
# ---------------------------------------------------------------------------


def test_extractor_prefers_the_first_qualifying_block() -> None:
    attempt_id = "2026-11-21T09:55:06.061488Z"
    other = _note_body(issue_number=25, started_at=attempt_id, last_work_commit="abc1234")
    good = _note_body(started_at=attempt_id, last_work_commit="abc1234")
    text = f"notes:\n```\n{other}\n```\nand:\n```markdown\n{good}\n```\n"
    assert _extract_recovered_handoff_note(text, 24, attempt_id) == good.strip()


def test_extractor_rejects_unusable_blocks() -> None:
    attempt_id = "2026-11-21T09:55:06.061488Z"
    assert _extract_recovered_handoff_note(None, 24, attempt_id) is None
    assert _extract_recovered_handoff_note("", 24, attempt_id) is None
    assert (
        _extract_recovered_handoff_note("no fences here", 24, attempt_id) is None
    )
    bad_reason = _note_body(
        started_at=attempt_id, last_work_commit="abc1234", reason="bogus"
    )
    assert (
        _extract_recovered_handoff_note(f"```\n{bad_reason}\n```", 24, attempt_id)
        is None
    )
    unclosed = f"```\n{_note_body(started_at=attempt_id, last_work_commit='abc1234')}"
    assert _extract_recovered_handoff_note(unclosed, 24, attempt_id) is None


def test_recovered_block_must_name_the_expected_work_commit(tmp_path: Path) -> None:
    expected = "commit-0007"
    good = _note_body(started_at="x", last_work_commit=expected)
    assert _recovered_note_names_expected_work(good, expected, tmp_path) is True
    prefix = _note_body(started_at="x", last_work_commit=expected[:9])
    assert _recovered_note_names_expected_work(prefix, expected, tmp_path) is True
    stale = _note_body(started_at="x", last_work_commit="deadbeef1234abcd")
    assert _recovered_note_names_expected_work(stale, expected, tmp_path) is False


# ---------------------------------------------------------------------------
# End to end: the attempt ends as a handoff with a committed note
# ---------------------------------------------------------------------------


def test_handoff_without_a_note_ends_as_handoff_with_an_emergency_note(
    tmp_path: Path,
) -> None:
    remote, _ = repository_with_main(tmp_path)
    github = FakeGitHub(
        issue(24, body="Rewrite the parser.", labels=frozenset({"ready-for-agent"}))
    )
    clone_dir = tmp_path / "clone"
    events, event_log = _capture_events()

    def do_handoff(working_directory: Path) -> None:
        write_and_commit(working_directory, "parser.py", "half done", "Half-finish the parser")

    executor = FakeModelExecutor(actions=[do_handoff], handoff=True)
    lifecycle = build_lifecycle(
        remote, clone_dir, tmp_path / "data", github, executor, event_log=event_log
    )

    result = lifecycle.run_once()

    assert result.outcome is AttemptOutcome.HANDOFF
    assert remote_has_branch(remote, "agent/issue-24")
    assert github.added_labels == [("issue-24", "round-finished")]
    [comment] = [c for c in github.comments if "Agent Attempt Result: handoff" in c.body]
    assert "token_soft_threshold" in comment.body
    subjects = remote_branch_subjects(remote, "agent/issue-24")
    assert subjects[:2] == ["Handoff note: issue #24", "Half-finish the parser"]
    note = git(clone_dir, "show", "agent/issue-24:.agent/handoff/24.md")
    assert "- reason: token_soft_threshold" in note
    assert "written by the runtime" in note
    kinds = [event for event, _, _, _ in events]
    assert "handoff_note_emergency_written" in kinds
    assert "handoff_note_recovered" not in kinds


def test_handoff_with_a_note_in_the_final_response_recovers_it(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    github = FakeGitHub(
        issue(24, body="Rewrite the parser.", labels=frozenset({"ready-for-agent"}))
    )
    clone_dir = tmp_path / "clone"
    events, event_log = _capture_events()

    def do_handoff(working_directory: Path) -> None:
        write_and_commit(working_directory, "parser.py", "half done", "Half-finish the parser")
        # A stray file left uncommitted, so the seal commit lands on top of
        # the recovered note like an ordinary token handoff.
        (working_directory / "stray.tmp").write_text("noise")

    def text_for(attempt_id: str | None, head_sha: str) -> str:
        return (
            "I cannot write the handoff note file — the `Write` tool is not "
            "available. Below is the complete handoff note content:\n\n"
            "```markdown\n"
            f"{_note_body(started_at=attempt_id or '', last_work_commit=head_sha)}"
            "```\n"
        )

    class RecoveringExecutor:
        def __init__(self) -> None:
            self.calls = 0
            self.captured_prompts: list[str] = []

        async def execute(
            self,
            *,
            issue_body: str,
            working_directory: Path,
            issue_number: int | None = None,
            archive=None,
            attempt_id: str | None = None,
            branch: str | None = None,
            base_revision: str | None = None,
        ) -> ModelExecution:
            self.calls += 1
            self.captured_prompts.append(issue_body)
            do_handoff(working_directory)
            head_sha = git(working_directory, "rev-parse", "HEAD")
            return ModelExecution(
                status=ModelExecutionStatus.HANDOFF_REQUESTED,
                explanation="scripted handoff",
                stop_reason="end_turn",
                model_usage=None,
                observed_models=(),
                skill_events=(
                    SkillEvent(
                        phase="PreToolUse",
                        name="handoff",
                        agent_id=None,
                        timestamp="2026-09-22T00:00:00Z",
                    ),
                ),
                last_assistant_text=text_for(attempt_id, head_sha),
            )

    lifecycle = build_lifecycle(
        remote, clone_dir, tmp_path / "data", github, RecoveringExecutor(),
        event_log=event_log,
    )

    result = lifecycle.run_once()

    assert result.outcome is AttemptOutcome.HANDOFF
    assert remote_has_branch(remote, "agent/issue-24")
    assert github.added_labels == [("issue-24", "round-finished")]
    subjects = remote_branch_subjects(remote, "agent/issue-24")
    assert subjects[:3] == [
        "Seal attempt work for #24 (handoff_requested)",
        "Handoff note: issue #24",
        "Half-finish the parser",
    ]
    note = git(clone_dir, "show", "agent/issue-24:.agent/handoff/24.md")
    assert "Half done." in note
    assert "Recovered by the runtime from the model's final response" in note
    assert "written by the runtime" not in note
    [comment] = [c for c in github.comments if "Agent Attempt Result: handoff" in c.body]
    assert "Half done." in comment.body
    kinds = [event for event, _, _, _ in events]
    assert "handoff_note_recovered" in kinds
    assert "handoff_note_emergency_written" not in kinds
