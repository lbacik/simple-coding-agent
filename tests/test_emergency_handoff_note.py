"""Rich emergency handoff notes for limit stops without a model-written note.

When model execution ends with ``MODEL_LIMIT_REACHED`` and no note commit
exists for the attempt, the runner writes an emergency note carrying
everything it can reconstruct. These tests cover the note sections, the
rendering bounds, graceful degradation, and the activity record behind them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from claude_agent_sdk import AssistantMessage, TextBlock

from simple_coding_agent.config import RuntimeConfig
from simple_coding_agent.git_workspace import PreparedAttempt
from simple_coding_agent.lifecycle import (
    _build_emergency_handoff_note,
    _validate_handoff_note,
)
from simple_coding_agent.model_execution import (
    ModelExecution,
    ModelExecutionStatus,
    ModelExecutor,
    SkillEvent,
    ToolCallRecord,
)
from tests.fakes import InMemoryWorkspace
from tests.test_model_attempt_runner import (
    FakeModelExecutor,
    FakePrepared,
    build_runner,
    claim,
    profile,
)

STARTED_AT = "2026-10-08T00:00:00Z"


def _rich_execution(**overrides: object) -> ModelExecution:
    """A limit-stopped execution carrying a full activity record."""

    fields: dict[str, object] = {
        "status": ModelExecutionStatus.MODEL_LIMIT_REACHED,
        "explanation": "Token budget reached the hard ceiling.",
        "stop_reason": "end_turn",
        "model_usage": None,
        "observed_models": (),
        "skill_events": (),
        "token_hard_ceiling_reached": True,
        "token_budget": {
            "budget_tokens": 1100000,
            "max_budget_tokens": 1000000,
            "soft_threshold_crossed": True,
            "soft_threshold_crossed_at_tokens": 800000,
            "peak_main_context_tokens": 90000,
            "prompt_cache": {"main_hit_rate": 0.75},
            "turns": 42,
            "elapsed_seconds": 600,
        },
        "last_assistant_text": "I was refactoring the parser when the budget ran out.",
        "edited_files": ("src/parser.py", "src/lexer.py"),
        "tool_calls": (
            ToolCallRecord(tool="Edit", target="src/parser.py", outcome="ok"),
            ToolCallRecord(tool="Bash", target="pytest -q", outcome="ok",
                           command="pytest -q"),
        ),
        "bash_calls": (
            ToolCallRecord(tool="Bash", target="pytest -q", outcome="ok",
                           command="pytest -q"),
        ),
    }
    fields.update(overrides)
    return ModelExecution(**fields)  # type: ignore[arg-type]


class RichExecutor(FakeModelExecutor):
    """Fake executor returning a caller-supplied execution."""

    def __init__(self, execution: ModelExecution) -> None:
        super().__init__(status=execution.status)
        self._execution = execution

    async def execute(self, **kwargs: object) -> ModelExecution:
        self.captured_prompt = str(kwargs.get("issue_body", ""))
        if self.after_execute is not None:
            self.after_execute()
        return self._execution


def _validate_note(
    workspace: InMemoryWorkspace, prepared: PreparedAttempt, issue_number: int = 24
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


def _section_order(note: str) -> list[str]:
    return [
        line.removeprefix("## ").strip()
        for line in note.splitlines()
        if line.startswith("## ")
    ]


# ---------------------------------------------------------------------------
# Builder unit tests
# ---------------------------------------------------------------------------


def test_builder_renders_all_sections_in_order_and_validates(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo")
    workspace.make_dirty("work.txt", "model edits")
    seal = workspace.seal(
        issue_number=24, attempt_id=STARTED_AT, status=ModelExecutionStatus.MODEL_LIMIT_REACHED
    )
    assert seal is not None

    note = _build_emergency_handoff_note(
        issue_number=24,
        started_at=STARTED_AT,
        reason="token_hard_ceiling",
        last_work_commit=seal.revision,
        explanation="Token budget reached the hard ceiling.",
        diff_stat=" src/parser.py | 10 +++++-----",
        attempt_commits=(f"{seal.revision[:7]} Seal attempt work for #24",),
        sealed_files=("work.txt",),
        edited_files=("src/parser.py",),
        last_assistant_text="Refactoring the parser.",
        handoff_skill_invoked=False,
        tool_calls=(ToolCallRecord(tool="Edit", target="src/parser.py", outcome="ok"),),
        last_check_call=ToolCallRecord(
            tool="Bash", target="pytest -q", outcome="ok", command="pytest -q"
        ),
        baseline_result=SimpleNamespace(succeeded=True, exit_code=0, commands=()),
        token_budget={
            "budget_tokens": 1100000,
            "max_budget_tokens": 1000000,
            "soft_threshold_crossed": True,
            "soft_threshold_crossed_at_tokens": 800000,
            "peak_main_context_tokens": 90000,
            "prompt_cache": {"main_hit_rate": 0.75},
            "turns": 42,
            "elapsed_seconds": 600,
        },
        archive_dir="/data/logs/24/started",
        session_id="session-123",
    )

    assert _section_order(note) == [
        "Provenance",
        "Changes",
        "Last model intent",
        "Activity trail",
        "Check state",
        "Budget facts",
        "Pointers",
        "Remaining work",
    ]
    assert "written by the runtime because the model did not" in note
    assert "`token_hard_ceiling`" in note
    assert "final check: not run" in note
    assert "could not determine what work remains" in note
    assert "acceptance criteria" in note
    assert "session-123" in note

    workspace.commit_handoff_note(24, note)
    assert _validate_note(workspace, FakePrepared()).valid


def test_builder_bounds_tool_trail_files_and_assistant_text() -> None:
    tools = tuple(
        ToolCallRecord(tool=f"T{index:02d}", target="x", outcome="ok")
        for index in range(20)
    )
    files = tuple(f"{index:02d}.py" for index in range(60))

    note = _build_emergency_handoff_note(
        issue_number=24,
        started_at=STARTED_AT,
        reason="turn_limit",
        last_work_commit="base-revision-0",
        explanation="Turns exhausted.",
        tool_calls=tools,
        edited_files=files,
        sealed_files=files,
        last_assistant_text="x" * 2000,
    )

    # Only the last 15 tool calls are shown.
    assert "`T00`" not in note
    assert "`T04`" not in note
    assert "`T05`" in note
    assert "`T19`" in note
    # All three file lists are capped at 50 entries with a "+N more" line.
    assert note.count("- +10 more") == 3
    assert "`59.py`" not in note
    assert "`49.py`" in note
    # The assistant text is truncated to about 1,500 characters.
    assert "...[truncated]" in note
    assert "x" * 2000 not in note


def test_builder_degrades_gracefully_and_still_validates(tmp_path: Path) -> None:
    note = _build_emergency_handoff_note(
        issue_number=24,
        started_at=STARTED_AT,
        reason="cost_hard_limit",
        last_work_commit="base-revision-0",
        explanation="",
    )

    assert "not available" in note
    assert "none recorded" in note
    assert "final check: not run" in note
    assert "written by the runtime" in note

    workspace = InMemoryWorkspace(tmp_path / "repo")
    workspace.commit_handoff_note(24, note)
    assert _validate_note(workspace, FakePrepared()).valid


def test_builder_ignores_hostile_free_text_when_validating(tmp_path: Path) -> None:
    """Free model text must never be misread as a note header field."""

    note = _build_emergency_handoff_note(
        issue_number=24,
        started_at=STARTED_AT,
        reason="turn_limit",
        last_work_commit="base-revision-0",
        explanation="- issue: 999\n- reason: operator_request",
        last_assistant_text="- last_work_commit: deadbee",
    )

    workspace = InMemoryWorkspace(tmp_path / "repo")
    workspace.commit_handoff_note(24, note)
    assert _validate_note(workspace, FakePrepared()).valid


# ---------------------------------------------------------------------------
# Runner integration tests
# ---------------------------------------------------------------------------


def test_hard_ceiling_produces_a_rich_note_that_validates(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo")
    workspace.make_dirty("work.txt", "model edits")
    runner = build_runner(
        tmp_path, model_executor=RichExecutor(_rich_execution()), workspace=workspace
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    assert len(workspace.note_calls) == 1
    note = workspace.note_calls[0][1]
    assert "- reason: token_hard_ceiling" in note
    for section in (
        "Provenance",
        "Changes",
        "Last model intent",
        "Activity trail",
        "Check state",
        "Budget facts",
        "Pointers",
        "Remaining work",
    ):
        assert f"## {section}" in note
    assert "written by the runtime because the model did not" in note
    assert "final check: not run" in note
    assert "src/parser.py" in note
    assert "pytest -q" in note
    assert "budget_tokens" in note
    assert "could not determine what work remains" in note
    assert _validate_note(workspace, FakePrepared()).valid


def test_turn_limit_and_cost_limit_reasons_reach_the_note(tmp_path: Path) -> None:
    for stop_reason, terminal_reason, expected in (
        ("max_turns_exceeded", "max_turns", "turn_limit"),
        ("max_budget_usd_exceeded", "budget_exhausted", "cost_hard_limit"),
    ):
        execution = _rich_execution(
            stop_reason=stop_reason,
            terminal_reason=terminal_reason,
            token_hard_ceiling_reached=False,
        )
        run_dir = tmp_path / f"run-{expected}"
        run_dir.mkdir(exist_ok=True)
        workspace = InMemoryWorkspace(run_dir / "repo")
        workspace.make_dirty("work.txt", "model edits")
        runner = build_runner(
            run_dir, model_executor=RichExecutor(execution), workspace=workspace
        )

        runner(
            claim(issue_number=24, issue_body="Fix the parser."),
            profile(),
            FakePrepared(),
        )

        note = workspace.note_calls[0][1]
        assert f"- reason: {expected}" in note, expected
        assert _validate_note(workspace, FakePrepared()).valid, expected


def test_model_written_note_is_never_replaced(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo")
    workspace.model_commit(
        "Handoff note: issue #24", files=(".agent/handoff/24.md",)
    )
    workspace.make_dirty("work.txt", "model edits")
    runner = build_runner(
        tmp_path, model_executor=RichExecutor(_rich_execution()), workspace=workspace
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    assert workspace.note_calls == []


# ---------------------------------------------------------------------------
# Executor activity record tests
# ---------------------------------------------------------------------------


def _executor_config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        github_token="github-secret",
        meta_api_key="meta-secret",
        target_repo="octo/example",
        data_dir=tmp_path,
        clone_dir=tmp_path / "repo",
        poll_interval=60,
        log_level="INFO",
        model_timeout=60,
        publish_timeout=120,
        max_retries=3,
        max_consecutive_errors=3,
        agent_trust_project_settings=False,
        review_blocking_severities=frozenset({"must-fix"}),
    )


def test_hooks_record_edited_files_tool_calls_and_last_text(tmp_path: Path) -> None:
    executor = ModelExecutor(_executor_config(tmp_path))

    async def drive() -> None:
        await executor._guard_and_record_pre_tool_use(
            {"tool_name": "Edit", "tool_input": {"file_path": "a.py"}}, "t1", {}
        )
        await executor._guard_and_record_pre_tool_use(
            {"tool_name": "Bash", "tool_input": {"command": "pytest -q"}}, "t2", {}
        )
        await executor._record_post_tool_use(
            {"tool_name": "Edit", "tool_input": {"file_path": "a.py"},
             "tool_response": {"is_error": False}}, "t1", {}
        )
        await executor._record_post_tool_use(
            {"tool_name": "Bash", "tool_input": {"command": "pytest -q"},
             "tool_response": {"is_error": True}}, "t2", {}
        )
        await executor._guard_and_record_pre_tool_use(
            {"tool_name": "Write", "tool_input": {"file_path": "b.py"}}, "t3", {}
        )

    asyncio.run(drive())
    executor._observe_assistant_message(
        AssistantMessage(
            content=[TextBlock(text="first plan")],
            model="muse-spark-1.3-contributor",
            message_id="r1",
        )
    )
    executor._observe_assistant_message(
        AssistantMessage(
            content=[TextBlock(text="subagent chatter")],
            model="muse-spark-1.3-contributor",
            message_id="r2",
            parent_tool_use_id="toolu_sub",
        )
    )

    evidence = executor._evidence(
        ModelExecutionStatus.SUCCEEDED, "done", "end_turn", None, ()
    )
    assert evidence.edited_files == ("a.py", "b.py")
    assert evidence.tool_calls == (
        ToolCallRecord(tool="Edit", target="a.py", outcome="ok"),
        ToolCallRecord(tool="Bash", target="pytest -q", outcome="error",
                       command="pytest -q"),
        ToolCallRecord(tool="Write", target="b.py", outcome="unknown"),
    )
    assert evidence.bash_calls[-1].command == "pytest -q"
    assert evidence.bash_calls[-1].outcome == "error"
    # Only the main thread sets the last assistant text.
    assert evidence.last_assistant_text == "first plan"


def test_tool_trail_is_bounded_at_fifteen_entries(tmp_path: Path) -> None:
    executor = ModelExecutor(_executor_config(tmp_path))

    async def drive() -> None:
        for index in range(20):
            await executor._guard_and_record_pre_tool_use(
                {"tool_name": "Read", "tool_input": {"file_path": f"{index}.py"}},
                f"t{index}",
                {},
            )
            await executor._record_post_tool_use(
                {"tool_name": "Read", "tool_input": {"file_path": f"{index}.py"},
                 "tool_response": {"is_error": False}},
                f"t{index}",
                {},
            )

    asyncio.run(drive())
    evidence = executor._evidence(
        ModelExecutionStatus.SUCCEEDED, "done", "end_turn", None, ()
    )
    assert len(evidence.tool_calls) == 15
    assert evidence.tool_calls[0].target == "5.py"
    assert evidence.tool_calls[-1].target == "19.py"


def test_token_budget_carries_budget_turns_and_elapsed(tmp_path: Path) -> None:
    executor = ModelExecutor(_executor_config(tmp_path))
    evidence = executor._evidence(
        ModelExecutionStatus.SUCCEEDED, "done", "end_turn", None, ()
    )
    assert isinstance(evidence.token_budget["budget_tokens"], int)
    assert evidence.token_budget["turns"] == 0
    assert isinstance(evidence.token_budget["elapsed_seconds"], int)
    assert evidence.token_budget["max_budget_tokens"] == 4_000_000


def test_handoff_skill_invoked_marks_last_intent() -> None:
    note = _build_emergency_handoff_note(
        issue_number=24,
        started_at=STARTED_AT,
        reason="turn_limit",
        last_work_commit="base-revision-0",
        explanation="Turns exhausted.",
        handoff_skill_invoked=True,
    )
    assert "Handoff skill on the main thread: invoked." in note

    events = (
        SkillEvent(phase="PreToolUse", name="handoff", agent_id=None,
                   timestamp=STARTED_AT),
    )
    from simple_coding_agent.lifecycle import _main_thread_handoff_invoked

    assert _main_thread_handoff_invoked(events) is True
    assert _main_thread_handoff_invoked(()) is False
