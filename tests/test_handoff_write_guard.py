"""The handoff guards steer the model to the `Write` tool and close the Bash redirect hole.

After the token soft threshold (and during an operator handoff) only handoff
work stays allowed. The denial must name the `Write` tool and the exact note
path, Bash output redirects must not escape the note directory, and commands
the shell splitter cannot tokenize are denied without raising.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from simple_coding_agent.config import RuntimeConfig
from simple_coding_agent.model_execution import (
    ModelExecutor,
    _deletes_handoff_note,
    _is_handoff_command,
    _token_handoff_steps,
)


class FakeClient:
    def __init__(self, options: object, messages: list[object]) -> None:
        self.options = options
        self._messages = messages

    async def disconnect(self) -> None:
        self.disconnect_count = getattr(self, "disconnect_count", 0) + 1

    async def connect(self, prompt: str) -> None:
        return None

    async def interrupt(self) -> None:
        return None

    async def stop_task(self, task_id: str) -> None:
        self.stopped_tasks = [*getattr(self, 'stopped_tasks', []), task_id]

    async def query(self, prompt: str) -> None:
        return None

    async def receive_response(self):
        for message in self._messages:
            yield message


def _runtime_config(tmp_path: Path) -> RuntimeConfig:
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


def _result() -> object:
    return SimpleNamespace(
        is_error=False,
        stop_reason="end_turn",
        model_usage={"muse-spark-1.3-contributor": {"input_tokens": 12}},
    )


def _hooked_executor(tmp_path: Path, *, issue_number: int = 287):
    captured: list[FakeClient] = []

    def client_factory(options: object) -> FakeClient:
        client = FakeClient(options, [_result()])
        captured.append(client)
        return client

    executor = ModelExecutor(
        _runtime_config(tmp_path), client_factory=client_factory
    )
    asyncio.run(
        executor.execute(
            issue_body="Fix it.",
            working_directory=tmp_path,
            issue_number=issue_number,
        )
    )
    pre_hook = captured[0].options.hooks["PreToolUse"][0].hooks[0]
    return executor, pre_hook


def _deny_reason(decision: dict) -> str:
    return decision["hookSpecificOutput"]["permissionDecisionReason"]


REJECTED_COMMANDS = (
    "echo 'x = 1' > simple_coding_agent/foo.py",
    "echo x >simple_coding_agent/foo.py",
    "cat > src.py <<'EOF'\nprint(1)\nEOF",
    "git log >> out.txt",
    "git status 2>err.log",
    "echo x > .agent/handoff_backup/1.md",
)

ALLOWED_COMMANDS = (
    'git add .agent/handoff/287.md && git commit -m "Handoff note: issue #287"',
    "git log --oneline | tail -3",
    "git status 2>&1",
    "git diff > /dev/null",
    "echo x > .agent/handoff/287.md",
    "mkdir -p .agent/handoff",
)


def test_handoff_allowlist_rejects_redirects_outside_the_note_dir() -> None:
    for command in REJECTED_COMMANDS:
        assert _is_handoff_command(command) is False, command


def test_handoff_allowlist_keeps_handoff_work_allowed() -> None:
    for command in ALLOWED_COMMANDS:
        assert _is_handoff_command(command) is True, command


def test_archived_heredoc_note_with_semicolon_body_is_denied() -> None:
    """The shape of run 2026-10-21T18-40-02 call 0189: denied, not half-run."""

    command = (
        "cat > .agent/handoff/283.md <<'NOTE_EOF'\n"
        "# Handoff note: issue #283\n"
        "\n"
        "implementation is complete in main; remaining work is listed below\n"
        "NOTE_EOF"
    )
    assert _is_handoff_command(command) is False


def test_unbalanced_quote_is_denied_without_raising() -> None:
    command = "cat <<'EOF'\nthe run didn't finish\nEOF"
    assert _is_handoff_command(command) is False
    assert _deletes_handoff_note(command) is False


def test_soft_threshold_denial_directs_to_write_tool(tmp_path: Path) -> None:
    executor, pre_hook = _hooked_executor(tmp_path)
    executor._soft_threshold_crossed = True

    denied = asyncio.run(
        pre_hook(
            {"tool_name": "Bash", "tool_input": {"command": "pytest -q"}},
            None,
            {},
        )
    )
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    reason = _deny_reason(denied)
    assert "`Write`" in reason
    assert ".agent/handoff/287.md" in reason
    assert "Bash cannot create or modify files" in reason


def test_soft_threshold_redirect_denial_directs_to_write_tool(
    tmp_path: Path,
) -> None:
    executor, pre_hook = _hooked_executor(tmp_path)
    executor._soft_threshold_crossed = True

    for command in REJECTED_COMMANDS:
        denied = asyncio.run(
            pre_hook({"tool_name": "Bash", "tool_input": {"command": command}}, None, {})
        )
        assert denied["hookSpecificOutput"]["permissionDecision"] == "deny", command
        assert "`Write`" in _deny_reason(denied), command

    archived = (
        "cat > .agent/handoff/283.md <<'NOTE_EOF'\n"
        "implementation is complete in main; remaining work below\n"
        "NOTE_EOF"
    )
    denied = asyncio.run(
        pre_hook({"tool_name": "Bash", "tool_input": {"command": archived}}, None, {})
    )
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "`Write`" in _deny_reason(denied)


def test_soft_threshold_unbalanced_quote_denied_without_raising(
    tmp_path: Path,
) -> None:
    executor, pre_hook = _hooked_executor(tmp_path)
    executor._soft_threshold_crossed = True

    denied = asyncio.run(
        pre_hook(
            {"tool_name": "Bash", "tool_input": {"command": "echo didn't finish"}},
            None,
            {},
        )
    )
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "`Write`" in _deny_reason(denied)


def test_soft_threshold_allows_handoff_bash_commands(tmp_path: Path) -> None:
    executor, pre_hook = _hooked_executor(tmp_path)
    executor._soft_threshold_crossed = True

    for command in ALLOWED_COMMANDS:
        assert (
            asyncio.run(
                pre_hook(
                    {"tool_name": "Bash", "tool_input": {"command": command}}, None, {}
                )
            )
            == {}
        ), command


def test_operator_handoff_denial_names_write_and_note_path(
    tmp_path: Path,
) -> None:
    executor, pre_hook = _hooked_executor(tmp_path)
    executor._operator_context_delivered = True

    denied_bash = asyncio.run(
        pre_hook(
            {"tool_name": "Bash", "tool_input": {"command": "pytest -q"}},
            None,
            {},
        )
    )
    assert denied_bash["hookSpecificOutput"]["permissionDecision"] == "deny"
    bash_reason = _deny_reason(denied_bash)
    assert "`Write`" in bash_reason
    assert ".agent/handoff/287.md" in bash_reason

    denied_edit = asyncio.run(
        pre_hook(
            {"tool_name": "Edit", "tool_input": {"file_path": "src/a.py"}},
            None,
            {},
        )
    )
    assert denied_edit["hookSpecificOutput"]["permissionDecision"] == "deny"
    edit_reason = _deny_reason(denied_edit)
    assert "`Write`" in edit_reason
    assert ".agent/handoff/287.md" in edit_reason


def test_operator_handoff_redirect_and_unbalanced_quote_denied(
    tmp_path: Path,
) -> None:
    executor, pre_hook = _hooked_executor(tmp_path)
    executor._operator_context_delivered = True

    for command in (*REJECTED_COMMANDS, "echo didn't finish"):
        denied = asyncio.run(
            pre_hook({"tool_name": "Bash", "tool_input": {"command": command}}, None, {})
        )
        assert denied["hookSpecificOutput"]["permissionDecision"] == "deny", command

    for command in ALLOWED_COMMANDS:
        assert (
            asyncio.run(
                pre_hook(
                    {"tool_name": "Bash", "tool_input": {"command": command}}, None, {}
                )
            )
            == {}
        ), command


def test_no_guard_keeps_bash_behavior_unchanged(tmp_path: Path) -> None:
    _, pre_hook = _hooked_executor(tmp_path)

    for command in (*REJECTED_COMMANDS, *ALLOWED_COMMANDS, "pytest -q"):
        assert (
            asyncio.run(
                pre_hook(
                    {"tool_name": "Bash", "tool_input": {"command": command}}, None, {}
                )
            )
            == {}
        ), command


def test_token_handoff_steps_name_the_write_tool() -> None:
    steps = _token_handoff_steps(287)
    assert "`Write`" in steps
    assert ".agent/handoff/287.md" in steps
    assert "git add" in steps
    assert "git commit" in steps
    assert "Bash cannot create or modify files" in steps


def test_handoff_skill_names_the_write_tool() -> None:
    skill = (Path(__file__).resolve().parent.parent / "skills" / "handoff" / "SKILL.md").read_text()
    assert "`Write` tool" in skill
    assert "Bash cannot" in skill
    assert "git add" in skill
    assert "git commit" in skill
