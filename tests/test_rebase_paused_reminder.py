"""Behaviour tests for the paused-rebase PostToolUse reminder.

While a rebase is paused in the attempt workspace (``.git/rebase-merge`` or
``.git/rebase-apply`` exists), every PostToolUse hook call reminds the model
to finish the rebase before doing issue work. The reminder is suppressed once
the soft threshold has crossed or an Operator Handoff context was delivered,
is logged exactly once per run, and never fails the tool call.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from simple_coding_agent.config import RuntimeConfig
from simple_coding_agent.model_execution import (
    ModelExecutor,
    OperatorHandoff,
)

_REMINDER_SENTENCE = "A git rebase is still paused"


class FakeClient:
    def __init__(self, options: object, messages: list[object]) -> None:
        self.options = options
        self._messages = messages

    async def disconnect(self) -> None:
        pass

    async def connect(self, prompt: str) -> None:
        pass

    async def interrupt(self) -> None:
        pass

    async def stop_task(self, task_id: str) -> None:
        pass

    async def query(self, prompt: str) -> None:
        pass

    async def receive_response(self):
        for message in self._messages:
            yield message


def runtime_config(tmp_path: Path) -> RuntimeConfig:
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


def result(*, is_error: bool = False, stop_reason: str | None = "end_turn") -> object:
    return SimpleNamespace(
        is_error=is_error,
        stop_reason=stop_reason,
        model_usage={"muse-spark-1.3-contributor": {"input_tokens": 12}},
    )


def _hooked_executor(tmp_path: Path, repo: Path):
    """An executed executor with its PostToolUse hook and event log."""

    captured: list[FakeClient] = []
    events: list[tuple[str, str]] = []

    def client_factory(options: object) -> FakeClient:
        client = FakeClient(options, [result()])
        captured.append(client)
        return client

    executor = ModelExecutor(
        replace(runtime_config(tmp_path), max_budget_tokens=1_000_000),
        client_factory=client_factory,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail)
        ),
    )
    asyncio.run(
        executor.execute(
            issue_body="Fix it.",
            working_directory=repo,
            issue_number=24,
            attempt_id="2026-10-11T13:48:18.485562Z",
        )
    )
    post_hook = captured[0].options.hooks["PostToolUse"][0].hooks[0]
    return executor, events, post_hook


def _pause_rebase(repo: Path, *, current: int | None = 3, total: int | None = 4) -> None:
    """Leave a paused rebase behind, optionally with pick counters."""

    state_dir = repo / ".git" / "rebase-merge"
    state_dir.mkdir(parents=True, exist_ok=True)
    if current is not None:
        (state_dir / "msgnum").write_text(f"{current}\n")
    if total is not None:
        (state_dir / "end").write_text(f"{total}\n")


def _cross_soft_threshold(executor: ModelExecutor) -> None:
    executor._ledger.observe_response("main", "r-big", 0)
    executor._ledger.observe_usage("main", "r-big", {"input_tokens": 900_000})
    executor._check_limits("main")
    assert executor._soft_threshold_crossed is True


def test_main_thread_post_tool_use_reminds_while_rebase_paused(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    executor, _, post_hook = _hooked_executor(tmp_path, repo)
    _pause_rebase(repo)

    reply = asyncio.run(
        post_hook(
            {"tool_name": "Bash", "tool_input": {"command": "pytest"}, "tool_response": "ok"},
            "call-1",
            {},
        )
    )

    context = reply["hookSpecificOutput"]["additionalContext"]
    assert reply["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert _REMINDER_SENTENCE in context
    assert "git rebase --continue" in context
    assert "Do not start or continue issue work" in context
    assert "2 of 4 picks remaining" in context


def test_subagent_post_tool_use_reminds_while_rebase_paused(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _, _, post_hook = _hooked_executor(tmp_path, repo)
    _pause_rebase(repo)

    reply = asyncio.run(
        post_hook(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "pytest"},
                "tool_response": "ok",
                "agent_id": "agent-1",
            },
            "call-1",
            {},
        )
    )

    assert _REMINDER_SENTENCE in reply["hookSpecificOutput"]["additionalContext"]


def test_no_paused_rebase_returns_todays_result(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _, _, post_hook = _hooked_executor(tmp_path, repo)

    outcome = asyncio.run(
        post_hook(
            {"tool_name": "Bash", "tool_input": {"command": "pytest"}, "tool_response": "ok"},
            "call-1",
            {},
        )
    )

    assert outcome == {}


def test_soft_threshold_crossing_suppresses_the_reminder(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    executor, _, post_hook = _hooked_executor(tmp_path, repo)
    _pause_rebase(repo)
    _cross_soft_threshold(executor)

    reply = asyncio.run(
        post_hook(
            {"tool_name": "Bash", "tool_input": {"command": "pytest"}, "tool_response": "ok"},
            "call-1",
            {},
        )
    )

    assert _REMINDER_SENTENCE not in reply["hookSpecificOutput"]["additionalContext"]


def test_delivered_operator_handoff_suppresses_the_reminder(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    executor, _, post_hook = _hooked_executor(tmp_path, repo)
    _pause_rebase(repo)
    executor.set_operator_handoff_provider(
        lambda: OperatorHandoff(request_id="req-1", accepted_at=datetime.now(UTC))
    )

    reply = asyncio.run(
        post_hook(
            {"tool_name": "Bash", "tool_input": {"command": "pytest"}, "tool_response": "ok"},
            "call-1",
            {},
        )
    )

    assert _REMINDER_SENTENCE not in reply["hookSpecificOutput"]["additionalContext"]
    assert "handoff" in reply["hookSpecificOutput"]["additionalContext"]


def test_missing_pick_counters_still_remind_without_a_count(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _, _, post_hook = _hooked_executor(tmp_path, repo)
    _pause_rebase(repo, current=None, total=None)

    reply = asyncio.run(
        post_hook(
            {"tool_name": "Bash", "tool_input": {"command": "pytest"}, "tool_response": "ok"},
            "call-1",
            {},
        )
    )

    context = reply["hookSpecificOutput"]["additionalContext"]
    assert _REMINDER_SENTENCE in context
    assert "remaining" not in context


def test_only_the_first_reminder_in_a_run_is_logged(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _, events, post_hook = _hooked_executor(tmp_path, repo)
    _pause_rebase(repo)
    hook_input = {
        "tool_name": "Bash",
        "tool_input": {"command": "pytest"},
        "tool_response": "ok",
    }

    first = asyncio.run(post_hook(hook_input, "call-1", {}))
    second = asyncio.run(post_hook(hook_input, "call-2", {}))

    assert _REMINDER_SENTENCE in first["hookSpecificOutput"]["additionalContext"]
    assert _REMINDER_SENTENCE in second["hookSpecificOutput"]["additionalContext"]
    reminders = [detail for event, detail in events if event == "rebase_paused_reminder"]
    assert len(reminders) == 1
    assert "remaining_picks=2" in reminders[0]


def test_unreadable_workspace_state_falls_back_to_todays_result(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    executor, _, post_hook = _hooked_executor(tmp_path, repo)
    executor._working_directory = tmp_path / "missing"

    outcome = asyncio.run(
        post_hook(
            {"tool_name": "Bash", "tool_input": {"command": "pytest"}, "tool_response": "ok"},
            "call-1",
            {},
        )
    )

    assert outcome == {}
