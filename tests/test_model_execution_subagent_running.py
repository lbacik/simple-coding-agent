"""Background subagent telemetry stays running until the task really ends."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    TextBlock,
    ToolResultBlock,
    UserMessage,
)

from simple_coding_agent.config import RuntimeConfig
from simple_coding_agent.model_execution import ModelExecutor


def _config(tmp_path: Path) -> RuntimeConfig:
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


def _executor(tmp_path: Path) -> tuple[ModelExecutor, list[tuple[str, str]]]:
    events: list[tuple[str, str]] = []
    executor = ModelExecutor(
        _config(tmp_path),
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail)
        ),
    )
    return executor, events


def _launch(executor: ModelExecutor, tool_use_id: str) -> None:
    executor._ensure_ledger_thread("main", prompt_chars=10)
    executor._start_launch_thread(
        {"tool_name": "Agent", "tool_input": {"prompt": "do work", "subagent_type": "Explore"}},
        tool_use_id,
    )


def _ack(tool_use_id: str, agent_id: str) -> UserMessage:
    return UserMessage(
        content=[
            ToolResultBlock(
                tool_use_id=tool_use_id,
                content=json.dumps(
                    {"agentId": agent_id, "isAsync": True, "outputFile": f"/tmp/{agent_id}.output"}
                ),
            )
        ]
    )


def _sub_response(thread_id: str, message_id: str, text: str = "work") -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=text)],
        model="muse-spark-1.3-contributor",
        message_id=message_id,
        parent_tool_use_id=thread_id,
    )


def _limits_for(events: list[tuple[str, str]], thread_id: str) -> list[str]:
    return [detail for name, detail in events if name == "limits_checked" and detail.startswith(f"thread={thread_id};")]


def test_foreground_result_marks_finished_and_settles_thread(tmp_path: Path) -> None:
    executor, events = _executor(tmp_path)
    _launch(executor, "toolu-fg")
    assert executor._subagent_info["toolu-fg"]["running"] is True

    executor._observe_assistant_message(_sub_response("toolu-fg", "s1", "sub work"))
    executor._observe_assistant_message(_sub_response("toolu-fg", "s2", "sub done"))
    executor._observe_user_message(
        UserMessage(content=[ToolResultBlock(tool_use_id="toolu-fg", content="done")])
    )

    assert executor._subagent_info["toolu-fg"]["running"] is False
    sub_entries = _limits_for(events, "toolu-fg")
    assert len(sub_entries) == executor._ledger.response_count("toolu-fg") == 2


def test_background_ack_keeps_running_and_settles_later(tmp_path: Path) -> None:
    executor, events = _executor(tmp_path)
    _launch(executor, "toolu-bg")
    executor._observe_assistant_message(_sub_response("toolu-bg", "s1", "partial"))
    assert _limits_for(events, "toolu-bg") == []

    executor._observe_user_message(_ack("toolu-bg", "agent-1"))

    assert executor._subagent_info["toolu-bg"]["running"] is True
    # The acknowledgement does not settle the in-flight subagent response.
    assert _limits_for(events, "toolu-bg") == []
    view = executor._live_attempt_view()
    by_id = {entry["id"]: entry for entry in view["subagents"]}
    assert by_id["toolu-bg"]["running"] is True

    # A response streaming afterwards settles the first one normally.
    executor._observe_assistant_message(_sub_response("toolu-bg", "s2", "more"))
    assert len(_limits_for(events, "toolu-bg")) == 1

    # The terminal task notification settles the last response: exactly one
    # entry per settled response, no duplicates.
    pending: dict = {}
    finished: list = []
    executor._observe_background_task(
        TaskStartedMessage(
            task_type="local_agent",
            subtype="task_started",
            data={},
            task_id="task-1",
            description="bg",
            uuid="u-started",
            session_id="sess-1",
            tool_use_id="toolu-bg",
        ),
        pending,
        finished,
    )
    executor._observe_background_task(
        TaskNotificationMessage(
            subtype="task_notification",
            data={},
            task_id="task-1",
            status="completed",  # type: ignore[arg-type]
            output_file="/tmp/task-1.txt",
            summary="done",
            uuid="u-notified",
            session_id="sess-1",
            tool_use_id="toolu-bg",
        ),
        pending,
        finished,
    )

    assert executor._subagent_info["toolu-bg"]["running"] is False
    sub_entries = _limits_for(events, "toolu-bg")
    assert len(sub_entries) == executor._ledger.response_count("toolu-bg") == 2


def test_background_completes_on_updated_with_only_task_id(tmp_path: Path) -> None:
    executor, _ = _executor(tmp_path)
    _launch(executor, "toolu-bg2")
    executor._observe_user_message(_ack("toolu-bg2", "agent-2"))
    assert executor._subagent_info["toolu-bg2"]["running"] is True

    pending: dict = {}
    finished: list = []
    executor._observe_background_task(
        TaskStartedMessage(
            task_type="local_agent",
            subtype="task_started",
            data={},
            task_id="task-2",
            description="bg2",
            uuid="u2-started",
            session_id="sess-1",
            tool_use_id="toolu-bg2",
        ),
        pending,
        finished,
    )
    assert executor._subagent_info["toolu-bg2"]["running"] is True
    executor._observe_background_task(
        TaskUpdatedMessage(
            subtype="task_updated",
            data={},
            task_id="task-2",
            patch={"status": "completed"},
            status="completed",  # type: ignore[arg-type]
        ),
        pending,
        finished,
    )

    assert executor._subagent_info["toolu-bg2"]["running"] is False
    assert pending == {}


def test_background_stopped_at_soft_threshold(tmp_path: Path) -> None:
    executor, events = _executor(tmp_path)
    _launch(executor, "toolu-bg3")
    executor._observe_user_message(_ack("toolu-bg3", "agent-3"))
    assert executor._subagent_info["toolu-bg3"]["running"] is True

    # The soft-threshold path tracks hook agent ids; the ack links agent-3
    # back to the launch tool_use_id.
    executor._track_subagent_seen({"tool_name": "Read", "agent_id": "agent-3"})
    executor._stop_running_subagents()

    assert executor._subagent_info["toolu-bg3"]["running"] is False
    assert any(name == "token_soft_threshold_subagents_stopped" for name, _ in events)
    view = executor._live_attempt_view()
    by_id = {entry["id"]: entry for entry in view["subagents"]}
    assert by_id["toolu-bg3"]["running"] is False


def test_background_stopped_by_executor_at_terminal(tmp_path: Path) -> None:
    executor, _ = _executor(tmp_path)
    _launch(executor, "toolu-bg4")
    executor._observe_user_message(_ack("toolu-bg4", "agent-4"))

    executor._task_tool_links["task-4"] = "toolu-bg4"

    class _Client:
        def __init__(self) -> None:
            self.stopped: list[str] = []

        async def stop_task(self, task_id: str) -> None:
            self.stopped.append(task_id)

    client = _Client()
    asyncio.run(executor._stop_background_tasks(client, ["task-4"], "result is terminal"))

    assert client.stopped == ["task-4"]
    assert executor._subagent_info["toolu-bg4"]["running"] is False


def test_background_failed_stop_keeps_running(tmp_path: Path) -> None:
    executor, _ = _executor(tmp_path)
    _launch(executor, "toolu-bg5")
    executor._observe_user_message(_ack("toolu-bg5", "agent-5"))

    executor._task_tool_links["task-5"] = "toolu-bg5"

    class _FailingClient:
        async def stop_task(self, task_id: str) -> None:
            raise RuntimeError(f"cannot stop {task_id}")

    asyncio.run(executor._stop_background_tasks(_FailingClient(), ["task-5"], "result is terminal"))

    assert executor._subagent_info["toolu-bg5"]["running"] is True
