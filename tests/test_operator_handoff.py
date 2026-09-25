"""Behaviour tests for ``agentctl handoff now`` during an active attempt.

The operator request is delivered at the next safe SDK boundary (after an
in-flight tool finishes) or once through the defined fallback, never by
interrupting an in-flight tool early. Model handoff work ends by the
original 240-second deadline and never bypasses the hard cost limit.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from claude_agent_sdk import AssistantMessage

from simple_coding_agent.config import RuntimeConfig
from simple_coding_agent.model_execution import (
    ModelExecutor,
    ModelExecutionStatus,
    OperatorHandoff,
)


class FakeClient:
    def __init__(
        self,
        options: object,
        messages: list[object],
        *,
        followup_messages: list[object] | None = None,
    ) -> None:
        self.options = options
        self._messages = messages
        self._followup_messages = followup_messages or []
        self.prompt: str | None = None
        self.interrupted = False
        self.queried_prompts: list[str] = []

    async def __aenter__(self) -> FakeClient:
        return self

    async def __aexit__(self, *unused: object) -> None:
        return None

    async def connect(self, prompt: str) -> None:
        self.prompt = prompt

    async def interrupt(self) -> None:
        self.interrupted = True

    async def query(self, prompt: str) -> None:
        self.queried_prompts.append(prompt)

    async def receive_response(self):
        if self.queried_prompts:
            for message in self._followup_messages:
                yield message
            return
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


def skill_hook_input(name: str) -> dict:
    return {"tool_name": "Skill", "tool_input": {"skill": name}}


def make_executor(
    tmp_path: Path,
    client: FakeClient,
    *,
    handoff: OperatorHandoff | None = None,
    now: datetime | None = None,
) -> tuple[ModelExecutor, list[tuple[str, str]]]:
    current = now or datetime.now(UTC)
    reported: list[tuple[str, str]] = []
    executor = ModelExecutor(
        runtime_config(tmp_path),
        client_factory=lambda options: client,
        clock=lambda: current,
    )
    if handoff is not None:
        executor.set_operator_handoff_provider(lambda: handoff)
    executor.set_operator_handoff_reporter(
        lambda event, request_id: reported.append((event, request_id))
    )
    return executor, reported


def test_operator_instruction_is_delivered_once_at_a_post_tool_boundary(
    tmp_path: Path,
) -> None:
    client = FakeClient(object(), [result()])
    executor, reported = make_executor(
        tmp_path, client, handoff=OperatorHandoff("req-1", datetime.now(UTC))
    )

    first = asyncio.run(
        executor._record_post_tool_use(
            {"tool_name": "Edit", "tool_input": {"file_path": "src/a.py"}}, None, {}
        )
    )
    second = asyncio.run(
        executor._record_post_tool_use(
            {"tool_name": "Edit", "tool_input": {"file_path": "src/b.py"}}, None, {}
        )
    )

    context = first["hookSpecificOutput"]["additionalContext"]
    assert "operator_request" in context
    assert "handoff" in context
    assert reported == [("delivered", "req-1")]
    assert second == {}


def test_operator_delivery_merges_with_cost_soft_threshold_instruction(
    tmp_path: Path,
) -> None:
    client = FakeClient(object(), [result()])
    executor, _ = make_executor(
        tmp_path, client, handoff=OperatorHandoff("req-1", datetime.now(UTC))
    )
    executor._soft_threshold_crossed = True

    reply = asyncio.run(
        executor._record_post_tool_use(
            {"tool_name": "Edit", "tool_input": {"file_path": "src/a.py"}}, None, {}
        )
    )

    context = reply["hookSpecificOutput"]["additionalContext"]
    assert "operator_request" in context
    assert "cost budget" in context


def test_begun_is_reported_only_when_the_handoff_skill_is_observed(
    tmp_path: Path,
) -> None:
    client = FakeClient(object(), [result()])
    executor, reported = make_executor(
        tmp_path, client, handoff=OperatorHandoff("req-1", datetime.now(UTC))
    )

    asyncio.run(executor._record_post_tool_use(skill_hook_input("implement"), None, {}))
    asyncio.run(
        executor._record_skill_event(
            "PreToolUse", skill_hook_input("implement"), "tool-1", {}
        )
    )
    assert reported == [("delivered", "req-1")]
    assert executor.operator_request_id == "req-1"

    asyncio.run(
        executor._record_skill_event(
            "PreToolUse", skill_hook_input("handoff"), "tool-2", {}
        )
    )
    assert reported == [("delivered", "req-1"), ("begun", "req-1")]


def test_pre_tool_guard_denies_non_handoff_work_only_after_delivery(
    tmp_path: Path,
) -> None:
    client = FakeClient(object(), [result()])
    executor, _ = make_executor(
        tmp_path, client, handoff=OperatorHandoff("req-1", datetime.now(UTC))
    )

    before = asyncio.run(
        executor._guard_and_record_pre_tool_use(
            {"tool_name": "Edit", "tool_input": {"file_path": "src/a.py"}}, None, {}
        )
    )
    assert before == {}

    asyncio.run(
        executor._record_post_tool_use(
            {"tool_name": "Edit", "tool_input": {"file_path": "src/a.py"}}, None, {}
        )
    )

    denied = asyncio.run(
        executor._guard_and_record_pre_tool_use(
            {"tool_name": "Edit", "tool_input": {"file_path": "src/a.py"}}, None, {}
        )
    )
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    denied_skill = asyncio.run(
        executor._guard_and_record_pre_tool_use(
            skill_hook_input("code-review"), None, {}
        )
    )
    assert denied_skill["hookSpecificOutput"]["permissionDecision"] == "deny"
    denied_push = asyncio.run(
        executor._guard_and_record_pre_tool_use(
            {"tool_name": "Bash", "tool_input": {"command": "git push origin HEAD"}},
            None,
            {},
        )
    )
    assert denied_push["hookSpecificOutput"]["permissionDecision"] == "deny"

    assert (
        asyncio.run(
            executor._guard_and_record_pre_tool_use(
                skill_hook_input("handoff"), None, {}
            )
        )
        == {}
    )
    assert (
        asyncio.run(
            executor._guard_and_record_pre_tool_use(
                {
                    "tool_name": "Write",
                    "tool_input": {"file_path": ".agent/handoff/24.md"},
                },
                None,
                {},
            )
        )
        == {}
    )
    assert (
        asyncio.run(
            executor._guard_and_record_pre_tool_use(
                {"tool_name": "Bash", "tool_input": {"command": "git commit -m work"}},
                None,
                {},
            )
        )
        == {}
    )


def test_execution_without_a_provider_is_untouched(tmp_path: Path) -> None:
    client = FakeClient(object(), [result()])
    executor = ModelExecutor(
        runtime_config(tmp_path), client_factory=lambda options: client
    )

    execution = asyncio.run(
        executor.execute(issue_body="Fix it.", working_directory=tmp_path)
    )

    assert execution.status is ModelExecutionStatus.SUCCEEDED
    assert client.queried_prompts == []
    assert client.interrupted is False


def test_fallback_runs_once_when_the_safe_boundary_is_missed(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    client = FakeClient(object(), [result()], followup_messages=[result()])
    executor, reported = make_executor(
        tmp_path,
        client,
        handoff=OperatorHandoff("req-1", now - timedelta(seconds=61)),
        now=now,
    )

    execution = asyncio.run(
        executor.execute(issue_body="Fix it.", working_directory=tmp_path)
    )

    assert client.interrupted is True
    assert len(client.queried_prompts) == 1
    assert "operator_request" in client.queried_prompts[0]
    assert execution.status is ModelExecutionStatus.SUCCEEDED
    assert reported == [("delivered", "req-1")]


def test_no_fallback_inside_the_delivery_window(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    client = FakeClient(object(), [result()])
    executor, _ = make_executor(
        tmp_path,
        client,
        handoff=OperatorHandoff("req-1", now - timedelta(seconds=30)),
        now=now,
    )

    execution = asyncio.run(
        executor.execute(issue_body="Fix it.", working_directory=tmp_path)
    )

    assert execution.status is ModelExecutionStatus.SUCCEEDED
    assert client.queried_prompts == []
    assert client.interrupted is False


def test_fallback_never_bypasses_the_hard_cost_limit(tmp_path: Path) -> None:
    # The request arrives mid-stream after heavy usage already pushed the
    # estimate past the hard ceiling: the single fallback must be consumed
    # without issuing its query, and the stream's own terminal result wins.
    now = datetime.now(UTC)
    heavy_usage = AssistantMessage(
        content=[],
        model="muse-spark-1.3-contributor",
        usage={"input_tokens": 400_000},
    )
    client = FakeClient(object(), [heavy_usage, result()])
    current: list[OperatorHandoff | None] = [None]
    executor = ModelExecutor(
        runtime_config(tmp_path),
        client_factory=lambda options: client,
        clock=lambda: now,
    )
    executor.set_operator_handoff_provider(lambda: current[0])

    async def run() -> object:
        task = asyncio.ensure_future(
            executor.execute(issue_body="Fix it.", working_directory=tmp_path)
        )
        for _ in range(1000):
            await asyncio.sleep(0)
            if executor._cost_estimator.estimated_cost_usd >= 5:
                break
        current[0] = OperatorHandoff("req-1", now - timedelta(seconds=61))
        return await task

    execution = asyncio.run(run())

    assert execution.status is ModelExecutionStatus.SUCCEEDED
    assert client.queried_prompts == []
    assert client.interrupted is False


def test_model_handoff_work_ends_at_the_original_deadline(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    client = FakeClient(object(), [result()])
    executor, _ = make_executor(
        tmp_path,
        client,
        handoff=OperatorHandoff("req-1", now - timedelta(seconds=241)),
        now=now,
    )

    execution = asyncio.run(
        executor.execute(issue_body="Fix it.", working_directory=tmp_path)
    )

    assert execution.status is ModelExecutionStatus.OPERATOR_HANDOFF_EXPIRED
    assert "req-1" in execution.explanation
    assert "240" in execution.explanation
    assert client.interrupted is True
    assert client.queried_prompts == []


def test_stream_end_without_an_operator_request_stays_an_infrastructure_error(
    tmp_path: Path,
) -> None:
    client = FakeClient(object(), [])
    executor = ModelExecutor(
        runtime_config(tmp_path), client_factory=lambda options: client
    )

    execution = asyncio.run(
        executor.execute(issue_body="Fix it.", working_directory=tmp_path)
    )

    assert execution.status is ModelExecutionStatus.INFRASTRUCTURE_ERROR


def test_handoff_requested_when_the_skill_is_observed_after_delivery(
    tmp_path: Path,
) -> None:
    client = FakeClient(object(), [result()])
    executor, _ = make_executor(
        tmp_path, client, handoff=OperatorHandoff("req-1", datetime.now(UTC))
    )
    asyncio.run(
        executor._record_post_tool_use(
            {"tool_name": "Edit", "tool_input": {"file_path": "src/a.py"}}, None, {}
        )
    )
    asyncio.run(
        executor._record_skill_event(
            "PreToolUse", skill_hook_input("handoff"), "tool-1", {}
        )
    )

    execution = executor._classify(result(), ())

    assert execution.status is ModelExecutionStatus.HANDOFF_REQUESTED


def test_predelivery_adopts_a_newer_request_but_delivery_pins_the_first(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    client = FakeClient(object(), [result()])
    current: list[OperatorHandoff | None] = [OperatorHandoff("req-1", now)]
    executor = ModelExecutor(
        runtime_config(tmp_path), client_factory=lambda options: client
    )
    executor.set_operator_handoff_provider(lambda: current[0])

    executor._poll_operator_handoff()
    assert executor.operator_request_id == "req-1"

    current[0] = OperatorHandoff("req-2", now)
    executor._poll_operator_handoff()
    assert executor.operator_request_id == "req-2"

    asyncio.run(
        executor._record_post_tool_use(
            {"tool_name": "Edit", "tool_input": {"file_path": "src/a.py"}}, None, {}
        )
    )
    current[0] = OperatorHandoff("req-3", now)
    executor._poll_operator_handoff()
    assert executor.operator_request_id == "req-2"


def test_a_failing_reporter_never_breaks_delivery(tmp_path: Path) -> None:
    client = FakeClient(object(), [result()])
    executor = ModelExecutor(
        runtime_config(tmp_path), client_factory=lambda options: client
    )
    executor.set_operator_handoff_provider(
        lambda: OperatorHandoff("req-1", datetime.now(UTC))
    )

    def bad_reporter(event: str, request_id: str) -> None:
        raise RuntimeError("report channel down")

    executor.set_operator_handoff_reporter(bad_reporter)

    reply = asyncio.run(
        executor._record_post_tool_use(
            {"tool_name": "Edit", "tool_input": {"file_path": "src/a.py"}}, None, {}
        )
    )

    assert "operator_request" in reply["hookSpecificOutput"]["additionalContext"]
