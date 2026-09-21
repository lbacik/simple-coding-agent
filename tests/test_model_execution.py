"""Behaviour tests for the model-execution boundary."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from claude_agent_sdk import AssistantMessage, TextBlock

from simple_coding_agent.config import RuntimeConfig
from simple_coding_agent.model_execution import (
    ModelExecutor,
    ModelExecutionStatus,
    publication_guard,
)


class FakeClient:
    def __init__(self, options: object, messages: list[object]) -> None:
        self.options = options
        self._messages = messages
        self.prompt: str | None = None
        self.interrupted = False

    async def __aenter__(self) -> FakeClient:
        return self

    async def __aexit__(self, *unused: object) -> None:
        return None

    async def connect(self, prompt: str) -> None:
        self.prompt = prompt

    async def interrupt(self) -> None:
        self.interrupted = True

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


def test_dispatches_the_issue_body_to_the_pinned_sdk_and_returns_execution_evidence(
    tmp_path: Path,
) -> None:
    captured: list[FakeClient] = []

    def client_factory(options: object) -> FakeClient:
        client = FakeClient(
            options,
            [
                SimpleNamespace(model="muse-spark-1.3-contributor"),
                result(),
            ],
        )
        captured.append(client)
        return client

    execution = asyncio.run(
        ModelExecutor(runtime_config(tmp_path), client_factory=client_factory).execute(
            issue_body="Fix the parser.", working_directory=tmp_path / "repo"
        )
    )

    assert execution.status is ModelExecutionStatus.SUCCEEDED
    assert execution.stop_reason == "end_turn"
    assert execution.model_usage == {"muse-spark-1.3-contributor": {"input_tokens": 12}}
    assert execution.observed_models == ("muse-spark-1.3-contributor",)
    assert captured[0].prompt == "/implement\n\nFix the parser."
    options = captured[0].options
    assert options.model == "muse-spark-1.3-contributor"
    assert options.fallback_model is None
    assert options.permission_mode == "bypassPermissions"
    assert options.max_turns == 60
    assert options.max_budget_usd == 5
    assert options.setting_sources == ["user"]
    assert options.skills == ["implement", "tdd", "code-review", "codebase-design"]
    assert options.env == {
        "ANTHROPIC_BASE_URL": "https://api.meta.ai",
        "ANTHROPIC_AUTH_TOKEN": "meta-secret",
        "ANTHROPIC_MODEL": "muse-spark-1.3-contributor",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "muse-spark-1.3-contributor",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "muse-spark-1.3-contributor",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "muse-spark-1.3-contributor",
        "CLAUDE_CODE_SUBAGENT_MODEL": "muse-spark-1.3-contributor",
        "CLAUDE_STREAM_IDLE_TIMEOUT_MS": "60000",
    }


def test_enables_project_settings_only_when_the_operator_explicitly_trusts_them(
    tmp_path: Path,
) -> None:
    captured: list[FakeClient] = []

    def client_factory(options: object) -> FakeClient:
        client = FakeClient(options, [result()])
        captured.append(client)
        return client

    asyncio.run(
        ModelExecutor(
            replace(runtime_config(tmp_path), agent_trust_project_settings=True),
            client_factory=client_factory,
        ).execute(issue_body="Fix it.", working_directory=tmp_path)
    )

    assert captured[0].options.setting_sources == ["user", "project"]
    assert captured[0].options.strict_mcp_config is True
    assert captured[0].options.extra_args == {"disable-all-hooks": None}


@pytest.mark.parametrize(
    ("terminal", "expected"),
    [
        (result(stop_reason="max_turns_exceeded"), ModelExecutionStatus.MODEL_LIMIT_REACHED),
        (result(stop_reason="aborted_by_user"), ModelExecutionStatus.INFRASTRUCTURE_ERROR),
        (result(stop_reason="timeout"), ModelExecutionStatus.INFRASTRUCTURE_ERROR),
        (result(is_error=True), ModelExecutionStatus.MODEL_LIMIT_REACHED),
    ],
)
def test_classifies_terminal_results_without_claiming_success(
    tmp_path: Path, terminal: object, expected: ModelExecutionStatus
) -> None:
    execution = asyncio.run(
        ModelExecutor(
            runtime_config(tmp_path), client_factory=lambda options: FakeClient(options, [terminal])
        ).execute(issue_body="Fix it.", working_directory=tmp_path)
    )

    assert execution.status is expected


def test_treats_an_observed_model_mismatch_as_an_infrastructure_error(tmp_path: Path) -> None:
    execution = asyncio.run(
        ModelExecutor(
            runtime_config(tmp_path),
            client_factory=lambda options: FakeClient(
                options, [SimpleNamespace(model="unexpected-model"), result()]
            ),
        ).execute(issue_body="Fix it.", working_directory=tmp_path)
    )

    assert execution.status is ModelExecutionStatus.INFRASTRUCTURE_ERROR
    assert "unexpected-model" in execution.explanation


def test_interrupts_and_drains_a_stalled_stream_after_model_timeout(tmp_path: Path) -> None:
    class StalledClient(FakeClient):
        async def receive_response(self):
            if not self.interrupted:
                await asyncio.Event().wait()
            yield result(stop_reason="aborted_by_timeout")

    captured: list[StalledClient] = []

    def client_factory(options: object) -> StalledClient:
        client = StalledClient(options, [])
        captured.append(client)
        return client

    execution = asyncio.run(
        ModelExecutor(
            replace(runtime_config(tmp_path), model_timeout=0), client_factory=client_factory
        ).execute(issue_body="Fix it.", working_directory=tmp_path)
    )

    assert execution.status is ModelExecutionStatus.INFRASTRUCTURE_ERROR
    assert captured[0].interrupted is True


@pytest.mark.parametrize(
    "command",
    [
        "git push origin HEAD",
        "git -C repository push origin HEAD",
        "cd repository && git push origin HEAD",
        "git status; gh issue close 21",
        "gh pr merge 42",
    ],
)
def test_publication_guard_denies_model_publication_commands_in_bypass_mode(command: str) -> None:
    decision = asyncio.run(
        publication_guard(
            {"tool_name": "Bash", "tool_input": {"command": command}}, None, {}
        )
    )

    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_publication_guard_allows_non_publication_commands() -> None:
    decision = asyncio.run(
        publication_guard(
            {"tool_name": "Bash", "tool_input": {"command": "git commit -m 'work'"}},
            None,
            {},
        )
    )

    assert decision == {}


def test_captures_skill_provenance_from_pre_and_post_tool_hooks(tmp_path: Path) -> None:
    captured: list[FakeClient] = []

    def client_factory(options: object) -> FakeClient:
        client = FakeClient(options, [result()])
        captured.append(client)
        return client

    executor = ModelExecutor(runtime_config(tmp_path), client_factory=client_factory)
    asyncio.run(executor.execute(issue_body="Fix it.", working_directory=tmp_path))
    pre_hook = captured[0].options.hooks["PreToolUse"][0].hooks[0]
    post_hook = captured[0].options.hooks["PostToolUse"][0].hooks[0]

    asyncio.run(
        pre_hook(
            {"tool_name": "Skill", "tool_input": {"skill": "code-review"}, "agent_id": "review-1"},
            None,
            {},
        )
    )
    asyncio.run(
        post_hook(
            {"tool_name": "Skill", "tool_input": {"skill": "code-review"}, "agent_id": "review-1"},
            None,
            {},
        )
    )

    assert [(event.phase, event.name, event.agent_id) for event in executor.skill_events] == [
        ("PreToolUse", "code-review", "review-1"),
        ("PostToolUse", "code-review", "review-1"),
    ]
    assert all(event.timestamp.endswith("Z") for event in executor.skill_events)


def test_logs_skill_invocations_under_distinct_event_names(tmp_path: Path) -> None:
    events: list[tuple[str, str]] = []
    captured: list[FakeClient] = []

    def client_factory(options: object) -> FakeClient:
        client = FakeClient(options, [result()])
        captured.append(client)
        return client

    executor = ModelExecutor(
        runtime_config(tmp_path),
        client_factory=client_factory,
        event_log=lambda event, detail="", issue_number=None: events.append((event, detail)),
    )
    asyncio.run(executor.execute(issue_body="Fix it.", working_directory=tmp_path))
    pre_hook = captured[0].options.hooks["PreToolUse"][0].hooks[0]
    post_hook = captured[0].options.hooks["PostToolUse"][0].hooks[0]

    asyncio.run(
        pre_hook(
            {"tool_name": "Skill", "tool_input": {"skill": "code-review"}, "agent_id": "review-1"},
            None,
            {},
        )
    )
    asyncio.run(
        post_hook(
            {"tool_name": "Skill", "tool_input": {"skill": "code-review"}, "agent_id": "review-1"},
            None,
            {},
        )
    )

    names = [event for event, _ in events]
    assert "tool_call" not in names
    assert "tool_result" not in names
    assert any(name == "skill_call" and "code-review" in detail for name, detail in events)
    assert any(name == "skill_result" and "code-review" in detail for name, detail in events)


class FakeArchive:
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self.written: dict[str, str] = {}

    def write_text(self, name: str, content: str) -> Path:
        path = self._directory / name
        path.write_text(content)
        self.written[name] = content
        return path


def test_offloads_tool_and_model_response_evidence_to_the_attempt_archive(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, str]] = []
    captured: list[FakeClient] = []
    archive = FakeArchive(tmp_path)

    def client_factory(options: object) -> FakeClient:
        client = FakeClient(
            options,
            [
                AssistantMessage(
                    content=[TextBlock(text="A" * 5000)],
                    model="muse-spark-1.3-contributor",
                ),
                result(),
            ],
        )
        captured.append(client)
        return client

    executor = ModelExecutor(
        runtime_config(tmp_path),
        client_factory=client_factory,
        event_log=lambda event, detail="", issue_number=None: events.append((event, detail)),
    )
    asyncio.run(
        executor.execute(
            issue_body="Fix the parser.", working_directory=tmp_path, archive=archive
        )
    )
    pre_hook = captured[0].options.hooks["PreToolUse"][0].hooks[0]
    post_hook = captured[0].options.hooks["PostToolUse"][0].hooks[0]
    asyncio.run(
        pre_hook({"tool_name": "Bash", "tool_input": {"command": "pytest -q" * 200}}, None, {})
    )
    asyncio.run(
        post_hook(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "pytest -q"},
                "tool_response": {"stdout": "ok" * 500, "interrupted": False, "is_error": False},
            },
            None,
            {},
        )
    )

    call_event = next(detail for name, detail in events if name == "tool_call")
    result_event = next(detail for name, detail in events if name == "tool_result")
    response_event = next(detail for name, detail in events if name == "model_response")

    assert len(call_event) < 200
    assert len(result_event) < 200
    assert len(response_event) < 200
    assert "(ok)" in result_event
    assert "chars=5000" in response_event
    assert any(
        "pytest -q" * 200 in content for content in archive.written.values()
    )
    assert any("A" * 5000 in content for content in archive.written.values())


def test_logs_the_process_even_when_the_attempt_succeeds(tmp_path: Path) -> None:
    events: list[tuple[str, str, int | None]] = []
    captured: list[FakeClient] = []

    def client_factory(options: object) -> FakeClient:
        client = FakeClient(
            options,
            [
                AssistantMessage(
                    content=[TextBlock(text="Looking at the parser now.")],
                    model="muse-spark-1.3-contributor",
                ),
                result(),
            ],
        )
        captured.append(client)
        return client

    executor = ModelExecutor(
        runtime_config(tmp_path),
        client_factory=client_factory,
        event_log=lambda event, detail="", issue_number=None: events.append(
            (event, detail, issue_number)
        ),
    )
    asyncio.run(
        executor.execute(
            issue_body="Fix the parser.", working_directory=tmp_path, issue_number=42
        )
    )
    pre_hook = captured[0].options.hooks["PreToolUse"][0].hooks[0]
    post_hook = captured[0].options.hooks["PostToolUse"][0].hooks[0]
    asyncio.run(
        pre_hook({"tool_name": "Bash", "tool_input": {"command": "pytest"}}, None, {})
    )
    asyncio.run(
        post_hook(
            {"tool_name": "Bash", "tool_input": {"command": "pytest"}, "tool_response": "ok"},
            None,
            {},
        )
    )

    names = [event for event, _, _ in events]
    assert names[0] == "model_execution_started"
    assert "model_response" in names
    assert "model_execution_finished" in names
    assert any(name == "tool_call" and "pytest" in detail for name, detail, _ in events)
    assert any(name == "tool_result" and "ok" in detail for name, detail, _ in events)
    assert all(issue_number == 42 for _, _, issue_number in events)
    assert any(
        "Looking at the parser now." in detail for name, detail, _ in events if name == "model_response"
    )


def test_blocks_a_third_repair_cycle_after_three_code_reviews(tmp_path: Path) -> None:
    captured: list[FakeClient] = []

    def client_factory(options: object) -> FakeClient:
        client = FakeClient(options, [result()])
        captured.append(client)
        return client

    executor = ModelExecutor(runtime_config(tmp_path), client_factory=client_factory)
    asyncio.run(executor.execute(issue_body="Fix it.", working_directory=tmp_path))
    pre_hook = captured[0].options.hooks["PreToolUse"][0].hooks[0]
    review = {"tool_name": "Skill", "tool_input": {"skill": "code-review"}}

    decisions = [asyncio.run(pre_hook(review, None, {})) for _ in range(4)]

    assert decisions[:3] == [{}, {}, {}]
    assert decisions[3]["hookSpecificOutput"]["permissionDecision"] == "deny"
