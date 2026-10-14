"""Context figures in telemetry (issue #245).

``context.main``/``main_peak`` come from the ledger, ``window`` and
``auto_compact_threshold`` from the SDK context-usage call (session start
and after each compaction only), ``compactions`` from compaction boundary
events, and ``subagents`` from ledger threads with running flags.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import MappingProxyType

from claude_agent_sdk import AssistantMessage, SystemMessage

from simple_coding_agent.config import RuntimeConfig
from simple_coding_agent.model_execution import ModelExecutor
from tests.test_telemetry_live_figures import (
    claim_seed,
    make_reporter,
    raw_view,
)


class StubControl:
    def __init__(self) -> None:
        self.section = {"control_observed_at": "2026-10-14T12:00:00Z"}

    def telemetry_control_snapshot(self) -> dict:
        return dict(self.section)


def executor_config(tmp_path: Path, **overrides: object) -> RuntimeConfig:
    values: dict[str, object] = {
        "github_token": "github-secret",
        "meta_api_key": "meta-secret",
        "target_repo": "octo/example",
        "data_dir": tmp_path,
        "clone_dir": tmp_path / "repo",
        "poll_interval": 60,
        "log_level": "INFO",
        "model_timeout": 60,
        "publish_timeout": 120,
        "max_retries": 3,
        "max_consecutive_errors": 3,
        "agent_trust_project_settings": False,
        "review_blocking_severities": frozenset({"must-fix"}),
    }
    values.update(overrides)
    return RuntimeConfig(**values)  # type: ignore[arg-type]


def usage_message(index: int, input_tokens: int) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        model="muse-spark-1.3-contributor",
        usage={
            "input_tokens": input_tokens,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 10,
        },
        message_id=f"r{index}",
    )


def terminal_result() -> object:
    from types import SimpleNamespace

    return SimpleNamespace(
        is_error=False,
        stop_reason="end_turn",
        model_usage={"muse-spark-1.3-contributor": {"input_tokens": 12}},
    )


class FakeClient:
    """SDK fake with a scripted context-usage response."""

    instances: list["FakeClient"] = []

    def __init__(
        self,
        options: object,
        messages: list[object],
        usage: dict | Exception | None = None,
    ) -> None:
        self.options = options
        self._messages = messages
        self._usage = (
            {
                "rawMaxTokens": 200000,
                "maxTokens": 200000,
                "autoCompactThreshold": 190000,
                "totalTokens": 50000,
            }
            if usage is None
            else usage
        )
        self.usage_calls = 0
        FakeClient.instances.append(self)

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, *unused: object) -> None:
        return None

    async def connect(self, prompt: str) -> None:
        pass

    async def interrupt(self) -> None:
        pass

    async def query(self, prompt: str) -> None:
        pass

    async def get_context_usage(self) -> dict:
        self.usage_calls += 1
        if isinstance(self._usage, Exception):
            raise self._usage
        self._usage["totalTokens"] = 50000 + self.usage_calls * 1000
        return dict(self._usage)

    async def receive_response(self):
        for message in self._messages:
            yield message


def run_execute(tmp_path: Path, messages: list[object], usage=None) -> tuple[ModelExecutor, FakeClient]:
    FakeClient.instances.clear()
    published: list[object] = []
    executor = ModelExecutor(
        executor_config(tmp_path),
        client_factory=lambda options: FakeClient(options, messages, usage),
        attempt_publisher=published.append,
    )
    asyncio.run(executor.execute(issue_body="Fix it.", working_directory=tmp_path / "repo"))
    assert FakeClient.instances
    return executor, FakeClient.instances[0]


def context_view(**overrides: object) -> dict:
    view = raw_view()
    view.update(
        {
            "main_context_tokens": 148000.0,
            "main_peak_context_tokens": 151200.0,
            "context_window": 200000,
            "context_window_basis": "sdk_assumed",
            "context_window_reason": None,
            "auto_compact_threshold": 190000,
            "auto_compact_threshold_basis": "sdk_assumed",
            "auto_compact_threshold_reason": None,
            "compaction_count": 0,
            "compaction_last": None,
            "subagents": (),
        }
    )
    view.update(overrides)
    return view


# ---------------------------------------------------------------------------
# reporter shape
# ---------------------------------------------------------------------------


def test_reporter_context_mains_use_ledger_basis() -> None:
    reporter = make_reporter()
    reporter.begin_attempt(claim_seed())
    reporter.publish_attempt_view(context_view(main_context_basis="measured"))

    current = reporter.snapshot(StubControl())["current_attempt"]

    assert current is not None
    assert current["context"]["main"] == {"value": 148000, "basis": "measured"}
    assert current["context"]["main_peak"] == {"value": 151200, "basis": "measured"}
    assert current["context"]["window"] == {"value": 200000, "basis": "sdk_assumed"}
    assert current["context"]["auto_compact_threshold"] == {
        "value": 190000,
        "basis": "sdk_assumed",
    }
    assert current["context"]["compactions"] == {"count": 0, "last": None}
    assert current["context"]["subagents"] == []


def test_reporter_window_configured_basis() -> None:
    reporter = make_reporter()
    reporter.begin_attempt(claim_seed())
    reporter.publish_attempt_view(
        context_view(
            context_window_basis="configured",
            auto_compact_threshold_basis="configured",
        )
    )

    current = reporter.snapshot(StubControl())["current_attempt"]

    assert current is not None
    assert current["context"]["window"] == {"value": 200000, "basis": "configured"}
    assert current["context"]["auto_compact_threshold"] == {
        "value": 190000,
        "basis": "configured",
    }


def test_reporter_window_unavailable_carries_reason() -> None:
    reporter = make_reporter()
    reporter.begin_attempt(claim_seed())
    reporter.publish_attempt_view(
        context_view(
            context_window=None,
            context_window_basis="unavailable",
            context_window_reason="context usage failed: RuntimeError",
            auto_compact_threshold=None,
            auto_compact_threshold_basis="unavailable",
            auto_compact_threshold_reason="context usage failed: RuntimeError",
        )
    )

    current = reporter.snapshot(StubControl())["current_attempt"]

    assert current is not None
    assert current["context"]["window"] == {
        "value": None,
        "basis": "unavailable",
        "reason": "context usage failed: RuntimeError",
    }


def test_reporter_subagents_running_first_and_capped() -> None:
    subagents = tuple(
        {
            "id": f"sub-{index:02d}",
            "type": "Explore",
            "context": 1000.0 + index,
            "running": index % 2 == 0,
        }
        for index in range(25)
    )
    reporter = make_reporter()
    reporter.begin_attempt(claim_seed())
    reporter.publish_attempt_view(context_view(subagents=subagents))

    current = reporter.snapshot(StubControl())["current_attempt"]

    assert current is not None
    listed = current["context"]["subagents"]
    assert len(listed) == 20
    # Running entries come first.
    seen_finished = False
    for entry in listed:
        if not entry["running"]:
            seen_finished = True
        else:
            assert not seen_finished
        assert entry["context"]["basis"] == "estimated"
        assert isinstance(entry["context"]["value"], int)


def test_reporter_tracer_context_is_unavailable() -> None:
    reporter = make_reporter()
    reporter.begin_attempt(claim_seed())

    current = reporter.snapshot(StubControl())["current_attempt"]

    assert current is not None
    assert current["context"]["main"]["basis"] == "unavailable"
    assert current["context"]["window"]["basis"] == "unavailable"
    assert current["context"]["compactions"] == {"count": 0, "last": None}
    assert current["context"]["subagents"] == []


# ---------------------------------------------------------------------------
# executor behaviour
# ---------------------------------------------------------------------------


def test_window_basis_sdk_assumed_without_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("CLAUDE_CODE_MAX_CONTEXT_TOKENS", raising=False)
    executor, client = run_execute(tmp_path, [usage_message(0, 1000), terminal_result()])

    assert client.usage_calls == 1
    view = dict(executor._live_attempt_view())  # noqa: SLF001
    assert view["context_window"] == 200000
    assert view["context_window_basis"] == "sdk_assumed"
    assert view["auto_compact_threshold"] == 190000
    assert view["auto_compact_threshold_basis"] == "sdk_assumed"


def test_window_basis_configured_with_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "200000")
    executor, client = run_execute(tmp_path, [usage_message(0, 1000), terminal_result()])

    assert client.usage_calls == 1
    view = dict(executor._live_attempt_view())  # noqa: SLF001
    assert view["context_window_basis"] == "configured"
    assert view["auto_compact_threshold_basis"] == "configured"


def test_failed_context_usage_yields_unavailable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("CLAUDE_CODE_MAX_CONTEXT_TOKENS", raising=False)
    executor, client = run_execute(
        tmp_path,
        [usage_message(0, 1000), terminal_result()],
        usage=RuntimeError("boom"),
    )

    # The attempt still succeeds; only the figures are unavailable.
    assert client.usage_calls == 1
    view = dict(executor._live_attempt_view())  # noqa: SLF001
    assert view["context_window"] is None
    assert view["context_window_basis"] == "unavailable"
    assert isinstance(view["context_window_reason"], str)

    reporter = make_reporter()
    reporter.begin_attempt(claim_seed())
    reporter.publish_attempt_view(MappingProxyType(dict(view)))
    current = reporter.snapshot(StubControl())["current_attempt"]
    assert current is not None
    assert current["context"]["window"]["basis"] == "unavailable"
    assert current["context"]["window"]["value"] is None


def test_compaction_updates_count_last_and_triggers_reading(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("CLAUDE_CODE_MAX_CONTEXT_TOKENS", raising=False)
    executor, client = run_execute(
        tmp_path,
        [
            usage_message(0, 1000),
            SystemMessage(
                subtype="compact_boundary",
                data={"compact_metadata": {"trigger": "auto", "pre_tokens": 150000}},
            ),
            usage_message(1, 2000),
            terminal_result(),
        ],
    )

    # Session start + one compaction.
    assert client.usage_calls == 2
    view = dict(executor._live_attempt_view())  # noqa: SLF001
    assert view["compaction_count"] == 1
    last = dict(view["compaction_last"])
    assert last["trigger"] == "auto"
    assert last["pre_tokens"] == 150000
    # Post tokens filled from the post-compaction context-usage total.
    assert isinstance(last["post_tokens"], int) and last["post_tokens"] > 0
    assert isinstance(last["at"], str) and last["at"]

    reporter = make_reporter()
    reporter.begin_attempt(claim_seed())
    reporter.publish_attempt_view(MappingProxyType(dict(view)))
    current = reporter.snapshot(StubControl())["current_attempt"]
    assert current is not None
    assert current["context"]["compactions"]["count"] == 1
    snapshot_last = current["context"]["compactions"]["last"]
    assert snapshot_last["trigger"] == "auto"
    assert snapshot_last["pre_tokens"] == {"value": 150000, "basis": "measured"}
    assert snapshot_last["post_tokens"]["basis"] == "measured"


def test_subagent_running_flag_and_ordering(tmp_path: Path) -> None:
    executor = ModelExecutor(executor_config(tmp_path))
    executor._reset_ledger_state()  # noqa: SLF001
    executor._ensure_ledger_thread("main", prompt_chars=10)  # noqa: SLF001

    # Three launches: sub-1 finishes, sub-2 and sub-3 stay running.
    for tool_id, sub_type in [("tool-1", "Explore"), ("tool-2", "Plan"), ("tool-3", "Explore")]:
        executor._ensure_ledger_thread(tool_id, subagent_type=sub_type, prompt_chars=10)  # noqa: SLF001
        executor._subagent_info[tool_id] = {"type": sub_type, "running": True}  # noqa: SLF001
        executor._subagent_order.append(tool_id)  # noqa: SLF001
    executor._subagent_info["tool-1"]["running"] = False  # noqa: SLF001

    ordered = executor._telemetry_subagents()  # noqa: SLF001
    assert [entry["id"] for entry in ordered] == ["tool-3", "tool-2", "tool-1"]
    assert [entry["running"] for entry in ordered] == [True, True, False]

    view = dict(executor._live_attempt_view())  # noqa: SLF001
    reporter = make_reporter()
    reporter.begin_attempt(claim_seed())
    reporter.publish_attempt_view(MappingProxyType(dict(view)))
    current = reporter.snapshot(StubControl())["current_attempt"]
    assert current is not None
    listed = current["context"]["subagents"]
    assert [entry["id"] for entry in listed] == ["tool-3", "tool-2", "tool-1"]


def test_subagent_view_capped_at_20(tmp_path: Path) -> None:
    executor = ModelExecutor(executor_config(tmp_path))
    executor._reset_ledger_state()  # noqa: SLF001
    executor._ensure_ledger_thread("main", prompt_chars=10)  # noqa: SLF001
    for index in range(25):
        tool_id = f"tool-{index:02d}"
        executor._ensure_ledger_thread(tool_id, subagent_type="Explore", prompt_chars=10)  # noqa: SLF001
        executor._subagent_info[tool_id] = {"type": "Explore", "running": True}  # noqa: SLF001
        executor._subagent_order.append(tool_id)  # noqa: SLF001

    ordered = executor._telemetry_subagents()  # noqa: SLF001
    assert len(ordered) == 20
