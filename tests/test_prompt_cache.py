"""Per-category token usage and prompt-cache effectiveness.

Covers the ``TokenLedger`` cache readings (per-category totals, hit rate,
miss tokens, compaction handling) and the executor side (``limits_checked``
detail, the one-time ``prompt_cache_ineffective`` warning, and the
``token_budget`` additions). The replay tests pin the acceptance figures for
``issue109_0758`` and ``issue109_0913`` from
``tests/fixtures/token_ledger_replay.json``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from claude_agent_sdk import AssistantMessage, SystemMessage, UserMessage

from simple_coding_agent.config import RuntimeConfig
from simple_coding_agent.model_execution import (
    PROMPT_CACHE_HIT_RATE_THRESHOLD,
    PROMPT_CACHE_MIN_MEASURED_RESPONSES,
    ModelExecutor,
)
from simple_coding_agent.token_ledger import TokenLedger, usage_by_category

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "token_ledger_replay.json"
RECORDINGS = json.loads(FIXTURE_PATH.read_text())["recordings"]


def _usage(input_tokens: int, cache_read: int, cache_creation: int, output: int) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_creation,
        "output_tokens": output,
    }


def _measured_ledger() -> TokenLedger:
    ledger = TokenLedger()
    ledger.start_thread("main", kind="main", prompt_chars=0)
    return ledger


def _respond(ledger: TokenLedger, response_id: str, usage: dict[str, int]) -> None:
    ledger.observe_response("main", response_id, visible_chars=0)
    ledger.observe_usage("main", response_id, usage)


# --- usage_by_category -------------------------------------------------------


def test_usage_by_category_reads_both_naming_shapes() -> None:
    assert usage_by_category(_usage(1, 2, 3, 4)) == {
        "input_tokens": 1,
        "cache_read_input_tokens": 2,
        "cache_creation_input_tokens": 3,
        "output_tokens": 4,
    }
    assert usage_by_category(
        {
            "inputTokens": 1,
            "cacheReadInputTokens": 2,
            "cacheCreationInputTokens": 3,
            "outputTokens": 4,
            "thinkingTokens": 9,
        }
    ) == {
        "input_tokens": 1,
        "cache_read_input_tokens": 2,
        "cache_creation_input_tokens": 3,
        "output_tokens": 4,
    }


# --- measured_by_category ----------------------------------------------------


def test_measured_by_category_sums_all_threads() -> None:
    ledger = _measured_ledger()
    ledger.start_thread("toolu-1", kind="subagent", prompt_chars=0, subagent_type="Explore")
    _respond(ledger, "r1", _usage(100, 200, 300, 40))
    ledger.observe_response("toolu-1", "s1", visible_chars=0)
    ledger.observe_usage("toolu-1", "s1", _usage(10, 20, 30, 4))

    assert ledger.measured_by_category == {
        "input_tokens": 110,
        "cache_read_input_tokens": 220,
        "cache_creation_input_tokens": 330,
        "output_tokens": 44,
    }


def test_measured_by_category_ignores_estimates() -> None:
    ledger = _measured_ledger()
    ledger.observe_response("main", "r1", visible_chars=300)

    assert ledger.measured_by_category == {
        "input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "output_tokens": 0,
    }


# --- hit rate ----------------------------------------------------------------


def test_main_hit_rate_is_none_without_measured_responses() -> None:
    ledger = _measured_ledger()
    ledger.observe_response("main", "r1", visible_chars=100)

    assert ledger.main_hit_rate() is None


def test_main_hit_rate_is_none_when_total_input_is_zero() -> None:
    ledger = _measured_ledger()
    _respond(ledger, "r1", _usage(0, 0, 0, 50))

    assert ledger.main_hit_rate() is None


def test_main_hit_rate_divides_read_by_total_input() -> None:
    ledger = _measured_ledger()
    _respond(ledger, "r1", _usage(100, 300, 0, 10))
    _respond(ledger, "r2", _usage(100, 100, 0, 10))

    assert ledger.main_hit_rate() == pytest.approx(400 / 600)


def test_main_hit_rate_limit_covers_only_the_first_k_measured() -> None:
    ledger = _measured_ledger()
    _respond(ledger, "r1", _usage(0, 100, 0, 0))
    _respond(ledger, "r2", _usage(100, 0, 0, 0))

    assert ledger.main_hit_rate(limit=1) == pytest.approx(1.0)
    assert ledger.main_hit_rate() == pytest.approx(0.5)


# --- miss tokens --------------------------------------------------------------


def test_first_measured_main_response_is_excluded_from_miss() -> None:
    ledger = _measured_ledger()
    _respond(ledger, "r1", _usage(100, 50, 0, 10))

    stats = ledger.main_cache_stats()

    assert (stats.miss_tokens, stats.counted, stats.excluded) == (0, 0, 1)
    metrics = ledger.response_cache_metrics("main", "r1")
    assert metrics is not None
    assert metrics.miss_tokens is None


def test_miss_is_previous_total_input_minus_read() -> None:
    ledger = _measured_ledger()
    _respond(ledger, "r1", _usage(100, 300, 50, 10))  # total input 450
    _respond(ledger, "r2", _usage(400, 100, 0, 10))  # miss 350

    stats = ledger.main_cache_stats()

    assert (stats.miss_tokens, stats.counted, stats.excluded) == (350, 1, 1)
    metrics = ledger.response_cache_metrics("main", "r2")
    assert metrics is not None
    assert metrics.miss_tokens == 350
    assert metrics.hit_rate == pytest.approx(100 / 500)


def test_full_cache_read_has_no_miss() -> None:
    ledger = _measured_ledger()
    _respond(ledger, "r1", _usage(100, 300, 0, 10))  # total input 400
    _respond(ledger, "r2", _usage(0, 500, 0, 10))  # miss max(0, 400-500) = 0

    stats = ledger.main_cache_stats()

    assert (stats.miss_tokens, stats.counted, stats.excluded) == (0, 1, 1)


def test_response_after_estimated_predecessor_is_excluded() -> None:
    ledger = _measured_ledger()
    _respond(ledger, "r1", _usage(100, 300, 0, 10))
    ledger.observe_response("main", "r2", visible_chars=100)  # estimated
    _respond(ledger, "r3", _usage(100, 300, 0, 10))

    stats = ledger.main_cache_stats()

    assert (stats.miss_tokens, stats.counted, stats.excluded) == (0, 0, 2)
    metrics = ledger.response_cache_metrics("main", "r3")
    assert metrics is not None
    assert metrics.miss_tokens is None
    # Exclusion never affects the hit rate.
    assert metrics.hit_rate == pytest.approx(300 / 400)


def test_response_after_compaction_is_excluded() -> None:
    ledger = _measured_ledger()
    _respond(ledger, "r1", _usage(100, 300, 0, 10))
    ledger.notify_compaction("main")
    _respond(ledger, "r2", _usage(100, 0, 0, 10))

    stats = ledger.main_cache_stats()

    assert (stats.miss_tokens, stats.counted, stats.excluded) == (0, 0, 2)
    assert ledger.response_cache_metrics("main", "r2") is not None
    assert ledger.response_cache_metrics("main", "r2").miss_tokens is None  # type: ignore[union-attr]


def test_compaction_only_excludes_the_next_response() -> None:
    ledger = _measured_ledger()
    _respond(ledger, "r1", _usage(100, 300, 0, 10))  # total input 400
    ledger.notify_compaction("main")
    _respond(ledger, "r2", _usage(100, 0, 0, 10))  # excluded
    _respond(ledger, "r3", _usage(100, 0, 0, 10))  # miss 100 - 0

    stats = ledger.main_cache_stats()

    assert (stats.miss_tokens, stats.counted, stats.excluded) == (100, 1, 2)


def test_response_cache_metrics_is_none_without_usage() -> None:
    ledger = _measured_ledger()
    ledger.observe_response("main", "r1", visible_chars=100)

    assert ledger.response_cache_metrics("main", "r1") is None
    assert ledger.response_cache_metrics("main", "unknown") is None
    assert ledger.response_cache_metrics("unknown", "r1") is None


def test_subagent_measured_response_has_no_miss() -> None:
    ledger = _measured_ledger()
    ledger.start_thread("toolu-1", kind="subagent", prompt_chars=0, subagent_type="Explore")
    ledger.observe_response("toolu-1", "s1", visible_chars=0)
    ledger.observe_usage("toolu-1", "s1", _usage(100, 300, 0, 10))

    metrics = ledger.response_cache_metrics("toolu-1", "s1")

    assert metrics is not None
    assert metrics.miss_tokens is None
    assert metrics.hit_rate == pytest.approx(300 / 400)


# --- replay acceptance figures ------------------------------------------------


def _replay_live(recording: list) -> TokenLedger:
    """Feed recording rows through a ledger with main-loop usage reported."""

    ledger = TokenLedger()
    for row in recording:
        if row[0] == "s":
            _, thread, kind, prompt_chars = row
            ledger.start_thread(thread, kind=kind, prompt_chars=prompt_chars)
        elif row[0] == "r":
            _, thread, response_id, visible_chars, v_in, v_read, v_creation, v_out = row
            ledger.observe_response(thread, response_id, visible_chars)
            ledger.observe_usage(
                thread,
                response_id,
                _usage(v_in, v_read, v_creation, v_out),
            )
        elif row[0] == "i":
            _, thread, chars = row
            ledger.observe_input(thread, chars)
    return ledger


def test_0758_replay_cache_figures() -> None:
    ledger = _replay_live(RECORDINGS["issue109_0758"])

    assert ledger.measured_by_category == {
        "input_tokens": 3569248,
        "cache_read_input_tokens": 2826295,
        "cache_creation_input_tokens": 0,
        "output_tokens": 36825,
    }
    assert ledger.main_hit_rate() == pytest.approx(0.4419, abs=1e-4)
    stats = ledger.main_cache_stats()
    assert stats.miss_tokens == 3441919
    assert (stats.counted, stats.excluded) == (76, 1)

    first_hit = ledger.main_hit_rate(limit=20)
    assert first_hit == pytest.approx(0.1171, abs=1e-4)

    hot = ledger.response_cache_metrics("main", "f9764c8e")
    assert hot is not None
    assert hot.hit_rate == pytest.approx(0.7723, abs=1e-4)
    assert hot.miss_tokens == 9918

    cold = ledger.response_cache_metrics("main", "ced645cb")
    assert cold is not None
    assert cold.hit_rate == 0
    assert cold.miss_tokens == 47714


def test_0913_replay_cache_figures() -> None:
    ledger = _replay_live(RECORDINGS["issue109_0913"])

    assert ledger.measured_by_category == {
        "input_tokens": 3763348,
        "cache_read_input_tokens": 1559772,
        "cache_creation_input_tokens": 0,
        "output_tokens": 23834,
    }
    assert ledger.main_hit_rate() == pytest.approx(0.2930, abs=1e-4)
    stats = ledger.main_cache_stats()
    assert stats.miss_tokens == 3662466
    assert (stats.counted, stats.excluded) == (76, 1)
    assert ledger.main_hit_rate(limit=20) == pytest.approx(0.2830, abs=1e-4)


# --- executor: limits_checked -------------------------------------------------


class FakeClient:
    def __init__(self, options: object, messages: list[object]) -> None:
        self.options = options
        self._messages = messages
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


def _ok_result(model_usage: dict | None = None) -> object:
    return SimpleNamespace(
        is_error=False,
        stop_reason="end_turn",
        model_usage=(
            model_usage
            if model_usage is not None
            else {"muse-spark-1.3-contributor": {"input_tokens": 12}}
        ),
    )


def _execute_with(
    tmp_path: Path, messages: list[object], **overrides: object
) -> tuple[object, list[tuple[str, str, str]], ModelExecutor]:
    events: list[tuple[str, str, str]] = []
    executor = ModelExecutor(
        replace(_runtime_config(tmp_path), **overrides),
        client_factory=lambda options: FakeClient(options, messages),
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level)
        ),
    )
    execution = asyncio.run(executor.execute(issue_body="Fix it.", working_directory=tmp_path))
    return execution, events, executor


def _limits_details(events: list[tuple[str, str, str]]) -> list[str]:
    return [detail for name, detail, _ in events if name == "limits_checked"]


def test_limits_checked_appends_cache_fields_for_a_measured_response(tmp_path: Path) -> None:
    _, events, _ = _execute_with(
        tmp_path,
        [
            AssistantMessage(
                content=[],
                model="muse-spark-1.3-contributor",
                message_id="r1",
                usage={"input_tokens": 1000, "cache_read_input_tokens": 3000, "output_tokens": 100},
            ),
            AssistantMessage(
                content=[],
                model="muse-spark-1.3-contributor",
                message_id="r2",
            ),
            _ok_result(),
        ],
    )

    details = _limits_details(events)
    assert len(details) == 1
    assert "; input_tokens=1000" in details[0]
    assert "; cache_read_tokens=3000" in details[0]
    assert "; cache_creation_tokens=0" in details[0]
    assert "; cache_hit_rate=0.75" in details[0]
    # The first measured main-thread response is always excluded from miss accounting.
    assert "; cache_miss_tokens=excluded" in details[0]


def test_limits_checked_reports_miss_for_a_second_measured_response(tmp_path: Path) -> None:
    _, events, _ = _execute_with(
        tmp_path,
        [
            AssistantMessage(
                content=[],
                model="muse-spark-1.3-contributor",
                message_id="r1",
                usage={"input_tokens": 1000, "cache_read_input_tokens": 3000, "output_tokens": 100},
            ),
            AssistantMessage(
                content=[],
                model="muse-spark-1.3-contributor",
                message_id="r2",
                usage={"input_tokens": 3900, "cache_read_input_tokens": 100, "output_tokens": 50},
            ),
            AssistantMessage(
                content=[],
                model="muse-spark-1.3-contributor",
                message_id="r3",
            ),
            _ok_result(),
        ],
    )

    details = _limits_details(events)
    assert len(details) == 2
    assert "; cache_miss_tokens=excluded" in details[0]
    # Expected prefix 4000, read 100.
    assert "; cache_miss_tokens=3900" in details[1]
    assert "; cache_hit_rate=0.025" in details[1]


def test_limits_checked_is_unchanged_without_reported_usage(tmp_path: Path) -> None:
    _, events, _ = _execute_with(
        tmp_path,
        [
            AssistantMessage(
                content=[],
                model="muse-spark-1.3-contributor",
                message_id="r1",
                usage={"input_tokens": 0, "output_tokens": 0},
            ),
            UserMessage(content="ok"),
            _ok_result(),
        ],
    )

    details = _limits_details(events)
    assert len(details) == 1
    assert "cache_" not in details[0]
    assert "input_tokens" not in details[0]
    assert "miss" not in details[0]


def test_limits_checked_after_compaction_logs_miss_excluded(tmp_path: Path) -> None:
    _, events, _ = _execute_with(
        tmp_path,
        [
            AssistantMessage(
                content=[],
                model="muse-spark-1.3-contributor",
                message_id="r1",
                usage={"input_tokens": 1000, "output_tokens": 100},
            ),
            SystemMessage(
                subtype="compact_boundary",
                data={"compact_metadata": {"trigger": "auto", "pre_tokens": 150000}},
            ),
            AssistantMessage(
                content=[],
                model="muse-spark-1.3-contributor",
                message_id="r2",
                usage={"input_tokens": 500, "output_tokens": 50},
            ),
            AssistantMessage(
                content=[],
                model="muse-spark-1.3-contributor",
                message_id="r3",
            ),
            _ok_result(),
        ],
    )

    details = _limits_details(events)
    assert len(details) == 2
    assert "; cache_miss_tokens=excluded" in details[0]
    assert "; cache_miss_tokens=excluded" in details[1]
    assert [name for name, _, _ in events if name == "context_compacted"] != []


# --- executor: prompt_cache_ineffective ----------------------------------------


def _low_hit_messages(count: int, start: int = 0) -> list[object]:
    return [
        AssistantMessage(
            content=[],
            model="muse-spark-1.3-contributor",
            message_id=f"r{index}",
            usage={"input_tokens": 10_000, "output_tokens": 100},
        )
        for index in range(start, start + count)
    ]


def test_cache_warning_fires_once_at_k_low_hit_responses(tmp_path: Path) -> None:
    assert PROMPT_CACHE_MIN_MEASURED_RESPONSES == 20
    assert PROMPT_CACHE_HIT_RATE_THRESHOLD == 0.5

    _, events, _ = _execute_with(
        tmp_path,
        [
            *_low_hit_messages(21),
            AssistantMessage(content=[], model="muse-spark-1.3-contributor", message_id="r21b"),
            _ok_result(),
        ],
        max_budget_tokens=1_000_000_000,
    )

    warnings = [
        (detail, level)
        for name, detail, level in events
        if name == "prompt_cache_ineffective"
    ]
    assert len(warnings) == 1
    detail, level = warnings[0]
    assert level == "WARNING"
    assert "measured_responses=20" in detail
    assert "main_hit_rate=0.0" in detail
    assert "hit_rate_threshold=0.5" in detail
    assert "input_tokens=200000" in detail
    assert "cache_read_tokens=0" in detail
    assert "cache_creation_tokens=0" in detail


def test_cache_warning_stays_silent_above_the_threshold(tmp_path: Path) -> None:
    messages: list[object] = [
        AssistantMessage(
            content=[],
            model="muse-spark-1.3-contributor",
            message_id=f"r{index}",
            usage={
                "input_tokens": 1_000,
                "cache_read_input_tokens": 9_000,
                "output_tokens": 100,
            },
        )
        for index in range(21)
    ]
    messages.append(
        AssistantMessage(content=[], model="muse-spark-1.3-contributor", message_id="tail")
    )
    messages.append(_ok_result())

    _, events, _ = _execute_with(tmp_path, messages, max_budget_tokens=1_000_000_000)

    assert [name for name, _, _ in events if name == "prompt_cache_ineffective"] == []


def test_cache_warning_stays_silent_with_fewer_than_k_responses(tmp_path: Path) -> None:
    _, events, _ = _execute_with(
        tmp_path,
        [
            *_low_hit_messages(5),
            AssistantMessage(content=[], model="muse-spark-1.3-contributor", message_id="tail"),
            _ok_result(),
        ],
        max_budget_tokens=1_000_000_000,
    )

    assert [name for name, _, _ in events if name == "prompt_cache_ineffective"] == []


# --- executor: token_budget additions ------------------------------------------


def test_token_budget_carries_per_category_and_cache_fields(tmp_path: Path) -> None:
    execution, events, _ = _execute_with(
        tmp_path,
        [
            AssistantMessage(
                content=[],
                model="muse-spark-1.3-contributor",
                message_id="r1",
                usage={
                    "input_tokens": 1000,
                    "cache_read_input_tokens": 3000,
                    "cache_creation_input_tokens": 500,
                    "output_tokens": 200,
                },
            ),
            _ok_result(
                {
                    "muse-spark-1.3-contributor": {
                        "input_tokens": 1000,
                        "cache_read_input_tokens": 3000,
                        "cache_creation_input_tokens": 500,
                        "output_tokens": 200,
                    }
                }
            ),
        ],
    )

    budget = execution.token_budget
    assert budget["measured_by_category"] == {
        "input": 1000,
        "cache_read": 3000,
        "cache_creation": 500,
        "output": 200,
    }
    assert budget["actual_by_category"] == {
        "input": 1000,
        "cache_read": 3000,
        "cache_creation": 500,
        "output": 200,
    }
    assert budget["prompt_cache"] == {
        "main_hit_rate": pytest.approx(3000 / 4500, abs=1e-4),
        "all_threads_hit_rate": pytest.approx(3000 / 4500, abs=1e-4),
        "main_cache_miss_tokens": 0,
        "main_miss_responses_counted": 0,
        "main_miss_responses_excluded": 1,
    }
    reconciled = [detail for name, detail, _ in events if name == "token_budget_reconciled"]
    assert len(reconciled) == 1
    assert '"main_hit_rate"' in reconciled[0]


def test_token_budget_without_a_result_message_nulls_actual_fields(tmp_path: Path) -> None:
    execution, _, _ = _execute_with(
        tmp_path,
        [
            AssistantMessage(
                content=[],
                model="muse-spark-1.3-contributor",
                message_id="r1",
                usage={"input_tokens": 1000, "output_tokens": 100},
            ),
        ],
    )

    budget = execution.token_budget
    assert budget["actual_tokens"] is None
    assert budget["actual_by_category"] is None
    assert budget["prompt_cache"]["all_threads_hit_rate"] is None
    assert budget["prompt_cache"]["main_hit_rate"] == pytest.approx(0.0)
    assert budget["measured_by_category"] == {
        "input": 1000,
        "cache_read": 0,
        "cache_creation": 0,
        "output": 100,
    }


# --- executor: full-recording replay -------------------------------------------


def _recording_messages(key: str) -> list[object]:
    messages: list[object] = []
    for row in RECORDINGS[key]:
        if row[0] == "r":
            _, thread, response_id, _visible, v_in, v_read, v_creation, v_out = row
            kwargs: dict[str, object] = {"message_id": response_id}
            if thread != "main":
                kwargs["parent_tool_use_id"] = thread
            messages.append(
                AssistantMessage(
                    content=[],
                    model="muse-spark-1.3-contributor",
                    usage=_usage(v_in, v_read, v_creation, v_out),
                    **kwargs,  # type: ignore[arg-type]
                )
            )
        elif row[0] == "i":
            messages.append(UserMessage(content="x" * 64))
    totals = {"input_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": 0}
    for row in RECORDINGS[key]:
        if row[0] == "r":
            for name, index in (
                ("input_tokens", 4),
                ("cache_read_input_tokens", 5),
                ("cache_creation_input_tokens", 6),
                ("output_tokens", 7),
            ):
                totals[name] += row[index]
    messages.append(_ok_result({"muse-spark-1.3-contributor": totals}))
    return messages


@pytest.mark.parametrize(
    ("key", "measured", "hit_rate", "miss"),
    [
        (
            "issue109_0758",
            {"input": 3569248, "cache_read": 2826295, "cache_creation": 0, "output": 36825},
            0.4419,
            3441919,
        ),
        (
            "issue109_0913",
            {"input": 3763348, "cache_read": 1559772, "cache_creation": 0, "output": 23834},
            0.2930,
            3662466,
        ),
    ],
)
def test_full_recording_replay_warns_once_and_reports_cache_figures(
    tmp_path: Path, key: str, measured: dict[str, int], hit_rate: float, miss: int
) -> None:
    execution, events, _ = _execute_with(
        tmp_path, _recording_messages(key), max_budget_tokens=1_000_000_000
    )

    budget = execution.token_budget
    assert budget["measured_by_category"] == measured
    assert budget["prompt_cache"]["main_hit_rate"] == pytest.approx(hit_rate, abs=1e-4)
    assert budget["prompt_cache"]["main_cache_miss_tokens"] == miss
    assert budget["prompt_cache"]["main_miss_responses_counted"] == 76
    assert budget["prompt_cache"]["main_miss_responses_excluded"] == 1
    assert budget["actual_by_category"] == measured
    assert budget["prompt_cache"]["all_threads_hit_rate"] == pytest.approx(hit_rate, abs=1e-4)

    warnings = [name for name, _, _ in events if name == "prompt_cache_ineffective"]
    assert len(warnings) == 1
