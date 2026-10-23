"""Live attempt figures published during model execution (telemetry).

During model execution the executor publishes an immutable attempt view
after every limit check; the reporter swaps the reference under a lock
held only for the swap, and the listener thread serves the stored copy
without ever reading the token ledger. When model execution ends the
served figures — including ``observed_at`` — stay frozen until
finalisation.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

from claude_agent_sdk import AssistantMessage

from simple_coding_agent.config import RuntimeConfig
from simple_coding_agent.lifecycle import AgentLifecycle, ModelAttemptRunner
from simple_coding_agent.model_execution import ModelExecutionStatus, ModelExecutor
from simple_coding_agent.telemetry import TelemetryReporter
from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.control import ControlStore
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace


EXAMPLES = Path(__file__).parent.parent / "examples" / "telemetry"


def load_example(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text())


def scripted_clock(start: datetime = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)):
    """A clock that advances one second per call (deterministic publications)."""

    state = {"now": start}

    def tick() -> datetime:
        state["now"] = state["now"] + timedelta(seconds=1)
        return state["now"]

    return tick


def make_reporter(**overrides: object) -> TelemetryReporter:
    values: dict[str, object] = {
        "repository": "owner/repo",
        "agent_version": "0.0-test",
        "sdk_version": "0.2.163-test",
        "cli_version": "2.1.286-test",
        "process_started_at": "2026-01-01T00:00:00Z",
        "hard_ceiling": 4_000_000,
        "turns_max": 60,
        "timeout_seconds": 3600,
        "continuation_max": 0,
        "clock": scripted_clock(),
    }
    values.update(overrides)
    return TelemetryReporter(**values)  # type: ignore[arg-type]


def claim_seed(**overrides: object) -> dict:
    seed: dict[str, object] = {
        "attempt_id": "2026-10-14T12:00:00-attempt-0977",
        "issue_number": 977,
        "issue_title": "Example: add a synthetic widget panel",
        "continuation_index": 0,
    }
    seed.update(overrides)
    return seed


def raw_view(**overrides: object) -> dict:
    """One executor-shaped raw live view (pre-basis readings)."""

    view: dict[str, object] = {
        "budget_tokens": 640000.0,
        "measured_tokens": 512000.0,
        "estimated_tokens": 128000.0,
        "hard_ceiling_reached": False,
        "soft_threshold_value": 3200000,
        "soft_threshold_rule": "percentage",
        "soft_threshold_percentage": 0.2,
        "soft_threshold_reserve_turns": 6,
        "soft_threshold_crossed": False,
        "soft_threshold_crossed_at_tokens": None,
        "soft_threshold_crossed_at": None,
        "main_context_basis": "measured",
        "tokens_by_category": {
            "input_tokens": 120000,
            "cache_read_input_tokens": 350000,
            "cache_creation_input_tokens": 0,
            "output_tokens": 42000,
        },
        "turns": 18,
        "cache_hit_rate": 0.86,
        "cache_counted": 22,
        "elapsed_seconds": 2700,
    }
    view.update(overrides)
    return view


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


class FakeClient:
    def __init__(self, options: object, messages: list[object]) -> None:
        self.options = options
        self._messages = messages
        self.queried_prompts: list[str] = []

    async def disconnect(self) -> None:
        self.disconnect_count = getattr(self, "disconnect_count", 0) + 1

    async def connect(self, prompt: str) -> None:
        pass

    async def interrupt(self) -> None:
        pass

    async def stop_task(self, task_id: str) -> None:
        self.stopped_tasks = [*getattr(self, 'stopped_tasks', []), task_id]

    async def query(self, prompt: str) -> None:
        self.queried_prompts.append(prompt)

    async def receive_response(self):
        if self.queried_prompts:
            return
        for message in self._messages:
            yield message


def terminal_result() -> object:
    return SimpleNamespace(
        is_error=False,
        stop_reason="end_turn",
        model_usage={"muse-spark-1.3-contributor": {"input_tokens": 12}},
    )


# ---------------------------------------------------------------------------
# shape conformance with the contract examples
# ---------------------------------------------------------------------------


def test_live_figures_match_the_attempt_in_progress_example_shape() -> None:
    example = load_example("snapshot-attempt-in-progress.json")["current_attempt"]
    reporter = make_reporter()
    reporter.begin_attempt(claim_seed())
    reporter.publish_attempt_view(raw_view())

    current = reporter.snapshot(StubControl())["current_attempt"]

    assert current is not None
    assert set(current["budget"]) == set(example["budget"])
    assert current["budget"]["used"] == {"value": 640000, "basis": "estimated"}
    assert current["budget"]["measured"] == {"value": 512000, "basis": "measured"}
    assert current["budget"]["estimated"] == {"value": 128000, "basis": "estimated"}
    assert current["budget"]["hard_ceiling"] == {"value": 4000000, "basis": "configured"}
    assert current["budget"]["hard_ceiling_reached"] is False
    assert set(current["budget"]["soft_threshold"]) == set(example["budget"]["soft_threshold"])
    assert current["budget"]["soft_threshold"]["basis"] == "configured"
    assert current["budget"]["soft_threshold"]["rule"] == "percentage"
    assert current["budget"]["soft_threshold"]["crossed"] is False
    assert current["budget"]["soft_threshold"]["crossed_at_tokens"] is None
    assert current["budget"]["soft_threshold"]["crossed_at"] is None
    assert set(current["tokens_by_category"]) == set(example["tokens_by_category"])
    for figure in current["tokens_by_category"].values():
        assert figure["basis"] == "measured"
    assert current["tokens_by_category"]["input_tokens"] == {
        "value": 120000,
        "basis": "measured",
    }
    assert set(current["turns"]) == set(example["turns"])
    assert current["turns"]["value"] == 18
    assert current["turns"]["basis"] == "measured"
    assert set(current["cache"]["main_hit_rate"]) == set(example["cache"]["main_hit_rate"])
    assert current["cache"]["main_hit_rate"]["basis"] == "measured"
    assert current["cache"]["main_hit_rate"]["counted"] == 22
    assert current["provider_rate_limit"] == example["provider_rate_limit"]
    assert set(current["elapsed_seconds"]) == set(example["elapsed_seconds"])
    assert current["elapsed_seconds"] == {"value": 2700, "basis": "measured"}
    assert isinstance(current["observed_at"], str) and current["observed_at"]
    assert "usd" not in json.dumps(current).lower()


def test_soft_threshold_crossed_matches_the_handoff_example_shape() -> None:
    example = load_example("snapshot-soft-threshold-handoff.json")["current_attempt"]
    reporter = make_reporter()
    reporter.begin_attempt(claim_seed())
    reporter.publish_attempt_view(
        raw_view(
            budget_tokens=3050000.0,
            measured_tokens=2900000.0,
            estimated_tokens=150000.0,
            soft_threshold_value=2950000,
            soft_threshold_rule="reserve",
            soft_threshold_crossed=True,
            soft_threshold_crossed_at_tokens=2960000,
            soft_threshold_crossed_at="2026-09-30T11:55:00Z",
            main_context_basis="estimated",
            turns=47,
            cache_hit_rate=0.91,
            cache_counted=47,
            elapsed_seconds=10500,
        )
    )

    current = reporter.snapshot(StubControl())["current_attempt"]

    assert current is not None
    assert set(current["budget"]["soft_threshold"]) == set(
        example["budget"]["soft_threshold"]
    )
    soft = current["budget"]["soft_threshold"]
    # Under the reserve rule the threshold inherits the main context's basis.
    assert soft["basis"] == "estimated"
    assert soft["rule"] == "reserve"
    assert soft["crossed"] is True
    assert soft["crossed_at_tokens"] == 2960000
    assert soft["crossed_at"] == "2026-09-30T11:55:00Z"
    assert current["budget"]["used"] == {"value": 3050000, "basis": "estimated"}
    assert current["turns"]["value"] == 47
    assert "usd" not in json.dumps(current).lower()


def test_used_basis_is_measured_when_nothing_is_estimated() -> None:
    reporter = make_reporter()
    reporter.begin_attempt(claim_seed())
    reporter.publish_attempt_view(
        raw_view(
            budget_tokens=512000.0,
            measured_tokens=512000.0,
            estimated_tokens=0.0,
        )
    )

    current = reporter.snapshot(StubControl())["current_attempt"]

    assert current is not None
    assert current["budget"]["used"] == {"value": 512000, "basis": "measured"}


def test_hit_rate_is_unavailable_below_the_minimum_never_zero() -> None:
    reporter = make_reporter()
    reporter.begin_attempt(claim_seed())
    reporter.publish_attempt_view(raw_view(cache_hit_rate=0.9, cache_counted=5))

    current = reporter.snapshot(StubControl())["current_attempt"]

    assert current is not None
    hit_rate = current["cache"]["main_hit_rate"]
    assert hit_rate["value"] is None
    assert hit_rate["basis"] == "unavailable"
    assert isinstance(hit_rate["reason"], str) and hit_rate["reason"]
    assert hit_rate["counted"] == 5


def test_observed_at_and_sequence_advance_with_every_publication() -> None:
    reporter = make_reporter()
    reporter.begin_attempt(claim_seed())
    reporter.publish_attempt_view(raw_view(turns=1))

    first = reporter.snapshot(StubControl())

    reporter.publish_attempt_view(raw_view(turns=2))
    second = reporter.snapshot(StubControl())

    assert first["current_attempt"] is not None
    assert second["current_attempt"] is not None
    assert first["current_attempt"]["observed_at"] != second["current_attempt"]["observed_at"]
    assert second["sequence"] == first["sequence"] + 1
    assert second["current_attempt"]["turns"]["value"] == 2


# ---------------------------------------------------------------------------
# executor publication behaviour
# ---------------------------------------------------------------------------


def test_executor_publishes_an_immutable_view_after_every_limit_check(
    tmp_path: Path,
) -> None:
    published: list[object] = []
    executor = ModelExecutor(
        executor_config(tmp_path),
        attempt_publisher=published.append,
    )

    executor._observe_assistant_message(usage_message(0, 1000))
    executor._observe_assistant_message(usage_message(1, 2000))

    assert len(published) == 2
    for view in published:
        assert isinstance(view, MappingProxyType)
        # The nested category mapping is frozen too, not just the top level.
        assert isinstance(view["tokens_by_category"], MappingProxyType)  # type: ignore[index]
        try:
            view["tokens_by_category"]["input_tokens"] = 0  # type: ignore[index]
        except TypeError:
            pass
        else:
            raise AssertionError("nested tokens_by_category must be immutable")
    first = dict(published[0])  # type: ignore[arg-type]
    second = dict(published[1])  # type: ignore[arg-type]
    # The second publication carries the later readings; the first is untouched.
    assert second["budget_tokens"] > first["budget_tokens"]
    assert second["turns"] == 2
    assert first["turns"] == 1
    assert first["tokens_by_category"]["input_tokens"] == 1000
    assert second["tokens_by_category"]["input_tokens"] == 3000


def test_failing_publish_does_not_affect_the_attempt(tmp_path: Path) -> None:
    events: list[tuple] = []

    def exploding_publisher(view: object) -> None:
        raise RuntimeError("telemetry sink is down")

    executor = ModelExecutor(
        executor_config(tmp_path, max_budget_tokens=1000),
        attempt_publisher=exploding_publisher,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, level)
        ),
    )

    # A raising publisher must not escape the limit check, and the
    # soft-threshold latch still engages on the same check.
    executor._observe_assistant_message(usage_message(0, 1000))

    assert executor._soft_threshold_crossed is True  # noqa: SLF001
    assert ("telemetry_publish_failed", "WARNING") in events

    execution = asyncio.run(
        ModelExecutor(
            executor_config(tmp_path),
            client_factory=lambda options: FakeClient(
                options, [usage_message(0, 1000), terminal_result()]
            ),
            attempt_publisher=exploding_publisher,
        ).execute(issue_body="Fix the widget.", working_directory=tmp_path / "repo")
    )

    assert execution.status is ModelExecutionStatus.SUCCEEDED


def test_views_freeze_after_execution_ends(tmp_path: Path) -> None:
    reporter = make_reporter()
    reporter.begin_attempt(claim_seed())
    publications: list[object] = []

    def counting_publisher(view: object) -> None:
        publications.append(view)
        reporter.publish_attempt_view(view)

    executor = ModelExecutor(
        executor_config(tmp_path),
        client_factory=lambda options: FakeClient(
            options,
            [
                usage_message(0, 1000),
                usage_message(1, 2000),
                terminal_result(),
            ],
        ),
        attempt_publisher=counting_publisher,
    )

    execution = asyncio.run(
        executor.execute(issue_body="Fix the widget.", working_directory=tmp_path / "repo")
    )

    assert execution.status is ModelExecutionStatus.SUCCEEDED
    frozen_count = len(publications)
    assert frozen_count >= 2

    first = reporter.snapshot(StubControl())
    second = reporter.snapshot(StubControl())

    # Serving never publishes: the count is stable and the attempt section
    # — including observed_at — is identical while the top-level sequence
    # still advances per served snapshot.
    assert len(publications) == frozen_count
    assert first["current_attempt"] == second["current_attempt"]
    assert second["sequence"] == first["sequence"] + 1
    assert first["current_attempt"]["turns"] == {"value": 2, "basis": "measured", "max": {"value": 60, "basis": "configured"}}


def test_listener_serves_a_detached_copy_without_ledger_access() -> None:
    reporter = make_reporter()
    assert not hasattr(reporter, "_ledger")
    reporter.begin_attempt(claim_seed())

    source = raw_view()
    reporter.publish_attempt_view(source)
    source["turns"] = 999
    assert isinstance(source["tokens_by_category"], dict)
    source["tokens_by_category"]["input_tokens"] = 999999

    current = reporter.snapshot(StubControl())["current_attempt"]

    assert current is not None
    assert current["turns"]["value"] == 18
    assert current["tokens_by_category"]["input_tokens"]["value"] == 120000


def test_begin_and_finish_clear_the_published_view() -> None:
    reporter = make_reporter()
    reporter.begin_attempt(claim_seed())
    reporter.publish_attempt_view(raw_view())

    assert reporter.snapshot(StubControl())["current_attempt"]["turns"]["value"] == 18

    reporter.begin_attempt(claim_seed())
    reseeded = reporter.snapshot(StubControl())["current_attempt"]
    assert reseeded is not None
    assert reseeded["turns"]["value"] is None
    assert reseeded["budget"]["used"]["basis"] == "unavailable"

    reporter.publish_attempt_view(raw_view())
    assert reporter.snapshot(StubControl())["current_attempt"] is not None
    reporter.finish_attempt()
    assert reporter.snapshot(StubControl())["current_attempt"] is None


# ---------------------------------------------------------------------------
# ledger helpers and lifecycle wiring
# ---------------------------------------------------------------------------


def test_ledger_main_basis_and_measured_count(tmp_path: Path) -> None:
    executor = ModelExecutor(executor_config(tmp_path))

    assert executor._ledger.main_context_basis() == "estimated"  # noqa: SLF001
    assert executor._ledger.main_measured_response_count == 0  # noqa: SLF001

    executor._observe_assistant_message(usage_message(0, 1000))

    assert executor._ledger.main_context_basis() == "measured"  # noqa: SLF001
    assert executor._ledger.main_measured_response_count == 1  # noqa: SLF001


def test_lifecycle_wires_the_reporter_into_the_executor(tmp_path: Path) -> None:
    reporter = make_reporter()
    executor = ModelExecutor(executor_config(tmp_path))
    lifecycle = AgentLifecycle(
        tracker=FakeTracker(),
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=ModelAttemptRunner(
            attempt_state=AttemptStateStore(tmp_path),
            workspace=InMemoryWorkspace(tmp_path / "repo"),
            verifier=None,  # type: ignore[arg-type]
            evaluator=None,  # type: ignore[arg-type]
            model_executor=executor,
        ),
        control_store=ControlStore(tmp_path),
        repository="owner/repo",
        telemetry=reporter,
        sleeper=lambda seconds: None,
    )

    lifecycle._wire_attempt_telemetry_publisher()  # noqa: SLF001

    assert executor._attempt_publisher == reporter.publish_attempt_view  # noqa: SLF001


def test_lifecycle_wiring_leaves_plain_function_runners_alone(tmp_path: Path) -> None:
    lifecycle = AgentLifecycle(
        tracker=FakeTracker(),
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=lambda claim, profile, prepared: (_ for _ in ()).throw(
            AssertionError("must not run")
        ),
        control_store=ControlStore(tmp_path),
        repository="owner/repo",
        telemetry=make_reporter(),
        sleeper=lambda seconds: None,
    )

    lifecycle._wire_attempt_telemetry_publisher()  # noqa: SLF001
