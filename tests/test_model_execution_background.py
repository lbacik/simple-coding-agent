"""Behaviour tests for background subagents keeping model execution open.

The CLI runs ``Agent`` tool calls asynchronously by default: the model can
end its turn (``stop_reason=end_turn``) while subagents are still running,
expecting to be woken later by their completion notifications. The execution
must not treat that first ``ResultMessage`` as terminal while background
tasks launched in this session are still active.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    TextBlock,
)

from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.completion import (
    AttemptOutcome,
    CompletionDecision,
    PublicationPath,
)
from simple_coding_agent.config import RepositoryProfile, RuntimeConfig
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.git_workspace import PreparedAttempt
from simple_coding_agent.lifecycle import ModelAttemptRunner, _review_findings_text
from simple_coding_agent.completion import ReviewEvidence, ReviewFinding
from simple_coding_agent.model_execution import (
    ModelExecution,
    ModelExecutionStatus,
    ModelExecutor,
    OperatorHandoff,
    SkillEvent,
)
from tests.fakes import InMemoryWorkspace


MODEL = "muse-spark-1.3-contributor"


class BackgroundFakeClient:
    """An SDK client double whose stream continues past ``ResultMessage``.

    ``receive_messages()`` is the continuing stream the production boundary
    reads (the real ``receive_response()`` stops after the first result);
    ``receive_response()`` mirrors it for the interrupt-drain path. Once
    ``query()`` has been called the stream serves the follow-up turn, like
    the handoff follow-up fakes in ``test_model_execution.py``.
    """

    def __init__(
        self,
        options: object,
        messages: list[object],
        *,
        followup_messages: list[object] | None = None,
    ) -> None:
        self.options = options
        self._messages = list(messages)
        self._followup_messages = list(followup_messages or [])
        self.prompt: str | None = None
        self.queried_prompts: list[str] = []
        self.interrupted = False

    async def disconnect(self) -> None:
        self.disconnect_count = getattr(self, "disconnect_count", 0) + 1

    async def connect(self, prompt: str) -> None:
        self.prompt = prompt

    async def interrupt(self) -> None:
        self.interrupted = True

    async def stop_task(self, task_id: str) -> None:
        self.stopped_tasks = [*getattr(self, 'stopped_tasks', []), task_id]

    async def query(self, prompt: str) -> None:
        self.queried_prompts.append(prompt)

    async def receive_messages(self):
        if self.queried_prompts:
            for message in self._followup_messages:
                yield message
            return
        for message in self._messages:
            yield message

    async def receive_response(self):
        async for message in self.receive_messages():
            yield message


class HangingBackgroundClient(BackgroundFakeClient):
    """A background stream whose tail hangs until the execution interrupts it."""

    async def receive_messages(self):
        if self.queried_prompts:
            for message in self._followup_messages:
                yield message
            return
        for message in self._messages:
            yield message
        while not self.interrupted:
            await asyncio.sleep(0)

    async def receive_response(self):
        if self.queried_prompts:
            for message in self._followup_messages:
                yield message
            return
        for message in self._messages:
            yield message
        return


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


def started(task_id: str, task_type: str = "local_agent") -> TaskStartedMessage:
    return TaskStartedMessage(
        task_type=task_type,
        subtype="task_started",
        data={},
        task_id=task_id,
        description=f"review agent {task_id}",
        uuid=f"uuid-{task_id}-started",
        session_id="sess-1",
    )


def notified(
    task_id: str, *, status: str = "completed", summary: str = "review done"
) -> TaskNotificationMessage:
    return TaskNotificationMessage(
        subtype="task_notification",
        data={},
        task_id=task_id,
        status=status,  # type: ignore[arg-type]
        output_file=f"/tmp/{task_id}.txt",
        summary=summary,
        uuid=f"uuid-{task_id}-notified",
        session_id="sess-1",
    )


def updated(task_id: str, status: str) -> TaskUpdatedMessage:
    return TaskUpdatedMessage(
        subtype="task_updated",
        data={},
        task_id=task_id,
        patch={"status": status},
        status=status,  # type: ignore[arg-type]
    )


def result(*, input_tokens: int) -> object:
    return SimpleNamespace(
        is_error=False,
        stop_reason="end_turn",
        model_usage={MODEL: {"input_tokens": input_tokens}},
    )


def assistant_text(text: str, *, message_id: str = "r-woken") -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=text)],
        model=MODEL,
        message_id=message_id,
    )


def run_executor(
    tmp_path: Path,
    client: BackgroundFakeClient,
    **overrides: Any,
) -> tuple[ModelExecution, list[tuple[str, str]], BackgroundFakeClient]:
    events: list[tuple[str, str]] = []
    executor = ModelExecutor(
        replace(runtime_config(tmp_path), **overrides),
        client_factory=lambda options: client,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail)
        ),
    )
    execution = asyncio.run(
        executor.execute(issue_body="Fix it.", working_directory=tmp_path)
    )
    return execution, events, client


def test_deferred_result_waits_for_the_notification_then_uses_the_second_result(
    tmp_path: Path,
) -> None:
    client = BackgroundFakeClient(
        None,
        [
            started("A"),
            result(input_tokens=12),
            notified("A", summary="Standards review done"),
            assistant_text("Aggregating the review reports."),
            result(input_tokens=99),
        ],
    )

    execution, events, _ = run_executor(tmp_path, client)

    assert execution.status is ModelExecutionStatus.SUCCEEDED
    # The terminal evidence comes from the second result, not the deferred one.
    assert execution.model_usage == {MODEL: {"input_tokens": 99}}
    # The woken assistant turn is observed and logged.
    assert any(
        name == "model_response" and "Aggregating the review reports." in detail
        for name, detail in events
    )
    # No wake-up follow-up was needed: the CLI woke the model itself.
    assert client.queried_prompts == []


def test_deferral_and_settle_events_carry_the_task_ids(tmp_path: Path) -> None:
    client = BackgroundFakeClient(
        None,
        [
            started("A"),
            started("B"),
            result(input_tokens=12),
            notified("A"),
            notified("B"),
            assistant_text("Both reviews are in."),
            result(input_tokens=99),
        ],
    )

    _, events, _ = run_executor(tmp_path, client)

    deferred = [detail for name, detail in events if name == "background_result_deferred"]
    assert len(deferred) == 1
    assert "pending_task_ids=A,B" in deferred[0]
    settled = [detail for name, detail in events if name == "background_tasks_settled"]
    assert len(settled) == 1
    assert "finished_task_ids=A,B" in settled[0]


def test_task_updated_killed_is_a_terminal_signal_without_any_notification(
    tmp_path: Path,
) -> None:
    client = BackgroundFakeClient(
        None,
        [
            started("A"),
            result(input_tokens=12),
            updated("A", "killed"),
            assistant_text("The review agent was stopped; proceeding anyway."),
            result(input_tokens=99),
        ],
    )

    execution, events, _ = run_executor(tmp_path, client)

    assert execution.status is ModelExecutionStatus.SUCCEEDED
    assert execution.model_usage == {MODEL: {"input_tokens": 99}}
    assert any(name == "background_result_deferred" for name, _ in events)
    assert any(name == "background_tasks_settled" for name, _ in events)
    assert client.queried_prompts == []


def test_non_terminal_task_updated_keeps_waiting(tmp_path: Path) -> None:
    client = BackgroundFakeClient(
        None,
        [
            started("A"),
            result(input_tokens=12),
            updated("A", "running"),
            result(input_tokens=34),
            notified("A"),
        ],
        followup_messages=[result(input_tokens=56)],
    )

    execution, events, _ = run_executor(tmp_path, client)

    # Both in-stream results were deferred; the follow-up turn is terminal.
    assert execution.model_usage == {MODEL: {"input_tokens": 56}}
    deferred = [detail for name, detail in events if name == "background_result_deferred"]
    assert len(deferred) == 2
    assert "deferred_results=2" in deferred[1]
    assert len(client.queried_prompts) == 1


def test_first_result_stays_terminal_without_background_tasks(tmp_path: Path) -> None:
    client = BackgroundFakeClient(None, [result(input_tokens=12)])

    execution, events, _ = run_executor(tmp_path, client)

    assert execution.status is ModelExecutionStatus.SUCCEEDED
    assert execution.model_usage == {MODEL: {"input_tokens": 12}}
    assert not [name for name, _ in events if name.startswith("background_")]
    assert client.queried_prompts == []


def test_wake_up_followup_fires_once_when_no_new_turn_follows(tmp_path: Path) -> None:
    client = BackgroundFakeClient(
        None,
        [
            started("A"),
            result(input_tokens=12),
            notified("A", summary="Standards review done"),
        ],
        followup_messages=[result(input_tokens=77)],
    )

    execution, events, _ = run_executor(tmp_path, client)

    assert len(client.queried_prompts) == 1
    prompt = client.queried_prompts[0]
    assert "Standards review done" in prompt
    assert "/tmp/A.txt" in prompt
    assert any(name == "background_tasks_followup" for name, _ in events)
    # The result of the woken turn is terminal.
    assert execution.status is ModelExecutionStatus.SUCCEEDED
    assert execution.model_usage == {MODEL: {"input_tokens": 77}}


def test_timeout_while_a_task_is_pending_is_not_success(tmp_path: Path) -> None:
    client = HangingBackgroundClient(None, [started("A")])

    execution, _, _ = run_executor(tmp_path, client, model_timeout=0.2)

    assert execution.status is not ModelExecutionStatus.SUCCEEDED
    assert execution.status is ModelExecutionStatus.INFRASTRUCTURE_ERROR
    assert client.interrupted is True


def test_token_hard_ceiling_interrupts_while_waiting_on_background_tasks(
    tmp_path: Path,
) -> None:
    client = BackgroundFakeClient(
        None,
        [
            started("A"),
            result(input_tokens=12),
            AssistantMessage(
                content=[],
                model=MODEL,
                message_id="r-big",
                usage={"input_tokens": 1_100_000},
            ),
        ],
    )

    execution, events, _ = run_executor(
        tmp_path, client, max_budget_tokens=1_000_000
    )

    assert execution.status is ModelExecutionStatus.MODEL_LIMIT_REACHED
    assert execution.token_hard_ceiling_reached is True
    assert client.interrupted is True
    assert client.queried_prompts == []
    assert any(name == "background_result_deferred" for name, _ in events)
    assert any(name == "token_hard_ceiling_reached" for name, _ in events)


def test_operator_handoff_expiry_interrupts_while_waiting_on_background_tasks(
    tmp_path: Path,
) -> None:
    client = HangingBackgroundClient(
        None,
        [
            started("A"),
            result(input_tokens=12),
        ],
    )
    polls = 0

    def provider() -> OperatorHandoff | None:
        nonlocal polls
        polls += 1
        # Let the first two loop iterations consume the start and defer the
        # result; only then does the expired request surface.
        if polls < 5:
            return None
        return OperatorHandoff(
            request_id="req-1",
            accepted_at=datetime.now(UTC) - timedelta(seconds=500),
        )

    events: list[tuple[str, str]] = []
    executor = ModelExecutor(
        runtime_config(tmp_path),
        client_factory=lambda options: client,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail)
        ),
    )
    executor.set_operator_handoff_provider(provider)
    execution = asyncio.run(
        executor.execute(issue_body="Fix it.", working_directory=tmp_path)
    )

    assert execution.status is ModelExecutionStatus.OPERATOR_HANDOFF_EXPIRED
    assert any(name == "background_result_deferred" for name, _ in events)


# --- Honest "Review findings" line (issue #177) --------------------------------


class FakeModelExecutor:
    def __init__(
        self,
        *,
        status: ModelExecutionStatus = ModelExecutionStatus.MODEL_LIMIT_REACHED,
        skill_events: tuple[SkillEvent, ...] = (),
    ) -> None:
        self._status = status
        self._skill_events = skill_events

    async def execute(
        self, *, issue_body: str, working_directory: Path, issue_number=None, archive=None
    ) -> ModelExecution:
        return ModelExecution(
            status=self._status,
            explanation="stub",
            stop_reason="end_turn",
            model_usage=None,
            observed_models=(),
            skill_events=self._skill_events,
        )


class FakeVerifier:
    def prepare(self, profile: object, working_directory: Path, **kwargs: object):
        setup = SimpleNamespace(succeeded=True, commands=())
        baseline = SimpleNamespace(succeeded=True, commands=(), exit_code=0)
        return SimpleNamespace(setup=setup, baseline=baseline)

    def mark_review_complete(self, **kwargs: object) -> None:
        return None

    def final_check(self, profile: object, working_directory: Path):
        return SimpleNamespace(succeeded=True, commands=(), exit_code=0)


class FakeEvaluator:
    def evaluate(self, **kwargs: object) -> CompletionDecision:
        return CompletionDecision(AttemptOutcome.INCOMPLETE, False, PublicationPath.NONE, ("stub",))


def attempt_claim() -> Claim:
    return Claim(
        TrackerIssue(
            id="issue-24",
            number=24,
            title="Implement lifecycle",
            body="Fix the parser.",
            created_at=datetime(2026, 9, 20, tzinfo=UTC),
            state="OPEN",
            labels=frozenset({"ready-for-agent"}),
            assignee_logins=(),
            blocked_by=0,
            author_login="reporter",
        ),
        Assignment("issue-24", "agent-id"),
    )


def attempt_profile() -> RepositoryProfile:
    return RepositoryProfile(
        setup=(), check=("pytest",), base_branch="main", timeout=60, setup_timeout=60, env={}
    )


def run_attempt(tmp_path: Path, executor: FakeModelExecutor):
    attempt_state = AttemptStateStore(tmp_path)
    attempt_state.start(issue_number=24, branch="agent/issue-24")
    attempt_state.transition(AttemptPhase.SETUP)
    runner = ModelAttemptRunner(
        attempt_state=attempt_state,
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        verifier=FakeVerifier(),  # type: ignore[arg-type]
        evaluator=FakeEvaluator(),  # type: ignore[arg-type]
        model_executor=executor,  # type: ignore[arg-type]
    )
    return runner(
        attempt_claim(),
        attempt_profile(),
        PreparedAttempt(branch="agent/issue-24", base_revision="base-revision-0"),
    )


def test_review_findings_not_run_without_code_review(tmp_path: Path) -> None:
    evidence = run_attempt(tmp_path, FakeModelExecutor(skill_events=()))

    assert evidence.review_cycles == 0
    assert evidence.review_findings == "not run"


def test_review_findings_not_collected_after_code_review(tmp_path: Path) -> None:
    executor = FakeModelExecutor(
        skill_events=(
            SkillEvent(
                phase="PreToolUse",
                name="code-review",
                agent_id=None,
                timestamp="2026-10-11T00:00:00Z",
            ),
        )
    )

    evidence = run_attempt(tmp_path, executor)

    assert evidence.review_cycles == 0
    assert evidence.review_findings == "not collected"


def test_review_findings_join_when_findings_exist() -> None:
    assert (
        _review_findings_text(
            ReviewEvidence((ReviewFinding("must-fix", "data race"),), 1), 2
        )
        == "data race"
    )
    assert _review_findings_text(ReviewEvidence((), 1), 2) == "not collected"
    assert _review_findings_text(ReviewEvidence((), 0), 0) == "not run"


class RaisingStopClient(BackgroundFakeClient):
    async def stop_task(self, task_id: str) -> None:
        await super().stop_task(task_id)
        raise RuntimeError("stop failed")


class HangingRaisingStopClient(RaisingStopClient, HangingBackgroundClient):
    pass


def test_non_deferring_task_does_not_defer_and_is_stopped_on_exit(tmp_path: Path) -> None:
    client = BackgroundFakeClient(
        None, [started("B", "local_bash"), result(input_tokens=12)]
    )
    execution, events, client = run_executor(tmp_path, client)

    assert not any(name == "background_result_deferred" for name, _ in events)
    assert client.stopped_tasks == ["B"]
    stopped = [detail for name, detail in events if name == "background_tasks_stopped"]
    assert len(stopped) == 1 and "B" in stopped[0]


def test_grace_expiry_stops_deferring_tasks_and_uses_deferred_result(
    tmp_path: Path,
) -> None:
    client = HangingBackgroundClient(
        None, [started("A"), result(input_tokens=12)]
    )
    execution, events, client = run_executor(
        tmp_path, client, background_task_grace_seconds=1
    )

    assert client.stopped_tasks == ["A"]
    stopped = [detail for name, detail in events if name == "background_tasks_stopped"]
    assert len(stopped) == 1 and "A" in stopped[0] and "waited_seconds=" in stopped[0]
    assert not client.interrupted


def test_failing_stop_task_is_logged_and_does_not_fail_the_attempt(
    tmp_path: Path,
) -> None:
    client = HangingRaisingStopClient(None, [started("A"), result(input_tokens=12)])
    execution, events, client = run_executor(
        tmp_path, client, background_task_grace_seconds=1
    )

    assert client.stopped_tasks == ["A"]
    assert any(name == "background_task_stop_failed" for name, _ in events)
    assert execution.status is ModelExecutionStatus.SUCCEEDED
