"""Behaviour tests for the model attempt runner's starting-prompt assembly."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.completion import (
    AttemptOutcome,
    CompletionDecision,
    PublicationPath,
)
from simple_coding_agent.config import RepositoryProfile
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.lifecycle import ModelAttemptRunner
from simple_coding_agent.model_execution import ModelExecution, ModelExecutionStatus


def test_prompt_appends_trusted_comments_in_order(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    comments = ("please also handle timeouts", "keep the old retry behaviour")
    runner = build_runner(tmp_path, model_executor=executor, issue_comments=lambda issue: comments)

    runner(claim(issue_body="Fix the parser."), profile(), FakePrepared())

    assert executor.captured_prompt == (
        "Fix the parser.\n\nplease also handle timeouts\n\nkeep the old retry behaviour"
    )


def test_prompt_degrades_to_the_bare_issue_body_with_no_comments_or_continuation(
    tmp_path: Path,
) -> None:
    executor = FakeModelExecutor()
    runner = build_runner(tmp_path, model_executor=executor)

    runner(claim(issue_body="Fix the parser."), profile(), FakePrepared())

    assert executor.captured_prompt == "Fix the parser."


def test_continuation_adds_only_a_pointer_to_the_handoff_note(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    workspace = FakeWorkspace(commits=("Previous attempt work",))
    runner = build_runner(tmp_path, model_executor=executor, workspace=workspace, issue_number=24)

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    assert executor.captured_prompt == (
        "Fix the parser.\n\n"
        "This is a continued attempt; if it exists, the handoff note from the "
        "previous attempt is at .agent/handoff/24.md."
    )


def test_missing_continuation_branch_skips_the_model_when_label_present(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    runner = build_runner(tmp_path, model_executor=executor)

    evidence = runner(
        claim(issue_number=24, labels=frozenset({"ready-for-agent", "round-finished"})),
        profile(),
        FakePrepared(restored_from_remote=False),
    )

    assert executor.captured_prompt is None
    assert evidence.decision.outcome is AttemptOutcome.INCOMPLETE
    assert not evidence.decision.publication_eligible
    assert "agent/issue-24" in evidence.details
    assert "not found on origin" in evidence.details


def test_missing_continuation_branch_skips_the_model_with_a_prior_handoff_comment(
    tmp_path: Path,
) -> None:
    executor = FakeModelExecutor()
    comments = ("## Agent Attempt Result: incomplete\n<!-- agent-attempt: earlier -->",)
    runner = build_runner(tmp_path, model_executor=executor, issue_comments=lambda issue: comments)

    evidence = runner(claim(issue_number=24), profile(), FakePrepared(restored_from_remote=False))

    assert executor.captured_prompt is None
    assert evidence.decision.outcome is AttemptOutcome.INCOMPLETE


def test_restored_continuation_branch_runs_the_model_normally(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    runner = build_runner(tmp_path, model_executor=executor)

    runner(
        claim(issue_number=24, labels=frozenset({"ready-for-agent", "round-finished"})),
        profile(),
        FakePrepared(restored_from_remote=True),
    )

    assert executor.captured_prompt is not None


def test_normal_claim_with_no_continuation_signal_runs_the_model_normally(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    runner = build_runner(tmp_path, model_executor=executor)

    runner(claim(issue_number=24), profile(), FakePrepared(restored_from_remote=False))

    assert executor.captured_prompt is not None


def build_runner(
    tmp_path: Path,
    *,
    model_executor: "FakeModelExecutor",
    workspace: "FakeWorkspace | None" = None,
    issue_comments=lambda issue: (),
    issue_number: int = 24,
) -> ModelAttemptRunner:
    attempt_state = AttemptStateStore(tmp_path)
    attempt_state.start(issue_number=issue_number, branch=f"agent/issue-{issue_number}")
    attempt_state.transition(AttemptPhase.SETUP)
    return ModelAttemptRunner(
        attempt_state=attempt_state,
        workspace=workspace or FakeWorkspace(),
        verifier=FakeVerifier(),
        evaluator=FakeEvaluator(),
        model_executor=model_executor,
        issue_comments=issue_comments,
    )


class FakeModelExecutor:
    def __init__(self, *, status: ModelExecutionStatus = ModelExecutionStatus.MODEL_LIMIT_REACHED) -> None:
        self.captured_prompt: str | None = None
        self._status = status

    async def execute(
        self, *, issue_body: str, working_directory: Path, issue_number: int | None = None, archive=None
    ) -> ModelExecution:
        self.captured_prompt = issue_body
        return ModelExecution(
            status=self._status,
            explanation="stub",
            stop_reason="end_turn",
            model_usage=None,
            observed_models=(),
            skill_events=(),
        )


class FakeVerifier:
    def prepare(self, profile: object, working_directory: Path):
        result = SimpleNamespace(succeeded=True, commands=())
        return SimpleNamespace(setup=result, baseline=result)

    def mark_review_complete(self, **kwargs: object) -> None:
        return None

    def final_check(self, profile: object, working_directory: Path):
        return SimpleNamespace(succeeded=True, commands=(), exit_code=0)


class FakeEvaluator:
    def evaluate(self, **kwargs: object) -> CompletionDecision:
        return CompletionDecision(AttemptOutcome.INCOMPLETE, False, PublicationPath.NONE, ("stub",))


class FakeWorkspace:
    working_directory = Path("/repository")

    def __init__(self, *, commits: tuple[str, ...] = ()) -> None:
        self._commits = commits

    def commits_added(self, prepared: object) -> tuple[str, ...]:
        return self._commits


class FakePrepared:
    def __init__(self, *, branch: str = "agent/issue-24", restored_from_remote: bool = True) -> None:
        self.branch = branch
        self.restored_from_remote = restored_from_remote


def claim(
    *,
    issue_number: int = 24,
    issue_body: str = "Fix the parser.",
    labels: frozenset[str] = frozenset({"ready-for-agent"}),
) -> Claim:
    return Claim(
        TrackerIssue(
            id=f"issue-{issue_number}",
            number=issue_number,
            title="Implement lifecycle",
            body=issue_body,
            created_at=datetime(2026, 9, 20, tzinfo=UTC),
            state="OPEN",
            labels=labels,
            assignee_logins=(),
            blocked_by=0,
            author_login="reporter",
        ),
        Assignment(f"issue-{issue_number}", "agent-id"),
    )


def profile() -> RepositoryProfile:
    return RepositoryProfile(
        setup=(), check=("pytest",), base_branch="main", timeout=60, setup_timeout=60, env={}
    )
