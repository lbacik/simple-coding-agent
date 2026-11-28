"""Fresh attempts may start on a red baseline to fix the base branch (issue #198).

Either the ``fix-baseline`` issue label or the repository profile's
``baseline: tolerate`` setting opts a fresh (non-continuation) attempt into
the same baseline tolerance continuations already enjoy. The tolerated
failure rides along in the model's starting prompt, the final check stays
authoritative, and the attempt records that it started red.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.command_runner import CommandResult, CommandStatus
from simple_coding_agent.completion import (
    AttemptOutcome,
    CompletionDecision,
    CompletionEvaluator,
    PublicationPath,
    ReviewEvidence,
    VerificationRunner,
    _build_baseline_failure_section,
)
from simple_coding_agent.config import RepositoryProfile, ProfileError, load_repository_profile
from simple_coding_agent.github_tracker import FIX_BASELINE, Assignment, Claim, TrackerIssue
from simple_coding_agent.git_workspace import PreparedAttempt
from simple_coding_agent.lifecycle import (
    ModelAttemptRunner,
    _build_starting_prompt,
    _red_baseline_tolerance_sources,
    _tolerates_red_baseline,
)
from simple_coding_agent.model_execution import ModelExecution, ModelExecutionStatus
from tests.fakes import InMemoryWorkspace


def _command_result(status: CommandStatus = CommandStatus.SUCCEEDED) -> CommandResult:
    from simple_coding_agent.command_runner import CommandExecution

    exit_code = 0 if status is CommandStatus.SUCCEEDED else 1
    return CommandResult(status, (CommandExecution("check", exit_code, "", "", False),), 0.1)


def _profile(**overrides: object) -> RepositoryProfile:
    fields: dict[str, object] = {
        "setup": (),
        "check": ("pytest",),
        "base_branch": "main",
        "timeout": 60,
        "setup_timeout": 60,
        "env": {},
        "model_instructions": "",
    }
    fields.update(overrides)
    return RepositoryProfile(**fields)  # type: ignore[arg-type]


def _claim(
    *,
    issue_number: int = 24,
    labels: frozenset[str] = frozenset({"ready-for-agent"}),
) -> Claim:
    return Claim(
        TrackerIssue(
            id=f"issue-{issue_number}",
            number=issue_number,
            title="Fix the fixtures",
            body="Fix the fixtures.",
            created_at=datetime(2026, 9, 20, tzinfo=UTC),
            state="OPEN",
            labels=labels,
            assignee_logins=(),
            blocked_by=0,
            author_login="reporter",
        ),
        Assignment(f"issue-{issue_number}", "agent-id"),
    )


def test_fix_baseline_constant_value() -> None:
    assert FIX_BASELINE == "fix-baseline"


def test_tolerates_label_without_profile_flag() -> None:
    issue = _claim(labels=frozenset({"ready-for-agent", FIX_BASELINE})).issue
    assert _tolerates_red_baseline(issue, _profile()) is True
    assert _tolerates_red_baseline(_claim().issue, _profile()) is False


def test_tolerates_profile_flag_without_label() -> None:
    assert _tolerates_red_baseline(_claim().issue, _profile(tolerate_red_baseline=True)) is True


def test_tolerance_sources_name_label_and_profile() -> None:
    issue = _claim(labels=frozenset({"ready-for-agent", FIX_BASELINE})).issue
    assert _red_baseline_tolerance_sources(issue, _profile()) == (f"label:{FIX_BASELINE}",)
    assert _red_baseline_tolerance_sources(_claim().issue, _profile(tolerate_red_baseline=True)) == (
        "profile:baseline=tolerate",
    )


def test_verification_tolerates_red_baseline_on_fresh_attempt(tmp_path: Path) -> None:
    class Runner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def run(self, commands, **kwargs):
            self.calls.append(tuple(commands))
            if len(self.calls) == 2:
                return _command_result(CommandStatus.FAILED)
            return _command_result()

    profile = _profile()
    verifier = VerificationRunner(Runner())
    preparation = verifier.prepare(
        profile, tmp_path, continuation=False, tolerate_red_baseline=True
    )
    assert preparation.setup.succeeded
    assert not preparation.baseline.succeeded

    verifier.mark_review_complete(
        model_status=ModelExecutionStatus.SUCCEEDED,
        commit_count=1,
        review=ReviewEvidence((), 0),
    )
    assert verifier.final_check(profile, tmp_path).succeeded


def test_verification_still_blocks_red_baseline_by_default(tmp_path: Path) -> None:
    class Runner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def run(self, commands, **kwargs):
            self.calls.append(tuple(commands))
            if len(self.calls) == 2:
                return _command_result(CommandStatus.FAILED)
            return _command_result()

    profile = _profile()
    verifier = VerificationRunner(Runner())
    preparation = verifier.prepare(profile, tmp_path)
    assert not preparation.baseline.succeeded
    try:
        verifier.mark_review_complete(
            model_status=ModelExecutionStatus.SUCCEEDED,
            commit_count=1,
            review=ReviewEvidence((), 0),
        )
    except Exception:
        pass
    else:
        raise AssertionError("expected VerificationOrderError for red baseline")


def test_evaluator_tolerates_red_baseline_for_fixup(tmp_path: Path) -> None:
    evaluator = CompletionEvaluator({"must-fix"})
    common = dict(
        setup=_command_result(),
        baseline=_command_result(CommandStatus.FAILED),
        model_status=ModelExecutionStatus.SUCCEEDED,
        commit_count=1,
        acceptance_criteria_satisfied=True,
        review=ReviewEvidence((), 0),
        final_check=_command_result(CommandStatus.SUCCEEDED),
        tolerate_red_baseline=True,
    )
    eligible = evaluator.evaluate(**common)
    complete = evaluator.evaluate(**common, push_succeeded=True, pr_exists=True)
    assert eligible.outcome is None
    assert complete.outcome is AttemptOutcome.COMPLETE


def test_evaluator_still_blocks_red_baseline_by_default() -> None:
    decision = CompletionEvaluator({"must-fix"}).evaluate(
        setup=_command_result(),
        baseline=_command_result(CommandStatus.FAILED),
        model_status=None,
        commit_count=0,
        acceptance_criteria_satisfied=False,
        review=ReviewEvidence((), 0),
        final_check=None,
    )
    assert decision.outcome is AttemptOutcome.INCOMPLETE


def test_fresh_tolerated_prompt_names_base_branch_not_prior_commits() -> None:
    baseline = SimpleNamespace(
        succeeded=False,
        commands=(
            SimpleNamespace(
                command="npm test",
                exit_code=1,
                stdout="4 failing",
                stderr="assertion error",
                timed_out=False,
            ),
        ),
    )
    section = _build_baseline_failure_section(baseline, fresh_tolerated=True)
    assert "base branch" in section
    assert "npm test" in section
    assert "exit 1" in section
    assert "4 failing" in section
    continuation_section = _build_baseline_failure_section(baseline)
    assert "earlier attempt" in continuation_section
    assert "base branch" not in continuation_section


def test_starting_prompt_fresh_tolerated_omits_handoff_but_keeps_failure() -> None:
    baseline = SimpleNamespace(
        succeeded=False,
        commands=(
            SimpleNamespace(
                command="npm test",
                exit_code=1,
                stdout="4 failing",
                stderr="",
                timed_out=False,
            ),
        ),
    )
    prompt = _build_starting_prompt(
        "Fix the fixtures.",
        (),
        continuation=False,
        issue_number=24,
        profile=_profile(),
        baseline=baseline,
        fresh_baseline_tolerated=True,
    )
    assert "Baseline check failure" in prompt
    assert "base branch" in prompt
    assert ".agent/handoff/24.md" not in prompt


def test_profile_baseline_tolerate_loads(tmp_path: Path) -> None:
    profile_path = tmp_path / "docs" / "agents" / "simple-coding-agent-profile.yml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("setup: pytest\ncheck: pytest\nbaseline: tolerate\n")
    profile = load_repository_profile(tmp_path)
    assert profile.tolerate_red_baseline is True


def test_profile_baseline_defaults_to_require(tmp_path: Path) -> None:
    profile_path = tmp_path / "docs" / "agents" / "simple-coding-agent-profile.yml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("setup: pytest\ncheck: pytest\n")
    assert load_repository_profile(tmp_path).tolerate_red_baseline is False


def test_profile_baseline_rejects_unknown_value(tmp_path: Path) -> None:
    profile_path = tmp_path / "docs" / "agents" / "simple-coding-agent-profile.yml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("setup: pytest\ncheck: pytest\nbaseline: whatever\n")
    try:
        load_repository_profile(tmp_path)
    except ProfileError:
        pass
    else:
        raise AssertionError("expected ProfileError for unknown baseline value")


class _FakeModelExecutor:
    def __init__(self) -> None:
        self.captured_prompt: str | None = None

    def set_new_commits_provider(self, provider) -> None:
        return None

    async def execute(self, **kwargs) -> ModelExecution:
        self.captured_prompt = kwargs.get("issue_body")
        return ModelExecution(
            status=ModelExecutionStatus.MODEL_LIMIT_REACHED,
            explanation="stub",
            stop_reason="end_turn",
            model_usage=None,
            observed_models=(),
            skill_events=(),
        )


class _FakeVerifier:
    def __init__(self, *, baseline_succeeded: bool = True) -> None:
        self._baseline_succeeded = baseline_succeeded
        self.seen: dict[str, object] = {}

    def prepare(self, profile: object, working_directory: Path, **kwargs: object):
        self.seen.update(kwargs)
        setup = SimpleNamespace(succeeded=True, commands=())
        baseline = SimpleNamespace(
            succeeded=self._baseline_succeeded,
            commands=()
            if self._baseline_succeeded
            else (
                SimpleNamespace(
                    command="npm test",
                    exit_code=1,
                    stdout="4 failing",
                    stderr="assertion error",
                    timed_out=False,
                ),
            ),
            exit_code=0 if self._baseline_succeeded else 1,
        )
        return SimpleNamespace(setup=setup, baseline=baseline)

    def mark_review_complete(self, **kwargs: object) -> None:
        return None

    def final_check(self, profile: object, working_directory: Path):
        return SimpleNamespace(succeeded=True, commands=(), exit_code=0)


class _FakeArchive:
    def __init__(self, directory: Path | None = None) -> None:
        self.merged: dict[str, object] = {}
        self.directory: Path = directory if directory is not None else Path("/tmp")

    def write_attempt(self, metadata: dict[str, object]) -> None:
        self.merged.update(metadata)

    def write_text(self, name: str, content: str) -> Path:
        return Path(name)

    def record_evidence_metadata(self, name: str, metadata: dict[str, object]) -> None:
        return None


class _FakeEvaluator:
    def evaluate(self, **kwargs: object) -> CompletionDecision:
        return CompletionDecision(AttemptOutcome.INCOMPLETE, False, PublicationPath.NONE, ("stub",))


def _runner(
    tmp_path: Path,
    *,
    executor: _FakeModelExecutor,
    verifier: _FakeVerifier,
    events: list,
    archive: _FakeArchive,
) -> ModelAttemptRunner:
    attempt_state = AttemptStateStore(tmp_path / "state")
    attempt_state.start(issue_number=24, branch="agent/issue-24")
    attempt_state.transition(AttemptPhase.SETUP)
    runner = ModelAttemptRunner(
        attempt_state=attempt_state,
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        verifier=verifier,  # type: ignore[arg-type]
        evaluator=_FakeEvaluator(),  # type: ignore[arg-type]
        model_executor=executor,  # type: ignore[arg-type]
        issue_comments=lambda issue: (),
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level)
        ),
    )
    runner._attempt_archive_factory = lambda issue_number, started_at: archive  # type: ignore[attr-defined]
    return runner


def _prepared() -> PreparedAttempt:
    return PreparedAttempt(
        branch="agent/issue-24",
        base_revision="base-revision-0",
        restored_from_remote=True,
        rebase_conflicts=(),
        pre_rebase_revision=None,
    )


def test_fresh_fix_baseline_label_reaches_model_with_failure_and_record(tmp_path: Path) -> None:
    executor = _FakeModelExecutor()
    verifier = _FakeVerifier(baseline_succeeded=False)
    events: list = []
    archive = _FakeArchive(tmp_path)
    runner = _runner(tmp_path, executor=executor, verifier=verifier, events=events, archive=archive)

    runner(
        _claim(labels=frozenset({"ready-for-agent", FIX_BASELINE})),
        _profile(),
        _prepared(),
    )

    assert executor.captured_prompt is not None
    assert "Baseline check failure" in executor.captured_prompt
    assert "npm test" in executor.captured_prompt
    assert "4 failing" in executor.captured_prompt
    assert ".agent/handoff/24.md" not in executor.captured_prompt
    assert verifier.seen.get("tolerate_red_baseline") is True
    kinds = [event for event, _, _ in events]
    assert "baseline_check_failed" in kinds
    assert "red_baseline_tolerated" in kinds
    assert "model_dispatch_starting" in kinds
    assert "red_baseline_tolerated" in archive.merged


def test_fresh_profile_tolerate_reaches_model(tmp_path: Path) -> None:
    executor = _FakeModelExecutor()
    verifier = _FakeVerifier(baseline_succeeded=False)
    events: list = []
    archive = _FakeArchive(tmp_path)
    runner = _runner(tmp_path, executor=executor, verifier=verifier, events=events, archive=archive)

    runner(_claim(), _profile(tolerate_red_baseline=True), _prepared())

    assert executor.captured_prompt is not None
    assert "Baseline check failure" in executor.captured_prompt
    assert "red_baseline_tolerated" in archive.merged


def test_fresh_without_tolerance_still_stops_before_model(tmp_path: Path) -> None:
    executor = _FakeModelExecutor()
    verifier = _FakeVerifier(baseline_succeeded=False)
    events: list = []
    archive = _FakeArchive(tmp_path)
    runner = _runner(tmp_path, executor=executor, verifier=verifier, events=events, archive=archive)

    runner(_claim(), _profile(), _prepared())

    assert executor.captured_prompt is None
    assert "red_baseline_tolerated" not in archive.merged
