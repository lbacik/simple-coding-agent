"""A red deferred baseline must end the attempt without a redundant final check.

When setup was deferred past unresolved rebase conflicts, the baseline check
runs only after the model resolved them and its session has ended. A red
deferred baseline used to flow into the final check -- the same gate on the
same tree -- which failed the same way and ended the attempt ``incomplete``
with the generic final-check reason. The attempt now ends ``incomplete`` with
a named reason instead: the final check is skipped, and the reason plus the
failure detail travel in the completion event, a dedicated event,
``attempt.json``, and the result comment, so the next round's prompt carries
them whether or not it defers setup again.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.completion import (
    AttemptOutcome,
    CompletionEvaluator,
    PublicationPath,
)
from simple_coding_agent.config import RepositoryProfile
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.git_workspace import PreparedAttempt
from simple_coding_agent.lifecycle import ModelAttemptRunner, _build_starting_prompt
from simple_coding_agent.model_execution import (
    ModelExecution,
    ModelExecutionStatus,
    SkillEvent,
)
from tests.fakes import InMemoryWorkspace


def _claim(
    *,
    issue_number: int = 24,
    issue_body: str = "Fix the parser.",
    labels: frozenset[str] = frozenset({"ready-for-agent"}),
) -> Claim:
    return Claim(
        TrackerIssue(
            id=f"issue-{issue_number}",
            number=issue_number,
            title="Fix the parser",
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


def _prepared() -> PreparedAttempt:
    return PreparedAttempt(
        branch="agent/issue-24",
        base_revision="base-revision-0",
        restored_from_remote=True,
    )


def _reviewed() -> tuple[SkillEvent, ...]:
    """One observed ``code-review`` invocation, so acceptance can be satisfied."""

    return (
        SkillEvent(
            phase="PreToolUse",
            name="code-review",
            agent_id=None,
            timestamp="2026-10-11T10:00:00Z",
        ),
    )


class _SucceededExecutor:
    """Model double that resolves the conflict and leaves dirty work to seal."""

    def __init__(self, workspace: InMemoryWorkspace) -> None:
        self._workspace = workspace
        self.captured_prompt: str | None = None
        self.after_execute = workspace.note_model_finished

    def set_new_commits_provider(self, provider: object) -> None:
        return None

    async def execute(self, **kwargs: object) -> ModelExecution:
        self.captured_prompt = kwargs.get("issue_body")  # type: ignore[assignment]
        self._workspace.make_dirty("src/parser.py", "fixed")
        if callable(self.after_execute):
            self.after_execute()
        return ModelExecution(
            status=ModelExecutionStatus.SUCCEEDED,
            explanation="Resolved the conflicts and fixed the parser.",
            stop_reason="end_turn",
            model_usage=None,
            observed_models=(),
            skill_events=_reviewed(),
        )


def _failing_check_command(
    *,
    command: str = "pytest",
    exit_code: int | None = 1,
    stdout: str = "1 failed",
    stderr: str = "assertion error",
) -> SimpleNamespace:
    return SimpleNamespace(
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
    )


class _DeferredVerifier:
    """Setup/baseline double with a counting final check."""

    def __init__(
        self,
        *,
        setup_succeeded: bool = True,
        baseline_succeeded: bool = True,
    ) -> None:
        self._setup_succeeded = setup_succeeded
        self._baseline_succeeded = baseline_succeeded
        self.final_check_calls = 0

    def prepare(self, profile: object, working_directory: Path, **kwargs: object):
        setup = SimpleNamespace(
            succeeded=self._setup_succeeded,
            commands=(
                SimpleNamespace(
                    command="setup",
                    exit_code=0 if self._setup_succeeded else 1,
                    stdout="",
                    stderr="" if self._setup_succeeded else "broken",
                    timed_out=False,
                ),
            ),
        )
        baseline = SimpleNamespace(
            succeeded=self._baseline_succeeded,
            commands=(_failing_check_command(),)
            if not self._baseline_succeeded
            else (),
            exit_code=0 if self._baseline_succeeded else 1,
        )
        return SimpleNamespace(setup=setup, baseline=baseline)

    def mark_review_complete(self, **kwargs: object) -> None:
        return None

    def final_check(self, profile: object, working_directory: Path):
        self.final_check_calls += 1
        return SimpleNamespace(succeeded=True, commands=(), exit_code=0)


class _RecordingArchive:
    def __init__(self) -> None:
        self.merged: dict[str, object] = {}
        self.texts: dict[str, str] = {}

    def write_attempt(self, metadata: dict[str, object]) -> None:
        self.merged.update(metadata)

    def write_text(self, name: str, content: str) -> Path:
        self.texts[name] = content
        return Path(name)


def _run(
    tmp_path: Path,
    workspace: InMemoryWorkspace,
    verifier: _DeferredVerifier,
    events: list[tuple[str, str, str]],
    archive: _RecordingArchive,
) -> tuple[object, _SucceededExecutor]:
    attempt_state = AttemptStateStore(tmp_path / "state")
    attempt_state.start(issue_number=24, branch="agent/issue-24")
    attempt_state.transition(AttemptPhase.SETUP)
    executor = _SucceededExecutor(workspace)
    runner = ModelAttemptRunner(
        attempt_state=attempt_state,
        workspace=workspace,  # type: ignore[arg-type]
        verifier=verifier,  # type: ignore[arg-type]
        evaluator=CompletionEvaluator(frozenset()),
        model_executor=executor,  # type: ignore[arg-type]
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level)
        ),
        attempt_archive_factory=lambda issue_number, started_at: archive,  # type: ignore[return-value]
    )
    return (
        runner(_claim(), _profile(), _prepared()),
        executor,
    )


def _event_names(events: list[tuple[str, str, str]]) -> list[str]:
    return [event for event, _, _ in events]


def _event_details(events: list[tuple[str, str, str]], name: str) -> list[str]:
    return [detail for event, detail, _ in events if event == name]


def _seeded_conflict_workspace(root: Path) -> InMemoryWorkspace:
    workspace = InMemoryWorkspace(root)
    workspace.model_commit("Retained work", files=("Retained work",))
    workspace.leave_conflict(files=("SubscriptionController.php",))
    return workspace


def test_red_deferred_baseline_skips_final_check_and_ends_incomplete(tmp_path: Path) -> None:
    workspace = _seeded_conflict_workspace(tmp_path / "repo")
    events: list[tuple[str, str, str]] = []
    archive = _RecordingArchive()
    verifier = _DeferredVerifier(baseline_succeeded=False)

    evidence, _ = _run(tmp_path, workspace, verifier, events, archive)

    assert verifier.final_check_calls == 0
    assert "final_check_started" not in _event_names(events)
    assert evidence.decision.outcome is AttemptOutcome.INCOMPLETE
    assert evidence.decision.publication_path is PublicationPath.PARTIAL
    assert not evidence.decision.publication_eligible


def test_red_deferred_baseline_events_carry_reason_and_failure_detail(tmp_path: Path) -> None:
    workspace = _seeded_conflict_workspace(tmp_path / "repo")
    events: list[tuple[str, str, str]] = []
    archive = _RecordingArchive()

    evidence, _ = _run(tmp_path, workspace, _DeferredVerifier(baseline_succeeded=False), events, archive)

    reason = "; ".join(evidence.decision.reasons)
    assert "conflicts were resolved" in reason
    assert "must start by making the check pass" in reason
    assert "pytest" in reason
    assert "exit 1" in reason
    assert "1 failed" in reason
    assert "assertion error" in reason
    assert reason != "Final authoritative check did not succeed."

    decided = _event_details(events, "completion_decided")
    assert len(decided) == 1
    assert "conflicts were resolved" in decided[0]
    assert "pytest" in decided[0]
    assert "1 failed" in decided[0]

    dedicated = _event_details(events, "deferred_baseline_failed")
    assert len(dedicated) == 1
    assert "conflicts were resolved" in dedicated[0]
    assert "pytest" in dedicated[0]
    assert "1 failed" in dedicated[0]
    assert "assertion error" in dedicated[0]


def test_red_deferred_baseline_is_recorded_in_attempt_json(tmp_path: Path) -> None:
    workspace = _seeded_conflict_workspace(tmp_path / "repo")
    events: list[tuple[str, str, str]] = []
    archive = _RecordingArchive()

    evidence, _ = _run(tmp_path, workspace, _DeferredVerifier(baseline_succeeded=False), events, archive)

    record = archive.merged.get("deferred_baseline_failed")
    assert isinstance(record, dict)
    assert "conflicts were resolved" in record["reason"]
    assert "pytest" in record["reason"]
    assert "1 failed" in record["reason"]
    assert record["reason"] == "; ".join(evidence.decision.reasons)


def test_red_deferred_baseline_result_comment_signals_continuation(tmp_path: Path) -> None:
    """The published comment promises a branch, so the next round continues.

    Trusted result comments reach the starting prompt verbatim; the comment
    must therefore carry the named reason and the failure detail.
    """

    from simple_coding_agent.publication import _result_comment

    workspace = _seeded_conflict_workspace(tmp_path / "repo")
    events: list[tuple[str, str, str]] = []
    archive = _RecordingArchive()

    evidence, _ = _run(tmp_path, workspace, _DeferredVerifier(baseline_succeeded=False), events, archive)
    assert evidence.decision.publication_path is PublicationPath.PARTIAL

    comment = _result_comment(
        SimpleNamespace(
            issue_number=24,
            branch="agent/issue-24",
            started_at="2026-10-11T09:44:17Z",
            decision=evidence.decision,
            check_command="pytest",
            check_exit_code=None,
            review_cycles=0,
            reviews_performed=1,
            review_findings="not collected",
            details=evidence.details,
            requeue_source="human",
            note_commit_sha=None,
        ),
        AttemptOutcome.INCOMPLETE,
        "https://github.com/o/r/tree/agent/issue-24",
        None,
        evidence.details,
        "<!-- agent-attempt: 2026-10-11T09:44:17Z -->",
    )
    assert "Agent Attempt Result: incomplete" in comment
    assert "no branch created" not in comment
    assert "conflicts were resolved" in comment
    assert "must start by making the check pass" in comment
    assert "pytest" in comment
    assert "1 failed" in comment
    assert "assertion error" in comment

    from simple_coding_agent.lifecycle import _continuation_expected

    issue = _claim().issue
    assert _continuation_expected(issue, lambda unused: (comment,)) is True
    prompt = _build_starting_prompt(
        issue.body,
        (comment,),
        continuation=True,
        issue_number=24,
        profile=_profile(),
        setup_deferred=True,
    )
    assert "conflicts were resolved" in prompt
    assert "pytest" in prompt
    assert "1 failed" in prompt


def test_next_round_prompt_carries_reason_without_deferring_setup(tmp_path: Path) -> None:
    """A non-deferred next round sees the reason via the trusted comment.

    When the next round runs its own baseline and that baseline is also red,
    the fresh #176 section appears as well; the two sections share the same
    form but describe different moments, so they must not contradict.
    """

    from simple_coding_agent.publication import _result_comment

    workspace = _seeded_conflict_workspace(tmp_path / "repo")
    events: list[tuple[str, str, str]] = []
    archive = _RecordingArchive()

    evidence, _ = _run(tmp_path, workspace, _DeferredVerifier(baseline_succeeded=False), events, archive)
    comment = _result_comment(
        SimpleNamespace(
            issue_number=24,
            branch="agent/issue-24",
            started_at="2026-10-11T09:44:17Z",
            decision=evidence.decision,
            check_command="pytest",
            check_exit_code=None,
            review_cycles=0,
            reviews_performed=1,
            review_findings="not collected",
            details=evidence.details,
            requeue_source="human",
            note_commit_sha=None,
        ),
        AttemptOutcome.INCOMPLETE,
        "https://github.com/o/r/tree/agent/issue-24",
        None,
        evidence.details,
        "<!-- agent-attempt: 2026-10-11T09:44:17Z -->",
    )

    issue = _claim().issue
    prompt = _build_starting_prompt(
        issue.body,
        (comment,),
        continuation=True,
        issue_number=24,
        profile=_profile(),
    )
    assert "conflicts were resolved" in prompt
    assert "must start by making the check pass" in prompt
    assert "1 failed" in prompt

    fresh_baseline = SimpleNamespace(
        succeeded=False,
        commands=(_failing_check_command(stdout="2 failed", stderr="other error"),),
    )
    prompt_with_fresh_failure = _build_starting_prompt(
        issue.body,
        (comment,),
        continuation=True,
        issue_number=24,
        profile=_profile(),
        baseline=fresh_baseline,
    )
    # The deferred account survives alongside the fresh pre-model failure...
    assert "after the rebase conflicts were resolved" in prompt_with_fresh_failure
    # ...which keeps the #176 wording about the current branch.
    assert "before this attempt started" in prompt_with_fresh_failure
    assert "2 failed" in prompt_with_fresh_failure
    assert "other error" in prompt_with_fresh_failure


def test_green_deferred_baseline_runs_final_check_unchanged(tmp_path: Path) -> None:
    workspace = _seeded_conflict_workspace(tmp_path / "repo")
    events: list[tuple[str, str, str]] = []
    archive = _RecordingArchive()
    verifier = _DeferredVerifier(baseline_succeeded=True)

    evidence, _ = _run(tmp_path, workspace, verifier, events, archive)

    assert verifier.final_check_calls == 1
    assert "final_check_started" in _event_names(events)
    assert _event_details(events, "deferred_baseline_failed") == []
    assert "deferred_baseline_failed" not in archive.merged
    assert evidence.decision.publication_eligible
    assert evidence.decision.outcome is None


def test_deferred_setup_failure_keeps_early_return_without_deferred_event(tmp_path: Path) -> None:
    workspace = _seeded_conflict_workspace(tmp_path / "repo")
    events: list[tuple[str, str, str]] = []
    archive = _RecordingArchive()
    verifier = _DeferredVerifier(setup_succeeded=False, baseline_succeeded=True)

    evidence, _ = _run(tmp_path, workspace, verifier, events, archive)

    assert verifier.final_check_calls == 0
    assert "setup_failed" in _event_names(events)
    assert "final_check_started" not in _event_names(events)
    assert _event_details(events, "deferred_baseline_failed") == []
    assert "deferred_baseline_failed" not in archive.merged
    assert evidence.check_exit_code is None
