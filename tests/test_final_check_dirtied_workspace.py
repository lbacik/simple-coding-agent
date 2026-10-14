"""A successful final check must not leave the workspace dirty unnoticed.

Regression tests for the ``next build`` report: the final check runs after
the attempt work is sealed, so a check command that rewrites a tracked file
(``tsconfig.json``) or generates a non-ignored file (``next-env.d.ts``) used
to stay uncommitted. The attempt was still published as ``complete`` and the
dirty tree surfaced only in cleanup as a workspace hold.

The runner now verifies the tree is clean after a successful final check and
before any publication. When the check dirtied the tree the attempt ends
``incomplete`` on the partial path with a cleanup-only reason, the
side-effect files are sealed onto the attempt branch (so cleanup stays
clean), and the next round's starting prompt carries the reason verbatim.
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
from simple_coding_agent.lifecycle import (
    ModelAttemptRunner,
    _build_starting_prompt,
    _continuation_expected,
)
from simple_coding_agent.model_execution import (
    ModelExecution,
    ModelExecutionStatus,
    SkillEvent,
)
from simple_coding_agent.publication import _result_comment
from tests.fakes import InMemoryWorkspace


def _claim(
    *,
    issue_number: int = 24,
    issue_body: str = "Scaffold the project.",
    labels: frozenset[str] = frozenset({"ready-for-agent"}),
) -> Claim:
    return Claim(
        TrackerIssue(
            id=f"issue-{issue_number}",
            number=issue_number,
            title="Scaffold the project",
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
        "check": ("npm run build",),
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
    """Model double that finishes cleanly after leaving one dirty work file."""

    def __init__(self, workspace: InMemoryWorkspace) -> None:
        self._workspace = workspace
        self.captured_prompt: str | None = None

    def set_new_commits_provider(self, provider: object) -> None:
        return None

    async def execute(self, **kwargs: object) -> ModelExecution:
        self.captured_prompt = kwargs.get("issue_body")  # type: ignore[assignment]
        self._workspace.make_dirty("app/page.tsx", "export default function Page() {}")
        return ModelExecution(
            status=ModelExecutionStatus.SUCCEEDED,
            explanation="Scaffolded the project.",
            stop_reason="end_turn",
            model_usage=None,
            observed_models=(),
            skill_events=_reviewed(),
        )


class _NextBuildVerifier:
    """Final-check double reproducing the ``next build`` side effects.

    Rewrites the tracked ``tsconfig.json`` and generates the non-ignored
    ``next-env.d.ts``, then reports success -- exactly what the report's
    profile check did.
    """

    def __init__(self, workspace: InMemoryWorkspace, *, succeeded: bool = True) -> None:
        self._workspace = workspace
        self._succeeded = succeeded

    def prepare(self, profile: object, working_directory: Path, **kwargs: object):
        setup = SimpleNamespace(succeeded=True, commands=())
        baseline = SimpleNamespace(succeeded=True, commands=(), exit_code=0)
        return SimpleNamespace(setup=setup, baseline=baseline)

    def mark_review_complete(self, **kwargs: object) -> None:
        return None

    def final_check(self, profile: object, working_directory: Path):
        self._workspace.make_dirty("tsconfig.json", '{"compilerOptions": {"jsx": "react-jsx"}}')
        self._workspace.make_dirty("next-env.d.ts", "/// <reference types=\"next\" />")
        return SimpleNamespace(succeeded=self._succeeded, commands=(), exit_code=0 if self._succeeded else 1)


class _CleanVerifier(_NextBuildVerifier):
    """Final-check double that leaves the tree untouched."""

    def final_check(self, profile: object, working_directory: Path):
        return SimpleNamespace(succeeded=True, commands=(), exit_code=0)


class _IgnoredOnlyVerifier(_NextBuildVerifier):
    """Final-check double whose only side effect is a branch-ignored artifact."""

    def final_check(self, profile: object, working_directory: Path):
        self._workspace.make_ignored(".next/build-manifest.json", "{}")
        return SimpleNamespace(succeeded=True, commands=(), exit_code=0)


class _RecordingArchive:
    def __init__(self) -> None:
        self.merged: dict[str, object] = {}

    def write_attempt(self, metadata: dict[str, object]) -> None:
        self.merged.update(metadata)

    def write_text(self, name: str, content: str) -> Path:
        return Path(name)


def _run(
    tmp_path: Path,
    workspace: InMemoryWorkspace,
    verifier: object,
    events: list[tuple[str, str]],
    archive: _RecordingArchive,
) -> object:
    attempt_state = AttemptStateStore(tmp_path / "state")
    attempt_state.start(issue_number=24, branch="agent/issue-24")
    attempt_state.transition(AttemptPhase.SETUP)
    runner = ModelAttemptRunner(
        attempt_state=attempt_state,
        workspace=workspace,  # type: ignore[arg-type]
        verifier=verifier,  # type: ignore[arg-type]
        evaluator=CompletionEvaluator(frozenset()),
        model_executor=_SucceededExecutor(workspace),  # type: ignore[arg-type]
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail)
        ),
        attempt_archive_factory=lambda issue_number, started_at: archive,  # type: ignore[return-value]
    )
    return runner(_claim(), _profile(), _prepared())


def _event_details(events: list[tuple[str, str]], name: str) -> list[str]:
    return [detail for event, detail in events if event == name]


def test_dirtied_workspace_ends_incomplete_with_cleanup_only_reason(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo")
    events: list[tuple[str, str]] = []
    archive = _RecordingArchive()

    evidence = _run(tmp_path, workspace, _NextBuildVerifier(workspace), events, archive)

    assert evidence.decision.outcome is AttemptOutcome.INCOMPLETE
    assert evidence.decision.publication_path is PublicationPath.PARTIAL
    assert not evidence.decision.publication_eligible
    reason = "; ".join(evidence.decision.reasons)
    assert "tsconfig.json" in reason
    assert "next-env.d.ts" in reason
    assert "No re-implementation is needed" in reason
    assert "cleanup" in reason
    assert "final_check_succeeded" in [event for event, _ in events]
    dirtied = _event_details(events, "final_check_dirtied_workspace")
    assert len(dirtied) == 1
    assert "tsconfig.json" in dirtied[0]
    assert "next-env.d.ts" in dirtied[0]
    decided = _event_details(events, "completion_decided")
    assert decided and "tsconfig.json" in decided[0] and "next-env.d.ts" in decided[0]
    assert "No re-implementation is needed" in decided[0]


def test_dirtied_files_are_sealed_so_cleanup_stays_clean(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo")
    events: list[tuple[str, str]] = []
    archive = _RecordingArchive()

    evidence = _run(tmp_path, workspace, _NextBuildVerifier(workspace), events, archive)

    assert workspace.is_clean()
    subjects = [commit.subject for commit in workspace.commits_added(_prepared())]
    assert any("Seal attempt work for #24" in subject for subject in subjects)
    assert len(subjects) == 2
    # The result comment names the preservation commit.
    assert evidence.details
    revisions = [commit.revision for commit in workspace.commits_added(_prepared())]
    assert any(revision in evidence.details for revision in revisions)
    # Cleanup (the step that used to raise DirtyWorkspaceError) now passes.
    workspace.cleanup(base_branch="main", prepared=_prepared(), retain_branch=False)
    assert workspace.current_branch == "main"
    # attempt.json carries the reason and the paths.
    record = archive.merged.get("final_check_dirtied_workspace")
    assert isinstance(record, dict)
    assert "tsconfig.json" in record["paths"]
    assert "next-env.d.ts" in record["paths"]
    assert "No re-implementation is needed" in record["reason"]


def test_ignored_only_side_effects_do_not_trigger(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo")
    events: list[tuple[str, str]] = []
    archive = _RecordingArchive()

    evidence = _run(tmp_path, workspace, _IgnoredOnlyVerifier(workspace), events, archive)

    assert _event_details(events, "final_check_dirtied_workspace") == []
    assert evidence.decision.publication_eligible
    assert evidence.decision.outcome is None


def test_clean_final_check_behaves_as_before(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo")
    events: list[tuple[str, str]] = []
    archive = _RecordingArchive()

    evidence = _run(tmp_path, workspace, _CleanVerifier(workspace), events, archive)

    assert _event_details(events, "final_check_dirtied_workspace") == []
    assert evidence.decision.publication_eligible
    assert evidence.decision.outcome is None
    assert [event for event, _ in events][-1] == "completion_decided"


def test_failed_final_check_keeps_existing_reason(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo")
    events: list[tuple[str, str]] = []
    archive = _RecordingArchive()

    evidence = _run(
        tmp_path,
        workspace,
        _NextBuildVerifier(workspace, succeeded=False),
        events,
        archive,
    )

    assert evidence.decision.outcome is AttemptOutcome.INCOMPLETE
    assert _event_details(events, "final_check_dirtied_workspace") == []
    assert evidence.decision.reasons == ("Final authoritative check did not succeed.",)


def test_result_comment_is_a_continuation_signal_for_the_next_round(tmp_path: Path) -> None:
    """The published ``incomplete`` comment promises a branch, so the next
    round treats the issue as a continuation; trusted comments reach the
    starting prompt verbatim, carrying the cleanup-only reason.

    Requeue needs no human action and no automatic-continuation budget: a
    non-handoff release re-applies ``ready-for-agent`` (ordinary queue), while
    the ``round-finished`` + auto-continuation path applies to handoffs only.
    """

    workspace = InMemoryWorkspace(tmp_path / "repo")
    events: list[tuple[str, str]] = []
    archive = _RecordingArchive()

    evidence = _run(tmp_path, workspace, _NextBuildVerifier(workspace), events, archive)
    assert evidence.decision.publication_path is PublicationPath.PARTIAL

    comment = _result_comment(
        SimpleNamespace(
            issue_number=24,
            branch="agent/issue-24",
            started_at="2026-10-11T09:44:17Z",
            decision=evidence.decision,
            check_command="npm run build",
            check_exit_code=0,
            review_cycles=0,
            reviews_performed=1,
            review_findings="none",
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
    assert "tsconfig.json" in comment and "next-env.d.ts" in comment
    assert "No re-implementation is needed" in comment

    issue = _claim().issue
    assert _continuation_expected(issue, lambda unused: (comment,)) is True

    prompt = _build_starting_prompt(
        issue.body,
        (comment,),
        continuation=True,
        issue_number=24,
        profile=_profile(),
    )
    assert comment in prompt
    assert "No re-implementation is needed" in prompt
    assert "tsconfig.json" in prompt and "next-env.d.ts" in prompt
