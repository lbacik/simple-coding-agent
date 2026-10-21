"""A continued attempt resumes from one current Handoff note, not history.

The starting prompt must exclude every historical generated handoff result
(including its note payload) and strip machine bookkeeping from generated
result comments that remain, while keeping ordinary trusted human guidance
verbatim in order -- even when the operator and the agent share one GitHub
identity, so the filter is structural, never identity-based. Filtering is
presentation-only: continuation detection, budget counting, and publication
recovery still see the full comment history.
"""

from __future__ import annotations

from pathlib import Path

from simple_coding_agent.completion import AttemptOutcome
from simple_coding_agent.github_tracker import GitHubIdentity, GitHubTracker, IssueComment
from tests.fakes import InMemoryWorkspace
from tests.test_emergency_handoff_note import RichExecutor, _rich_execution, _validate_note
from tests.test_github_tracker import issue as tracker_issue
from tests.test_model_attempt_runner import (
    FakeModelExecutor,
    FakePrepared,
    build_runner,
    claim,
    profile,
    seeded_workspace,
)


def handoff_comment(payload: str, *, marker: str, source: str = "automatic") -> str:
    """A production-shaped handoff attempt-result comment carrying a note payload."""

    return (
        "## Agent Attempt Result: handoff\n\n"
        "**Branch**: [agent/issue-24](https://github.com/o/r/tree/agent/issue-24)\n"
        "**PR**: none\n"
        "**Handoff note commit**: `abc123def456`\n\n"
        "### Evidence\n"
        "- Check command: `pytest` → exit code None\n"
        "- Reviews performed: 0\n"
        "- code-review was not run\n"
        "- Review findings: not run\n\n"
        "### Details\n"
        f"{payload}\n\n"
        "Automatic continuation 1/2 used; the agent will continue automatically.\n\n"
        f"<!-- agent-attempt: {marker} -->\n"
        f"<!-- requeue-source: {source} -->"
    )


FIRST_PAYLOAD = (
    "# Handoff note: issue #24\n\n"
    "## Summary\n\nFirst-round slice work with obsolete routing instructions.\n\n"
    "## Remaining work\n\nFirst payload marker: wire the stale integration branch."
)
SECOND_PAYLOAD = (
    "# Handoff note: issue #24\n\n"
    "## Summary\n\nSecond-round progress superseding the first round.\n\n"
    "## Remaining work\n\nSecond payload marker: finish the v3 journey port."
)


def test_continuation_prompt_excludes_every_historical_handoff_comment(
    tmp_path: Path,
) -> None:
    executor = FakeModelExecutor()
    comments = (
        "please also handle timeouts",
        handoff_comment(FIRST_PAYLOAD, marker="2026-10-19T20:32:39.838475Z", source="human"),
        handoff_comment(SECOND_PAYLOAD, marker="2026-10-19T20:51:48.155629Z"),
        "correction: the de-facto PR target is main, not the stale branch",
    )
    workspace = seeded_workspace(tmp_path / "repo", "Previous attempt work")
    runner = build_runner(
        tmp_path, model_executor=executor, workspace=workspace,
        issue_comments=lambda issue: comments, issue_number=24,
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    prompt = executor.captured_prompt
    assert prompt is not None
    assert "please also handle timeouts" in prompt
    assert "correction: the de-facto PR target is main" in prompt
    assert prompt.index("please also handle timeouts") < prompt.index(
        "correction: the de-facto PR target is main"
    )
    assert "Agent Attempt Result: handoff" not in prompt
    assert "First payload marker" not in prompt
    assert "Second payload marker" not in prompt
    assert "agent-attempt" not in prompt
    assert "requeue-source" not in prompt
    assert ".agent/handoff/24.md" in prompt
    assert "## Completion gate" in prompt


def test_legacy_handoff_comment_without_markers_is_excluded(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    comments = (
        "keep the old retry behaviour",
        "## Agent Attempt Result: handoff\n\nLegacy payload marker: old routing plan.",
    )
    workspace = seeded_workspace(tmp_path / "repo", "Previous attempt work")
    runner = build_runner(
        tmp_path, model_executor=executor, workspace=workspace,
        issue_comments=lambda issue: comments, issue_number=24,
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    prompt = executor.captured_prompt
    assert prompt is not None
    assert "keep the old retry behaviour" in prompt
    assert "Legacy payload marker" not in prompt
    assert "Agent Attempt Result: handoff" not in prompt


def test_fresh_attempt_prompt_also_excludes_stale_handoff_payloads(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    comments = (
        "keep the old retry behaviour",
        handoff_comment(FIRST_PAYLOAD, marker="2026-10-19T20:32:39.838475Z", source="human"),
    )
    runner = build_runner(
        tmp_path, model_executor=executor, issue_comments=lambda issue: comments
    )

    runner(claim(issue_body="Fix the parser."), profile(), FakePrepared())

    prompt = executor.captured_prompt
    assert prompt is not None
    assert "keep the old retry behaviour" in prompt
    assert "First payload marker" not in prompt
    assert ".agent/handoff" not in prompt


def test_continuation_pointer_instructs_reading_the_local_note_first(
    tmp_path: Path,
) -> None:
    executor = FakeModelExecutor()
    workspace = seeded_workspace(tmp_path / "repo", "Previous attempt work")
    runner = build_runner(
        tmp_path, model_executor=executor, workspace=workspace, issue_number=24
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    prompt = executor.captured_prompt
    assert prompt is not None
    assert "Read the current Handoff note" in prompt
    assert ".agent/handoff/24.md" in prompt
    assert "before implementing" in prompt
    assert "code-review" in prompt


def test_bookkeeping_stripped_from_remaining_generated_results(tmp_path: Path) -> None:
    executor = FakeModelExecutor()
    comments = (
        "## Agent Attempt Result: incomplete\n\n"
        "**Branch**: [agent/issue-24](https://github.com/o/r/tree/agent/issue-24)\n"
        "**PR**: none\n\n"
        "### Details\n"
        "Final check failed with work preserved on the branch.\n\n"
        "Automatic continuation 1/2 used; the agent will continue automatically.\n\n"
        "<!-- agent-attempt: 2026-10-19T20:32:39.838475Z -->\n"
        "<!-- requeue-source: human -->",
    )
    workspace = seeded_workspace(tmp_path / "repo", "Previous attempt work")
    runner = build_runner(
        tmp_path, model_executor=executor, workspace=workspace,
        issue_comments=lambda issue: comments, issue_number=24,
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    prompt = executor.captured_prompt
    assert prompt is not None
    assert "Final check failed with work preserved on the branch." in prompt
    assert "agent-attempt" not in prompt
    assert "requeue-source" not in prompt
    assert "**Branch**" not in prompt
    assert "Automatic continuation 1/2 used" not in prompt


def test_prompt_filtering_leaves_continuation_detection_untouched(
    tmp_path: Path,
) -> None:
    """The missing-branch short-circuit still sees the raw handoff comment."""

    executor = FakeModelExecutor()
    comments = (
        handoff_comment(SECOND_PAYLOAD, marker="2026-10-19T20:51:48.155629Z"),
    )
    runner = build_runner(tmp_path, model_executor=executor, issue_comments=lambda issue: comments)

    evidence = runner(
        claim(issue_number=24), profile(), FakePrepared(restored_from_remote=False)
    )

    assert executor.captured_prompt is None
    assert evidence.decision.outcome is AttemptOutcome.INCOMPLETE


class SharedIdentityTransport:
    """Viewer and issue author share one login, as on lbacik/simple-rpg-engine."""

    def __init__(self, comments: tuple[IssueComment, ...]) -> None:
        self._comments = comments

    def viewer(self) -> GitHubIdentity:
        return GitHubIdentity(id="viewer-id", login="lbacik")

    def list_issue_comments(self, repository: str, issue_number: int) -> tuple[IssueComment, ...]:
        assert repository == "octo/example"
        return self._comments


def test_shared_identity_human_guidance_survives_prompt_filtering() -> None:
    transport = SharedIdentityTransport(
        comments=(
            IssueComment(author_login="lbacik", body="hold the old retry behaviour"),
            IssueComment(
                author_login="lbacik",
                body=handoff_comment(SECOND_PAYLOAD, marker="2026-10-19T20:51:48.155629Z"),
            ),
            IssueComment(author_login="lbacik", body="correction: target main now"),
        )
    )
    tracker = GitHubTracker(transport, "octo/example")
    issue = tracker_issue(1, author_login="lbacik")

    trusted = tracker.trusted_comments(issue)
    assert len(trusted) == 3

    prompt_comments = tracker.prompt_comments(issue)

    assert prompt_comments == (
        "hold the old retry behaviour",
        "correction: target main now",
    )


PRIOR_NOTE = (
    "# Handoff note: issue #24\n\n"
    "- issue: 24\n"
    "- started_at: 2026-10-19T20:32:39.838475Z\n"
    "- reason: token_soft_threshold\n"
    "- last_work_commit: abc123def456\n\n"
    "## Summary\n\nPrior round wired the journey port.\n\n"
    "## Remaining work\n\nPrior remaining marker: run the full gate.\n"
)


def write_prior_round_note(workspace: InMemoryWorkspace, content: str = PRIOR_NOTE) -> str:
    """Commit a previous round's complete note, returning its immutable revision."""

    note_path = workspace.working_directory / ".agent/handoff/24.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(content)
    commit = workspace.prior_round_commit(
        "Handoff note: issue #24", (".agent/handoff/24.md",)
    )
    return commit.revision


def test_emergency_note_preserves_last_complete_note_at_immutable_revision(
    tmp_path: Path,
) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo")
    prior_revision = write_prior_round_note(workspace)
    workspace.make_dirty("work.txt", "model edits")
    runner = build_runner(
        tmp_path, model_executor=RichExecutor(_rich_execution()), workspace=workspace
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    assert len(workspace.note_calls) == 1
    note = workspace.note_calls[0][1]
    assert "## Previous complete note" in note
    assert prior_revision in note
    assert "Prior remaining marker" in note
    assert "older context" in note
    assert "runtime evidence" in note
    assert _validate_note(workspace, FakePrepared()).valid


def test_emergency_note_without_previous_note_says_so_explicitly(
    tmp_path: Path,
) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo")
    workspace.make_dirty("work.txt", "model edits")
    runner = build_runner(
        tmp_path, model_executor=RichExecutor(_rich_execution()), workspace=workspace
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    note = workspace.note_calls[0][1]
    assert "## Previous complete note" in note
    assert "No previous complete Handoff note was found on this branch" in note
    assert _validate_note(workspace, FakePrepared()).valid


def test_emergency_note_with_unreadable_previous_note_names_its_revision(
    tmp_path: Path,
) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo")
    ghost = workspace.prior_round_commit(
        "Handoff note: issue #24", (".agent/handoff/24.md",)
    )
    workspace.make_dirty("work.txt", "model edits")
    runner = build_runner(
        tmp_path, model_executor=RichExecutor(_rich_execution()), workspace=workspace
    )

    runner(claim(issue_number=24, issue_body="Fix the parser."), profile(), FakePrepared())

    note = workspace.note_calls[0][1]
    assert "## Previous complete note" in note
    assert ghost.revision in note
    assert "could not be read" in note
    assert "Prior remaining marker" not in note
    assert _validate_note(workspace, FakePrepared()).valid
