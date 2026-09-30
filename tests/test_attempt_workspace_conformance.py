"""Workspace conformance suite: the same tests against both adapters.

``GitWorkspace`` (temporary bare remote, as the end-to-end handoff test
does) and ``InMemoryWorkspace`` (the shared fake) both satisfy
``AttemptWorkspace``; every test here runs against both so the seam stays
real. Covers ``seal``, ``commits_added``, ``conflict_state``,
``commit_handoff_note``, ``handoff_note_commit`` and dirty ``cleanup``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_coding_agent.git_workspace import (
    DirtyWorkspaceError,
    GitWorkspace,
    GitWorkspaceError,
)
from simple_coding_agent.model_execution import ModelExecutionStatus
from tests.fakes import (
    InMemoryWorkspace,
    git,
    repository_with_main,
    run_command,
    write_and_commit,
)


def _attempt_fixture(request: pytest.FixtureRequest, tmp_path: Path) -> SimpleNamespace:
    # The attempt starts before any of its commits, as the checkpoint does.
    attempt_id = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    if request.param == "git":
        remote, seed = repository_with_main(tmp_path)
        clone = tmp_path / "clone"
        workspace: GitWorkspace | InMemoryWorkspace = GitWorkspace(
            clone, str(remote), token_provider=lambda: "secret-token"
        )
        prepared = workspace.prepare_attempt(base_branch="main", issue_number=24)
        git(clone, "config", "user.name", "Test User")
        git(clone, "config", "user.email", "test@example.com")
        return SimpleNamespace(
            workspace=workspace,
            prepared=prepared,
            root=clone,
            seed=seed,
            kind="git",
            attempt_id=attempt_id,
        )
    root = tmp_path / "mem"
    memory = InMemoryWorkspace(root, issue_number=24)
    prepared = memory.prepare_attempt(base_branch="main", issue_number=24)
    return SimpleNamespace(
        workspace=memory,
        prepared=prepared,
        root=root,
        seed=None,
        kind="memory",
        attempt_id=attempt_id,
    )


@pytest.fixture(params=["git", "memory"])
def attempt(request: pytest.FixtureRequest, tmp_path: Path) -> SimpleNamespace:
    return _attempt_fixture(request, tmp_path)


def dirty_file(attempt: SimpleNamespace, name: str, contents: str = "dirty work") -> None:
    """Leave ``name`` uncommitted in the attempt worktree."""

    if attempt.kind == "git":
        path = attempt.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    else:
        attempt.workspace.make_dirty(name, contents)


def model_commit(attempt: SimpleNamespace, name: str, contents: str, message: str) -> None:
    """Commit ``name`` directly, as the model would with its own commit."""

    if attempt.kind == "git":
        path = attempt.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_and_commit(attempt.root, name, contents, message)
    else:
        attempt.workspace.make_dirty(name, contents)
        attempt.workspace.model_commit(message, files=(name,))


def prior_round_commit(attempt: SimpleNamespace, name: str, contents: str, message: str) -> None:
    """Commit ``name`` as an earlier round of a continued branch did."""

    if attempt.kind == "git":
        path = attempt.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        git(attempt.root, "add", name)
        yesterday = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        run_command(
            ("git", "commit", "-m", message),
            cwd=attempt.root,
            env={**os.environ, "GIT_AUTHOR_DATE": yesterday, "GIT_COMMITTER_DATE": yesterday},
        )
    else:
        attempt.workspace.prior_round_commit(message, files=(name,))


def leave_conflict(attempt: SimpleNamespace) -> None:
    """Leave an unresolved rebase in the worktree, as preparation does."""

    if attempt.kind == "git":
        write_and_commit(
            attempt.root, "README.md", "branch change", "Conflicting branch commit"
        )
        write_and_commit(
            attempt.seed, "README.md", "upstream fix", "Conflicting upstream commit"
        )
        git(attempt.seed, "push", "origin", "main")
        attempt.prepared = attempt.workspace.prepare_attempt(
            base_branch="main", issue_number=24
        )
    else:
        attempt.workspace.leave_conflict(("README.md",))


def current_branch(attempt: SimpleNamespace) -> str:
    if attempt.kind == "git":
        return git(attempt.root, "branch", "--show-current")
    return attempt.workspace.current_branch


def attempt_id_trailer(attempt: SimpleNamespace) -> str:
    """The Attempt-Id trailer of HEAD (empty when the commit carries none)."""

    if attempt.kind == "git":
        return git(attempt.root, "log", "--format=%(trailers:key=Attempt-Id,valueonly)", "-1")
    bodies = attempt.workspace.commit_bodies()
    assert bodies, "expected at least one commit"
    for line in bodies[0].splitlines():
        if line.startswith("Attempt-Id:"):
            return line.partition(":")[2].strip()
    return ""


def head_files(attempt: SimpleNamespace) -> tuple[str, ...]:
    """Repository-relative paths touched by HEAD."""

    if attempt.kind == "git":
        output = git(attempt.root, "show", "--name-only", "--format=", "HEAD")
        return tuple(line for line in output.splitlines() if line.strip())
    files = attempt.workspace.commit_files()
    assert files, "expected at least one commit"
    return tuple(sorted(files[0]))


def test_seal_is_a_noop_on_a_clean_tree(attempt: SimpleNamespace) -> None:
    assert (
        attempt.workspace.seal(
            issue_number=24,
            attempt_id="2026-09-29T00:00:00Z",
            status=ModelExecutionStatus.SUCCEEDED,
        )
        is None
    )
    assert attempt.workspace.commits_added(attempt.prepared) == ()


def test_seal_commits_dirty_and_untracked_work_attributed_to_the_attempt(
    attempt: SimpleNamespace,
) -> None:
    dirty_file(attempt, "tracked.txt", "model edits")
    dirty_file(attempt, "untracked.txt", "new file")

    commit = attempt.workspace.seal(
        issue_number=24,
        attempt_id="2026-09-29T00:00:00Z",
        status=ModelExecutionStatus.SUCCEEDED,
    )

    assert commit is not None
    assert commit.subject == "Seal attempt work for #24 (succeeded)"
    assert attempt_id_trailer(attempt) == "2026-09-29T00:00:00Z"
    assert attempt.workspace.is_clean()
    commits = attempt.workspace.commits_added(attempt.prepared)
    assert [entry.subject for entry in commits] == [commit.subject]
    assert commits[0].revision == commit.revision


def test_seal_carries_the_execution_status_in_its_subject(
    attempt: SimpleNamespace,
) -> None:
    dirty_file(attempt, "work.txt")

    commit = attempt.workspace.seal(
        issue_number=24,
        attempt_id="attempt-7",
        status=ModelExecutionStatus.INFRASTRUCTURE_ERROR,
    )

    assert commit is not None
    assert commit.subject == "Seal attempt work for #24 (infrastructure_error)"


def test_seal_refuses_an_unresolved_rebase_without_committing(
    attempt: SimpleNamespace,
) -> None:
    leave_conflict(attempt)
    before = attempt.workspace.commits_added(attempt.prepared)

    with pytest.raises(GitWorkspaceError, match="unresolved"):
        attempt.workspace.seal(
            issue_number=24,
            attempt_id="2026-09-29T00:00:00Z",
            status=ModelExecutionStatus.SUCCEEDED,
        )

    assert attempt.workspace.commits_added(attempt.prepared) == before


def test_conflict_state_reports_a_clean_tree(attempt: SimpleNamespace) -> None:
    state = attempt.workspace.conflict_state()

    assert state.in_progress is False
    assert state.conflicted_files == ()
    assert state.resolution_problems == ()


def test_conflict_state_reports_an_unresolved_rebase(
    attempt: SimpleNamespace,
) -> None:
    leave_conflict(attempt)

    state = attempt.workspace.conflict_state()

    assert state.in_progress is True
    assert state.conflicted_files != ()
    assert state.resolution_problems != ()


def test_commits_added_lists_attempt_commits_newest_first(
    attempt: SimpleNamespace,
) -> None:
    model_commit(attempt, "first.txt", "one", "First model commit")
    model_commit(attempt, "second.txt", "two", "Second model commit")

    commits = attempt.workspace.commits_added(attempt.prepared)

    assert [entry.subject for entry in commits] == [
        "Second model commit",
        "First model commit",
    ]


def test_commit_handoff_note_commits_only_the_note(attempt: SimpleNamespace) -> None:
    dirty_file(attempt, "work.txt", "unrelated edits")

    commit = attempt.workspace.commit_handoff_note(24, "# Handoff note: issue #24\n")

    assert commit.subject == "Handoff note: issue #24"
    assert (attempt.root / ".agent" / "handoff" / "24.md").read_text() == (
        "# Handoff note: issue #24\n"
    )
    assert head_files(attempt) == (".agent/handoff/24.md",)
    # Other uncommitted work is left alone for the seal commit.
    assert not attempt.workspace.is_clean()


def test_handoff_note_commit_survives_a_seal_commit_on_top(
    attempt: SimpleNamespace,
) -> None:
    model_commit(
        attempt,
        ".agent/handoff/24.md",
        "# Handoff note: issue #24\n",
        "Model progress",
    )
    dirty_file(attempt, "work.txt", "uncommitted edits")
    seal_commit = attempt.workspace.seal(
        issue_number=24,
        attempt_id="2026-09-29T00:00:00Z",
        status=ModelExecutionStatus.HANDOFF_REQUESTED,
    )
    assert seal_commit is not None

    note_commit = attempt.workspace.handoff_note_commit(
        attempt.prepared, 24, attempt_id=attempt.attempt_id
    )

    assert note_commit is not None
    assert note_commit.subject == "Model progress"
    assert note_commit.revision != seal_commit.revision


def test_handoff_note_commit_is_none_without_a_note(
    attempt: SimpleNamespace,
) -> None:
    model_commit(attempt, "work.txt", "work", "Model work")

    assert (
        attempt.workspace.handoff_note_commit(
            attempt.prepared, 24, attempt_id=attempt.attempt_id
        )
        is None
    )


def test_handoff_note_commit_ignores_a_note_from_an_earlier_round(
    attempt: SimpleNamespace,
) -> None:
    """A continued branch keeps earlier rounds' notes; they are not this attempt's."""

    prior_round_commit(
        attempt, ".agent/handoff/24.md", "# Handoff note: issue #24\n", "Handoff note: issue #24"
    )
    model_commit(attempt, "work.txt", "work", "Model work")

    assert (
        attempt.workspace.handoff_note_commit(
            attempt.prepared, 24, attempt_id=attempt.attempt_id
        )
        is None
    )

    note = attempt.workspace.commit_handoff_note(24, "# Handoff note: issue #24\n\nnew\n")

    found = attempt.workspace.handoff_note_commit(
        attempt.prepared, 24, attempt_id=attempt.attempt_id
    )
    assert found is not None
    assert found.revision == note.revision


def test_cleanup_on_a_dirty_tree_stays_on_the_attempt_branch(
    attempt: SimpleNamespace,
) -> None:
    dirty_file(attempt, "work.txt", "uncommitted edits")

    with pytest.raises(DirtyWorkspaceError, match="uncommitted or untracked"):
        attempt.workspace.cleanup(
            base_branch="main", prepared=attempt.prepared, retain_branch=True
        )

    assert current_branch(attempt) == "agent/issue-24"
    assert not attempt.workspace.is_clean()


def test_cleanup_on_a_clean_tree_returns_to_the_base_branch(
    attempt: SimpleNamespace,
) -> None:
    attempt.workspace.cleanup(
        base_branch="main", prepared=attempt.prepared, retain_branch=True
    )

    assert current_branch(attempt) == "main"
