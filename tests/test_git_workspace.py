from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from simple_coding_agent.git_workspace import (
    DirtyWorkspaceError,
    GitWorkspace,
    GitWorkspaceError,
    GitWorkspaceRecoveryError,
)


def test_clones_once_fetches_and_prepares_an_attempt_from_the_verified_base(
    tmp_path: Path,
) -> None:
    remote, seed = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "secret-token")

    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)

    assert prepared.branch == "agent/issue-19"
    assert prepared.base_revision == git(seed, "rev-parse", "main")
    assert git(tmp_path / "clone", "branch", "--show-current") == "agent/issue-19"
    assert git(tmp_path / "clone", "remote", "get-url", "origin") == str(remote)

    workspace.prepare_attempt(base_branch="main", issue_number=19)

    assert git(tmp_path / "clone", "branch", "--show-current") == "agent/issue-19"


def test_repairs_a_base_branch_that_diverges_from_origin_before_the_attempt(
    tmp_path: Path,
) -> None:
    remote, seed = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    workspace.prepare_attempt(base_branch="main", issue_number=19)
    git(clone, "checkout", "main")
    write_and_commit(clone, "local.txt", "local change", "local base")

    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)

    assert prepared.base_revision == git(seed, "rev-parse", "main")
    assert git(clone, "rev-parse", "main") == git(seed, "rev-parse", "main")


def test_repairs_a_stale_base_after_origin_advances(tmp_path: Path) -> None:
    remote, seed = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    workspace.prepare_attempt(base_branch="main", issue_number=19)
    write_and_commit(seed, "upstream.txt", "upstream change", "Advance base")
    git(seed, "push", "origin", "main")

    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)

    assert prepared.base_revision == git(seed, "rev-parse", "main")
    assert git(clone, "rev-parse", "main") == git(seed, "rev-parse", "main")


def test_raises_a_recovery_error_when_the_diverged_base_cannot_be_repaired(
    tmp_path: Path,
) -> None:
    remote, seed = repository_with_main(tmp_path)
    clone = tmp_path / "clone"

    def failing_reset(command: tuple[str, ...], cwd: Path | None, env: dict[str, str]) -> str:
        if command[1] == "branch" and command[2] == "-f":
            raise subprocess.CalledProcessError(1, command)
        return run_command(command, cwd=cwd, env=env)

    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    workspace.prepare_attempt(base_branch="main", issue_number=19)
    write_and_commit(seed, "upstream.txt", "upstream change", "Advance base")
    git(seed, "push", "origin", "main")

    broken_workspace = GitWorkspace(
        clone, str(remote), token_provider=lambda: "secret-token", run=failing_reset
    )

    with pytest.raises(GitWorkspaceRecoveryError, match="diverged"):
        broken_workspace.prepare_attempt(base_branch="main", issue_number=19)


def test_rebases_a_reused_attempt_branch_onto_an_advanced_base(tmp_path: Path) -> None:
    remote, seed = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    workspace.prepare_attempt(base_branch="main", issue_number=19)
    write_and_commit(clone, "local.txt", "unpublished work", "Retain me")

    write_and_commit(seed, "README.md", "fixed setup command", "Fix setup command")
    git(seed, "push", "origin", "main")

    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)

    assert (clone / "README.md").read_text() == "fixed setup command"
    assert [commit.subject for commit in workspace.commits_added(prepared)] == ["Retain me"]


def test_leaves_a_conflicting_rebase_in_place_for_the_model_to_resolve(
    tmp_path: Path,
) -> None:
    """A conflicting rebase is left in progress instead of aborted.

    The model session gets a chance to resolve the conflict as the first
    step of its normal session; aborting happens only later, when the model
    leaves the conflict unresolved.
    """

    remote, seed = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    workspace.prepare_attempt(base_branch="main", issue_number=19)
    write_and_commit(clone, "README.md", "branch change", "Conflicting branch commit")
    pre_rebase_revision = git(clone, "rev-parse", "agent/issue-19")

    write_and_commit(seed, "README.md", "upstream fix", "Conflicting upstream commit")
    git(seed, "push", "origin", "main")

    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)

    assert prepared.branch == "agent/issue-19"
    assert prepared.rebase_conflicts == ("README.md",)
    assert prepared.pre_rebase_revision == pre_rebase_revision
    # The rebase is still in progress and the markers are visible ...
    assert workspace.has_unresolved_conflicts() is True
    assert "<<<<<<<" in (clone / "README.md").read_text()
    assert workspace.conflicted_files() == ("README.md",)
    # ... while the branch ref still points at the pre-rebase tip.
    assert git(clone, "rev-parse", "agent/issue-19") == pre_rebase_revision


def test_clean_rebase_reports_no_conflicts_and_no_pre_rebase_revision(
    tmp_path: Path,
) -> None:
    remote, seed = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    workspace.prepare_attempt(base_branch="main", issue_number=19)
    write_and_commit(clone, "local.txt", "unpublished work", "Retain me")

    write_and_commit(seed, "README.md", "upstream fix", "Upstream commit")
    git(seed, "push", "origin", "main")

    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)

    assert prepared.rebase_conflicts == ()
    assert prepared.pre_rebase_revision is None
    assert workspace.has_unresolved_conflicts() is False
    assert workspace.rebase_resolution_problems() == ()


def test_resolution_problems_are_empty_once_the_model_resolves_the_rebase(
    tmp_path: Path,
) -> None:
    """Simulate a model resolving the conflicted rebase, then verify clean."""

    remote, seed = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    workspace.prepare_attempt(base_branch="main", issue_number=19)
    write_and_commit(clone, "README.md", "branch change", "Conflicting branch commit")

    write_and_commit(seed, "README.md", "upstream fix", "Conflicting upstream commit")
    git(seed, "push", "origin", "main")

    workspace.prepare_attempt(base_branch="main", issue_number=19)

    assert workspace.rebase_resolution_problems() != ()

    (clone / "README.md").write_text("resolved content")
    git(clone, "add", "README.md")
    git(
        clone,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.com",
        "rebase",
        "--continue",
    )

    assert workspace.has_unresolved_conflicts() is False
    assert workspace.conflicted_files() == ()
    assert workspace.conflict_marker_files() == ()
    assert workspace.diff_check_clean() is True
    assert workspace.rebase_resolution_problems() == ()


def test_resolution_problems_report_markers_and_whitespace_errors(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    workspace.prepare_attempt(base_branch="main", issue_number=19)

    assert workspace.conflict_marker_files() == ()
    assert workspace.diff_check_clean() is True

    write_and_commit(clone, "notes.txt", "clean content", "Add notes")
    with (clone / "notes.txt").open("a", encoding="utf-8") as handle:
        handle.write("<<<<<<< HEAD\nstale marker\n=======\nother side\n>>>>>>> branch\n")
    with (clone / "other.txt").open("w", encoding="utf-8") as handle:
        handle.write("trailing whitespace \n")
    with (clone / "longer.txt").open("w", encoding="utf-8") as handle:
        handle.write("stale center marker with extra equals\n==========\n")

    assert workspace.conflict_marker_files() == ("notes.txt", "longer.txt")
    git(clone, "add", "-A")
    assert workspace.diff_check_clean() is False
    problems = workspace.rebase_resolution_problems()
    assert any("notes.txt" in problem for problem in problems)
    assert any("diff --check" in problem for problem in problems)


def test_abort_unresolved_rebase_restores_the_pre_rebase_branch_state(
    tmp_path: Path,
) -> None:
    """The fallback after an unresolvable conflict matches the old behavior."""

    remote, seed = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    workspace.prepare_attempt(base_branch="main", issue_number=19)
    write_and_commit(clone, "README.md", "branch change", "Conflicting branch commit")
    pre_rebase_revision = git(clone, "rev-parse", "agent/issue-19")

    write_and_commit(seed, "README.md", "upstream fix", "Conflicting upstream commit")
    git(seed, "push", "origin", "main")

    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)

    workspace.abort_unresolved_rebase(prepared)

    assert "<<<<<<<" not in (clone / "README.md").read_text()
    assert git(clone, "status", "--porcelain") == ""
    with pytest.raises(subprocess.CalledProcessError):
        git(clone, "rev-parse", "--verify", "REBASE_HEAD")
    assert (clone / "README.md").read_text() == "branch change"
    assert git(clone, "rev-parse", "agent/issue-19") == pre_rebase_revision
    assert workspace.has_unresolved_conflicts() is False
    assert workspace.is_clean()


def test_abort_unresolved_rebase_restores_after_rebase_quit(tmp_path: Path) -> None:
    """Even a rebase state the model dropped with --quit is restored."""

    remote, seed = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    workspace.prepare_attempt(base_branch="main", issue_number=19)
    write_and_commit(clone, "README.md", "branch change", "Conflicting branch commit")
    pre_rebase_revision = git(clone, "rev-parse", "agent/issue-19")

    write_and_commit(seed, "README.md", "upstream fix", "Conflicting upstream commit")
    git(seed, "push", "origin", "main")

    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)
    git(clone, "rebase", "--quit")

    assert workspace.has_unresolved_conflicts() is True

    workspace.abort_unresolved_rebase(prepared)

    assert workspace.has_unresolved_conflicts() is False
    assert git(clone, "rev-parse", "agent/issue-19") == pre_rebase_revision
    assert (clone / "README.md").read_text() == "branch change"


def test_reports_no_unresolved_conflicts_on_a_clean_worktree(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "secret-token")

    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)

    assert workspace.has_unresolved_conflicts() is False
    assert workspace.is_clean()


def test_cleanup_aborts_an_unresolved_rebase_left_by_the_model_session(tmp_path: Path) -> None:
    remote, seed = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)
    write_and_commit(clone, "README.md", "branch change", "Conflicting branch commit")

    write_and_commit(seed, "README.md", "upstream fix", "Conflicting upstream commit")
    git(seed, "push", "origin", "main")
    git(clone, "fetch", "origin")
    # Simulate the model leaving an unresolved rebase behind: start the
    # rebase directly instead of going through prepare_attempt (which leaves
    # the same state in place when the reused branch conflicts with base).
    with pytest.raises(subprocess.CalledProcessError):
        git(clone, "rebase", "origin/main")
    git(clone, "rev-parse", "--verify", "REBASE_HEAD")

    workspace.cleanup(base_branch="main", prepared=prepared, retain_branch=False)

    assert git(clone, "branch", "--show-current") == "main"
    assert git(clone, "status", "--porcelain") == ""
    with pytest.raises(subprocess.CalledProcessError):
        git(clone, "rev-parse", "--verify", "REBASE_HEAD")


def test_resumes_a_published_handoff_branch_on_a_fresh_clone(tmp_path: Path) -> None:
    remote, seed = repository_with_main(tmp_path)
    published = tmp_path / "published"
    git(tmp_path, "clone", str(remote), str(published))
    git(published, "config", "user.name", "Test User")
    git(published, "config", "user.email", "test@example.com")
    git(published, "checkout", "-b", "agent/issue-19")
    write_and_commit(published, "handoff.txt", "handoff work", "Handoff note")
    git(published, "push", "-u", "origin", "agent/issue-19")

    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")

    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)

    assert prepared.branch == "agent/issue-19"
    assert prepared.restored_from_remote is True
    assert (clone / "handoff.txt").read_text() == "handoff work"
    assert [commit.subject for commit in workspace.commits_added(prepared)] == ["Handoff note"]


def test_reports_restored_from_remote_as_false_with_no_published_branch(tmp_path: Path) -> None:
    remote, seed = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "secret-token")

    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)

    assert prepared.restored_from_remote is False


def test_prefers_the_published_branch_over_diverged_local_state(tmp_path: Path) -> None:
    remote, seed = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    workspace.prepare_attempt(base_branch="main", issue_number=19)
    write_and_commit(clone, "local-only.txt", "should be discarded", "Local-only commit")

    published = tmp_path / "published"
    git(tmp_path, "clone", str(remote), str(published))
    git(published, "config", "user.name", "Test User")
    git(published, "config", "user.email", "test@example.com")
    git(published, "checkout", "-b", "agent/issue-19")
    write_and_commit(published, "handoff.txt", "handoff work", "Handoff note")
    git(published, "push", "-u", "origin", "agent/issue-19")

    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)

    assert (clone / "handoff.txt").read_text() == "handoff work"
    assert not (clone / "local-only.txt").exists()
    assert [commit.subject for commit in workspace.commits_added(prepared)] == ["Handoff note"]


def test_reports_the_verified_base_and_commits_added_by_the_attempt(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)
    write_and_commit(clone, "change.txt", "change", "Implement workspace")

    commits = workspace.commits_added(prepared)

    assert [commit.subject for commit in commits] == ["Implement workspace"]
    assert commits[0].revision == git(clone, "rev-parse", "HEAD")


def test_commit_dirty_work_preserves_untracked_and_modified_changes(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)
    (clone / "untracked.txt").write_text("dirty work")

    committed = workspace.commit_dirty_work("Preserve uncommitted work before handoff")

    assert committed is True
    commits = workspace.commits_added(prepared)
    assert [commit.subject for commit in commits] == ["Preserve uncommitted work before handoff"]
    assert git(clone, "status", "--porcelain") == ""


def test_commit_dirty_work_is_a_noop_on_a_clean_worktree(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)

    committed = workspace.commit_dirty_work("Preserve uncommitted work before handoff")

    assert committed is False
    assert workspace.commits_added(prepared) == ()


def test_cleanup_preserves_untracked_files_and_attempt_branch(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)
    (clone / ".venv").mkdir()
    (clone / ".venv" / "marker").write_text("keep")
    (clone / "untracked.txt").write_text("preserve")

    workspace.cleanup(base_branch="main", prepared=prepared, retain_branch=False)

    assert (clone / "untracked.txt").exists()
    assert (clone / ".venv" / "marker").read_text() == "keep"
    assert "agent/issue-19" in git(clone, "branch", "--format=%(refname:short)").splitlines()
    with pytest.raises(DirtyWorkspaceError):
        workspace.prepare_attempt(base_branch="main", issue_number=20)


def test_cleanup_retains_unpublished_commits_when_requested(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)
    write_and_commit(clone, "change.txt", "change", "Retain me")

    workspace.cleanup(base_branch="main", prepared=prepared, retain_branch=True)

    assert git(clone, "branch", "--show-current") == "main"
    git(clone, "rev-parse", "agent/issue-19")
    assert [commit.subject for commit in workspace.commits_added(prepared)] == ["Retain me"]

    reused = workspace.prepare_attempt(base_branch="main", issue_number=19)

    assert reused.branch == prepared.branch
    assert [commit.subject for commit in workspace.commits_added(reused)] == ["Retain me"]


def test_pushes_with_an_explicit_empty_lease_only_after_confirming_remote_absence(
    tmp_path: Path,
) -> None:
    remote, _ = repository_with_main(tmp_path)
    commands: list[tuple[str, ...]] = []

    def recording_run(command: tuple[str, ...], cwd: Path | None, env: dict[str, str]) -> str:
        commands.append(command)
        return run_command(command, cwd=cwd, env=env)

    workspace = GitWorkspace(
        tmp_path / "clone", str(remote), token_provider=lambda: "token", run=recording_run
    )
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement feature")

    workspace.push_attempt_branch(prepared.branch, max_retries=1)

    push = next(command for command in commands if command[1] == "push")
    assert push[2] == "--force-with-lease=agent/issue-23:"
    assert ("git", "fetch", "origin", "agent/issue-23") in commands
    assert ("git", "ls-remote", "--heads", "origin", "refs/heads/agent/issue-23") in commands


def test_verifies_an_ambiguous_push_before_retrying(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    push_calls = 0

    def ambiguous_run(command: tuple[str, ...], cwd: Path | None, env: dict[str, str]) -> str:
        nonlocal push_calls
        result = run_command(command, cwd=cwd, env=env)
        if command[1] == "push":
            push_calls += 1
            raise subprocess.CalledProcessError(1, command)
        return result

    workspace = GitWorkspace(
        tmp_path / "clone", str(remote), token_provider=lambda: "token", run=ambiguous_run,
        sleeper=lambda _: None,
    )
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement feature")

    pushed = workspace.push_attempt_branch(prepared.branch, max_retries=3)

    assert pushed == git(tmp_path / "clone", "rev-parse", "HEAD")
    assert push_calls == 1


def test_git_operations_resolve_credentials_with_askpass_without_persisting_them(
    tmp_path: Path,
) -> None:
    remote, _ = repository_with_main(tmp_path)
    environments: list[dict[str, str]] = []
    askpass_values: list[str] = []
    token = "secret-token"

    def recording_run(command: tuple[str, ...], cwd: Path | None, env: dict[str, str]) -> str:
        environments.append(env)
        askpass_values.append(
            run_command(
                (env["GIT_ASKPASS"], "Password for https://github.com:"),
                env=env,
            )
        )
        return run_command(command, cwd=cwd, env=env)

    workspace = GitWorkspace(
        tmp_path / "clone", str(remote), token_provider=lambda: token, run=recording_run
    )
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)
    token = "rotated-token"
    workspace.commits_added(prepared)

    assert environments
    assert all(environment["GITHUB_TOKEN"] == "secret-token" for environment in environments[:-1])
    assert environments[-1]["GITHUB_TOKEN"] == "rotated-token"
    assert all(environment["GIT_TERMINAL_PROMPT"] == "0" for environment in environments)
    assert all(os.path.isfile(environment["GIT_ASKPASS"]) is False for environment in environments)
    assert askpass_values[:-1] == ["secret-token"] * (len(askpass_values) - 1)
    assert askpass_values[-1] == "rotated-token"
    assert "secret-token" not in git(tmp_path / "clone", "config", "--local", "--list")


def test_refuses_a_configured_repository_url_with_embedded_credentials(tmp_path: Path) -> None:
    workspace = GitWorkspace(
        tmp_path / "clone",
        "https://x-access-token:secret-token@github.com/octo/example.git",
        token_provider=lambda: "secret-token",
    )

    with pytest.raises(GitWorkspaceError, match="credentials"):
        workspace.prepare_attempt(base_branch="main", issue_number=19)


def test_refuses_reusing_a_clone_for_a_different_repository(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    other_remote, _ = repository_with_main(tmp_path / "other")
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    workspace.prepare_attempt(base_branch="main", issue_number=19)
    git(clone, "remote", "set-url", "origin", str(other_remote))

    with pytest.raises(GitWorkspaceError, match="configured repository"):
        workspace.prepare_attempt(base_branch="main", issue_number=19)


def test_profile_bootstrap_never_rebases_an_attempt_branch_onto_the_wrong_base(
    tmp_path: Path,
) -> None:
    """Reading the profile must not touch attempt branches.

    With ``develop`` diverged from ``main`` and a remote attempt branch built
    on ``develop``, bootstrapping the workspace for the profile read must
    succeed without rebasing anything; the single real preparation onto
    ``develop`` is then a clean no-op rebase instead of a conflict.
    """

    remote, seed = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")

    git(seed, "checkout", "-b", "develop")
    write_and_commit(seed, "README.md", "develop content", "Develop work")
    git(seed, "push", "-u", "origin", "develop")
    git(seed, "checkout", "main")
    write_and_commit(seed, "README.md", "main content", "Main work")
    git(seed, "push", "origin", "main")

    git(seed, "checkout", "develop")
    git(seed, "checkout", "-b", "agent/issue-38")
    write_and_commit(seed, "feature.txt", "attempt work", "Attempt work")
    git(seed, "push", "-u", "origin", "agent/issue-38")

    workspace.prepare_for_profile_read()

    assert git(clone, "branch", "--show-current") == "main"
    assert "agent/issue-38" not in git(clone, "branch", "--format=%(refname:short)").splitlines()

    prepared = workspace.prepare_attempt(base_branch="develop", issue_number=38)

    assert prepared.branch == "agent/issue-38"
    assert (clone / "feature.txt").read_text() == "attempt work"
    assert (clone / "README.md").read_text() == "develop content"
    assert [commit.subject for commit in workspace.commits_added(prepared)] == ["Attempt work"]


def repository_with_main(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    git(tmp_path, "init", "--bare", str(remote))
    git(tmp_path, "init", str(seed))
    git(seed, "checkout", "-b", "main")
    git(seed, "config", "user.name", "Test User")
    git(seed, "config", "user.email", "test@example.com")
    write_and_commit(seed, "README.md", "seed", "Initial commit")
    write_and_commit(seed, ".gitignore", ".venv/\n", "Ignore environment")
    git(seed, "remote", "add", "origin", str(remote))
    git(seed, "push", "-u", "origin", "main")
    git(remote, "symbolic-ref", "HEAD", "refs/heads/main")
    return remote, seed


def write_and_commit(repository: Path, name: str, contents: str, message: str) -> None:
    (repository / name).write_text(contents)
    git(repository, "add", name)
    git(repository, "commit", "-m", message)


def git(cwd: Path, *arguments: str) -> str:
    return run_command(("git", *arguments), cwd=cwd)


def run_command(
    command: tuple[str, ...], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> str:
    if env is None:
        # Never let a git subcommand (e.g. "rebase --continue") fall back to
        # the caller's real $EDITOR: on a real terminal that spawns an
        # interactive editor attached to it and hangs the test run.
        env = {**os.environ, "GIT_EDITOR": "true", "GIT_SEQUENCE_EDITOR": "true"}
    return subprocess.run(
        command, cwd=cwd, env=env, check=True, text=True, capture_output=True
    ).stdout.strip()
