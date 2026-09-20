from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from simple_coding_agent.git_workspace import GitWorkspace, GitWorkspaceError


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


def test_refuses_a_base_branch_that_diverges_from_origin(tmp_path: Path) -> None:
    remote, seed = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "secret-token")
    workspace.prepare_attempt(base_branch="main", issue_number=19)
    git(tmp_path / "clone", "checkout", "main")
    write_and_commit(tmp_path / "clone", "local.txt", "local change", "local base")

    with pytest.raises(GitWorkspaceError, match="diverged"):
        workspace.prepare_attempt(base_branch="main", issue_number=19)

    assert git(seed, "rev-parse", "main") != git(tmp_path / "clone", "rev-parse", "main")


def test_refuses_a_stale_base_after_origin_advances(tmp_path: Path) -> None:
    remote, seed = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    workspace.prepare_attempt(base_branch="main", issue_number=19)
    write_and_commit(seed, "upstream.txt", "upstream change", "Advance base")
    git(seed, "push", "origin", "main")

    with pytest.raises(GitWorkspaceError, match="diverged"):
        workspace.prepare_attempt(base_branch="main", issue_number=19)

    assert git(seed, "rev-parse", "main") != git(clone, "rev-parse", "main")


def test_reports_the_verified_base_and_commits_added_by_the_attempt(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)
    write_and_commit(clone, "change.txt", "change", "Implement workspace")

    commits = workspace.commits_added(prepared)

    assert [commit.subject for commit in commits] == ["Implement workspace"]
    assert commits[0].revision == git(clone, "rev-parse", "HEAD")


def test_cleanup_restores_base_and_preserves_ignored_dependencies(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)
    (clone / ".venv").mkdir()
    (clone / ".venv" / "marker").write_text("keep")
    (clone / "untracked.txt").write_text("discard")

    workspace.cleanup(base_branch="main", prepared=prepared, retain_branch=False)

    assert git(clone, "branch", "--show-current") == "main"
    assert not (clone / "untracked.txt").exists()
    assert (clone / ".venv" / "marker").read_text() == "keep"
    assert "agent/issue-19" not in git(clone, "branch", "--format=%(refname:short)").splitlines()


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
    return subprocess.run(
        command, cwd=cwd, env=env, check=True, text=True, capture_output=True
    ).stdout.strip()
