from __future__ import annotations

import subprocess
from pathlib import Path

from simple_coding_agent.git_workspace import GitWorkspace
from simple_coding_agent.model_execution import ModelExecutionStatus
from tests.fakes import git, repository_with_main, write_and_commit


def test_debug_workspace_finish_with_note_commit(tmp_path: Path) -> None:
    remote, seed = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    workspace.prepare_attempt(base_branch="main", issue_number=19)
    git(clone, "config", "user.name", "Test User")
    git(clone, "config", "user.email", "test@example.com")
    write_and_commit(clone, "README.md", "branch change", "Branch pick one")
    write_and_commit(clone, "feature.txt", "feature work", "Branch pick two")
    write_and_commit(seed, "README.md", "upstream fix", "Upstream commit")
    git(seed, "push", "origin", "main")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=19)
    assert workspace.has_unresolved_conflicts() is True

    (clone / "README.md").write_text("resolved content\n")
    git(clone, "add", "README.md")
    (clone / ".agent" / "handoff").mkdir(parents=True, exist_ok=True)
    (clone / ".agent" / "handoff" / "19.md").write_text("# Handoff note\n")
    git(clone, "add", ".agent/handoff/19.md")
    git(clone, "commit", "-m", "Handoff note: issue #19")
    (clone / ".gitignore").write_text(".venv/\n.env\n")
    print("PORCELAIN:", git(clone, "status", "--porcelain"))
    print("REBASE_HEAD:", end=" ")
    try:
        print(git(clone, "rev-parse", "--verify", "REBASE_HEAD"))
    except Exception as error:  # noqa: BLE001
        print("MISSING", error)

    finished = workspace.finish_paused_rebase(
        prepared,
        issue_number=19,
        attempt_id="attempt-finish-2",
        status=ModelExecutionStatus.HANDOFF_REQUESTED,
    )
    print("FINISHED:", finished)
    print("PORCELAIN2:", git(clone, "status", "--porcelain"))
    proc = subprocess.run(
        ("git", "log", "--oneline", "-8"),
        cwd=clone,
        text=True,
        capture_output=True,
    )
    print("LOG:", proc.stdout)
    proc = subprocess.run(
        ("git", "rebase", "--show-current-patch"),
        cwd=clone,
        text=True,
        capture_output=True,
    )
    print("SHOW-CURRENT rc:", proc.returncode, proc.stdout[-1000:], proc.stderr[-1000:])
