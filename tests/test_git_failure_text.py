"""Git failure text stays out of issue comments (issue #397).

A failed git command carries its operation, exit code and redacted, bounded
output as structured data. Only a short, sanitised one-line summary reaches
an issue comment; the full redacted output goes to WARNING/ERROR events.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from simple_coding_agent.config import RepositoryProfile
from simple_coding_agent.git_workspace import (
    GitCommandError,
    GitWorkspace,
    GitWorkspaceError,
    git_failure_detail,
    git_failure_summary,
)
from simple_coding_agent.github_tracker import GitHubGraphQLTransport
from tests.fakes import (
    FakeGitHub,
    FakeModelExecutor,
    build_lifecycle,
    git,
    lifecycle_issue as issue,
    repository_with_main,
    run_command,
    write_and_commit,
)

GITHUB_TOKEN = "ghp_github_secret_value"
META_API_KEY = "meta-secret-value"
MODEL_API_KEY = "sk-model-secret-value"
CREDENTIALS = (GITHUB_TOKEN, META_API_KEY, MODEL_API_KEY)
MARKER = "<!-- conflict-attempt: success -->"


def hostile_stderr() -> str:
    return (
        f"remote: model key {MODEL_API_KEY} and meta key {META_API_KEY}\n"
        f"remote: token {GITHUB_TOKEN} in /workspace/clone/secret.txt\n"
        f"remote: ping @someone {MARKER} `code` -->\n"
        "\n"
        f"fatal: unable to push for @someone {MARKER} with {GITHUB_TOKEN} `x`\n"
    )


def failing_git(operation: str, stderr: str):
    def run(command: tuple[str, ...], cwd: Path | None, env: dict[str, str]) -> str:
        if command[1] == operation:
            raise subprocess.CalledProcessError(128, command, stderr=stderr)
        return run_command(command, cwd=cwd, env=env)

    return run


def test_failed_git_command_carries_structured_redacted_output(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(
        tmp_path / "clone",
        str(remote),
        token_provider=lambda: GITHUB_TOKEN,
        redactions=CREDENTIALS,
        run=failing_git("status", hostile_stderr()),
    )
    (tmp_path / "clone").mkdir()
    (tmp_path / "clone" / ".git").mkdir()

    with pytest.raises(GitCommandError) as raised:
        workspace.dirty_status()

    error = raised.value
    assert isinstance(error, GitWorkspaceError)
    assert str(error) == "Git status failed (exit code 128)"
    assert (error.operation, error.exit_code) == ("status", 128)
    assert "unable to push" in error.output
    assert "[REDACTED]" in error.output
    assert all(secret not in error.output for secret in CREDENTIALS)
    assert error.__cause__ is None


def test_redaction_happens_before_the_output_bound(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    stderr = "x" * 1995 + MODEL_API_KEY
    workspace = GitWorkspace(
        tmp_path / "clone",
        str(remote),
        token_provider=lambda: GITHUB_TOKEN,
        redactions=CREDENTIALS,
        run=failing_git("clone", stderr),
    )

    with pytest.raises(GitCommandError) as raised:
        workspace.prepare_for_profile_read()

    assert len(raised.value.output) == 2000
    assert MODEL_API_KEY[:5] not in raised.value.output


def test_git_that_cannot_start_has_a_short_form_without_exit_code(tmp_path: Path) -> None:
    def missing(command: tuple[str, ...], cwd: Path | None, env: dict[str, str]) -> str:
        raise FileNotFoundError(2, f"No such file: git {MODEL_API_KEY}")

    workspace = GitWorkspace(
        tmp_path / "clone",
        "https://example.test/repo.git",
        token_provider=lambda: GITHUB_TOKEN,
        redactions=CREDENTIALS,
        run=missing,
    )

    with pytest.raises(GitCommandError) as raised:
        workspace.prepare_for_profile_read()

    assert str(raised.value) == "Git clone could not start"
    assert raised.value.exit_code is None
    assert MODEL_API_KEY not in raised.value.output
    assert "No such file" in raised.value.output


def test_summary_is_one_neutralised_bounded_line() -> None:
    error = GitCommandError("push", 128, "first line\n" + "@someone `x` <!-- marker --> " + "y" * 400)

    summary = git_failure_summary(error)

    assert summary.startswith("GitCommandError: Git push failed (exit code 128): `")
    assert "\n" not in summary
    assert "first line" not in summary
    code = summary.split(": `", 1)[1]
    assert code.endswith("`")
    assert code.count("`") == 1
    assert len(code) <= 202
    assert "@" not in summary
    assert "<!--" not in summary and "-->" not in summary


def test_summary_of_a_wrapping_error_names_the_underlying_git_failure() -> None:
    try:
        try:
            raise GitCommandError("push", 1, "remote: rejected\nfatal: hung up")
        except GitCommandError as cause:
            raise GitWorkspaceError("Attempt branch could not be pushed safely") from cause
    except GitWorkspaceError as error:
        summary = git_failure_summary(error)
        detail = git_failure_detail(error)

    assert summary == (
        "GitWorkspaceError: Attempt branch could not be pushed safely;"
        " Git push failed (exit code 1): `fatal: hung up`"
    )
    assert "remote: rejected\nfatal: hung up" in detail
    assert "Git push failed (exit code 1)" in detail


def test_summary_of_a_non_git_error_is_its_type_and_message() -> None:
    assert git_failure_summary(TimeoutError("PUBLISH_TIMEOUT")) == "TimeoutError: PUBLISH_TIMEOUT"
    assert git_failure_detail(TimeoutError("PUBLISH_TIMEOUT")) == "TimeoutError: PUBLISH_TIMEOUT"


def incomplete_profile(_: Path) -> RepositoryProfile:
    return RepositoryProfile(
        # The baseline passes; the final check fails once the work exists, so
        # the attempt publishes partial work and the push failure is reported.
        setup=(),
        check=("test ! -f work.txt",),
        base_branch="main",
        timeout=30,
        setup_timeout=30,
        env={},
    )


def test_push_failure_comment_holds_only_a_sanitised_summary_and_the_event_the_output(
    tmp_path: Path,
) -> None:
    remote, _ = repository_with_main(tmp_path)
    events: list[tuple[str, str, str]] = []
    github = FakeGitHub(issue(23))
    lifecycle = build_lifecycle(
        remote,
        tmp_path / "clone",
        tmp_path / "data",
        github,
        FakeModelExecutor(
            actions=[lambda wd: write_and_commit(wd, "work.txt", "partial", "Partial work")]
        ),
        profile_fn=incomplete_profile,
        workspace_redactions=CREDENTIALS,
        workspace_run=failing_git("push", hostile_stderr()),
        event_log=lambda event, detail="", level="INFO", **_: events.append(
            (event, detail, level)
        ),
    )

    lifecycle.run_once()

    result_comments = [c.body for c in github.comments if "Agent Attempt Result" in c.body]
    assert len(result_comments) == 1
    comment = result_comments[0]
    assert "Partial branch push failed" in comment
    assert all(secret not in comment for secret in CREDENTIALS)
    assert "@someone" not in comment
    assert MARKER not in comment
    assert "<!-- conflict" not in comment
    assert "/workspace/clone/secret.txt" not in comment
    push_line = next(line for line in comment.splitlines() if "push failed" in line)
    git_output = push_line.split(": `", 1)[1]
    assert "unable to push" in git_output
    assert len(git_output) <= 202
    failures = [(detail, level) for event, detail, level in events if event == "publication_failed"]
    assert len(failures) == 1
    detail, level = failures[0]
    assert level == "ERROR"
    assert "/workspace/clone/secret.txt" in detail
    assert "[REDACTED]" in detail
    assert all(secret not in detail for secret in CREDENTIALS)
    assert git(remote, "branch", "--list", "agent/issue-23") == ""


def test_seal_failure_reason_uses_the_short_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote, _ = repository_with_main(tmp_path)
    github = FakeGitHub(issue(23))
    lifecycle = build_lifecycle(
        remote,
        tmp_path / "clone",
        tmp_path / "data",
        github,
        FakeModelExecutor(actions=[lambda wd: (wd / "work.txt").write_text("dirty")]),
        workspace_redactions=CREDENTIALS,
    )

    def seal_fails(**_: object) -> None:
        raise GitCommandError("commit", 1, f"line one\nfatal: @someone {MARKER} broke")

    monkeypatch.setattr(lifecycle._workspace, "seal", seal_fails)

    lifecycle.run_once()

    comment = next(c.body for c in github.comments if "Agent Attempt Result" in c.body)
    assert "Failed to seal attempt work after model execution" in comment
    assert "Git commit failed (exit code 1)" in comment
    assert "line one" not in comment
    assert "@someone" not in comment
    assert MARKER not in comment


@pytest.mark.parametrize("method", ["add_comment", "create_handoff_record", "update_handoff_record"])
def test_every_comment_body_is_redacted_before_it_is_sent(method: str) -> None:
    transport = GitHubGraphQLTransport(GITHUB_TOKEN, redactions=CREDENTIALS)
    sent: list[str] = []

    def execute(query: str, variables: dict[str, object]) -> dict:
        if "body" in variables:
            sent.append(str(variables["body"]))
            return {"addComment": {"commentEdge": {"node": {"id": "comment-1"}}}}
        return {"repository": {"issue": {"id": "issue-id"}}}

    transport._execute = execute  # type: ignore[method-assign]
    body = f"model said {MODEL_API_KEY}; meta {META_API_KEY}; token {GITHUB_TOKEN}"

    if method == "update_handoff_record":
        transport.update_handoff_record("octo/example", "comment-1", body)
    else:
        getattr(transport, method)("octo/example", 23, body)

    assert sent == ["model said [REDACTED]; meta [REDACTED]; token [REDACTED]"]


def test_long_output_keeps_the_final_line_for_the_comment(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    stderr = "remote: noise\n" * 400 + "fatal: the decisive reason\n"
    workspace = GitWorkspace(
        tmp_path / "clone",
        str(remote),
        token_provider=lambda: GITHUB_TOKEN,
        run=failing_git("clone", stderr),
    )

    with pytest.raises(GitCommandError) as raised:
        workspace.prepare_for_profile_read()

    assert len(raised.value.output) == 2000
    assert git_failure_summary(raised.value).endswith(": `fatal: the decisive reason`")
