from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from simple_coding_agent.completion import AttemptOutcome, CompletionDecision, PublicationPath
from simple_coding_agent.git_workspace import GitWorkspace
from simple_coding_agent.publication import PublicationRequest, Publisher, PullRequest

from test_git_workspace import git, repository_with_main, write_and_commit


def test_publishes_a_complete_attempt_once_and_renders_the_standard_result(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub()

    result = Publisher(workspace, github, "octo/example").publish(
        PublicationRequest(
            issue_number=23,
            issue_title="Implement publication",
            branch=prepared.branch,
            started_at="2026-09-20T13:00:00Z",
            decision=CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)),
            check_command="pytest",
            check_exit_code=0,
            review_cycles=1,
            review_findings="all clear",
            details="Implemented the publication boundary.",
            base_branch="main",
        )
    )

    assert result.outcome is AttemptOutcome.COMPLETE
    assert result.pull_request == PullRequest(1, "https://example.test/pulls/1")
    assert git(tmp_path / "clone", "ls-remote", "origin", "refs/heads/agent/issue-23")
    assert github.created == [("octo/example", "agent/issue-23", "main", "agent: Implement publication")]
    assert "Closes #23" in github.pull_request_bodies[0]
    assert "## Agent Attempt Result: complete" in github.comments[0][1]
    assert "<!-- agent-attempt: 2026-09-20T13:00:00Z -->" in github.comments[0][1]

    repeated = Publisher(workspace, github, "octo/example").publish(
        PublicationRequest(
            issue_number=23,
            issue_title="Implement publication",
            branch=prepared.branch,
            started_at="2026-09-20T13:00:00Z",
            decision=CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)),
            check_command="pytest",
            check_exit_code=0,
            review_cycles=1,
            review_findings="all clear",
            details="Implemented the publication boundary.",
            base_branch="main",
        )
    )
    assert repeated.outcome is AttemptOutcome.COMPLETE
    assert len(github.comments) == 1
    assert len(github.created) == 1


def test_publishes_incomplete_commits_without_creating_a_pull_request(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "partial.txt", "partial", "Partial work")
    github = FakeGitHub()

    result = Publisher(workspace, github, "octo/example").publish(
        request(CompletionDecision(AttemptOutcome.INCOMPLETE, False, PublicationPath.PARTIAL, ("check failed",)), prepared.branch)
    )

    assert result.outcome is AttemptOutcome.INCOMPLETE
    assert result.pull_request is None
    assert not github.created
    assert result.branch_url == "https://github.com/octo/example/tree/agent/issue-23"


def test_no_changes_does_not_push_or_create_a_pull_request(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    github = FakeGitHub()

    result = Publisher(workspace, github, "octo/example").publish(
        request(CompletionDecision(AttemptOutcome.NO_CHANGES, False, PublicationPath.NONE, ("no commits",)), prepared.branch)
    )

    assert result.outcome is AttemptOutcome.NO_CHANGES
    assert result.branch_url is None
    assert not github.created
    assert not git(tmp_path / "clone", "ls-remote", "origin", "refs/heads/agent/issue-23")


def test_verifies_a_pull_request_created_before_an_ambiguous_response(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub(fail_creation_after_write=True)

    result = Publisher(workspace, github, "octo/example").publish(
        request(CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)), prepared.branch)
    )

    assert result.outcome is AttemptOutcome.COMPLETE
    assert result.pull_request == PullRequest(1, "https://example.test/pulls/1")
    assert len(github.created) == 1


def request(decision: CompletionDecision, branch: str) -> PublicationRequest:
    return PublicationRequest(23, "Implement publication", branch, "2026-09-20T13:00:00Z", decision, "pytest", 1, 0, "all clear", "Check did not pass.", "main")


@dataclass
class FakeGitHub:
    pull_request: PullRequest | None = None
    fail_creation_after_write: bool = False

    def __post_init__(self) -> None:
        self.created: list[tuple[str, str, str, str]] = []
        self.pull_request_bodies: list[str] = []
        self.comments: list[tuple[int, str]] = []

    def find_pull_request(self, repository: str, head: str) -> PullRequest | None:
        return self.pull_request

    def create_pull_request(self, repository: str, head: str, base: str, title: str, body: str) -> PullRequest:
        self.created.append((repository, head, base, title))
        self.pull_request_bodies.append(body)
        self.pull_request = PullRequest(1, "https://example.test/pulls/1")
        if self.fail_creation_after_write:
            raise OSError("connection closed after write")
        return self.pull_request

    def find_attempt_comment(self, repository: str, issue_number: int, marker: str) -> bool:
        return any(number == issue_number and marker in body for number, body in self.comments)

    def add_comment(self, repository: str, issue_number: int, body: str) -> None:
        self.comments.append((issue_number, body))
