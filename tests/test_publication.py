from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from simple_coding_agent.completion import AttemptOutcome, CompletionDecision, PublicationPath
from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.git_workspace import GitWorkspace, GitWorkspaceError
from simple_coding_agent.publication import PublicationRequest, Publisher, PullRequest

from test_git_workspace import git, repository_with_main, write_and_commit


def test_publishes_a_complete_attempt_once_and_renders_the_standard_result(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub()
    state = publishing_state(tmp_path)

    result = Publisher(workspace, github, "octo/example", attempt_state=state).publish(
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
    assert result.pull_request == PullRequest(1, "https://example.test/pulls/1", "pr-node-1")
    assert git(tmp_path / "clone", "ls-remote", "origin", "refs/heads/agent/issue-23")
    assert github.created == [("octo/example", "agent/issue-23", "main", "agent: Implement publication")]
    assert "Closes #23" in github.pull_request_bodies[0]
    assert "## Agent Attempt Result: complete" in github.comments[0][1]
    assert "<!-- agent-attempt: 2026-09-20T13:00:00Z -->" in github.comments[0][1]
    assert state.read().phase is AttemptPhase.PUBLISHING

    repeated = Publisher(workspace, github, "octo/example", attempt_state=state).publish(
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


def test_posts_a_distinct_result_for_a_later_attempt_on_the_same_issue(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    github = FakeGitHub()
    publisher = Publisher(workspace, github, "octo/example")
    decision = CompletionDecision(AttemptOutcome.NO_CHANGES, False, PublicationPath.NONE, ("no commits",))

    publisher.publish(request(decision, prepared.branch))
    publisher.publish(
        PublicationRequest(
            23, "Implement publication", prepared.branch, "2026-09-20T14:00:00Z", decision,
            "pytest", 1, 0, "all clear", "Check did not pass.", "main",
        )
    )

    assert len(github.comments) == 2
    assert "2026-09-20T13:00:00Z" in github.comments[0][1]
    assert "2026-09-20T14:00:00Z" in github.comments[1][1]


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
    assert result.pull_request == PullRequest(1, "https://example.test/pulls/1", "pr-node-1")
    assert len(github.created) == 1


def test_per_call_timeout_overrides_the_configured_publish_timeout(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "partial.txt", "partial", "Partial work")
    github = FakeGitHub()

    result = Publisher(
        workspace,
        github,
        "octo/example",
        publish_timeout=120,
        monotonic=lambda: 50.0,
    ).publish(
        request(CompletionDecision(AttemptOutcome.INCOMPLETE, False, PublicationPath.PARTIAL, ("check failed",)), prepared.branch),
        timeout=0,
    )

    assert result.outcome is AttemptOutcome.INCOMPLETE
    assert "PUBLISH_TIMEOUT exceeded" in result.details
    assert not git(tmp_path / "clone", "ls-remote", "origin", "refs/heads/agent/issue-23")


def test_enforces_the_shared_publish_timeout_across_pull_request_work(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    state = publishing_state(tmp_path)
    state.transition(AttemptPhase.PUBLISHING)
    moments = iter((0.0, 120.0))

    result = Publisher(
        workspace,
        FakeGitHub(),
        "octo/example",
        attempt_state=state,
        publish_timeout=120,
        monotonic=lambda: next(moments),
    ).publish(request(CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)), prepared.branch))

    assert result.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR


def test_partial_push_failure_keeps_incomplete_and_reports_both_reasons(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "partial.txt", "partial", "Partial work")
    github = FakeGitHub()
    events: list[tuple[str, str, str, int | None]] = []

    result = Publisher(
        PushFailingWorkspace(),
        github,
        "octo/example",
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level, issue_number)
        ),
    ).publish(
        PublicationRequest(
            23,
            "Implement publication",
            prepared.branch,
            "2026-09-20T13:00:00Z",
            CompletionDecision(
                AttemptOutcome.INCOMPLETE,
                False,
                PublicationPath.PARTIAL,
                ("Final authoritative check did not succeed.",),
            ),
            "APP_ENV=test vendor/bin/phpunit",
            2,
            0,
            "all clear",
            "Model explanation.",
            "main",
        )
    )

    assert result.outcome is AttemptOutcome.INCOMPLETE
    assert result.branch_url is None
    assert len(github.comments) == 1
    assert "Implementation succeeded" not in github.comments[0][1]
    assert "Final authoritative check did not succeed." in github.comments[0][1]
    assert "Partial branch push failed" in github.comments[0][1]
    assert "remote hung up unexpectedly" in github.comments[0][1]
    assert "Final authoritative check did not succeed." in result.details
    assert "Partial branch push failed" in result.details
    assert len(events) == 1
    assert events[0][0] == "publication_failed"
    assert "remote hung up unexpectedly" in events[0][1]
    assert events[0][2] == "ERROR"
    assert events[0][3] == 23


def test_pr_creation_failure_keeps_the_established_infrastructure_error_details(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = AlwaysFailingGitHub()

    result = Publisher(workspace, github, "octo/example", sleeper=lambda _: None).publish(
        request(CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)), prepared.branch)
    )

    assert result.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    assert len(github.comments) == 1
    assert "Implementation succeeded but publication failed." in github.comments[0][1]


def request(decision: CompletionDecision, branch: str) -> PublicationRequest:
    return PublicationRequest(23, "Implement publication", branch, "2026-09-20T13:00:00Z", decision, "pytest", 1, 0, "all clear", "Check did not pass.", "main")


def publishing_state(tmp_path: Path) -> AttemptStateStore:
    state = AttemptStateStore(tmp_path / "state")
    state.start(issue_number=23, branch="agent/issue-23")
    state.transition(AttemptPhase.SETUP)
    state.transition(AttemptPhase.MODEL_RUNNING)
    state.transition(AttemptPhase.PUSHING)
    return state


@dataclass
class FakeGitHub:
    pull_request: PullRequest | None = None
    fail_creation_after_write: bool = False
    enable_error: Exception | None = None
    merge_error: Exception | None = None

    def __post_init__(self) -> None:
        self.created: list[tuple[str, str, str, str]] = []
        self.pull_request_bodies: list[str] = []
        self.comments: list[tuple[int, str]] = []
        self.auto_merge_calls: list[tuple[str, int, str]] = []
        self.merge_calls: list[tuple[str, int, str]] = []

    def find_pull_request(self, repository: str, head: str) -> PullRequest | None:
        return self.pull_request

    def create_pull_request(self, repository: str, head: str, base: str, title: str, body: str) -> PullRequest:
        self.created.append((repository, head, base, title))
        self.pull_request_bodies.append(body)
        self.pull_request = PullRequest(1, "https://example.test/pulls/1", "pr-node-1")
        if self.fail_creation_after_write:
            raise OSError("connection closed after write")
        return self.pull_request

    def enable_auto_merge(
        self, repository: str, pull_request: PullRequest, merge_method: str
    ) -> None:
        self.auto_merge_calls.append((repository, pull_request.number, merge_method))
        if self.enable_error is not None:
            raise self.enable_error

    def merge_pull_request(
        self, repository: str, pull_request: PullRequest, merge_method: str
    ) -> None:
        self.merge_calls.append((repository, pull_request.number, merge_method))
        if self.merge_error is not None:
            raise self.merge_error

    def find_attempt_comment(self, repository: str, issue_number: int, marker: str) -> bool:
        return any(number == issue_number and marker in body for number, body in self.comments)

    def add_comment(self, repository: str, issue_number: int, body: str) -> None:
        self.comments.append((issue_number, body))


class PushFailingWorkspace:
    def push_attempt_branch(self, branch: str, *, max_retries: int, deadline: float | None = None) -> str:
        raise GitWorkspaceError("remote hung up unexpectedly")


@dataclass
class AlwaysFailingGitHub(FakeGitHub):
    def find_pull_request(self, repository: str, head: str) -> PullRequest | None:
        return None

    def create_pull_request(self, repository: str, head: str, base: str, title: str, body: str) -> PullRequest:
        raise OSError("pull request creation unavailable")


def test_auto_merge_off_makes_no_merge_calls_and_reports_nothing(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub()

    result = Publisher(workspace, github, "octo/example").publish(
        request(CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)), prepared.branch)
    )

    assert result.outcome is AttemptOutcome.COMPLETE
    assert result.merge_result is None
    assert github.auto_merge_calls == []
    assert github.merge_calls == []
    assert "**Merge**" not in github.comments[0][1]


def test_auto_merge_enabled_after_pr_creation(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub()

    result = Publisher(
        workspace, github, "octo/example", auto_merge=True, merge_method="squash"
    ).publish(
        request(CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)), prepared.branch)
    )

    assert result.outcome is AttemptOutcome.COMPLETE
    assert result.merge_result == "auto-merge enabled (squash)"
    assert github.auto_merge_calls == [("octo/example", 1, "squash")]
    assert github.merge_calls == []
    assert "**Merge**: auto-merge enabled (squash)" in github.comments[0][1]


def test_auto_merge_runs_for_an_already_existing_pr(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub(pull_request=PullRequest(7, "https://example.test/pulls/7", "pr-node-7"))

    result = Publisher(
        workspace, github, "octo/example", auto_merge=True, merge_method="rebase"
    ).publish(
        request(CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)), prepared.branch)
    )

    assert result.outcome is AttemptOutcome.COMPLETE
    assert result.pull_request == PullRequest(7, "https://example.test/pulls/7", "pr-node-7")
    assert github.created == []
    assert github.auto_merge_calls == [("octo/example", 7, "rebase")]
    assert result.merge_result == "auto-merge enabled (rebase)"


def test_auto_merge_falls_back_to_direct_merge_when_already_mergeable(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub(enable_error=OSError("Pull request is already mergeable"))

    result = Publisher(
        workspace, github, "octo/example", auto_merge=True, merge_method="squash"
    ).publish(
        request(CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)), prepared.branch)
    )

    assert result.outcome is AttemptOutcome.COMPLETE
    assert result.merge_result == "merged"
    assert len(github.auto_merge_calls) == 1
    assert github.merge_calls == [("octo/example", 1, "squash")]
    assert "**Merge**: merged" in github.comments[0][1]


def test_auto_merge_failure_keeps_complete_and_reports_the_reason(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub(enable_error=OSError("auto-merge not allowed on this repository"))

    result = Publisher(workspace, github, "octo/example", auto_merge=True).publish(
        request(CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)), prepared.branch)
    )

    assert result.outcome is AttemptOutcome.COMPLETE
    assert result.merge_result == "auto-merge failed: auto-merge not allowed on this repository"
    assert github.merge_calls == []
    assert "auto-merge failed: auto-merge not allowed on this repository" in github.comments[0][1]


def test_direct_merge_failure_keeps_complete_and_reports_the_reason(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub(
        enable_error=OSError("Pull request is already mergeable"),
        merge_error=OSError("merge conflict: 2 files"),
    )

    result = Publisher(workspace, github, "octo/example", auto_merge=True).publish(
        request(CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)), prepared.branch)
    )

    assert result.outcome is AttemptOutcome.COMPLETE
    assert result.merge_result == "auto-merge failed: merge conflict: 2 files"
    assert "**Merge**: auto-merge failed: merge conflict: 2 files" in github.comments[0][1]


def test_non_complete_outcomes_never_trigger_the_merge_step(tmp_path: Path) -> None:
    decisions = [
        CompletionDecision(AttemptOutcome.INCOMPLETE, False, PublicationPath.PARTIAL, ("check failed",)),
        CompletionDecision(AttemptOutcome.NO_CHANGES, False, PublicationPath.NONE, ("no commits",)),
        CompletionDecision(AttemptOutcome.INFRASTRUCTURE_ERROR, False, PublicationPath.NONE, ("broken",)),
        CompletionDecision(AttemptOutcome.HANDOFF, False, PublicationPath.PARTIAL, ("handoff",)),
    ]
    for decision in decisions:
        remote, _ = repository_with_main(tmp_path / f"repo-{decision.outcome.value}")
        workspace = GitWorkspace(
            tmp_path / f"clone-{decision.outcome.value}",
            str(remote),
            token_provider=lambda: "token",
        )
        prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
        if decision.publication_path is PublicationPath.PARTIAL:
            write_and_commit(
                tmp_path / f"clone-{decision.outcome.value}",
                "partial.txt",
                "partial",
                "Partial work",
            )
        github = FakeGitHub()

        result = Publisher(workspace, github, "octo/example", auto_merge=True).publish(
            request(decision, prepared.branch)
        )

        assert result.outcome is decision.outcome
        assert result.merge_result is None
        assert github.auto_merge_calls == []
        assert github.merge_calls == []


def test_merge_step_is_idempotent_for_already_enabled_pr(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub(enable_error=OSError("auto-merge already enabled for this pull request"))

    result = Publisher(workspace, github, "octo/example", auto_merge=True).publish(
        request(CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)), prepared.branch)
    )

    assert result.outcome is AttemptOutcome.COMPLETE
    assert result.merge_result == "auto-merge enabled (merge)"
    assert "auto-merge failed" not in github.comments[0][1]


def test_merge_step_is_idempotent_for_an_already_merged_pr(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub(enable_error=OSError("Pull request was already merged"))

    result = Publisher(workspace, github, "octo/example", auto_merge=True).publish(
        request(CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)), prepared.branch)
    )

    assert result.outcome is AttemptOutcome.COMPLETE
    assert result.merge_result == "merged"
    assert github.merge_calls == []
    assert "auto-merge failed" not in github.comments[0][1]


def test_merge_step_is_idempotent_when_direct_merge_reports_already_merged(
    tmp_path: Path,
) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub(
        enable_error=OSError("Pull request is already mergeable"),
        merge_error=OSError("Pull request was already merged"),
    )

    result = Publisher(workspace, github, "octo/example", auto_merge=True).publish(
        request(CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)), prepared.branch)
    )

    assert result.outcome is AttemptOutcome.COMPLETE
    assert result.merge_result == "merged"
    assert "auto-merge failed" not in github.comments[0][1]


def test_label_mode_merges_when_the_issue_was_opted_in_at_claim_time(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub()

    published = publish_label_mode(workspace, github, prepared.branch, opt_in=True)

    assert published.outcome is AttemptOutcome.COMPLETE
    assert published.merge_result == "auto-merge enabled (merge)"
    assert github.auto_merge_calls == [("octo/example", 1, "merge")]
    assert github.merge_calls == []
    assert "**Merge**: auto-merge enabled (merge)" in github.comments[0][1]


def test_label_mode_skips_the_merge_step_without_opt_in(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub()

    published = publish_label_mode(workspace, github, prepared.branch, opt_in=False)

    assert published.outcome is AttemptOutcome.COMPLETE
    assert published.pull_request == PullRequest(1, "https://example.test/pulls/1", "pr-node-1")
    assert published.merge_result == (
        "auto-merge skipped (issue not labelled `agent-auto-merge` at claim time)"
    )
    assert github.auto_merge_calls == []
    assert github.merge_calls == []
    assert (
        "**Merge**: auto-merge skipped (issue not labelled `agent-auto-merge` at claim time)"
        in github.comments[0][1]
    )


def test_label_mode_skips_when_the_request_carries_no_claim_time_decision(
    tmp_path: Path,
) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub()

    result = Publisher(workspace, github, "octo/example", auto_merge="label").publish(
        request(CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)), prepared.branch)
    )

    assert result.outcome is AttemptOutcome.COMPLETE
    assert result.pull_request is not None
    assert result.merge_result == (
        "auto-merge skipped (issue not labelled `agent-auto-merge` at claim time)"
    )
    assert github.auto_merge_calls == []
    assert github.merge_calls == []


def test_true_mode_still_merges_without_an_opt_in(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub()

    result = Publisher(workspace, github, "octo/example", auto_merge=True).publish(
        request(CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)), prepared.branch)
    )

    assert result.outcome is AttemptOutcome.COMPLETE
    assert result.merge_result == "auto-merge enabled (merge)"
    assert github.auto_merge_calls == [("octo/example", 1, "merge")]


def test_false_mode_never_merges_even_when_opted_in(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub()

    result = Publisher(workspace, github, "octo/example", auto_merge="false").publish(
        publish_request(prepared.branch, opt_in=True)
    )

    assert result.outcome is AttemptOutcome.COMPLETE
    assert result.merge_result is None
    assert github.auto_merge_calls == []
    assert github.merge_calls == []
    assert "**Merge**" not in github.comments[0][1]


def test_rejects_an_unknown_auto_merge_mode(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")

    with pytest.raises(ValueError, match="auto_merge"):
        Publisher(workspace, FakeGitHub(), "octo/example", auto_merge="sometimes")


def publish_request(branch: str, *, opt_in: bool) -> PublicationRequest:
    base = request(CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)), branch)
    return PublicationRequest(
        base.issue_number,
        base.issue_title,
        base.branch,
        base.started_at,
        base.decision,
        base.check_command,
        base.check_exit_code,
        base.review_cycles,
        base.review_findings,
        base.details,
        base.base_branch,
        auto_merge_opt_in=opt_in,
    )


def publish_label_mode(workspace, github: FakeGitHub, branch: str, *, opt_in: bool):
    return Publisher(workspace, github, "octo/example", auto_merge="label").publish(
        publish_request(branch, opt_in=opt_in)
    )


# --- Reviews performed vs repair cycles (issue #178) ---------------------------


def complete_request(branch: str, **overrides: object) -> PublicationRequest:
    fields: dict[str, object] = {
        "issue_number": 23,
        "issue_title": "Implement publication",
        "branch": branch,
        "started_at": "2026-09-20T13:00:00Z",
        "decision": CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)),
        "check_command": "pytest",
        "check_exit_code": 0,
        "review_cycles": 0,
        "review_findings": "not collected",
        "details": "Implemented the publication boundary.",
        "base_branch": "main",
    }
    fields.update(overrides)
    return PublicationRequest(**fields)  # type: ignore[arg-type]


def test_result_comment_reports_zero_reviews_explicitly(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub()

    Publisher(workspace, github, "octo/example").publish(
        complete_request(prepared.branch, reviews_performed=0, review_cycles=0)
    )

    (comment,) = [body for _, body in github.comments]
    assert "Reviews performed: 0" in comment
    assert "Review repair cycles: 0/2" in comment
    assert "code-review was not run" in comment


def test_result_comment_distinguishes_reviews_from_repair_cycles(
    tmp_path: Path,
) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub()

    Publisher(workspace, github, "octo/example").publish(
        complete_request(prepared.branch, reviews_performed=1, review_cycles=0)
    )

    (comment,) = [body for _, body in github.comments]
    assert "Reviews performed: 1" in comment
    assert "Review repair cycles: 0/2" in comment
    assert "code-review was not run" not in comment


def test_result_comment_reports_repair_cycles_for_repeated_reviews(
    tmp_path: Path,
) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub()

    Publisher(workspace, github, "octo/example").publish(
        complete_request(prepared.branch, reviews_performed=3, review_cycles=2)
    )

    (comment,) = [body for _, body in github.comments]
    assert "Reviews performed: 3" in comment
    assert "Review repair cycles: 2/2" in comment
    assert "code-review was not run" not in comment


def test_pull_request_body_reports_reviews_performed(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=23)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    github = FakeGitHub()

    Publisher(workspace, github, "octo/example").publish(
        complete_request(prepared.branch, reviews_performed=0, review_cycles=0)
    )

    (body,) = github.pull_request_bodies
    assert "Reviews performed: 0" in body
    assert "Review repair cycles: 0/2" in body
    assert "code-review was not run" in body
