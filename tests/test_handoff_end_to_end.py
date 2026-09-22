"""End-to-end acceptance tests for the handoff/continuation flow (issue #49).

These wire the real ``GitWorkspace``, ``GitHubTracker``, ``Publisher``, and
``AgentLifecycle`` together against a local git remote and a fake GitHub
transport, following the fakes/patterns already used in ``test_lifecycle.py``,
``test_git_workspace.py``, ``test_github_tracker.py``, and
``test_publication.py``.

Only the continuation side of the flow (restoring a published branch,
short-circuiting a missing one, filtering trusted comments, releasing
``round-finished`` on claim) is implemented in this codebase; see #46-#48.
The mechanism that *produces* a handoff (a soft cost threshold, a
project-owned ``handoff`` skill, writing ``.agent/handoff/<n>.md``, adding
``round-finished``) is not implemented yet. Scenarios that describe a prior
handoff (1, 2, 4, 6) therefore set up that prior state directly -- a pushed
attempt branch, a handoff note file, a `round-finished` label, and a comment
carrying the attempt-result marker -- the same durable artifacts a real
handoff would have produced, and exercise the real continuation path against
them.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
import subprocess

from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.command_runner import CommandRunner
from simple_coding_agent.completion import AttemptOutcome, CompletionEvaluator, VerificationRunner
from simple_coding_agent.config import RepositoryProfile
from simple_coding_agent.git_workspace import GitWorkspace
from simple_coding_agent.github_tracker import (
    Assignment,
    GitHubIdentity,
    GitHubTracker,
    IssueComment,
    IssuePage,
    TrackerIssue,
)
from simple_coding_agent.lifecycle import AgentLifecycle, LifecycleStatus, ModelAttemptRunner
from simple_coding_agent.model_execution import ModelExecution, ModelExecutionStatus, SkillEvent
from simple_coding_agent.publication import Publisher, PullRequest

REPO = "octo/example"


# --- Scenario 1: cost soft-threshold handoff -> continuation -> PR ---------


def test_cost_soft_threshold_handoff_continuation_completes_with_a_pull_request(
    tmp_path: Path,
) -> None:
    remote, github = run_published_handoff_continuation(
        tmp_path,
        issue_number=24,
        issue_body="Rewrite the parser.",
        note_body="# Handoff note\n\nCost threshold reached; parser rewrite half done.\n",
        note_message="Add handoff note for cost threshold",
        handoff_comment="## Agent Attempt Result: incomplete\n\nCost threshold reached; handed off.",
        finish=lambda wd: write_and_commit(wd, "parser.py", "done", "Finish the parser"),
    )

    assert github.removed_labels == [
        ("issue-24", "round-finished"),
        ("issue-24", "ready-for-agent"),
    ]
    assert "agent/issue-24" in github.pull_requests
    assert remote_branch_subjects(remote, "agent/issue-24")[:2] == [
        "Finish the parser",
        "Add handoff note for cost threshold",
    ]


# --- Scenario 2: turns/timeout hard-limit handoff -> same continuation path -


def test_turns_timeout_hard_limit_handoff_continuation_completes_with_a_pull_request(
    tmp_path: Path,
) -> None:
    """The continuation contract cannot distinguish a cost from a hard-limit
    handoff -- both leave the same durable artifacts (branch, note, label,
    comment) -- so this only differs from scenario 1 in the comment's stated
    cause, proving that difference is irrelevant to how continuation behaves.
    """

    remote, github = run_published_handoff_continuation(
        tmp_path,
        issue_number=25,
        issue_body="Migrate the config loader.",
        note_body="# Handoff note\n\nInterrupted at MODEL_TIMEOUT; migration half done.\n",
        note_message="Add handoff note for hard-limit interrupt",
        handoff_comment=(
            "## Agent Attempt Result: incomplete\n\nMax turns exceeded; interrupted, drained, handed off."
        ),
        finish=lambda wd: write_and_commit(wd, "config_loader.py", "done", "Finish the migration"),
    )

    assert "agent/issue-25" in github.pull_requests
    assert remote_branch_subjects(remote, "agent/issue-25")[:2] == [
        "Finish the migration",
        "Add handoff note for hard-limit interrupt",
    ]


# --- Scenario 3: restart between publication steps --------------------------


def test_restart_between_publication_steps_recovers_and_completes(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    clone_dir = tmp_path / "clone"
    data_dir = tmp_path / "data"
    github = FakeGitHub(issue(24, body="Add logging."))

    # First process: model committed, checkpoint reached PUSHING, then "crashed"
    # before the push happened.
    workspace = GitWorkspace(clone_dir, str(remote), token_provider=lambda: "token")
    workspace.prepare_attempt(base_branch="main", issue_number=24)
    write_and_commit(clone_dir, "logging.py", "print('hi')", "Add logging")
    attempt_state = AttemptStateStore(data_dir)
    attempt_state.start(issue_number=24, branch="agent/issue-24")
    attempt_state.transition(AttemptPhase.SETUP)
    attempt_state.transition(AttemptPhase.MODEL_RUNNING)
    attempt_state.transition(AttemptPhase.PUSHING)

    # Restart: a fresh AgentLifecycle reconciles the interrupted attempt at
    # startup, reusing the same on-disk clone and checkpoint.
    executor = FakeModelExecutor()
    lifecycle = build_lifecycle(remote, clone_dir, data_dir, github, executor)

    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.ATTEMPTED
    assert result.outcome is AttemptOutcome.COMPLETE
    assert executor.calls == 0
    assert "agent/issue-24" in github.pull_requests
    assert remote_branch_subjects(remote, "agent/issue-24")[0] == "Add logging"
    assert AttemptStateStore(data_dir).read() is None


# --- Scenario 4: continuation from a fresh clone -----------------------------


def test_continuation_from_a_fresh_clone_restores_the_published_branch(tmp_path: Path) -> None:
    clone_dir = tmp_path / "brand-new-clone"
    assert not clone_dir.exists()

    remote, _ = run_published_handoff_continuation(
        tmp_path,
        issue_number=31,
        issue_body="Migrate the database.",
        note_body="# Handoff note\n\nHalfway through the migration.\n",
        note_message="Add handoff note",
        handoff_comment="## Agent Attempt Result: incomplete\n\nhanded off",
        finish=lambda wd: write_and_commit(wd, "migration.py", "done", "Finish the migration"),
        clone_dir=clone_dir,
    )

    assert remote_branch_subjects(remote, "agent/issue-31")[:2] == [
        "Finish the migration",
        "Add handoff note",
    ]


# --- Scenario 5: repeated handoff keeps both comments in the filtered feed --


def test_repeated_handoff_keeps_both_comments_in_the_filtered_feed() -> None:
    github = FakeGitHub(
        issue(24, author_login="reporter"),
        comments=(
            IssueComment(author_login="agent", body="## Agent Attempt Result: incomplete\n\nFirst handoff"),
            IssueComment(author_login="agent", body="## Agent Attempt Result: incomplete\n\nSecond handoff"),
        ),
    )
    tracker = GitHubTracker(github, REPO)

    trusted = tracker.trusted_comments(issue(24, author_login="reporter"))

    assert trusted == (
        "## Agent Attempt Result: incomplete\n\nFirst handoff",
        "## Agent Attempt Result: incomplete\n\nSecond handoff",
    )


# --- Scenario 6: rebase conflict on continuation -----------------------------


def test_rebase_conflict_on_continuation_is_resolved_by_the_model(tmp_path: Path) -> None:
    remote, seed = repository_with_main(tmp_path)
    publish_attempt_branch(tmp_path, remote, 24, {"README.md": "branch change\n"}, "Conflicting branch commit")
    write_and_commit(seed, "README.md", "upstream fix\n", "Conflicting upstream commit")
    git(seed, "push", "origin", "main")

    def resolve(working_directory: Path) -> None:
        assert "<<<<<<<" in (working_directory / "README.md").read_text()
        (working_directory / "README.md").write_text("resolved\n")
        git(working_directory, "add", "README.md")
        git(working_directory, "rebase", "--continue")
        write_and_commit(working_directory, "feature.txt", "done", "Finish after resolving conflict")

    github = FakeGitHub(
        issue(24, body="Fix the README workflow.", labels=frozenset({"ready-for-agent", "round-finished"})),
        comments=(IssueComment(author_login="agent", body="## Agent Attempt Result: incomplete\n\nhanded off"),),
    )
    executor = FakeModelExecutor(actions=[resolve])
    lifecycle = build_lifecycle(remote, tmp_path / "clone", tmp_path / "data", github, executor)

    result = lifecycle.run_once()

    assert result.outcome is AttemptOutcome.COMPLETE
    subjects = remote_branch_subjects(remote, "agent/issue-24")
    assert "Finish after resolving conflict" in subjects
    assert "Conflicting branch commit" in subjects


# --- Scenario 7: missing branch despite continuation signals ----------------


def test_missing_continuation_branch_short_circuits_to_incomplete(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    github = FakeGitHub(
        issue(24, body="Fix the thing.", labels=frozenset({"ready-for-agent", "round-finished"})),
    )
    executor = FakeModelExecutor()
    lifecycle = build_lifecycle(remote, tmp_path / "clone", tmp_path / "data", github, executor)

    result = lifecycle.run_once()

    assert result.outcome is AttemptOutcome.INCOMPLETE
    assert executor.calls == 0
    [comment] = github.comments
    assert "not found on origin" in comment.body
    assert "agent/issue-24" in comment.body
    assert not remote_has_branch(remote, "agent/issue-24")


# --- Scenario 8: zero-commit handoff stays no_changes ------------------------


def test_zero_commit_attempt_stays_no_changes_and_never_publishes(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    github = FakeGitHub(issue(24, body="Investigate flaky test."))
    executor = FakeModelExecutor()
    lifecycle = build_lifecycle(remote, tmp_path / "clone", tmp_path / "data", github, executor)

    result = lifecycle.run_once()

    assert result.outcome is AttemptOutcome.NO_CHANGES
    assert executor.calls == 1
    assert github.pull_requests == {}
    assert not remote_has_branch(remote, "agent/issue-24")


# --- Scenario 9: author/operator guidance reaches the prompt ----------------


def test_author_and_operator_comments_reach_the_prompt(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    github = FakeGitHub(
        issue(24, body="Improve error messages.", author_login="reporter"),
        comments=(
            IssueComment(author_login="reporter", body="Please also cover the CLI path."),
            IssueComment(author_login="agent", body="Focus on the JSON formatter first."),
        ),
    )
    executor = FakeModelExecutor(
        actions=[lambda wd: write_and_commit(wd, "errors.py", "done", "Improve errors")]
    )
    lifecycle = build_lifecycle(remote, tmp_path / "clone", tmp_path / "data", github, executor)

    lifecycle.run_once()

    [prompt] = executor.captured_prompts
    assert "Please also cover the CLI path." in prompt
    assert "Focus on the JSON formatter first." in prompt


# --- Scenario 10: third-party comment excluded from the prompt --------------


def test_third_party_comment_is_excluded_from_the_prompt(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    github = FakeGitHub(
        issue(24, body="Improve error messages.", author_login="reporter"),
        comments=(
            IssueComment(author_login="reporter", body="Please also cover the CLI path."),
            IssueComment(author_login="random-passerby", body="I think this is fine as-is."),
        ),
    )
    executor = FakeModelExecutor(
        actions=[lambda wd: write_and_commit(wd, "errors.py", "done", "Improve errors")]
    )
    lifecycle = build_lifecycle(remote, tmp_path / "clone", tmp_path / "data", github, executor)

    lifecycle.run_once()

    [prompt] = executor.captured_prompts
    assert "Please also cover the CLI path." in prompt
    assert "I think this is fine as-is." not in prompt


# --- Scenario 11: real handoff production, publication, and label sequence --


def test_real_handoff_is_produced_pushed_and_labeled_without_a_pull_request(
    tmp_path: Path,
) -> None:
    remote, _ = repository_with_main(tmp_path)
    github = FakeGitHub(
        issue(24, body="Rewrite the parser.", labels=frozenset({"ready-for-agent"})),
    )

    def do_handoff(working_directory: Path) -> None:
        write_and_commit(working_directory, "parser.py", "half done", "Half-finish the parser")
        write_handoff_note(working_directory, 24, "# Handoff note: issue #24\n\nHalfway done.\n")

    executor = FakeModelExecutor(actions=[do_handoff], handoff=True)
    lifecycle = build_lifecycle(remote, tmp_path / "clone", tmp_path / "data", github, executor)

    result = lifecycle.run_once()

    assert result.outcome is AttemptOutcome.HANDOFF
    assert github.pull_requests == {}
    subjects = remote_branch_subjects(remote, "agent/issue-24")
    assert subjects[:2] == ["Handoff note: issue #24", "Half-finish the parser"]
    [comment] = github.comments
    assert "## Agent Attempt Result: handoff" in comment.body
    assert "Halfway done." in comment.body
    assert github.removed_labels == [("issue-24", "ready-for-agent")]
    assert github.added_labels == [("issue-24", "round-finished")]
    assert "round-finished" in github.labels
    assert github.assignee_logins == []


# --- Scenario 12: note-only handoff downgrades to no_changes and is discarded


def test_note_only_handoff_downgrades_to_no_changes_and_is_not_published(
    tmp_path: Path,
) -> None:
    remote, _ = repository_with_main(tmp_path)
    github = FakeGitHub(issue(24, body="Investigate flaky test."))

    def note_only(working_directory: Path) -> None:
        write_handoff_note(working_directory, 24, "# Handoff note: issue #24\n\nNothing to preserve.\n")

    executor = FakeModelExecutor(actions=[note_only], handoff=True)
    lifecycle = build_lifecycle(remote, tmp_path / "clone", tmp_path / "data", github, executor)

    result = lifecycle.run_once()

    assert result.outcome is AttemptOutcome.NO_CHANGES
    assert github.pull_requests == {}
    assert github.added_labels == []
    assert not remote_has_branch(remote, "agent/issue-24")


# --- Scenario 13: repeated handoff retains note history and both comments ---


def test_repeated_real_handoff_retains_note_history_and_both_comments(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    clone_dir = tmp_path / "clone"
    data_dir = tmp_path / "data"
    github = FakeGitHub(issue(24, body="Rewrite the parser.", labels=frozenset({"ready-for-agent"})))

    def first_handoff(working_directory: Path) -> None:
        write_and_commit(working_directory, "parser.py", "step one", "Start the parser rewrite")
        write_handoff_note(working_directory, 24, "# Handoff note: issue #24\n\nStep one done.\n")

    first_executor = FakeModelExecutor(actions=[first_handoff], handoff=True)
    first_lifecycle = build_lifecycle(remote, clone_dir, data_dir, github, first_executor)
    first_result = first_lifecycle.run_once()

    assert first_result.outcome is AttemptOutcome.HANDOFF
    assert "round-finished" in github.labels

    # A human restores ready-for-agent to requeue; claim_next clears
    # round-finished the same way it already does for a fabricated one.
    github.labels.add("ready-for-agent")
    github.assignee_logins.clear()

    def second_handoff(working_directory: Path) -> None:
        write_and_commit(working_directory, "parser.py", "step two", "Finish the parser rewrite")
        write_handoff_note(working_directory, 24, "# Handoff note: issue #24\n\nStep two done.\n")

    second_executor = FakeModelExecutor(actions=[second_handoff], handoff=True)
    second_lifecycle = build_lifecycle(remote, clone_dir, data_dir, github, second_executor)
    second_result = second_lifecycle.run_once()

    assert second_result.outcome is AttemptOutcome.HANDOFF
    subjects = remote_branch_subjects(remote, "agent/issue-24")
    assert subjects[:4] == [
        "Handoff note: issue #24",
        "Finish the parser rewrite",
        "Handoff note: issue #24",
        "Start the parser rewrite",
    ]
    handoff_comments = [c for c in github.comments if "Agent Attempt Result: handoff" in c.body]
    assert len(handoff_comments) == 2
    assert "Step one done." in handoff_comments[0].body
    assert "Step two done." in handoff_comments[1].body


# --- Harness -----------------------------------------------------------------


def run_published_handoff_continuation(
    tmp_path: Path,
    *,
    issue_number: int,
    issue_body: str,
    note_body: str,
    note_message: str,
    handoff_comment: str,
    finish,
    clone_dir: Path | None = None,
) -> tuple[Path, "FakeGitHub"]:
    """Resume a previously published handoff branch and run one attempt to completion.

    Shared by the scenarios that only differ in the prior handoff's cause and
    the work the model does to finish: sets up the pushed branch, handoff
    note, ``round-finished`` label, and handoff comment a real handoff would
    have left behind, then runs the real continuation path against them.
    """

    remote, _ = repository_with_main(tmp_path)
    publish_attempt_branch(
        tmp_path, remote, issue_number, {f".agent/handoff/{issue_number}.md": note_body}, note_message
    )
    github = FakeGitHub(
        issue(issue_number, body=issue_body, labels=frozenset({"ready-for-agent", "round-finished"})),
        comments=(IssueComment(author_login="agent", body=handoff_comment),),
    )
    executor = FakeModelExecutor(actions=[finish])
    lifecycle = build_lifecycle(remote, clone_dir or tmp_path / "clone", tmp_path / "data", github, executor)

    result = lifecycle.run_once()

    assert result.outcome is AttemptOutcome.COMPLETE
    return remote, github


def build_lifecycle(
    remote: Path,
    clone_dir: Path,
    data_dir: Path,
    github: "FakeGitHub",
    executor: "FakeModelExecutor",
) -> AgentLifecycle:
    attempt_state = AttemptStateStore(data_dir)
    workspace = GitWorkspace(clone_dir, str(remote), token_provider=lambda: "token")
    tracker = GitHubTracker(github, REPO)
    runner = ModelAttemptRunner(
        attempt_state=attempt_state,
        workspace=workspace,
        verifier=VerificationRunner(CommandRunner()),
        evaluator=CompletionEvaluator(frozenset({"blocking"})),
        model_executor=executor,
        issue_comments=tracker.trusted_comments,
    )
    publisher = Publisher(workspace, github, REPO, attempt_state=attempt_state, sleeper=lambda _: None)
    return AgentLifecycle(
        tracker=tracker,
        attempt_state=attempt_state,
        workspace=workspace,
        profile_loader=lambda _: profile(),
        publisher=publisher,
        attempt_runner=runner,
        sleeper=lambda _: None,
    )


def profile() -> RepositoryProfile:
    return RepositoryProfile(setup=(), check=("true",), base_branch="main", timeout=30, setup_timeout=30, env={})


class FakeModelExecutor:
    """Runs a scripted local git action to stand in for one model turn.

    Defaults to succeeding with one completed code-review, the minimum
    evidence the real evaluator requires to reach ``complete``. Pass
    ``handoff=True`` for an action that stands in for the model invoking the
    handoff skill instead (write/commit the note, then stop): the returned
    evidence carries HANDOFF_REQUESTED and a ``handoff`` skill event instead.
    """

    def __init__(self, *, actions: list | None = None, handoff: bool = False) -> None:
        self._actions = list(actions or [])
        self._handoff = handoff
        self.calls = 0
        self.captured_prompts: list[str] = []

    async def execute(
        self, *, issue_body: str, working_directory: Path, issue_number: int | None = None, archive=None
    ) -> ModelExecution:
        self.calls += 1
        self.captured_prompts.append(issue_body)
        if self._actions:
            self._actions.pop(0)(working_directory)
        if self._handoff:
            return ModelExecution(
                status=ModelExecutionStatus.HANDOFF_REQUESTED,
                explanation="scripted handoff",
                stop_reason="end_turn",
                model_usage=None,
                observed_models=(),
                skill_events=(
                    SkillEvent(phase="PreToolUse", name="handoff", agent_id=None, timestamp="2026-09-22T00:00:00Z"),
                ),
            )
        return ModelExecution(
            status=ModelExecutionStatus.SUCCEEDED,
            explanation="scripted",
            stop_reason="end_turn",
            model_usage=None,
            observed_models=(),
            skill_events=(SkillEvent(phase="PreToolUse", name="code-review", agent_id=None, timestamp="2026-09-22T00:00:00Z"),),
        )


class FakeGitHub:
    """A single-issue GitHub double for both the tracker and publisher seams."""

    def __init__(self, initial: TrackerIssue, *, comments: tuple[IssueComment, ...] = ()) -> None:
        self._issue = initial
        self.labels: set[str] = set(initial.labels)
        self.assignee_logins: list[str] = list(initial.assignee_logins)
        self.comments: list[IssueComment] = list(comments)
        self.pull_requests: dict[str, PullRequest] = {}
        self.removed_labels: list[tuple[str, str]] = []
        self.added_labels: list[tuple[str, str]] = []
        self._next_pr = 100

    def _current(self) -> TrackerIssue:
        return replace(self._issue, labels=frozenset(self.labels), assignee_logins=tuple(self.assignee_logins))

    # GitHubTransport
    def list_issues(self, repository: str, cursor: str | None) -> IssuePage:
        return IssuePage(issues=(self._current(),), next_cursor=None)

    def get_issue(self, repository: str, number: int) -> TrackerIssue | None:
        return self._current() if number == self._issue.number else None

    def viewer(self) -> GitHubIdentity:
        return GitHubIdentity(id="agent-id", login="agent")

    def assign_issue(self, issue_id: str, assignee_id: str) -> Assignment:
        if "agent" not in self.assignee_logins:
            self.assignee_logins.append("agent")
        return Assignment(issue_id=issue_id, assignee_id=assignee_id)

    def remove_assignee(self, issue_id: str, assignee_id: str) -> None:
        if "agent" in self.assignee_logins:
            self.assignee_logins.remove("agent")

    def remove_label(self, issue_id: str, label: str) -> None:
        self.removed_labels.append((issue_id, label))
        self.labels.discard(label)

    def add_label(self, issue_id: str, label: str) -> None:
        self.added_labels.append((issue_id, label))
        self.labels.add(label)

    def list_issue_comments(self, repository: str, issue_number: int) -> tuple[IssueComment, ...]:
        return tuple(self.comments)

    # PublicationTransport
    def find_pull_request(self, repository: str, head: str) -> PullRequest | None:
        return self.pull_requests.get(head)

    def create_pull_request(self, repository: str, head: str, base: str, title: str, body: str) -> PullRequest:
        pull_request = PullRequest(number=self._next_pr, url=f"https://github.com/{repository}/pull/{self._next_pr}")
        self._next_pr += 1
        self.pull_requests[head] = pull_request
        return pull_request

    def find_attempt_comment(self, repository: str, issue_number: int, marker: str) -> bool:
        return any(marker in comment.body for comment in self.comments)

    def add_comment(self, repository: str, issue_number: int, body: str) -> None:
        self.comments.append(IssueComment(author_login="agent", body=body))


def issue(
    number: int,
    *,
    body: str = "Fix the thing.",
    labels: frozenset[str] = frozenset({"ready-for-agent"}),
    author_login: str = "reporter",
) -> TrackerIssue:
    return TrackerIssue(
        id=f"issue-{number}",
        number=number,
        title=f"Issue {number}",
        body=body,
        created_at=datetime(2026, 9, 20, tzinfo=UTC),
        state="OPEN",
        labels=labels,
        assignee_logins=(),
        blocked_by=0,
        author_login=author_login,
    )


def repository_with_main(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    git(tmp_path, "init", "--bare", str(remote))
    git(tmp_path, "init", str(seed))
    git(seed, "checkout", "-b", "main")
    write_and_commit(seed, "README.md", "seed", "Initial commit")
    git(seed, "remote", "add", "origin", str(remote))
    git(seed, "push", "-u", "origin", "main")
    git(remote, "symbolic-ref", "HEAD", "refs/heads/main")
    return remote, seed


def publish_attempt_branch(
    tmp_path: Path,
    remote: Path,
    issue_number: int,
    files: dict[str, str],
    message: str,
) -> None:
    """Simulate a prior attempt's push -- a handoff's durable artifact -- via a throwaway clone."""

    published = tmp_path / f"published-{issue_number}"
    git(tmp_path, "clone", str(remote), str(published))
    git(published, "config", "user.name", "Test User")
    git(published, "config", "user.email", "test@example.com")
    git(published, "checkout", "-b", f"agent/issue-{issue_number}")
    for filename, contents in files.items():
        path = published / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        git(published, "add", filename)
    git(published, "commit", "-m", message)
    git(published, "push", "-u", "origin", f"agent/issue-{issue_number}")


def write_and_commit(repository: Path, name: str, contents: str, message: str) -> None:
    (repository / name).write_text(contents)
    git(repository, "add", name)
    git(repository, "commit", "-m", message)


def write_handoff_note(working_directory: Path, issue_number: int, body: str) -> None:
    """Stand in for the handoff skill's own final, separate note commit."""

    (working_directory / ".agent" / "handoff").mkdir(parents=True, exist_ok=True)
    write_and_commit(working_directory, f".agent/handoff/{issue_number}.md", body, f"Handoff note: issue #{issue_number}")


def remote_branch_subjects(remote: Path, branch: str) -> list[str]:
    output = subprocess.run(
        ("git", "-C", str(remote), "log", branch, "--format=%s"),
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return output.strip().splitlines()


def remote_has_branch(remote: Path, branch: str) -> bool:
    result = subprocess.run(
        ("git", "-C", str(remote), "rev-parse", "--verify", "--quiet", branch),
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def git(cwd: Path, *arguments: str) -> str:
    return subprocess.run(("git", *arguments), cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()
