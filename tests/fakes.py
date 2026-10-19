"""Shared test doubles and repository helpers for the whole suite.

``FakeTracker``, ``InMemoryWorkspace`` and ``FakePublisher`` are defined
exactly once, here: test modules must import them instead of redefining
their own. ``InMemoryWorkspace`` is the in-memory :class:`AttemptWorkspace
<simple_coding_agent.attempt_workspace.AttemptWorkspace>` adapter; the
conformance suite runs the same tests against it and the production
``GitWorkspace``. The ``repository_with_main``/``git``/``write_and_commit``
helpers build a temporary bare remote the way the end-to-end handoff test
does.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
import shutil
import subprocess

from simple_coding_agent.attempt_workspace import (
    ConflictState,
    handoff_note_relpath,
)
from simple_coding_agent.completion import AttemptOutcome
from simple_coding_agent.git_workspace import (
    AttemptCommit,
    DirtyWorkspaceError,
    GitWorkspaceError,
    PreparedAttempt,
)
from simple_coding_agent.github_tracker import (
    MERGE_CONFLICT,
    READY_FOR_AGENT,
    ROUND_FINISHED,
    Assignment,
    Claim,
    TrackerIssue,
    conflict_branch_issue_number,
    is_conflict_detection_eligible,
)
from simple_coding_agent.command_runner import (
    CommandExecution,
    CommandResult,
    CommandStatus,
)
from simple_coding_agent.model_execution import ModelExecution, ModelExecutionStatus
from simple_coding_agent.publication import PullRequest


# ---------------------------------------------------------------------------
# git repository helpers (temporary bare remote)
# ---------------------------------------------------------------------------


def git(cwd: Path, *arguments: str) -> str:
    """Run a git subcommand in ``cwd`` with a non-interactive editor."""

    return run_command(("git", *arguments), cwd=cwd)


def run_command(
    command: tuple[str, ...], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> str:
    """Run a subcommand, never falling back to the caller's real $EDITOR."""

    if env is None:
        # Never let a git subcommand (e.g. "rebase --continue") fall back to
        # the caller's real $EDITOR: on a real terminal that spawns an
        # interactive editor attached to it and hangs the test run.
        env = {**os.environ, "GIT_EDITOR": "true", "GIT_SEQUENCE_EDITOR": "true"}
    return subprocess.run(
        command, cwd=cwd, env=env, check=True, text=True, capture_output=True
    ).stdout.strip()


def write_and_commit(repository: Path, name: str, contents: str, message: str) -> None:
    """Write ``name`` into ``repository`` and commit it with ``message``."""

    (repository / name).write_text(contents)
    git(repository, "add", name)
    git(repository, "commit", "-m", message)


def repository_with_main(tmp_path: Path) -> tuple[Path, Path]:
    """Create a bare remote plus a seeded ``main`` checkout; return both."""

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


# ---------------------------------------------------------------------------
# InMemoryWorkspace
# ---------------------------------------------------------------------------


@dataclass
class _RecordedCommit:
    revision: str
    subject: str
    body: str
    files: frozenset[str]
    prior_round: bool = False
    existing: frozenset[str] = frozenset()


class InMemoryWorkspace:
    """The in-memory attempt-workspace adapter shared by every test module.

    Simulates dirty and untracked files (a set of repository-relative
    paths), commits added since the base (newest first, like ``git log``),
    conflict state including an unresolved rebase, and branch position for
    cleanup assertions. Handoff notes are written as real files under
    ``working_directory`` so lifecycle note reads work unchanged.
    ``leave_conflict`` plus ``note_model_finished`` (wired as the model
    executor's ``after_execute`` hook) simulates the model resolving the
    conflict during its turn; pass ``resolve_after_model=False`` to simulate
    the model leaving it unresolved.
    """

    def __init__(
        self,
        root: Path,
        *,
        issue_number: int = 24,
        base_revision: str = "base-revision-0",
        resolve_after_model: bool = True,
    ) -> None:
        self.working_directory = root
        root.mkdir(parents=True, exist_ok=True)
        self._branch = f"agent/issue-{issue_number}"
        self._base_revision = base_revision
        self._resolve_after_model = resolve_after_model
        self._model_finished = False
        self.current_branch = self._branch
        self.dirty: set[str] = set()
        # Artifacts ignored on the attempt branch but exposed as untracked on
        # the base branch (removed by cleanup, like the production sweep);
        # and artifacts ignored on both branches (invisible, always kept).
        self.ignored: set[str] = set()
        self.base_ignored: set[str] = set()
        self._commits: list[_RecordedCommit] = []
        self._sequence = 0
        self._conflict = ConflictState(in_progress=False)
        # Conflicted files ``prepare_attempt`` reports (and leaves in
        # progress) to simulate a conflicting rebase for conflict attempts.
        self.prepare_conflicts: tuple[str, ...] = ()
        self.abort_calls: list[object] = []
        self.cleanup_calls: list[tuple[str, bool]] = []
        self.seal_calls: list[tuple[int, str, ModelExecutionStatus]] = []
        self.note_calls: list[tuple[int, str]] = []
        self.bootstrap_bases: list[str] = []
        self.prepared_bases: list[str] = []

    # -- file simulation ----------------------------------------------------

    def make_dirty(self, name: str, contents: str = "dirty work") -> Path:
        """Write ``name`` under the working directory and leave it uncommitted."""

        path = self.working_directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        self.dirty.add(name)
        return path

    def make_ignored(self, name: str, contents: str = "generated") -> Path:
        """Leave a branch-ignored artifact: invisible to ``is_clean``.

        Mirrors a generated file the attempt branch gitignores but the base
        branch does not: ``cleanup`` removes it after switching to the base
        instead of holding intake on it.
        """

        path = self.make_dirty(name, contents)
        self.ignored.add(name)
        return path

    def make_base_ignored(self, name: str, contents: str = "cached") -> Path:
        """Leave an artifact ignored on both branches: invisible, always kept.

        Mirrors dependency installs and virtualenvs: ``cleanup`` never
        removes these because they stay ignored after the switch to base.
        """

        path = self.make_dirty(name, contents)
        self.base_ignored.add(name)
        return path

    # -- AttemptWorkspace ----------------------------------------------------

    def prepare_for_profile_read(self, *, base_branch: str = "main") -> str:
        self.bootstrap_bases.append(base_branch)
        return self._base_revision

    def prepare_attempt(self, *, base_branch: str, issue_number: int) -> PreparedAttempt:
        self.prepared_bases.append(base_branch)
        self._branch = f"agent/issue-{issue_number}"
        self.current_branch = self._branch
        if self.prepare_conflicts:
            # Simulate a published branch whose rebase onto the base hits
            # conflicts: the rebase stays in progress with its files reported,
            # like the production workspace, for the caller to abort.
            self.leave_conflict(self.prepare_conflicts)
            return PreparedAttempt(
                branch=self._branch,
                base_revision=self._base_revision,
                rebase_conflicts=tuple(self.prepare_conflicts),
            )
        return PreparedAttempt(branch=self._branch, base_revision=self._base_revision)

    def is_clean(self) -> bool:
        return not (self.dirty - self.ignored - self.base_ignored)

    def dirty_paths(self) -> tuple[str, ...]:
        return tuple(sorted(self.dirty - self.ignored - self.base_ignored))

    def commits_added(self, prepared: PreparedAttempt) -> tuple[AttemptCommit, ...]:
        return tuple(
            AttemptCommit(revision=record.revision, subject=record.subject)
            for record in reversed(self._commits)
        )

    def files_changed(self, revision: str) -> tuple[str, ...]:
        """Repository-relative paths one recorded commit touched, if known."""

        record = next((entry for entry in self._commits if entry.revision == revision), None)
        if record is None:
            return ()
        return tuple(sorted(record.files))

    def drop_newest_commits(self, count: int) -> None:
        """Remove the newest ``count`` commits, simulating a squash or drop."""

        for _ in range(max(0, count)):
            if self._commits:
                self._commits.pop()

    def commit_bodies(self) -> tuple[str, ...]:
        """Newest-first commit bodies (subjects excluded); for trailer assertions."""

        return tuple(record.body for record in reversed(self._commits))

    def commit_files(self) -> tuple[frozenset[str], ...]:
        """Newest-first paths touched by each commit; for note-only assertions."""

        return tuple(record.files for record in reversed(self._commits))

    def conflict_state(self) -> ConflictState:
        return self._conflict

    def leave_conflict(
        self,
        files: tuple[str, ...] = ("conflicted.txt",),
        problems: tuple[str, ...] | None = None,
    ) -> None:
        """Simulate an unresolved rebase/merge left in the worktree."""

        if problems is None:
            details = f" Conflicting files: {', '.join(files)}." if files else ""
            problems = (f"A rebase or merge is still in progress.{details}",)
        self._conflict = ConflictState(
            in_progress=True, conflicted_files=tuple(files), resolution_problems=tuple(problems)
        )

    def resolve_conflict(self) -> None:
        """Clear a simulated conflict, as if the model resolved it."""

        self._conflict = ConflictState(in_progress=False)

    def note_model_finished(self) -> None:
        """Simulate the model's turn ending so a resolution can take effect.

        Wire as the model executor's ``after_execute`` hook: when
        ``resolve_after_model`` holds (the default), a conflict left via
        ``leave_conflict`` is resolved as if the model resolved it during
        its turn; otherwise the conflict stays in place.
        """

        self._model_finished = True
        if self._resolve_after_model:
            self.resolve_conflict()

    def abort_unresolved_rebase(self, prepared: PreparedAttempt) -> None:
        self.abort_calls.append(prepared)
        self._conflict = ConflictState(in_progress=False)

    def seal(
        self,
        *,
        issue_number: int,
        attempt_id: str,
        status: ModelExecutionStatus,
    ) -> AttemptCommit | None:
        self.seal_calls.append((issue_number, attempt_id, status))
        if self._conflict.in_progress:
            raise GitWorkspaceError(
                "Cannot seal attempt work while a rebase or merge is unresolved"
            )
        if not self.dirty:
            return None
        subject = f"Seal attempt work for #{issue_number} ({status})"
        body = f"Attempt-Id: {attempt_id}\n"
        return self._record_commit(subject, body, frozenset(self.dirty), clear_dirty=True)

    def commit_handoff_note(self, issue_number: int, content: str) -> AttemptCommit:
        self.note_calls.append((issue_number, content))
        relpath = handoff_note_relpath(issue_number)
        path = self.working_directory / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        subject = f"Handoff note: issue #{issue_number}"
        return self._record_commit(subject, "", frozenset({relpath}), clear_dirty=False)

    def model_commit(self, subject: str, files: tuple[str, ...] = ()) -> AttemptCommit:
        """Simulate the model committing its own work (for regression tests)."""

        touched = frozenset(files) if files else frozenset(self.dirty)
        for name in touched:
            self.dirty.discard(name)
        return self._record_commit(subject, "", touched, clear_dirty=False)

    def prior_round_commit(self, subject: str, files: tuple[str, ...]) -> AttemptCommit:
        """Simulate a commit carried over from an earlier round of the branch.

        ``handoff_note_commit`` ignores it, as ``GitWorkspace`` ignores
        commits authored before the attempt started.
        """

        return self._record_commit(subject, "", frozenset(files), clear_dirty=False, prior_round=True)

    def handoff_note_commits(
        self, prepared: PreparedAttempt, issue_number: int, *, attempt_id: str
    ) -> tuple[AttemptCommit, ...]:
        relpath = handoff_note_relpath(issue_number)
        return tuple(
            AttemptCommit(revision=record.revision, subject=record.subject)
            for record in reversed(self._commits)
            if relpath in record.files and not record.prior_round
        )

    def handoff_note_commit(
        self, prepared: PreparedAttempt, issue_number: int, *, attempt_id: str
    ) -> AttemptCommit | None:
        commits = self.handoff_note_commits(
            prepared, issue_number, attempt_id=attempt_id
        )
        return commits[0] if commits else None

    def handoff_note_commit_carries_work(self, revision: str, issue_number: int) -> bool:
        relpath = handoff_note_relpath(issue_number)
        record = next((entry for entry in self._commits if entry.revision == revision), None)
        if record is None:
            return True
        touches_other_files = any(name != relpath for name in record.files)
        return touches_other_files and relpath not in record.existing

    def cleanup(
        self, *, base_branch: str, prepared: PreparedAttempt, retain_branch: bool
    ) -> None:
        self.cleanup_calls.append((base_branch, retain_branch))
        self._conflict = ConflictState(in_progress=False)
        if self.dirty - self.ignored - self.base_ignored:
            raise DirtyWorkspaceError(
                "Repository working tree contains uncommitted or untracked changes; "
                "leaving the workspace on the attempt branch for operator inspection."
            )
        # The base branch exposes branch-ignored artifacts as untracked, so
        # they are removed instead of holding intake; base-ignored paths stay
        # ignored after the switch and are kept. Anything unremovable stops
        # counting as ignored, so it holds intake like the production sweep.
        removed: set[str] = set()
        for name in sorted(self.dirty & self.ignored):
            path = self.working_directory / name
            removable = path.is_symlink() or path.is_file() or path.is_dir()
            if removable:
                try:
                    if path.is_symlink() or path.is_file():
                        path.unlink(missing_ok=True)
                    else:
                        shutil.rmtree(path)
                except OSError:
                    removable = False
            if not removable:
                self.ignored.discard(name)
                continue
            removed.add(name)
        self.dirty -= removed
        self.ignored -= removed
        self.current_branch = base_branch

    # -- internals ------------------------------------------------------------

    def _record_commit(
        self,
        subject: str,
        body: str,
        files: frozenset[str],
        *,
        clear_dirty: bool,
        prior_round: bool = False,
    ) -> AttemptCommit:
        self._sequence += 1
        record = _RecordedCommit(
            revision=f"commit-{self._sequence:04d}",
            subject=subject,
            body=body,
            files=files,
            prior_round=prior_round,
            existing=frozenset(
                name for name in files if (self.working_directory / name).exists()
            ),
        )
        self._commits.append(record)
        if clear_dirty:
            self.dirty.clear()
        return AttemptCommit(revision=record.revision, subject=record.subject)


# ---------------------------------------------------------------------------
# FakeTracker
# ---------------------------------------------------------------------------


class FakeTracker:
    """The tracker double shared by every test module.

    ``issues`` maps numbers to current remote state (``fetch_issue``,
    ``recover_claim``). ``fifo`` is the claim queue drained by
    ``claim_next``; the one-shot ``next_claim`` takes precedence and may
    also be assigned after construction. ``fetch_errors``/``assign_errors``
    inject transport failures per issue number. Identity follows the
    production convention: the viewer login is ``agent`` and fresh claims
    carry assignee id ``agent-id``.
    """

    def __init__(
        self,
        issues: dict[int, TrackerIssue] | None = None,
        *,
        fifo: list[int] | None = None,
        next_claim: Claim | None = None,
        auto_counts: dict[int, int] | None = None,
        open_pull_requests: dict[str, PullRequest] | None = None,
        max_conflict_attempts: int = 0,
        event_log: Callable[..., None] | None = None,
    ) -> None:
        self.issues = dict(issues or {})
        self.fifo: list[int] = list(fifo or [])
        self.next_claim = next_claim
        self.auto_counts = dict(auto_counts or {})
        if next_claim is not None:
            self.issues.setdefault(next_claim.issue.number, next_claim.issue)
        self.fetch_errors: dict[int, Exception] = {}
        self.assign_errors: dict[int, Exception] = {}
        self.claimed: list[int] = []
        self.fifo_claims: list[int] = []
        self.claimed_verified: list[int] = []
        self.cleanup: list[tuple[int, str, str]] = []
        self.removed_labels: list[tuple[str, str]] = []
        self.added_labels: list[tuple[str, str]] = []
        # Open pull requests keyed by head branch, mirroring the production
        # detector's paged query; ``None`` means the listing is unavailable.
        self.open_pull_requests = dict(open_pull_requests or {})
        self.has_pull_request_listing = open_pull_requests is not None
        self.max_conflict_attempts = max_conflict_attempts
        self._event_log = (
            event_log
            if event_log is not None
            else lambda event, detail="", level="INFO", issue_number=None: None
        )
        # One entry per poll that ran the detector query (budget above zero).
        self.pr_queries: list[None] = []

    def detect_conflict_requeues(self) -> list[tuple[int, int, str]]:
        """Mirror the production detector: label conflicting published branches.

        With a positive conflict budget and a pull-request listing, add
        ``merge-conflict`` to the issue behind every open, non-draft
        ``agent/issue-N`` branch that reports ``CONFLICTING``, under the same
        rule as :func:`is_conflict_detection_eligible
        <simple_coding_agent.github_tracker.is_conflict_detection_eligible>`.
        Newly labelled issues join the front of the claim queue in FIFO order
        so the same poll can claim them. Returns the labelled ``(issue, pull
        request, head sha)`` triples in listing order.
        """

        if self.max_conflict_attempts <= 0 or not self.has_pull_request_listing:
            return []
        self.pr_queries.append(None)
        detected: list[tuple[int, int, str]] = []
        for head, pull_request in self.open_pull_requests.items():
            number = conflict_branch_issue_number(head)
            if number is None:
                continue
            issue = self.issues.get(number)
            if issue is None:
                continue
            if not is_conflict_detection_eligible(
                issue, draft=pull_request.draft, mergeable=pull_request.mergeable
            ):
                continue
            self.issues[number] = replace(issue, labels=issue.labels | {MERGE_CONFLICT})
            self.added_labels.append((issue.id, MERGE_CONFLICT))
            detected.append((number, pull_request.number, pull_request.head_sha))
            self._event_log(
                "conflict_requeue_detected",
                f"issue #{number}; pull request #{pull_request.number}"
                f" ({pull_request.head_sha}) reports merge conflicts",
                level="INFO",
                issue_number=number,
            )
        newly = sorted(
            (number for number, _, _ in detected if number not in self.fifo),
            key=lambda number: (self.issues[number].created_at, number),
        )
        self.fifo = newly + self.fifo
        return detected

    def claim_next(self) -> Claim | None:
        self.detect_conflict_requeues()
        if self.next_claim is not None:
            claim, self.next_claim = self.next_claim, None
            self.claimed.append(claim.issue.number)
            return claim
        if not self.fifo:
            return None
        number = self.fifo.pop(0)
        self.fifo_claims.append(number)
        self.claimed.append(number)
        found = self.issues.get(number)
        if found is None:  # pragma: no cover - tests seed the queue from issues
            raise AssertionError(f"FakeTracker fifo issue #{number} is not seeded")
        return Claim(issue=found, assignment=Assignment(issue_id=found.id, assignee_id="agent-id"))

    def fetch_issue(self, number: int) -> TrackerIssue | None:
        if number in self.fetch_errors:
            raise self.fetch_errors[number]
        return self.issues.get(number)

    def is_self_assigned(self, issue: TrackerIssue) -> bool:
        return "agent" in issue.assignee_logins

    def automatic_continuation_count(self, issue: TrackerIssue) -> int:
        """The recorded automatic-continuation count for ``issue`` (default 0)."""

        return self.auto_counts.get(issue.number, 0)

    def claim_verified(self, issue: TrackerIssue) -> Claim:
        if issue.number in self.assign_errors:
            raise self.assign_errors[issue.number]
        current = self.issues.get(issue.number)
        assert current is not None
        if ROUND_FINISHED in current.labels:
            self.removed_labels.append((current.id, ROUND_FINISHED))
        if MERGE_CONFLICT in current.labels and (
            READY_FOR_AGENT in current.labels or ROUND_FINISHED in current.labels
        ):
            # The ordinary (or continuation) path takes precedence over the
            # conflict signal, so the stale label goes at the claim, exactly
            # like the production tracker.
            self.removed_labels.append((current.id, MERGE_CONFLICT))
        self.claimed_verified.append(issue.number)
        return Claim(issue=current, assignment=Assignment(issue_id=current.id, assignee_id="agent-id"))

    def recover_claim(self, issue_number: int) -> Claim | None:
        found = self.issues.get(issue_number)
        if found is None:
            return None
        return Claim(issue=found, assignment=Assignment(issue_id=found.id, assignee_id="agent-id"))

    def release_attempt(self, issue_number: int, label: str, assignee_id: str) -> None:
        self.cleanup.append((issue_number, label, assignee_id))

    def release_handoff(self, issue_number: int, assignee_id: str) -> None:
        self.cleanup.append((issue_number, "round-finished", assignee_id))


# ---------------------------------------------------------------------------
# FakePublisher
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublishedAttempt:
    """The verified remote result, mirroring ``PublicationResult`` fields."""

    outcome: AttemptOutcome
    branch_url: str | None
    comment_posted: bool = True
    pull_request: object = None


class FakePublisher:
    """The publisher double shared by every test module.

    By default an infrastructure-error outcome publishes no branch (like the
    production publisher), while every other outcome publishes the example
    URL. Pass ``branch_url`` explicitly (including ``None``) or ``outcome``
    to pin publication behaviour, and ``comment_posted=False`` to simulate
    an unconfirmed result comment. Pass ``pull_request`` (a ``PullRequest``)
    to simulate a published pull request. Per-call publication budgets
    arrive via the ``timeout`` keyword (like the production publisher) and
    are recorded on ``timeouts``.
    """

    _DEFAULT_BRANCH_URL = "https://example.test/tree/agent/issue-24"

    def __init__(
        self,
        *,
        outcome: AttemptOutcome | None = None,
        branch_url: str | None = _DEFAULT_BRANCH_URL,
        comment_posted: bool = True,
        pull_request: object = None,
        pull_requests: dict[str, object] | None = None,
    ) -> None:
        self._outcome = outcome
        self._branch_url = branch_url
        self._comment_posted = comment_posted
        self._pull_request = pull_request
        self._pull_requests = dict(pull_requests or {})
        self.requests: list[object] = []
        self.outcomes: list[AttemptOutcome] = []
        self.timeouts: list[float | None] = []
        self.pr_lookups: list[str] = []
        self.conflict_requests: list[object] = []

    def publish(self, request: object, *, timeout: float | None = None) -> PublishedAttempt:
        self.requests.append(request)
        self.timeouts.append(timeout)
        outcome = self._outcome or request.decision.outcome or AttemptOutcome.COMPLETE
        self.outcomes.append(outcome)
        if self._branch_url == FakePublisher._DEFAULT_BRANCH_URL:
            branch_url = None if outcome is AttemptOutcome.INFRASTRUCTURE_ERROR else self._branch_url
        else:
            branch_url = self._branch_url
        return PublishedAttempt(
            outcome=outcome,
            branch_url=branch_url,
            comment_posted=self._comment_posted,
            pull_request=self._pull_request,
        )

    def find_pull_request(self, head: str) -> object | None:
        """Return the seeded open pull request for an attempt branch, if any."""

        self.pr_lookups.append(head)
        return self._pull_requests.get(head)

    def publish_conflict(self, request: object, *, timeout: float | None = None) -> PublishedAttempt:
        """Publish one conflict attempt: a branch URL only for the push path."""

        self.requests.append(request)
        self.conflict_requests.append(request)
        self.timeouts.append(timeout)
        outcome = self._outcome or request.outcome
        self.outcomes.append(outcome)
        if self._branch_url == FakePublisher._DEFAULT_BRANCH_URL:
            branch_url = (
                f"https://example.test/tree/{request.branch}" if request.push_branch else None
            )
        else:
            branch_url = self._branch_url
        return PublishedAttempt(
            outcome=outcome,
            branch_url=branch_url,
            comment_posted=self._comment_posted,
            pull_request=request.pull_request,
        )


def ok_result(command: str = "pytest") -> CommandResult:
    """A succeeded profile-command result for conflict-attempt tests."""

    return CommandResult(
        CommandStatus.SUCCEEDED,
        (CommandExecution(command, 0, "", "", False),),
        0.1,
    )


def failed_result(command: str = "pytest", exit_code: int = 1) -> CommandResult:
    """A failed profile-command result for conflict-attempt tests."""

    return CommandResult(
        CommandStatus.FAILED,
        (CommandExecution(command, exit_code, "", "boom", False),),
        0.1,
    )


class FakeCommandRunner:
    """Serve scripted profile-command results to conflict attempts."""

    def __init__(self, results: list[CommandResult]) -> None:
        self._results = list(results)
        self.calls: list[tuple[tuple[str, ...], float, Path, object]] = []

    def run(
        self,
        commands: tuple[str, ...],
        *,
        timeout: float,
        cwd: Path,
        environment: object = None,
    ) -> CommandResult:
        self.calls.append((tuple(commands), timeout, cwd, environment))
        return self._results.pop(0)


class FakeConflictExecutor:
    """Serve one scripted model resolution session to a conflict attempt."""

    def __init__(
        self,
        *,
        status: ModelExecutionStatus = ModelExecutionStatus.SUCCEEDED,
        calls: list[str] | None = None,
    ) -> None:
        self.captured_prompt: str | None = None
        self._status = status
        self._calls = calls
        self.after_execute = None

    async def execute(
        self,
        *,
        issue_body: str,
        working_directory: Path,
        issue_number: int | None = None,
        archive: object = None,
        attempt_id: str | None = None,
        branch: str | None = None,
        base_revision: str | None = None,
    ) -> ModelExecution:
        if self._calls is not None:
            self._calls.append("execute")
        self.captured_prompt = issue_body
        if self.after_execute is not None:
            self.after_execute()
        return ModelExecution(
            status=self._status,
            explanation="stub",
            stop_reason="end_turn",
            model_usage=None,
            observed_models=(),
            skill_events=(),
        )
