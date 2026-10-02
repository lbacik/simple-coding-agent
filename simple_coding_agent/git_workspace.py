"""Git workspace boundary for one persistent repository clone."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time
from urllib.parse import urlsplit

from simple_coding_agent.model_execution import ModelExecutionStatus


class GitWorkspaceError(RuntimeError):
    """Raised when the persistent git workspace cannot be used safely."""


class GitWorkspaceRecoveryError(GitWorkspaceError):
    """Raised when the local workspace is broken in a way this attempt cannot fix.

    Unlike other ``GitWorkspaceError`` cases, this is a fault in the agent's own
    persistent clone rather than in the claimed issue, so callers must not treat
    it as attempt failure: no result comment, no label release.
    """


class DirtyWorkspaceError(GitWorkspaceError):
    """Raised when the working tree has uncommitted changes before an attempt starts."""


class RebaseConflictError(GitWorkspaceError):
    """Raised when a reused attempt branch still conflicts after the attempt's turn.

    Workspace preparation leaves a conflicting rebase in place for the
    implementation attempt to resolve as the first step of its normal model
    session. This is
    raised only when the conflict is still unresolved afterwards: the rebase
    has been aborted, the branch restored to its pre-rebase state, and the
    working tree no longer holds conflict markers. Setup must not run after
    this error: the branch is stale relative to base and the attempt fails
    fast as an infrastructure error with a ``rebase_conflict`` cause instead
    of running setup.
    """

    def __init__(
        self,
        message: str,
        *,
        branch: str,
        base_branch: str,
        conflicted_files: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.branch = branch
        self.base_branch = base_branch
        self.conflicted_files = conflicted_files


@dataclass(frozen=True)
class PreparedAttempt:
    """The branch and verified base revision for one implementation attempt."""

    branch: str
    base_revision: str
    restored_from_remote: bool = False
    rebase_conflicts: tuple[str, ...] = ()
    pre_rebase_revision: str | None = None


@dataclass(frozen=True)
class AttemptCommit:
    """A commit added on an attempt branch after its verified base."""

    revision: str
    subject: str


@dataclass(frozen=True)
class ConflictState:
    """One snapshot of the workspace's rebase/merge conflict state.

    ``in_progress`` reports an unfinished rebase/merge, ``conflicted_files``
    the paths the model must resolve, and ``resolution_problems`` the
    deterministic post-model verification problems (markers, whitespace
    errors, unmerged paths). An empty ``resolution_problems`` means the
    tree is safe for setup. The state fails safe: a tree whose state cannot
    be read reports ``in_progress=True`` with a verification problem
    instead of looking clean.
    """

    in_progress: bool
    conflicted_files: tuple[str, ...] = ()
    resolution_problems: tuple[str, ...] = ()


def handoff_note_relpath(issue_number: int) -> str:
    """The repository-relative path of an issue's handoff note."""

    return f".agent/handoff/{issue_number}.md"


def attempt_start_epoch(attempt_id: str) -> int | None:
    """The attempt start as whole epoch seconds, or ``None`` if unparseable.

    Truncated to seconds because git author dates have second precision: a
    commit made in the same second the attempt started still belongs to it.
    """

    try:
        started = datetime.fromisoformat(attempt_id.replace("Z", "+00:00"))
    except ValueError:
        return None
    if started.tzinfo is None:
        return None
    return int(started.timestamp())


GitRunner = Callable[[tuple[str, ...], Path | None, dict[str, str]], str]


class GitWorkspace:
    """Own the clone, branch preparation, inspection, and local cleanup."""

    def __init__(
        self,
        clone_dir: Path,
        repository_url: str,
        *,
        token_provider: Callable[[], str],
        run: GitRunner | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self._clone_dir = clone_dir
        self._repository_url = repository_url
        self._token_provider = token_provider
        self._run = run or _run_git
        self._sleeper = sleeper or time.sleep

    @property
    def working_directory(self) -> Path:
        """The persistent clone directory used by profile commands and the SDK."""

        return self._clone_dir

    def is_clean(self) -> bool:
        """Return True if working tree has no uncommitted or untracked changes."""
        if not (self._clone_dir / ".git").exists():
            return True
        return not bool(self.dirty_status().strip())

    def dirty_status(self) -> str:
        """Return porcelain status of dirty or untracked changes."""
        return self._git("status", "--porcelain")

    def dirty_paths(self) -> tuple[str, ...]:
        """Return repository-relative paths of dirty or untracked changes.

        Parsed from ``git status --porcelain`` (v1): the path follows the
        two-column status and separator, so ``XY path`` yields ``path`` and
        a rename ``R  old -> new`` yields ``new``. Unreadable state
        propagates, so callers fail closed instead of treating the tree as
        clean.
        """

        return tuple(
            _porcelain_path(line)
            for line in self.dirty_status().splitlines()
            if line.strip()
        )

    def has_unresolved_conflicts(self) -> bool:
        """Return True when the working tree holds an unfinished rebase/merge.

        ``prepare_attempt`` deliberately leaves a conflicting rebase in place
        for the model session to resolve, so callers must not run repository
        setup commands against such a tree: the conflict markers look like
        syntax errors to every parser the setup may invoke.
        """

        git_dir = self._clone_dir / ".git"
        if not git_dir.is_dir():
            return False
        if (
            (git_dir / "rebase-merge").exists()
            or (git_dir / "rebase-apply").exists()
            or (git_dir / "MERGE_HEAD").exists()
        ):
            return True
        return bool(self._git("ls-files", "--unmerged").strip())

    def prepare_for_profile_read(self, *, base_branch: str = "main") -> str:
        """Fetch and check out the base branch so the repository profile can be read.

        Unlike ``prepare_attempt``, this never touches attempt branches: no
        branch is created, checked out, or rebased. It exists so callers can
        load the profile (which names the real base branch) before preparing
        the attempt exactly once onto that base. Returns the verified base
        revision.
        """

        self._ensure_clone()
        if not self.is_clean():
            raise DirtyWorkspaceError(
                f"Repository working tree contains uncommitted or untracked changes:\n{self.dirty_status()}"
            )
        self._git("fetch", "origin")
        base_revision = self._ensure_verified_base(base_branch)
        self._git("checkout", base_branch)
        return base_revision

    def prepare_attempt(self, *, base_branch: str, issue_number: int) -> PreparedAttempt:
        """Fetch, verify, and check out the branch used by an attempt.

        The local base is first repaired to match origin (see
        ``_ensure_verified_base``): setup must never run against stale or
        divergent code.

        A previously published attempt branch (for example, a handoff) is
        always preferred over whatever the local clone happens to have: the
        remote ref is fetched and, when present, used as the branch's
        starting point regardless of local state, so a fresh clone resumes
        published work instead of silently starting over from base.
        """

        branch = _attempt_branch(issue_number)
        self._ensure_clone()
        if not self.is_clean():
            raise DirtyWorkspaceError(
                f"Repository working tree contains uncommitted or untracked changes:\n{self.dirty_status()}"
            )
        self._git("fetch", "origin")
        base_revision = self._ensure_verified_base(base_branch)

        remote_branch_revision = self._fetch_attempt_branch(branch)
        if remote_branch_revision is not None:
            self._git("checkout", base_branch)
            self._git("branch", "-f", branch, remote_branch_revision)
            self._git("checkout", branch)
            pre_rebase_revision, rebase_conflicts = self._rebase_onto_base(branch, base_branch)
        elif self._branch_exists(branch):
            self._git("checkout", branch)
            pre_rebase_revision, rebase_conflicts = self._rebase_onto_base(branch, base_branch)
        else:
            self._git("checkout", "-b", branch, base_branch)
            pre_rebase_revision, rebase_conflicts = None, ()
        return PreparedAttempt(
            branch=branch,
            base_revision=base_revision,
            restored_from_remote=remote_branch_revision is not None,
            rebase_conflicts=rebase_conflicts,
            pre_rebase_revision=pre_rebase_revision,
        )

    def _ensure_verified_base(self, base_branch: str) -> str:
        """Fetch-verified revision of the base branch, repairing a stale local base.

        A local base that differs from ``origin/<base_branch>`` is never used
        as-is: setup must never run against stale or divergent code. This is
        normally just a leftover from an interrupted previous run, so it is
        first repaired by resetting the local base to match origin. Only a
        base that still disagrees after that repair is a broken workspace,
        raised as ``GitWorkspaceRecoveryError``.
        """

        base_revision = self._revision(f"origin/{base_branch}", remote=True)
        try:
            local_base_revision: str | None = self._revision(base_branch, remote=False)
        except GitWorkspaceError:
            # A fresh clone has no local branch for a non-default base yet;
            # repairing below creates it from origin.
            local_base_revision = None
        if local_base_revision != base_revision:
            self._reset_local_base(base_branch)
            local_base_revision = self._revision(base_branch, remote=False)
            if local_base_revision != base_revision:
                raise GitWorkspaceRecoveryError(
                    "Local base branch has diverged from origin and could not be repaired"
                )
        return base_revision

    def _fetch_attempt_branch(self, branch: str) -> str | None:
        """Best-effort fetch of a published attempt branch; a missing ref is not an error."""

        try:
            self._git("fetch", "origin", branch)
        except GitWorkspaceError:
            return None
        return self._remote_branch_revision(branch)

    def commits_added(self, prepared: PreparedAttempt) -> tuple[AttemptCommit, ...]:
        """Return commits reachable from the attempt branch but not its base."""

        output = self._git(
            "log",
            "--format=%H%x00%s",
            f"{prepared.base_revision}..{prepared.branch}",
        )
        if not output:
            return ()
        commits: list[AttemptCommit] = []
        for line in output.splitlines():
            revision, separator, subject = line.partition("\x00")
            if not separator or not revision:
                raise GitWorkspaceError("Git returned an invalid commit listing")
            commits.append(AttemptCommit(revision=revision, subject=subject))
        return tuple(commits)

    def conflict_state(self) -> ConflictState:
        """Return one snapshot of the rebase/merge conflict state.

        Fails safe: a tree whose state cannot be read is reported as
        conflicted with a verification problem, so setup never runs
        against a tree that could not be verified.
        """

        try:
            in_progress = self.has_unresolved_conflicts()
        except Exception:
            return ConflictState(
                in_progress=True,
                conflicted_files=(),
                resolution_problems=(
                    "Conflict state could not be verified; treating the tree as conflicted.",
                ),
            )
        try:
            files = self.conflicted_files()
        except Exception:
            files = ()
        try:
            problems = self.rebase_resolution_problems()
        except Exception:
            problems = (
                "Conflict-resolution verification failed; treating the tree as conflicted.",
            )
        if in_progress and not problems:
            problems = (
                "A rebase or merge is still in progress or paths remain unmerged.",
            )
        return ConflictState(
            in_progress=in_progress,
            conflicted_files=files,
            resolution_problems=problems,
        )

    def seal(
        self,
        *,
        issue_number: int,
        attempt_id: str,
        status: ModelExecutionStatus,
    ) -> AttemptCommit | None:
        """Commit every dirty and untracked change on the attempt branch.

        This is sealing: the runner calls it unconditionally after every
        exit from model execution, whatever the model execution status, so
        new statuses cannot silently drop work. Returns ``None`` when the
        tree is already clean. Refuses (raises) while a rebase/merge is
        unresolved, so conflict markers are never committed. The commit is
        attributed to its issue and status in the subject and to its attempt
        by an ``Attempt-Id`` trailer.
        """

        if self.conflict_state().in_progress:
            raise GitWorkspaceError(
                "Cannot seal attempt work while a rebase or merge is unresolved"
            )
        if self.is_clean():
            return None
        subject = f"Seal attempt work for #{issue_number} ({status})"
        self._git("add", "-A")
        self._git("commit", "--message", f"{subject}\n\nAttempt-Id: {attempt_id}\n")
        revision = self._git("rev-parse", "HEAD")
        return AttemptCommit(revision=revision, subject=subject)

    def commit_handoff_note(self, issue_number: int, content: str) -> AttemptCommit:
        """Write the handoff note file and commit only the note.

        Owns the note path (``.agent/handoff/<N>.md``): unlike sealing,
        this stages just the note file, so other uncommitted work is left
        for the seal commit that always runs first.
        """

        relpath = handoff_note_relpath(issue_number)
        note_path = self._clone_dir / relpath
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(content)
        self._git("add", "--", relpath)
        subject = f"Handoff note: issue #{issue_number}"
        self._git("commit", "--message", subject)
        revision = self._git("rev-parse", "HEAD")
        return AttemptCommit(revision=revision, subject=subject)

    def handoff_note_commit(
        self, prepared: PreparedAttempt, issue_number: int, *, attempt_id: str
    ) -> AttemptCommit | None:
        """Return the newest commit of this attempt touching the handoff note path.

        Position-independent: a seal commit landing on top of a
        model-committed note does not hide the note. Notes carried over from
        earlier rounds of a continued branch are ignored: only commits
        authored at or after the attempt start (``attempt_id`` is the
        checkpoint's ``started_at``) count. Author dates survive the rebase
        onto the base branch, unlike revisions. An unparseable
        ``attempt_id`` applies no time bound.
        """

        started = attempt_start_epoch(attempt_id)
        output = self._git(
            "log",
            "--format=%H%x00%at%x00%s",
            f"{prepared.base_revision}..{prepared.branch}",
            "--",
            handoff_note_relpath(issue_number),
        )
        for line in output.splitlines():
            revision, _, rest = line.partition("\x00")
            authored, separator, subject = rest.partition("\x00")
            if not separator or not revision or not authored.isdigit():
                raise GitWorkspaceError("Git returned an invalid commit listing")
            if started is None or int(authored) >= started:
                return AttemptCommit(revision=revision, subject=subject)
        return None

    def cleanup(
        self, *, base_branch: str, prepared: PreparedAttempt, retain_branch: bool = True
    ) -> None:
        """Best-effort abort of an unresolved rebase and switch to base_branch.

        Never delete files or branches: uncommitted/untracked files and attempt
        branches are preserved as-is for human operator inspection and cleanup.

        Never switch to the base branch while the tree is dirty: the
        workspace stays on the attempt branch and raises
        ``DirtyWorkspaceError`` so the caller can hold intake for operator
        inspection instead of carrying uncommitted work onto the base.
        """

        try:
            self._git("rebase", "--abort")
        except GitWorkspaceError:
            pass
        if not self.is_clean():
            raise DirtyWorkspaceError(
                "Repository working tree contains uncommitted or untracked changes; "
                "leaving the workspace on the attempt branch for operator inspection."
            )
        try:
            self._git("checkout", base_branch)
        except GitWorkspaceError:
            pass

    def push_attempt_branch(
        self, branch: str, *, max_retries: int, deadline: float | None = None
    ) -> str:
        """Push an attempt branch using an explicit, observed force-with-lease.

        The operation is independently callable during recovery.  A failed
        push is first reconciled with the remote because the server may have
        accepted it before the client lost its response.
        """

        if max_retries <= 0:
            raise GitWorkspaceError("Push retry count must be positive")
        local_revision = self._revision(branch, remote=False)
        last_error: GitWorkspaceError | None = None
        for attempt in range(max_retries):
            if deadline is not None and time.monotonic() >= deadline:
                raise GitWorkspaceError("PUBLISH_TIMEOUT exceeded while pushing")
            try:
                observed = self._fetch_remote_branch(branch)
                expected = observed or ""
                self._git(
                    "push",
                    f"--force-with-lease={branch}:{expected}",
                    "origin",
                    f"{branch}:refs/heads/{branch}",
                )
                if self._remote_branch_revision(branch) == local_revision:
                    return local_revision
                raise GitWorkspaceError("Pushed branch does not match the local revision")
            except GitWorkspaceError as error:
                last_error = error
                try:
                    if self._remote_branch_revision(branch) == local_revision:
                        return local_revision
                except GitWorkspaceError:
                    pass
                if attempt + 1 < max_retries:
                    delay = min(2 * 2**attempt, 30)
                    if deadline is not None and time.monotonic() + delay >= deadline:
                        raise GitWorkspaceError("PUBLISH_TIMEOUT exceeded while retrying push") from error
                    self._sleeper(delay)
        raise GitWorkspaceError("Attempt branch could not be pushed safely") from last_error

    def _fetch_remote_branch(self, branch: str) -> str | None:
        """Fetch an attempt branch, confirming an absent ref rather than guessing."""

        try:
            self._git("fetch", "origin", branch)
        except GitWorkspaceError as fetch_error:
            try:
                observed = self._remote_branch_revision(branch)
            except GitWorkspaceError:
                raise fetch_error from None
            if observed is not None:
                raise fetch_error
            return None
        return self._remote_branch_revision(branch)

    def _remote_branch_revision(self, branch: str) -> str | None:
        output = self._git("ls-remote", "--heads", "origin", f"refs/heads/{branch}")
        if not output:
            return None
        revision, separator, reference = output.partition("\t")
        if not separator or reference != f"refs/heads/{branch}" or not revision:
            raise GitWorkspaceError("Remote branch revision is invalid")
        return revision

    def _ensure_clone(self) -> None:
        _reject_url_credentials(self._repository_url)
        if self._clone_dir.exists():
            if not self._clone_dir.is_dir() or not (self._clone_dir / ".git").exists():
                raise GitWorkspaceError("CLONE_DIR exists but is not a git clone")
            if self._git("remote", "get-url", "origin") != self._repository_url:
                raise GitWorkspaceError("CLONE_DIR is not the configured repository")
            return
        self._clone_dir.parent.mkdir(parents=True, exist_ok=True)
        self._git("clone", self._repository_url, str(self._clone_dir), cwd=None)

    def _revision(self, reference: str, *, remote: bool) -> str:
        try:
            return self._git("rev-parse", "--verify", reference)
        except GitWorkspaceError as error:
            location = "origin" if remote else "local clone"
            raise GitWorkspaceError(f"Base branch is unavailable in the {location}") from error

    def _rebase_onto_base(self, branch: str, base_branch: str) -> tuple[str | None, tuple[str, ...]]:
        """Replay a reused attempt branch onto the just-verified base.

        A retained branch from a failed prior attempt (for example, a setup
        failure) must never keep running against the base as it stood back
        then: repository-owned config such as the setup profile has to be
        read fresh. Rebasing preserves any commits on the branch, published
        or not.

        A conflicting rebase is deliberately left in progress and reported
        through the returned conflicted files, so the model session can
        resolve it as the first step of its normal session before setup ever
        runs. The branch ref still points at the pre-rebase tip while the
        working tree shows the conflict markers. Returns the pre-rebase
        branch revision (used to restore the branch when the model leaves
        the conflict unresolved) together with the conflicted files, or
        ``(None, ())`` when the rebase applied cleanly.
        """

        pre_rebase_revision = self._git("rev-parse", "--verify", branch)
        try:
            self._git("rebase", base_branch)
        except GitWorkspaceError:
            conflicted = self.conflicted_files()
            if conflicted or self._rebase_in_progress():
                return pre_rebase_revision, conflicted
            raise
        return None, ()

    def conflicted_files(self) -> tuple[str, ...]:
        """Return paths with unresolved merge conflicts, or () when unavailable."""

        try:
            output = self._git("diff", "--name-only", "--diff-filter=U")
        except GitWorkspaceError:
            return ()
        return tuple(line for line in output.splitlines() if line.strip())

    def conflict_marker_files(self) -> tuple[str, ...]:
        """Return worktree files that still contain conflict markers.

        Covers tracked files plus untracked, non-ignored files, since the
        model may leave markers in either. Conservative by design: a file
        that merely mentions markers is reported too, and the attempt then
        falls back to the pre-rebase state instead of running setup.
        """

        try:
            tracked = self._git("ls-files", "-z")
            untracked = self._git("ls-files", "--others", "--exclude-standard", "-z")
        except GitWorkspaceError:
            return ()
        names = [name for name in tracked.split("\0") + untracked.split("\0") if name.strip()]
        marked: list[str] = []
        for name in names:
            path = self._clone_dir / name
            try:
                content = path.read_bytes()
            except OSError:
                continue
            for line in content.splitlines():
                stripped = line.strip()
                if (
                    stripped.startswith(b"<<<<<<<")
                    or (
                        len(stripped) >= 7
                        and stripped.startswith(b"=======")
                        and stripped.strip(b"=") == b""
                    )
                    or stripped.startswith(b">>>>>>>")
                ):
                    marked.append(name)
                    break
        return tuple(marked)

    def diff_check_clean(self) -> bool:
        """Return True when ``git diff --check`` reports no whitespace errors.

        Covers staged changes as well as unstaged ones, since the model
        stages its conflict resolution with ``git add`` before continuing
        the rebase.
        """

        try:
            self._git("diff", "--check")
            self._git("diff", "--cached", "--check")
        except GitWorkspaceError:
            return False
        return True

    def rebase_resolution_problems(self) -> tuple[str, ...]:
        """Deterministic post-model verification of a left-in-place rebase.

        Returns an empty tuple when the working tree is safe for setup:
        no rebase/merge in progress, no unmerged paths, no conflict
        markers left in any file, and ``git diff --check`` clean. Any
        remaining problem is described, so the caller can abort the rebase,
        restore the branch, and report the conflict instead of proceeding.
        """

        problems: list[str] = []
        if self.has_unresolved_conflicts():
            conflicted = self.conflicted_files()
            files = f" Conflicting files: {', '.join(conflicted)}." if conflicted else ""
            problems.append(f"A rebase or merge is still in progress.{files}")
        marked = self.conflict_marker_files()
        if marked:
            problems.append(f"Conflict markers remain in: {', '.join(marked)}.")
        if not self.diff_check_clean():
            problems.append("`git diff --check` reports whitespace errors.")
        return tuple(problems)

    def abort_unresolved_rebase(self, prepared: PreparedAttempt) -> None:
        """Abort a model-unresolved rebase and restore the pre-rebase branch state.

        The abort alone restores the branch when the rebase is still in
        progress; the hard reset to the recorded pre-rebase revision also
        covers states the model left behind without rebase metadata (for
        example after ``git rebase --quit``). Untracked files are preserved
        as-is, matching the cleanup path. Raises
        ``GitWorkspaceRecoveryError`` when the branch cannot be restored and
        needs human cleanup.
        """

        try:
            self._git("rebase", "--abort")
        except GitWorkspaceError:
            pass
        pre_rebase_revision = prepared.pre_rebase_revision
        if pre_rebase_revision is None:
            return
        try:
            self._git("reset", "--hard", pre_rebase_revision)
            self._git("checkout", prepared.branch)
        except GitWorkspaceError as error:
            raise GitWorkspaceRecoveryError(
                f"Rebase of {prepared.branch} could not be aborted and the branch "
                "could not be restored; human cleanup is required"
            ) from error
        if self.has_unresolved_conflicts():
            raise GitWorkspaceRecoveryError(
                f"Rebase of {prepared.branch} still holds unresolved conflicts "
                "after abort; human cleanup is required"
            )

    def _rebase_in_progress(self) -> bool:
        """Return True if git still reports an unresolved rebase state."""

        git_dir = self._clone_dir / ".git"
        if git_dir.is_dir() and (
            (git_dir / "rebase-merge").exists() or (git_dir / "rebase-apply").exists()
        ):
            return True
        try:
            self._git("rev-parse", "--verify", "REBASE_HEAD")
        except GitWorkspaceError:
            return False
        return True

    def _reset_local_base(self, base_branch: str) -> None:
        """Best-effort repair of a local base branch that fell out of sync with origin."""

        try:
            current_branch = self._git("rev-parse", "--abbrev-ref", "HEAD")
            if current_branch == base_branch:
                self._git("reset", "--hard", f"origin/{base_branch}")
            else:
                self._git("branch", "-f", base_branch, f"origin/{base_branch}")
        except GitWorkspaceError:
            pass

    def _branch_exists(self, branch: str) -> bool:
        try:
            self._git("show-ref", "--verify", "--quiet", f"refs/heads/{branch}")
        except GitWorkspaceError:
            return False
        return True

    def _git(self, *arguments: str, cwd: Path | None = None) -> str:
        working_directory = self._clone_dir if cwd is None and arguments[0] != "clone" else cwd
        github_token = self._token_provider()
        if not isinstance(github_token, str) or not github_token:
            raise GitWorkspaceError("GITHUB_TOKEN is unavailable")
        with _askpass_environment(github_token) as environment:
            try:
                return self._run(("git", *arguments), working_directory, environment)
            except (OSError, subprocess.CalledProcessError) as error:
                raise GitWorkspaceError("Git command failed") from error


def _porcelain_path(line: str) -> str:
    """Return the repository-relative path of one ``status --porcelain`` line.

    Porcelain v1 lines are ``XY path`` (or ``XY origin -> destination`` for
    renames); the destination is what blocks a clean tree.
    """

    path = line[3:] if len(line) > 3 else line.strip()
    _, separator, destination = path.rpartition(" -> ")
    return destination.strip() if separator else path.strip()


def _attempt_branch(issue_number: int) -> str:
    if isinstance(issue_number, bool) or not isinstance(issue_number, int) or issue_number <= 0:
        raise GitWorkspaceError("Issue number must be a positive integer")
    return f"agent/issue-{issue_number}"


def _reject_url_credentials(repository_url: str) -> None:
    parsed = urlsplit(repository_url)
    if parsed.username is not None or parsed.password is not None:
        raise GitWorkspaceError("Repository URL must not contain credentials")


class _askpass_environment:
    """Provide a per-operation credential resolver without persisting a token."""

    def __init__(self, github_token: str) -> None:
        self._github_token = github_token
        self._path: Path | None = None

    def __enter__(self) -> dict[str, str]:
        descriptor, name = tempfile.mkstemp(prefix="simple-coding-agent-askpass-")
        self._path = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as script:
            script.write(
                "#!/bin/sh\n"
                'case "$1" in\n'
                '  *Username*) printf "%s\\n" "x-access-token" ;;\n'
                '  *) printf "%s\\n" "$GITHUB_TOKEN" ;;\n'
                "esac\n"
            )
        self._path.chmod(self._path.stat().st_mode | stat.S_IXUSR)
        environment = os.environ.copy()
        environment.update(
            {
                "GITHUB_TOKEN": self._github_token,
                "GIT_ASKPASS": str(self._path),
                "GIT_TERMINAL_PROMPT": "0",
            }
        )
        return environment

    def __exit__(self, *unused: object) -> None:
        if self._path is not None:
            self._path.unlink(missing_ok=True)


def _run_git(command: tuple[str, ...], cwd: Path | None, env: dict[str, str]) -> str:
    completed = subprocess.run(command, cwd=cwd, env=env, check=True, text=True, capture_output=True)
    return completed.stdout.strip()
