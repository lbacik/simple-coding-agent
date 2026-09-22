"""Git workspace boundary for one persistent repository clone."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time
from urllib.parse import urlsplit


class GitWorkspaceError(RuntimeError):
    """Raised when the persistent git workspace cannot be used safely."""


class GitWorkspaceRecoveryError(GitWorkspaceError):
    """Raised when the local workspace is broken in a way this attempt cannot fix.

    Unlike other ``GitWorkspaceError`` cases, this is a fault in the agent's own
    persistent clone rather than in the claimed issue, so callers must not treat
    it as attempt failure: no result comment, no label release.
    """


@dataclass(frozen=True)
class PreparedAttempt:
    """The branch and verified base revision for one implementation attempt."""

    branch: str
    base_revision: str
    restored_from_remote: bool = False


@dataclass(frozen=True)
class AttemptCommit:
    """A commit added on an attempt branch after its verified base."""

    revision: str
    subject: str


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

    def prepare_attempt(self, *, base_branch: str, issue_number: int) -> PreparedAttempt:
        """Fetch, verify, and check out the branch used by an attempt.

        A local base that differs from ``origin/<base_branch>`` is never used
        as-is: setup must never run against stale or divergent code. This is
        normally just a leftover from an interrupted previous run, so it is
        first repaired by resetting the local base to match origin. Only a
        base that still disagrees after that repair is a broken workspace,
        raised as ``GitWorkspaceRecoveryError`` for the caller to handle
        separately from an ordinary attempt failure.

        A previously published attempt branch (for example, a handoff) is
        always preferred over whatever the local clone happens to have: the
        remote ref is fetched and, when present, used as the branch's
        starting point regardless of local state, so a fresh clone resumes
        published work instead of silently starting over from base.
        """

        branch = _attempt_branch(issue_number)
        self._ensure_clone()
        self._git("fetch", "origin")
        base_revision = self._revision(f"origin/{base_branch}", remote=True)
        local_base_revision = self._revision(base_branch, remote=False)
        if local_base_revision != base_revision:
            self._reset_local_base(base_branch)
            local_base_revision = self._revision(base_branch, remote=False)
            if local_base_revision != base_revision:
                raise GitWorkspaceRecoveryError(
                    "Local base branch has diverged from origin and could not be repaired"
                )

        remote_branch_revision = self._fetch_attempt_branch(branch)
        if remote_branch_revision is not None:
            self._git("checkout", base_branch)
            self._git("branch", "-f", branch, remote_branch_revision)
            self._git("checkout", branch)
            self._rebase_onto_base(branch, base_branch)
        elif self._branch_exists(branch):
            self._git("checkout", branch)
            self._rebase_onto_base(branch, base_branch)
        else:
            self._git("checkout", "-b", branch, base_branch)
        return PreparedAttempt(
            branch=branch,
            base_revision=base_revision,
            restored_from_remote=remote_branch_revision is not None,
        )

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

    def cleanup(
        self, *, base_branch: str, prepared: PreparedAttempt, retain_branch: bool
    ) -> None:
        """Restore the base worktree and optionally delete a disposable branch.

        A rebase left conflicted because the model session could not resolve
        it is aborted here, best-effort, before switching branches: checking
        out another branch mid-rebase fails in git, and this runs
        unconditionally from a ``finally`` block, so it must not raise.
        """

        try:
            self._git("rebase", "--abort")
        except GitWorkspaceError:
            pass
        self._git("checkout", base_branch)
        self._git("reset", "--hard", f"origin/{base_branch}")
        self._git("clean", "-fd")
        if not retain_branch:
            self._git("branch", "-D", prepared.branch)

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
                raise fetch_error
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

    def _rebase_onto_base(self, branch: str, base_branch: str) -> None:
        """Replay a reused attempt branch onto the just-verified base.

        A retained branch from a failed prior attempt (for example, a setup
        failure) must never keep running against the base as it stood back
        then: repository-owned config such as the setup profile has to be
        read fresh. Rebasing preserves any commits on the branch, published
        or not. A conflicting rebase is left in its conflicted state rather
        than aborted and recreated from base: discarding the branch here
        would destroy exactly the work a resumed handoff exists to preserve,
        so the conflict is left for the attempt's model session to resolve
        as ordinary remaining work, using normal git commands.
        """

        try:
            self._git("rebase", base_branch)
        except GitWorkspaceError:
            pass

    def _reset_local_base(self, base_branch: str) -> None:
        """Best-effort repair of a local base branch that fell out of sync with origin."""

        try:
            current_branch = self._git("rev-parse", "--abbrev-ref", "HEAD")
            if current_branch == base_branch:
                self._git("reset", "--hard", f"origin/{base_branch}")
                self._git("clean", "-fd")
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
