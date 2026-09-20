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


@dataclass(frozen=True)
class PreparedAttempt:
    """The branch and verified base revision for one implementation attempt."""

    branch: str
    base_revision: str


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

    def prepare_attempt(self, *, base_branch: str, issue_number: int) -> PreparedAttempt:
        """Fetch, verify, and check out the branch used by an attempt.

        A local base that differs from ``origin/<base_branch>`` is deliberately
        refused: setup must never run against stale or divergent code.
        """

        branch = _attempt_branch(issue_number)
        self._ensure_clone()
        self._git("fetch", "origin")
        base_revision = self._revision(f"origin/{base_branch}", remote=True)
        local_base_revision = self._revision(base_branch, remote=False)
        if local_base_revision != base_revision:
            raise GitWorkspaceError("Local base branch has diverged from origin")

        if self._branch_exists(branch):
            self._git("checkout", branch)
        else:
            self._git("checkout", "-b", branch, base_branch)
        return PreparedAttempt(branch=branch, base_revision=base_revision)

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
        """Restore the base worktree and optionally delete a disposable branch."""

        self._git("checkout", base_branch)
        self._git("reset", "--hard", f"origin/{base_branch}")
        self._git("clean", "-fd")
        if not retain_branch:
            self._git("branch", "-D", prepared.branch)

    def push_attempt_branch(self, branch: str, *, max_retries: int) -> str:
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
            try:
                observed = self._fetch_remote_branch(branch)
                expected = observed or ""
                self._git(
                    "push",
                    f"--force-with-lease={branch}:{expected}",
                    "origin",
                    f"{branch}:refs/heads/{branch}",
                )
                return local_revision
            except GitWorkspaceError as error:
                last_error = error
                try:
                    if self._remote_branch_revision(branch) == local_revision:
                        return local_revision
                except GitWorkspaceError:
                    pass
                if attempt + 1 < max_retries:
                    self._sleeper(2**attempt)
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
