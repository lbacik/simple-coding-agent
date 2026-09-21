"""Bounded, isolated execution of repository profile commands."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
import os
from pathlib import Path
import subprocess
import time


class CommandStatus(StrEnum):
    """The terminal state of a profile command sequence."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class CommandExecution:
    """Captured evidence for one command in a profile sequence."""

    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool


@dataclass(frozen=True)
class CommandResult:
    """Captured evidence for an entire profile command sequence."""

    status: CommandStatus
    commands: tuple[CommandExecution, ...]
    duration: float

    @property
    def succeeded(self) -> bool:
        return self.status is CommandStatus.SUCCEEDED

    @property
    def exit_code(self) -> int | None:
        return self.commands[-1].exit_code if self.commands else None


class CommandRunner:
    """Execute profile commands without inheriting the orchestrator environment."""

    def __init__(self, *, redactions: Iterable[str] = (), extra_path: str = "") -> None:
        self._redactions = tuple(value for value in redactions if value)
        self._extra_path = extra_path

    def run(
        self,
        commands: tuple[str, ...],
        *,
        timeout: float,
        cwd: Path,
        environment: Mapping[str, str] | None = None,
    ) -> CommandResult:
        """Run commands through ``sh -c`` with one timeout budget for all commands."""

        started = time.monotonic()
        deadline = started + timeout
        executions: list[CommandExecution] = []
        subprocess_environment = _environment(environment or {}, self._extra_path)

        for command in commands:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                executions.append(CommandExecution(command, None, "", "", True))
                return CommandResult(CommandStatus.TIMED_OUT, tuple(executions), time.monotonic() - started)
            try:
                completed = subprocess.run(
                    ["sh", "-c", command],
                    cwd=cwd,
                    env=subprocess_environment,
                    text=True,
                    capture_output=True,
                    timeout=remaining,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                executions.append(
                    CommandExecution(
                        command,
                        None,
                        self._scrub(error.stdout),
                        self._scrub(error.stderr),
                        True,
                    )
                )
                return CommandResult(CommandStatus.TIMED_OUT, tuple(executions), time.monotonic() - started)

            executions.append(
                CommandExecution(
                    command,
                    completed.returncode,
                    self._scrub(completed.stdout),
                    self._scrub(completed.stderr),
                    False,
                )
            )
            if completed.returncode != 0:
                return CommandResult(CommandStatus.FAILED, tuple(executions), time.monotonic() - started)

        return CommandResult(CommandStatus.SUCCEEDED, tuple(executions), time.monotonic() - started)

    def _scrub(self, output: str | bytes | None) -> str:
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        value = output or ""
        for secret in self._redactions:
            value = value.replace(secret, "[REDACTED]")
        return value


def _environment(profile_environment: Mapping[str, str], extra_path: str = "") -> dict[str, str]:
    """Build the deliberately small environment exposed to repository commands."""

    path = f"{extra_path}:{os.defpath}" if extra_path else os.defpath
    environment = {"PATH": path}
    environment.update(profile_environment)
    # Credentials belong exclusively to the driving process and model boundary.
    environment.pop("GITHUB_TOKEN", None)
    environment.pop("META_API_KEY", None)
    return environment
