"""Pre-flight verification of the pinned SDK, CLI, and upstream skill bundle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any

from simple_coding_agent.config import CLAUDE_AGENT_SDK_VERSION, CLAUDE_CODE_VERSION


UPSTREAM_SKILLS_COMMIT = "c55ee46073ed923f86ce59a5eb3b6d895095d1b7"
REQUIRED_SKILLS = ("implement", "tdd", "code-review", "codebase-design")


class ProvenanceError(RuntimeError):
    """Raised when the approved model-execution environment is not intact."""


@dataclass(frozen=True)
class ProvenanceEvidence:
    """Non-secret evidence retained with an attempt's operational record."""

    skill_commit: str
    skill_hashes: dict[str, str]
    sdk_version: str
    cli_version: str


class ProvenanceVerifier:
    """Reject model execution unless all pinned runtime inputs can be verified."""

    def __init__(
        self,
        *,
        home: Path,
        run: Callable[[tuple[str, ...]], str] | None = None,
        sdk_version: Callable[[], str] | None = None,
    ) -> None:
        self._home = home
        self._run = run or _run
        self._sdk_version = sdk_version or _installed_sdk_version

    def verify(self) -> ProvenanceEvidence:
        artifacts = _artifacts(self._run(("agent-installer", "list", "--json")))
        hashes: dict[str, str] = {}
        for skill in REQUIRED_SKILLS:
            artifact = artifacts.get(f"skill:{skill}")
            if artifact is None:
                raise ProvenanceError(f"Missing required skill: {skill}")
            if artifact.get("resolvedCommit") != UPSTREAM_SKILLS_COMMIT:
                raise ProvenanceError(f"Required skill has an unexpected source commit: {skill}")
            digest = artifact.get("hash")
            if not isinstance(digest, str) or not digest:
                raise ProvenanceError(f"Required skill has no installed hash: {skill}")
            _verify_link(self._home, skill)
            hashes[skill] = digest
        sdk_version = self._sdk_version()
        if sdk_version != CLAUDE_AGENT_SDK_VERSION:
            raise ProvenanceError("Claude Agent SDK version does not match the pinned runtime")
        cli_version = _version(self._run(("claude", "--version")))
        if cli_version != CLAUDE_CODE_VERSION:
            raise ProvenanceError("Claude Code CLI version does not match the pinned runtime")
        return ProvenanceEvidence(UPSTREAM_SKILLS_COMMIT, hashes, sdk_version, cli_version)


def _artifacts(output: str) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as error:
        raise ProvenanceError("agent-installer did not return valid JSON") from error
    values = payload.get("artifacts") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        raise ProvenanceError("agent-installer returned an invalid artifact list")
    return {
        artifact["id"]: artifact
        for artifact in values
        if isinstance(artifact, dict) and isinstance(artifact.get("id"), str)
    }


def _verify_link(home: Path, skill: str) -> None:
    source = home / ".agents" / "skills" / skill
    link = home / ".claude" / "skills" / skill
    if not source.is_dir() or not link.is_symlink() or link.resolve() != source.resolve():
        raise ProvenanceError(f"Required skill HOME link is invalid: {skill}")


def _installed_sdk_version() -> str:
    try:
        from importlib.metadata import version

        return version("claude-agent-sdk")
    except Exception as error:
        raise ProvenanceError("Claude Agent SDK version is unavailable") from error


def _run(command: tuple[str, ...]) -> str:
    try:
        return subprocess.run(command, text=True, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProvenanceError(f"Could not run {' '.join(command[:2])}") from error


def _version(output: str) -> str:
    words = output.strip().split()
    return words[-1].removeprefix("v") if words else ""
