"""Pre-flight verification of the pinned SDK, CLI, and upstream skill bundle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from simple_coding_agent.agent_version import UNKNOWN_VERSION, agent_version


REQUIRED_SKILLS = ("implement", "tdd", "code-review", "codebase-design")

# Vendored in this repository under skills/<name>, not installed via
# agent-installer: there is no upstream artifact or pinned commit to check,
# only the HOME symlink shape shared with the upstream skills and a content
# hash retained for the operational record.
PROJECT_OWNED_SKILLS = ("handoff",)


class ProvenanceError(RuntimeError):
    """Raised when the approved model-execution environment is not intact."""


@dataclass(frozen=True)
class ProvenanceEvidence:
    """Non-secret evidence retained with an attempt's operational record."""

    skill_hashes: dict[str, str]
    sdk_version: str
    cli_version: str
    cli_path: Path
    agent_version: str = UNKNOWN_VERSION

    def summary(self) -> str:
        """One-line startup log detail naming the agent, SDK and CLI versions."""

        return (
            f"agent={self.agent_version}; sdk={self.sdk_version}; "
            f"cli={self.cli_version} ({self.cli_path})"
        )


class ProvenanceVerifier:
    """Reject model execution unless all pinned runtime inputs can be verified."""

    def __init__(
        self,
        *,
        home: Path,
        expected_sdk_version: str,
        run: Callable[[tuple[str, ...]], str] | None = None,
        sdk_version: Callable[[], str] | None = None,
        cli_path: str | Path | None = None,
        bundled_cli_version: Callable[[], str] | None = None,
    ) -> None:
        self._home = home
        self._expected_sdk_version = expected_sdk_version
        self._run = run or _run
        self._sdk_version = sdk_version or _installed_sdk_version
        self._cli_path = cli_path
        self._bundled_cli_version = bundled_cli_version or _expected_bundled_cli_version

    def verify(self) -> ProvenanceEvidence:
        artifacts = _artifacts(self._run(("agent-installer", "list", "--json")))
        hashes: dict[str, str] = {}
        for skill in REQUIRED_SKILLS:
            artifact = artifacts.get(f"skill:{skill}")
            if artifact is None:
                raise ProvenanceError(f"Missing required skill: {skill}")
            digest = artifact.get("installedHash")
            if not isinstance(digest, str) or not digest:
                raise ProvenanceError(f"Required skill has no installed hash: {skill}")
            _verify_link(self._home, skill)
            hashes[skill] = digest
        for skill in PROJECT_OWNED_SKILLS:
            _verify_link(self._home, skill)
            hashes[skill] = _skill_content_hash(self._home / ".agents" / "skills" / skill)
        sdk_version = self._sdk_version()
        if sdk_version != self._expected_sdk_version:
            raise ProvenanceError("Claude Agent SDK version does not match the pinned runtime")
        cli_path = Path(self._cli_path).expanduser() if self._cli_path is not None else _bundled_cli_path()
        if not cli_path.is_file():
            raise ProvenanceError(
                f"Bundled Claude Code CLI is missing at {cli_path}; "
                "the pinned runtime cannot be verified"
            )
        expected_cli_version = self._bundled_cli_version()
        cli_version = _version(self._run((str(cli_path), "--version")))
        if cli_version != expected_cli_version:
            raise ProvenanceError("Claude Code CLI version does not match the pinned runtime")
        return ProvenanceEvidence(hashes, sdk_version, cli_version, cli_path, agent_version())


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


def _skill_content_hash(skill_dir: Path) -> str:
    """Hash a project-owned skill's files for the operational record.

    There is no upstream commit to pin against, so this is retained evidence
    of what was actually installed, not a match against an expected value.
    """

    digest = hashlib.sha256()
    for path in sorted(skill_dir.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(skill_dir).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _verify_link(home: Path, skill: str) -> None:
    source = home / ".agents" / "skills" / skill
    link = home / ".claude" / "skills" / skill
    if not source.is_dir() or not link.is_symlink() or link.resolve() != source.resolve():
        raise ProvenanceError(f"Required skill HOME link is invalid: {skill}")


def _bundled_cli_path() -> Path:
    """Resolve the Claude Code CLI bundled inside the installed SDK package.

    This is the only reliance on SDK package internals: the SDK runs this
    same bundled binary in preference to ``PATH`` unless ``cli_path`` is set,
    so provenance must verify exactly this file and the executor must run it.
    """

    try:
        import claude_agent_sdk

        return Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    except Exception as error:
        raise ProvenanceError("Bundled Claude Code CLI location is unavailable") from error


def _expected_bundled_cli_version() -> str:
    """Read the CLI version the installed SDK release ships."""

    try:
        from claude_agent_sdk._cli_version import __cli_version__

        if not isinstance(__cli_version__, str) or not __cli_version__.strip():
            raise ValueError("bundled CLI version is empty")
        return __cli_version__
    except ProvenanceError:
        raise
    except Exception as error:
        raise ProvenanceError("Bundled Claude Code CLI version is unavailable") from error


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
    return words[0] if words else ""
