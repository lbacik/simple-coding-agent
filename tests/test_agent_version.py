"""The agent reports its own installed version in every operator-visible place."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from simple_coding_agent import agent_version as agent_version_module
from simple_coding_agent.agent_version import UNKNOWN_VERSION, agent_version
from simple_coding_agent.agentctl import format_status
from simple_coding_agent.control import IntakeState, build_status
from simple_coding_agent.observability import AttemptArchive

ROOT = Path(__file__).parent.parent


def _status() -> dict:
    return build_status(
        repository="owner/repo",
        intake=IntakeState.RUNNING,
        recovering=False,
        active_attempt=None,
        pending_command_id=None,
        get_command=lambda _id: None,
        recent_commands=lambda: [],
    )


def _break_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(agent_version_module, "version", missing)


def test_version_matches_pyproject() -> None:
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert agent_version() == declared


def test_unreadable_metadata_yields_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    _break_metadata(monkeypatch)
    assert agent_version() == UNKNOWN_VERSION


def test_attempt_json_records_agent_version(tmp_path: Path) -> None:
    archive = AttemptArchive(tmp_path, issue_number=7, started_at="2026-09-20T13:00:00Z")
    archive.write_attempt({"outcome": "complete"})
    record = json.loads((archive.directory / "attempt.json").read_text())
    assert record["agent_version"] == agent_version()


def test_attempt_json_survives_unreadable_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _break_metadata(monkeypatch)
    archive = AttemptArchive(tmp_path, issue_number=7, started_at="2026-09-20T13:00:00Z")
    archive.write_attempt({"outcome": "complete"})
    assert archive.read_attempt()["agent_version"] == UNKNOWN_VERSION


def test_status_payload_and_formatted_output_carry_version() -> None:
    status = _status()
    assert status["agent_version"] == agent_version()
    assert f"agent version: {agent_version()}" in format_status(status)


def test_status_reports_placeholder_when_lookup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _break_metadata(monkeypatch)
    assert "agent version: unknown" in format_status(_status())


def _check(tag: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_tag_version.py"), tag, str(ROOT / "pyproject.toml")],
        capture_output=True,
        text=True,
    )


def test_tag_guard_accepts_matching_tag() -> None:
    assert _check(f"v{agent_version()}").returncode == 0


def test_tag_guard_names_both_values_on_mismatch() -> None:
    result = _check("v9.9.9")
    assert result.returncode == 1
    assert "v9.9.9" in result.stderr
    assert agent_version() in result.stderr
