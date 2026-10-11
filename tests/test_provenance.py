"""Behaviour tests for the model-execution provenance gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simple_coding_agent.provenance import ProvenanceError, ProvenanceVerifier

_UPSTREAM_SKILLS = ("implement", "tdd", "code-review", "codebase-design")


def _link_skill(home: Path, skill: str, *, content: str = "") -> None:
    store = home / ".agents" / "skills" / skill
    store.mkdir(parents=True)
    (store / "SKILL.md").write_text(content or f"# {skill}\n")
    links = home / ".claude" / "skills"
    links.mkdir(parents=True, exist_ok=True)
    (links / skill).symlink_to(store)


def _upstream_listing(*, extra: dict[str, str] | None = None) -> str:
    return json.dumps(
        {
            "artifacts": [
                {
                    "id": f"skill:{skill}",
                    "resolvedCommit": "some-other-commit",
                    "installedHash": "abc",
                    **(extra or {}),
                }
                for skill in _UPSTREAM_SKILLS
            ]
        }
    )


def _write_bundled_cli(tmp_path: Path) -> Path:
    cli = tmp_path / "claude"
    cli.write_text("#!/bin/sh\necho bundled\n")
    return cli


def _verifier(
    home: Path,
    cli: Path,
    *,
    run=None,
    sdk_version: str = "0.2.163",
    bundled_version: str = "2.1.286",
):
    listing = _upstream_listing()

    def default_run(command: tuple[str, ...]) -> str:
        if command[:2] == ("agent-installer", "list"):
            return listing
        assert command[0] == str(cli), f"expected the bundled CLI, got {command[0]!r}"
        assert command[1:] == ("--version",)
        return "2.1.286"

    return ProvenanceVerifier(
        home=home,
        expected_sdk_version="0.2.163",
        run=run or default_run,
        sdk_version=lambda: sdk_version,
        cli_path=cli,
        bundled_cli_version=lambda: bundled_version,
    )


def _linked_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    for skill in _UPSTREAM_SKILLS:
        _link_skill(home, skill)
    _link_skill(home, "handoff")
    return home


def test_accepts_the_pinned_skill_closure_versions_and_home_links(tmp_path: Path) -> None:
    home = _linked_home(tmp_path)
    cli = _write_bundled_cli(tmp_path)
    verifier = _verifier(home, cli)

    evidence = verifier.verify()

    assert evidence.cli_version == "2.1.286"
    assert evidence.cli_path == cli
    assert set(evidence.skill_hashes) == {*_UPSTREAM_SKILLS, "handoff"}


def test_ignores_the_claude_on_path(tmp_path: Path) -> None:
    """A different PATH CLI version no longer affects the outcome."""
    home = _linked_home(tmp_path)
    cli = _write_bundled_cli(tmp_path)
    listing = _upstream_listing()

    def run(command: tuple[str, ...]) -> str:
        if command[:2] == ("agent-installer", "list"):
            return listing
        # The verified binary is the injected bundled CLI, never `claude`.
        assert command[0] == str(cli)
        return "2.1.286"

    evidence = _verifier(home, cli, run=run).verify()

    assert evidence.cli_version == "2.1.286"
    # Even if `claude` on PATH reports something else, verification stands:
    # the run seam above would fail if the verifier ever consulted PATH.


def test_blocks_model_execution_when_skill_provenance_is_tampered(tmp_path: Path) -> None:
    cli = _write_bundled_cli(tmp_path)
    verifier = ProvenanceVerifier(
        home=tmp_path,
        expected_sdk_version="0.2.163",
        run=lambda _: '{"artifacts": []}',
        sdk_version=lambda: "0.2.163",
        cli_path=cli,
        bundled_cli_version=lambda: "2.1.286",
    )

    with pytest.raises(ProvenanceError, match="required skill"):
        verifier.verify()


def test_blocks_model_execution_when_the_project_owned_skill_link_is_missing(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    for skill in _UPSTREAM_SKILLS:
        _link_skill(home, skill)
    cli = _write_bundled_cli(tmp_path)
    verifier = _verifier(home, cli)

    with pytest.raises(ProvenanceError, match="handoff"):
        verifier.verify()


def test_project_owned_skill_hash_reflects_its_installed_content(tmp_path: Path) -> None:
    home = tmp_path / "home"
    for skill in _UPSTREAM_SKILLS:
        _link_skill(home, skill)
    _link_skill(home, "handoff", content="# handoff\n\nOne body.\n")
    cli = _write_bundled_cli(tmp_path)
    verifier = _verifier(home, cli)
    first_hash = verifier.verify().skill_hashes["handoff"]

    (home / ".agents" / "skills" / "handoff" / "SKILL.md").write_text("# handoff\n\nA different body.\n")
    second_hash = verifier.verify().skill_hashes["handoff"]

    assert first_hash != second_hash


def test_parses_the_cli_version_from_its_trailing_label(tmp_path: Path) -> None:
    home = _linked_home(tmp_path)
    cli = _write_bundled_cli(tmp_path)
    listing = _upstream_listing()
    verifier = ProvenanceVerifier(
        home=home,
        expected_sdk_version="0.2.163",
        run=lambda command: listing
        if command[:2] == ("agent-installer", "list")
        else "2.1.286 (Claude Code)",
        sdk_version=lambda: "0.2.163",
        cli_path=cli,
        bundled_cli_version=lambda: "2.1.286",
    )

    evidence = verifier.verify()

    assert evidence.cli_version == "2.1.286"


def test_rejects_a_bundled_cli_version_different_from_the_sdk_pin(
    tmp_path: Path,
) -> None:
    home = _linked_home(tmp_path)
    cli = _write_bundled_cli(tmp_path)
    listing = _upstream_listing()
    verifier = ProvenanceVerifier(
        home=home,
        expected_sdk_version="0.2.163",
        run=lambda command: listing
        if command[:2] == ("agent-installer", "list")
        else "9.9.9",
        sdk_version=lambda: "0.2.163",
        cli_path=cli,
        bundled_cli_version=lambda: "2.1.286",
    )

    with pytest.raises(ProvenanceError, match="CLI version does not match"):
        verifier.verify()


def test_missing_bundled_cli_fails_closed_without_a_path_fallback(
    tmp_path: Path,
) -> None:
    home = _linked_home(tmp_path)
    missing = tmp_path / "does-not-exist" / "claude"
    calls: list[tuple[str, ...]] = []
    listing = _upstream_listing()

    def run(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command[:2] == ("agent-installer", "list"):
            return listing
        raise AssertionError(f"must not consult PATH, got {command!r}")

    verifier = ProvenanceVerifier(
        home=home,
        expected_sdk_version="0.2.163",
        run=run,
        sdk_version=lambda: "0.2.163",
        cli_path=missing,
        bundled_cli_version=lambda: "2.1.286",
    )

    with pytest.raises(ProvenanceError, match="Bundled Claude Code CLI is missing"):
        verifier.verify()

    assert all(command[:2] == ("agent-installer", "list") for command in calls)


def test_unavailable_bundled_cli_version_fails_closed(tmp_path: Path) -> None:
    home = _linked_home(tmp_path)
    cli = _write_bundled_cli(tmp_path)

    def broken() -> str:
        raise ProvenanceError("Bundled Claude Code CLI version is unavailable")

    verifier = ProvenanceVerifier(
        home=home,
        expected_sdk_version="0.2.163",
        run=_verifier(home, cli)._run,
        sdk_version=lambda: "0.2.163",
        cli_path=cli,
        bundled_cli_version=broken,
    )

    with pytest.raises(ProvenanceError, match="Bundled Claude Code CLI version"):
        verifier.verify()


def _verified_evidence(tmp_path: Path):
    home = _linked_home(tmp_path)
    cli = _write_bundled_cli(tmp_path)
    return _verifier(home, cli).verify()


def test_startup_summary_names_the_agent_version_next_to_sdk_and_cli(tmp_path: Path) -> None:
    from simple_coding_agent.agent_version import agent_version

    evidence = _verified_evidence(tmp_path)
    summary = evidence.summary()

    assert summary == (
        f"agent={agent_version()}; sdk=0.2.163; cli=2.1.286 ({evidence.cli_path})"
    )


def test_startup_verification_survives_an_unreadable_agent_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.metadata

    from simple_coding_agent import agent_version as agent_version_module

    def missing(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(agent_version_module, "version", missing)

    evidence = _verified_evidence(tmp_path)

    assert evidence.agent_version == "unknown"
    assert evidence.summary().startswith("agent=unknown;")
