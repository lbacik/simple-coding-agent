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


def test_accepts_the_pinned_skill_closure_versions_and_home_links(tmp_path: Path) -> None:
    home = tmp_path / "home"
    for skill in _UPSTREAM_SKILLS:
        _link_skill(home, skill)
    _link_skill(home, "handoff")
    verifier = ProvenanceVerifier(
        home=home,
        expected_sdk_version="0.2.163",
        expected_cli_version="2.1.286",
        run=lambda command: _upstream_listing()
        if command[:2] == ("agent-installer", "list")
        else "2.1.286",
        sdk_version=lambda: "0.2.163",
    )

    evidence = verifier.verify()

    assert evidence.cli_version == "2.1.286"
    assert set(evidence.skill_hashes) == {*_UPSTREAM_SKILLS, "handoff"}


def test_blocks_model_execution_when_skill_provenance_is_tampered(tmp_path: Path) -> None:
    verifier = ProvenanceVerifier(
        home=tmp_path,
        expected_sdk_version="0.2.163",
        expected_cli_version="2.1.286",
        run=lambda _: '{"artifacts": []}',
        sdk_version=lambda: "0.2.163",
    )

    with pytest.raises(ProvenanceError, match="required skill"):
        verifier.verify()


def test_blocks_model_execution_when_the_project_owned_skill_link_is_missing(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    for skill in _UPSTREAM_SKILLS:
        _link_skill(home, skill)
    verifier = ProvenanceVerifier(
        home=home,
        expected_sdk_version="0.2.163",
        expected_cli_version="2.1.286",
        run=lambda command: _upstream_listing()
        if command[:2] == ("agent-installer", "list")
        else "2.1.286",
        sdk_version=lambda: "0.2.163",
    )

    with pytest.raises(ProvenanceError, match="handoff"):
        verifier.verify()


def test_project_owned_skill_hash_reflects_its_installed_content(tmp_path: Path) -> None:
    home = tmp_path / "home"
    for skill in _UPSTREAM_SKILLS:
        _link_skill(home, skill)
    _link_skill(home, "handoff", content="# handoff\n\nOne body.\n")
    verifier = ProvenanceVerifier(
        home=home,
        expected_sdk_version="0.2.163",
        expected_cli_version="2.1.286",
        run=lambda command: _upstream_listing()
        if command[:2] == ("agent-installer", "list")
        else "2.1.286",
        sdk_version=lambda: "0.2.163",
    )
    first_hash = verifier.verify().skill_hashes["handoff"]

    (home / ".agents" / "skills" / "handoff" / "SKILL.md").write_text("# handoff\n\nA different body.\n")
    second_hash = verifier.verify().skill_hashes["handoff"]

    assert first_hash != second_hash


def test_parses_the_cli_version_from_its_trailing_label(tmp_path: Path) -> None:
    home = tmp_path / "home"
    for skill in _UPSTREAM_SKILLS:
        _link_skill(home, skill)
    _link_skill(home, "handoff")
    listing = _upstream_listing()
    verifier = ProvenanceVerifier(
        home=home,
        expected_sdk_version="0.2.163",
        expected_cli_version="2.1.286",
        run=lambda command: listing
        if command[:2] == ("agent-installer", "list")
        else "2.1.286 (Claude Code)",
        sdk_version=lambda: "0.2.163",
    )

    evidence = verifier.verify()

    assert evidence.cli_version == "2.1.286"


def _verified_evidence(tmp_path: Path):
    home = tmp_path / "home"
    for skill in (*_UPSTREAM_SKILLS, "handoff"):
        _link_skill(home, skill)
    return ProvenanceVerifier(
        home=home,
        expected_sdk_version="0.2.163",
        expected_cli_version="2.1.286",
        run=lambda command: _upstream_listing()
        if command[:2] == ("agent-installer", "list")
        else "2.1.286",
        sdk_version=lambda: "0.2.163",
    ).verify()


def test_startup_summary_names_the_agent_version_next_to_sdk_and_cli(tmp_path: Path) -> None:
    from simple_coding_agent.agent_version import agent_version

    summary = _verified_evidence(tmp_path).summary()

    assert summary == f"agent={agent_version()}; sdk=0.2.163; cli=2.1.286"


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
