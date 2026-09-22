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
        expected_sdk_version="0.2.156",
        expected_cli_version="2.1.276",
        run=lambda command: _upstream_listing()
        if command[:2] == ("agent-installer", "list")
        else "2.1.276",
        sdk_version=lambda: "0.2.156",
    )

    evidence = verifier.verify()

    assert evidence.cli_version == "2.1.276"
    assert set(evidence.skill_hashes) == {*_UPSTREAM_SKILLS, "handoff"}


def test_blocks_model_execution_when_skill_provenance_is_tampered(tmp_path: Path) -> None:
    verifier = ProvenanceVerifier(
        home=tmp_path,
        expected_sdk_version="0.2.156",
        expected_cli_version="2.1.276",
        run=lambda _: '{"artifacts": []}',
        sdk_version=lambda: "0.2.156",
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
        expected_sdk_version="0.2.156",
        expected_cli_version="2.1.276",
        run=lambda command: _upstream_listing()
        if command[:2] == ("agent-installer", "list")
        else "2.1.276",
        sdk_version=lambda: "0.2.156",
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
        expected_sdk_version="0.2.156",
        expected_cli_version="2.1.276",
        run=lambda command: _upstream_listing()
        if command[:2] == ("agent-installer", "list")
        else "2.1.276",
        sdk_version=lambda: "0.2.156",
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
        expected_sdk_version="0.2.156",
        expected_cli_version="2.1.278",
        run=lambda command: listing
        if command[:2] == ("agent-installer", "list")
        else "2.1.278 (Claude Code)",
        sdk_version=lambda: "0.2.156",
    )

    evidence = verifier.verify()

    assert evidence.cli_version == "2.1.278"
