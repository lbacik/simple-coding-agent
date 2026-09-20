"""Behaviour tests for the model-execution provenance gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simple_coding_agent.provenance import ProvenanceError, ProvenanceVerifier


def test_accepts_the_pinned_skill_closure_versions_and_home_links(tmp_path: Path) -> None:
    home = tmp_path / "home"
    store = home / ".agents" / "skills"
    links = home / ".claude" / "skills"
    for skill in ("implement", "tdd", "code-review", "codebase-design"):
        (store / skill).mkdir(parents=True)
        links.mkdir(parents=True, exist_ok=True)
        (links / skill).symlink_to(store / skill)
    listing = json.dumps(
        {"artifacts": [
            {"id": f"skill:{skill}", "resolvedCommit": "c55ee46073ed923f86ce59a5eb3b6d895095d1b7", "hash": "abc"}
            for skill in ("implement", "tdd", "code-review", "codebase-design")
        ]}
    )
    verifier = ProvenanceVerifier(
        home=home,
        run=lambda command: listing if command[:2] == ("agent-installer", "list") else "2.1.276",
        sdk_version=lambda: "0.2.156",
    )

    evidence = verifier.verify()

    assert evidence.skill_commit == "c55ee46073ed923f86ce59a5eb3b6d895095d1b7"
    assert evidence.cli_version == "2.1.276"
    assert set(evidence.skill_hashes) == {"implement", "tdd", "code-review", "codebase-design"}


def test_blocks_model_execution_when_skill_provenance_is_tampered(tmp_path: Path) -> None:
    verifier = ProvenanceVerifier(
        home=tmp_path,
        run=lambda _: '{"artifacts": []}',
        sdk_version=lambda: "0.2.156",
    )

    with pytest.raises(ProvenanceError, match="required skill"):
        verifier.verify()
