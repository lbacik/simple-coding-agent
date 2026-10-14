"""Conformance tests for the synthetic telemetry snapshot examples (schema v1).

Each file under ``examples/telemetry/`` must parse as JSON. Every
``snapshot-*.json`` file must carry the stable fields
(``schema_version``, ``generated_at``, ``identity.repository``,
``identity.agent_version``) and every figure object (any object with a
``basis`` key) must satisfy the contract invariant: ``value`` is ``null``
exactly when ``basis`` is ``unavailable``. Every ``error-*.json`` file must
be the machine-readable ``agentctl telemetry`` error shape with a known code.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = ROOT / "examples" / "telemetry"

ALLOWED_BASES = {"measured", "estimated", "configured", "sdk_assumed", "unavailable"}
ALLOWED_ERROR_CODES = {"agent_unreachable", "unsupported_operation", "snapshot_failed"}

EXPECTED_SNAPSHOTS = {
    "snapshot-idle-with-last-finished.json",
    "snapshot-attempt-in-progress.json",
    "snapshot-soft-threshold-handoff.json",
    "snapshot-recovery-hold.json",
    "snapshot-truncated.json",
}

EXPECTED_ERRORS = {
    "error-agent-unreachable.json",
    "error-unsupported-operation.json",
    "error-snapshot-failed.json",
}


def _load(name: str) -> dict:
    path = EXAMPLES_DIR / name
    assert path.is_file(), f"telemetry example {name} must exist"
    return json.loads(path.read_text())


def _figures(node: object):
    """Yield every figure object (any dict carrying a ``basis`` key)."""

    if isinstance(node, dict):
        if "basis" in node:
            yield node
        for value in node.values():
            yield from _figures(value)
    elif isinstance(node, list):
        for item in node:
            yield from _figures(item)


def test_expected_snapshot_examples_exist() -> None:
    found = {path.name for path in EXAMPLES_DIR.glob("snapshot-*.json")}
    assert EXPECTED_SNAPSHOTS <= found, (
        f"missing snapshot examples: {sorted(EXPECTED_SNAPSHOTS - found)}"
    )


def test_expected_error_examples_exist() -> None:
    found = {path.name for path in EXAMPLES_DIR.glob("error-*.json")}
    assert EXPECTED_ERRORS <= found, (
        f"missing error examples: {sorted(EXPECTED_ERRORS - found)}"
    )


def test_snapshots_carry_the_stable_fields() -> None:
    for name in sorted(EXPECTED_SNAPSHOTS):
        snapshot = _load(name)
        assert snapshot["schema_version"] == 1, f"{name} must be schema version 1"
        assert isinstance(snapshot["generated_at"], str) and snapshot["generated_at"], (
            f"{name} must carry generated_at"
        )
        identity = snapshot["identity"]
        assert isinstance(identity["repository"], str) and identity["repository"], (
            f"{name} must carry identity.repository"
        )
        assert isinstance(identity["agent_version"], str) and identity["agent_version"], (
            f"{name} must carry identity.agent_version"
        )
        assert isinstance(snapshot["truncated"], bool), (
            f"{name} must carry a boolean truncated flag"
        )


def test_figure_invariant_value_null_exactly_when_unavailable() -> None:
    for name in sorted(EXPECTED_SNAPSHOTS):
        snapshot = _load(name)
        figures = list(_figures(snapshot))
        assert figures, f"{name} must contain at least one figure object"
        for figure in figures:
            assert figure["basis"] in ALLOWED_BASES, (
                f"{name}: unknown basis {figure['basis']!r}"
            )
            if figure["basis"] == "unavailable":
                assert figure.get("value") is None, (
                    f"{name}: unavailable figure must have value null: {figure!r}"
                )
            else:
                assert figure.get("value") is not None, (
                    f"{name}: {figure['basis']} figure must not have value null: {figure!r}"
                )


def test_error_examples_cover_each_code() -> None:
    seen = set()
    for name in sorted(EXPECTED_ERRORS):
        reply = _load(name)
        assert reply["ok"] is False, f"{name} must be an error reply"
        code = reply["error"]["code"]
        assert code in ALLOWED_ERROR_CODES, f"{name} has unknown code {code!r}"
        assert isinstance(reply["error"]["message"], str) and reply["error"]["message"], (
            f"{name} must carry an error message"
        )
        seen.add(code)
    assert seen == ALLOWED_ERROR_CODES, (
        f"error examples must cover each code, saw {sorted(seen)}"
    )


def test_dockerfile_sets_the_agent_discovery_label() -> None:
    content = (ROOT / "Dockerfile").read_text()
    assert 'io.github.lbacik.simple-coding-agent.role="agent"' in content, (
        "Dockerfile must set the agent discovery label in the image"
    )


def test_onboarding_mentions_the_discovery_label() -> None:
    content = (ROOT / "docs" / "agents" / "operator-onboarding.md").read_text()
    assert "io.github.lbacik.simple-coding-agent.role" in content, (
        "Operator onboarding docs must mention the discovery label"
    )
