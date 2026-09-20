"""Behaviour tests for structured, credential-safe attempt observability."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

from simple_coding_agent.observability import AttemptArchive, JsonEventLogger


def test_emits_required_json_fields_and_redacts_credentials() -> None:
    stream = io.StringIO()
    logger = JsonEventLogger(stream, redactions=("github-secret",))

    logger.emit("attempt_finished", issue_number=26, phase="publishing", detail="github-secret failed")

    event = json.loads(stream.getvalue())
    assert set(event) == {"timestamp", "level", "event", "issue_number", "phase", "detail"}
    assert event["detail"] == "[REDACTED] failed"


def test_archives_attempt_metadata_without_credentials(tmp_path: Path) -> None:
    archive = AttemptArchive(
        tmp_path,
        issue_number=26,
        started_at="2026-09-20T13:00:00Z",
        redactions=("meta-secret",),
    )

    archive.write_attempt({"outcome": "infrastructure_error", "token_estimate_usd": "meta-secret"})
    archive.write_text("setup_stdout.log", "meta-secret")

    directory = tmp_path / "logs" / "26" / "2026-09-20T13-00-00Z"
    metadata = json.loads((directory / "attempt.json").read_text())
    assert metadata["token_estimate_usd"] == "[REDACTED]"
    assert metadata["token_estimate_authority"] == "non-authoritative"
    assert (directory / "setup_stdout.log").read_text() == "[REDACTED]"
