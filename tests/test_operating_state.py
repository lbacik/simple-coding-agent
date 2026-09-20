"""Behaviour tests for persistent process-wide operating limits."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from simple_coding_agent.completion import AttemptOutcome
from simple_coding_agent.operating import ConsecutiveErrorStore


def test_persists_errors_across_restart_and_deduplicates_a_reconciled_attempt(tmp_path: Path) -> None:
    store = ConsecutiveErrorStore(tmp_path, clock=lambda: datetime(2026, 9, 20, tzinfo=UTC))

    assert store.record(AttemptOutcome.INFRASTRUCTURE_ERROR, "attempt-a") == 1
    restarted = ConsecutiveErrorStore(tmp_path)

    assert restarted.record(AttemptOutcome.INFRASTRUCTURE_ERROR, "attempt-a") == 1
    assert restarted.record(AttemptOutcome.INFRASTRUCTURE_ERROR, "attempt-b") == 2
    assert restarted.read().count == 2


def test_resets_the_persistent_guard_for_every_non_infrastructure_outcome(tmp_path: Path) -> None:
    store = ConsecutiveErrorStore(tmp_path, clock=lambda: datetime(2026, 9, 20, tzinfo=UTC))
    store.record(AttemptOutcome.INFRASTRUCTURE_ERROR, "attempt-a")

    assert store.record(AttemptOutcome.INCOMPLETE, "attempt-b") == 0

    state = store.read()
    assert state.count == 0
    assert state.last_success_at == "2026-09-20T00:00:00Z"
