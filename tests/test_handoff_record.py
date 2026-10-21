"""Unit tests for the single discoverable handoff record (issue #282).

The shared handoff comment carries one short branch-following link plus a
hidden versioned marker. These tests pin the marker shape, the durable
intent store, and the transparency of the shared record to the legacy
readers (automatic-continuation count, conflict streak, trusted comments).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from simple_coding_agent.handoff_record import (
    HANDOFF_RECORD_VERSION,
    HandoffPublicationStore,
    HandoffPublicationStoreError,
    HandoffRecord,
    build_handoff_record_body,
    parse_handoff_record,
)
from simple_coding_agent.recovery import (
    REQUEUE_SOURCE_AUTOMATIC,
    REQUEUE_SOURCE_HUMAN,
    count_automatic_continuations,
    count_conflict_failures,
    is_conflict_attempt,
    is_handoff_record,
    requeue_source_marker,
)


def record(**overrides: object) -> HandoffRecord:
    fields: dict[str, object] = {
        "version": HANDOFF_RECORD_VERSION,
        "repository": "octo/example",
        "issue_number": 24,
        "attempt_id": "2026-09-20T13:00:00Z",
        "requeue_source": REQUEUE_SOURCE_HUMAN,
        "continuation_count": 0,
        "branch": "agent/issue-24",
        "branch_sha": "a" * 40,
        "note_sha": "b" * 40,
    }
    fields.update(overrides)
    return HandoffRecord(**fields)  # type: ignore[arg-type]


def test_marker_round_trip_preserves_every_field() -> None:
    from simple_coding_agent.handoff_record import handoff_record_marker

    expected = record(
        requeue_source=REQUEUE_SOURCE_AUTOMATIC,
        continuation_count=2,
    )
    parsed = parse_handoff_record(handoff_record_marker(expected))

    assert parsed == expected


def test_marker_is_versioned() -> None:
    from simple_coding_agent.handoff_record import handoff_record_marker

    assert f"v={HANDOFF_RECORD_VERSION}" in handoff_record_marker(record())


def test_parse_rejects_bodies_without_the_marker() -> None:
    assert parse_handoff_record("no marker here") is None
    assert parse_handoff_record("<!-- handoff-record: unterminated") is None
    assert parse_handoff_record("## Agent Attempt Result: handoff\n\n<!-- agent-attempt: x -->") is None


def test_parse_rejects_a_marker_missing_required_fields() -> None:
    assert parse_handoff_record("<!-- handoff-record: v=1 repo=octo/example -->") is None


def test_is_handoff_record_matches_only_the_shared_marker() -> None:
    from simple_coding_agent.handoff_record import handoff_record_marker

    assert is_handoff_record(handoff_record_marker(record()))
    assert is_handoff_record(build_handoff_record_body(record()))
    assert not is_handoff_record("## Agent Attempt Result: handoff\n\n<!-- agent-attempt: x -->")
    assert not is_handoff_record("<!-- conflict-attempt: outcome=success sha=none -->")
    assert not is_handoff_record("")


def test_body_links_to_the_branch_following_note_and_hides_legacy_markers() -> None:
    body = build_handoff_record_body(record())

    assert "https://github.com/octo/example/blob/agent/issue-24/.agent/handoff/24.md" in body
    assert is_handoff_record(body)
    # The shared record must never read as a legacy attempt-result comment.
    assert "Agent Attempt Result" not in body
    assert "Conflict Attempt Result" not in body
    assert "<!-- agent-attempt:" not in body
    assert "<!-- conflict-attempt:" not in body
    assert requeue_source_marker(REQUEUE_SOURCE_HUMAN) not in body
    assert requeue_source_marker(REQUEUE_SOURCE_AUTOMATIC) not in body
    assert not is_conflict_attempt(body)


def test_automatic_continuation_count_skips_the_shared_record() -> None:
    from simple_coding_agent.handoff_record import handoff_record_marker

    shared = handoff_record_marker(record(attempt_id="2026-09-20T14:00:00Z"))
    automatic = (
        "## Agent Attempt Result: handoff\n\nHalf done.\n\n"
        "<!-- agent-attempt: 2026-09-20T13:00:00Z -->\n"
        f"{requeue_source_marker(REQUEUE_SOURCE_AUTOMATIC)}"
    )

    # Newest first: the shared record sits on top of a legacy automatic row.
    assert count_automatic_continuations([shared, automatic]) == 1
    assert count_automatic_continuations([automatic, shared]) == 1


def test_automatic_continuation_count_ignores_a_shared_record_row() -> None:
    from simple_coding_agent.handoff_record import handoff_record_marker

    shared = handoff_record_marker(record())

    assert count_automatic_continuations([shared]) == 0


def test_conflict_streak_skips_the_shared_record() -> None:
    from simple_coding_agent.handoff_record import handoff_record_marker

    shared = handoff_record_marker(record())
    failure = "<!-- conflict-attempt: outcome=rebase_conflict sha=none -->"

    assert count_conflict_failures([shared, failure]) == 1
    assert count_conflict_failures([failure, shared]) == 1
    assert count_conflict_failures([shared]) == 0


def test_store_persists_the_intended_payload_across_instances(tmp_path: Path) -> None:
    store = HandoffPublicationStore(tmp_path)
    expected = record()

    assert store.load(expected.attempt_id) is None
    stored = store.save_if_absent(expected)

    assert stored == expected
    assert HandoffPublicationStore(tmp_path).load(expected.attempt_id) == expected


def test_store_first_intent_wins_so_a_retry_never_increments_the_count(tmp_path: Path) -> None:
    store = HandoffPublicationStore(tmp_path)
    first = record(continuation_count=1)
    retry = record(continuation_count=2)

    assert store.save_if_absent(first) == first
    assert store.save_if_absent(retry) == first
    assert store.load(first.attempt_id) == first


def test_store_rejects_an_invalid_record() -> None:
    with pytest.raises(HandoffPublicationStoreError):
        HandoffPublicationStore(Path("/tmp")).save_if_absent(
            record(issue_number=0)  # type: ignore[arg-type]
        )


def test_store_survives_a_missing_file_as_empty(tmp_path: Path) -> None:
    assert HandoffPublicationStore(tmp_path).load("2026-09-20T13:00:00Z") is None
    assert HandoffPublicationStore(tmp_path).read_all() == {}
