from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from simple_coding_agent.attempt_state import (
    AttemptPhase,
    AttemptStateError,
    AttemptStateStore,
)


class SequenceClock:
    def __init__(self, *values: datetime) -> None:
        self._values = iter(values)

    def __call__(self) -> datetime:
        return next(self._values)


def test_starts_and_reads_an_attempt_with_a_stable_identity(tmp_path: Path) -> None:
    started = datetime(2026, 9, 20, 13, 0, tzinfo=UTC)
    store = AttemptStateStore(tmp_path, clock=SequenceClock(started))

    attempt = store.start(issue_number=42, branch="agent/issue-42")

    assert attempt.issue_number == 42
    assert attempt.branch == "agent/issue-42"
    assert attempt.phase is AttemptPhase.CLAIMED
    assert attempt.started_at == "2026-09-20T13:00:00Z"
    assert attempt.updated_at == "2026-09-20T13:00:00Z"
    assert store.read() == attempt


def test_transitions_in_order_without_changing_started_at(tmp_path: Path) -> None:
    started = datetime(2026, 9, 20, 13, 0, tzinfo=UTC)
    updated = started + timedelta(seconds=10)
    store = AttemptStateStore(tmp_path, clock=SequenceClock(started, updated))
    attempt = store.start(issue_number=42, branch="agent/issue-42")

    transitioned = store.transition(AttemptPhase.SETUP)

    assert transitioned.phase is AttemptPhase.SETUP
    assert transitioned.started_at == attempt.started_at
    assert transitioned.updated_at == "2026-09-20T13:00:10Z"


def test_transitions_through_every_attempt_phase_in_order(tmp_path: Path) -> None:
    started = datetime(2026, 9, 20, 13, 0, tzinfo=UTC)
    store = AttemptStateStore(
        tmp_path,
        clock=SequenceClock(
            started,
            started + timedelta(seconds=1),
            started + timedelta(seconds=2),
            started + timedelta(seconds=3),
            started + timedelta(seconds=4),
        ),
    )
    store.start(issue_number=42, branch="agent/issue-42")

    phases = [
        store.transition(AttemptPhase.SETUP).phase,
        store.transition(AttemptPhase.MODEL_RUNNING).phase,
        store.transition(AttemptPhase.PUSHING).phase,
        store.transition(AttemptPhase.PUBLISHING).phase,
    ]

    assert phases == [
        AttemptPhase.SETUP,
        AttemptPhase.MODEL_RUNNING,
        AttemptPhase.PUSHING,
        AttemptPhase.PUBLISHING,
    ]


def test_rejects_non_sequential_phase_transitions(tmp_path: Path) -> None:
    store = AttemptStateStore(tmp_path)
    store.start(issue_number=42, branch="agent/issue-42")

    with pytest.raises(AttemptStateError, match="claimed.*model_running"):
        store.transition(AttemptPhase.MODEL_RUNNING)


def test_deletes_the_active_attempt(tmp_path: Path) -> None:
    store = AttemptStateStore(tmp_path)
    store.start(issue_number=42, branch="agent/issue-42")

    store.delete()

    assert store.read() is None


def test_reports_a_malformed_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "state" / "attempt.json"
    path.parent.mkdir()
    path.write_text("{not JSON")

    with pytest.raises(AttemptStateError, match="malformed"):
        AttemptStateStore(tmp_path).read()


def test_ignores_a_temporary_file_left_by_an_interrupted_write(tmp_path: Path) -> None:
    store = AttemptStateStore(tmp_path)
    attempt = store.start(issue_number=42, branch="agent/issue-42")
    temporary = tmp_path / "state" / ".attempt-interrupted.tmp"
    temporary.write_text('{"incomplete": true}')

    assert store.read() == attempt


def test_keeps_the_previous_checkpoint_when_replacement_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AttemptStateStore(tmp_path)
    original = store.start(issue_number=42, branch="agent/issue-42")

    def fail_replacement(source: Path, destination: Path) -> None:
        raise OSError("disk failure")

    monkeypatch.setattr("simple_coding_agent.attempt_state.os.replace", fail_replacement)

    with pytest.raises(AttemptStateError, match="written"):
        store.transition(AttemptPhase.SETUP)

    assert store.read() == original


def test_gives_each_new_attempt_a_distinct_started_at(tmp_path: Path) -> None:
    first = datetime(2026, 9, 20, 13, 0, tzinfo=UTC)
    store = AttemptStateStore(tmp_path, clock=SequenceClock(first, first))

    first_attempt = store.start(issue_number=42, branch="agent/issue-42")
    store.delete()
    second_attempt = store.start(issue_number=42, branch="agent/issue-42")

    assert second_attempt.started_at != first_attempt.started_at


def test_never_moves_updated_at_before_started_at_when_the_clock_repeats(
    tmp_path: Path,
) -> None:
    timestamp = datetime(2026, 9, 20, 13, 0, tzinfo=UTC)
    store = AttemptStateStore(tmp_path, clock=SequenceClock(timestamp, timestamp, timestamp))
    store.start(issue_number=42, branch="agent/issue-42")
    store.delete()
    attempt = store.start(issue_number=42, branch="agent/issue-42")

    transitioned = store.transition(AttemptPhase.SETUP)

    assert transitioned.started_at == "2026-09-20T13:00:00.000001Z"
    assert attempt.started_at == transitioned.started_at
    assert transitioned.updated_at == "2026-09-20T13:00:00.000002Z"


def test_claim_time_auto_merge_opt_in_defaults_to_false(tmp_path: Path) -> None:
    store = AttemptStateStore(tmp_path)

    attempt = store.start(issue_number=42, branch="agent/issue-42")

    assert attempt.auto_merge_opt_in is False


def test_claim_time_auto_merge_opt_in_survives_transitions(tmp_path: Path) -> None:
    started = datetime(2026, 9, 20, 13, 0, tzinfo=UTC)
    store = AttemptStateStore(tmp_path, clock=SequenceClock(started, started))

    store.start(issue_number=42, branch="agent/issue-42", auto_merge_opt_in=True)

    transitioned = store.transition(AttemptPhase.SETUP)

    assert transitioned.auto_merge_opt_in is True
    assert store.read() is not None
    assert store.read().auto_merge_opt_in is True


def test_checkpoint_without_opt_in_reads_as_not_opted_in(tmp_path: Path) -> None:
    path = tmp_path / "state" / "attempt.json"
    path.parent.mkdir()
    path.write_text(
        '{"branch": "agent/issue-42", "issue_number": 42, "phase": "claimed",'
        ' "started_at": "2026-09-20T13:00:00Z", "updated_at": "2026-09-20T13:00:00Z"}'
    )

    assert AttemptStateStore(tmp_path).read().auto_merge_opt_in is False


def test_checkpoint_with_a_non_boolean_opt_in_is_malformed(tmp_path: Path) -> None:
    path = tmp_path / "state" / "attempt.json"
    path.parent.mkdir()
    path.write_text(
        '{"auto_merge_opt_in": "yes", "branch": "agent/issue-42", "issue_number": 42,'
        ' "phase": "claimed", "started_at": "2026-09-20T13:00:00Z",'
        ' "updated_at": "2026-09-20T13:00:00Z"}'
    )

    with pytest.raises(AttemptStateError, match="malformed"):
        AttemptStateStore(tmp_path).read()


def test_claim_time_requeue_source_defaults_to_human(tmp_path: Path) -> None:
    store = AttemptStateStore(tmp_path)

    attempt = store.start(issue_number=42, branch="agent/issue-42")

    assert attempt.requeue_source == "human"


def test_claim_time_requeue_source_survives_transitions(tmp_path: Path) -> None:
    started = datetime(2026, 9, 20, 13, 0, tzinfo=UTC)
    store = AttemptStateStore(tmp_path, clock=SequenceClock(started, started))

    store.start(issue_number=42, branch="agent/issue-42", requeue_source="automatic")

    transitioned = store.transition(AttemptPhase.SETUP)

    assert transitioned.requeue_source == "automatic"
    assert store.read() is not None
    assert store.read().requeue_source == "automatic"


def test_checkpoint_without_requeue_source_reads_as_human(tmp_path: Path) -> None:
    path = tmp_path / "state" / "attempt.json"
    path.parent.mkdir()
    path.write_text(
        '{"branch": "agent/issue-42", "issue_number": 42, "phase": "claimed",'
        ' "started_at": "2026-09-20T13:00:00Z", "updated_at": "2026-09-20T13:00:00Z"}'
    )

    assert AttemptStateStore(tmp_path).read().requeue_source == "human"


def test_checkpoint_with_an_unknown_requeue_source_is_malformed(tmp_path: Path) -> None:
    path = tmp_path / "state" / "attempt.json"
    path.parent.mkdir()
    path.write_text(
        '{"branch": "agent/issue-42", "issue_number": 42, "phase": "claimed",'
        ' "requeue_source": "sometimes", "started_at": "2026-09-20T13:00:00Z",'
        ' "updated_at": "2026-09-20T13:00:00Z"}'
    )

    with pytest.raises(AttemptStateError, match="malformed"):
        AttemptStateStore(tmp_path).read()


def test_start_rejects_an_unknown_requeue_source(tmp_path: Path) -> None:
    store = AttemptStateStore(tmp_path)

    with pytest.raises(AttemptStateError, match="Requeue source"):
        store.start(issue_number=42, branch="agent/issue-42", requeue_source="sometimes")


def test_claim_lease_defaults_to_unrecorded(tmp_path: Path) -> None:
    store = AttemptStateStore(tmp_path)
    store.start(issue_number=42, branch="agent/issue-42")

    checkpoint = store.read()

    assert checkpoint is not None
    assert checkpoint.claim_branch_revision is None
    assert checkpoint.claim_lease_recorded is False


def test_record_claim_lease_stores_the_observed_remote_revision(tmp_path: Path) -> None:
    store = AttemptStateStore(tmp_path)
    store.start(issue_number=42, branch="agent/issue-42")

    recorded = store.record_claim_lease("abc123")

    assert recorded.claim_branch_revision == "abc123"
    assert recorded.claim_lease_recorded is True
    assert store.read() == recorded


def test_record_claim_lease_stores_an_absent_branch(tmp_path: Path) -> None:
    store = AttemptStateStore(tmp_path)
    store.start(issue_number=42, branch="agent/issue-42")

    recorded = store.record_claim_lease(None)

    assert recorded.claim_branch_revision is None
    assert recorded.claim_lease_recorded is True


def test_record_claim_lease_keeps_the_first_recording(tmp_path: Path) -> None:
    store = AttemptStateStore(tmp_path)
    store.start(issue_number=42, branch="agent/issue-42")
    store.record_claim_lease("first")

    rerecorded = store.record_claim_lease("second")

    assert rerecorded.claim_branch_revision == "first"
    assert rerecorded.claim_lease_recorded is True


def test_advance_claim_lease_moves_to_the_pushed_revision(tmp_path: Path) -> None:
    store = AttemptStateStore(tmp_path)
    store.start(issue_number=42, branch="agent/issue-42")
    store.record_claim_lease(None)

    advanced = store.advance_claim_lease("pushed-sha")

    assert advanced.claim_branch_revision == "pushed-sha"
    assert advanced.claim_lease_recorded is True
    assert store.read() == advanced


def test_advance_claim_lease_records_a_lease_for_an_older_checkpoint(tmp_path: Path) -> None:
    store = AttemptStateStore(tmp_path)
    store.start(issue_number=42, branch="agent/issue-42")

    advanced = store.advance_claim_lease("pushed-sha")

    assert advanced.claim_branch_revision == "pushed-sha"
    assert advanced.claim_lease_recorded is True


def test_claim_lease_survives_transitions_and_reopening(tmp_path: Path) -> None:
    store = AttemptStateStore(tmp_path)
    store.start(issue_number=42, branch="agent/issue-42")
    store.record_claim_lease("abc123")
    store.transition(AttemptPhase.SETUP)

    reopened = AttemptStateStore(tmp_path).read()

    assert reopened is not None
    assert reopened.claim_branch_revision == "abc123"
    assert reopened.claim_lease_recorded is True


def test_checkpoint_without_a_claim_lease_reads_as_unrecorded(tmp_path: Path) -> None:
    path = tmp_path / "state" / "attempt.json"
    path.parent.mkdir()
    path.write_text(
        '{"branch": "agent/issue-42", "issue_number": 42, "phase": "claimed",'
        ' "started_at": "2026-09-20T13:00:00Z", "updated_at": "2026-09-20T13:00:00Z"}'
    )

    checkpoint = AttemptStateStore(tmp_path).read()

    assert checkpoint is not None
    assert checkpoint.claim_branch_revision is None
    assert checkpoint.claim_lease_recorded is False


def test_checkpoint_with_a_claim_lease_but_no_flag_reads_as_unrecorded(tmp_path: Path) -> None:
    path = tmp_path / "state" / "attempt.json"
    path.parent.mkdir()
    path.write_text(
        '{"branch": "agent/issue-42", "issue_number": 42, "phase": "claimed",'
        ' "started_at": "2026-09-20T13:00:00Z", "updated_at": "2026-09-20T13:00:00Z",'
        ' "claim_branch_revision": "abc123"}'
    )

    checkpoint = AttemptStateStore(tmp_path).read()

    assert checkpoint is not None
    assert checkpoint.claim_branch_revision is None
    assert checkpoint.claim_lease_recorded is False


def test_record_claim_lease_rejects_a_blank_revision(tmp_path: Path) -> None:
    store = AttemptStateStore(tmp_path)
    store.start(issue_number=42, branch="agent/issue-42")

    with pytest.raises(AttemptStateError, match="non-empty"):
        store.record_claim_lease("  ")


def test_advance_claim_lease_rejects_an_absent_revision(tmp_path: Path) -> None:
    store = AttemptStateStore(tmp_path)
    store.start(issue_number=42, branch="agent/issue-42")

    with pytest.raises(AttemptStateError, match="advance requires a revision"):
        store.advance_claim_lease(None)
