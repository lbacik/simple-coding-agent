"""``last_finished`` agreement and the lost-claim release path.

The Status Snapshot's ``last_finished`` must agree with the Attempt
history: a finished Handoff (automatic or operator) sets no operator
alert by itself, and an Attempt dropped through the lost-claim path
(``recovery release`` when the issue is no longer claimed by this
Worker, or startup reconciliation across a restart) still finalizes as
``released`` with the recorded outcome.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from simple_coding_agent.attempt_history import (
    read_sealing_record,
    requires_attention,
    show_attempt,
)
from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.contract_status import project_last_finished
from simple_coding_agent.control import ControlStore
from simple_coding_agent.observability import AttemptArchive

REPOSITORY = "octo/example"


def _tracker_issue(number: int, **overrides: object):
    from simple_coding_agent.github_tracker import TrackerIssue

    base: dict[str, object] = {
        "id": f"issue-{number}",
        "number": number,
        "title": f"Issue {number}",
        "body": f"Implement {number}.",
        "created_at": datetime(2026, 9, 20, 13, 0, tzinfo=UTC),
        "state": "OPEN",
        "labels": frozenset({"ready-for-agent"}),
        "assignee_logins": (),
        "blocked_by": 0,
        "author_login": "reporter",
    }
    base.update(overrides)
    return TrackerIssue(**base)  # type: ignore[arg-type]


def _scripted_clock(start: datetime = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)):
    state = {"now": start}

    def tick() -> datetime:
        state["now"] = state["now"] + timedelta(seconds=1)
        return state["now"]

    return tick


def _make_lifecycle(tmp_path: Path, tracker):
    """Wire the real lifecycle with control, archives and telemetry."""

    from simple_coding_agent.config import RepositoryProfile
    from simple_coding_agent.lifecycle import AgentLifecycle
    from simple_coding_agent.telemetry import TelemetryReporter
    from tests.fakes import FakePublisher, InMemoryWorkspace

    events: list[tuple[str, str, str]] = []
    lifecycle = AgentLifecycle(
        tracker=tracker,  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: RepositoryProfile(
            setup=("setup.sh",),
            check=("pytest",),
            base_branch="main",
            timeout=300,
            setup_timeout=120,
            env={},
        ),
        publisher=FakePublisher(),
        attempt_runner=None,
        control_store=ControlStore(tmp_path),
        repository=REPOSITORY,
        telemetry=TelemetryReporter(
            repository=REPOSITORY,
            agent_version="0.0-test",
            sdk_version="0.2.163-test",
            cli_version="2.1.286-test",
            process_started_at="2026-01-01T00:00:00Z",
            hard_ceiling=4_000_000,
            turns_max=60,
            usd_ceiling=20,
            timeout_seconds=3600,
            continuation_max=0,
            clock=_scripted_clock(),
        ),
        data_dir=tmp_path,
        sleeper=lambda seconds: None,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, level, detail)
        ),
        attempt_archive_factory=lambda issue_number, started_at: AttemptArchive(
            tmp_path, issue_number=issue_number, started_at=started_at
        ),
    )
    return lifecycle, events


def _last_finished(lifecycle) -> dict | None:
    return lifecycle._telemetry.contract_source().get("last_finished")  # noqa: SLF001


def _handoff_summary(**overrides: object) -> dict:
    summary: dict[str, object] = {
        "attempt_id": "2026-11-20T15:04:14.752341Z",
        "issue_number": 573,
        "issue_title": "Do the thing",
        "outcome": "handoff",
        "finished_at": "2026-11-20T15:05:00.000000Z",
        "lifecycle_state": "finished",
    }
    summary.update(overrides)
    return summary


# ---------------------------------------------------------------------------
# attention_required agreement
# ---------------------------------------------------------------------------


def test_finished_handoff_sets_no_attention() -> None:
    """An automatic Handoff must not raise an operator alert."""

    projected = project_last_finished(_handoff_summary())

    assert projected is not None
    assert projected["outcome"]["native"] == "handoff"
    assert projected["attention_required"] is False


def test_last_finished_attention_matches_attempt_history(tmp_path: Path) -> None:
    """``status`` and ``attempts`` report the same ``attention_required``."""

    attempt_id = "2026-11-20T15:04:14.752341Z"
    archive = AttemptArchive(tmp_path, issue_number=573, started_at=attempt_id)
    archive.write_attempt(
        {
            "started_at": attempt_id,
            "issue_number": 573,
            "issue_title": "Do the thing",
            "outcome": "handoff",
            "completed_at": "2026-11-20T15:05:00.000000Z",
        }
    )

    shown = show_attempt(tmp_path, attempt_id)["attempt"]
    projected = project_last_finished(_handoff_summary())

    assert projected is not None
    assert projected["attention_required"] is False
    assert shown["attention_required"] is False
    assert projected["attention_required"] == shown["attention_required"]
    assert projected["attention_required"] == requires_attention(attempt_id, frozenset())


def test_held_attempt_still_requires_attention() -> None:
    """The shared helper keeps flagging work held for recovery."""

    attempt_id = "2026-11-20T15:04:14.752341Z"

    assert requires_attention(attempt_id, {attempt_id}) is True
    projected = project_last_finished(_handoff_summary(), held={attempt_id})

    assert projected is not None
    assert projected["attention_required"] is True


# ---------------------------------------------------------------------------
# lost-claim finalization through recovery release
# ---------------------------------------------------------------------------


def _start_retained_attempt(tmp_path: Path, issue_number: int = 1):
    """Open a checkpoint plus an archive carrying the recorded verdict."""

    checkpoint = AttemptStateStore(tmp_path).start(
        issue_number=issue_number,
        branch=f"agent/issue-{issue_number}",
        issue_title=f"Issue {issue_number}",
    )
    archive = AttemptArchive(
        tmp_path, issue_number=issue_number, started_at=checkpoint.started_at
    )
    # Publication recorded its verdict (ADR 0010): the outcome only, the
    # completion boundary never ran, so there is no completed_at yet.
    archive.write_attempt(
        {
            "started_at": checkpoint.started_at,
            "issue_number": issue_number,
            "issue_title": f"Issue {issue_number}",
            "outcome": "infrastructure_error",
        }
    )
    return checkpoint, archive


def test_recovery_release_lost_claim_finalizes_as_released(tmp_path: Path) -> None:
    """A release that loses the claim still finalizes ``last_finished``."""

    from tests.fakes import FakeTracker

    tracker = FakeTracker(
        issues={1: _tracker_issue(1)},
        worker_id="worker-a",
    )
    lifecycle, events = _make_lifecycle(tmp_path, tracker)
    checkpoint, archive = _start_retained_attempt(tmp_path)
    attempt_id = checkpoint.started_at

    cleared, _detail = lifecycle._release_retained_attempt(  # noqa: SLF001
        checkpoint, "s3://saved"
    )

    assert cleared is True
    assert any(event == "claim_lost" for event, _, _ in events)
    # Nothing was published and no Terminal Outcome reached the ledger.
    assert lifecycle._publisher.requests == []  # noqa: SLF001
    # The checkpoint is gone and the archive is Sealed.
    assert AttemptStateStore(tmp_path).read() is None
    assert read_sealing_record(archive.directory) is not None
    # The archive record finalizes as a release without rewriting the
    # recorded outcome.
    record = archive.read_attempt()
    assert record.get("lifecycle_state") == "released"
    assert record.get("outcome") == "infrastructure_error"
    assert isinstance(record.get("completed_at"), str)
    # Telemetry names the released Attempt with the recorded outcome.
    last = _last_finished(lifecycle)
    assert last is not None
    assert last["attempt_id"] == attempt_id
    assert last["lifecycle_state"] == "released"
    assert last["outcome"] == "infrastructure_error"
    # The history agrees: released, same outcome, no attention.
    shown = lifecycle.control_attempts_show(attempt_id)["attempt"]  # noqa: SLF001
    assert shown["lifecycle_state"] == "released"
    assert shown["outcome"] is not None
    assert shown["outcome"]["native"] == last["outcome"]
    assert shown["attention_required"] is False
    # And so does the Status Snapshot.
    finished = lifecycle.contract_status_snapshot()["last_finished"]  # noqa: SLF001
    assert finished["attempt_id"] == attempt_id
    assert finished["lifecycle_state"] == "released"
    assert finished["outcome"]["native"] == shown["outcome"]["native"]
    assert finished["attention_required"] is False
    assert finished["attention_required"] == shown["attention_required"]


def test_recovery_release_lost_claim_without_recorded_outcome(tmp_path: Path) -> None:
    """Without a recorded verdict the lost-claim outcome is infra error."""

    from tests.fakes import FakeTracker

    tracker = FakeTracker(
        issues={1: _tracker_issue(1)},
        worker_id="worker-a",
    )
    lifecycle, _events = _make_lifecycle(tmp_path, tracker)
    checkpoint = AttemptStateStore(tmp_path).start(
        issue_number=1, branch="agent/issue-1", issue_title="Issue 1"
    )
    archive = AttemptArchive(
        tmp_path, issue_number=1, started_at=checkpoint.started_at
    )
    archive.write_attempt(
        {
            "started_at": checkpoint.started_at,
            "issue_number": 1,
            "issue_title": "Issue 1",
        }
    )

    cleared, _detail = lifecycle._release_retained_attempt(  # noqa: SLF001
        checkpoint, "s3://saved"
    )

    assert cleared is True
    last = _last_finished(lifecycle)
    assert last is not None
    assert last["attempt_id"] == checkpoint.started_at
    assert last["lifecycle_state"] == "released"
    assert last["outcome"] == "infrastructure_error"
    assert archive.read_attempt().get("lifecycle_state") == "released"


def test_startup_reconciliation_lost_claim_finalizes_as_released(
    tmp_path: Path,
) -> None:
    """A claim lost across a restart finalizes exactly like a release."""

    from tests.fakes import FakeTracker

    tracker = FakeTracker(
        issues={1: _tracker_issue(1)},
        worker_id="worker-a",
    )
    lifecycle, events = _make_lifecycle(tmp_path, tracker)
    checkpoint, archive = _start_retained_attempt(tmp_path)
    attempt_id = checkpoint.started_at

    lifecycle.run_once()

    assert any(event == "claim_lost" for event, _, _ in events)
    assert AttemptStateStore(tmp_path).read() is None
    assert read_sealing_record(archive.directory) is not None
    record = archive.read_attempt()
    assert record.get("lifecycle_state") == "released"
    assert record.get("outcome") == "infrastructure_error"
    last = _last_finished(lifecycle)
    assert last is not None
    assert last["attempt_id"] == attempt_id
    assert last["lifecycle_state"] == "released"
    assert last["outcome"] == "infrastructure_error"
    shown = lifecycle.control_attempts_show(attempt_id)["attempt"]  # noqa: SLF001
    assert shown["lifecycle_state"] == "released"
    assert shown["outcome"] is not None
    assert shown["outcome"]["native"] == last["outcome"]
    assert shown["attention_required"] is False
    finished = lifecycle.contract_status_snapshot()["last_finished"]  # noqa: SLF001
    assert finished["attempt_id"] == attempt_id
    assert finished["lifecycle_state"] == "released"
    assert finished["outcome"]["native"] == shown["outcome"]["native"]
    assert finished["attention_required"] == shown["attention_required"]
