"""Restart recovery of a retained conflict attempt (issue #273).

A Worker that restarts in the middle of a conflict attempt must recover it
as a conflict attempt, not an ordinary one: the checkpoint and
``attempt.json`` record the attempt kind, and startup recovery of a
retained conflict attempt releases ``merge-conflict`` (never
``ready-for-agent``/``round-finished``), posts a failure result carrying
the conflict-attempt marker and the ``conflict`` requeue source, and
removes the assignee and the Claim while leaving the remote branch
untouched.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from simple_coding_agent.attempt_state import (
    ATTEMPT_KIND_CONFLICT,
    ATTEMPT_KIND_ORDINARY,
    AttemptStateStore,
)
from simple_coding_agent.completion import AttemptOutcome
from simple_coding_agent.config import RepositoryProfile
from simple_coding_agent.github_tracker import TrackerIssue
from simple_coding_agent.lifecycle import AgentLifecycle
from simple_coding_agent.observability import AttemptArchive
from simple_coding_agent.publication import (
    ConflictPublicationRequest,
    PublicationRequest,
    PullRequest,
    _conflict_result_comment,
)
from simple_coding_agent.recovery import (
    REQUEUE_SOURCE_AUTOMATIC,
    REQUEUE_SOURCE_CONFLICT,
    REQUEUE_SOURCE_HUMAN,
    attempt_marker,
    conflict_attempt_marker,
    count_automatic_continuations,
    count_conflict_failures,
    is_conflict_failure,
    parse_requeue_source,
    requeue_source_marker,
)
from tests.fakes import (
    FakeCommandRunner,
    FakeConflictExecutor,
    FakePublisher,
    FakeTracker,
    InMemoryWorkspace,
)

CONFLICT = frozenset({"merge-conflict"})
READY = frozenset({"ready-for-agent"})

_CREATED_BASE = datetime(2026, 9, 20, 13, 0, tzinfo=UTC)


def issue(number: int = 7, **overrides: object) -> TrackerIssue:
    base: dict[str, object] = {
        "id": f"issue-{number}",
        "number": number,
        "title": f"Issue {number}",
        "body": f"Implement {number}.",
        "created_at": _CREATED_BASE + timedelta(seconds=number),
        "state": "OPEN",
        "labels": CONFLICT,
        "assignee_logins": (),
        "blocked_by": 0,
        "author_login": "reporter",
    }
    base.update(overrides)
    return TrackerIssue(**base)  # type: ignore[arg-type]


def pull_request(number: int = 7, **overrides: object) -> PullRequest:
    base: dict[str, object] = {
        "number": number,
        "url": f"https://example.test/pull/{number}",
        "node_id": f"pr-{number}",
        "draft": False,
        "mergeable": "CONFLICTING",
        "base_ref": "main",
    }
    base.update(overrides)
    return PullRequest(**base)  # type: ignore[arg-type]


def profile() -> RepositoryProfile:
    return RepositoryProfile(
        setup=("setup.sh",),
        check=("pytest",),
        base_branch="main",
        timeout=300,
        setup_timeout=120,
        env={},
    )


def raising_runner(claim: object, profile: object, prepared: object) -> object:
    raise AssertionError("a conflict attempt must never call the ordinary model path")


def make_lifecycle(
    tmp_path: Path,
    tracker: FakeTracker,
    store_dir: Path,
    publisher: FakePublisher,
    *,
    data_dir: Path | None = None,
) -> AgentLifecycle:
    return AgentLifecycle(
        tracker=tracker,  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(store_dir),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: profile(),
        publisher=publisher,
        attempt_runner=raising_runner,  # type: ignore[arg-type]
        control_store=None,
        sleeper=lambda seconds: None,
        event_log=lambda event, detail="", level="INFO", issue_number=None: None,
        max_conflict_attempts=2,
        command_runner=FakeCommandRunner([]),
        conflict_model_executor=FakeConflictExecutor(),
        conflict_issue_comments=lambda _: (),
        attempt_archive_factory=(
            (lambda issue_number, started_at: AttemptArchive(
                data_dir, issue_number=issue_number, started_at=started_at
            ))
            if data_dir is not None
            else None
        ),
    )


# ---------------------------------------------------------------------------
# checkpoint kind: round-trip and legacy defaults
# ---------------------------------------------------------------------------


def test_conflict_kind_round_trips_through_checkpoint_json(tmp_path: Path) -> None:
    store = AttemptStateStore(tmp_path / "store")
    checkpoint = store.start(
        issue_number=7,
        branch="agent/issue-7",
        requeue_source=REQUEUE_SOURCE_CONFLICT,
        attempt_kind=ATTEMPT_KIND_CONFLICT,
    )

    assert checkpoint.attempt_kind == ATTEMPT_KIND_CONFLICT
    assert checkpoint.requeue_source == REQUEUE_SOURCE_CONFLICT
    reread = store.read()
    assert reread is not None
    assert reread.attempt_kind == ATTEMPT_KIND_CONFLICT
    assert reread.requeue_source == REQUEUE_SOURCE_CONFLICT
    raw = json.loads((tmp_path / "store" / "state" / "attempt.json").read_text())
    assert raw["attempt_kind"] == ATTEMPT_KIND_CONFLICT
    assert raw["requeue_source"] == REQUEUE_SOURCE_CONFLICT


def test_checkpoint_without_kind_reads_ordinary(tmp_path: Path) -> None:
    store = AttemptStateStore(tmp_path / "store")
    store.start(issue_number=7, branch="agent/issue-7", requeue_source="human")
    path = tmp_path / "store" / "state" / "attempt.json"
    raw = json.loads(path.read_text())
    del raw["attempt_kind"]
    path.write_text(json.dumps(raw))

    reread = store.read()

    assert reread is not None
    assert reread.attempt_kind == ATTEMPT_KIND_ORDINARY


def test_legacy_conflict_flag_maps_to_conflict_kind(tmp_path: Path) -> None:
    store = AttemptStateStore(tmp_path / "store")
    store.start(issue_number=7, branch="agent/issue-7", requeue_source="human")
    path = tmp_path / "store" / "state" / "attempt.json"
    raw = json.loads(path.read_text())
    del raw["attempt_kind"]
    raw["is_conflict_claim"] = True
    path.write_text(json.dumps(raw))

    reread = store.read()

    assert reread is not None
    assert reread.attempt_kind == ATTEMPT_KIND_CONFLICT


# ---------------------------------------------------------------------------
# conflict requeue source: marker render/parse, budget transparency
# ---------------------------------------------------------------------------


def test_conflict_requeue_source_marker_round_trips() -> None:
    assert (
        requeue_source_marker(REQUEUE_SOURCE_CONFLICT)
        == "<!-- requeue-source: conflict -->"
    )
    assert (
        parse_requeue_source(requeue_source_marker(REQUEUE_SOURCE_CONFLICT))
        == REQUEUE_SOURCE_CONFLICT
    )
    assert parse_requeue_source("no markers here") is None
    with pytest.raises(ValueError):
        requeue_source_marker("bogus")


def test_conflict_results_stay_transparent_to_the_automatic_count() -> None:
    automatic = (
        "## Agent Attempt Result: complete\n\nDone.\n\n"
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->\n"
        f"{requeue_source_marker(REQUEUE_SOURCE_AUTOMATIC)}"
    )
    conflict_failure = (
        "## Conflict Attempt Result: error\n\nRestarted.\n\n"
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->\n"
        f"{conflict_attempt_marker('error', sha='none')}\n"
        f"{requeue_source_marker(REQUEUE_SOURCE_CONFLICT)}"
    )

    assert count_automatic_continuations([automatic]) == 1
    assert count_automatic_continuations([conflict_failure, automatic]) == 1
    assert count_conflict_failures([conflict_failure]) == 1


# ---------------------------------------------------------------------------
# live claim: kind, source, and attempt.json
# ---------------------------------------------------------------------------


def test_live_conflict_claim_records_kind_source_and_archive(tmp_path: Path) -> None:
    tracker = FakeTracker({7: issue()}, fifo=[7])
    publisher = FakePublisher(pull_requests={"agent/issue-7": pull_request()})
    data_dir = tmp_path / "data"
    lifecycle = make_lifecycle(
        tmp_path, tracker, tmp_path / "store", publisher, data_dir=data_dir
    )

    claim, checkpoint = lifecycle._gated_claim()

    assert claim is not None and checkpoint is not None
    assert checkpoint.attempt_kind == ATTEMPT_KIND_CONFLICT
    assert checkpoint.requeue_source == REQUEUE_SOURCE_CONFLICT
    archive = AttemptArchive(
        data_dir, issue_number=7, started_at=checkpoint.started_at
    )
    assert archive.read_attempt().get("attempt_kind") == ATTEMPT_KIND_CONFLICT


def test_live_ordinary_claim_leaves_kind_out_of_the_archive(tmp_path: Path) -> None:
    tracker = FakeTracker({7: issue(labels=READY)}, fifo=[7])
    publisher = FakePublisher()
    data_dir = tmp_path / "data"
    lifecycle = make_lifecycle(
        tmp_path, tracker, tmp_path / "store", publisher, data_dir=data_dir
    )

    claim, checkpoint = lifecycle._gated_claim()

    assert claim is not None and checkpoint is not None
    assert checkpoint.attempt_kind == ATTEMPT_KIND_ORDINARY
    assert checkpoint.requeue_source == REQUEUE_SOURCE_HUMAN
    archive = AttemptArchive(
        data_dir, issue_number=7, started_at=checkpoint.started_at
    )
    assert "attempt_kind" not in archive.read_attempt()


# ---------------------------------------------------------------------------
# startup recovery: a retained conflict attempt recovers as a conflict error
# ---------------------------------------------------------------------------


def test_startup_recovery_of_conflict_attempt(tmp_path: Path) -> None:
    tracker = FakeTracker({7: issue()}, comments={7: []})
    publisher = FakePublisher(pull_requests={"agent/issue-7": pull_request()})
    store = AttemptStateStore(tmp_path / "store")
    store.start(
        issue_number=7,
        branch="agent/issue-7",
        requeue_source=REQUEUE_SOURCE_CONFLICT,
        attempt_kind=ATTEMPT_KIND_CONFLICT,
    )
    lifecycle = make_lifecycle(tmp_path, tracker, tmp_path / "store", publisher)

    assert lifecycle._reconcile_startup() is AttemptOutcome.INFRASTRUCTURE_ERROR

    # Exactly one result, and it is the conflict error: nothing pushed, so
    # the remote branch is left as it was.
    assert len(publisher.conflict_requests) == 1
    assert all(
        isinstance(request, ConflictPublicationRequest)
        for request in publisher.requests
    )
    request = publisher.conflict_requests[0]
    assert request.conflict_outcome == "error"
    assert request.push_branch is False
    assert request.requeue_source == REQUEUE_SOURCE_CONFLICT

    # The release drops the conflict signal and the assignee, and never
    # re-adds the ordinary requeue labels or a handoff label.
    assert tracker.cleanup == [(7, "merge-conflict", "agent-id")]
    assert all(
        label not in ("ready-for-agent", "round-finished")
        for _, label in tracker.added_labels
    )

    # The rendered comment carries the conflict-attempt marker (a failure
    # that counts toward the budget) and the conflict requeue source, and
    # no handoff note.
    body = _conflict_result_comment(
        request,
        request.outcome,
        request.conflict_outcome,
        None,
        request.pull_request,
        request.details,
        attempt_marker(request.started_at),
        None,
        None,
    )
    assert "## Conflict Attempt Result: error" in body
    assert is_conflict_failure(body) is True
    assert count_conflict_failures([body]) == 1
    assert parse_requeue_source(body) == REQUEUE_SOURCE_CONFLICT
    assert "## Handoff" not in body

    # The Claim is gone: no checkpoint survives recovery.
    assert lifecycle._attempt_state.read() is None


def test_startup_recovery_of_ordinary_attempt_is_unchanged(tmp_path: Path) -> None:
    tracker = FakeTracker({7: issue(labels=READY)}, comments={7: []})
    publisher = FakePublisher(pull_requests={"agent/issue-7": pull_request()})
    store = AttemptStateStore(tmp_path / "store")
    store.start(issue_number=7, branch="agent/issue-7", requeue_source="human")
    lifecycle = make_lifecycle(tmp_path, tracker, tmp_path / "store", publisher)

    assert lifecycle._reconcile_startup() is AttemptOutcome.INFRASTRUCTURE_ERROR

    assert publisher.conflict_requests == []
    assert any(
        isinstance(request, PublicationRequest) for request in publisher.requests
    )
    assert tracker.cleanup == [(7, "ready-for-agent", "agent-id")]
    assert lifecycle._attempt_state.read() is None


def test_legacy_checkpoint_without_kind_recovers_ordinary(tmp_path: Path) -> None:
    tracker = FakeTracker({7: issue(labels=READY)}, comments={7: []})
    publisher = FakePublisher(pull_requests={"agent/issue-7": pull_request()})
    store = AttemptStateStore(tmp_path / "store")
    store.start(issue_number=7, branch="agent/issue-7", requeue_source="human")
    path = tmp_path / "store" / "state" / "attempt.json"
    raw = json.loads(path.read_text())
    del raw["attempt_kind"]
    path.write_text(json.dumps(raw))
    lifecycle = make_lifecycle(tmp_path, tracker, tmp_path / "store", publisher)

    assert lifecycle._reconcile_startup() is AttemptOutcome.INFRASTRUCTURE_ERROR

    assert publisher.conflict_requests == []
    assert tracker.cleanup == [(7, "ready-for-agent", "agent-id")]
    assert lifecycle._attempt_state.read() is None
