"""Claim hardening: two workers with distinct logins never work one issue.

Post-assign verification on both claim paths (``claim_next`` for the Queue
and ``claim_verified`` for the ``agentctl next issue`` priority target):
after assigning, the worker waits out a short settle delay, re-reads the
issue, and yields when anyone else is assigned. Yielding removes only this
worker's own assignment (no labels, comments, or checkpoint) and emits a
``claim_lost`` event. Only the verified winner removes a stale
``round-finished``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from simple_coding_agent.lifecycle import AgentLifecycle, AttemptEvidence

import simple_coding_agent.github_tracker as tracker_module
from simple_coding_agent.completion import AttemptOutcome, CompletionDecision, PublicationPath
from simple_coding_agent.control import CommandAcknowledgement, ControlStore
from simple_coding_agent.github_tracker import (
    CLAIM_SETTLE_DELAY_SECONDS,
    Assignment,
    ClaimLostError,
    GitHubIdentity,
    GitHubTracker,
    IssuePage,
    TrackerIssue,
)
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace


def issue(number: int, **overrides: object) -> TrackerIssue:
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


class RaceTransport:
    """A fake transport where rivals appear between assign and re-read.

    Reads before this worker assigns return the seeded snapshot; reads
    after this worker assigns return it with this worker (``agent``) plus
    the seeded rivals present, simulating a second assignee arriving
    during the settle window. This mirrors both claim paths: ``claim_next``
    reads once for eligibility (clean) and once for verification
    (contested), while ``claim_verified`` assigns first and its single
    verification read is contested. Numbers absent from ``rivals`` verify
    cleanly with only this worker assigned.
    """

    def __init__(
        self,
        issues: dict[int, TrackerIssue],
        rivals: dict[int, tuple[str, ...]] | None = None,
    ) -> None:
        self._issues = dict(issues)
        self._rivals = dict(rivals or {})
        self._assigned: set[str] = set()
        self.assignments: list[tuple[str, str]] = []
        self.removed_assignees: list[tuple[str, str]] = []
        self.removed_labels: list[tuple[str, str]] = []
        self.events: list[tuple[str, str, str, int | None]] = []

    def event_log(
        self,
        event: str,
        detail: str = "",
        level: str = "INFO",
        issue_number: int | None = None,
    ) -> None:
        self.events.append((event, detail, level, issue_number))

    def list_issues(self, repository: str, cursor: str | None) -> IssuePage:
        assert repository == "octo/example"
        assert cursor is None
        return IssuePage(issues=tuple(self._issues.values()), next_cursor=None)

    def get_issue(self, repository: str, number: int) -> TrackerIssue | None:
        assert repository == "octo/example"
        base = self._issues.get(number)
        if base is None:
            return None
        if base.id not in self._assigned:
            return base
        rivals = self._rivals.get(number, ())
        return TrackerIssue(
            id=base.id,
            number=base.number,
            title=base.title,
            body=base.body,
            created_at=base.created_at,
            state=base.state,
            labels=base.labels,
            assignee_logins=("agent", *rivals),
            blocked_by=base.blocked_by,
            author_login=base.author_login,
        )

    def viewer(self) -> GitHubIdentity:
        return GitHubIdentity(id="viewer-id", login="agent")

    def assign_issue(self, issue_id: str, assignee_id: str) -> Assignment:
        self.assignments.append((issue_id, assignee_id))
        self._assigned.add(issue_id)
        return Assignment(issue_id=issue_id, assignee_id=assignee_id)

    def remove_assignee(self, issue_id: str, assignee_id: str) -> None:
        self.removed_assignees.append((issue_id, assignee_id))

    def remove_label(self, issue_id: str, label: str) -> None:
        self.removed_labels.append((issue_id, label))

    def list_issue_comments(self, repository: str, issue_number: int) -> tuple:
        return ()


def record_settle(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record settle-window sleeps without waiting (overrides the suite stub)."""

    calls: list[float] = []
    monkeypatch.setattr(
        tracker_module.time, "sleep", lambda seconds: calls.append(seconds)
    )
    return calls


# ---------------------------------------------------------------------------
# settle delay
# ---------------------------------------------------------------------------


def test_settle_delay_is_a_short_constant_without_a_config_knob() -> None:
    assert 2.0 <= CLAIM_SETTLE_DELAY_SECONDS <= 3.0


def test_claim_next_waits_the_settle_delay_before_verifying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = record_settle(monkeypatch)
    transport = RaceTransport({1: issue(1)})
    tracker = GitHubTracker(transport, "octo/example", event_log=transport.event_log)

    assert tracker.claim_next() is not None

    assert calls == [CLAIM_SETTLE_DELAY_SECONDS]


def test_claim_verified_waits_the_settle_delay_before_verifying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = record_settle(monkeypatch)
    transport = RaceTransport({1: issue(1)})
    tracker = GitHubTracker(transport, "octo/example", event_log=transport.event_log)

    tracker.claim_verified(transport._issues[1])

    assert calls == [CLAIM_SETTLE_DELAY_SECONDS]


# ---------------------------------------------------------------------------
# claim_next: yield and continue
# ---------------------------------------------------------------------------


def test_claim_next_yields_on_a_rival_and_claims_the_next_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_settle(monkeypatch)
    first = issue(
        1,
        labels=frozenset({"ready-for-agent", "round-finished"}),
        created_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
    )
    second = issue(2, created_at=datetime(2026, 9, 20, 13, 0, tzinfo=UTC))
    transport = RaceTransport({1: first, 2: second}, rivals={1: ("rival-worker",)})
    tracker = GitHubTracker(transport, "octo/example", event_log=transport.event_log)

    claim = tracker.claim_next()

    assert claim is not None
    assert claim.issue.number == 2
    # Yielding removed only our own assignment: no labels, no comments.
    assert transport.removed_assignees == [("issue-1", "viewer-id")]
    assert ("issue-1", "round-finished") not in transport.removed_labels
    # The contested candidate keeps its queue signal for a later poll.
    assert transport.removed_labels == []
    lost = [row for row in transport.events if row[0] == "claim_lost"]
    assert len(lost) == 1
    assert lost[0][3] == 1
    assert "rival-worker" in lost[0][1]


def test_claim_next_yielding_twice_returns_none_without_stranding_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_settle(monkeypatch)
    transport = RaceTransport(
        {
            1: issue(1, labels=frozenset({"ready-for-agent", "round-finished"})),
            2: issue(2),
        },
        rivals={1: ("rival-a",), 2: ("rival-b",)},
    )
    tracker = GitHubTracker(transport, "octo/example", event_log=transport.event_log)

    assert tracker.claim_next() is None

    assert sorted(transport.removed_assignees) == [
        ("issue-1", "viewer-id"),
        ("issue-2", "viewer-id"),
    ]
    assert transport.removed_labels == []
    assert [row[3] for row in transport.events if row[0] == "claim_lost"] == [1, 2]


def test_claim_next_winner_removes_round_finished_only_after_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_settle(monkeypatch)
    transport = RaceTransport(
        {1: issue(1, labels=frozenset({"ready-for-agent", "round-finished"}))}
    )
    tracker = GitHubTracker(transport, "octo/example", event_log=transport.event_log)

    claim = tracker.claim_next()

    assert claim is not None
    assert transport.removed_labels == [("issue-1", "round-finished")]
    assert transport.removed_assignees == []
    assert [row for row in transport.events if row[0] == "claim_lost"] == []


def test_claim_next_treats_a_human_rival_like_any_other_assignee(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_settle(monkeypatch)
    transport = RaceTransport(
        {1: issue(1), 2: issue(2)}, rivals={1: ("human-reviewer",)}
    )
    tracker = GitHubTracker(transport, "octo/example", event_log=transport.event_log)

    claim = tracker.claim_next()

    assert claim is not None
    assert claim.issue.number == 2
    assert transport.removed_assignees == [("issue-1", "viewer-id")]


# ---------------------------------------------------------------------------
# claim_verified: yield terminally
# ---------------------------------------------------------------------------


def test_claim_verified_yields_on_a_rival_without_touching_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_settle(monkeypatch)
    target = issue(1, labels=frozenset({"ready-for-agent", "round-finished"}))
    transport = RaceTransport({1: target}, rivals={1: ("rival-worker",)})
    tracker = GitHubTracker(transport, "octo/example", event_log=transport.event_log)

    with pytest.raises(ClaimLostError) as excinfo:
        tracker.claim_verified(target)

    assert excinfo.value.issue_number == 1
    assert excinfo.value.other_assignees == ("rival-worker",)
    assert transport.removed_assignees == [("issue-1", "viewer-id")]
    assert transport.removed_labels == []
    lost = [row for row in transport.events if row[0] == "claim_lost"]
    assert len(lost) == 1
    assert lost[0][3] == 1
    assert "rival-worker" in lost[0][1]


def test_claim_verified_winner_removes_round_finished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_settle(monkeypatch)
    target = issue(1, labels=frozenset({"ready-for-agent", "round-finished"}))
    transport = RaceTransport({1: target})
    tracker = GitHubTracker(transport, "octo/example", event_log=transport.event_log)

    claim = tracker.claim_verified(target)

    assert claim.issue.number == 1
    assert transport.removed_labels == [("issue-1", "round-finished")]
    assert transport.removed_assignees == []


# ---------------------------------------------------------------------------
# lifecycle: priority loss falls back to FIFO; FIFO loss continues
# ---------------------------------------------------------------------------


def make_lifecycle(
    tmp_path: Path, issues: dict[int, TrackerIssue], *, fifo: list[int]
) -> tuple[AgentLifecycle, FakeTracker, ControlStore, list[tuple]]:
    from simple_coding_agent.attempt_state import AttemptStateStore
    from simple_coding_agent.lifecycle import AgentLifecycle

    events: list[tuple] = []

    def event_log(
        event: str,
        detail: str = "",
        level: str = "INFO",
        issue_number: int | None = None,
    ) -> None:
        events.append((event, detail, level, issue_number))

    tracker = FakeTracker(issues, fifo=list(fifo), event_log=event_log)
    store = ControlStore(tmp_path)
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=lambda received_claim, profile, prepared: _evidence(),
        control_store=store,
        sleeper=lambda seconds: None,
        event_log=event_log,
    )
    return lifecycle, tracker, store, events


def _evidence() -> AttemptEvidence:
    from simple_coding_agent.lifecycle import AttemptEvidence

    return AttemptEvidence(
        decision=CompletionDecision(AttemptOutcome.COMPLETE, True, PublicationPath.COMPLETE, ("ready",)),
        check_command="pytest",
        check_exit_code=0,
        review_cycles=1,
        review_findings="all clear",
        details="Implemented.",
    )


def test_priority_loss_is_not_fulfilled_and_falls_back_to_fifo(tmp_path: Path) -> None:
    lifecycle, tracker, store, events = make_lifecycle(
        tmp_path, {24: issue(24), 25: issue(25)}, fifo=[25]
    )
    lifecycle.submit_next_issue("req-next", 24)
    tracker.claim_race[24] = ("rival-worker",)

    result = lifecycle.run_once()

    assert result.status.value == "attempted"
    assert tracker.claimed_verified == []
    assert tracker.fifo_claims == [25]
    assert tracker.yielded == [24]
    failed = store.get_command("req-next")
    assert failed is not None
    assert failed.acknowledgement is CommandAcknowledgement.NOT_FULFILLED
    assert "claimed by another assignee" in failed.detail
    assert "rival-worker" in failed.detail
    assert [row for row in events if row[0] == "claim_lost"]
    not_fulfilled = [row for row in events if row[0] == "next_issue_not_fulfilled"]
    assert not_fulfilled and not_fulfilled[0][3] == 24


def test_fifo_loss_continues_with_the_next_candidate(tmp_path: Path) -> None:
    lifecycle, tracker, store, _events = make_lifecycle(
        tmp_path, {24: issue(24), 25: issue(25)}, fifo=[24, 25]
    )
    tracker.claim_race[24] = ("rival-worker",)

    result = lifecycle.run_once()

    assert result.status.value == "attempted"
    assert tracker.fifo_claims == [24, 25]
    assert tracker.yielded == [24]


# ---------------------------------------------------------------------------
# wording: no single-claimant or human-only assumptions
# ---------------------------------------------------------------------------


def test_is_self_assigned_docstring_no_longer_assumes_a_serialized_boundary() -> None:
    doc = GitHubTracker.is_self_assigned.__doc__ or ""

    assert "only this process claims through the serialized boundary" not in doc


def test_recovery_retry_messages_name_another_assignee_not_a_human(tmp_path: Path) -> None:
    lifecycle, tracker, _store, _events = make_lifecycle(
        tmp_path,
        {
            24: issue(24, assignee_logins=("rival-worker",)),
            25: issue(25),
        },
        fifo=[],
    )

    class _Checkpoint:
        issue_number = 24

    taken = lifecycle._recovery_conflict_reason(None, _Checkpoint())  # type: ignore[arg-type]

    assert taken is not None
    assert "another assignee" in taken
    assert "human changes" not in taken

    class _Bare:
        issue_number = 25

    bare = lifecycle._recovery_conflict_reason(None, _Bare())  # type: ignore[arg-type]

    assert bare is not None
    assert "another assignee" in bare
    assert "human changes" not in bare


def test_context_claim_names_the_only_assignee_rule_and_yielding() -> None:
    context = Path(__file__).resolve().parent.parent / "CONTEXT.md"
    claim_section = context.read_text().split("**Claim**:", 1)[1].split("**", 1)[0]

    assert "only assignee" in claim_section
    assert "yields" in claim_section
