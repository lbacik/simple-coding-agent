"""Randomised claim timing: Workers that start together stop yielding in lockstep.

Post-assign verification waits a fixed settle delay and then yields when
another claimant is present, so two Workers starting at the same instant
claim the same issue in every cycle, both see each other after the settle
window, and both yield -- forever, while their polls stay in phase. Three
always-on randomisations break the symmetry:

- a random back-off after every lost claim, uniform within
  ``[0, CLAIM_LOST_BACKOFF_MAX_SECONDS]``, before the next assign attempt;
- jitter on every idle poll sleep, uniform within ±20% of ``POLL_INTERVAL``,
  with the event detail naming the exact duration slept;
- a random startup delay, uniform within ``[0, POLL_INTERVAL / 2]``, taken
  once before the first claim attempt (startup reconciliation is not
  delayed).

The safety properties are unchanged: yielding still touches no queue labels,
comments, or checkpoints, each loser removes only its own marker, and the
settle delay itself stays fixed.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path

import pytest

import simple_coding_agent.github_tracker as tracker_module
from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.completion import (
    AttemptOutcome,
    CompletionDecision,
    PublicationPath,
)
from simple_coding_agent.github_tracker import (
    CLAIM_LOST_BACKOFF_MAX_SECONDS,
    CLAIM_SETTLE_DELAY_SECONDS,
    Assignment,
    Claim,
    ClaimLostError,
    GitHubIdentity,
    GitHubTracker,
    IssuePage,
    TrackerIssue,
    claim_marker_label,
)
from simple_coding_agent.lifecycle import (
    AgentLifecycle,
    AttemptEvidence,
    LifecycleStatus,
)
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace

REPO = "octo/example"
POLL_INTERVAL = 60


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


class FakeClock:
    """A virtual clock: sleeps advance it instead of waiting."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Advance a virtual clock on tracker sleeps (overrides the suite stub)."""

    made = FakeClock()
    monkeypatch.setattr(tracker_module.time, "sleep", made.sleep)
    return made


# ---------------------------------------------------------------------------
# two-Worker simulation on one shared login
# ---------------------------------------------------------------------------


class PoolTransport:
    """Mutable shared-login transport for the two-Worker timing simulation.

    Both Workers share the viewer login ``agent``; assignees and labels
    mutate in place so interleavings are observable. Mirrors the production
    transport surface the real ``GitHubTracker`` claim path uses.
    """

    def __init__(self, issues: dict[int, TrackerIssue]) -> None:
        self._issues = dict(issues)
        # Frozen verification snapshot plus how many reads it still serves.
        # A simultaneous race stages both claims before either Worker
        # re-reads; in production both re-reads then land before either
        # yield completes (the settle window dwarfs the start offset while a
        # yield costs further API round trips), so both racers see both
        # markers and both yield. The freeze models exactly that overlap:
        # each Worker's verification re-read sees the staged snapshot while
        # the yields that follow read live state.
        self._frozen: TrackerIssue | None = None
        self._frozen_reads = 0
        self.assignments: list[tuple[str, str]] = []
        self.added_labels: list[tuple[str, str]] = []
        self.removed_labels: list[tuple[str, str]] = []
        self.removed_assignees: list[tuple[str, str]] = []
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
        assert repository == REPO
        assert cursor is None
        return IssuePage(issues=tuple(self._issues.values()), next_cursor=None)

    def freeze_verification(self, snapshot: TrackerIssue, reads: int = 2) -> None:
        """Serve ``snapshot`` for the next ``reads`` issue reads."""

        self._frozen = snapshot
        self._frozen_reads = reads

    def unfreeze_verification(self) -> None:
        self._frozen = None
        self._frozen_reads = 0

    def get_issue(self, repository: str, number: int) -> TrackerIssue | None:
        assert repository == REPO
        if self._frozen is not None and self._frozen_reads > 0:
            self._frozen_reads -= 1
            return self._frozen
        return self._issues.get(number)

    def viewer(self) -> GitHubIdentity:
        return GitHubIdentity(id="viewer-id", login="agent")

    def assign_issue(self, issue_id: str, assignee_id: str) -> Assignment:
        self.assignments.append((issue_id, assignee_id))
        for number, current in self._issues.items():
            if current.id == issue_id and "agent" not in current.assignee_logins:
                self._issues[number] = tracker_module.replace(
                    current,
                    assignee_logins=current.assignee_logins + ("agent",),
                )
        return Assignment(issue_id=issue_id, assignee_id=assignee_id)

    def remove_assignee(self, issue_id: str, assignee_id: str) -> None:
        self.removed_assignees.append((issue_id, assignee_id))
        for number, current in self._issues.items():
            if current.id == issue_id:
                self._issues[number] = tracker_module.replace(
                    current,
                    assignee_logins=tuple(
                        login for login in current.assignee_logins if login != "agent"
                    ),
                )

    def add_label(self, issue_id: str, label: str) -> None:
        self.added_labels.append((issue_id, label))
        for number, current in self._issues.items():
            if current.id == issue_id and label not in current.labels:
                self._issues[number] = tracker_module.replace(
                    current, labels=current.labels | {label}
                )

    def remove_label(self, issue_id: str, label: str) -> None:
        self.removed_labels.append((issue_id, label))
        for number, current in self._issues.items():
            if current.id == issue_id:
                self._issues[number] = tracker_module.replace(
                    current, labels=current.labels - {label}
                )

    def list_issue_comments(self, repository: str, issue_number: int) -> tuple:
        return ()


def _pool_worker(
    transport: PoolTransport, worker_id: str, source: object
) -> GitHubTracker:
    return GitHubTracker(
        transport,
        REPO,
        worker_id=worker_id,
        random_source=source,  # type: ignore[arg-type]
        event_log=transport.event_log,
    )


def _recording_source(
    rng: random.Random, draws: list[float],
) -> object:
    def draw() -> float:
        value = rng.random()
        draws.append(value)
        return value

    return draw


def _stage_same_instant(
    transport: PoolTransport, trackers: tuple[GitHubTracker, ...], number: int
) -> None:
    """Assign and mark every Worker at the same virtual instant.

    Staging mirrors the per-candidate preamble of ``claim_next``
    (re-read, assign, add this Worker's marker) for each racer before any
    of them verifies, which is exactly what happens when a whole Pool
    starts together. Verification itself always runs through the real
    ``_verify_claim``, so the settle wait, the re-read, the yield, the
    back-off, and ``ClaimLostError`` are production code, not test logic.
    """

    identity = transport.viewer()
    for tracker in trackers:
        current = transport.get_issue(REPO, number)
        assert current is not None
        transport.assign_issue(current.id, identity.id)
        transport.add_label(current.id, claim_marker_label(tracker.worker_id))


def _verify(
    tracker: GitHubTracker, transport: PoolTransport, snapshot: TrackerIssue
) -> tuple[str, tuple[str, ...]]:
    """Run the real settle-window verification for one staged Worker."""

    try:
        tracker._verify_claim(snapshot, transport.viewer())  # noqa: SLF001 - race seam
    except ClaimLostError as error:
        return ("lost", error.other_assignees)
    return ("held", ())


def _run_lockstep_simulation(
    numbers: list[int], *, seed: int, bound: int = 3
) -> dict[str, object]:
    """Drive two same-instant Workers until one claim sticks.

    Both Workers share one seeded RNG (standing in for two independent
    unseeded sources: the draw order is deterministic and the draws
    differ), one fake transport, and one fake clock. Simultaneous attempts
    stage both claims at the same virtual instant and verify each through
    the real ``_verify_claim``; a desynchronised attempt runs the real
    ``claim_next`` end to end. Returns the per-cycle held/lost record, the
    back-off draws, and the final holdings.
    """

    created = {
        number: issue(number, created_at=datetime(2026, 9, 20, 12, number, tzinfo=UTC))
        for number in numbers
    }
    transport = PoolTransport(created)
    rng = random.Random(seed)
    draws_a: list[float] = []
    draws_b: list[float] = []
    worker_a = _pool_worker(transport, "worker-a", _recording_source(rng, draws_a))
    worker_b = _pool_worker(transport, "worker-b", _recording_source(rng, draws_b))
    workers = {"worker-a": worker_a, "worker-b": worker_b}

    cycles: list[dict[str, tuple[str, tuple[str, ...]]]] = []
    holdings: dict[str, int] = {}
    # Next virtual attempt time per Worker; both start at the same instant.
    next_attempt = {"worker-a": 0.0, "worker-b": 0.0}

    for _ in range(bound):
        first, second = sorted(next_attempt, key=next_attempt.__getitem__)
        if abs(next_attempt[first] - next_attempt[second]) <= CLAIM_SETTLE_DELAY_SECONDS:
            at = max(next_attempt.values())
            head = worker_a.runnable_queue()[0].number
            _stage_same_instant(transport, (worker_a, worker_b), head)
            # Snapshots first: only the verification re-reads inside
            # ``_verify_claim`` may consume the frozen overlap window.
            snap_a = transport.get_issue(REPO, head)
            snap_b = transport.get_issue(REPO, head)
            assert snap_a is not None and snap_b is not None
            # One frozen read per verification: the re-read inside
            # ``_verify_claim`` sees the staged overlap while the yields
            # that follow read live state.
            transport.freeze_verification(snap_a, reads=1)
            outcome_a = _verify(worker_a, transport, snap_a)
            transport.freeze_verification(snap_b, reads=1)
            try:
                outcome_b = _verify(worker_b, transport, snap_b)
            finally:
                transport.unfreeze_verification()
            outcome = {"worker-a": outcome_a, "worker-b": outcome_b}
            cycles.append(outcome)
            # The reported lockstep, reproduced: both re-reads land before
            # either yield completes, so both Workers see both markers and
            # both yield; the issue returns to the queue pristine.
            assert outcome["worker-a"][0] == "lost"
            assert outcome["worker-b"][0] == "lost"
            if outcome["worker-a"][0] == "held" and outcome["worker-b"][0] == "held":
                raise AssertionError("both Workers proceeded past verification")
            if outcome["worker-a"][0] == "held":
                holdings["worker-a"] = head
                break
            if outcome["worker-b"][0] == "held":
                holdings["worker-b"] = head
                break
            # Both yielded (the reported lockstep): each retries after the
            # settle window plus its own recorded back-off draw.
            next_attempt = {
                "worker-a": at + CLAIM_SETTLE_DELAY_SECONDS + draws_a[-1]
                * CLAIM_LOST_BACKOFF_MAX_SECONDS,
                "worker-b": at + CLAIM_SETTLE_DELAY_SECONDS + draws_b[-1]
                * CLAIM_LOST_BACKOFF_MAX_SECONDS,
            }
            continue
        # Desynchronised: the earlier Worker claims alone through the real
        # ``claim_next`` and must hold the issue; the later Worker retries
        # after and takes what is left.
        claim = workers[first].claim_next()
        assert claim is not None
        holdings[first] = claim.issue.number
        cycles.append({first: ("held", ()), second: ("pending", ())})
        rest = workers[second].claim_next()
        if rest is not None:
            holdings[second] = rest.issue.number
            cycles.append({second: ("held", ())})
        else:
            cycles.append({second: ("lost-or-empty", ())})
        break

    return {
        "cycles": cycles,
        "draws_a": draws_a,
        "draws_b": draws_b,
        "holdings": holdings,
    }


def test_two_workers_starting_together_converge_on_one_issue(
    clock: FakeClock,
) -> None:
    assert clock is not None  # the fixture drives tracker sleeps

    sim = _run_lockstep_simulation([1], seed=1, bound=3)

    assert len(sim["cycles"]) <= 3
    assert len(sim["holdings"]) == 1
    # At no point did both Workers proceed past verification on the issue.
    for outcome in sim["cycles"]:  # type: ignore[union-attr]
        held = [name for name, (state, _) in outcome.items() if state == "held"]
        assert len(held) <= 1


def test_two_workers_starting_together_split_two_issues(
    clock: FakeClock,
) -> None:
    assert clock is not None  # the fixture drives tracker sleeps

    sim = _run_lockstep_simulation([1, 2], seed=1, bound=3)

    assert len(sim["cycles"]) <= 3
    assert len(sim["holdings"]) == 2
    assert set(sim["holdings"].values()) == {1, 2}  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# back-off unit bounds
# ---------------------------------------------------------------------------


class BackoffTransport:
    """Two candidates: the first always has a rival, the second is clean."""

    def __init__(self, timeline: list[tuple[str, object]]) -> None:
        self._timeline = timeline
        self._issues = {
            1: issue(1, created_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC)),
            2: issue(2, created_at=datetime(2026, 9, 20, 13, 0, tzinfo=UTC)),
        }
        self._assigned = False
        self.removed_assignees: list[tuple[str, str]] = []
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
        return IssuePage(issues=tuple(self._issues.values()), next_cursor=None)

    def get_issue(self, repository: str, number: int) -> TrackerIssue | None:
        base = self._issues.get(number)
        if base is None:
            return None
        if base.id == "issue-1" and self._assigned:
            return tracker_module.replace(
                base, assignee_logins=("agent", "rival-worker")
            )
        if "agent" in base.assignee_logins:
            return base
        return base

    def viewer(self) -> GitHubIdentity:
        return GitHubIdentity(id="viewer-id", login="agent")

    def assign_issue(self, issue_id: str, assignee_id: str) -> Assignment:
        self._timeline.append(("assign", issue_id))
        self._assigned = self._assigned or issue_id == "issue-1"
        if issue_id == "issue-1":
            self._issues[1] = tracker_module.replace(
                self._issues[1], assignee_logins=("agent",)
            )
        else:
            self._issues[2] = tracker_module.replace(
                self._issues[2], assignee_logins=("agent",)
            )
        return Assignment(issue_id=issue_id, assignee_id=assignee_id)

    def remove_assignee(self, issue_id: str, assignee_id: str) -> None:
        self.removed_assignees.append((issue_id, assignee_id))
        for number, current in self._issues.items():
            if current.id == issue_id:
                self._issues[number] = tracker_module.replace(
                    current, assignee_logins=()
                )

    def remove_label(self, issue_id: str, label: str) -> None:
        raise AssertionError("a lost claim must not touch labels")

    def add_label(self, issue_id: str, label: str) -> None:
        raise AssertionError("no Worker id is set, so no marker is added")

    def list_issue_comments(self, repository: str, issue_number: int) -> tuple:
        return ()


def test_post_yield_backoff_within_bound_and_before_next_assign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeline: list[tuple[str, object]] = []
    monkeypatch.setattr(
        tracker_module.time, "sleep", lambda seconds: timeline.append(("sleep", seconds))
    )
    transport = BackoffTransport(timeline)
    tracker = GitHubTracker(
        transport,
        REPO,
        random_source=random.Random(11).random,
        event_log=transport.event_log,
    )

    claim = tracker.claim_next()

    assert claim is not None
    assert claim.issue.number == 2
    kinds = [kind for kind, _ in timeline]
    assert kinds == ["assign", "sleep", "sleep", "assign", "sleep"]
    assert timeline[0] == ("assign", "issue-1")
    assert timeline[1] == ("sleep", CLAIM_SETTLE_DELAY_SECONDS)
    backoff = timeline[2][1]
    assert isinstance(backoff, float)
    assert 0 <= backoff <= CLAIM_LOST_BACKOFF_MAX_SECONDS
    assert timeline[3] == ("assign", "issue-2")
    assert timeline[4] == ("sleep", CLAIM_SETTLE_DELAY_SECONDS)
    # Yielding still removes only our own assignment and emits ``claim_lost``.
    assert transport.removed_assignees == [("issue-1", "viewer-id")]
    lost = [row for row in transport.events if row[0] == "claim_lost"]
    assert len(lost) == 1
    assert lost[0][3] == 1


def test_successful_first_claim_sleeps_no_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []
    monkeypatch.setattr(
        tracker_module.time, "sleep", lambda seconds: calls.append(seconds)
    )
    transport = BackoffTransport([])
    tracker = GitHubTracker(
        transport,
        REPO,
        random_source=random.Random(11).random,
        event_log=transport.event_log,
    )

    claim = tracker.claim_verified(transport._issues[2])

    assert claim.issue.number == 2
    assert calls == [CLAIM_SETTLE_DELAY_SECONDS]


# ---------------------------------------------------------------------------
# poll jitter and startup delay
# ---------------------------------------------------------------------------


def _evidence() -> AttemptEvidence:
    return AttemptEvidence(
        decision=CompletionDecision(
            AttemptOutcome.COMPLETE, True, PublicationPath.COMPLETE, ("ready",)
        ),
        check_command="pytest",
        check_exit_code=0,
        review_cycles=1,
        review_findings="all clear",
        details="Implemented.",
    )


def _lifecycle(
    tmp_path: Path,
    tracker: FakeTracker,
    *,
    poll_interval: int = POLL_INTERVAL,
    seed: int,
    sleeps: list[float],
    events: list[tuple],
) -> AgentLifecycle:
    return AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=lambda received_claim, profile, prepared: _evidence(),
        poll_interval=poll_interval,
        sleeper=sleeps.append,
        random_source=random.Random(seed).random,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level, issue_number)
        ),
    )


def test_idle_poll_sleep_is_jittered_and_reported(tmp_path: Path) -> None:
    sleeps: list[float] = []
    events: list[tuple] = []
    lifecycle = _lifecycle(
        tmp_path, FakeTracker(), seed=5, sleeps=sleeps, events=events
    )

    assert lifecycle.run_once().status is LifecycleStatus.IDLE
    assert len(sleeps) == 2
    startup, idle = sleeps
    assert 0 <= startup <= POLL_INTERVAL / 2
    assert 0.8 * POLL_INTERVAL <= idle <= 1.2 * POLL_INTERVAL
    announced = [detail for event, detail, _, _ in events if event == "no_eligible_issue_found"]
    assert announced == [f"no eligible issue available; sleeping {idle}s"]

    assert lifecycle.run_once().status is LifecycleStatus.IDLE
    assert len(sleeps) == 3
    assert 0.8 * POLL_INTERVAL <= sleeps[2] <= 1.2 * POLL_INTERVAL


def test_startup_delay_happens_once_before_the_first_claim(
    tmp_path: Path,
) -> None:
    sleeps: list[float] = []
    events: list[tuple] = []
    lifecycle = _lifecycle(
        tmp_path,
        FakeTracker(next_claim=Claim(issue(24), Assignment("issue-24", "agent-id"))),
        seed=5,
        sleeps=sleeps,
        events=events,
    )

    assert lifecycle.run_once().status is LifecycleStatus.ATTEMPTED
    assert len(sleeps) == 1
    assert 0 <= sleeps[0] <= POLL_INTERVAL / 2

    assert lifecycle.run_once().status is LifecycleStatus.IDLE
    assert len(sleeps) == 2
    assert 0.8 * POLL_INTERVAL <= sleeps[1] <= 1.2 * POLL_INTERVAL


def test_startup_reconciliation_is_not_delayed(tmp_path: Path) -> None:
    state = AttemptStateStore(tmp_path)
    state.start(issue_number=24, branch="agent/issue-24")
    sleeps: list[float] = []
    events: list[tuple] = []
    lifecycle = AgentLifecycle(
        tracker=FakeTracker({24: issue(24)}),
        attempt_state=state,
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=lambda received_claim, profile, prepared: _evidence(),
        poll_interval=POLL_INTERVAL,
        sleeper=sleeps.append,
        random_source=random.Random(5).random,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level, issue_number)
        ),
    )

    assert lifecycle.run_once().status is LifecycleStatus.ATTEMPTED
    assert sleeps == []
