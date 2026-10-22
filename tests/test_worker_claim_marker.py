"""Shared-login pool: per-Worker claim markers (``WORKER_ID``).

With one GitHub login shared by several Workers the assignment is
idempotent, so the settle-window verification alone cannot tell pool
Workers apart. With ``WORKER_ID`` set a claimant is a ``(login, worker
id)`` pair: after assigning, the Worker also adds its
``claimed-by:<worker-id>`` marker label, yields when it sees another
Work's marker, and releases its marker before the shared assignment
(which goes only when no marker remains).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import simple_coding_agent.github_tracker as tracker_module
from simple_coding_agent.config import ConfigurationError, load_runtime_config
from simple_coding_agent.github_tracker import (
    CLAIM_SETTLE_DELAY_SECONDS,
    Assignment,
    ClaimLostError,
    GitHubIdentity,
    GitHubTracker,
    GitHubTrackerError,
    IssuePage,
    TrackerIssue,
    claim_marker_label,
    has_claim_marker,
)

if TYPE_CHECKING:
    from simple_coding_agent.lifecycle import AgentLifecycle

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


def _operator_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    return {
        "GITHUB_TOKEN": "github-secret",
        "META_API_KEY": "meta-secret",
        "TARGET_REPO": "octo/example",
        "DATA_DIR": str(tmp_path),
        **overrides,
    }


class SharedTransport:
    """A mutable single-login transport shared by pool Worker trackers.

    Both Workers see the viewer login ``agent`` (the shared account); labels
    and assignees mutate in place so interleavings are observable.
    """

    def __init__(self, initial: TrackerIssue) -> None:
        self.current = initial
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
        assert repository == "octo/example"
        assert cursor is None
        return IssuePage(issues=(self.current,), next_cursor=None)

    def get_issue(self, repository: str, number: int) -> TrackerIssue | None:
        assert repository == "octo/example"
        return self.current if number == self.current.number else None

    def viewer(self) -> GitHubIdentity:
        return GitHubIdentity(id="viewer-id", login="agent")

    def assign_issue(self, issue_id: str, assignee_id: str) -> Assignment:
        self.assignments.append((issue_id, assignee_id))
        if "agent" not in self.current.assignee_logins:
            self.current = tracker_module.replace(
                self.current,
                assignee_logins=self.current.assignee_logins + ("agent",),
            )
        return Assignment(issue_id=issue_id, assignee_id=assignee_id)

    def remove_assignee(self, issue_id: str, assignee_id: str) -> None:
        self.removed_assignees.append((issue_id, assignee_id))
        self.current = tracker_module.replace(
            self.current,
            assignee_logins=tuple(
                login for login in self.current.assignee_logins if login != "agent"
            ),
        )

    def add_label(self, issue_id: str, label: str) -> None:
        self.added_labels.append((issue_id, label))
        if label not in self.current.labels:
            self.current = tracker_module.replace(
                self.current, labels=self.current.labels | {label}
            )

    def remove_label(self, issue_id: str, label: str) -> None:
        self.removed_labels.append((issue_id, label))
        self.current = tracker_module.replace(
            self.current, labels=self.current.labels - {label}
        )

    def list_issue_comments(self, repository: str, issue_number: int) -> tuple:
        return ()


class LabelRaceTransport(SharedTransport):
    """A transport where a rival marker appears during the settle window.

    Reads before this Worker assigns return the clean snapshot; reads after
    return it with this Worker (``agent``) plus the rival's
    ``claimed-by:*`` marker present, simulating a second pool Worker
    assigning and marking during the settle delay.
    """

    def __init__(self, initial: TrackerIssue, rival_marker: str) -> None:
        super().__init__(initial)
        self._rival_marker = rival_marker
        self._assigned = False

    def assign_issue(self, issue_id: str, assignee_id: str) -> Assignment:
        self._assigned = True
        return super().assign_issue(issue_id, assignee_id)

    def get_issue(self, repository: str, number: int) -> TrackerIssue | None:
        current = super().get_issue(repository, number)
        if current is None or not self._assigned:
            return current
        if self._rival_marker not in current.labels:
            current = tracker_module.replace(
                self.current,
                labels=self.current.labels | {self._rival_marker},
            )
            self.current = current
        return current


def worker_tracker(
    transport: SharedTransport, worker_id: str
) -> GitHubTracker:
    return GitHubTracker(
        transport, "octo/example", worker_id=worker_id, event_log=transport.event_log
    )


# ---------------------------------------------------------------------------
# WORKER_ID setting
# ---------------------------------------------------------------------------


def test_worker_id_defaults_to_empty(tmp_path: Path) -> None:
    assert load_runtime_config(_operator_env(tmp_path)).worker_id == ""


@pytest.mark.parametrize("worker_id", ["a", "worker-1", "x" * 32, "0-9-a-z"])
def test_worker_id_accepts_lowercase_digits_and_dashes(
    tmp_path: Path, worker_id: str
) -> None:
    assert (
        load_runtime_config(_operator_env(tmp_path, WORKER_ID=worker_id)).worker_id
        == worker_id
    )


@pytest.mark.parametrize(
    "worker_id",
    ["Worker-1", "worker_1", "worker 1", "worker.1", "x" * 33, "white space", "-UP-"],
)
def test_worker_id_rejects_anything_but_lowercase_digits_and_dashes(
    tmp_path: Path, worker_id: str
) -> None:
    with pytest.raises(ConfigurationError) as error:
        load_runtime_config(_operator_env(tmp_path, WORKER_ID=worker_id))

    assert "WORKER_ID" in str(error.value)
    assert "github-secret" not in str(error.value)
    assert "meta-secret" not in str(error.value)


def test_blank_worker_id_means_no_pool_identity(tmp_path: Path) -> None:
    assert load_runtime_config(_operator_env(tmp_path, WORKER_ID="  ")).worker_id == ""


def test_tracker_rejects_an_invalid_worker_id() -> None:
    transport = SharedTransport(issue(1))

    with pytest.raises(GitHubTrackerError):
        GitHubTracker(transport, "octo/example", worker_id="Worker-1")


# ---------------------------------------------------------------------------
# without WORKER_ID nothing changes
# ---------------------------------------------------------------------------


def test_without_worker_id_no_marker_is_ever_written() -> None:
    transport = SharedTransport(issue(1))
    tracker = GitHubTracker(
        transport, "octo/example", event_log=transport.event_log
    )

    claim = tracker.claim_next()

    assert claim is not None
    assert transport.added_labels == []
    assert transport.removed_labels == []
    assert not any(
        label.startswith("claimed-by:") for label in transport.current.labels
    )


def test_without_worker_id_release_removes_the_assignment_unconditionally() -> None:
    transport = SharedTransport(
        issue(
            24,
            labels=frozenset({"ready-for-agent", "claimed-by:other"}),
            assignee_logins=("agent",),
        )
    )
    tracker = GitHubTracker(
        transport, "octo/example", event_log=transport.event_log
    )

    tracker.release_attempt(24, "ready-for-agent", "viewer-id")

    assert transport.removed_assignees == [("issue-24", "viewer-id")]
    assert ("issue-24", "claimed-by:other") not in transport.removed_labels


# ---------------------------------------------------------------------------
# claim with WORKER_ID: at most one keeper
# ---------------------------------------------------------------------------


def test_winner_adds_its_marker_and_clears_stale_labels() -> None:
    transport = SharedTransport(
        issue(1, labels=frozenset({"ready-for-agent", "round-finished"}))
    )
    tracker = worker_tracker(transport, "worker-a")

    claim = tracker.claim_next()

    assert claim is not None
    assert ("issue-1", "claimed-by:worker-a") in transport.added_labels
    assert ("issue-1", "round-finished") in transport.removed_labels
    assert transport.removed_assignees == []
    assert "claimed-by:worker-a" in transport.current.labels


def test_second_worker_on_one_login_never_claims_over_the_winner() -> None:
    transport = SharedTransport(issue(1))
    first = worker_tracker(transport, "worker-a")
    second = worker_tracker(transport, "worker-b")

    assert first.claim_next() is not None
    assert second.claim_next() is None

    # The loser-to-be writes nothing: no second marker, no assignment churn,
    # and the winner's claim (shared assignment plus its marker) is intact.
    assert ("issue-1", "claimed-by:worker-b") not in transport.added_labels
    assert transport.removed_assignees == []
    assert transport.current.assignee_logins == ("agent",)
    assert transport.current.labels == frozenset(
        {"ready-for-agent", "claimed-by:worker-a"}
    )


def test_loser_yields_removes_only_its_marker_and_leaves_the_assignment() -> None:
    transport = LabelRaceTransport(issue(1), "claimed-by:worker-winner")
    tracker = worker_tracker(transport, "worker-loser")

    assert tracker.claim_next() is None

    assert ("issue-1", "claimed-by:worker-loser") in transport.added_labels
    assert ("issue-1", "claimed-by:worker-loser") in transport.removed_labels
    # The winner's marker keeps the shared assignment: no assignee removal.
    assert transport.removed_assignees == []
    assert transport.current.assignee_logins == ("agent",)
    assert "claimed-by:worker-winner" in transport.current.labels
    assert "claimed-by:worker-loser" not in transport.current.labels
    lost = [row for row in transport.events if row[0] == "claim_lost"]
    assert len(lost) == 1
    assert lost[0][3] == 1


def test_priority_claim_yields_on_another_workers_marker() -> None:
    transport = LabelRaceTransport(issue(1), "claimed-by:worker-winner")
    tracker = worker_tracker(transport, "worker-loser")

    with pytest.raises(ClaimLostError):
        tracker.claim_verified(transport.current)

    assert ("issue-1", "claimed-by:worker-loser") in transport.added_labels
    assert ("issue-1", "claimed-by:worker-loser") in transport.removed_labels
    assert transport.removed_labels.count(("issue-1", "round-finished")) == 0
    assert transport.removed_assignees == []


def test_last_yielder_out_removes_the_now_unmarked_assignment() -> None:
    # Both racers yielded: the first left the assignment while the winner's
    # marker existed; once every marker is gone the loser path removes the
    # shared assignment too, so the issue returns to the queue.
    fresh = issue(
        1, labels=frozenset({"ready-for-agent"}), assignee_logins=("agent",)
    )
    transport = SharedTransport(fresh)
    tracker = worker_tracker(transport, "worker-loser")
    # The loser's own marker is already gone (removed just before this
    # re-read); no foreign marker remains either.
    tracker._yield_losing_claim(fresh, GitHubIdentity(id="viewer-id", login="agent"))

    assert transport.removed_assignees == [("issue-1", "viewer-id")]


# ---------------------------------------------------------------------------
# release order with WORKER_ID
# ---------------------------------------------------------------------------


def test_release_attempt_removes_own_marker_but_keeps_assignment_for_the_winner() -> None:
    transport = SharedTransport(
        issue(
            24,
            labels=frozenset(
                {"ready-for-agent", "claimed-by:worker-a", "claimed-by:worker-b"}
            ),
            assignee_logins=("agent",),
        )
    )
    tracker = worker_tracker(transport, "worker-a")

    tracker.release_attempt(24, "ready-for-agent", "viewer-id")

    assert ("issue-24", "ready-for-agent") in transport.removed_labels
    assert ("issue-24", "claimed-by:worker-a") in transport.removed_labels
    assert ("issue-24", "claimed-by:worker-b") not in transport.removed_labels
    assert transport.removed_assignees == []
    assert transport.current.labels == frozenset({"claimed-by:worker-b"})
    assert transport.current.assignee_logins == ("agent",)


def test_release_attempt_removes_the_assignment_when_no_marker_remains() -> None:
    transport = SharedTransport(
        issue(
            24,
            labels=frozenset({"ready-for-agent", "claimed-by:worker-a"}),
            assignee_logins=("agent",),
        )
    )
    tracker = worker_tracker(transport, "worker-a")

    tracker.release_attempt(24, "ready-for-agent", "viewer-id")

    assert ("issue-24", "claimed-by:worker-a") in transport.removed_labels
    assert transport.removed_assignees == [("issue-24", "viewer-id")]
    assert transport.current.labels == frozenset()
    assert transport.current.assignee_logins == ()


def test_release_handoff_keeps_assignment_while_another_marker_remains() -> None:
    transport = SharedTransport(
        issue(
            24,
            labels=frozenset(
                {"ready-for-agent", "claimed-by:worker-a", "claimed-by:worker-b"}
            ),
            assignee_logins=("agent",),
        )
    )
    tracker = worker_tracker(transport, "worker-a")

    tracker.release_handoff(24, "viewer-id")

    assert ("issue-24", "ready-for-agent") in transport.removed_labels
    assert ("issue-24", "round-finished") in transport.added_labels
    assert ("issue-24", "claimed-by:worker-a") in transport.removed_labels
    assert transport.removed_assignees == []
    assert "claimed-by:worker-b" in transport.current.labels
    assert transport.current.assignee_logins == ("agent",)


def test_release_handoff_removes_the_assignment_when_no_marker_remains() -> None:
    transport = SharedTransport(
        issue(
            24,
            labels=frozenset({"ready-for-agent", "claimed-by:worker-a"}),
            assignee_logins=("agent",),
        )
    )
    tracker = worker_tracker(transport, "worker-a")

    tracker.release_handoff(24, "viewer-id")

    assert transport.removed_assignees == [("issue-24", "viewer-id")]
    assert transport.current.labels == frozenset({"round-finished"})
    assert transport.current.assignee_logins == ()


# ---------------------------------------------------------------------------
# eligibility and self with WORKER_ID
# ---------------------------------------------------------------------------


def test_any_marker_makes_an_issue_ineligible() -> None:
    marked = issue(
        7,
        labels=frozenset({"ready-for-agent", "claimed-by:worker-a"}),
    )

    assert marked.is_eligible is False
    assert marked.meets_base_queue_rules is False
    assert has_claim_marker(marked.labels) is True
    assert claim_marker_label("worker-a") == "claimed-by:worker-a"


def test_marked_issues_never_enter_the_runnable_queue() -> None:
    transport = SharedTransport(
        issue(
            1,
            labels=frozenset({"ready-for-agent", "claimed-by:worker-a"}),
        )
    )
    tracker = worker_tracker(transport, "worker-b")

    assert tracker.runnable_queue() == ()


def test_is_self_assigned_requires_the_own_marker() -> None:
    transport = SharedTransport(issue(1))
    tracker = worker_tracker(transport, "worker-a")
    own = issue(
        1,
        labels=frozenset({"ready-for-agent", "claimed-by:worker-a"}),
        assignee_logins=("agent",),
    )
    foreign = issue(
        2,
        labels=frozenset({"ready-for-agent", "claimed-by:worker-b"}),
        assignee_logins=("agent",),
    )

    assert tracker.is_self_assigned(own) is True
    assert tracker.is_self_assigned(foreign) is False


def test_recover_claim_requires_the_own_marker() -> None:
    transport = SharedTransport(issue(1))
    tracker = worker_tracker(transport, "worker-a")
    transport.current = issue(
        1,
        labels=frozenset({"ready-for-agent", "claimed-by:worker-a"}),
        assignee_logins=("agent",),
    )

    assert tracker.recover_claim(1) is not None

    transport.current = issue(
        1,
        labels=frozenset({"ready-for-agent", "claimed-by:worker-b"}),
        assignee_logins=("agent",),
    )

    assert tracker.recover_claim(1) is None


def test_recover_claim_survives_a_restart_between_marker_and_assignment_removal() -> None:
    # This Worker's own partial release removed its marker but crashed
    # before the conditional assignment removal: no marker remains, so
    # finishing the release on restart is safe and must stay possible.
    transport = SharedTransport(issue(1))
    tracker = worker_tracker(transport, "worker-a")
    transport.current = issue(
        1, labels=frozenset({"ready-for-agent"}), assignee_logins=("agent",)
    )

    assert tracker.recover_claim(1) is not None


def test_recover_claim_without_an_assignment_is_nothing_to_recover() -> None:
    transport = SharedTransport(issue(1))
    tracker = worker_tracker(transport, "worker-a")
    transport.current = issue(
        1, labels=frozenset({"ready-for-agent", "claimed-by:worker-a"})
    )

    assert tracker.recover_claim(1) is None


# ---------------------------------------------------------------------------
# retry and priority handling with WORKER_ID
# ---------------------------------------------------------------------------


def _lifecycle(tmp_path: Path, tracker: FakeTracker) -> AgentLifecycle:
    from simple_coding_agent.attempt_state import AttemptStateStore
    from simple_coding_agent.lifecycle import AgentLifecycle

    return AgentLifecycle(
        tracker=tracker,  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=lambda received_claim, profile, prepared: None,  # type: ignore[return-value]
        event_log=lambda event, detail="", level="INFO", issue_number=None: None,
    )


def test_retry_conflict_names_another_workers_marker_on_the_shared_login(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    tracker = FakeTracker(
        issues={
            24: issue(
                24,
                labels=frozenset({"ready-for-agent", "claimed-by:worker-b"}),
                assignee_logins=("agent",),
            )
        },
        worker_id="worker-a",
    )
    lifecycle = _lifecycle(tmp_path, tracker)

    reason = lifecycle._recovery_conflict_reason(
        SimpleNamespace(issue_number=24),  # type: ignore[arg-type]
        SimpleNamespace(issue_number=24),  # type: ignore[arg-type]
    )

    assert reason is not None
    assert "another assignee" in reason
    assert "claimed-by:worker-b" in reason


def test_retry_conflict_without_a_marker_reads_as_before(tmp_path: Path) -> None:
    from types import SimpleNamespace

    tracker = FakeTracker(
        issues={24: issue(24, assignee_logins=("rival-worker",))},
        worker_id="worker-a",
    )
    lifecycle = _lifecycle(tmp_path, tracker)

    reason = lifecycle._recovery_conflict_reason(
        SimpleNamespace(issue_number=24),  # type: ignore[arg-type]
        SimpleNamespace(issue_number=24),  # type: ignore[arg-type]
    )

    assert reason is not None
    assert "another assignee" in reason
    assert "claimed-by" not in reason


def test_lifecycle_self_check_requires_the_own_marker(tmp_path: Path) -> None:
    tracker = FakeTracker(issues={}, worker_id="worker-a")
    lifecycle = _lifecycle(tmp_path, tracker)
    own = issue(
        24,
        labels=frozenset({"ready-for-agent", "claimed-by:worker-a"}),
        assignee_logins=("agent",),
    )
    foreign = issue(
        25,
        labels=frozenset({"ready-for-agent", "claimed-by:worker-b"}),
        assignee_logins=("agent",),
    )

    assert lifecycle._is_self_assignment(own) is True
    assert lifecycle._is_self_assignment(foreign) is False


def test_settle_delay_still_applies_with_a_worker_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []
    monkeypatch.setattr(
        tracker_module.time, "sleep", lambda seconds: calls.append(seconds)
    )
    transport = SharedTransport(issue(1))
    tracker = worker_tracker(transport, "worker-a")

    assert tracker.claim_next() is not None

    assert calls == [CLAIM_SETTLE_DELAY_SECONDS]


def test_priority_target_with_only_a_foreign_marker_falls_back_to_fifo(
    tmp_path: Path,
) -> None:
    """A priority target the winner claimed first ends ``not fulfilled``.

    Worker B wins the race after the priority is accepted (the shared-login
    assignment plus only B's marker): the target revalidates as claimed by
    another assignee, so the priority ends ``not fulfilled`` and the same
    cycle falls back to FIFO.
    """
    from simple_coding_agent.attempt_state import AttemptStateStore
    from simple_coding_agent.completion import (
        AttemptOutcome,
        CompletionDecision,
        PublicationPath,
    )
    from simple_coding_agent.control import CommandAcknowledgement, ControlStore
    from simple_coding_agent.lifecycle import AgentLifecycle, AttemptEvidence

    events: list[tuple] = []

    def event_log(
        event: str,
        detail: str = "",
        level: str = "INFO",
        issue_number: int | None = None,
    ) -> None:
        events.append((event, detail, level, issue_number))

    tracker = FakeTracker(
        issues={24: issue(24), 25: issue(25)},
        fifo=[25],
        worker_id="worker-a",
        event_log=event_log,
    )
    store = ControlStore(tmp_path)
    lifecycle = AgentLifecycle(
        tracker=tracker,  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=lambda received_claim, profile, prepared: AttemptEvidence(
            decision=CompletionDecision(
                AttemptOutcome.COMPLETE, True, PublicationPath.COMPLETE, ("ready",)
            ),
            check_command="pytest",
            check_exit_code=0,
            review_cycles=1,
            review_findings="all clear",
            details="Implemented.",
        ),
        control_store=store,
        sleeper=lambda seconds: None,
        event_log=event_log,
    )
    lifecycle.submit_next_issue("req-next", 24)
    # Worker B wins after acceptance: shared-login assignment plus only B's
    # marker, so the target is this login assigned to another claimant.
    tracker.issues[24] = issue(
        24,
        labels=frozenset({"ready-for-agent", "claimed-by:worker-b"}),
        assignee_logins=("agent",),
    )

    result = lifecycle.run_once()

    assert result.status.value == "attempted"
    assert tracker.fifo_claims == [25]
    failed = store.get_command("req-next")
    assert failed is not None
    assert failed.acknowledgement is CommandAcknowledgement.NOT_FULFILLED
    assert "already assigned to agent" in failed.detail
    assert "claimed by another worker (claimed-by:worker-b)" in failed.detail
    not_fulfilled = [row for row in events if row[0] == "next_issue_not_fulfilled"]
    assert not_fulfilled and not_fulfilled[0][3] == 24
