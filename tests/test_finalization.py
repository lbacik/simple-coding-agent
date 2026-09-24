"""Restart-safe finalization: stable identity, durable boundary, preserved work."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.completion import AttemptOutcome, CompletionDecision, PublicationPath
from simple_coding_agent.finalization import AttemptCompletionStore
from simple_coding_agent.git_workspace import DirtyWorkspaceError
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.lifecycle import AgentLifecycle, LifecycleStatus
from simple_coding_agent.operating import ConsecutiveErrorStore


def test_completion_store_records_each_attempt_once(tmp_path: Path) -> None:
    store = AttemptCompletionStore(tmp_path)

    assert store.record(attempt_id="2026-09-20T13:00:00Z", issue_number=24, branch="agent/issue-24", outcome=AttemptOutcome.COMPLETE) is True
    assert store.record(attempt_id="2026-09-20T13:00:00Z", issue_number=24, branch="agent/issue-24", outcome=AttemptOutcome.COMPLETE) is False

    assert store.is_finalized("2026-09-20T13:00:00Z") is True
    assert store.is_finalized("2026-09-20T14:00:00Z") is False
    recorded = store.read_all()["2026-09-20T13:00:00Z"]
    assert recorded.issue_number == 24
    assert recorded.outcome is AttemptOutcome.COMPLETE


def test_completion_store_survives_restart(tmp_path: Path) -> None:
    AttemptCompletionStore(tmp_path).record(
        attempt_id="attempt-a", issue_number=24, branch="agent/issue-24", outcome=AttemptOutcome.INCOMPLETE
    )

    restarted = AttemptCompletionStore(tmp_path)

    assert restarted.is_finalized("attempt-a") is True
    assert restarted.record(attempt_id="attempt-a", issue_number=24, branch="agent/issue-24", outcome=AttemptOutcome.INCOMPLETE) is False


def test_completion_store_rejects_malformed_record(tmp_path: Path) -> None:
    path = tmp_path / "state" / "completed_attempts.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not JSON")

    from simple_coding_agent.finalization import CompletionStoreError

    with pytest.raises(CompletionStoreError, match="malformed"):
        AttemptCompletionStore(tmp_path).is_finalized("attempt-a")


def test_finalized_attempt_is_recorded_once_before_a_new_claim(tmp_path: Path) -> None:
    """A fully finalized attempt lands in the ledger exactly once with its accounting."""
    claim = Claim(_issue(24), Assignment("issue-24", "agent-id"))
    tracker = _FakeTracker(claim)
    workspace = _FakeWorkspace()
    publisher = _FakePublisher()
    completions = AttemptCompletionStore(tmp_path)
    errors = ConsecutiveErrorStore(tmp_path)
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: _profile(),
        publisher=publisher,
        attempt_runner=_evidence_runner(AttemptOutcome.INCOMPLETE),
        error_store=errors,
        completion_store=completions,
    )

    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.ATTEMPTED
    # The fresh claim created a checkpoint that finalization must remove
    # after recording the attempt exactly once.
    assert completions.read_all()
    attempt_id = next(iter(completions.read_all()))
    assert completions.is_finalized(attempt_id) is True
    assert AttemptStateStore(tmp_path).read() is None
    assert errors.read().last_attempt_id == attempt_id


def test_crash_between_finalization_and_checkpoint_removal_does_not_republish(tmp_path: Path) -> None:
    """Ledger present + checkpoint present means: only remove the checkpoint."""
    state = _state_at(tmp_path, AttemptPhase.PUBLISHING)
    checkpoint = state.read()
    assert checkpoint is not None
    completions = AttemptCompletionStore(tmp_path)
    completions.record(
        attempt_id=checkpoint.started_at, issue_number=24, branch="agent/issue-24", outcome=AttemptOutcome.COMPLETE
    )
    tracker = _FakeTracker(None)
    publisher = _FakePublisher()
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=_FakeWorkspace(commits=("completed",)),
        profile_loader=lambda _: _profile(),
        publisher=publisher,
        completion_store=completions,
    )

    lifecycle.run_once()

    assert publisher.requests == []
    assert tracker.cleanup == []
    assert state.read() is None
    # Still exactly one ledger entry: no duplicate accounting.
    assert len(completions.read_all()) == 1


def test_release_failure_holds_the_attempt_without_accounting(tmp_path: Path) -> None:
    state = _state_at(tmp_path, AttemptPhase.PUSHING)
    checkpoint = state.read()
    assert checkpoint is not None
    completions = AttemptCompletionStore(tmp_path)
    errors = ConsecutiveErrorStore(tmp_path)
    lifecycle = AgentLifecycle(
        tracker=_FailingReleaseTracker(None),
        attempt_state=state,
        workspace=_FakeWorkspace(commits=("completed",)),
        profile_loader=lambda _: _profile(),
        publisher=_FakePublisher(),
        error_store=errors,
        completion_store=completions,
    )

    with pytest.raises(OSError, match="GitHub unavailable"):
        lifecycle.run_once()

    assert state.read() is not None
    assert completions.read_all() == {}
    assert errors.read().last_attempt_id is None


def test_comment_failure_holds_without_ledger_then_recovers_once(tmp_path: Path) -> None:
    state = _state_at(tmp_path, AttemptPhase.PUSHING)
    completions = AttemptCompletionStore(tmp_path)
    errors = ConsecutiveErrorStore(tmp_path)
    workspace = _FakeWorkspace(commits=("completed",))
    first = AgentLifecycle(
        tracker=_FakeTracker(None),
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: _profile(),
        publisher=_CommentFailingPublisher(),
        error_store=errors,
        completion_store=completions,
    )

    first.run_once()

    assert state.read() is not None
    assert completions.read_all() == {}
    assert errors.read().last_attempt_id is None

    tracker = _FakeTracker(None)
    second = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: _profile(),
        publisher=_FakePublisher(),
        error_store=errors,
        completion_store=completions,
    )

    second.run_once()

    assert tracker.cleanup == [(24, "ready-for-agent", "agent-id")]
    assert state.read() is None
    assert len(completions.read_all()) == 1
    assert errors.read().last_attempt_id is not None


def test_dirty_worktree_during_startup_holds_without_publishing(tmp_path: Path) -> None:
    """Unexplained dirty work keeps the checkpoint, branch, and tree for inspection."""
    state = _state_at(tmp_path, AttemptPhase.MODEL_RUNNING)
    completions = AttemptCompletionStore(tmp_path)
    publisher = _FakePublisher()
    tracker = _FakeTracker(None)
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=_DirtyWorkspace(),
        profile_loader=lambda _: _profile(),
        publisher=publisher,
        completion_store=completions,
    )

    with pytest.raises(SystemExit):
        lifecycle.run_once()

    assert state.read() is not None
    assert publisher.requests == []
    assert tracker.cleanup == []
    assert completions.read_all() == {}


def test_malformed_checkpoint_holds_intake_without_claiming(tmp_path: Path) -> None:
    path = tmp_path / "state" / "attempt.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not JSON")
    tracker = _FakeTracker(Claim(_issue(25), Assignment("issue-25", "agent-id")))
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=_FakeWorkspace(),
        profile_loader=lambda _: _profile(),
        publisher=_FakePublisher(),
        completion_store=AttemptCompletionStore(tmp_path),
    )

    with pytest.raises(SystemExit):
        lifecycle.run_once()

    assert tracker.claimed == []


def test_dirty_tree_without_checkpoint_holds_intake(tmp_path: Path) -> None:
    tracker = _FakeTracker(Claim(_issue(25), Assignment("issue-25", "agent-id")))
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=_DirtyWorkspace(),
        profile_loader=lambda _: _profile(),
        publisher=_FakePublisher(),
        completion_store=AttemptCompletionStore(tmp_path),
    )

    with pytest.raises(SystemExit):
        lifecycle.run_once()

    assert tracker.claimed == []
    assert AttemptStateStore(tmp_path).read() is None


# --- Fakes ---


def _evidence_runner(outcome: AttemptOutcome):
    from simple_coding_agent.lifecycle import AttemptEvidence

    def run(claim: Claim, profile: object, prepared: object) -> AttemptEvidence:
        if outcome is AttemptOutcome.INCOMPLETE:
            decision = CompletionDecision(outcome, False, PublicationPath.PARTIAL, ("check failed",))
        else:
            decision = CompletionDecision(outcome, False, PublicationPath.NONE, ("done",))
        return AttemptEvidence(decision, "pytest", 1, 0, "all clear", "details")

    return run


class _FakeTracker:
    def __init__(self, next_claim: Claim | None) -> None:
        self.next_claim = next_claim
        self.cleanup: list[tuple[int, str, str]] = []
        self.claimed: list[int] = []

    def claim_next(self) -> Claim | None:
        claim, self.next_claim = self.next_claim, None
        if claim is not None:
            self.claimed.append(claim.issue.number)
        return claim

    def recover_claim(self, issue_number: int) -> Claim | None:
        return Claim(_issue(issue_number), Assignment(f"issue-{issue_number}", "agent-id"))

    def release_attempt(self, issue_number: int, label: str, assignee_id: str) -> None:
        self.cleanup.append((issue_number, label, assignee_id))

    def release_handoff(self, issue_number: int, assignee_id: str) -> None:
        self.cleanup.append((issue_number, "round-finished", assignee_id))


class _FailingReleaseTracker(_FakeTracker):
    def release_attempt(self, issue_number: int, label: str, assignee_id: str) -> None:
        raise OSError("GitHub unavailable")

    def release_handoff(self, issue_number: int, assignee_id: str) -> None:
        raise OSError("GitHub unavailable")


class _FakeWorkspace:
    working_directory = Path("/repository")

    def __init__(self, *, commits: tuple[str, ...] = ()) -> None:
        self.cleanup_calls: list[tuple[str, bool]] = []
        self._commits = commits

    def is_clean(self) -> bool:
        return True

    def prepare_for_profile_read(self, *, base_branch: str = "main") -> None:
        return None

    def prepare_attempt(self, *, base_branch: str, issue_number: int):
        return type("Prepared", (), {"branch": f"agent/issue-{issue_number}"})()

    def cleanup(self, *, base_branch: str, prepared: object, retain_branch: bool) -> None:
        self.cleanup_calls.append((base_branch, retain_branch))

    def commits_added(self, prepared: object) -> tuple[str, ...]:
        return self._commits


class _DirtyWorkspace(_FakeWorkspace):
    def is_clean(self) -> bool:
        return False

    def prepare_for_profile_read(self, *, base_branch: str = "main") -> None:
        raise DirtyWorkspaceError("Repository working tree contains uncommitted changes")

    def prepare_attempt(self, *, base_branch: str, issue_number: int):
        raise DirtyWorkspaceError("Repository working tree contains uncommitted changes")


class _FakePublisher:
    def __init__(self) -> None:
        self.outcomes: list[AttemptOutcome | None] = []
        self.requests: list[object] = []

    def publish(self, request: object):
        self.requests.append(request)
        self.outcomes.append(request.decision.outcome)
        outcome = request.decision.outcome or AttemptOutcome.COMPLETE
        branch_url = None if outcome is AttemptOutcome.INFRASTRUCTURE_ERROR else "https://example.test/tree/x"
        return type("Published", (), {"outcome": outcome, "branch_url": branch_url, "comment_posted": True})()


class _CommentFailingPublisher(_FakePublisher):
    def publish(self, request: object):
        self.requests.append(request)
        self.outcomes.append(request.decision.outcome)
        return type("Published", (), {"outcome": request.decision.outcome, "comment_posted": False})()


def _profile(base_branch: str = "main"):
    return type("Profile", (), {"base_branch": base_branch})()


def _state_at(tmp_path: Path, phase: AttemptPhase) -> AttemptStateStore:
    state = AttemptStateStore(tmp_path)
    state.start(issue_number=24, branch="agent/issue-24")
    for next_phase in (AttemptPhase.SETUP, AttemptPhase.MODEL_RUNNING, AttemptPhase.PUSHING, AttemptPhase.PUBLISHING):
        if state.read() and state.read().phase is phase:  # type: ignore[union-attr]
            break
        state.transition(next_phase)
    return state


def _issue(number: int) -> TrackerIssue:
    return TrackerIssue(
        id=f"issue-{number}",
        number=number,
        title="Implement lifecycle",
        body="Connect boundaries.",
        created_at=datetime(2026, 9, 20, tzinfo=UTC),
        state="OPEN",
        labels=frozenset({"ready-for-agent"}),
        assignee_logins=(),
        blocked_by=0,
        author_login="reporter",
    )
