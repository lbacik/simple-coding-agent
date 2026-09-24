"""Behaviour tests for the single-process implementation lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.completion import AttemptOutcome, CompletionDecision, PublicationPath
from simple_coding_agent.git_workspace import GitWorkspaceRecoveryError
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.lifecycle import AgentLifecycle, AttemptEvidence, LifecycleStatus
from simple_coding_agent.operating import ConsecutiveErrorStore


def test_interruption_handler_logs_a_terminal_event_then_resignals(tmp_path: Path) -> None:
    """SIGTERM/SIGINT must not kill the process silently mid-attempt."""

    import signal

    from simple_coding_agent.lifecycle import AttemptInterruptionHandler

    events: list[tuple[str, str, str, int | None]] = []
    resignalled: list[int] = []
    handler = AttemptInterruptionHandler(
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level, issue_number)
        ),
        issue_number_provider=lambda: 24,
        resignal=resignalled.append,
    )

    handler(signal.SIGTERM, None)

    assert events == [
        ("attempt_interrupted", "Process received SIGTERM during the attempt.", "ERROR", 24)
    ]
    assert resignalled == [signal.SIGTERM]


def test_interruption_handler_restores_previous_handlers(tmp_path: Path) -> None:
    import signal

    from simple_coding_agent.lifecycle import AttemptInterruptionHandler

    before_term = signal.getsignal(signal.SIGTERM)
    before_int = signal.getsignal(signal.SIGINT)
    handler = AttemptInterruptionHandler(
        event_log=lambda *args, **kwargs: None,
        issue_number_provider=lambda: None,
        resignal=lambda signum: None,
    )

    restore = handler.install()
    try:
        assert signal.getsignal(signal.SIGTERM) is handler
        assert signal.getsignal(signal.SIGINT) is handler
    finally:
        restore()

    assert signal.getsignal(signal.SIGTERM) == before_term
    assert signal.getsignal(signal.SIGINT) == before_int


def test_keyboard_interrupt_still_publishes_a_terminal_outcome(tmp_path: Path) -> None:
    """KeyboardInterrupt bypasses `except Exception`; it must still log and publish."""

    claim = Claim(issue(24), Assignment("issue-24", "agent-id"))
    tracker = FakeTracker(claim)
    workspace = FakeWorkspace()
    publisher = FakePublisher()
    events: list[tuple[str, str, int | None]] = []
    state = AttemptStateStore(tmp_path)

    def interrupted_workflow(received_claim: Claim, profile: object, prepared: object):
        raise KeyboardInterrupt("operator stop")

    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: profile(),
        publisher=publisher,
        attempt_runner=interrupted_workflow,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, issue_number)
        ),
    )

    with pytest.raises(KeyboardInterrupt):
        lifecycle.run_once()

    assert ("attempt_exception", "KeyboardInterrupt: operator stop", 24) in events
    assert publisher.outcomes == [AttemptOutcome.INFRASTRUCTURE_ERROR]
    assert tracker.cleanup == [(24, "ready-for-agent", "agent-id")]
    assert state.read() is None


def test_setup_failure_posts_result_then_releases_only_the_agent_claim(
    tmp_path: Path,
) -> None:
    claim = Claim(issue(24), Assignment("issue-24", "agent-id"))
    state = AttemptStateStore(tmp_path)
    tracker = FakeTracker(claim)
    workspace = FakeWorkspace()
    publisher = FakePublisher()
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: (_ for _ in ()).throw(OSError("profile is missing")),
        publisher=publisher,
    )

    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.ATTEMPTED
    assert result.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    assert publisher.outcomes == [AttemptOutcome.INFRASTRUCTURE_ERROR]
    assert tracker.cleanup == [(24, "ready-for-agent", "agent-id")]
    assert state.read() is None
    assert workspace.cleanup_calls == [("main", False)]


def test_empty_queue_sleeps_once_without_attempting_work(tmp_path: Path) -> None:
    sleeps: list[int] = []
    events: list[tuple[str, str, str]] = []
    lifecycle = AgentLifecycle(
        tracker=FakeTracker(None),
        attempt_state=AttemptStateStore(tmp_path),
        workspace=FakeWorkspace(),
        profile_loader=lambda _: None,
        publisher=FakePublisher(),
        poll_interval=17,
        sleeper=sleeps.append,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level)
        ),
    )

    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.IDLE
    assert sleeps == [17]
    assert [event for event, _, _ in events] == ["polling_for_issue", "no_eligible_issue_found"]
    assert events[1][1] == "no ready-for-agent issue available; sleeping 17s"
    assert events[1][2] == "INFO"


def test_stops_after_the_persisted_consecutive_infrastructure_error_limit(tmp_path: Path) -> None:
    events: list[tuple[str, str]] = []
    lifecycle = AgentLifecycle(
        tracker=FakeTracker(Claim(issue(24), Assignment("issue-24", "agent-id"))),
        attempt_state=AttemptStateStore(tmp_path),
        workspace=FakeWorkspace(),
        profile_loader=lambda _: (_ for _ in ()).throw(OSError("profile is missing")),
        publisher=FakePublisher(),
        error_store=ConsecutiveErrorStore(tmp_path),
        max_consecutive_errors=1,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, issue_number)
        ),
    )

    with pytest.raises(SystemExit) as stopped:
        lifecycle.run_once()

    assert stopped.value.code == 1
    assert [event for event, _, _ in events] == [
        "polling_for_issue",
        "issue_claimed",
        "workspace_prepared",
        "attempt_exception",
        "consecutive_error_limit_reached",
    ]
    assert events[3][1] == "OSError: profile is missing"
    assert events[3][2] == 24
    assert events[4][2] == 24
    assert ConsecutiveErrorStore(tmp_path).read().count == 1


@pytest.mark.parametrize("phase", [AttemptPhase.CLAIMED, AttemptPhase.SETUP])
def test_startup_recovers_pre_model_checkpoint_before_claiming_new_work(
    tmp_path: Path, phase: AttemptPhase
) -> None:
    state = state_at(tmp_path, phase)
    tracker = FakeTracker(Claim(issue(25), Assignment("issue-25", "agent-id")))
    workspace = FakeWorkspace()
    publisher = FakePublisher()
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: profile(),
        publisher=publisher,
    )

    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.ATTEMPTED
    assert publisher.outcomes == [AttemptOutcome.INFRASTRUCTURE_ERROR]
    assert tracker.cleanup[0] == (24, "ready-for-agent", "agent-id")
    assert tracker.claimed == []
    assert state.read() is None
    assert workspace.cleanup_calls[0] == ("main", False)


def test_startup_recovers_an_interrupted_final_check_as_infrastructure_error(
    tmp_path: Path,
) -> None:
    """A process that dies mid final-check must still produce a terminal event.

    The runner records `final_check_started` in the attempt archive before
    running the check; when recovery sees that marker without a matching
    finish, the outcome is infrastructure_error (not a mislabelled
    model-interruption), with committed work preserved.
    """

    from simple_coding_agent.observability import AttemptArchive

    state = state_at(tmp_path, AttemptPhase.MODEL_RUNNING)
    checkpoint = state.read()
    assert checkpoint is not None
    AttemptArchive(tmp_path, issue_number=24, started_at=checkpoint.started_at).write_attempt(
        {"final_check_started": "2026-09-24T08:15:00Z"}
    )
    events: list[tuple[str, int | None]] = []
    workspace = FakeWorkspace(commits=("completed",))
    publisher = FakePublisher()
    lifecycle = AgentLifecycle(
        tracker=FakeTracker(None),
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: profile(),
        publisher=publisher,
        sleeper=lambda _: None,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, issue_number)
        ),
        attempt_archive_factory=lambda issue_number, started_at: AttemptArchive(
            tmp_path, issue_number=issue_number, started_at=started_at
        ),
    )

    lifecycle.run_once()

    assert publisher.outcomes == [AttemptOutcome.INFRASTRUCTURE_ERROR]
    assert ("final_check_interrupted", 24) in events
    assert "interrupted before completing" in publisher.requests[0].decision.reasons[0]
    assert workspace.cleanup_calls == [("main", True)]
    assert state.read() is None


def test_startup_recovers_a_finished_final_check_as_model_work(tmp_path: Path) -> None:
    from simple_coding_agent.observability import AttemptArchive

    state = state_at(tmp_path, AttemptPhase.MODEL_RUNNING)
    checkpoint = state.read()
    assert checkpoint is not None
    AttemptArchive(tmp_path, issue_number=24, started_at=checkpoint.started_at).write_attempt(
        {
            "final_check_started": "2026-09-24T08:15:00Z",
            "final_check_finished": "2026-09-24T08:16:00Z",
        }
    )
    workspace = FakeWorkspace(commits=("completed",))
    publisher = FakePublisher()
    lifecycle = AgentLifecycle(
        tracker=FakeTracker(None),
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: profile(),
        publisher=publisher,
        sleeper=lambda _: None,
        attempt_archive_factory=lambda issue_number, started_at: AttemptArchive(
            tmp_path, issue_number=issue_number, started_at=started_at
        ),
    )

    lifecycle.run_once()

    assert publisher.outcomes == [AttemptOutcome.INCOMPLETE]


def test_startup_recovers_model_work_by_publishing_partial_branch_without_sdk_resume(
    tmp_path: Path,
) -> None:
    state = state_at(tmp_path, AttemptPhase.MODEL_RUNNING)
    workspace = FakeWorkspace(commits=("unfinished",))
    publisher = FakePublisher()
    lifecycle = AgentLifecycle(
        tracker=FakeTracker(None),
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: profile(),
        publisher=publisher,
        sleeper=lambda _: None,
    )

    lifecycle.run_once()

    assert publisher.outcomes == [AttemptOutcome.INCOMPLETE]
    assert publisher.requests[0].decision.publication_path is PublicationPath.PARTIAL
    assert state.read() is None
    assert workspace.cleanup_calls == [("main", False)]


def test_startup_recovers_model_checkpoint_without_commits_as_infrastructure_error(
    tmp_path: Path,
) -> None:
    state = state_at(tmp_path, AttemptPhase.MODEL_RUNNING)
    publisher = FakePublisher()
    lifecycle = AgentLifecycle(
        tracker=FakeTracker(None),
        attempt_state=state,
        workspace=FakeWorkspace(),
        profile_loader=lambda _: profile(),
        publisher=publisher,
        sleeper=lambda _: None,
    )

    lifecycle.run_once()

    assert publisher.outcomes == [AttemptOutcome.INFRASTRUCTURE_ERROR]


def test_startup_recovers_a_missing_profile_with_standard_cleanup(tmp_path: Path) -> None:
    state = state_at(tmp_path, AttemptPhase.SETUP)
    tracker = FakeTracker(None)
    workspace = FakeWorkspace()
    publisher = FakePublisher()
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: (_ for _ in ()).throw(OSError("profile is missing")),
        publisher=publisher,
        sleeper=lambda _: None,
    )

    lifecycle.run_once()

    assert publisher.outcomes == [AttemptOutcome.INFRASTRUCTURE_ERROR]
    assert tracker.cleanup == [(24, "ready-for-agent", "agent-id")]
    assert workspace.cleanup_calls == [("main", False)]
    assert state.read() is None


@pytest.mark.parametrize("phase", [AttemptPhase.PUSHING, AttemptPhase.PUBLISHING])
def test_startup_retries_completed_publication_and_deduplicates_cleanup(
    tmp_path: Path, phase: AttemptPhase
) -> None:
    state = state_at(tmp_path, phase)
    tracker = FakeTracker(None)
    publisher = FakePublisher()
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=FakeWorkspace(commits=("completed",)),
        profile_loader=lambda _: profile(),
        publisher=publisher,
        sleeper=lambda _: None,
    )

    lifecycle.run_once()

    assert publisher.outcomes == [None]
    assert publisher.requests[0].decision.publication_eligible
    assert tracker.cleanup == [(24, "ready-for-agent", "agent-id")]
    assert state.read() is None


def test_startup_preserves_commits_when_recovered_push_fails(tmp_path: Path) -> None:
    state = state_at(tmp_path, AttemptPhase.PUSHING)
    workspace = FakeWorkspace(commits=("completed",))
    lifecycle = AgentLifecycle(
        tracker=FakeTracker(None),
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: profile(),
        publisher=PushFailingPublisher(),
        sleeper=lambda _: None,
    )

    result = lifecycle.run_once()

    assert result.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    assert workspace.cleanup_calls == [("main", True)]
    assert state.read() is None


def test_startup_retries_recovery_after_result_comment_failure(tmp_path: Path) -> None:
    state = state_at(tmp_path, AttemptPhase.PUSHING)
    workspace = FakeWorkspace(commits=("completed",))
    first = AgentLifecycle(
        tracker=FakeTracker(None),
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: profile(),
        publisher=CommentFailingPublisher(),
        sleeper=lambda _: None,
    )

    first.run_once()

    assert state.read() is not None
    assert workspace.cleanup_calls == [("main", True)]

    tracker = FakeTracker(None)
    second = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: profile(),
        publisher=FakePublisher(),
        sleeper=lambda _: None,
    )

    second.run_once()

    assert tracker.cleanup == [(24, "ready-for-agent", "agent-id")]
    assert state.read() is None


def test_attempt_runs_the_injected_workflow_then_publishes_before_cleanup(tmp_path: Path) -> None:
    claim = Claim(issue(24), Assignment("issue-24", "agent-id"))
    tracker = FakeTracker(claim)
    workspace = FakeWorkspace()
    publisher = FakePublisher()
    calls: list[str] = []

    def run_workflow(received_claim: Claim, profile: object, prepared: object) -> AttemptEvidence:
        calls.append("workflow")
        assert received_claim == claim
        return AttemptEvidence(
            decision=CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)),
            check_command="pytest",
            check_exit_code=0,
            review_cycles=1,
            review_findings="all clear",
            details="Implemented the lifecycle.",
        )

    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: profile(),
        publisher=publisher,
        attempt_runner=run_workflow,
    )

    result = lifecycle.run_once()

    assert result.outcome is AttemptOutcome.COMPLETE
    assert calls == ["workflow"]
    assert publisher.outcomes == [None]
    assert tracker.cleanup == [(24, "ready-for-agent", "agent-id")]
    assert workspace.cleanup_calls == [("main", False)]


def test_partial_push_failure_keeps_incomplete_and_retains_the_local_branch(
    tmp_path: Path,
) -> None:
    claim = Claim(issue(24), Assignment("issue-24", "agent-id"))
    tracker = FakeTracker(claim)
    workspace = FakeWorkspace()
    publisher = IncompletePushFailingPublisher()
    store = ConsecutiveErrorStore(tmp_path)

    def run_workflow(received_claim: Claim, profile: object, prepared: object) -> AttemptEvidence:
        return AttemptEvidence(
            decision=CompletionDecision(
                AttemptOutcome.INCOMPLETE,
                False,
                PublicationPath.PARTIAL,
                ("Final authoritative check did not succeed.",),
            ),
            check_command="pytest",
            check_exit_code=2,
            review_cycles=1,
            review_findings="all clear",
            details="Partial implementation.",
        )

    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: profile(),
        publisher=publisher,
        attempt_runner=run_workflow,
        error_store=store,
        max_consecutive_errors=3,
    )

    result = lifecycle.run_once()

    assert result.outcome is AttemptOutcome.INCOMPLETE
    assert workspace.cleanup_calls == [("main", True)]
    assert store.read().count == 0


def test_logs_every_stage_from_claiming_the_issue_to_dispatching_the_model(
    tmp_path: Path,
) -> None:
    """Nothing was logged between claiming an issue and dispatching the model,

    so an operator reading the log stream during that window (workspace
    prep, profile load, setup/baseline checks) couldn't tell which of those
    steps an attempt was stuck on.
    """

    claim = Claim(issue(24), Assignment("issue-24", "agent-id"))
    tracker = FakeTracker(claim)
    workspace = FakeWorkspace()
    publisher = FakePublisher()
    events: list[tuple[str, int | None]] = []

    def run_workflow(received_claim: Claim, profile: object, prepared: object) -> AttemptEvidence:
        return AttemptEvidence(
            decision=CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)),
            check_command="pytest",
            check_exit_code=0,
            review_cycles=1,
            review_findings="all clear",
            details="Implemented the lifecycle.",
        )

    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: profile(),
        publisher=publisher,
        attempt_runner=run_workflow,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, issue_number)
        ),
    )

    result = lifecycle.run_once()

    assert result.outcome is AttemptOutcome.COMPLETE
    assert [event for event, _ in events] == [
        "polling_for_issue",
        "issue_claimed",
        "workspace_prepared",
        "profile_loaded",
        "attempt_phase_transitioned",
    ]
    assert all(issue_number == 24 for _, issue_number in events[1:])


def test_keeps_the_checkpoint_when_remote_cleanup_is_interrupted(tmp_path: Path) -> None:
    state = AttemptStateStore(tmp_path)
    tracker = FailingCleanupTracker(Claim(issue(24), Assignment("issue-24", "agent-id")))
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=FakeWorkspace(),
        profile_loader=lambda _: (_ for _ in ()).throw(OSError("profile is missing")),
        publisher=FakePublisher(),
    )

    with pytest.raises(OSError, match="GitHub unavailable"):
        lifecycle.run_once()

    assert state.read() is not None


def test_recreates_the_attempt_branch_from_a_profile_base_branch(tmp_path: Path) -> None:
    workspace = FakeWorkspace()
    lifecycle = AgentLifecycle(
        tracker=FakeTracker(Claim(issue(24), Assignment("issue-24", "agent-id"))),
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: profile("release"),
        publisher=FakePublisher(),
    )

    lifecycle.run_once()

    assert workspace.prepared_bases == ["main", "release"]
    assert workspace.cleanup_calls == [("main", False), ("release", False)]


def test_does_not_start_cleanup_when_the_result_comment_was_not_posted(tmp_path: Path) -> None:
    state = AttemptStateStore(tmp_path)
    tracker = FakeTracker(Claim(issue(24), Assignment("issue-24", "agent-id")))
    workspace = FakeWorkspace()
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: (_ for _ in ()).throw(OSError("profile is missing")),
        publisher=CommentFailingPublisher(),
    )

    lifecycle.run_once()

    assert tracker.cleanup == []
    assert state.read() is not None
    assert workspace.cleanup_calls == [("main", True)]


def test_stops_without_touching_the_issue_when_the_workspace_cannot_be_repaired(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, str, str]] = []
    state = AttemptStateStore(tmp_path)
    tracker = FakeTracker(Claim(issue(24), Assignment("issue-24", "agent-id")))
    workspace = RecoveryFailingWorkspace()
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: (_ for _ in ()).throw(AssertionError("must not be reached")),
        publisher=FakePublisher(),
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level, issue_number)
        ),
    )

    with pytest.raises(SystemExit):
        lifecycle.run_once()

    assert tracker.cleanup == []
    assert workspace.cleanup_calls == []
    assert state.read() is not None
    assert (
        "git_workspace_unrecoverable",
        "GitWorkspaceRecoveryError: workspace is broken",
        "ERROR",
        24,
    ) in events


@dataclass
class FakeTracker:
    next_claim: Claim | None

    def __post_init__(self) -> None:
        self.cleanup: list[tuple[int, str, str]] = []
        self.claimed: list[int] = []

    def claim_next(self) -> Claim | None:
        claim, self.next_claim = self.next_claim, None
        if claim is not None:
            self.claimed.append(claim.issue.number)
        return claim

    def recover_claim(self, issue_number: int) -> Claim | None:
        return Claim(issue(issue_number), Assignment(f"issue-{issue_number}", "agent-id"))

    def release_attempt(self, issue_number: int, label: str, assignee_id: str) -> None:
        self.cleanup.append((issue_number, label, assignee_id))


class FailingCleanupTracker(FakeTracker):
    def release_attempt(self, issue_number: int, label: str, assignee_id: str) -> None:
        raise OSError("GitHub unavailable")


class FakeWorkspace:
    working_directory = Path("/repository")

    def __init__(self, *, commits: tuple[str, ...] = ()) -> None:
        self.cleanup_calls: list[tuple[str, bool]] = []
        self.prepared_bases: list[str] = []
        self._commits = commits

    def prepare_attempt(self, *, base_branch: str, issue_number: int):
        self.prepared_bases.append(base_branch)
        return type("Prepared", (), {"branch": f"agent/issue-{issue_number}"})()

    def cleanup(self, *, base_branch: str, prepared: object, retain_branch: bool) -> None:
        self.cleanup_calls.append((base_branch, retain_branch))

    def commits_added(self, prepared: object) -> tuple[str, ...]:
        return self._commits


class RecoveryFailingWorkspace(FakeWorkspace):
    def prepare_attempt(self, *, base_branch: str, issue_number: int):
        raise GitWorkspaceRecoveryError("workspace is broken")


class FakePublisher:
    def __init__(self) -> None:
        self.outcomes: list[AttemptOutcome] = []
        self.requests: list[object] = []

    def publish(self, request: object):
        self.requests.append(request)
        self.outcomes.append(request.decision.outcome)
        outcome = request.decision.outcome or AttemptOutcome.COMPLETE
        branch_url = (
            None
            if outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
            else "https://example.test/tree/agent/issue-24"
        )
        return type("Published", (), {"outcome": outcome, "branch_url": branch_url})()


class CommentFailingPublisher(FakePublisher):
    def publish(self, request: object):
        return type("Published", (), {"outcome": request.decision.outcome, "comment_posted": False})()


class PushFailingPublisher(FakePublisher):
    def publish(self, request: object):
        self.requests.append(request)
        self.outcomes.append(AttemptOutcome.INFRASTRUCTURE_ERROR)
        return type("Published", (), {
            "outcome": AttemptOutcome.INFRASTRUCTURE_ERROR,
            "branch_url": None,
        })()


class IncompletePushFailingPublisher(FakePublisher):
    def publish(self, request: object):
        self.requests.append(request)
        self.outcomes.append(AttemptOutcome.INCOMPLETE)
        return type("Published", (), {
            "outcome": AttemptOutcome.INCOMPLETE,
            "branch_url": None,
        })()


def profile(base_branch: str = "main"):
    return type("Profile", (), {"base_branch": base_branch})()


def state_at(tmp_path: Path, phase: AttemptPhase) -> AttemptStateStore:
    state = AttemptStateStore(tmp_path)
    state.start(issue_number=24, branch="agent/issue-24")
    for next_phase in (AttemptPhase.SETUP, AttemptPhase.MODEL_RUNNING, AttemptPhase.PUSHING, AttemptPhase.PUBLISHING):
        if state.read().phase is phase:
            break
        state.transition(next_phase)
    return state


def issue(number: int) -> TrackerIssue:
    from datetime import UTC, datetime

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
