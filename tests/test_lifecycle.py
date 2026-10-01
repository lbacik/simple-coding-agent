"""Behaviour tests for the single-process implementation lifecycle."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.completion import AttemptOutcome, CompletionDecision, PublicationPath
from simple_coding_agent.git_workspace import GitWorkspaceRecoveryError, RebaseConflictError
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.lifecycle import AgentLifecycle, AttemptEvidence, LifecycleStatus
from simple_coding_agent.operating import ConsecutiveErrorStore
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace


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
    tracker = FakeTracker(next_claim=claim)
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
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
    tracker = FakeTracker(next_claim=claim)
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
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
    # The profile never loaded, so no attempt branch was ever prepared and
    # there is nothing to clean up.
    assert workspace.cleanup_calls == []


def test_empty_queue_sleeps_once_without_attempting_work(tmp_path: Path) -> None:
    sleeps: list[int] = []
    events: list[tuple[str, str, str]] = []
    lifecycle = AgentLifecycle(
        tracker=FakeTracker(),
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo", issue_number=24),
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
        tracker=FakeTracker(next_claim=Claim(issue(24), Assignment("issue-24", "agent-id"))),
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo", issue_number=24),
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
        "workspace_prepared_for_profile_read",
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
    tracker = FakeTracker(
        {24: issue(24)}, next_claim=Claim(issue(25), Assignment("issue-25", "agent-id"))
    )
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
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
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
    workspace.model_commit("completed")
    publisher = FakePublisher()
    lifecycle = AgentLifecycle(
        tracker=FakeTracker({24: issue(24)}),
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
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
    workspace.model_commit("completed")
    publisher = FakePublisher()
    lifecycle = AgentLifecycle(
        tracker=FakeTracker({24: issue(24)}),
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
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
    workspace.model_commit("unfinished")
    publisher = FakePublisher()
    lifecycle = AgentLifecycle(
        tracker=FakeTracker({24: issue(24)}),
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
        tracker=FakeTracker({24: issue(24)}),
        attempt_state=state,
        workspace=InMemoryWorkspace(tmp_path / "repo", issue_number=24),
        profile_loader=lambda _: profile(),
        publisher=publisher,
        sleeper=lambda _: None,
    )

    lifecycle.run_once()

    assert publisher.outcomes == [AttemptOutcome.INFRASTRUCTURE_ERROR]


def test_startup_recovers_a_missing_profile_with_standard_cleanup(tmp_path: Path) -> None:
    state = state_at(tmp_path, AttemptPhase.SETUP)
    tracker = FakeTracker({24: issue(24)})
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
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
    # The profile never loaded, so no attempt branch was ever prepared.
    assert workspace.cleanup_calls == []
    assert state.read() is None


@pytest.mark.parametrize("phase", [AttemptPhase.PUSHING, AttemptPhase.PUBLISHING])
def test_startup_retries_completed_publication_and_deduplicates_cleanup(
    tmp_path: Path, phase: AttemptPhase
) -> None:
    state = state_at(tmp_path, phase)
    tracker = FakeTracker({24: issue(24)})
    publisher = FakePublisher()
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
    workspace.model_commit("completed")
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: profile(),
        publisher=publisher,
        sleeper=lambda _: None,
    )

    lifecycle.run_once()

    assert publisher.outcomes == [AttemptOutcome.COMPLETE]
    assert publisher.requests[0].decision.publication_eligible
    assert tracker.cleanup == [(24, "ready-for-agent", "agent-id")]
    assert state.read() is None


def test_startup_preserves_commits_when_recovered_push_fails(tmp_path: Path) -> None:
    state = state_at(tmp_path, AttemptPhase.PUSHING)
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
    workspace.model_commit("completed")
    lifecycle = AgentLifecycle(
        tracker=FakeTracker({24: issue(24)}),
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: profile(),
        publisher=FakePublisher(outcome=AttemptOutcome.INFRASTRUCTURE_ERROR),
        sleeper=lambda _: None,
    )

    result = lifecycle.run_once()

    assert result.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    assert workspace.cleanup_calls == [("main", True)]
    assert state.read() is None


def test_startup_retries_recovery_after_result_comment_failure(tmp_path: Path) -> None:
    state = state_at(tmp_path, AttemptPhase.PUSHING)
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
    workspace.model_commit("completed")
    first = AgentLifecycle(
        tracker=FakeTracker({24: issue(24)}),
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: profile(),
        publisher=FakePublisher(comment_posted=False),
        sleeper=lambda _: None,
    )

    first.run_once()

    assert state.read() is not None
    assert workspace.cleanup_calls == [("main", True)]

    tracker = FakeTracker({24: issue(24)})
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
    tracker = FakeTracker(next_claim=claim)
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
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
    assert publisher.outcomes == [AttemptOutcome.COMPLETE]
    assert tracker.cleanup == [(24, "ready-for-agent", "agent-id")]
    assert workspace.cleanup_calls == [("main", False)]


def test_partial_push_failure_keeps_incomplete_and_retains_the_local_branch(
    tmp_path: Path,
) -> None:
    claim = Claim(issue(24), Assignment("issue-24", "agent-id"))
    tracker = FakeTracker(next_claim=claim)
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
    publisher = FakePublisher(outcome=AttemptOutcome.INCOMPLETE, branch_url=None)
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


def test_dirty_tree_at_cleanup_holds_the_workspace_on_the_attempt_branch(
    tmp_path: Path,
) -> None:
    """Uncommitted work left at cleanup holds intake with the branch preserved.

    The workflow leaves dirty work behind (as an unverified final check
    would); cleanup must refuse to check out the base branch, emit
    ``workspace_hold``, and pause the loop via ``SystemExit(1)`` without
    finalizing the checkpoint.
    """

    claim = Claim(issue(24), Assignment("issue-24", "agent-id"))
    tracker = FakeTracker(next_claim=claim)
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
    publisher = FakePublisher()
    events: list[tuple[str, str, str, int | None]] = []
    state = AttemptStateStore(tmp_path)

    def run_workflow(received_claim: Claim, profile: object, prepared: object) -> AttemptEvidence:
        workspace.make_dirty("uncommitted.py", "leftover work")
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
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: profile(),
        publisher=publisher,
        attempt_runner=run_workflow,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level, issue_number)
        ),
    )

    with pytest.raises(SystemExit) as held:
        lifecycle.run_once()

    assert held.value.code == 1
    # Cleanup was attempted but refused to leave the attempt branch.
    assert workspace.cleanup_calls == [("main", False)]
    assert workspace.current_branch == "agent/issue-24"
    assert any(
        event == "workspace_hold" and level == "WARNING" and issue_number == 24
        for event, _, level, issue_number in events
    )
    # The checkpoint is retained for operator recovery, not finalized.
    assert state.read() is not None


def test_logs_every_stage_from_claiming_the_issue_to_dispatching_the_model(
    tmp_path: Path,
) -> None:
    """Nothing was logged between claiming an issue and dispatching the model,

    so an operator reading the log stream during that window (workspace
    prep, profile load, setup/baseline checks) couldn't tell which of those
    steps an attempt was stuck on.
    """

    claim = Claim(issue(24), Assignment("issue-24", "agent-id"))
    tracker = FakeTracker(next_claim=claim)
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
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
        "workspace_prepared_for_profile_read",
        "profile_loaded",
        "workspace_prepared",
        "attempt_phase_transitioned",
    ]
    assert all(issue_number == 24 for _, issue_number in events[1:])


def test_keeps_the_checkpoint_when_remote_cleanup_is_interrupted(tmp_path: Path) -> None:
    state = AttemptStateStore(tmp_path)
    tracker = FailingCleanupTracker(
        next_claim=Claim(issue(24), Assignment("issue-24", "agent-id"))
    )
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=InMemoryWorkspace(tmp_path / "repo", issue_number=24),
        profile_loader=lambda _: (_ for _ in ()).throw(OSError("profile is missing")),
        publisher=FakePublisher(),
    )

    with pytest.raises(OSError, match="GitHub unavailable"):
        lifecycle.run_once()

    assert state.read() is not None


def test_recreates_the_attempt_branch_from_a_profile_base_branch(tmp_path: Path) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
    lifecycle = AgentLifecycle(
        tracker=FakeTracker(next_claim=Claim(issue(24), Assignment("issue-24", "agent-id"))),
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: profile("release"),
        publisher=FakePublisher(),
    )

    lifecycle.run_once()

    # The profile names the base, so the attempt is prepared exactly once,
    # onto that base; the bootstrap for the profile read never prepares.
    assert workspace.bootstrap_bases == ["main"]
    assert workspace.prepared_bases == ["release"]
    assert workspace.cleanup_calls == [("release", False)]


def test_does_not_start_cleanup_when_the_result_comment_was_not_posted(tmp_path: Path) -> None:
    state = AttemptStateStore(tmp_path)
    tracker = FakeTracker(next_claim=Claim(issue(24), Assignment("issue-24", "agent-id")))
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: (_ for _ in ()).throw(OSError("profile is missing")),
        publisher=FakePublisher(comment_posted=False),
    )

    lifecycle.run_once()

    assert tracker.cleanup == []
    assert state.read() is not None
    # The profile never loaded, so no attempt branch was ever prepared.
    assert workspace.cleanup_calls == []


def test_stops_without_touching_the_issue_when_the_workspace_cannot_be_repaired(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, str, str]] = []
    state = AttemptStateStore(tmp_path)
    tracker = FakeTracker(next_claim=Claim(issue(24), Assignment("issue-24", "agent-id")))
    workspace = RecoveryFailingWorkspace(tmp_path / "repo")
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


def test_unresolved_rebase_conflict_after_the_model_reports_infrastructure_error(
    tmp_path: Path,
) -> None:
    """A conflict the model leaves unresolved fails as rebase_conflict.

    Preparation leaves the conflicting rebase in place for the model, so
    the model session still runs; only when the attempt runner reports the
    conflict unresolved (as RebaseConflictError) does the attempt end as an
    infrastructure error without setup ever running on the conflicted tree.
    """

    events: list[tuple[str, str, str]] = []
    claim = Claim(issue(24), Assignment("issue-24", "agent-id"))
    workspace = RebaseConflictingWorkspace(tmp_path / "repo")
    publisher = FakePublisher()
    calls: list[str] = []

    def attempt_runner(received_claim: Claim, profile: object, prepared: object) -> AttemptEvidence:
        calls.append("model")
        raise RebaseConflictError(
            "Rebase of agent/issue-24 onto develop hit conflicts"
            " (rebase_conflict). Conflicting files: SubscriptionController.php."
            " The model left conflicts unresolved, so the rebase was aborted"
            " and setup was not started.",
            branch="agent/issue-24",
            base_branch="develop",
            conflicted_files=("SubscriptionController.php",),
        )

    lifecycle = AgentLifecycle(
        tracker=FakeTracker(next_claim=claim),
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: profile("develop"),
        publisher=publisher,
        attempt_runner=attempt_runner,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level)
        ),
    )

    result = lifecycle.run_once()

    assert result.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    assert calls == ["model"]
    assert any(event == "rebase_conflict" for event, _, _ in events)
    prepared_details = [detail for event, detail, _ in events if event == "workspace_prepared"]
    assert len(prepared_details) == 1
    assert "SubscriptionController.php" in prepared_details[0]
    assert "setup_started" not in [event for event, _, _ in events]
    assert publisher.outcomes == [AttemptOutcome.INFRASTRUCTURE_ERROR]
    details = publisher.requests[0].details
    assert "rebase_conflict" in details
    assert "Setup was not started" in details


def test_rebase_conflict_on_the_profile_base_reports_infrastructure_error(tmp_path: Path) -> None:
    """A conflict on the single preparation onto the profile base fails fast.

    The bootstrap for the profile read never touches the attempt branch, so
    by the time ``prepare_attempt`` runs the profile is already loaded. The
    conflict is left for the model; when the runner reports it unresolved,
    the attempt ends as an infrastructure error without setup running.
    """

    events: list[tuple[str, str, str]] = []
    claim = Claim(issue(24), Assignment("issue-24", "agent-id"))
    workspace = RebaseConflictingWorkspace(tmp_path / "repo", conflict_bases=("main",))
    publisher = FakePublisher()
    calls: list[str] = []

    def attempt_runner(received_claim: Claim, profile: object, prepared: object) -> AttemptEvidence:
        calls.append("model")
        raise RebaseConflictError(
            "Rebase of agent/issue-24 onto main hit conflicts (rebase_conflict)."
            " The model left conflicts unresolved, so the rebase was aborted"
            " and setup was not started.",
            branch="agent/issue-24",
            base_branch="main",
        )

    lifecycle = AgentLifecycle(
        tracker=FakeTracker(next_claim=claim),
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: profile("main"),
        publisher=publisher,
        attempt_runner=attempt_runner,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level)
        ),
    )

    result = lifecycle.run_once()

    assert result.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    assert calls == ["model"]
    assert any(event == "rebase_conflict" for event, _, _ in events)
    assert workspace.bootstrap_bases == ["main"]
    assert workspace.prepared_bases == ["main"]
    assert publisher.outcomes == [AttemptOutcome.INFRASTRUCTURE_ERROR]


def test_conflicted_prepare_still_dispatches_the_model(tmp_path: Path) -> None:
    """A reused branch whose rebase conflicts reaches the model session.

    The attempt no longer always ends as infrastructure_error: preparation
    succeeds with the conflict left in place and the model gets its chance
    to resolve it before setup runs.
    """

    claim = Claim(issue(24), Assignment("issue-24", "agent-id"))
    workspace = RebaseConflictingWorkspace(tmp_path / "repo")
    publisher = FakePublisher()
    calls: list[str] = []

    def attempt_runner(received_claim: Claim, profile: object, prepared: object) -> AttemptEvidence:
        calls.append("model")
        assert tuple(getattr(prepared, "rebase_conflicts", ())) == ("SubscriptionController.php",)
        return AttemptEvidence(
            decision=CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)),
            check_command="pytest",
            check_exit_code=0,
            review_cycles=1,
            review_findings="all clear",
            details="Resolved the conflict and implemented the issue.",
        )

    lifecycle = AgentLifecycle(
        tracker=FakeTracker(next_claim=claim),
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: profile("develop"),
        publisher=publisher,
        attempt_runner=attempt_runner,
    )

    result = lifecycle.run_once()

    assert calls == ["model"]
    assert result.outcome is AttemptOutcome.COMPLETE


def test_non_main_profile_prepares_the_attempt_only_onto_the_profile_base(
    tmp_path: Path,
) -> None:
    """The attempt branch must be rebased only onto the profile base branch.

    Regression test: bootstrapping the workspace to read the profile used to
    run a full ``prepare_attempt`` on ``main`` first, which rebased a
    ``develop``-based attempt branch onto ``main`` and failed with a rebase
    conflict before the profile was ever loaded.
    """

    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
    lifecycle = AgentLifecycle(
        tracker=FakeTracker(next_claim=Claim(issue(24), Assignment("issue-24", "agent-id"))),
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: profile("develop"),
        publisher=FakePublisher(),
    )

    lifecycle.run_once()

    assert workspace.bootstrap_bases == ["main"]
    assert workspace.prepared_bases == ["develop"]
    assert workspace.cleanup_calls == [("develop", False)]


def test_startup_recovery_prepares_only_onto_the_profile_base(tmp_path: Path) -> None:
    """The continuation path must share the single-prepare bootstrap behavior."""

    state = state_at(tmp_path, AttemptPhase.SETUP)
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
    lifecycle = AgentLifecycle(
        tracker=FakeTracker({24: issue(24)}),
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: profile("develop"),
        publisher=FakePublisher(),
        sleeper=lambda _: None,
    )

    lifecycle.run_once()

    assert workspace.bootstrap_bases == ["main"]
    assert workspace.prepared_bases == ["develop"]
    assert workspace.cleanup_calls == [("develop", False)]


class FailingCleanupTracker(FakeTracker):
    def release_attempt(self, issue_number: int, label: str, assignee_id: str) -> None:
        raise OSError("GitHub unavailable")


class RecoveryFailingWorkspace(InMemoryWorkspace):
    def prepare_for_profile_read(self, *, base_branch: str = "main") -> str:
        raise GitWorkspaceRecoveryError("workspace is broken")

    def prepare_attempt(self, *, base_branch: str, issue_number: int):
        raise GitWorkspaceRecoveryError("workspace is broken")


class RebaseConflictingWorkspace(InMemoryWorkspace):
    """Report a conflicting rebase on the configured bases, leaving it in place."""

    def __init__(self, root: Path, *, conflict_bases: tuple[str, ...] = ("develop",)) -> None:
        super().__init__(root, issue_number=24)
        self._conflict_bases = conflict_bases

    def prepare_attempt(self, *, base_branch: str, issue_number: int):
        prepared = super().prepare_attempt(base_branch=base_branch, issue_number=issue_number)
        if base_branch in self._conflict_bases:
            self.leave_conflict(("SubscriptionController.php",))
            prepared = replace(prepared, rebase_conflicts=("SubscriptionController.php",))
        return prepared


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


def test_classify_handoff_reason_prefers_structured_signals_over_explanation() -> None:
    from types import SimpleNamespace

    from simple_coding_agent.lifecycle import _classify_handoff_reason

    assert (
        _classify_handoff_reason(
            SimpleNamespace(
                token_hard_ceiling_reached=True,
                stop_reason="max_turns_exceeded",
                terminal_reason="max_turns",
                explanation="Model execution reached max_turns.",
            )
        )
        == "token_hard_ceiling"
    )
    assert (
        _classify_handoff_reason(
            SimpleNamespace(
                token_hard_ceiling_reached=False,
                stop_reason="tool_use",
                terminal_reason="budget_exhausted",
                explanation="anything at all",
            )
        )
        == "cost_hard_limit"
    )
    assert (
        _classify_handoff_reason(
            SimpleNamespace(
                stop_reason="max_budget_usd_exceeded",
                terminal_reason=None,
                explanation="",
            )
        )
        == "cost_hard_limit"
    )
    assert (
        _classify_handoff_reason(
            SimpleNamespace(
                stop_reason="max_turns_exceeded",
                terminal_reason="max_turns",
                explanation="",
            )
        )
        == "turn_limit"
    )
    assert (
        _classify_handoff_reason(
            SimpleNamespace(stop_reason="timeout", terminal_reason=None, explanation="")
        )
        == "time_limit"
    )
