"""Operator recovery: retry and release retained attempts (issue #84)."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.completion import AttemptOutcome, CompletionDecision, PublicationPath
from simple_coding_agent.control import CommandAcknowledgement, ControlStore
from simple_coding_agent.finalization import AttemptCompletionStore
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.lifecycle import AgentLifecycle, LifecycleStatus
from simple_coding_agent.recovery import (
    RecoveryRejectedError,
    parse_attempt_id,
    parse_saved_at,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _issue(number: int, assignees: tuple[str, ...] = ()) -> TrackerIssue:
    return TrackerIssue(
        id=f"issue-{number}",
        number=number,
        title="Implement recovery",
        body="Recover retained work.",
        created_at=datetime(2026, 9, 20, tzinfo=UTC),
        state="OPEN",
        labels=frozenset({"ready-for-agent"}),
        assignee_logins=assignees,
        blocked_by=0,
        author_login="reporter",
    )


def _claim(number: int) -> Claim:
    return Claim(_issue(number), Assignment(f"issue-{number}", "agent-id"))


class FakeTracker:
    def __init__(self, next_claim: Claim | None = None) -> None:
        self.next_claim = next_claim
        self.cleanup: list[tuple] = []
        self.claimed: list[int] = []

    def claim_next(self) -> Claim | None:
        claim_value, self.next_claim = self.next_claim, None
        if claim_value is not None:
            self.claimed.append(claim_value.issue.number)
        return claim_value

    def recover_claim(self, issue_number: int) -> Claim | None:
        return Claim(_issue(issue_number), Assignment(f"issue-{issue_number}", "agent-id"))

    def release_attempt(self, issue_number: int, label: str, assignee_id: str) -> None:
        self.cleanup.append((issue_number, label, assignee_id))

    def release_handoff(self, issue_number: int, assignee_id: str) -> None:
        self.cleanup.append((issue_number, "round-finished", assignee_id))


class FakeWorkspace:
    working_directory = Path("/repository")

    def __init__(self, commits: tuple = ()) -> None:
        self._commits = commits
        self.cleanup_calls: list[tuple] = []
        self.dirty = False

    def is_clean(self) -> bool:
        return not self.dirty

    def prepare_for_profile_read(self, *, base_branch: str = "main") -> None:
        return None

    def prepare_attempt(self, *, base_branch: str, issue_number: int):
        return type(
            "Prepared", (), {"branch": f"agent/issue-{issue_number}", "base_revision": "abc"}
        )()

    def cleanup(self, *, base_branch: str, prepared: object, retain_branch: bool) -> None:
        self.cleanup_calls.append((base_branch, retain_branch))

    def commits_added(self, prepared: object) -> tuple:
        return self._commits


class FakePublisher:
    _github = None
    _repository = "owner/repo"

    def __init__(self, comment_posted: bool = True) -> None:
        self.requests: list[object] = []
        self._comment_posted = comment_posted

    def publish(self, request: object):
        self.requests.append(request)
        outcome = request.decision.outcome or AttemptOutcome.COMPLETE
        return type(
            "Published",
            (),
            {"outcome": outcome, "branch_url": "https://example.test/x", "comment_posted": self._comment_posted},
        )()


def make_lifecycle(tmp_path: Path, *, tracker=None, workspace=None, publisher=None, store=None):
    tracker = tracker or FakeTracker()
    workspace = workspace or FakeWorkspace()
    publisher = publisher or FakePublisher()
    store = store or ControlStore(tmp_path)
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=publisher,
        attempt_runner=lambda c, p, pr: None,
        control_store=store,
        completion_store=AttemptCompletionStore(tmp_path),
        sleeper=lambda seconds: None,
        repository="owner/repo",
    )
    return lifecycle, tracker, workspace, publisher, store


def start_checkpoint(tmp_path: Path, phase: AttemptPhase = AttemptPhase.CLAIMED):
    state = AttemptStateStore(tmp_path)
    checkpoint = state.start(issue_number=24, branch="agent/issue-24")
    order = (AttemptPhase.SETUP, AttemptPhase.MODEL_RUNNING, AttemptPhase.PUSHING, AttemptPhase.PUBLISHING)
    for next_phase in order:
        current = state.read()
        assert current is not None
        if current.phase is phase:
            break
        state.transition(next_phase)
    return state.read()


# ---------------------------------------------------------------------------
# acceptance: restart reports recovering and blocks intake
# ---------------------------------------------------------------------------


def test_status_reports_recovery_hold_with_attempt_identity_and_next_action(tmp_path: Path) -> None:
    checkpoint = start_checkpoint(tmp_path)
    assert checkpoint is not None
    lifecycle, _, _, _, _ = make_lifecycle(tmp_path)

    snapshot = lifecycle.control_status("owner/repo")

    recovery = snapshot["recovery"]
    assert recovery is not None
    assert recovery["attempt_id"] == checkpoint.started_at
    assert recovery["issue_number"] == 24
    assert recovery["phase"] == checkpoint.phase.value
    assert recovery["branch"] == "agent/issue-24"
    assert recovery["workspace"]
    assert recovery["checkpoint"] is True
    assert "issue #24" in recovery["hold_reason"]
    assert checkpoint.started_at in recovery["hold_reason"]
    assert "recovery retry" in recovery["next_action"]
    assert "recovery release" in recovery["next_action"]
    assert recovery["publication"]["phase"] == checkpoint.phase.value


def test_status_reports_recovering_during_startup_reconciliation(tmp_path: Path) -> None:
    start_checkpoint(tmp_path)
    lifecycle, tracker, _, _, _ = make_lifecycle(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    original = tracker.recover_claim

    def blocking_recover(issue_number: int):
        entered.set()
        assert release.wait(timeout=30)
        return original(issue_number)

    tracker.recover_claim = blocking_recover  # type: ignore[method-assign]
    worker = threading.Thread(target=lifecycle.run_once)
    worker.start()
    assert entered.wait(timeout=30)
    try:
        snapshot = lifecycle.control_status("owner/repo")
    finally:
        release.set()
        worker.join(timeout=30)

    assert snapshot["recovering"] is True
    assert snapshot["intake"] == "recovering"
    assert snapshot["recovery"] is not None
    assert snapshot["recovery"]["attempt_id"] is not None


def test_retained_checkpoint_blocks_new_claims_until_recovery(tmp_path: Path) -> None:
    checkpoint = start_checkpoint(tmp_path)
    assert checkpoint is not None
    lifecycle, tracker, _, _, _ = make_lifecycle(tmp_path)
    tracker.next_claim = _claim(99)
    # Simulate an already-reconciled hold: startup reconciliation ran, the
    # retained checkpoint survived, and intake must sleep instead of
    # claiming new work (which would collide with the checkpoint).
    lifecycle._startup_reconciled = True

    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.IDLE
    assert result.outcome is None
    assert tracker.claimed == []
    assert AttemptStateStore(tmp_path).read() is not None

    # After the operator clears the hold, intake resumes on the next cycle.
    lifecycle.submit_recovery_retry("req-retry", checkpoint.started_at)
    lifecycle.run_once()
    assert tracker.claimed == [99]


def test_resume_stays_rejected_while_a_recovery_command_runs(tmp_path: Path) -> None:
    from simple_coding_agent.control import ResumeBlockedError

    checkpoint = start_checkpoint(tmp_path, AttemptPhase.PUSHING)
    assert checkpoint is not None
    entered = threading.Event()
    proceed = threading.Event()
    outcomes: list[object] = []

    class BlockingPublisher(FakePublisher):
        def publish(self, request: object):
            self.requests.append(request)
            entered.set()
            assert proceed.wait(timeout=30)
            return type(
                "Published",
                (),
                {
                    "outcome": AttemptOutcome.INFRASTRUCTURE_ERROR,
                    "branch_url": None,
                    "comment_posted": True,
                },
            )()

    lifecycle, _, _, _, store = make_lifecycle(tmp_path, publisher=BlockingPublisher())

    def run_retry() -> None:
        try:
            outcomes.append(lifecycle.submit_recovery_retry("req-retry", checkpoint.started_at))
        except Exception as error:  # pragma: no cover - surfaced below
            outcomes.append(error)

    worker = threading.Thread(target=run_retry)
    worker.start()
    assert entered.wait(timeout=30)
    try:
        # The retained checkpoint is owned by the running recovery command:
        # resume must stay rejected instead of slipping through the gap.
        with pytest.raises(ResumeBlockedError, match=checkpoint.started_at):
            lifecycle.submit_resume("req-resume-race")
    finally:
        proceed.set()
        worker.join(timeout=30)
    assert store.get_command("req-resume-race") is None
    assert len(outcomes) == 1 and not isinstance(outcomes[0], Exception)
    assert outcomes[0].acknowledgement is CommandAcknowledgement.COMPLETED


def test_resume_cannot_bypass_the_recovery_hold(tmp_path: Path) -> None:
    checkpoint = start_checkpoint(tmp_path)
    assert checkpoint is not None
    lifecycle, _, _, _, store = make_lifecycle(tmp_path)

    from simple_coding_agent.control import ResumeBlockedError

    with pytest.raises(ResumeBlockedError, match=checkpoint.started_at):
        lifecycle.submit_resume("req-resume")

    assert store.get_command("req-resume") is None


# ---------------------------------------------------------------------------
# acceptance: retry checks evidence; ambiguity/conflict/failure keeps the hold
# ---------------------------------------------------------------------------


def test_retry_clears_the_hold_and_finalizes_once(tmp_path: Path) -> None:
    checkpoint = start_checkpoint(tmp_path, AttemptPhase.PUSHING)
    assert checkpoint is not None
    lifecycle, tracker, _, publisher, store = make_lifecycle(tmp_path)

    record = lifecycle.submit_recovery_retry("req-retry", checkpoint.started_at)

    assert record.acknowledgement is CommandAcknowledgement.COMPLETED
    assert AttemptStateStore(tmp_path).read() is None
    assert AttemptCompletionStore(tmp_path).is_finalized(checkpoint.started_at)
    assert tracker.cleanup != []
    assert publisher.requests != []
    # idempotent retry with the same ID returns the terminal record.
    retry = lifecycle.submit_recovery_retry("req-retry", checkpoint.started_at)
    assert retry.sequence == record.sequence
    assert retry.acknowledgement is CommandAcknowledgement.COMPLETED
    assert len(AttemptCompletionStore(tmp_path).read_all()) == 1


def test_retry_rejects_unknown_mismatched_and_finalized_ids(tmp_path: Path) -> None:
    checkpoint = start_checkpoint(tmp_path)
    assert checkpoint is not None
    lifecycle, _, _, _, store = make_lifecycle(tmp_path)

    with pytest.raises(RecoveryRejectedError, match="does not match"):
        lifecycle.submit_recovery_retry("req-bad", "mismatched-id")
    with pytest.raises(RecoveryRejectedError, match="unknown attempt"):
        AgentLifecycle(
            tracker=FakeTracker(),
            attempt_state=AttemptStateStore(tmp_path.parent / "empty"),
            workspace=FakeWorkspace(),
            profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
            publisher=FakePublisher(),
            control_store=ControlStore(tmp_path.parent / "empty"),
            completion_store=AttemptCompletionStore(tmp_path.parent / "empty"),
            sleeper=lambda seconds: None,
        ).submit_recovery_retry("req-unknown", "no-such-attempt")

    lifecycle.submit_recovery_retry("req-ok", checkpoint.started_at)
    assert store.get_command("req-ok") is not None  # sanity: retry cleared it
    # A leftover checkpoint whose ID is already in the ledger (crash between
    # finalization and checkpoint removal) is finalized: rejected, never
    # re-executed. Simulate it with a fresh checkpoint + manual ledger entry.
    leftover = AttemptStateStore(tmp_path).start(issue_number=24, branch="agent/issue-24")
    AttemptCompletionStore(tmp_path).record(
        attempt_id=leftover.started_at,
        issue_number=24,
        branch="agent/issue-24",
        outcome=AttemptOutcome.INFRASTRUCTURE_ERROR,
    )
    with pytest.raises(RecoveryRejectedError, match="already finalized"):
        lifecycle.submit_recovery_retry("req-again", leftover.started_at)

    assert store.get_command("req-bad") is None
    assert store.get_command("req-again") is None


def test_retry_rejected_while_reconciliation_is_running(tmp_path: Path) -> None:
    checkpoint = start_checkpoint(tmp_path)
    assert checkpoint is not None
    lifecycle, tracker, _, _, store = make_lifecycle(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    original = tracker.recover_claim

    def blocking_recover(issue_number: int):
        entered.set()
        assert release.wait(timeout=30)
        return original(issue_number)

    tracker.recover_claim = blocking_recover  # type: ignore[method-assign]
    worker = threading.Thread(target=lifecycle.run_once)
    worker.start()
    assert entered.wait(timeout=30)
    try:
        with pytest.raises(RecoveryRejectedError, match="still running|concurrently"):
            lifecycle.submit_recovery_retry("req-race", checkpoint.started_at)
    finally:
        release.set()
        worker.join(timeout=30)
    assert store.get_command("req-race") is None


def test_retry_keeps_the_hold_on_ambiguous_remote_state(tmp_path: Path) -> None:
    checkpoint = start_checkpoint(tmp_path, AttemptPhase.PUSHING)
    assert checkpoint is not None

    class AmbiguousPublisher(FakePublisher):
        def publish(self, request: object):
            self.requests.append(request)
            return type(
                "Published", (), {"outcome": AttemptOutcome.INFRASTRUCTURE_ERROR, "branch_url": None, "comment_posted": False}
            )()

    lifecycle, _, _, _, store = make_lifecycle(tmp_path, publisher=AmbiguousPublisher())

    record = lifecycle.submit_recovery_retry("req-amb", checkpoint.started_at)

    assert record.acknowledgement is CommandAcknowledgement.NOT_FULFILLED
    assert "unconfirmed" in record.detail
    assert AttemptStateStore(tmp_path).read() is not None
    assert AttemptCompletionStore(tmp_path).read_all() == {}


def test_retry_keeps_the_hold_on_conflicting_human_changes(tmp_path: Path) -> None:
    checkpoint = start_checkpoint(tmp_path)
    assert checkpoint is not None

    class ConflictingTracker(FakeTracker):
        def fetch_issue(self, number: int):
            return _issue(number, assignees=("human",))

        def is_self_assigned(self, found: TrackerIssue) -> bool:
            return False

    lifecycle, _, _, publisher, _ = make_lifecycle(tmp_path, tracker=ConflictingTracker())

    record = lifecycle.submit_recovery_retry("req-conflict", checkpoint.started_at)

    assert record.acknowledgement is CommandAcknowledgement.NOT_FULFILLED
    assert "conflicting" in record.detail
    assert publisher.requests == []
    assert AttemptStateStore(tmp_path).read() is not None


def test_retry_never_invokes_the_model_or_renews_state(tmp_path: Path) -> None:
    checkpoint = start_checkpoint(tmp_path, AttemptPhase.MODEL_RUNNING)
    assert checkpoint is not None
    calls: list[object] = []

    def attempt_runner(claim: Claim, profile: object, prepared: object):
        calls.append(claim)
        raise AssertionError("retry must never invoke the model")

    lifecycle, _, _, publisher, _ = make_lifecycle(tmp_path)
    lifecycle.set_attempt_runner(attempt_runner)

    record = lifecycle.submit_recovery_retry("req-nomodel", checkpoint.started_at)

    assert calls == []
    assert record.acknowledgement is CommandAcknowledgement.COMPLETED
    assert publisher.requests != []


# ---------------------------------------------------------------------------
# acceptance: release records saved work, confirms comment, never claims handoff
# ---------------------------------------------------------------------------


def test_release_confirms_comment_before_labels_and_records_saved_at(tmp_path: Path) -> None:
    checkpoint = start_checkpoint(tmp_path, AttemptPhase.PUBLISHING)
    assert checkpoint is not None
    events: list[str] = []

    class OrderedTracker(FakeTracker):
        def release_attempt(self, issue_number: int, label: str, assignee_id: str) -> None:
            events.append("release")
            super().release_attempt(issue_number, label, assignee_id)

    class CommentVerifyingTransport:
        def __init__(self) -> None:
            self.present = False
            self.bodies: list[str] = []

        def find_attempt_comment(self, repository: str, issue_number: int, marker: str) -> bool:
            events.append("find")
            assert checkpoint.started_at in marker
            return self.present

        def add_comment(self, repository: str, issue_number: int, body: str) -> None:
            events.append("comment")
            assert checkpoint.started_at in body
            assert "/tmp/secured-work" in body
            self.bodies.append(body)
            self.present = True

    transport = CommentVerifyingTransport()

    class GithubPublisher(FakePublisher):
        _repository = "owner/repo"

        def __init__(self) -> None:
            super().__init__()
            self._github = transport

        def publish(self, request: object):  # pragma: no cover - release bypasses publish
            raise AssertionError("release must use the comment transport, not publish")

    tracker = OrderedTracker()
    lifecycle, _, _, _, store = make_lifecycle(
        tmp_path, tracker=tracker, publisher=GithubPublisher()
    )

    record = lifecycle.submit_recovery_release(
        "req-release", checkpoint.started_at, "/tmp/secured-work"
    )

    assert record.acknowledgement is CommandAcknowledgement.COMPLETED
    assert "/tmp/secured-work" in record.detail
    assert AttemptOutcome.INFRASTRUCTURE_ERROR.value in record.detail or "released" in record.detail
    assert events.index("comment") < events.index("release")
    assert tracker.cleanup == [(24, "ready-for-agent", "agent-id")]
    assert AttemptStateStore(tmp_path).read() is None
    assert AttemptCompletionStore(tmp_path).is_finalized(checkpoint.started_at)
    # The saved-work reference remains visible in the command record.
    stored = store.get_command("req-release")
    assert stored is not None
    assert "/tmp/secured-work" in stored.detail


def test_release_never_claims_a_successful_handoff(tmp_path: Path) -> None:
    checkpoint = start_checkpoint(tmp_path, AttemptPhase.PUBLISHING)
    assert checkpoint is not None
    bodies: list[str] = []

    class Transport:
        def __init__(self) -> None:
            self.present = False

        def find_attempt_comment(self, repository: str, issue_number: int, marker: str) -> bool:
            assert checkpoint.started_at in marker
            return self.present

        def add_comment(self, repository: str, issue_number: int, body: str) -> None:
            bodies.append(body)
            self.present = True

    class GithubPublisher(FakePublisher):
        _repository = "owner/repo"

        def __init__(self) -> None:
            super().__init__()
            self._github = Transport()

        def publish(self, request: object):  # pragma: no cover
            raise AssertionError("release must not publish a handoff")

    class HandoffTracker(FakeTracker):
        def release_handoff(self, issue_number: int, assignee_id: str) -> None:
            raise AssertionError("release must never take the handoff label path")

    lifecycle, tracker, _, _, _ = make_lifecycle(
        tmp_path, tracker=HandoffTracker(), publisher=GithubPublisher()
    )

    record = lifecycle.submit_recovery_release("req-rel", checkpoint.started_at, "s3://bucket/work")

    assert record.acknowledgement is CommandAcknowledgement.COMPLETED
    assert bodies, "a release comment must be posted"
    first_line = bodies[0].splitlines()[0]
    assert "handoff" not in first_line.lower() or "infrastructure_error" in first_line
    assert "not published as a success" in bodies[0]
    assert "does not" in bodies[0] or "not fulfilled" in bodies[0].lower() or "never" in bodies[0].lower()
    assert tracker.cleanup == [(24, "ready-for-agent", "agent-id")]
    assert "round-finished" not in str(tracker.cleanup)


def test_release_keeps_the_hold_when_the_comment_cannot_be_confirmed(tmp_path: Path) -> None:
    checkpoint = start_checkpoint(tmp_path)
    assert checkpoint is not None

    class FailingTransport:
        def find_attempt_comment(self, repository: str, issue_number: int, marker: str) -> bool:
            return False

        def add_comment(self, repository: str, issue_number: int, body: str) -> None:
            raise OSError("GitHub unavailable")

    class GithubPublisher(FakePublisher):
        _repository = "owner/repo"

        def __init__(self) -> None:
            super().__init__()
            self._github = FailingTransport()

        def publish(self, request: object):  # pragma: no cover
            raise AssertionError("release uses the comment transport")

    lifecycle, tracker, _, _, _ = make_lifecycle(tmp_path, publisher=GithubPublisher())

    record = lifecycle.submit_recovery_release("req-fail", checkpoint.started_at, "/tmp/work")

    assert record.acknowledgement is CommandAcknowledgement.NOT_FULFILLED
    assert tracker.cleanup == []
    assert AttemptStateStore(tmp_path).read() is not None
    assert AttemptCompletionStore(tmp_path).read_all() == {}


def test_release_rejects_invalid_saved_at_without_a_control_change(tmp_path: Path) -> None:
    checkpoint = start_checkpoint(tmp_path)
    assert checkpoint is not None
    lifecycle, _, _, _, store = make_lifecycle(tmp_path)

    with pytest.raises(RecoveryRejectedError, match="saved-at"):
        lifecycle.submit_recovery_release("req-bad", checkpoint.started_at, "  ")

    assert store.get_command("req-bad") is None
    assert AttemptStateStore(tmp_path).read() is not None


def test_release_does_not_requeue_the_issue(tmp_path: Path) -> None:
    checkpoint = start_checkpoint(tmp_path)
    assert checkpoint is not None
    lifecycle, tracker, _, _, _ = make_lifecycle(tmp_path)

    lifecycle.submit_recovery_release("req-rel", checkpoint.started_at, "/tmp/work")

    assert tracker.claimed == []


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_parse_attempt_id_and_saved_at_validation() -> None:
    assert parse_attempt_id("  attempt-1 ") == "attempt-1"
    with pytest.raises(RecoveryRejectedError):
        parse_attempt_id("  ")
    with pytest.raises(RecoveryRejectedError):
        parse_attempt_id("")
    assert parse_saved_at("/tmp/work") == "/tmp/work"
    with pytest.raises(RecoveryRejectedError):
        parse_saved_at("  ")


# ---------------------------------------------------------------------------
# socket + CLI
# ---------------------------------------------------------------------------


def test_recovery_commands_travel_over_the_live_socket(tmp_path: Path) -> None:
    from simple_coding_agent.control_server import ControlServer

    checkpoint = start_checkpoint(tmp_path, AttemptPhase.PUSHING)
    assert checkpoint is not None
    lifecycle, _, _, _, _ = make_lifecycle(tmp_path)
    socket_path = tmp_path / "run" / "control.sock"
    server = ControlServer(socket_path, lifecycle)
    server.start()
    try:
        from simple_coding_agent.agentctl import run_recovery_retry, run_status

        reply = run_recovery_retry(socket_path, "req-sock", checkpoint.started_at)
        assert reply["ok"] is True
        assert reply["acknowledgement"] == "completed"

        status = run_status(socket_path)
        assert status["recovery"] is None
    finally:
        server.stop()


def test_agentctl_recovery_cli_parses_retry_and_release(tmp_path: Path, capsys) -> None:
    from simple_coding_agent import agentctl
    from simple_coding_agent.control_server import ControlServer

    checkpoint = start_checkpoint(tmp_path)
    assert checkpoint is not None
    lifecycle, _, _, _, _ = make_lifecycle(tmp_path)
    socket_path = tmp_path / "run" / "control.sock"
    server = ControlServer(socket_path, lifecycle)
    server.start()
    try:
        code = agentctl.main(
            ["--socket", str(socket_path), "recovery", "retry", checkpoint.started_at]
        )
        assert code == 0
        assert "completed" in capsys.readouterr().out
    finally:
        server.stop()


def test_status_format_shows_recovery_hold(tmp_path: Path) -> None:
    from simple_coding_agent import agentctl

    checkpoint = start_checkpoint(tmp_path)
    assert checkpoint is not None
    lifecycle, _, _, _, _ = make_lifecycle(tmp_path)

    rendered = agentctl.format_status(lifecycle.control_status("owner/repo"))

    assert f"attempt {checkpoint.started_at}" in rendered
    assert "hold_reason" in rendered
    assert "next_action" in rendered
    assert "agentctl recovery retry" in rendered
