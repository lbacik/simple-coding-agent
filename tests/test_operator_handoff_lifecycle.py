"""Operator-handoff tests above the model boundary: runner, lifecycle, store, CLI."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_coding_agent.agentctl import build_parser, format_status, run_handoff
from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.completion import (
    AttemptOutcome,
    CompletionDecision,
    CompletionEvaluator,
    PublicationPath,
)
from simple_coding_agent.control import (
    CommandAcknowledgement,
    ControlStore,
    HandoffRejectedError,
    IntakeState,
    RequestIdError,
)
from simple_coding_agent.control_server import ControlServer, send_request
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.lifecycle import (
    AgentLifecycle,
    AttemptEvidence,
    LifecycleStatus,
    ModelAttemptRunner,
    OperatorHandoff,
)
from simple_coding_agent.model_execution import (
    ModelExecution,
    ModelExecutionStatus,
    SkillEvent,
)


ISSUE_NUMBER = 24
WORK_REV = "c" * 40
NOTE_REV = "a" * 40


def issue(number: int = ISSUE_NUMBER) -> TrackerIssue:
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


def claim(number: int = ISSUE_NUMBER) -> Claim:
    return Claim(issue(number), Assignment(f"issue-{number}", "agent-id"))


def profile() -> object:
    return SimpleNamespace(
        setup=(), check=("true",), base_branch="main", timeout=30, env={}
    )


def write_operator_note(working_directory: Path, *, reason: str = "operator_request") -> None:
    note_dir = working_directory / ".agent" / "handoff"
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / f"{ISSUE_NUMBER}.md").write_text(
        f"# Handoff note: issue #{ISSUE_NUMBER}\n\n"
        f"- issue: {ISSUE_NUMBER}\n"
        "- started_at: 2026-09-22T00:00:00Z\n"
        f"- reason: {reason}\n"
        f"- last_work_commit: {WORK_REV}\n\n"
        "## Summary\n\nHalfway done.\n\n"
        "## Remaining work\n\nFinish the parser.\n\n"
        "## Evidence\n\nParser tests.\n"
    )


def note_commits() -> tuple[object, object]:
    return (
        SimpleNamespace(subject=f"Handoff note: issue #{ISSUE_NUMBER}", revision=NOTE_REV),
        SimpleNamespace(subject="Half-finish the parser", revision=WORK_REV),
    )


def accepted_snapshot(request_id: str) -> dict:
    return {
        "request_id": request_id,
        "accepted_at": "2026-09-22T00:00:00Z",
        "begun": False,
        "phase": "accepted",
        "acknowledgement": "accepted",
        "detail": "handoff requested",
    }


# --- runner fakes ------------------------------------------------------------


class ScriptedOperatorExecutor:
    """An executor double that wires the operator provider/reporter like the real one."""

    def __init__(
        self,
        *,
        status: ModelExecutionStatus,
        deliver: bool = False,
        begun: bool = False,
    ) -> None:
        self._status = status
        self._deliver = deliver
        self._begun = begun
        self.provider = None
        self.reporter = None
        self.provider_calls = 0
        self._latched: str | None = None

    def set_operator_handoff_provider(self, provider) -> None:
        self.provider = provider

    def set_operator_handoff_reporter(self, reporter) -> None:
        self.reporter = reporter

    @property
    def operator_request_id(self) -> str | None:
        return self._latched

    async def execute(self, **kwargs: object) -> ModelExecution:
        request = None
        if self.provider is not None:
            self.provider_calls += 1
            request = self.provider()
        if request is not None:
            self._latched = request.request_id
            if self._deliver and self.reporter is not None:
                self.reporter("delivered", request.request_id)
            if self._begun and self.reporter is not None:
                self.reporter("begun", request.request_id)
        return ModelExecution(
            status=self._status,
            explanation="scripted",
            stop_reason="end_turn",
            model_usage=None,
            observed_models=(),
            skill_events=(
                SkillEvent(
                    phase="PreToolUse",
                    name="handoff",
                    agent_id=None,
                    timestamp="2026-09-22T00:00:00Z",
                ),
            )
            if self._begun
            else (),
        )


class FakeOperatorControl:
    def __init__(self, snapshot: dict | None = None, *, hide_while_executing: bool = False) -> None:
        self._snapshot = snapshot
        self.hide_while_executing = hide_while_executing
        self.executing = False
        self.marks: list[tuple[str, str]] = []
        self.completed: list[tuple[str, str]] = []
        self.failed: list[tuple[str, str]] = []

    def snapshot(self) -> dict | None:
        if self.hide_while_executing and self.executing:
            return None
        return dict(self._snapshot) if self._snapshot is not None else None

    def mark_delivering(self, request_id: str) -> bool:
        self.marks.append(("delivering", request_id))
        return True

    def mark_begun(self, request_id: str) -> bool:
        self.marks.append(("begun", request_id))
        return True

    def complete(self, request_id: str, detail: str) -> None:
        self.completed.append((request_id, detail))

    def fail(self, request_id: str, reason: str) -> None:
        self.failed.append((request_id, reason))


class RunnerWorkspace:
    def __init__(self, working_directory: Path, commits: tuple[object, ...] = ()) -> None:
        self._working_directory = working_directory
        self._commits = commits
        self.preserved: list[str] = []

    @property
    def working_directory(self) -> Path:
        return self._working_directory

    def commits_added(self, prepared: object) -> tuple[object, ...]:
        return self._commits

    def has_unresolved_conflicts(self) -> bool:
        return False

    def commit_dirty_work(self, message: str) -> None:
        self.preserved.append(message)


class RunnerVerifier:
    def prepare(self, profile: object, working_directory: Path, **kwargs: object):
        setup = SimpleNamespace(succeeded=True, commands=())
        baseline = SimpleNamespace(succeeded=True, commands=(), exit_code=0)
        return SimpleNamespace(setup=setup, baseline=baseline)

    def mark_review_complete(self, **kwargs: object) -> None:
        return None

    def final_check(self, profile: object, working_directory: Path):
        return SimpleNamespace(succeeded=True, commands=(), exit_code=0)


def build_runner(
    tmp_path: Path,
    *,
    executor: ScriptedOperatorExecutor,
    workspace: RunnerWorkspace,
    operator: FakeOperatorControl | None,
) -> ModelAttemptRunner:
    attempt_state = AttemptStateStore(tmp_path)
    attempt_state.start(issue_number=ISSUE_NUMBER, branch=f"agent/issue-{ISSUE_NUMBER}")
    attempt_state.transition(AttemptPhase.SETUP)
    return ModelAttemptRunner(
        attempt_state=attempt_state,
        workspace=workspace,
        verifier=RunnerVerifier(),
        evaluator=CompletionEvaluator(frozenset()),
        model_executor=executor,
        operator_handoff=operator,
    )


def prepared() -> object:
    return SimpleNamespace(
        branch=f"agent/issue-{ISSUE_NUMBER}", base_revision="b" * 40
    )


# --- runner: served requests -------------------------------------------------


def test_setup_phase_request_waits_for_and_is_served_by_model_execution(
    tmp_path: Path,
) -> None:
    working_directory = tmp_path / "work"
    working_directory.mkdir()
    write_operator_note(working_directory)
    executor = ScriptedOperatorExecutor(
        status=ModelExecutionStatus.HANDOFF_REQUESTED, deliver=True, begun=True
    )
    operator = FakeOperatorControl(accepted_snapshot("req-1"))
    runner = build_runner(
        tmp_path,
        executor=executor,
        workspace=RunnerWorkspace(working_directory, note_commits()),
        operator=operator,
    )

    evidence = runner(claim(), profile(), prepared())

    assert executor.provider is not None
    assert executor.provider_calls >= 1
    assert executor.operator_request_id == "req-1"
    assert operator.marks == [("delivering", "req-1"), ("begun", "req-1")]
    assert evidence.decision.outcome is AttemptOutcome.HANDOFF
    # Completion waits for the durable publication boundary, not the runner.
    assert operator.completed == []
    assert operator.failed == []


def test_verification_phase_request_keeps_the_ordinary_outcome_and_is_not_fulfilled(
    tmp_path: Path,
) -> None:
    working_directory = tmp_path / "work"
    working_directory.mkdir()
    executor = ScriptedOperatorExecutor(status=ModelExecutionStatus.SUCCEEDED)
    operator = FakeOperatorControl(
        accepted_snapshot("req-late"), hide_while_executing=True
    )
    runner = build_runner(
        tmp_path,
        executor=executor,
        workspace=RunnerWorkspace(
            working_directory,
            (SimpleNamespace(subject="Implement", revision=WORK_REV),),
        ),
        operator=operator,
    )
    original_execute = executor.execute

    async def execute(**kwargs: object) -> ModelExecution:
        operator.executing = True
        try:
            return await original_execute(**kwargs)
        finally:
            operator.executing = False

    executor.execute = execute  # type: ignore[method-assign]

    evidence = runner(claim(), profile(), prepared())

    assert executor.operator_request_id is None
    assert evidence.decision.outcome is AttemptOutcome.INCOMPLETE
    assert len(operator.failed) == 1
    request_id, reason = operator.failed[0]
    assert request_id == "req-late"
    assert "incomplete" in reason


def test_served_handoff_with_an_invalid_note_is_incomplete_not_infrastructure_error(
    tmp_path: Path,
) -> None:
    working_directory = tmp_path / "work"
    working_directory.mkdir()
    write_operator_note(working_directory, reason="bogus_reason")
    executor = ScriptedOperatorExecutor(
        status=ModelExecutionStatus.HANDOFF_REQUESTED, deliver=True, begun=True
    )
    operator = FakeOperatorControl(accepted_snapshot("req-1"))
    runner = build_runner(
        tmp_path,
        executor=executor,
        workspace=RunnerWorkspace(working_directory, note_commits()),
        operator=operator,
    )

    evidence = runner(claim(), profile(), prepared())

    assert evidence.decision.outcome is AttemptOutcome.INCOMPLETE
    assert evidence.decision.publication_path is PublicationPath.PARTIAL
    assert len(operator.failed) == 1
    assert operator.failed[0][0] == "req-1"


def test_served_handoff_with_a_wrong_reason_still_publishes_but_fails_the_request(
    tmp_path: Path,
) -> None:
    working_directory = tmp_path / "work"
    working_directory.mkdir()
    write_operator_note(working_directory, reason="cost_soft_threshold")
    executor = ScriptedOperatorExecutor(
        status=ModelExecutionStatus.HANDOFF_REQUESTED, deliver=True, begun=True
    )
    operator = FakeOperatorControl(accepted_snapshot("req-1"))
    runner = build_runner(
        tmp_path,
        executor=executor,
        workspace=RunnerWorkspace(working_directory, note_commits()),
        operator=operator,
    )

    evidence = runner(claim(), profile(), prepared())

    assert evidence.decision.outcome is AttemptOutcome.HANDOFF
    assert len(operator.failed) == 1
    assert operator.failed[0][0] == "req-1"
    assert "operator_request" in operator.failed[0][1]


def test_served_handoff_without_handoff_outcome_fails_with_the_ordinary_outcome(
    tmp_path: Path,
) -> None:
    working_directory = tmp_path / "work"
    working_directory.mkdir()
    executor = ScriptedOperatorExecutor(
        status=ModelExecutionStatus.OPERATOR_HANDOFF_EXPIRED, deliver=True, begun=False
    )
    operator = FakeOperatorControl(accepted_snapshot("req-1"))
    runner = build_runner(
        tmp_path,
        executor=executor,
        workspace=RunnerWorkspace(
            working_directory,
            (SimpleNamespace(subject="Implement", revision=WORK_REV),),
        ),
        operator=operator,
    )

    evidence = runner(claim(), profile(), prepared())

    assert evidence.decision.outcome is AttemptOutcome.INCOMPLETE
    assert len(operator.failed) == 1
    assert operator.failed[0][0] == "req-1"


def test_runner_without_operator_control_behaves_as_before(tmp_path: Path) -> None:
    working_directory = tmp_path / "work"
    working_directory.mkdir()
    executor = ScriptedOperatorExecutor(status=ModelExecutionStatus.SUCCEEDED)
    runner = build_runner(
        tmp_path,
        executor=executor,
        workspace=RunnerWorkspace(working_directory, ()),
        operator=None,
    )

    evidence = runner(claim(), profile(), prepared())

    assert evidence.decision.outcome is AttemptOutcome.NO_CHANGES
    assert executor.provider is None


# --- lifecycle fakes ---------------------------------------------------------


class LifecycleTracker:
    def __init__(self, next_claim: Claim | None) -> None:
        self.next_claim = next_claim
        self.claimed: list[int] = []
        self.cleanup: list[tuple[int, str, str]] = []

    def claim_next(self) -> Claim | None:
        claim_value, self.next_claim = self.next_claim, None
        if claim_value is not None:
            self.claimed.append(claim_value.issue.number)
        return claim_value

    def recover_claim(self, issue_number: int) -> Claim | None:
        return Claim(issue(issue_number), Assignment(f"issue-{issue_number}", "agent-id"))

    def release_attempt(self, issue_number: int, label: str, assignee_id: str) -> None:
        self.cleanup.append((issue_number, label, assignee_id))

    def release_handoff(self, issue_number: int, assignee_id: str) -> None:
        self.cleanup.append((issue_number, "round-finished", assignee_id))


class LifecycleWorkspace:
    def __init__(self, working_directory: Path, commits: tuple[object, ...] = ()) -> None:
        self._working_directory = working_directory
        self._commits = commits
        self.cleanup_calls: list[tuple[str, bool]] = []

    @property
    def working_directory(self) -> Path:
        return self._working_directory

    def is_clean(self) -> bool:
        return True

    def prepare_for_profile_read(self, *, base_branch: str = "main") -> None:
        return None

    def prepare_attempt(self, *, base_branch: str, issue_number: int):
        return SimpleNamespace(
            branch=f"agent/issue-{issue_number}", base_revision="b" * 40
        )

    def commits_added(self, prepared: object) -> tuple[object, ...]:
        return self._commits

    def cleanup(self, *, base_branch: str, prepared: object, retain_branch: bool) -> None:
        self.cleanup_calls.append((base_branch, retain_branch))


class LifecyclePublisher:
    def __init__(self) -> None:
        self.requests: list[object] = []

    def publish(self, request: object):
        self.requests.append(request)
        outcome = request.decision.outcome or AttemptOutcome.COMPLETE
        return SimpleNamespace(outcome=outcome, branch_url="https://example.test/x")


def make_lifecycle(
    tmp_path: Path,
    *,
    next_claim: Claim | None,
    workspace: LifecycleWorkspace,
    attempt_runner,
):
    tracker = LifecycleTracker(next_claim)
    store = ControlStore(tmp_path)
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: SimpleNamespace(base_branch="main"),
        publisher=LifecyclePublisher(),
        attempt_runner=attempt_runner,
        control_store=store,
        sleeper=lambda seconds: None,
    )
    return lifecycle, tracker, store


def handoff_evidence() -> AttemptEvidence:
    return AttemptEvidence(
        decision=CompletionDecision(
            AttemptOutcome.HANDOFF, False, PublicationPath.PARTIAL, ("handoff note",)
        ),
        check_command="not run",
        check_exit_code=None,
        review_cycles=0,
        review_findings="not run",
        details="handoff note",
        note_commit_sha=NOTE_REV,
    )


# --- lifecycle: submit + finalize --------------------------------------------


def test_handoff_without_an_active_attempt_is_rejected(tmp_path: Path) -> None:
    workspace = LifecycleWorkspace(tmp_path / "work")
    lifecycle, _, store = make_lifecycle(
        tmp_path, next_claim=None, workspace=workspace, attempt_runner=lambda c, p, r: None
    )

    with pytest.raises(HandoffRejectedError):
        lifecycle.submit_handoff("req-1")

    assert store.intake_state() is IntakeState.RUNNING
    assert store.handoff_snapshot() is None


def test_unserved_handoff_finalizes_the_ordinary_outcome_as_not_fulfilled(
    tmp_path: Path,
) -> None:
    working_directory = tmp_path / "work"
    working_directory.mkdir()
    workspace = LifecycleWorkspace(working_directory)
    started = threading.Event()
    release = threading.Event()

    def blocking_workflow(received_claim: Claim, profile: object, prepared: object):
        started.set()
        assert release.wait(timeout=30)
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

    lifecycle, tracker, store = make_lifecycle(
        tmp_path, next_claim=claim(), workspace=workspace, attempt_runner=blocking_workflow
    )
    worker = threading.Thread(target=lifecycle.run_once)
    worker.start()
    assert started.wait(timeout=30)

    record = lifecycle.submit_handoff("req-handoff")
    assert record.acknowledgement is CommandAcknowledgement.ACCEPTED
    assert store.intake_state() is IntakeState.STOPPING
    release.set()
    worker.join(timeout=30)

    stored = store.get_command("req-handoff")
    assert stored is not None
    assert stored.acknowledgement is CommandAcknowledgement.NOT_FULFILLED
    assert store.intake_state() is IntakeState.STOPPED
    assert tracker.cleanup == [(ISSUE_NUMBER, "ready-for-agent", "agent-id")]


def test_served_valid_handoff_completes_after_durable_finalization(
    tmp_path: Path,
) -> None:
    working_directory = tmp_path / "work"
    working_directory.mkdir()
    write_operator_note(working_directory)
    workspace = LifecycleWorkspace(working_directory, note_commits())
    started = threading.Event()
    release = threading.Event()

    def blocking_workflow(received_claim: Claim, profile: object, prepared: object):
        started.set()
        assert release.wait(timeout=30)
        return handoff_evidence()

    lifecycle, tracker, store = make_lifecycle(
        tmp_path, next_claim=claim(), workspace=workspace, attempt_runner=blocking_workflow
    )
    worker = threading.Thread(target=lifecycle.run_once)
    worker.start()
    assert started.wait(timeout=30)

    lifecycle.submit_handoff("req-handoff")
    # The model observed the handoff skill before finalization.
    assert store.mark_handoff_begun("req-handoff") is True
    release.set()
    worker.join(timeout=30)

    stored = store.get_command("req-handoff")
    assert stored is not None
    assert stored.acknowledgement is CommandAcknowledgement.COMPLETED
    assert store.intake_state() is IntakeState.STOPPED
    assert tracker.cleanup == [(ISSUE_NUMBER, "round-finished", "agent-id")]


def test_orphan_handoff_with_no_checkpoint_fails_and_stops_intake(
    tmp_path: Path,
) -> None:
    workspace = LifecycleWorkspace(tmp_path / "work")
    lifecycle, _, store = make_lifecycle(
        tmp_path, next_claim=None, workspace=workspace, attempt_runner=lambda c, p, r: None
    )
    attempt_state = AttemptStateStore(tmp_path)
    attempt_state.start(issue_number=ISSUE_NUMBER, branch=f"agent/issue-{ISSUE_NUMBER}")
    lifecycle.submit_handoff("req-handoff")
    assert store.intake_state() is IntakeState.STOPPING
    attempt_state.delete()

    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.IDLE
    stored = store.get_command("req-handoff")
    assert stored is not None
    assert stored.acknowledgement is CommandAcknowledgement.NOT_FULFILLED
    assert store.intake_state() is IntakeState.STOPPED


def test_resume_after_a_fulfilled_handoff_permits_intake_without_requeueing(
    tmp_path: Path,
) -> None:
    workspace = LifecycleWorkspace(tmp_path / "work")
    lifecycle, tracker, store = make_lifecycle(
        tmp_path, next_claim=None, workspace=workspace, attempt_runner=lambda c, p, r: None
    )
    attempt_state = AttemptStateStore(tmp_path)
    attempt_state.start(issue_number=ISSUE_NUMBER, branch=f"agent/issue-{ISSUE_NUMBER}")
    lifecycle.submit_handoff("req-handoff")
    store.complete_handoff("req-handoff", "operator handoff fulfilled for issue #24")
    attempt_state.delete()

    record = lifecycle.submit_resume("req-resume")

    assert record.acknowledgement is CommandAcknowledgement.COMPLETED
    assert store.intake_state() is IntakeState.RUNNING
    assert lifecycle.run_once().status is LifecycleStatus.IDLE
    assert tracker.claimed == []


def test_status_and_format_include_the_tracked_handoff(tmp_path: Path) -> None:
    workspace = LifecycleWorkspace(tmp_path / "work")
    started = threading.Event()
    release = threading.Event()

    def blocking_workflow(received_claim: Claim, profile: object, prepared: object):
        started.set()
        assert release.wait(timeout=30)
        return handoff_evidence()

    lifecycle, _, store = make_lifecycle(
        tmp_path, next_claim=claim(), workspace=workspace, attempt_runner=blocking_workflow
    )
    worker = threading.Thread(target=lifecycle.run_once)
    worker.start()
    assert started.wait(timeout=30)

    lifecycle.submit_handoff("req-handoff")
    try:
        snapshot = lifecycle.control_status()
        assert snapshot["handoff"] is not None
        assert snapshot["handoff"]["request_id"] == "req-handoff"
        assert snapshot["handoff"]["acknowledgement"] == "accepted"
        text = format_status(snapshot)
        assert "req-handoff" in text
    finally:
        release.set()
        worker.join(timeout=30)


# --- store -------------------------------------------------------------------


def test_store_rejects_handoff_without_an_active_attempt(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)

    with pytest.raises(HandoffRejectedError):
        store.submit_handoff("req-1", has_active_attempt=False)

    assert store.intake_state() is IntakeState.RUNNING
    assert store.pending_command_id() is None
    assert store.handoff_snapshot() is None


def test_store_handoff_replaces_the_pending_stop_plan(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop("req-stop", has_active_attempt=True)

    record = store.submit_handoff("req-handoff", has_active_attempt=True)

    assert record.acknowledgement is CommandAcknowledgement.ACCEPTED
    assert store.pending_command_id() == "req-handoff"
    assert store.get_command("req-stop").acknowledgement is CommandAcknowledgement.SUPERSEDED
    snapshot = store.handoff_snapshot()
    assert snapshot is not None
    assert snapshot["request_id"] == "req-handoff"
    assert snapshot["begun"] is False
    assert snapshot["phase"] == "accepted"


def test_store_second_handoff_after_begun_is_rejected(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_handoff("req-1", has_active_attempt=True)
    assert store.mark_handoff_begun("req-1") is True

    with pytest.raises(HandoffRejectedError):
        store.submit_handoff("req-2", has_active_attempt=True)

    assert store.pending_command_id() == "req-1"


def test_store_stop_replaces_a_handoff_only_before_it_begins(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_handoff("req-handoff", has_active_attempt=True)

    store.submit_stop("req-stop", has_active_attempt=True)

    assert store.pending_command_id() == "req-stop"

    store2 = ControlStore(tmp_path / "second")
    store2.submit_handoff("req-handoff", has_active_attempt=True)
    assert store2.mark_handoff_begun("req-handoff") is True
    with pytest.raises(HandoffRejectedError):
        store2.submit_handoff("req-other", has_active_attempt=True)


def test_store_complete_and_fail_close_intake(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_handoff("req-1", has_active_attempt=True)

    completed = store.complete_handoff("req-1", "note published for issue #24")

    assert completed is not None
    assert completed.acknowledgement is CommandAcknowledgement.COMPLETED
    assert store.intake_state() is IntakeState.STOPPED
    assert store.handoff_snapshot()["phase"] == "completed"

    store = ControlStore(tmp_path / "second")
    store.submit_handoff("req-2", has_active_attempt=True)
    failed = store.fail_handoff("req-2", "model outcome incomplete without a note")

    assert failed is not None
    assert failed.acknowledgement is CommandAcknowledgement.NOT_FULFILLED
    assert store.intake_state() is IntakeState.STOPPED
    assert store.handoff_snapshot()["phase"] == "not_fulfilled"


def test_store_handoff_retry_with_the_identical_payload_is_idempotent(
    tmp_path: Path,
) -> None:
    store = ControlStore(tmp_path)
    first = store.submit_handoff("req-1", has_active_attempt=True)

    second = store.submit_handoff("req-1", has_active_attempt=True)

    assert second.sequence == first.sequence
    assert second.acknowledgement is CommandAcknowledgement.ACCEPTED


# --- CLI + socket -------------------------------------------------------------


def test_handoff_now_parses_with_an_optional_request_id() -> None:
    args = build_parser().parse_args(["handoff", "now"])

    assert args.command == "handoff"
    assert args.handoff_command == "now"
    assert args.request_id is None

    args = build_parser().parse_args(
        ["handoff", "now", "--request-id", "req-1"]
    )
    assert args.request_id == "req-1"


class StubControl:
    def __init__(self) -> None:
        self.repository = "octo/example"
        self.submitted: list[str] = []
        self.status_snapshot = {"repository": "octo/example", "handoff": None}

    def submit_handoff(self, request_id: str):
        if not isinstance(request_id, str) or not request_id:
            raise RequestIdError("A request ID is required.")
        self.submitted.append(request_id)
        if request_id == "req-bad":
            raise HandoffRejectedError("handoff now is rejected without an active attempt")
        return SimpleNamespace(
            request_id=request_id,
            sequence=1,
            kind="handoff",
            acknowledgement=SimpleNamespace(value="accepted"),
            detail="handoff requested",
            created_at="2026-09-25T00:00:00Z",
            updated_at="2026-09-25T00:00:00Z",
        )

    def control_status(self, repository=None) -> dict:
        return dict(self.status_snapshot)


def test_handoff_roundtrip_over_the_socket(tmp_path: Path) -> None:
    socket_path = tmp_path / "control.sock"
    server = ControlServer(socket_path, StubControl())
    server.start()
    try:
        reply = run_handoff(socket_path, "req-1")
        assert reply["ok"] is True
        assert reply["request_id"] == "req-1"

        rejected = run_handoff(socket_path, "req-bad")
        assert rejected["ok"] is False
        assert "rejected" in rejected["error"]
    finally:
        server.stop()


def test_unknown_handoff_request_lookup_reports_no_command(tmp_path: Path) -> None:
    socket_path = tmp_path / "control.sock"
    server = ControlServer(socket_path, StubControl())
    server.start()
    try:
        reply = send_request(socket_path, {"op": "handoff"})
        assert reply["ok"] is False
    finally:
        server.stop()


def test_format_status_renders_a_tracked_handoff() -> None:
    text = format_status(
        {
            "repository": "octo/example",
            "intake": "stopping",
            "active_attempt": None,
            "pending_command": None,
            "stop_plan": None,
            "pending_next_issue": None,
            "recovery": None,
            "handoff": {
                "request_id": "req-1",
                "accepted_at": "2026-09-22T00:00:00Z",
                "begun": True,
                "phase": "model_work",
                "acknowledgement": "accepted",
                "detail": "handoff requested",
            },
            "commands": {},
        }
    )

    assert "req-1" in text
    assert "model_work" in text
