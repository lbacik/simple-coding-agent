"""Exercise attempt finalization through its interface with real state stores."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.completion import AttemptOutcome
from simple_coding_agent.finalization import (
    AttemptCompletionStore,
    AttemptFinalizer,
    FinalizationEffects,
    FinalizationMode,
    FinalizationStage,
    Finalized,
    FinalizeRequest,
    Held,
    HeldReason,
    PublicationState,
)
from simple_coding_agent.git_workspace import DirtyWorkspaceError, PreparedAttempt

STARTED = datetime(2026, 9, 20, 13, tzinfo=UTC)
PREPARED = PreparedAttempt("agent/issue-24-prepared", "a" * 40)
CHECKPOINT_BRANCH = "agent/issue-24-restored"
REQUEST = FinalizeRequest(
    outcome=AttemptOutcome.COMPLETE,
    publication=PublicationState(confirmed=True, started_at=STARTED),
    prepared=PREPARED,
    base_branch="develop",
)


class Effects:
    def __init__(self, root: Path) -> None:
        self.state = AttemptStateStore(root, clock=lambda: STARTED)
        self.ledger = AttemptCompletionStore(root)
        self.checkpoint = self.state.start(issue_number=24, branch=CHECKPOINT_BRANCH)
        self.calls: list[str] = []
        self.handoff_checks: list[tuple[PreparedAttempt | None, datetime | None]] = []
        self.handoff_outcomes: list[AttemptOutcome] = []
        self.control_outcomes: list[AttemptOutcome] = []
        self.cleanup_args: list[
            tuple[PreparedAttempt, str, bool, FinalizationMode]
        ] = []
        self.held_messages: list[str] = []
        self.resolution_errors: list[Exception] = []
        self.hold_before = False
        self.hold_after = False
        self.handoff_failed = False
        self.release_error: Exception | None = None
        self.cleanup_error: Exception | None = None
        self.accounting_error: BaseException | None = None
        self.finish_error: Exception | None = None
        self.handoff_error: Exception | None = None

    def finalizer(self) -> AttemptFinalizer:
        return AttemptFinalizer(
            self.checkpoint,
            self.ledger,
            self.state,
            FinalizationEffects(
                release=self.release,
                cleanup=self.cleanup,
                check_handoff=self.check_handoff,
                resolve_handoff=self.resolve_handoff,
                write_outcome=self.write_outcome,
                finish_completion=self.finish_completion,
                record_control=self.record_control,
                handoff_owned=lambda: self.handoff_failed,
                finish_replay=lambda: self.calls.append("seal after replay"),
                workspace_held=lambda error, mode: self.calls.append("dirty"),
                report_hold=self.held_messages.append,
                resolution_unavailable=self.resolution_errors.append,
            ),
        )

    def release(self, outcome: AttemptOutcome, mode: FinalizationMode) -> None:
        self.calls.append("release")
        if self.release_error is not None:
            raise self.release_error

    def cleanup(
        self,
        prepared: PreparedAttempt,
        base_branch: str,
        retain: bool,
        mode: FinalizationMode,
    ) -> None:
        self.calls.append("cleanup")
        self.cleanup_args.append((prepared, base_branch, retain, mode))
        if self.cleanup_error is not None:
            raise self.cleanup_error

    def check_handoff(
        self, *, outcome, prepared, publication_started_at, check_allowance
    ) -> bool:
        self.calls.append(
            "pre-release hold" if check_allowance else "post-cleanup hold"
        )
        self.handoff_checks.append((prepared, publication_started_at))
        return self.hold_before if check_allowance else self.hold_after

    def resolve_handoff(
        self, *, outcome, prepared, publication_started_at, check_allowance
    ) -> bool:
        self.calls.append("resolve handoff")
        self.handoff_outcomes.append(outcome)
        if self.handoff_error is not None:
            raise self.handoff_error
        return not self.handoff_failed

    def write_outcome(self, outcome: AttemptOutcome) -> str:
        self.calls.append("write outcome")
        assert self.ledger.get(self.checkpoint.started_at) is None
        if self.accounting_error is not None:
            raise self.accounting_error
        return self.checkpoint.started_at

    def finish_completion(
        self, outcome: AttemptOutcome, completed_at: str, released: bool
    ) -> None:
        self.calls.append("finish release" if released else "finish completion")
        assert completed_at == self.checkpoint.started_at
        assert self.ledger.get(self.checkpoint.started_at).outcome is outcome
        if self.finish_error is not None:
            raise self.finish_error
        self.record_control(outcome)
        self.state.delete()

    def record_control(self, outcome: AttemptOutcome) -> None:
        self.calls.append("record control")
        self.control_outcomes.append(outcome)


def test_finalizes_in_order_and_checks_only_the_first_publication_allowance(
    tmp_path: Path,
) -> None:
    effects = Effects(tmp_path)

    assert effects.finalizer().finalize(REQUEST) == Finalized(AttemptOutcome.COMPLETE)

    assert effects.calls == [
        "pre-release hold",
        "release",
        "cleanup",
        "post-cleanup hold",
        "write outcome",
        "finish completion",
        "record control",
        "resolve handoff",
    ]
    assert effects.handoff_checks == [(PREPARED, STARTED), (None, STARTED)]
    assert effects.cleanup_args == [
        (PREPARED, "develop", False, FinalizationMode.NORMAL)
    ]
    assert effects.ledger.get(effects.checkpoint.started_at).branch == PREPARED.branch
    assert effects.state.read() is None


@pytest.mark.parametrize(
    ("failure", "reason", "stage", "calls"),
    [
        (
            "hold_before",
            HeldReason.HANDOFF_NOT_FULFILLED,
            FinalizationStage.BEFORE_RELEASE,
            ["pre-release hold"],
        ),
        (
            "release_error",
            HeldReason.RELEASE_FAILED,
            FinalizationStage.RELEASE,
            ["pre-release hold", "release", "cleanup"],
        ),
        (
            "cleanup_error",
            HeldReason.CLEANUP_FAILED,
            FinalizationStage.CLEANUP,
            ["pre-release hold", "release", "cleanup", "dirty"],
        ),
        (
            "hold_after",
            HeldReason.HANDOFF_NOT_FULFILLED,
            FinalizationStage.AFTER_CLEANUP,
            ["pre-release hold", "release", "cleanup", "post-cleanup hold"],
        ),
        (
            "accounting_error",
            HeldReason.ACCOUNTING_FAILED,
            FinalizationStage.ACCOUNTING,
            [
                "pre-release hold",
                "release",
                "cleanup",
                "post-cleanup hold",
                "write outcome",
            ],
        ),
    ],
)
def test_a_hold_preserves_the_checkpoint_and_skips_the_ledger(
    tmp_path: Path,
    failure: str,
    reason: HeldReason,
    stage: FinalizationStage,
    calls: list[str],
) -> None:
    effects = Effects(tmp_path)
    failures = {
        "hold_before": True,
        "release_error": RuntimeError("release failed"),
        "cleanup_error": DirtyWorkspaceError("dirty tree"),
        "hold_after": True,
        "accounting_error": RuntimeError("accounting unavailable"),
    }
    setattr(effects, failure, failures[failure])

    result = effects.finalizer().finalize(REQUEST)

    assert isinstance(result, Held)
    assert (result.reason, result.stage) == (reason, stage)
    assert effects.calls == calls
    assert effects.held_messages
    assert effects.state.read() == effects.checkpoint
    assert effects.ledger.read_all() == {}


def test_unconfirmed_publication_cleans_up_but_retains_the_branch(
    tmp_path: Path,
) -> None:
    effects = Effects(tmp_path)
    request = replace(
        REQUEST, publication=PublicationState(confirmed=False, started_at=STARTED)
    )

    result = effects.finalizer().finalize(request)

    assert isinstance(result, Held)
    assert result.reason is HeldReason.PUBLICATION_UNCONFIRMED
    assert effects.calls == ["cleanup"]
    assert effects.cleanup_args == [
        (PREPARED, "develop", True, FinalizationMode.NORMAL)
    ]
    assert effects.state.read() == effects.checkpoint
    assert effects.ledger.read_all() == {}


@pytest.mark.parametrize(
    ("mode", "retains_branch", "finish"),
    [
        (FinalizationMode.RECOVERY_RETRY, False, "finish completion"),
        (FinalizationMode.RECOVERY_RELEASE, True, "finish release"),
    ],
)
def test_operator_recovery_bypasses_handoff_holds(
    tmp_path: Path, mode: FinalizationMode, retains_branch: bool, finish: str
) -> None:
    effects = Effects(tmp_path)
    effects.hold_before = effects.hold_after = True

    assert effects.finalizer().finalize(replace(REQUEST, mode=mode)) == Finalized(
        REQUEST.outcome
    )

    assert effects.calls == [
        "release",
        "cleanup",
        "write outcome",
        finish,
        "record control",
        "resolve handoff",
    ]
    assert effects.cleanup_args == [(PREPARED, "develop", retains_branch, mode)]
    assert effects.state.read() is None


@pytest.mark.parametrize(
    "mode", [FinalizationMode.RECOVERY_RETRY, FinalizationMode.RECOVERY_RELEASE]
)
def test_operator_release_failure_stops_before_cleanup(
    tmp_path: Path, mode: FinalizationMode
) -> None:
    effects = Effects(tmp_path)
    effects.release_error = RuntimeError("release failed")

    result = effects.finalizer().finalize(replace(REQUEST, mode=mode))

    assert isinstance(result, Held)
    assert result.reason is HeldReason.RELEASE_FAILED
    assert effects.calls == ["release"]
    assert effects.state.read() == effects.checkpoint
    assert effects.ledger.read_all() == {}


def test_no_prepared_workspace_records_the_durable_checkpoint_branch(
    tmp_path: Path,
) -> None:
    effects = Effects(tmp_path)
    outcome = AttemptOutcome.INCOMPLETE
    request = replace(
        REQUEST, prepared=None, outcome=outcome, publication=PublicationState(True)
    )

    assert effects.finalizer().finalize(request) == Finalized(outcome)

    assert "cleanup" not in effects.calls
    assert effects.handoff_checks == [(None, None), (None, None)]
    entry = effects.ledger.get(effects.checkpoint.started_at)
    assert (entry.branch, entry.outcome) == (CHECKPOINT_BRANCH, outcome)


def test_crash_after_ledger_write_replays_only_the_recorded_completion(
    tmp_path: Path,
) -> None:
    effects = Effects(tmp_path)
    effects.finish_error = RuntimeError("control temporarily unavailable")
    initial = effects.finalizer().finalize(REQUEST)
    assert isinstance(initial, Held)
    assert initial.reason is HeldReason.ACCOUNTING_FAILED
    entry = effects.ledger.get(effects.checkpoint.started_at)
    assert entry is not None
    assert effects.state.read() == effects.checkpoint
    effects.calls.clear()
    request = replace(REQUEST, outcome=AttemptOutcome.INFRASTRUCTURE_ERROR)

    assert effects.finalizer().finalize(request) == Finalized(REQUEST.outcome)

    assert effects.calls == ["record control", "resolve handoff", "seal after replay"]
    assert effects.control_outcomes == [REQUEST.outcome]
    assert effects.handoff_outcomes == [REQUEST.outcome]
    assert effects.ledger.get(effects.checkpoint.started_at) == entry
    assert effects.state.read() is None


def test_replaying_a_completed_attempt_preserves_the_next_checkpoint(tmp_path: Path) -> None:
    effects = Effects(tmp_path)
    finalizer = effects.finalizer()
    assert finalizer.finalize(REQUEST) == Finalized(REQUEST.outcome)
    subsequent = effects.state.start(issue_number=25, branch="agent/issue-25")
    effects.calls.clear()

    assert finalizer.replay() == Finalized(REQUEST.outcome)

    assert effects.calls == []
    assert effects.state.read() == subsequent


def test_replay_preserves_a_checkpoint_owned_by_a_failed_handoff(
    tmp_path: Path,
) -> None:
    effects = Effects(tmp_path)
    effects.ledger.record(
        attempt_id=effects.checkpoint.started_at,
        issue_number=effects.checkpoint.issue_number,
        branch=CHECKPOINT_BRANCH,
        outcome=AttemptOutcome.HANDOFF,
    )
    effects.handoff_failed = True

    result = effects.finalizer().replay()

    assert isinstance(result, Held)
    assert result.reason is HeldReason.HANDOFF_NOT_FULFILLED
    assert result.stage is FinalizationStage.HANDOFF
    assert effects.calls == ["record control", "resolve handoff"]
    assert effects.state.read() == effects.checkpoint


def test_handoff_resolution_error_does_not_turn_a_completion_into_a_hold(
    tmp_path: Path,
) -> None:
    effects = Effects(tmp_path)
    effects.handoff_error = RuntimeError("handoff store temporarily unavailable")

    result = effects.finalizer().finalize(
        replace(REQUEST, mode=FinalizationMode.RECOVERY_RETRY)
    )

    assert result == Finalized(REQUEST.outcome)
    assert effects.resolution_errors == [effects.handoff_error]
    assert effects.ledger.get(effects.checkpoint.started_at).outcome is REQUEST.outcome
    assert effects.state.read() is None
