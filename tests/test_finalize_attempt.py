"""Table-driven tests for the deep finalize-attempt boundary.

The boundary owns one ordering — hold, release, cleanup, second hold,
ledger commit, handoff resolution — over injected seams, so these tests
drive ``finalize_attempt`` directly against real temp-dir completion
stores with small fakes. No ``AgentLifecycle`` is constructed here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from simple_coding_agent.completion import AttemptOutcome
from simple_coding_agent.finalization import (
    AttemptCompletionStore,
    FinalizeReason,
    FinalizeRequest,
    FinalizeStatus,
    commit_recovered_attempt,
    finalize_attempt,
    resolve_finalize_branch,
)
from simple_coding_agent.git_workspace import DirtyWorkspaceError, GitWorkspaceError


def _request(**overrides: object) -> FinalizeRequest:
    base: dict[str, object] = {
        "issue_number": 24,
        "attempt_id": "2026-09-20T13:00:00Z",
        "checkpoint_branch": "agent/issue-24",
        "prepared_branch": "agent/issue-24",
        "outcome": AttemptOutcome.COMPLETE,
        "comment_posted": True,
        "has_prepared": True,
        "publication_started_at": datetime(2026, 9, 20, 13, 0, 0, tzinfo=UTC),
    }
    base.update(overrides)
    return FinalizeRequest(**base)  # type: ignore[arg-type]


class _Seams:
    """Small fakes recording call order for one finalize run."""

    def __init__(
        self,
        *,
        first_hold: bool = False,
        second_hold: bool = False,
        release_error: Exception | None = None,
        cleanup_error: Exception | None = None,
        commit_error: BaseException | None = None,
        handoff: bool | None = True,
    ) -> None:
        self.calls: list[str] = []
        self.first_hold = first_hold
        self.second_hold = second_hold
        self.release_error = release_error
        self.cleanup_error = cleanup_error
        self.commit_error = commit_error
        self.handoff = handoff
        self.committed_branch: str | None = None
        self.held: list[tuple[FinalizeReason | None, str]] = []
        self.dirty: list[Exception] = []

    def is_finalized(self, attempt_id: str) -> bool:
        assert attempt_id == "2026-09-20T13:00:00Z"
        return False

    def check_hold(self, *, prepared: bool, check_allowance: bool) -> bool:
        self.calls.append(f"hold(prepared={prepared},allowance={check_allowance})")
        if len(self.calls) == 1:
            assert check_allowance is True
            return self.first_hold
        assert check_allowance is False
        return self.second_hold

    def release(self) -> None:
        self.calls.append("release")
        if self.release_error is not None:
            raise self.release_error

    def cleanup(self) -> None:
        self.calls.append("cleanup")
        if self.cleanup_error is not None:
            raise self.cleanup_error

    def commit(self, branch: str) -> None:
        self.calls.append(f"commit({branch})")
        self.committed_branch = branch
        if self.commit_error is not None:
            raise self.commit_error

    def resolve_handoff(self) -> bool | None:
        self.calls.append("resolve_handoff")
        return self.handoff

    def on_held(self, reason: FinalizeReason | None, detail: str) -> None:
        self.held.append((reason, detail))

    def on_dirty(self, error: Exception) -> None:
        self.dirty.append(error)


def _finalize(request: FinalizeRequest, seams: _Seams):
    return finalize_attempt(
        request,
        is_finalized=seams.is_finalized,
        check_hold=seams.check_hold,
        release=seams.release,
        cleanup=seams.cleanup,
        commit=seams.commit,
        resolve_handoff=seams.resolve_handoff,
        on_held=seams.on_held,
        on_dirty=seams.on_dirty,
    )


def test_finalize_orders_the_full_boundary(tmp_path: Path) -> None:
    store = AttemptCompletionStore(tmp_path)
    seams = _Seams()

    result = finalize_attempt(
        _request(),
        is_finalized=store.is_finalized,
        check_hold=seams.check_hold,
        release=seams.release,
        cleanup=seams.cleanup,
        commit=lambda branch: store.record(
            attempt_id="2026-09-20T13:00:00Z",
            issue_number=24,
            branch=branch,
            outcome=AttemptOutcome.COMPLETE,
        ),
        resolve_handoff=seams.resolve_handoff,
        on_held=seams.on_held,
        on_dirty=seams.on_dirty,
    )

    assert result.status is FinalizeStatus.FINALIZED
    assert result.released is True
    assert result.cleanup_ok is True
    assert result.handoff is True
    assert store.is_finalized("2026-09-20T13:00:00Z") is True
    assert seams.calls == [
        "hold(prepared=True,allowance=True)",
        "release",
        "cleanup",
        "hold(prepared=False,allowance=False)",
        "resolve_handoff",
    ]


def test_finalize_release_failure_still_cleans_up_then_raises() -> None:
    seams = _Seams(release_error=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        _finalize(_request(), seams)

    assert seams.calls == [
        "hold(prepared=True,allowance=True)",
        "release",
        "cleanup",
    ]
    assert len(seams.held) == 1
    assert seams.held[0][0] is FinalizeReason.RELEASE_UNCONFIRMED


def test_finalize_hold_after_cleanup_skips_the_ledger_write(tmp_path: Path) -> None:
    store = AttemptCompletionStore(tmp_path)
    seams = _Seams(second_hold=True)

    result = finalize_attempt(
        _request(),
        is_finalized=store.is_finalized,
        check_hold=seams.check_hold,
        release=seams.release,
        cleanup=seams.cleanup,
        commit=lambda branch: store.record(
            attempt_id="2026-09-20T13:00:00Z",
            issue_number=24,
            branch=branch,
            outcome=AttemptOutcome.COMPLETE,
        ),
        resolve_handoff=seams.resolve_handoff,
        on_held=seams.on_held,
        on_dirty=seams.on_dirty,
    )

    assert result.status is FinalizeStatus.HELD
    assert result.reason is FinalizeReason.HANDOFF_HOLD
    assert store.read_all() == {}
    assert "resolve_handoff" not in seams.calls


def test_finalize_first_hold_skips_release_and_cleanup() -> None:
    seams = _Seams(first_hold=True)

    result = _finalize(_request(), seams)

    assert result.status is FinalizeStatus.HELD
    assert result.reason is FinalizeReason.HANDOFF_HOLD
    assert result.released is False
    assert seams.calls == ["hold(prepared=True,allowance=True)"]


def test_finalize_unconfirmed_publication_holds_without_commit() -> None:
    seams = _Seams()

    result = _finalize(_request(comment_posted=False), seams)

    assert result.status is FinalizeStatus.HELD
    assert result.reason is FinalizeReason.PUBLICATION_UNCONFIRMED
    # Cleanup still runs (the tree must leave the attempt branch), but the
    # ledger commit and handoff resolution never happen.
    assert seams.calls == ["cleanup"]
    assert seams.committed_branch is None


def test_finalize_dirty_cleanup_holds_and_reports_dirty() -> None:
    seams = _Seams(
        cleanup_error=DirtyWorkspaceError("tree is dirty"),
    )

    result = _finalize(_request(), seams)

    assert result.status is FinalizeStatus.HELD
    assert result.reason is FinalizeReason.CLEANUP_UNCONFIRMED
    assert len(seams.dirty) == 1
    assert seams.committed_branch is None


def test_finalize_ledger_hit_is_idempotent_without_a_second_release(
    tmp_path: Path,
) -> None:
    store = AttemptCompletionStore(tmp_path)
    store.record(
        attempt_id="2026-09-20T13:00:00Z",
        issue_number=24,
        branch="agent/issue-24",
        outcome=AttemptOutcome.COMPLETE,
    )
    seams = _Seams()

    result = finalize_attempt(
        _request(),
        is_finalized=store.is_finalized,
        check_hold=seams.check_hold,
        release=seams.release,
        cleanup=seams.cleanup,
        commit=seams.commit,
        resolve_handoff=seams.resolve_handoff,
        on_held=seams.on_held,
        on_dirty=seams.on_dirty,
    )

    assert result.status is FinalizeStatus.FINALIZED
    assert result.idempotent is True
    assert seams.calls == []
    assert len(store.read_all()) == 1


def test_finalize_uses_checkpoint_branch_when_prepared_is_gone() -> None:
    seams = _Seams()

    result = _finalize(
        _request(
            prepared_branch=None,
            has_prepared=False,
            checkpoint_branch="agent/issue-24-restored",
        ),
        seams,
    )

    assert result.status is FinalizeStatus.FINALIZED
    assert seams.committed_branch == "agent/issue-24-restored"


def test_resolve_finalize_branch_prefers_prepared() -> None:
    assert (
        resolve_finalize_branch(
            prepared_branch="agent/issue-24", checkpoint_branch="agent/issue-24-old"
        )
        == "agent/issue-24"
    )
    assert (
        resolve_finalize_branch(
            prepared_branch=None, checkpoint_branch="agent/issue-24-old"
        )
        == "agent/issue-24-old"
    )


def test_commit_recovered_attempt_captures_release_failure() -> None:
    seams = _Seams(release_error=RuntimeError("release down"))

    result = commit_recovered_attempt(
        attempt_id="2026-09-20T13:00:00Z",
        branch="agent/issue-24",
        is_finalized=seams.is_finalized,
        release=seams.release,
        cleanup=seams.cleanup,
        commit=seams.commit,
        resolve_handoff=seams.resolve_handoff,
        on_held=seams.on_held,
        on_dirty=seams.on_dirty,
    )

    assert result.status is FinalizeStatus.HELD
    assert result.reason is FinalizeReason.RELEASE_UNCONFIRMED
    assert "release down" in result.detail


def test_commit_recovered_attempt_captures_accounting_hold() -> None:
    seams = _Seams(commit_error=SystemExit(3))

    result = commit_recovered_attempt(
        attempt_id="2026-09-20T13:00:00Z",
        branch="agent/issue-24",
        is_finalized=seams.is_finalized,
        release=seams.release,
        cleanup=seams.cleanup,
        commit=seams.commit,
        resolve_handoff=seams.resolve_handoff,
        on_held=seams.on_held,
        on_dirty=seams.on_dirty,
    )

    assert result.status is FinalizeStatus.HELD
    assert "accounting" in result.detail


def test_commit_recovered_attempt_dirty_cleanup_holds() -> None:
    seams = _Seams(cleanup_error=DirtyWorkspaceError("dirty"))

    result = commit_recovered_attempt(
        attempt_id="2026-09-20T13:00:00Z",
        branch="agent/issue-24",
        is_finalized=seams.is_finalized,
        release=seams.release,
        cleanup=seams.cleanup,
        commit=seams.commit,
        resolve_handoff=seams.resolve_handoff,
        on_held=seams.on_held,
        on_dirty=seams.on_dirty,
    )

    assert result.status is FinalizeStatus.HELD
    assert result.reason is FinalizeReason.CLEANUP_UNCONFIRMED
    assert len(seams.dirty) == 1
    assert seams.committed_branch is None


def test_commit_recovered_attempt_finalizes_in_order() -> None:
    seams = _Seams()

    result = commit_recovered_attempt(
        attempt_id="2026-09-20T13:00:00Z",
        branch="agent/issue-24",
        is_finalized=seams.is_finalized,
        release=seams.release,
        cleanup=seams.cleanup,
        commit=seams.commit,
        resolve_handoff=seams.resolve_handoff,
        on_held=seams.on_held,
        on_dirty=seams.on_dirty,
    )

    assert result.status is FinalizeStatus.FINALIZED
    assert seams.calls == ["release", "cleanup", "commit(agent/issue-24)", "resolve_handoff"]


def test_git_workspace_error_is_not_a_dirty_hold() -> None:
    assert not issubclass(GitWorkspaceError, DirtyWorkspaceError)
    assert issubclass(DirtyWorkspaceError, GitWorkspaceError)
