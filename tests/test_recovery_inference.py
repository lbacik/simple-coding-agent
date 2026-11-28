"""Retained-attempt inference owned by recovery (issue #111).

Decision-table tests through the public recovery interface: checkpoint
phase + workspace evidence -> what happened, what may safely be redone.
No lifecycle construction; the lifecycle delegates to these functions.
"""

from __future__ import annotations

from simple_coding_agent.attempt_state import AttemptPhase
from simple_coding_agent.completion import AttemptOutcome, PublicationPath
from simple_coding_agent.recovery import (
    ConflictEvidence,
    ReleaseEvidence,
    RetryEvidence,
    SharedRecordEvidence,
    dirty_tree_snapshot,
    final_check_interrupted,
    infer_conflict_reason,
    infer_release_outcome,
    infer_retry_decision,
    infer_shared_hold,
    publication_progress,
    retained_snapshot,
    startup_unreadable_snapshot,
)


def test_retry_model_running_with_commits_advances_to_pushing() -> None:
    inference = infer_retry_decision(
        RetryEvidence(phase=AttemptPhase.MODEL_RUNNING, has_commits=True)
    )

    assert inference.decision.outcome is AttemptOutcome.INCOMPLETE
    assert inference.decision.publication_path is PublicationPath.PARTIAL
    assert inference.decision.reasons == (
        "Model execution was interrupted; preserved committed work.",
    )
    assert inference.details == inference.decision.reasons[0]
    assert inference.note_sha is None
    assert inference.has_commits is True
    assert inference.advance_to_pushing is True


def test_retry_model_running_with_final_check_interrupted() -> None:
    inference = infer_retry_decision(
        RetryEvidence(
            phase=AttemptPhase.MODEL_RUNNING,
            has_commits=True,
            final_check_interrupted=True,
        )
    )

    assert inference.decision.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    assert inference.decision.publication_path is PublicationPath.PARTIAL
    assert inference.decision.reasons == (
        "Final check was interrupted before completing;"
        " committed work is preserved on the branch.",
    )
    assert inference.advance_to_pushing is False
    assert inference.has_commits is True


def test_retry_model_running_without_commits() -> None:
    inference = infer_retry_decision(
        RetryEvidence(phase=AttemptPhase.MODEL_RUNNING, has_commits=False)
    )

    assert inference.decision.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    assert inference.decision.publication_path is PublicationPath.NONE
    assert inference.decision.reasons == (
        "Model execution was interrupted without commits.",
    )
    assert inference.advance_to_pushing is False


def test_retry_pushing_with_valid_note_publishes_handoff() -> None:
    inference = infer_retry_decision(
        RetryEvidence(
            phase=AttemptPhase.PUSHING,
            has_commits=True,
            note_revision="abc123",
            note_valid=True,
            note_details="the note body",
        )
    )

    assert inference.decision.outcome is AttemptOutcome.HANDOFF
    assert inference.decision.publication_path is PublicationPath.PARTIAL
    assert inference.decision.reasons == (
        "Recovered a previously committed handoff note for publication.",
    )
    assert inference.details == "the note body"
    assert inference.note_sha == "abc123"


def test_retry_pushing_with_invalid_note_falls_back_to_incomplete() -> None:
    inference = infer_retry_decision(
        RetryEvidence(
            phase=AttemptPhase.PUBLISHING,
            has_commits=True,
            note_revision="abc123",
            note_valid=False,
            note_reason="the validation reason",
        )
    )

    assert inference.decision.outcome is AttemptOutcome.INCOMPLETE
    assert inference.decision.publication_path is PublicationPath.PARTIAL
    assert inference.decision.reasons == ("the validation reason",)
    assert inference.details == "the validation reason"
    assert inference.note_sha is None


def test_retry_pushing_without_note_recovers_completed_evidence() -> None:
    inference = infer_retry_decision(
        RetryEvidence(phase=AttemptPhase.PUBLISHING, has_commits=False)
    )

    assert inference.decision.outcome is None
    assert inference.decision.publication_eligible is True
    assert inference.decision.publication_path is PublicationPath.COMPLETE
    assert inference.decision.reasons == (
        "Recovered previously completed local evidence for publication.",
    )


def test_retry_before_model_execution() -> None:
    for phase in (AttemptPhase.CLAIMED, AttemptPhase.SETUP):
        inference = infer_retry_decision(
            RetryEvidence(phase=phase, has_commits=False)
        )

        assert inference.decision.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
        assert inference.decision.publication_path is PublicationPath.NONE
        assert inference.decision.reasons == (
            "Attempt was interrupted before model execution.",
        )


def test_release_prefers_valid_handoff_note() -> None:
    inference = infer_release_outcome(
        ReleaseEvidence(
            phase=AttemptPhase.PUBLISHING, has_commits=True, note_valid=True
        )
    )

    assert inference.outcome is AttemptOutcome.HANDOFF
    assert "not fulfilled" in inference.reason
    assert "phase publishing" in inference.reason


def test_release_with_commits_is_incomplete() -> None:
    inference = infer_release_outcome(
        ReleaseEvidence(phase=AttemptPhase.PUSHING, has_commits=True)
    )

    assert inference.outcome is AttemptOutcome.INCOMPLETE
    assert "secured the retained work" in inference.reason


def test_release_before_model_execution() -> None:
    inference = infer_release_outcome(
        ReleaseEvidence(phase=AttemptPhase.SETUP, has_commits=False)
    )

    assert inference.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    assert "before model execution" in inference.reason


def test_release_after_model_without_commits_is_finalization_hold() -> None:
    inference = infer_release_outcome(
        ReleaseEvidence(phase=AttemptPhase.PUBLISHING, has_commits=False)
    )

    assert inference.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    assert "completion boundary" in inference.reason


def test_conflict_fetch_failure_is_ambiguous() -> None:
    reason = infer_conflict_reason(
        ConflictEvidence(issue_number=24, fetch_error="ValueError: boom")
    )

    assert reason is not None
    assert "ambiguous" in reason
    assert "ValueError: boom" in reason


def test_conflict_missing_issue_holds() -> None:
    reason = infer_conflict_reason(
        ConflictEvidence(issue_number=24, missing=True)
    )

    assert reason == (
        "issue #24 is unavailable on GitHub;"
        " the retry must not replay a side effect it cannot verify."
    )


def test_conflict_with_holders_names_them() -> None:
    reason = infer_conflict_reason(
        ConflictEvidence(
            issue_number=24,
            self_assigned=False,
            holders=("other",),
            foreign_marker="",
        )
    )

    assert reason is not None
    assert "assigned to another assignee: other" in reason


def test_conflict_without_holders_keeps_agent_detail() -> None:
    reason = infer_conflict_reason(
        ConflictEvidence(issue_number=24, self_assigned=False)
    )

    assert reason is not None
    assert "no longer this agent" in reason


def test_no_conflict_when_self_assigned() -> None:
    assert (
        infer_conflict_reason(
            ConflictEvidence(issue_number=24, self_assigned=True)
        )
        is None
    )


def test_final_check_interrupted_table() -> None:
    assert final_check_interrupted(None) is False
    assert final_check_interrupted({}) is False
    assert (
        final_check_interrupted(
            {"final_check_started": True, "final_check_finished": True}
        )
        is False
    )
    assert (
        final_check_interrupted(
            {"final_check_started": True, "final_check_finished": False}
        )
        is True
    )


def test_publication_progress_shape() -> None:
    assert publication_progress(
        "pushing", comment_confirmed=True, pull_request={"number": 1, "url": "u"}
    ) == {
        "phase": "pushing",
        "comment_confirmed": True,
        "pull_request": {"number": 1, "url": "u"},
    }


def test_retained_snapshot_names_attempt_and_next_action() -> None:
    snapshot = retained_snapshot(
        attempt_id="attempt-1",
        issue_number=24,
        phase="pushing",
        outcome=None,
        branch="agent/issue-24",
        workspace="/work",
        workspace_dirty=False,
        publication=publication_progress(
            "pushing", comment_confirmed=None, pull_request=None
        ),
        hold_reason="the hold",
        recovering=False,
        finalized=False,
    )

    assert snapshot["attempt_id"] == "attempt-1"
    assert "attempt-1" in snapshot["hold_reason"]
    assert "agentctl recovery retry" in snapshot["next_action"]
    assert "agentctl recovery release" in snapshot["next_action"]


def test_startup_and_dirty_snapshots_cover_no_checkpoint() -> None:
    unreadable = startup_unreadable_snapshot(
        workspace="/work", workspace_dirty=True, reason="bad state"
    )
    assert unreadable["checkpoint"] is False
    assert unreadable["hold_reason"] == "bad state"

    dirty = dirty_tree_snapshot(workspace="/work", reason="dirty")
    assert dirty["workspace_dirty"] is True
    assert dirty["hold_reason"] == "dirty"


A1 = "2026-09-20T13:00:00Z"
A2 = "2026-09-20T14:00:00Z"
SHARED_REPO = "octo/example"


def _shared_evidence(**overrides: object) -> SharedRecordEvidence:
    params: dict[str, object] = {
        "issue_number": 24,
        "attempt_id": A1,
        "repository": SHARED_REPO,
        "outcome": AttemptOutcome.HANDOFF,
    }
    params.update(overrides)
    return SharedRecordEvidence(**params)  # type: ignore[arg-type]


def test_shared_hold_missing_record_proceeds() -> None:
    assert infer_shared_hold(_shared_evidence(record_missing=True)) is None


def test_shared_hold_fetch_error_is_ambiguous() -> None:
    hold = infer_shared_hold(_shared_evidence(fetch_error="boom"))
    assert hold is not None
    assert "ambiguous" in hold


def test_shared_hold_unreadable_record_retains() -> None:
    hold = infer_shared_hold(_shared_evidence(record_unreadable=True))
    assert hold is not None
    assert "unreadable" in hold


def test_shared_hold_wrong_issue_or_repo_retains() -> None:
    hold = infer_shared_hold(
        _shared_evidence(record_issue_number=25, record_repository=SHARED_REPO,
                         record_attempt_id=A1)
    )
    assert hold is not None
    assert "different issue or repository" in hold

    hold = infer_shared_hold(
        _shared_evidence(record_issue_number=24, record_repository="other/repo",
                         record_attempt_id=A1)
    )
    assert hold is not None
    assert "different issue or repository" in hold


def test_shared_hold_newer_attempt_owns_record() -> None:
    hold = infer_shared_hold(
        _shared_evidence(record_issue_number=24, record_repository=SHARED_REPO,
                         record_attempt_id=A2)
    )
    assert hold is not None
    assert "newer attempt" in hold


def test_shared_hold_same_attempt_non_handoff_retains() -> None:
    hold = infer_shared_hold(
        _shared_evidence(outcome=AttemptOutcome.INCOMPLETE,
                         record_issue_number=24, record_repository=SHARED_REPO,
                         record_attempt_id=A1)
    )
    assert hold is not None
    assert "must not change it" in hold


def test_shared_hold_same_attempt_handoff_proceeds() -> None:
    assert (
        infer_shared_hold(
            _shared_evidence(record_issue_number=24, record_repository=SHARED_REPO,
                             record_attempt_id=A1)
        )
        is None
    )


def test_shared_hold_older_record_proceeds() -> None:
    assert (
        infer_shared_hold(
            _shared_evidence(attempt_id=A2, record_issue_number=24,
                             record_repository=SHARED_REPO, record_attempt_id=A1)
        )
        is None
    )
