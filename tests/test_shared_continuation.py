"""Shared handoff record as the authority for continuation accounting (issue #283).

A routine handoff needs no individual result comment: the single shared
handoff comment carries the automatic-continuation standing, newer ordinary
results reconcile against it in attempt order, and recovery confirms a
published handoff from the shared record plus verified revisions. Older
issues without shared metadata migrate from trusted historical result
comments; unreadable or ambiguous metadata never grants a fresh budget.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.completion import AttemptOutcome
from simple_coding_agent.control import CommandAcknowledgement, ControlStore
from simple_coding_agent.finalization import AttemptCompletionStore
from simple_coding_agent.github_tracker import (
    Assignment,
    GitHubIdentity,
    GitHubTracker,
    IssueComment,
    IssuePage,
    TrackerIssue,
)
from simple_coding_agent.handoff_record import (
    HANDOFF_RECORD_VERSION,
    HandoffRecord,
    MemoryHandoffPublicationStore,
    build_handoff_record_body,
)
from simple_coding_agent.lifecycle import AgentLifecycle
from simple_coding_agent.publication import HandoffRecordRef, confirm_shared_handoff
from simple_coding_agent.recovery import (
    AmbiguousContinuationError,
    CONFLICT_OUTCOME_SUCCESS,
    REQUEUE_SOURCE_AUTOMATIC,
    REQUEUE_SOURCE_HUMAN,
    attempt_marker,
    conflict_attempt_marker,
    requeue_source_marker,
    resolve_automatic_continuation_count,
)
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace


A1 = "2026-09-20T13:00:00Z"
A2 = "2026-09-20T14:00:00Z"
A3 = "2026-09-20T15:00:00Z"
REPO = "octo/example"


def result_body(attempt: str, source: str, *, outcome: str = "handoff") -> str:
    return (
        f"## Agent Attempt Result: {outcome}\n\nHalf done.\n\n"
        f"{attempt_marker(attempt)}\n"
        f"{requeue_source_marker(source)}"
    )


def shared_body(
    attempt: str,
    count: int,
    *,
    issue_number: int = 24,
    repository: str = REPO,
    source: str = REQUEUE_SOURCE_AUTOMATIC,
    note_sha: str = "c" * 40,
) -> str:
    return build_handoff_record_body(
        HandoffRecord(
            version=HANDOFF_RECORD_VERSION,
            repository=repository,
            issue_number=issue_number,
            attempt_id=attempt,
            requeue_source=source,
            continuation_count=count,
            branch=f"agent/issue-{issue_number}",
            branch_sha="b" * 40,
            note_sha=note_sha,
        )
    )


def unsupported_body(attempt: str) -> str:
    return (
        "## Handoff note\n\nCurrent note.\n\n"
        "<!-- handoff-record: v=999 repo=octo/example issue=24"
        f" attempt={attempt} source=automatic count=1"
        " branch=agent/issue-24 branch_sha=" + "b" * 40 + " note_sha=" + "c" * 40 + " -->"
    )


# ---------------------------------------------------------------------------
# resolver: shared-record authority, reconciliation, migration, ambiguity
# ---------------------------------------------------------------------------


def test_resolver_falls_back_to_legacy_without_shared_metadata() -> None:
    bodies = (
        result_body(A2, REQUEUE_SOURCE_AUTOMATIC),
        result_body(A1, REQUEUE_SOURCE_AUTOMATIC),
    )
    assert resolve_automatic_continuation_count(bodies) == 2
    bodies = (
        result_body(A2, REQUEUE_SOURCE_AUTOMATIC),
        result_body(A1, REQUEUE_SOURCE_HUMAN),
    )
    assert resolve_automatic_continuation_count(bodies) == 1


def test_resolver_shared_record_needs_no_legacy_comment() -> None:
    assert resolve_automatic_continuation_count((shared_body(A2, 2),)) == 2


def test_resolver_overlapping_legacy_and_shared_count_once() -> None:
    bodies = (
        result_body(A2, REQUEUE_SOURCE_AUTOMATIC),
        shared_body(A2, 2),
        result_body(A1, REQUEUE_SOURCE_AUTOMATIC),
    )
    assert resolve_automatic_continuation_count(bodies) == 2


def test_resolver_newer_automatic_increments_shared() -> None:
    bodies = (
        result_body(A3, REQUEUE_SOURCE_AUTOMATIC),
        shared_body(A2, 2),
        result_body(A1, REQUEUE_SOURCE_AUTOMATIC),
    )
    assert resolve_automatic_continuation_count(bodies) == 3


def test_resolver_newer_human_requeue_resets_shared() -> None:
    bodies = (
        result_body(A3, REQUEUE_SOURCE_HUMAN),
        shared_body(A2, 2),
        result_body(A1, REQUEUE_SOURCE_AUTOMATIC),
    )
    assert resolve_automatic_continuation_count(bodies) == 0


def test_resolver_newer_ordinary_result_resets_shared() -> None:
    bodies = (
        result_body(A3, REQUEUE_SOURCE_HUMAN, outcome="incomplete"),
        shared_body(A2, 2),
    )
    assert resolve_automatic_continuation_count(bodies) == 0


def test_resolver_automatic_after_a_newer_human_counts() -> None:
    micro_newer = "2026-09-20T15:00:00.000001Z"
    bodies = (
        result_body(micro_newer, REQUEUE_SOURCE_AUTOMATIC),
        result_body(A3, REQUEUE_SOURCE_HUMAN),
        shared_body(A2, 2),
    )
    assert resolve_automatic_continuation_count(bodies) == 1


def test_resolver_conflict_results_stay_transparent() -> None:
    bodies = (
        "## Conflict Attempt Result: success\n\n"
        f"{attempt_marker(A3)}\n"
        f"{conflict_attempt_marker(CONFLICT_OUTCOME_SUCCESS, sha='d' * 40)}",
        shared_body(A2, 2),
        result_body(A2, REQUEUE_SOURCE_AUTOMATIC),
    )
    assert resolve_automatic_continuation_count(bodies) == 2


def test_resolver_orders_attempt_ids_chronologically_not_lexically() -> None:
    # Plain string comparison misorders "...00Z" past "...00.000001Z";
    # the shared record's attempt is the microsecond one, the newer plain
    # stamp must still reconcile on top of it.
    shared_micro = "2026-09-20T14:00:00.000001Z"
    bodies = (
        result_body(A3, REQUEUE_SOURCE_AUTOMATIC),
        shared_body(shared_micro, 2),
    )
    assert resolve_automatic_continuation_count(bodies) == 3


def test_resolver_duplicate_shared_records_are_ambiguous() -> None:
    with pytest.raises(AmbiguousContinuationError):
        resolve_automatic_continuation_count(
            (shared_body(A2, 2), shared_body(A3, 3))
        )


def test_resolver_malformed_shared_record_is_ambiguous() -> None:
    with pytest.raises(AmbiguousContinuationError):
        resolve_automatic_continuation_count(
            ("## Handoff note\n\n<!-- handoff-record: v=1 broken -->",)
        )


def test_resolver_unsupported_shared_version_is_ambiguous() -> None:
    with pytest.raises(AmbiguousContinuationError):
        resolve_automatic_continuation_count((unsupported_body(A2),))


def test_resolver_issue_or_repository_mismatch_is_ambiguous() -> None:
    with pytest.raises(AmbiguousContinuationError):
        resolve_automatic_continuation_count(
            (shared_body(A2, 2, issue_number=25),), issue_number=24
        )
    with pytest.raises(AmbiguousContinuationError):
        resolve_automatic_continuation_count(
            (shared_body(A2, 2, repository="other/repo"),),
            issue_number=24,
            repository=REPO,
        )


# ---------------------------------------------------------------------------
# tracker: eligibility from the shared record
# ---------------------------------------------------------------------------


class _CommentsTransport:
    """Minimal tracker transport serving one issue's comments."""

    def __init__(self, issue: TrackerIssue, bodies: list[str]) -> None:
        self._issue = issue
        self._bodies = list(bodies)

    def list_issues(self, repository: str, cursor: str | None) -> IssuePage:
        return IssuePage(issues=(self._issue,), next_cursor=None)

    def get_issue(self, repository: str, number: int) -> TrackerIssue | None:
        return self._issue

    def viewer(self) -> GitHubIdentity:
        return GitHubIdentity(id="viewer-id", login="agent")

    def assign_issue(self, issue_id: str, assignee_id: str) -> Assignment:
        raise AssertionError("no claims in eligibility tests")

    def remove_assignee(self, issue_id: str, assignee_id: str) -> None:
        raise AssertionError("no releases in eligibility tests")

    def remove_label(self, issue_id: str, label: str) -> None:
        raise AssertionError("no releases in eligibility tests")

    def add_label(self, issue_id: str, label: str) -> None:
        raise AssertionError("no labels in eligibility tests")

    def list_issue_comments(
        self, repository: str, issue_number: int
    ) -> tuple[IssueComment, ...]:
        return tuple(IssueComment(author_login="agent", body=body) for body in self._bodies)


def _continuation_issue() -> TrackerIssue:
    return TrackerIssue(
        id="issue-24",
        number=24,
        title="Continue",
        body="Keep going.",
        created_at=datetime(2026, 9, 20, tzinfo=UTC),
        state="OPEN",
        labels=frozenset({"round-finished"}),
        assignee_logins=(),
        blocked_by=0,
        author_login="reporter",
    )


def _tracker(bodies: list[str], *, budget: int = 5) -> GitHubTracker:
    return GitHubTracker(
        _CommentsTransport(_continuation_issue(), bodies),
        REPO,
        max_auto_continuations=budget,
    )


def test_tracker_eligible_from_shared_record_without_legacy_comment() -> None:
    tracker = _tracker([shared_body(A2, 2)])
    assert tracker.automatic_continuation_count(_continuation_issue()) == 2
    assert tracker.is_auto_continuation_eligible(_continuation_issue())


def test_tracker_over_budget_from_shared_record_stays_queued_out() -> None:
    tracker = _tracker([shared_body(A2, 5)], budget=5)
    assert not tracker.is_auto_continuation_eligible(_continuation_issue())


def test_tracker_newer_human_result_resets_shared_standing() -> None:
    tracker = _tracker(
        [shared_body(A2, 4), result_body(A3, REQUEUE_SOURCE_HUMAN)]
    )
    assert tracker.automatic_continuation_count(_continuation_issue()) == 0
    assert tracker.is_auto_continuation_eligible(_continuation_issue())


def test_tracker_ambiguous_shared_metadata_grants_no_budget() -> None:
    tracker = _tracker([shared_body(A2, 1), shared_body(A3, 2)])
    with pytest.raises(AmbiguousContinuationError):
        tracker.automatic_continuation_count(_continuation_issue())
    assert not tracker.is_auto_continuation_eligible(_continuation_issue())


# ---------------------------------------------------------------------------
# publication: shared-only confirmation
# ---------------------------------------------------------------------------


class _RecordTransport:
    def __init__(
        self,
        ref: HandoffRecordRef | None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._ref = ref
        self._error = error

    def find_handoff_record(
        self, repository: str, issue_number: int
    ) -> HandoffRecordRef | None:
        if self._error is not None:
            raise self._error
        return self._ref


def test_confirm_shared_handoff_matches_without_result_comment() -> None:
    body = shared_body(A1, 1, source=REQUEUE_SOURCE_HUMAN)
    record = confirm_shared_handoff(
        _RecordTransport(HandoffRecordRef(comment_id="c1", body=body)),
        REPO,
        24,
        A1,
    )
    assert record is not None
    assert record.attempt_id == A1
    assert record.continuation_count == 1


def test_confirm_shared_handoff_rejects_mismatches_and_ambiguity() -> None:
    body = shared_body(A1, 1, source=REQUEUE_SOURCE_HUMAN)
    assert (
        confirm_shared_handoff(
            _RecordTransport(HandoffRecordRef(comment_id="c1", body=body)),
            REPO,
            24,
            A2,
        )
        is None
    )
    assert (
        confirm_shared_handoff(
            _RecordTransport(
                HandoffRecordRef(comment_id="c1", body=body),
                error=OSError("down"),
            ),
            REPO,
            24,
            A1,
        )
        is None
    )
    assert confirm_shared_handoff(_RecordTransport(None), REPO, 24, A1) is None
    assert (
        confirm_shared_handoff(
            _RecordTransport(HandoffRecordRef(comment_id="c1", body="broken")),
            REPO,
            24,
            A1,
        )
        is None
    )


def test_confirm_shared_handoff_rejects_foreign_issue_or_repo() -> None:
    foreign_issue = shared_body(A1, 1, issue_number=25, source=REQUEUE_SOURCE_HUMAN)
    assert (
        confirm_shared_handoff(
            _RecordTransport(HandoffRecordRef(comment_id="c1", body=foreign_issue)),
            REPO,
            24,
            A1,
        )
        is None
    )
    foreign_repo = shared_body(
        A1, 1, repository="octo/other", source=REQUEUE_SOURCE_HUMAN
    )
    assert (
        confirm_shared_handoff(
            _RecordTransport(HandoffRecordRef(comment_id="c1", body=foreign_repo)),
            REPO,
            24,
            A1,
        )
        is None
    )


def test_confirm_shared_handoff_verifies_revisions_when_given() -> None:
    body = shared_body(A1, 1, source=REQUEUE_SOURCE_HUMAN)
    transport = _RecordTransport(HandoffRecordRef(comment_id="c1", body=body))
    assert (
        confirm_shared_handoff(transport, REPO, 24, A1, branch_sha="b" * 40) is not None
    )
    assert (
        confirm_shared_handoff(transport, REPO, 24, A1, branch_sha="0" * 40) is None
    )
    assert (
        confirm_shared_handoff(transport, REPO, 24, A1, note_sha="c" * 40) is not None
    )
    assert confirm_shared_handoff(transport, REPO, 24, A1, note_sha="0" * 40) is None


def test_handoff_intent_store_first_wins_across_retries() -> None:
    store = MemoryHandoffPublicationStore()
    first = HandoffRecord(
        version=HANDOFF_RECORD_VERSION,
        repository=REPO,
        issue_number=24,
        attempt_id=A1,
        requeue_source=REQUEUE_SOURCE_AUTOMATIC,
        continuation_count=2,
        branch="agent/issue-24",
        branch_sha="b" * 40,
        note_sha="c" * 40,
    )
    assert store.save_if_absent(first) == first
    recomputed = HandoffRecord(
        version=HANDOFF_RECORD_VERSION,
        repository=REPO,
        issue_number=24,
        attempt_id=A1,
        requeue_source=REQUEUE_SOURCE_AUTOMATIC,
        continuation_count=3,
        branch="agent/issue-24",
        branch_sha="b" * 40,
        note_sha="c" * 40,
    )
    assert store.save_if_absent(recomputed) == first
    assert store.load(A1) is not None
    assert store.load(A1).continuation_count == 2


# ---------------------------------------------------------------------------
# lifecycle: pre-publish shared hold during recovery retry
# ---------------------------------------------------------------------------


def _hold_lifecycle(tmp_path: Path, github):
    tracker = FakeTracker(
        issues={
            24: TrackerIssue(
                id="issue-24",
                number=24,
                title="Implement recovery",
                body="Recover retained work.",
                created_at=datetime(2026, 9, 20, tzinfo=UTC),
                state="OPEN",
                labels=frozenset({"ready-for-agent"}),
                assignee_logins=("agent",),
                blocked_by=0,
                author_login="reporter",
            )
        }
    )

    class _StubPublisher:
        def __init__(self) -> None:
            self._github = github
            self._repository = REPO

    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "workspace", issue_number=24),
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=_StubPublisher(),
        attempt_runner=lambda c, p, pr: None,
        control_store=ControlStore(tmp_path),
        completion_store=AttemptCompletionStore(tmp_path),
        sleeper=lambda seconds: None,
        repository=REPO,
    )
    return lifecycle


class _HoldGitHub:
    def __init__(self, ref: HandoffRecordRef | None = None, *, error=None) -> None:
        self._ref = ref
        self._error = error

    def find_handoff_record(self, repository: str, issue_number: int):
        if self._error is not None:
            raise self._error
        return self._ref


def test_shared_hold_skipped_without_the_seam(tmp_path: Path) -> None:
    lifecycle = _hold_lifecycle(tmp_path, object())
    assert (
        lifecycle._recovery_shared_hold(24, A1, AttemptOutcome.HANDOFF) is None
    )


def test_shared_hold_absent_record_proceeds(tmp_path: Path) -> None:
    lifecycle = _hold_lifecycle(tmp_path, _HoldGitHub(None))
    assert (
        lifecycle._recovery_shared_hold(24, A1, AttemptOutcome.HANDOFF) is None
    )


def test_shared_hold_ambiguous_read_retains(tmp_path: Path) -> None:
    lifecycle = _hold_lifecycle(tmp_path, _HoldGitHub(error=OSError("down")))
    assert (
        lifecycle._recovery_shared_hold(24, A1, AttemptOutcome.HANDOFF) is not None
    )


def test_shared_hold_unreadable_record_retains(tmp_path: Path) -> None:
    lifecycle = _hold_lifecycle(
        tmp_path, _HoldGitHub(HandoffRecordRef(comment_id="c1", body="broken"))
    )
    assert (
        lifecycle._recovery_shared_hold(24, A1, AttemptOutcome.HANDOFF) is not None
    )


def test_shared_hold_newer_remote_record_retains(tmp_path: Path) -> None:
    lifecycle = _hold_lifecycle(
        tmp_path,
        _HoldGitHub(HandoffRecordRef(comment_id="c1", body=shared_body(A2, 2))),
    )
    assert lifecycle._recovery_shared_hold(24, A1, AttemptOutcome.HANDOFF) is not None


def test_shared_hold_same_attempt_non_handoff_outcome_retains(tmp_path: Path) -> None:
    lifecycle = _hold_lifecycle(
        tmp_path,
        _HoldGitHub(HandoffRecordRef(comment_id="c1", body=shared_body(A1, 1))),
    )
    assert (
        lifecycle._recovery_shared_hold(24, A1, AttemptOutcome.INCOMPLETE) is not None
    )


def test_shared_hold_same_attempt_handoff_proceeds(tmp_path: Path) -> None:
    lifecycle = _hold_lifecycle(
        tmp_path,
        _HoldGitHub(HandoffRecordRef(comment_id="c1", body=shared_body(A1, 1))),
    )
    assert lifecycle._recovery_shared_hold(24, A1, AttemptOutcome.HANDOFF) is None


def test_shared_hold_older_remote_record_proceeds(tmp_path: Path) -> None:
    lifecycle = _hold_lifecycle(
        tmp_path,
        _HoldGitHub(HandoffRecordRef(comment_id="c1", body=shared_body(A1, 1))),
    )
    assert lifecycle._recovery_shared_hold(24, A2, AttemptOutcome.HANDOFF) is None


def test_shared_hold_wrong_issue_or_repo_retains(tmp_path: Path) -> None:
    lifecycle = _hold_lifecycle(
        tmp_path,
        _HoldGitHub(
            HandoffRecordRef(
                comment_id="c1", body=shared_body(A1, 1, issue_number=25)
            )
        ),
    )
    assert lifecycle._recovery_shared_hold(24, A1, AttemptOutcome.HANDOFF) is not None


# ---------------------------------------------------------------------------
# lifecycle: recovery retry confirms a handoff from the shared record alone
# ---------------------------------------------------------------------------


def _recovery_issue() -> TrackerIssue:
    return TrackerIssue(
        id="issue-24",
        number=24,
        title="Implement recovery",
        body="Recover retained work.",
        created_at=datetime(2026, 9, 20, tzinfo=UTC),
        state="OPEN",
        labels=frozenset({"ready-for-agent"}),
        assignee_logins=("agent",),
        blocked_by=0,
        author_login="reporter",
    )


def _note_content(attempt_id: str) -> str:
    return (
        "# Handoff note: issue #24\n\n"
        f"- issue: 24\n"
        f"- started_at: {attempt_id}\n"
        "- reason: token_soft_threshold\n"
        "- last_work_commit: base-revision-0\n\n"
        "Half done.\n"
    )


def test_retry_confirms_handoff_from_shared_record_without_result_comment(
    tmp_path: Path,
) -> None:
    state = AttemptStateStore(tmp_path)
    state.start(issue_number=24, branch="agent/issue-24")
    for phase in (
        AttemptPhase.SETUP,
        AttemptPhase.MODEL_RUNNING,
        AttemptPhase.PUSHING,
        AttemptPhase.PUBLISHING,
    ):
        state.transition(phase)
    checkpoint = state.read()
    assert checkpoint is not None

    workspace = InMemoryWorkspace(tmp_path / "workspace", issue_number=24)
    workspace.commit_handoff_note(24, _note_content(checkpoint.started_at))

    tracker = FakeTracker(issues={24: _recovery_issue()})
    # The individual result comment never posts, but the verified shared
    # record does: recovery must finalize without rerunning the model.
    publisher = FakePublisher(
        outcome=AttemptOutcome.HANDOFF, comment_posted=False, handoff_posted=True
    )
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=publisher,
        attempt_runner=lambda c, p, pr: (_ for _ in ()).throw(
            AssertionError("recovery must not invoke the model")
        ),
        control_store=ControlStore(tmp_path),
        completion_store=AttemptCompletionStore(tmp_path),
        sleeper=lambda seconds: None,
        repository=REPO,
    )

    record = lifecycle.submit_recovery_retry("req-shared", checkpoint.started_at)

    assert record.acknowledgement is CommandAcknowledgement.COMPLETED
    assert AttemptStateStore(tmp_path).read() is None
    assert AttemptCompletionStore(tmp_path).is_finalized(checkpoint.started_at)


# ---------------------------------------------------------------------------
# FakeTracker seeded comments take the shared-aware path
# ---------------------------------------------------------------------------


def _seeded_issue(number: int = 24) -> TrackerIssue:
    return TrackerIssue(
        id=f"issue-{number}",
        number=number,
        title="Continue",
        body="Keep going.",
        created_at=datetime(2026, 9, 20, tzinfo=UTC),
        state="OPEN",
        labels=frozenset({"round-finished"}),
        assignee_logins=(),
        blocked_by=0,
        author_login="reporter",
    )


def test_fake_tracker_seeded_comments_use_shared_authority() -> None:
    tracker = FakeTracker(
        issues={24: _seeded_issue()},
        comments={24: [("agent", shared_body(A2, 2))]},
    )
    assert tracker.automatic_continuation_count(_seeded_issue()) == 2


def test_fake_tracker_seeded_human_reset_hides_shared() -> None:
    tracker = FakeTracker(
        issues={24: _seeded_issue()},
        comments={
            24: [
                ("agent", shared_body(A2, 4)),
                ("agent", result_body(A3, REQUEUE_SOURCE_HUMAN)),
            ]
        },
    )
    assert tracker.automatic_continuation_count(_seeded_issue()) == 0


def test_fake_tracker_seeded_ambiguity_raises() -> None:
    tracker = FakeTracker(
        issues={24: _seeded_issue()},
        comments={24: [("agent", shared_body(A2, 1)), ("agent", shared_body(A3, 2))]},
    )
    with pytest.raises(AmbiguousContinuationError):
        tracker.automatic_continuation_count(_seeded_issue())


def test_resolver_unmarked_human_prose_does_not_reset_shared() -> None:
    bodies = (
        "A human reviewer note with no machine markers at all.",
        shared_body(A2, 2),
    )
    assert resolve_automatic_continuation_count(bodies) == 2


# ---------------------------------------------------------------------------
# lifecycle: stored intent wins; duplicates hold; re-read guards the flag
# ---------------------------------------------------------------------------


def test_handoff_used_prefers_the_stored_intent(tmp_path: Path) -> None:
    from simple_coding_agent.handoff_record import HandoffRecord

    lifecycle = _hold_lifecycle(tmp_path, object())
    stored = HandoffRecord(
        version=HANDOFF_RECORD_VERSION,
        repository=REPO,
        issue_number=24,
        attempt_id="2026-09-20T13:00:00Z",
        requeue_source=REQUEUE_SOURCE_AUTOMATIC,
        continuation_count=2,
        branch="agent/issue-24",
        branch_sha="b" * 40,
        note_sha="c" * 40,
    )
    lifecycle._publisher.handoff_intent = lambda attempt_id: stored  # type: ignore[attr-defined]
    from simple_coding_agent.github_tracker import Claim as TrackerClaim
    from simple_coding_agent.github_tracker import Assignment as TrackerAssignment

    claim = TrackerClaim(_seeded_issue(), TrackerAssignment("issue-24", "agent-id"))

    class _Checkpoint:
        started_at = "2026-09-20T13:00:00Z"
        requeue_source = REQUEUE_SOURCE_AUTOMATIC

    assert lifecycle._handoff_continuation_used(claim, _Checkpoint()) == 2  # type: ignore[arg-type]


def test_shared_hold_duplicate_records_retain(tmp_path: Path) -> None:
    from simple_coding_agent.github_tracker import GitHubTrackerError

    lifecycle = _hold_lifecycle(
        tmp_path,
        _HoldGitHub(error=GitHubTrackerError("more than one shared handoff comment")),
    )
    assert (
        lifecycle._recovery_shared_hold(24, A1, AttemptOutcome.HANDOFF) is not None
    )


def _shared_retry_lifecycle(tmp_path: Path, make_github, *, comment_posted: bool):
    state = AttemptStateStore(tmp_path)
    state.start(issue_number=24, branch="agent/issue-24")
    for phase in (
        AttemptPhase.SETUP,
        AttemptPhase.MODEL_RUNNING,
        AttemptPhase.PUSHING,
        AttemptPhase.PUBLISHING,
    ):
        state.transition(phase)
    checkpoint = state.read()
    assert checkpoint is not None

    workspace = InMemoryWorkspace(tmp_path / "workspace", issue_number=24)
    note_commit = workspace.commit_handoff_note(
        24, _note_content(checkpoint.started_at)
    )

    publisher = FakePublisher(
        outcome=AttemptOutcome.HANDOFF,
        comment_posted=comment_posted,
        handoff_posted=True,
    )
    publisher._github = make_github(checkpoint.started_at, note_commit.revision)  # type: ignore[attr-defined]
    publisher._repository = REPO  # type: ignore[attr-defined]
    lifecycle = AgentLifecycle(
        tracker=FakeTracker(issues={24: _recovery_issue()}),
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=publisher,
        attempt_runner=lambda c, p, pr: (_ for _ in ()).throw(
            AssertionError("recovery must not invoke the model")
        ),
        control_store=ControlStore(tmp_path),
        completion_store=AttemptCompletionStore(tmp_path),
        sleeper=lambda seconds: None,
        repository=REPO,
    )
    return lifecycle, checkpoint


def test_retry_rereads_shared_record_when_the_seam_exists(tmp_path: Path) -> None:
    def make_github(attempt_id: str, note_sha: str) -> _HoldGitHub:
        return _HoldGitHub(
            HandoffRecordRef(
                comment_id="c1",
                body=shared_body(
                    attempt_id, 0, source=REQUEUE_SOURCE_HUMAN, note_sha=note_sha
                ),
            )
        )

    lifecycle, checkpoint = _shared_retry_lifecycle(
        tmp_path, make_github, comment_posted=False
    )
    record = lifecycle.submit_recovery_retry("req-reread", checkpoint.started_at)
    assert record.acknowledgement is CommandAcknowledgement.COMPLETED
    assert AttemptStateStore(tmp_path).read() is None
    assert len(AttemptCompletionStore(tmp_path).read_all()) == 1


def test_retry_holds_when_the_flag_has_no_shared_record(tmp_path: Path) -> None:
    lifecycle, checkpoint = _shared_retry_lifecycle(
        tmp_path, lambda attempt_id, note_sha: _HoldGitHub(None), comment_posted=False
    )
    record = lifecycle.submit_recovery_retry("req-flag-only", checkpoint.started_at)
    assert record.acknowledgement is not CommandAcknowledgement.COMPLETED
    assert AttemptStateStore(tmp_path).read() is not None


def test_retry_holds_when_shared_note_revision_mismatches(tmp_path: Path) -> None:
    def make_github(attempt_id: str, note_sha: str) -> _HoldGitHub:
        return _HoldGitHub(
            HandoffRecordRef(
                comment_id="c1",
                body=shared_body(
                    attempt_id,
                    0,
                    source=REQUEUE_SOURCE_HUMAN,
                    note_sha="0" * 40,
                ),
            )
        )

    lifecycle, checkpoint = _shared_retry_lifecycle(
        tmp_path, make_github, comment_posted=False
    )
    record = lifecycle.submit_recovery_retry("req-note-mismatch", checkpoint.started_at)
    assert record.acknowledgement is not CommandAcknowledgement.COMPLETED
    assert AttemptStateStore(tmp_path).read() is not None
    assert len(AttemptCompletionStore(tmp_path).read_all()) == 0


class _ScriptedGitHub:
    """A shared-record seam that replays a scripted find sequence."""

    def __init__(self, script: list) -> None:
        self._script = list(script)

    def find_handoff_record(self, repository: str, issue_number: int):
        result = self._script.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_retry_lost_response_then_heal_replays_same_budget(tmp_path: Path) -> None:
    seen: list = []

    def make_github(attempt_id: str, note_sha: str) -> _ScriptedGitHub:
        ref = HandoffRecordRef(
            comment_id="c1",
            body=shared_body(
                attempt_id, 0, source=REQUEUE_SOURCE_HUMAN, note_sha=note_sha
            ),
        )
        seen.append(ref)
        # The pre-publish hold sees the record, but the post-publish
        # confirmation loses the response: the retry must retain the
        # attempt instead of treating the flag as proof.
        return _ScriptedGitHub([ref, None])

    lifecycle, checkpoint = _shared_retry_lifecycle(
        tmp_path, make_github, comment_posted=False
    )
    held = lifecycle.submit_recovery_retry("req-lost", checkpoint.started_at)
    assert held.acknowledgement is not CommandAcknowledgement.COMPLETED
    assert AttemptStateStore(tmp_path).read() is not None
    assert len(AttemptCompletionStore(tmp_path).read_all()) == 0

    # The record is readable again: repeating the retry replays the same
    # computed standing instead of consuming the budget twice, and the
    # attempt finalizes exactly once.
    lifecycle._publisher._github = _HoldGitHub(seen[0])  # type: ignore[attr-defined]
    done = lifecycle.submit_recovery_retry("req-healed", checkpoint.started_at)
    assert done.acknowledgement is CommandAcknowledgement.COMPLETED
    counts = [
        request.handoff_continuation_count  # type: ignore[attr-defined]
        for request in lifecycle._publisher.requests  # type: ignore[attr-defined]
    ]
    assert len(counts) == 2
    assert counts[0] == counts[1]
    assert len(AttemptCompletionStore(tmp_path).read_all()) == 1
