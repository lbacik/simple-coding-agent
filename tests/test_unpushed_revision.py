"""Unpushed-revision record and recovery (issue #400).

When an implementation attempt's branch push fails, the publisher durably
records the local tip (the Unpushed revision) with the attempt's claim-time
lease. On that Worker's next live implementation claim of the same issue the
lifecycle publishes the recorded revision when the remote branch still stands
at the recorded lease, and otherwise keeps it as a Preserved revision under a
local ref, so attempt preparation restoring from the remote never silently
orphans it. Recovering never changes the failed attempt's Recorded verdict.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateError, AttemptStateStore
from simple_coding_agent.completion import AttemptOutcome, CompletionDecision, PublicationPath
from simple_coding_agent.control import ControlStore
from simple_coding_agent.finalization import AttemptCompletionStore
from simple_coding_agent.git_workspace import (
    BranchMovedSinceClaimError,
    GitWorkspace,
    GitWorkspaceError,
)
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.lifecycle import AgentLifecycle, AttemptEvidence
from simple_coding_agent.model_execution import ModelExecutionStatus
from simple_coding_agent.publication import PublicationRequest, Publisher
from simple_coding_agent.unpushed_revision import (
    RECOVERY_ALREADY_PUBLISHED,
    RECOVERY_PRESERVED,
    RECOVERY_RECOVERED,
    UnpushedRevisionRecord,
    UnpushedRevisionStore,
    UnpushedRevisionStoreError,
    preserved_comment_line,
    preserved_ref_for_issue_attempt,
    recovered_comment_line,
)
from tests.fakes import (
    FakeGitHub,
    FakeModelExecutor,
    FakePublisher,
    FakeTracker,
    InMemoryWorkspace,
    build_lifecycle,
    git,
    lifecycle_issue,
    publish_attempt_branch,
    repository_with_main,
    run_command,
    write_and_commit,
)
from tests.test_publication import FakeGitHub as CommentGitHub

REMOTE_SHA = "b" * 40
LOCAL_SHA = "a" * 40
RIVAL_SHA = "c" * 40


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def publishing_request(branch: str = "agent/issue-23") -> PublicationRequest:
    return PublicationRequest(
        23,
        "Implement publication",
        branch,
        "2026-09-20T13:00:00Z",
        CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)),
        "pytest",
        1,
        0,
        "all clear",
        "Check did not pass.",
        "main",
    )


def publishing_state(tmp_path: Path) -> AttemptStateStore:
    state = AttemptStateStore(tmp_path / "state")
    state.start(issue_number=23, branch="agent/issue-23")
    state.transition(AttemptPhase.SETUP)
    state.transition(AttemptPhase.MODEL_RUNNING)
    state.transition(AttemptPhase.PUSHING)
    return state


class PushFailingWorkspace:
    """A pusher double whose push fails while the local tip stays readable."""

    def __init__(self, local_revision: str = LOCAL_SHA) -> None:
        self.local_revision = local_revision

    def push_attempt_branch(
        self,
        branch: str,
        *,
        max_retries: int,
        deadline: float | None = None,
        expected_revision: str | None = None,
        lease_known: bool = False,
        issue_number: int | None = None,
    ) -> str:
        raise GitWorkspaceError("remote hung up unexpectedly")

    def attempt_branch_revision(self, branch: str) -> str:
        return self.local_revision


class UnreadableTipWorkspace(PushFailingWorkspace):
    """A pusher double whose push fails and whose local tip cannot be read."""

    def attempt_branch_revision(self, branch: str) -> str:
        raise GitWorkspaceError("Attempt branch revision is unavailable in the local clone")


class BranchMovedWorkspace:
    """A lease-rejection double whose local tip stays readable."""

    def push_attempt_branch(
        self,
        branch: str,
        *,
        max_retries: int,
        deadline: float | None = None,
        expected_revision: str | None = None,
        lease_known: bool = False,
        issue_number: int | None = None,
    ) -> str:
        raise BranchMovedSinceClaimError(
            f"branch moved since claim: remote {branch} is rival-sha,"
            " claim-time lease was absent"
        )

    def attempt_branch_revision(self, branch: str) -> str:
        return LOCAL_SHA


class RecordingWorkspace:
    """A pusher double whose push succeeds and whose tip stays readable."""

    def __init__(self, sha: str = "new-sha") -> None:
        self.sha = sha

    def push_attempt_branch(
        self,
        branch: str,
        *,
        max_retries: int,
        deadline: float | None = None,
        expected_revision: str | None = None,
        lease_known: bool = False,
        issue_number: int | None = None,
    ) -> str:
        return self.sha

    def attempt_branch_revision(self, branch: str) -> str:
        return self.sha


def recording_publisher(
    workspace: object,
    events: list,
    state: AttemptStateStore | None,
    store: UnpushedRevisionStore | None,
) -> tuple[Publisher, CommentGitHub]:
    github = CommentGitHub()
    publisher = Publisher(
        workspace,  # type: ignore[arg-type]
        github,  # type: ignore[arg-type]
        "octo/example",
        attempt_state=state,
        unpushed_store=store,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level, issue_number)
        ),
    )
    return publisher, github


def live_issue(number: int = 24, labels: frozenset[str] = frozenset({"ready-for-agent"})) -> TrackerIssue:
    return TrackerIssue(
        id=f"issue-{number}",
        number=number,
        title="Implement lifecycle",
        body="Connect boundaries.",
        created_at=lifecycle_issue(number).created_at,
        state="OPEN",
        labels=labels,
        assignee_logins=(),
        blocked_by=0,
        author_login="reporter",
    )


def live_profile() -> object:
    return type("Profile", (), {"base_branch": "main"})()


def complete_workflow(
    received_claim: Claim, profile: object, prepared: object
) -> AttemptEvidence:
    return AttemptEvidence(
        decision=CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)),
        check_command="pytest",
        check_exit_code=0,
        review_cycles=1,
        review_findings="all clear",
        details="Implemented the lifecycle.",
    )


def run_live_claim(
    tmp_path: Path,
    *,
    record: UnpushedRevisionRecord | None = None,
    remote_branches: dict[str, str] | None = None,
    remote_unavailable: bool = False,
    recovery_push_error: Exception | None = None,
    labels: frozenset[str] = frozenset({"ready-for-agent"}),
    store_dir: str = "data",
) -> dict:
    """Drive one live implementation claim with an optional seeded record."""

    claim = Claim(live_issue(24, labels), Assignment("issue-24", "agent-id"))
    tracker = FakeTracker(next_claim=claim)
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
    if remote_branches:
        workspace.remote_branches.update(remote_branches)
    workspace.remote_unavailable = remote_unavailable
    workspace.recovery_push_error = recovery_push_error
    store = UnpushedRevisionStore(tmp_path / store_dir)
    if record is not None:
        store.save(record)
    events: list[tuple[str, str, str, int | None]] = []
    publisher = FakePublisher()
    lifecycle = AgentLifecycle(
        sleeper=lambda _: None,
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path / "state"),
        workspace=workspace,
        profile_loader=lambda _: live_profile(),
        publisher=publisher,
        attempt_runner=complete_workflow,
        unpushed_store=store,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level, issue_number)
        ),
    )
    result = lifecycle.run_once()
    return {
        "result": result,
        "tracker": tracker,
        "workspace": workspace,
        "store": store,
        "events": events,
        "publisher": publisher,
    }


# ---------------------------------------------------------------------------
# store + builders
# ---------------------------------------------------------------------------


def test_store_roundtrip_with_known_lease(tmp_path: Path) -> None:
    store = UnpushedRevisionStore(tmp_path / "data")
    record = UnpushedRevisionRecord(
        issue_number=23, revision=LOCAL_SHA, lease_revision=REMOTE_SHA, attempt_id="attempt-1"
    )

    store.save(record)

    assert store.read(23) == record
    assert store.read_all() == {23: record}


def test_store_roundtrip_with_absent_lease(tmp_path: Path) -> None:
    store = UnpushedRevisionStore(tmp_path / "data")
    record = UnpushedRevisionRecord(
        issue_number=23, revision=LOCAL_SHA, lease_revision=None, attempt_id="attempt-1"
    )

    store.save(record)

    assert store.read(23) == record


def test_store_replaces_older_record_for_same_issue(tmp_path: Path) -> None:
    store = UnpushedRevisionStore(tmp_path / "data")
    store.save(
        UnpushedRevisionRecord(
            issue_number=23, revision=LOCAL_SHA, lease_revision=REMOTE_SHA, attempt_id="first"
        )
    )

    store.save(
        UnpushedRevisionRecord(
            issue_number=23, revision=RIVAL_SHA, lease_revision=None, attempt_id="second"
        )
    )

    assert store.read(23) == UnpushedRevisionRecord(
        issue_number=23, revision=RIVAL_SHA, lease_revision=None, attempt_id="second"
    )


def test_store_keeps_one_record_per_issue(tmp_path: Path) -> None:
    store = UnpushedRevisionStore(tmp_path / "data")
    store.save(
        UnpushedRevisionRecord(
            issue_number=23, revision=LOCAL_SHA, lease_revision=None, attempt_id="a"
        )
    )
    store.save(
        UnpushedRevisionRecord(
            issue_number=24, revision=RIVAL_SHA, lease_revision=None, attempt_id="b"
        )
    )

    assert set(store.read_all()) == {23, 24}


def test_store_clear_reports_whether_a_record_existed(tmp_path: Path) -> None:
    store = UnpushedRevisionStore(tmp_path / "data")

    assert store.clear(23) is False
    store.save(
        UnpushedRevisionRecord(
            issue_number=23, revision=LOCAL_SHA, lease_revision=None, attempt_id="a"
        )
    )

    assert store.clear(23) is True
    assert store.read(23) is None
    assert store.clear(23) is False


def test_store_missing_file_reads_as_absent(tmp_path: Path) -> None:
    assert UnpushedRevisionStore(tmp_path / "data").read(23) is None


def test_store_malformed_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "data" / "state"
    path.mkdir(parents=True)
    (path / "unpushed_revisions.json").write_text("{not json")

    with pytest.raises(UnpushedRevisionStoreError):
        UnpushedRevisionStore(tmp_path / "data").read(23)


def test_store_wrong_shape_raises(tmp_path: Path) -> None:
    path = tmp_path / "data" / "state"
    path.mkdir(parents=True)
    (path / "unpushed_revisions.json").write_text(json.dumps({"23": {"revision": LOCAL_SHA}}))

    with pytest.raises(UnpushedRevisionStoreError):
        UnpushedRevisionStore(tmp_path / "data").read_all()


def test_store_rejects_invalid_record(tmp_path: Path) -> None:
    store = UnpushedRevisionStore(tmp_path / "data")

    with pytest.raises(UnpushedRevisionStoreError):
        store.save(
            UnpushedRevisionRecord(
                issue_number=0, revision=LOCAL_SHA, lease_revision=None, attempt_id="a"
            )
        )


def test_preserved_ref_names_issue_and_attempt() -> None:
    assert (
        preserved_ref_for_issue_attempt(24, "attempt-1")
        == "refs/agent/preserved/issue-24/attempt-1"
    )


def test_preserved_ref_sanitizes_ref_unsafe_characters() -> None:
    ref = preserved_ref_for_issue_attempt(24, "2026-09-20T13:00:00Z")

    assert ref == "refs/agent/preserved/issue-24/2026-09-20T13-00-00Z"
    assert preserved_ref_for_issue_attempt(24, "2026-09-20T13:00:00Z") == ref


def test_preserved_ref_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        preserved_ref_for_issue_attempt(0, "attempt-1")
    with pytest.raises(ValueError):
        preserved_ref_for_issue_attempt(24, "   ")
    with pytest.raises(ValueError):
        preserved_ref_for_issue_attempt(24, ":::")


def test_comment_lines() -> None:
    assert recovered_comment_line(LOCAL_SHA, "attempt-1") == (
        f"**Recovered unpushed revision**: `{LOCAL_SHA}` from attempt `attempt-1`"
    )
    assert "(already published)" in recovered_comment_line(
        LOCAL_SHA, "attempt-1", already_published=True
    )
    assert preserved_comment_line(LOCAL_SHA, "attempt-1", "refs/r", "moved") == (
        f"**Preserved unpushed revision**: `{LOCAL_SHA}` from attempt `attempt-1`"
        " at `refs/r` (moved)"
    )


# ---------------------------------------------------------------------------
# publisher recording
# ---------------------------------------------------------------------------


def test_push_failure_with_known_lease_writes_record(tmp_path: Path) -> None:
    events: list = []
    state = publishing_state(tmp_path)
    state.record_claim_lease(REMOTE_SHA)
    store = UnpushedRevisionStore(tmp_path / "data")
    publisher, github = recording_publisher(PushFailingWorkspace(), events, state, store)

    result = publisher.publish(publishing_request())

    assert result.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    assert store.read(23) == UnpushedRevisionRecord(
        issue_number=23,
        revision=LOCAL_SHA,
        lease_revision=REMOTE_SHA,
        attempt_id="2026-09-20T13:00:00Z",
    )
    # The failed attempt's outcome, result comment, and publication_failed
    # event are exactly what the push failure reports.
    (comment,) = [body for _, body in github.comments]
    assert "**Unpushed local revision**: `" + LOCAL_SHA + "`" in comment
    assert LOCAL_SHA in result.details
    failures = [event for event in events if event[0] == "publication_failed"]
    assert len(failures) == 1
    assert LOCAL_SHA in failures[0][1]


def test_branch_moved_failure_writes_record(tmp_path: Path) -> None:
    events: list = []
    state = publishing_state(tmp_path)
    state.record_claim_lease(REMOTE_SHA)
    store = UnpushedRevisionStore(tmp_path / "data")
    publisher, _ = recording_publisher(BranchMovedWorkspace(), events, state, store)

    result = publisher.publish(publishing_request())

    assert result.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    assert store.read(23) == UnpushedRevisionRecord(
        issue_number=23,
        revision=LOCAL_SHA,
        lease_revision=REMOTE_SHA,
        attempt_id="2026-09-20T13:00:00Z",
    )


def test_push_failure_with_absent_lease_records_absent(tmp_path: Path) -> None:
    events: list = []
    state = publishing_state(tmp_path)
    state.record_claim_lease(None)
    store = UnpushedRevisionStore(tmp_path / "data")
    publisher, _ = recording_publisher(PushFailingWorkspace(), events, state, store)

    publisher.publish(publishing_request())

    assert store.read(23) == UnpushedRevisionRecord(
        issue_number=23,
        revision=LOCAL_SHA,
        lease_revision=None,
        attempt_id="2026-09-20T13:00:00Z",
    )


def test_push_failure_with_unknown_lease_writes_no_record(tmp_path: Path) -> None:
    events: list = []
    state = publishing_state(tmp_path)
    store = UnpushedRevisionStore(tmp_path / "data")
    publisher, _ = recording_publisher(PushFailingWorkspace(), events, state, store)

    publisher.publish(publishing_request())

    assert store.read(23) is None
    assert [event for event in events if event[0] == "unpushed_revision_record_failed"] == []


def test_claim_lease_error_writes_no_record(tmp_path: Path) -> None:
    class AdvanceFailingStore(AttemptStateStore):
        def advance_claim_lease(self, revision: str | None):  # type: ignore[override]
            raise AttemptStateError("disk is gone")

    events: list = []
    inner = publishing_state(tmp_path)
    inner.record_claim_lease(REMOTE_SHA)
    state = AdvanceFailingStore(tmp_path)
    store = UnpushedRevisionStore(tmp_path / "data")
    publisher, _ = recording_publisher(RecordingWorkspace(), events, state, store)

    result = publisher.publish(publishing_request())

    assert result.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    assert "Claim lease advance failed" in result.details
    assert store.read(23) is None
    assert [event for event in events if event[0] == "unpushed_revision_record_failed"] == []


def test_unreadable_revision_emits_error_and_writes_no_record(tmp_path: Path) -> None:
    events: list = []
    state = publishing_state(tmp_path)
    state.record_claim_lease(REMOTE_SHA)
    store = UnpushedRevisionStore(tmp_path / "data")
    publisher, github = recording_publisher(UnreadableTipWorkspace(), events, state, store)

    result = publisher.publish(publishing_request())

    assert result.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    assert store.read(23) is None
    (comment,) = [body for _, body in github.comments]
    assert "Unpushed local revision" not in comment
    failures = [event for event in events if event[0] == "unpushed_revision_record_failed"]
    assert len(failures) == 1
    assert failures[0][2] == "ERROR"
    assert failures[0][3] == 23


def test_unreadable_lease_emits_error_and_writes_no_record(tmp_path: Path) -> None:
    class FlakyLeaseState:
        """A checkpoint store failing only at the Unpushed-record probe.

        The publish flow reads the checkpoint twice before recording (the
        pre-push phase check and the failure report's lease lookup); the
        third read -- the record probe -- hits the transient I/O failure.
        """

        def __init__(self, inner: AttemptStateStore) -> None:
            self._inner = inner
            self._reads = 0

        def read(self):  # type: ignore[no-untyped-def]
            self._reads += 1
            if self._reads >= 3:
                raise AttemptStateError("checkpoint is gone")
            return self._inner.read()

    events: list = []
    inner = publishing_state(tmp_path)
    inner.record_claim_lease(REMOTE_SHA)
    store = UnpushedRevisionStore(tmp_path / "data")
    publisher, github = recording_publisher(
        PushFailingWorkspace(), events, FlakyLeaseState(inner), store
    )

    result = publisher.publish(publishing_request())

    assert result.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    assert store.read(23) is None
    failures = [event for event in events if event[0] == "publication_failed"]
    assert len(failures) == 1
    assert LOCAL_SHA in failures[0][1]
    record_failures = [
        event for event in events if event[0] == "unpushed_revision_record_failed"
    ]
    assert len(record_failures) == 1
    assert record_failures[0][2] == "ERROR"
    assert record_failures[0][3] == 23
    assert "could not be read" in record_failures[0][1]
    (comment,) = [body for _, body in github.comments]
    assert "**Unpushed local revision**: `" + LOCAL_SHA + "`" in comment


def test_unwritable_store_emits_error_and_keeps_outcome(tmp_path: Path) -> None:
    events: list = []
    state = publishing_state(tmp_path)
    state.record_claim_lease(REMOTE_SHA)

    class BrokenStore(UnpushedRevisionStore):
        def save(self, record: UnpushedRevisionRecord) -> None:
            raise UnpushedRevisionStoreError("disk is gone")

    publisher, github = recording_publisher(
        PushFailingWorkspace(), events, state, BrokenStore(tmp_path / "data")
    )

    result = publisher.publish(publishing_request())

    assert result.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    (comment,) = [body for _, body in github.comments]
    assert "**Unpushed local revision**: `" + LOCAL_SHA + "`" in comment
    failures = [event for event in events if event[0] == "unpushed_revision_record_failed"]
    assert len(failures) == 1
    assert failures[0][2] == "ERROR"


def test_later_failure_replaces_record_for_same_issue(tmp_path: Path) -> None:
    events: list = []
    state = publishing_state(tmp_path)
    state.record_claim_lease(REMOTE_SHA)
    store = UnpushedRevisionStore(tmp_path / "data")
    store.save(
        UnpushedRevisionRecord(
            issue_number=23, revision=RIVAL_SHA, lease_revision=None, attempt_id="older"
        )
    )
    publisher, _ = recording_publisher(PushFailingWorkspace(), events, state, store)

    publisher.publish(publishing_request())

    assert store.read(23) == UnpushedRevisionRecord(
        issue_number=23,
        revision=LOCAL_SHA,
        lease_revision=REMOTE_SHA,
        attempt_id="2026-09-20T13:00:00Z",
    )


# ---------------------------------------------------------------------------
# GitWorkspace recovery operations against a real remote
# ---------------------------------------------------------------------------


def cloned_workspace(
    tmp_path: Path, remote: Path, *, issue_number: int = 24
) -> tuple[GitWorkspace, Path]:
    """A production workspace over a pre-cloned checkout with test identity."""

    clone = tmp_path / "clone"
    git(tmp_path, "clone", str(remote), str(clone))
    git(clone, "config", "user.name", "Test User")
    git(clone, "config", "user.email", "test@example.com")
    return (
        GitWorkspace(clone, str(remote), token_provider=lambda: "token"),
        clone,
    )


def test_remote_branch_revision_is_none_when_absent(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace, _ = cloned_workspace(tmp_path, remote)

    assert workspace.remote_branch_revision("agent/issue-24") is None


def test_remote_branch_revision_reads_the_pushed_tip(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    publish_attempt_branch(tmp_path, remote, 24, {"work.txt": "done"}, "Add work")
    workspace, _ = cloned_workspace(tmp_path, remote)

    assert (
        workspace.remote_branch_revision("agent/issue-24")
        == git(remote, "rev-parse", "refs/heads/agent/issue-24")
    )


def test_push_branch_revision_publishes_under_an_absent_lease(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace, clone = cloned_workspace(tmp_path, remote)
    workspace.prepare_attempt(base_branch="main", issue_number=24)
    write_and_commit(clone, "work.txt", "done", "Add work")
    revision = git(clone, "rev-parse", "HEAD")

    assert (
        workspace.push_branch_revision(
            "agent/issue-24", revision, expected_revision=None, issue_number=24
        )
        == revision
    )
    assert workspace.remote_branch_revision("agent/issue-24") == revision


def test_push_branch_revision_overwrites_at_the_recorded_lease(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    publish_attempt_branch(tmp_path, remote, 24, {"work.txt": "v1"}, "Add work")
    lease = git(remote, "rev-parse", "refs/heads/agent/issue-24")
    workspace, clone = cloned_workspace(tmp_path, remote)
    workspace.prepare_attempt(base_branch="main", issue_number=24)
    write_and_commit(clone, "work.txt", "v2", "Improve work")
    revision = git(clone, "rev-parse", "HEAD")

    assert (
        workspace.push_branch_revision(
            "agent/issue-24", revision, expected_revision=lease, issue_number=24
        )
        == revision
    )
    assert workspace.remote_branch_revision("agent/issue-24") == revision


def test_push_branch_revision_rejects_a_moved_remote(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    publish_attempt_branch(tmp_path, remote, 24, {"work.txt": "v1"}, "Add work")
    lease = git(remote, "rev-parse", "refs/heads/agent/issue-24")
    workspace, clone = cloned_workspace(tmp_path, remote)
    workspace.prepare_attempt(base_branch="main", issue_number=24)
    write_and_commit(clone, "work.txt", "v2", "Improve work")
    revision = git(clone, "rev-parse", "HEAD")
    other = tmp_path / "other"
    git(tmp_path, "clone", str(remote), str(other))
    git(other, "config", "user.name", "Test User")
    git(other, "config", "user.email", "test@example.com")
    git(other, "checkout", "agent/issue-24")
    write_and_commit(other, "rival.txt", "rival", "Rival work")
    git(other, "push", "origin", "agent/issue-24")
    moved = git(remote, "rev-parse", "refs/heads/agent/issue-24")
    assert moved != lease

    with pytest.raises(BranchMovedSinceClaimError):
        workspace.push_branch_revision(
            "agent/issue-24", revision, expected_revision=lease, issue_number=24
        )

    assert workspace.remote_branch_revision("agent/issue-24") == moved


def test_push_branch_revision_is_idempotent_when_already_published(
    tmp_path: Path,
) -> None:
    remote, _ = repository_with_main(tmp_path)
    publish_attempt_branch(tmp_path, remote, 24, {"work.txt": "v1"}, "Add work")
    revision = git(remote, "rev-parse", "refs/heads/agent/issue-24")
    workspace, _ = cloned_workspace(tmp_path, remote)

    assert (
        workspace.push_branch_revision(
            "agent/issue-24", revision, expected_revision="stale-lease", issue_number=24
        )
        == revision
    )


def test_preserve_revision_keeps_a_non_branch_ref(tmp_path: Path) -> None:
    remote, _ = repository_with_main(tmp_path)
    workspace, clone = cloned_workspace(tmp_path, remote)
    workspace.prepare_attempt(base_branch="main", issue_number=24)
    write_and_commit(clone, "work.txt", "done", "Add work")
    revision = git(clone, "rev-parse", "HEAD")
    ref = preserved_ref_for_issue_attempt(24, "attempt-1")

    assert workspace.preserve_revision(ref, revision) == ref
    assert git(clone, "rev-parse", ref) == revision
    assert "preserved" not in git(clone, "branch", "--list")


# ---------------------------------------------------------------------------
# lifecycle decision table against real git
# ---------------------------------------------------------------------------


def recovery_lifecycle(
    tmp_path: Path,
    workspace: GitWorkspace,
    store: UnpushedRevisionStore,
    events: list,
) -> AgentLifecycle:
    return AgentLifecycle(
        sleeper=lambda _: None,
        tracker=FakeTracker(),
        attempt_state=AttemptStateStore(tmp_path / "cycle"),
        workspace=workspace,  # type: ignore[arg-type]
        profile_loader=lambda _: live_profile(),
        publisher=FakePublisher(),
        unpushed_store=store,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level, issue_number)
        ),
    )


def seed_unpushed_clone(
    tmp_path: Path, *, with_remote_branch: bool
) -> tuple[Path, GitWorkspace, Path, str, str | None]:
    """A clone whose attempt branch holds one unpushed commit.

    Returns ``(remote, workspace, clone, revision, lease)`` where ``lease``
    is the remote tip observed at claim time (``None`` when the branch was
    still absent).
    """

    remote, _ = repository_with_main(tmp_path)
    lease: str | None = None
    if with_remote_branch:
        publish_attempt_branch(tmp_path, remote, 24, {"work.txt": "v1"}, "Add work")
        lease = git(remote, "rev-parse", "refs/heads/agent/issue-24")
    workspace, clone = cloned_workspace(tmp_path, remote)
    workspace.prepare_attempt(base_branch="main", issue_number=24)
    write_and_commit(clone, "work.txt", "v2", "Improve work")
    return remote, workspace, clone, git(clone, "rev-parse", "HEAD"), lease


def test_recovery_pushes_when_remote_tip_equals_lease(tmp_path: Path) -> None:
    remote, workspace, clone, revision, lease = seed_unpushed_clone(
        tmp_path, with_remote_branch=True
    )
    assert lease is not None
    store = UnpushedRevisionStore(tmp_path / "data")
    store.save(
        UnpushedRevisionRecord(
            issue_number=24, revision=revision, lease_revision=lease, attempt_id="attempt-1"
        )
    )
    events: list = []
    lifecycle = recovery_lifecycle(tmp_path, workspace, store, events)

    recovery = lifecycle._recover_unpushed_revision(24)

    assert recovery is not None
    assert recovery.outcome == RECOVERY_RECOVERED
    assert recovery.comment_line == recovered_comment_line(revision, "attempt-1")
    assert git(remote, "rev-parse", "refs/heads/agent/issue-24") == revision
    assert store.read(24) is None
    recovered = [event for event in events if event[0] == "unpushed_revision_recovered"]
    assert len(recovered) == 1
    assert recovered[0][2] == "INFO"
    assert revision in recovered[0][1]
    assert "attempt-1" in recovered[0][1]


def test_recovery_pushes_when_absent_lease_meets_absent_branch(tmp_path: Path) -> None:
    _, workspace, _, revision, lease = seed_unpushed_clone(tmp_path, with_remote_branch=False)
    assert lease is None
    store = UnpushedRevisionStore(tmp_path / "data")
    store.save(
        UnpushedRevisionRecord(
            issue_number=24, revision=revision, lease_revision=None, attempt_id="attempt-1"
        )
    )
    events: list = []
    lifecycle = recovery_lifecycle(tmp_path, workspace, store, events)

    recovery = lifecycle._recover_unpushed_revision(24)

    assert recovery is not None
    assert recovery.outcome == RECOVERY_RECOVERED
    assert workspace.remote_branch_revision("agent/issue-24") == revision
    assert store.read(24) is None


def test_recovery_reports_already_published_without_pushing(tmp_path: Path) -> None:
    remote, workspace, clone, revision, lease = seed_unpushed_clone(
        tmp_path, with_remote_branch=True
    )
    git(clone, "push", "origin", "agent/issue-24")
    assert git(remote, "rev-parse", "refs/heads/agent/issue-24") == revision
    store = UnpushedRevisionStore(tmp_path / "data")
    store.save(
        UnpushedRevisionRecord(
            issue_number=24, revision=revision, lease_revision=lease, attempt_id="attempt-1"
        )
    )
    events: list = []
    lifecycle = recovery_lifecycle(tmp_path, workspace, store, events)

    recovery = lifecycle._recover_unpushed_revision(24)

    assert recovery is not None
    assert recovery.outcome == RECOVERY_ALREADY_PUBLISHED
    assert "(already published)" in recovery.comment_line
    assert revision in recovery.comment_line
    assert store.read(24) is None
    assert [event for event in events if event[0] == "unpushed_revision_recovered"]


def test_recovery_preserves_when_remote_moved(tmp_path: Path) -> None:
    remote, workspace, clone, revision, lease = seed_unpushed_clone(
        tmp_path, with_remote_branch=True
    )
    assert lease is not None
    other = tmp_path / "other"
    git(tmp_path, "clone", str(remote), str(other))
    git(other, "config", "user.name", "Test User")
    git(other, "config", "user.email", "test@example.com")
    git(other, "checkout", "agent/issue-24")
    write_and_commit(other, "rival.txt", "rival", "Rival work")
    git(other, "push", "origin", "agent/issue-24")
    moved = git(remote, "rev-parse", "refs/heads/agent/issue-24")
    assert moved != lease and moved != revision
    store = UnpushedRevisionStore(tmp_path / "data")
    store.save(
        UnpushedRevisionRecord(
            issue_number=24, revision=revision, lease_revision=lease, attempt_id="attempt-1"
        )
    )
    events: list = []
    lifecycle = recovery_lifecycle(tmp_path, workspace, store, events)

    recovery = lifecycle._recover_unpushed_revision(24)

    assert recovery is not None
    assert recovery.outcome == RECOVERY_PRESERVED
    assert recovery.ref == preserved_ref_for_issue_attempt(24, "attempt-1")
    assert recovery.reason == "remote moved"
    assert git(remote, "rev-parse", "refs/heads/agent/issue-24") == moved
    assert git(clone, "rev-parse", recovery.ref) == revision
    assert store.read(24) is None
    preserved = [event for event in events if event[0] == "unpushed_revision_preserved"]
    assert len(preserved) == 1
    assert preserved[0][2] == "WARNING"
    assert recovery.ref in preserved[0][1]
    # Preparation restoring from the remote still leaves the kept commits
    # reachable from the Preserved ref.
    workspace.prepare_attempt(base_branch="main", issue_number=24)
    assert git(clone, "rev-parse", "agent/issue-24") == moved
    assert git(clone, "rev-parse", recovery.ref) == revision


def test_recovery_preserves_when_remote_branch_was_deleted(tmp_path: Path) -> None:
    remote, workspace, clone, revision, lease = seed_unpushed_clone(
        tmp_path, with_remote_branch=True
    )
    assert lease is not None
    git(remote, "update-ref", "-d", "refs/heads/agent/issue-24")
    assert workspace.remote_branch_revision("agent/issue-24") is None
    store = UnpushedRevisionStore(tmp_path / "data")
    store.save(
        UnpushedRevisionRecord(
            issue_number=24, revision=revision, lease_revision=lease, attempt_id="attempt-1"
        )
    )
    events: list = []
    lifecycle = recovery_lifecycle(tmp_path, workspace, store, events)

    recovery = lifecycle._recover_unpushed_revision(24)

    assert recovery is not None
    assert recovery.outcome == RECOVERY_PRESERVED
    assert recovery.reason == "remote deleted"
    assert git(clone, "rev-parse", recovery.ref) == revision
    assert store.read(24) is None


def test_recovery_preserves_when_the_recovery_push_is_refused(tmp_path: Path) -> None:
    remote, inner, clone, revision, lease = seed_unpushed_clone(
        tmp_path, with_remote_branch=True
    )
    assert lease is not None

    def refuse_push(
        command: tuple[str, ...], cwd: Path | None, env: dict[str, str]
    ) -> str:
        if command[1] == "push":
            raise subprocess.CalledProcessError(128, command, stderr="hook refused")
        return run_command(command, cwd=cwd, env=env)

    workspace = GitWorkspace(
        clone, str(remote), token_provider=lambda: "token", run=refuse_push
    )
    store = UnpushedRevisionStore(tmp_path / "data")
    store.save(
        UnpushedRevisionRecord(
            issue_number=24, revision=revision, lease_revision=lease, attempt_id="attempt-1"
        )
    )
    events: list = []
    lifecycle = recovery_lifecycle(tmp_path, workspace, store, events)

    recovery = lifecycle._recover_unpushed_revision(24)

    assert recovery is not None
    assert recovery.outcome == RECOVERY_PRESERVED
    assert recovery.reason == "recovery push refused"
    assert "GitCommandError" not in (recovery.reason or "")
    assert git(remote, "rev-parse", "refs/heads/agent/issue-24") == lease
    assert git(clone, "rev-parse", recovery.ref) == revision
    assert store.read(24) is None
    preserved = [event for event in events if event[0] == "unpushed_revision_preserved"]
    assert len(preserved) == 1
    assert preserved[0][2] == "WARNING"
    assert "reason=recovery push refused" in preserved[0][1]
    # The refusal's exception type rides along as event context, not in the
    # reason: the workspace surfaces the hook refusal as a GitCommandError.
    assert "GitCommandError" in preserved[0][1]


def test_recovery_leaves_record_untouched_when_remote_is_unreadable(
    tmp_path: Path,
) -> None:
    remote, _, clone, revision, lease = seed_unpushed_clone(
        tmp_path, with_remote_branch=True
    )

    def fail_ls_remote(
        command: tuple[str, ...], cwd: Path | None, env: dict[str, str]
    ) -> str:
        if command[1] == "ls-remote":
            raise subprocess.CalledProcessError(128, command, stderr="offline")
        return run_command(command, cwd=cwd, env=env)

    workspace = GitWorkspace(
        clone, str(remote), token_provider=lambda: "token", run=fail_ls_remote
    )
    store = UnpushedRevisionStore(tmp_path / "data")
    store.save(
        UnpushedRevisionRecord(
            issue_number=24, revision=revision, lease_revision=lease, attempt_id="attempt-1"
        )
    )
    events: list = []
    lifecycle = recovery_lifecycle(tmp_path, workspace, store, events)

    assert lifecycle._recover_unpushed_revision(24) is None
    assert store.read(24) == UnpushedRevisionRecord(
        issue_number=24, revision=revision, lease_revision=lease, attempt_id="attempt-1"
    )
    assert [event for event in events if event[0].startswith("unpushed_revision_")] == []
    assert git(clone, "for-each-ref", "refs/agent/preserved") == ""


def test_recovery_pushes_a_rebased_revision_the_remote_is_not_ahead_of(
    tmp_path: Path,
) -> None:
    """A moved base rewrites the local branch, so the lease (not ancestry) gates.

    The remote branch still stands at the failed attempt's claim-time lease
    while the recorded Unpushed revision was rebased onto a newer base: the
    lease commit is no ancestor of the revision, yet the recovery push is
    safe because nothing else published since the claim.
    """

    remote, seed = repository_with_main(tmp_path)
    publish_attempt_branch(tmp_path, remote, 24, {"work.txt": "v1"}, "Add work")
    lease = git(remote, "rev-parse", "refs/heads/agent/issue-24")
    write_and_commit(seed, "base2.txt", "new base", "Move the base")
    git(seed, "push", "origin", "main")
    workspace, clone = cloned_workspace(tmp_path, remote)
    # Preparation fetches the moved base and replays the published branch
    # onto it, like the failed attempt's own preparation did.
    workspace.prepare_attempt(base_branch="main", issue_number=24)
    revision = git(clone, "rev-parse", "HEAD")
    assert revision != lease
    with pytest.raises(subprocess.CalledProcessError):
        git(clone, "merge-base", "--is-ancestor", lease, revision)
    store = UnpushedRevisionStore(tmp_path / "data")
    store.save(
        UnpushedRevisionRecord(
            issue_number=24, revision=revision, lease_revision=lease, attempt_id="attempt-1"
        )
    )
    events: list = []
    lifecycle = recovery_lifecycle(tmp_path, workspace, store, events)

    recovery = lifecycle._recover_unpushed_revision(24)

    assert recovery is not None
    assert recovery.outcome == RECOVERY_RECOVERED
    assert git(remote, "rev-parse", "refs/heads/agent/issue-24") == revision
    assert store.read(24) is None


# ---------------------------------------------------------------------------
# lifecycle threading over run_once
# ---------------------------------------------------------------------------


def test_live_claim_threads_recovered_line_into_result_comment(tmp_path: Path) -> None:
    outcome = run_live_claim(
        tmp_path,
        record=UnpushedRevisionRecord(
            issue_number=24,
            revision=LOCAL_SHA,
            lease_revision=REMOTE_SHA,
            attempt_id="attempt-1",
        ),
        remote_branches={"agent/issue-24": REMOTE_SHA},
    )

    assert outcome["result"].outcome is AttemptOutcome.COMPLETE
    assert outcome["workspace"].remote_branches["agent/issue-24"] == LOCAL_SHA
    assert outcome["store"].read(24) is None
    (request,) = outcome["publisher"].requests
    assert request.unpushed_recovery_line == recovered_comment_line(LOCAL_SHA, "attempt-1")
    recovered = [
        event for event in outcome["events"] if event[0] == "unpushed_revision_recovered"
    ]
    assert len(recovered) == 1
    assert recovered[0][2] == "INFO"


def test_live_claim_threads_preserved_line_into_result_comment(tmp_path: Path) -> None:
    outcome = run_live_claim(
        tmp_path,
        record=UnpushedRevisionRecord(
            issue_number=24,
            revision=LOCAL_SHA,
            lease_revision=REMOTE_SHA,
            attempt_id="attempt-1",
        ),
        remote_branches={"agent/issue-24": RIVAL_SHA},
    )

    assert outcome["result"].outcome is AttemptOutcome.COMPLETE
    assert outcome["workspace"].remote_branches["agent/issue-24"] == RIVAL_SHA
    assert outcome["store"].read(24) is None
    ref = preserved_ref_for_issue_attempt(24, "attempt-1")
    assert outcome["workspace"].preserved_refs == {ref: LOCAL_SHA}
    (request,) = outcome["publisher"].requests
    assert request.unpushed_recovery_line == preserved_comment_line(
        LOCAL_SHA, "attempt-1", ref, "remote moved"
    )
    preserved = [
        event for event in outcome["events"] if event[0] == "unpushed_revision_preserved"
    ]
    assert len(preserved) == 1
    assert preserved[0][2] == "WARNING"
    assert ref in preserved[0][1]


def test_second_claim_of_same_issue_does_nothing(tmp_path: Path) -> None:
    first = run_live_claim(
        tmp_path,
        record=UnpushedRevisionRecord(
            issue_number=24,
            revision=LOCAL_SHA,
            lease_revision=REMOTE_SHA,
            attempt_id="attempt-1",
        ),
        remote_branches={"agent/issue-24": REMOTE_SHA},
        store_dir="data",
    )
    assert first["store"].read(24) is None
    tracker = first["tracker"]
    tracker.next_claim = Claim(live_issue(24), Assignment("issue-24", "agent-id"))
    workspace = first["workspace"]
    store = first["store"]
    events: list = []
    publisher = FakePublisher()
    lifecycle = AgentLifecycle(
        sleeper=lambda _: None,
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path / "state"),
        workspace=workspace,
        profile_loader=lambda _: live_profile(),
        publisher=publisher,
        attempt_runner=complete_workflow,
        unpushed_store=store,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level, issue_number)
        ),
    )

    lifecycle.run_once()

    (request,) = publisher.requests
    assert request.unpushed_recovery_line is None
    assert [event for event in events if event[0].startswith("unpushed_revision_")] == []


def test_unreadable_remote_keeps_record_and_continues_from_remote(tmp_path: Path) -> None:
    outcome = run_live_claim(
        tmp_path,
        record=UnpushedRevisionRecord(
            issue_number=24,
            revision=LOCAL_SHA,
            lease_revision=REMOTE_SHA,
            attempt_id="attempt-1",
        ),
        remote_branches={"agent/issue-24": REMOTE_SHA},
        remote_unavailable=True,
    )

    assert outcome["result"].outcome is AttemptOutcome.COMPLETE
    assert outcome["store"].read(24) == UnpushedRevisionRecord(
        issue_number=24, revision=LOCAL_SHA, lease_revision=REMOTE_SHA, attempt_id="attempt-1"
    )
    assert outcome["workspace"].preserved_refs == {}
    (request,) = outcome["publisher"].requests
    assert request.unpushed_recovery_line is None


def test_conflict_claim_leaves_record_untouched(tmp_path: Path) -> None:
    outcome = run_live_claim(
        tmp_path,
        record=UnpushedRevisionRecord(
            issue_number=24,
            revision=LOCAL_SHA,
            lease_revision=REMOTE_SHA,
            attempt_id="attempt-1",
        ),
        remote_branches={"agent/issue-24": REMOTE_SHA},
        labels=frozenset({"merge-conflict"}),
    )

    assert outcome["store"].read(24) == UnpushedRevisionRecord(
        issue_number=24, revision=LOCAL_SHA, lease_revision=REMOTE_SHA, attempt_id="attempt-1"
    )
    assert outcome["workspace"].remote_branches["agent/issue-24"] == REMOTE_SHA
    assert outcome["workspace"].preserved_refs == {}
    assert [event for event in outcome["events"] if event[0].startswith("unpushed_revision_")] == []


def test_startup_reconciliation_leaves_record_untouched(tmp_path: Path) -> None:
    state = AttemptStateStore(tmp_path / "state")
    state.start(issue_number=24, branch="agent/issue-24")
    for phase in (
        AttemptPhase.SETUP,
        AttemptPhase.MODEL_RUNNING,
        AttemptPhase.PUSHING,
    ):
        state.transition(phase)
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
    workspace.remote_branches["agent/issue-24"] = REMOTE_SHA
    workspace.model_commit("completed")
    store = UnpushedRevisionStore(tmp_path / "data")
    store.save(
        UnpushedRevisionRecord(
            issue_number=24,
            revision=LOCAL_SHA,
            lease_revision=REMOTE_SHA,
            attempt_id="attempt-1",
        )
    )
    events: list = []
    lifecycle = AgentLifecycle(
        sleeper=lambda _: None,
        tracker=FakeTracker({24: live_issue(24)}),
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: live_profile(),
        publisher=FakePublisher(),
        unpushed_store=store,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level, issue_number)
        ),
    )

    lifecycle.run_once()

    assert store.read(24) == UnpushedRevisionRecord(
        issue_number=24, revision=LOCAL_SHA, lease_revision=REMOTE_SHA, attempt_id="attempt-1"
    )
    assert workspace.remote_branches["agent/issue-24"] == REMOTE_SHA
    assert workspace.preserved_refs == {}
    assert [event for event in events if event[0].startswith("unpushed_revision_")] == []


# ---------------------------------------------------------------------------
# two-round end to end against real git: failed push, then recovery
# ---------------------------------------------------------------------------


def failing_push(
    command: tuple[str, ...], cwd: Path | None, env: dict[str, str]
) -> str:
    if command[1] == "push":
        raise subprocess.CalledProcessError(128, command, stderr="connection reset")
    return run_command(command, cwd=cwd, env=env)


def test_failed_push_then_next_claim_recovers_from_real_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(name, "Test User" if name.endswith("NAME") else "test@example.com")
    remote, seed = repository_with_main(tmp_path)
    data_dir = tmp_path / "data"
    store = UnpushedRevisionStore(data_dir)
    events_first: list = []
    github = FakeGitHub(lifecycle_issue(23))
    first = build_lifecycle(
        remote,
        tmp_path / "clone",
        data_dir,
        github,
        FakeModelExecutor(
            actions=[lambda wd: write_and_commit(wd, "work.txt", "partial", "Partial work")]
        ),
        workspace_run=failing_push,
        unpushed_store=store,
        event_log=lambda event, detail="", level="INFO", **_: events_first.append(
            (event, detail, level)
        ),
    )

    first.run_once()

    assert [event for event, _, _ in events_first if event == "publication_failed"]
    recorded = store.read(23)
    assert recorded is not None
    assert recorded.lease_revision is None
    clone = tmp_path / "clone"
    assert recorded.revision == git(clone, "rev-parse", "agent/issue-23")
    assert git(remote, "branch", "--list", "agent/issue-23") == ""
    first_comments = [comment.body for comment in github.comments]
    assert any("Agent Attempt Result" in body for body in first_comments)

    # The base moves between rounds: the recorded revision can no longer be
    # judged by ancestry, only by the recorded claim-time lease.
    write_and_commit(seed, "base2.txt", "new base", "Move the base")
    git(seed, "push", "origin", "main")

    recorded_claims: list = []
    original_record_claim_lease = AttemptStateStore.record_claim_lease

    def spy_record_claim_lease(self: AttemptStateStore, revision: str | None):  # type: ignore[no-untyped-def]
        recorded_claims.append(revision)
        return original_record_claim_lease(self, revision)

    monkeypatch.setattr(
        AttemptStateStore, "record_claim_lease", spy_record_claim_lease
    )
    events_second: list = []
    github2 = FakeGitHub(lifecycle_issue(23), comments=tuple(github.comments))
    second = build_lifecycle(
        remote,
        tmp_path / "clone",
        data_dir,
        github2,
        # The scripted model errors out, so round 2 publishes nothing: the
        # recovered remote tip stays intact for the assertions below.
        FakeModelExecutor(status=ModelExecutionStatus.INFRASTRUCTURE_ERROR),
        unpushed_store=store,
        event_log=lambda event, detail="", level="INFO", **_: events_second.append(
            (event, detail, level)
        ),
    )

    result = second.run_once()

    assert result.status.name == "ATTEMPTED"
    assert result.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    assert store.read(23) is None
    assert git(clone, "ls-remote", "origin", "refs/heads/agent/issue-23").startswith(
        recorded.revision
    )
    # The new attempt starts from the recovered revision: preparation
    # observed it on the remote and recorded it as the claim-time lease.
    assert recorded_claims == [recorded.revision]
    recovered = [event for event in events_second if event[0] == "unpushed_revision_recovered"]
    assert len(recovered) == 1
    assert recovered[0][2] == "INFO"
    assert recorded.revision in recovered[0][1]
    assert recorded.attempt_id in recovered[0][1]
    bodies = [comment.body for comment in github2.comments]
    assert any(
        recovered_comment_line(recorded.revision, recorded.attempt_id) in body
        for body in bodies
    )
    # The failed attempt's own result comment is never modified: the copy
    # round 2 inherited still matches the original byte for byte.
    bodies = [comment.body for comment in github2.comments]
    for body in first_comments:
        assert body in bodies


def test_recovery_reports_already_published_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(name, "Test User" if name.endswith("NAME") else "test@example.com")
    remote, _ = repository_with_main(tmp_path)
    publish_attempt_branch(tmp_path, remote, 23, {"work.txt": "v1"}, "Add work")
    revision = git(remote, "rev-parse", "refs/heads/agent/issue-23")
    data_dir = tmp_path / "data"
    store = UnpushedRevisionStore(data_dir)
    store.save(
        UnpushedRevisionRecord(
            issue_number=23,
            revision=revision,
            lease_revision="older-lease",
            attempt_id="attempt-1",
        )
    )
    events: list = []
    github = FakeGitHub(lifecycle_issue(23))
    lifecycle = build_lifecycle(
        remote,
        tmp_path / "clone",
        data_dir,
        github,
        FakeModelExecutor(),
        unpushed_store=store,
        event_log=lambda event, detail="", level="INFO", **_: events.append(
            (event, detail, level)
        ),
    )

    lifecycle.run_once()

    assert store.read(23) is None
    assert git(remote, "rev-parse", "refs/heads/agent/issue-23") == revision
    assert [event for event, _, _ in events if event == "unpushed_revision_recovered"]
    bodies = [comment.body for comment in github.comments]
    assert any("(already published)" in body and revision in body for body in bodies)


# ---------------------------------------------------------------------------
# recovery retry / release and startup reconciliation never recover
# ---------------------------------------------------------------------------


class RecoveryCallSpyWorkspace(InMemoryWorkspace):
    """An InMemoryWorkspace counting Unpushed-recovery calls (expect none)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.recovery_calls: list = []

    def remote_branch_revision(self, branch: str) -> str | None:
        self.recovery_calls.append(("remote_branch_revision", branch))
        return super().remote_branch_revision(branch)

    def push_branch_revision(
        self,
        branch: str,
        revision: str,
        *,
        expected_revision: str | None,
        issue_number: int | None = None,
    ) -> str:
        self.recovery_calls.append(("push_branch_revision", branch, revision))
        return super().push_branch_revision(
            branch, revision, expected_revision=expected_revision, issue_number=issue_number
        )

    def preserve_revision(self, ref: str, revision: str) -> str:
        self.recovery_calls.append(("preserve_revision", ref))
        return super().preserve_revision(ref, revision)


def _retained_lifecycle(
    tmp_path: Path,
    workspace: RecoveryCallSpyWorkspace,
    store: UnpushedRevisionStore,
    events: list,
) -> tuple[AgentLifecycle, object]:
    """A lifecycle holding a retained PUSHING attempt plus a seeded record."""

    state = AttemptStateStore(tmp_path / "state")
    started = state.start(issue_number=24, branch="agent/issue-24")
    for phase in (AttemptPhase.SETUP, AttemptPhase.MODEL_RUNNING, AttemptPhase.PUSHING):
        state.transition(phase)
    lifecycle = AgentLifecycle(
        sleeper=lambda _: None,
        tracker=FakeTracker(issues={24: live_issue(24)}),
        attempt_state=state,
        workspace=workspace,  # type: ignore[arg-type]
        profile_loader=lambda _: live_profile(),
        publisher=FakePublisher(),
        control_store=ControlStore(tmp_path),
        completion_store=AttemptCompletionStore(tmp_path),
        unpushed_store=store,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level, issue_number)
        ),
    )
    return lifecycle, started


def test_recovery_retry_leaves_record_untouched(tmp_path: Path) -> None:
    workspace = RecoveryCallSpyWorkspace(tmp_path / "repo", issue_number=24)
    store = UnpushedRevisionStore(tmp_path / "data")
    seeded = UnpushedRevisionRecord(
        issue_number=24,
        revision=LOCAL_SHA,
        lease_revision=REMOTE_SHA,
        attempt_id="attempt-1",
    )
    store.save(seeded)
    events: list = []
    lifecycle, started = _retained_lifecycle(tmp_path, workspace, store, events)

    lifecycle.submit_recovery_retry("req-retry", started.started_at)

    assert store.read(24) == seeded
    assert workspace.recovery_calls == []
    assert workspace.preserved_refs == {}
    assert [event for event in events if event[0].startswith("unpushed_revision_")] == []


def test_recovery_release_leaves_record_untouched(tmp_path: Path) -> None:
    workspace = RecoveryCallSpyWorkspace(tmp_path / "repo", issue_number=24)
    store = UnpushedRevisionStore(tmp_path / "data")
    seeded = UnpushedRevisionRecord(
        issue_number=24,
        revision=LOCAL_SHA,
        lease_revision=REMOTE_SHA,
        attempt_id="attempt-1",
    )
    store.save(seeded)
    events: list = []
    lifecycle, started = _retained_lifecycle(tmp_path, workspace, store, events)

    lifecycle.submit_recovery_release("req-release", started.started_at, "/tmp/secured-work")

    assert store.read(24) == seeded
    assert workspace.recovery_calls == []
    assert workspace.preserved_refs == {}
    assert [event for event in events if event[0].startswith("unpushed_revision_")] == []
