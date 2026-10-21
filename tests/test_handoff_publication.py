"""Publication tests for the single discoverable handoff comment (issue #282).

The first successful handoff creates one short shared comment linking to the
Handoff note on the attempt branch (branch-following URL) plus hidden
versioned metadata; later handoffs update that same comment in place. The
intended payload is persisted durably before any remote write, the branch is
pushed and verified before metadata claiming its revision is published, and
the remote record is confirmed before the step reports success. Retries reuse
the persisted intent (never a recomputed count, never a second comment) and
never overwrite a newer attempt's record.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from simple_coding_agent.attempt_state import AttemptPhase, AttemptStateStore
from simple_coding_agent.completion import AttemptOutcome, CompletionDecision, PublicationPath
from simple_coding_agent.git_workspace import GitWorkspaceError
from simple_coding_agent.handoff_record import (
    HandoffPublicationStore,
    MemoryHandoffPublicationStore,
    parse_handoff_record,
)
from simple_coding_agent.publication import (
    HandoffRecordRef,
    PublicationRequest,
    Publisher,
    PullRequest,
)
from simple_coding_agent.recovery import (
    REQUEUE_SOURCE_AUTOMATIC,
    REQUEUE_SOURCE_HUMAN,
    is_handoff_record,
)

from test_git_workspace import git, repository_with_main, write_and_commit


ATTEMPT_ONE = "2026-09-20T13:00:00Z"
ATTEMPT_TWO = "2026-09-20T14:00:00Z"
NOTE_ONE = "b" * 40
NOTE_TWO = "c" * 40


@dataclass
class HandoffGitHub:
    """An in-memory shared-record seam with failure injection.

    ``update_after_write``/``create_after_write`` simulate a lost response:
    the write lands, then the call raises, so confirmation must re-read.
    ``fail_update``/``fail_create``/``fail_find`` raise without writing, so
    the publisher must retain the attempt instead of guessing.
    """

    update_after_write: bool = False
    create_after_write: bool = False
    fail_update: bool = False
    fail_create: bool = False
    fail_find: bool = False

    def __post_init__(self) -> None:
        self.comments: list[tuple[int, str]] = []
        self.shared: dict[int, tuple[str, str, int]] = {}
        self._next_id = 1
        self.find_calls = 0

    def find_pull_request(self, repository: str, head: str) -> PullRequest | None:
        return None

    def create_pull_request(
        self, repository: str, head: str, base: str, title: str, body: str
    ) -> PullRequest:
        raise AssertionError("handoff tests never create a pull request")

    def find_attempt_comment(self, repository: str, issue_number: int, marker: str) -> bool:
        return any(number == issue_number and marker in body for number, body in self.comments)

    def add_comment(self, repository: str, issue_number: int, body: str) -> None:
        self.comments.append((issue_number, body))

    def find_handoff_record(self, repository: str, issue_number: int) -> HandoffRecordRef | None:
        self.find_calls += 1
        if self.fail_find:
            raise OSError("GitHub is unavailable")
        entry = self.shared.get(issue_number)
        if entry is None:
            return None
        return HandoffRecordRef(comment_id=entry[0], body=entry[1])

    def create_handoff_record(self, repository: str, issue_number: int, body: str) -> str:
        if self.fail_create:
            raise OSError("create failed")
        comment_id = f"handoff-{self._next_id}"
        self._next_id += 1
        self.comments.append((issue_number, body))
        self.shared[issue_number] = (comment_id, body, len(self.comments) - 1)
        if self.create_after_write:
            raise OSError("connection closed after write")
        return comment_id

    def update_handoff_record(self, repository: str, comment_id: str, body: str) -> None:
        if self.fail_update:
            raise OSError("update failed")
        for issue_number, (known_id, _, index) in self.shared.items():
            if known_id == comment_id:
                self.shared[issue_number] = (comment_id, body, index)
                self.comments[index] = (issue_number, body)
                break
        else:
            raise AssertionError(f"unknown handoff comment {comment_id!r}")
        if self.update_after_write:
            raise OSError("connection closed after write")


def handoff_request(
    branch: str,
    *,
    started_at: str = ATTEMPT_ONE,
    note_sha: str = NOTE_ONE,
    requeue_source: str = REQUEUE_SOURCE_HUMAN,
    count: int | None = 0,
) -> PublicationRequest:
    return PublicationRequest(
        issue_number=23,
        issue_title="Implement publication",
        branch=branch,
        started_at=started_at,
        decision=CompletionDecision(AttemptOutcome.HANDOFF, False, PublicationPath.PARTIAL, ("handoff",)),
        check_command="pytest",
        check_exit_code=0,
        review_cycles=1,
        review_findings="not run",
        details="Halfway done.",
        base_branch="main",
        note_commit_sha=note_sha,
        requeue_source=requeue_source,
        handoff_continuation_count=count,
    )


def prepared_branch(tmp_path: Path, *, issue_number: int = 23):
    remote, _ = repository_with_main(tmp_path)
    from simple_coding_agent.git_workspace import GitWorkspace

    workspace = GitWorkspace(tmp_path / "clone", str(remote), token_provider=lambda: "token")
    prepared = workspace.prepare_attempt(base_branch="main", issue_number=issue_number)
    write_and_commit(tmp_path / "clone", "feature.txt", "done", "Implement publication")
    return workspace, prepared


def shared_bodies(github: HandoffGitHub) -> list[str]:
    return [body for _, body in github.comments if is_handoff_record(body)]


def test_first_handoff_creates_one_shared_comment_with_branch_following_link(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub()

    result = Publisher(workspace, github, "octo/example").publish(handoff_request(prepared.branch))

    assert result.outcome is AttemptOutcome.HANDOFF
    assert result.handoff_posted is True
    assert len(github.comments) == 2
    assert len(shared_bodies(github)) == 1
    shared = shared_bodies(github)[0]
    assert "https://github.com/octo/example/blob/agent/issue-23/.agent/handoff/23.md" in shared
    assert "<!-- agent-attempt:" not in shared
    assert "Agent Attempt Result" not in shared
    record = parse_handoff_record(shared)
    assert record is not None
    assert record.attempt_id == ATTEMPT_ONE
    assert record.requeue_source == REQUEUE_SOURCE_HUMAN
    assert record.continuation_count == 0
    assert record.branch == "agent/issue-23"
    assert record.note_sha == NOTE_ONE
    pushed_sha = git(tmp_path / "clone", "rev-parse", "agent/issue-23").strip()
    assert record.branch_sha == pushed_sha
    # The legacy per-attempt comment is unchanged: still posted, still marked.
    legacy = [body for _, body in github.comments if not is_handoff_record(body)]
    assert len(legacy) == 1
    assert f"<!-- agent-attempt: {ATTEMPT_ONE} -->" in legacy[0]


def test_second_handoff_updates_the_same_shared_comment(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub()
    publisher = Publisher(workspace, github, "octo/example")

    first = publisher.publish(handoff_request(prepared.branch))
    write_and_commit(tmp_path / "clone", "more.txt", "more", "Continue publication")
    second = publisher.publish(
        handoff_request(prepared.branch, started_at=ATTEMPT_TWO, note_sha=NOTE_TWO, count=1)
    )

    assert first.handoff_posted is True
    assert second.handoff_posted is True
    assert len(shared_bodies(github)) == 1
    record = parse_handoff_record(shared_bodies(github)[0])
    assert record is not None
    assert record.attempt_id == ATTEMPT_TWO
    assert record.note_sha == NOTE_TWO
    assert record.continuation_count == 1
    # Two legacy per-attempt comments plus the one shared record.
    assert len(github.comments) == 3


def test_failed_push_before_publication_claims_no_revision(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub()
    store = MemoryHandoffPublicationStore()
    state = AttemptStateStore(tmp_path / "state")
    state.start(issue_number=23, branch=prepared.branch)
    for phase in (AttemptPhase.SETUP, AttemptPhase.MODEL_RUNNING, AttemptPhase.PUSHING, AttemptPhase.PUBLISHING):
        state.transition(phase)

    result = Publisher(
        PushFailingWorkspace(), github, "octo/example", attempt_state=state, handoff_store=store
    ).publish(handoff_request(prepared.branch))

    assert result.handoff_posted is False
    assert shared_bodies(github) == []
    # The branch was never verified, so no intent may be persisted claiming it.
    assert store.load(ATTEMPT_ONE) is None
    # The legacy per-attempt comment still posts; only the shared-record step is held.
    assert result.comment_posted is True
    assert len(github.comments) == 1


def test_failed_update_retains_the_attempt_without_touching_the_record(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub(fail_update=True)
    publisher = Publisher(workspace, github, "octo/example")

    assert publisher.publish(handoff_request(prepared.branch)).handoff_posted is True
    original = shared_bodies(github)[0]
    write_and_commit(tmp_path / "clone", "more.txt", "more", "Continue publication")

    result = publisher.publish(
        handoff_request(prepared.branch, started_at=ATTEMPT_TWO, note_sha=NOTE_TWO, count=1)
    )

    assert result.handoff_posted is False
    assert shared_bodies(github) == [original]
    assert parse_handoff_record(original) is not None
    assert parse_handoff_record(original).attempt_id == ATTEMPT_ONE  # type: ignore[union-attr]


def test_successful_update_with_a_lost_response_confirms_by_rereading(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub()
    publisher = Publisher(workspace, github, "octo/example")

    assert publisher.publish(handoff_request(prepared.branch)).handoff_posted is True
    github.update_after_write = True
    write_and_commit(tmp_path / "clone", "more.txt", "more", "Continue publication")

    result = publisher.publish(
        handoff_request(prepared.branch, started_at=ATTEMPT_TWO, note_sha=NOTE_TWO, count=1)
    )

    assert result.handoff_posted is True
    assert len(shared_bodies(github)) == 1
    record = parse_handoff_record(shared_bodies(github)[0])
    assert record is not None
    assert record.attempt_id == ATTEMPT_TWO


def test_successful_create_with_a_lost_response_confirms_by_rereading(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub(create_after_write=True)

    result = Publisher(workspace, github, "octo/example").publish(handoff_request(prepared.branch))

    assert result.handoff_posted is True
    assert len(shared_bodies(github)) == 1
    assert parse_handoff_record(shared_bodies(github)[0]) is not None


def test_failed_create_without_a_write_retains_the_attempt(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub(fail_create=True)

    result = Publisher(workspace, github, "octo/example").publish(handoff_request(prepared.branch))

    assert result.handoff_posted is False
    assert shared_bodies(github) == []


def test_retry_after_restart_reuses_the_persisted_intent(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub()
    data_dir = tmp_path / "data"

    first = Publisher(
        workspace, github, "octo/example", handoff_store=HandoffPublicationStore(data_dir)
    ).publish(handoff_request(prepared.branch, count=1))
    assert first.handoff_posted is True

    # After a restart the caller recomputes a different standing, but the
    # first persisted intent wins: no second comment, no incremented count.
    restarted = Publisher(
        workspace, github, "octo/example", handoff_store=HandoffPublicationStore(data_dir)
    ).publish(handoff_request(prepared.branch, count=5))
    assert restarted.handoff_posted is True
    assert len(shared_bodies(github)) == 1
    record = parse_handoff_record(shared_bodies(github)[0])
    assert record is not None
    assert record.continuation_count == 1
    assert record.attempt_id == ATTEMPT_ONE


def test_timeout_retry_confirms_without_recomputing_the_standing(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub(fail_find=True)
    store = MemoryHandoffPublicationStore()
    publisher = Publisher(workspace, github, "octo/example", handoff_store=store)

    assert publisher.publish(handoff_request(prepared.branch, count=1)).handoff_posted is False
    # The intent was persisted before the remote write failed; the retry
    # reuses it instead of the recomputed standing, and confirms by re-read.
    assert store.load(ATTEMPT_ONE) is not None
    github.fail_find = False
    retried = publisher.publish(handoff_request(prepared.branch, count=9))

    assert retried.handoff_posted is True
    assert len(shared_bodies(github)) == 1
    record = parse_handoff_record(shared_bodies(github)[0])
    assert record is not None
    assert record.continuation_count == 1
    assert record.attempt_id == ATTEMPT_ONE


def test_newer_attempt_record_is_never_overwritten(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub()
    publisher = Publisher(workspace, github, "octo/example")

    assert publisher.publish(
        handoff_request(prepared.branch, started_at=ATTEMPT_TWO, note_sha=NOTE_TWO, count=2)
    ).handoff_posted is True
    newer = shared_bodies(github)[0]

    result = publisher.publish(handoff_request(prepared.branch, count=0))

    assert result.handoff_posted is False
    assert shared_bodies(github) == [newer]


def test_ambiguous_remote_state_retains_the_attempt(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub(fail_find=True)

    result = Publisher(workspace, github, "octo/example").publish(handoff_request(prepared.branch))

    assert result.handoff_posted is False
    # The legacy comment still posts; only the shared-record step is held.
    assert result.comment_posted is True
    assert shared_bodies(github) == []


def test_automatic_continuation_standing_rides_in_the_hidden_record(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub()

    result = Publisher(workspace, github, "octo/example").publish(
        handoff_request(
            prepared.branch, requeue_source=REQUEUE_SOURCE_AUTOMATIC, count=3
        )
    )

    assert result.handoff_posted is True
    record = parse_handoff_record(shared_bodies(github)[0])
    assert record is not None
    assert record.requeue_source == REQUEUE_SOURCE_AUTOMATIC
    assert record.continuation_count == 3


class PushFailingWorkspace:
    def push_attempt_branch(self, branch: str, *, max_retries: int, deadline: float | None = None) -> str:
        raise GitWorkspaceError("remote hung up unexpectedly")


# --- lifecycle standing and gating ------------------------------------------
#
# The shared record's standing comes from the claim snapshot: a human requeue
# stands at zero while an automatic continuation stands one past the current
# row. An unconfirmed shared record retains the attempt (no release, no
# cleanup, no finalization) even though the legacy comment posted.

from datetime import UTC, datetime  # noqa: E402

from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue  # noqa: E402
from simple_coding_agent.lifecycle import (  # noqa: E402
    AgentLifecycle,
    AttemptEvidence,
    LifecycleStatus,
)

from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace  # noqa: E402


def lifecycle_issue(number: int, labels: frozenset[str]) -> TrackerIssue:
    return TrackerIssue(
        id=f"issue-{number}",
        number=number,
        title="Implement publication",
        body="Connect boundaries.",
        created_at=datetime(2026, 9, 20, tzinfo=UTC),
        state="OPEN",
        labels=labels,
        assignee_logins=(),
        blocked_by=0,
        author_login="reporter",
    )


def handoff_workflow() -> AttemptEvidence:
    return AttemptEvidence(
        decision=CompletionDecision(
            AttemptOutcome.HANDOFF, False, PublicationPath.PARTIAL, ("handoff",)
        ),
        check_command="pytest",
        check_exit_code=0,
        review_cycles=1,
        review_findings="not run",
        details="Halfway done.",
        note_commit_sha="d" * 40,
    )


def run_handoff_once(
    tmp_path: Path,
    *,
    labels: frozenset[str],
    auto_counts: dict[int, int] | None = None,
    handoff_posted: bool = True,
):
    from tests.test_lifecycle import profile

    claim = Claim(lifecycle_issue(24, labels), Assignment("issue-24", "agent-id"))
    tracker = FakeTracker(next_claim=claim, auto_counts=auto_counts)
    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
    publisher = FakePublisher(handoff_posted=handoff_posted)
    state = AttemptStateStore(tmp_path / "state")
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=workspace,
        profile_loader=lambda _: profile(),
        publisher=publisher,
        attempt_runner=lambda received_claim, profile, prepared: handoff_workflow(),
    )
    return lifecycle.run_once(), tracker, workspace, publisher, state


def test_human_requeue_stands_at_zero(tmp_path: Path) -> None:
    _, _, _, publisher, _ = run_handoff_once(
        tmp_path, labels=frozenset({"ready-for-agent"}), auto_counts={24: 2}
    )

    assert publisher.requests[0].handoff_continuation_count == 0


def test_automatic_continuation_stands_one_past_the_row(tmp_path: Path) -> None:
    _, _, _, publisher, _ = run_handoff_once(
        tmp_path, labels=frozenset({"round-finished"}), auto_counts={24: 2}
    )

    assert publisher.requests[0].handoff_continuation_count == 3


def test_confirmed_shared_record_releases_and_finalizes(tmp_path: Path) -> None:
    result, tracker, workspace, _, state = run_handoff_once(
        tmp_path, labels=frozenset({"ready-for-agent"})
    )

    assert result.status is LifecycleStatus.ATTEMPTED
    assert result.outcome is AttemptOutcome.HANDOFF
    assert tracker.cleanup == [(24, "round-finished", "agent-id")]
    assert workspace.cleanup_calls == [("main", False)]
    assert state.read() is None


def test_unconfirmed_shared_record_retains_the_attempt(tmp_path: Path) -> None:
    result, tracker, workspace, _, state = run_handoff_once(
        tmp_path, labels=frozenset({"ready-for-agent"}), handoff_posted=False
    )

    assert result.status is LifecycleStatus.ATTEMPTED
    assert result.outcome is AttemptOutcome.HANDOFF
    assert tracker.cleanup == []
    # Cleanup runs but retains the branch; the checkpoint stays for recovery.
    assert workspace.cleanup_calls == [("main", True)]
    assert state.read() is not None
