"""Operator comment-mode tests for handoff publication (issue #284).

``HANDOFF_COMMENT_MODE=link`` (the default) keeps the operator feed to a
single shared handoff comment per issue: the first handoff creates it and
later routine handoffs update its hidden record without posting an
individual result comment or copying the note body into GitHub.
``HANDOFF_COMMENT_MODE=verbose`` preserves the historical per-attempt
result comments for troubleshooting. Both modes share the same bookkeeping
(continuation counting, intent first-wins, recovery), post explicit
deduplicated notifications when human action is needed, and keep normal
completion reporting.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.completion import AttemptOutcome, CompletionDecision, PublicationPath
from simple_coding_agent.config import (
    ConfigurationError,
    ProfileError,
    load_repository_profile,
    load_runtime_config,
)
from simple_coding_agent.finalization import AttemptCompletionStore
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.handoff_record import (
    HandoffPublicationStore,
    MemoryHandoffPublicationStore,
    parse_handoff_record,
)
from simple_coding_agent.lifecycle import AgentLifecycle, _publication_confirmed
from simple_coding_agent.publication import PublicationRequest, Publisher
from simple_coding_agent.recovery import REQUEUE_SOURCE_AUTOMATIC
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace

from test_handoff_publication import (
    ATTEMPT_ONE,
    ATTEMPT_TWO,
    NOTE_TWO,
    HandoffGitHub,
    handoff_request,
    prepared_branch,
    shared_bodies,
)
from test_recovery import ControlStore


UTC = timezone.utc


def _operator_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = {
        "GITHUB_TOKEN": "github-secret",
        "META_API_KEY": "meta-secret",
        "TARGET_REPO": "octo/example",
        "DATA_DIR": str(tmp_path),
    }
    env.update(overrides)
    return env


def _issue(number: int = 24) -> TrackerIssue:
    return TrackerIssue(
        id=f"issue-{number}",
        number=number,
        title="Implement feature",
        body="Do the work.",
        created_at=datetime(2026, 9, 20, tzinfo=UTC),
        state="OPEN",
        labels=frozenset({"ready-for-agent"}),
        assignee_logins=("agent",),
        blocked_by=0,
        author_login="reporter",
    )


def _claim(number: int = 24) -> Claim:
    return Claim(_issue(number), Assignment(f"issue-{number}", "agent-id"))


def _lifecycle(tmp_path: Path, *, tracker: FakeTracker, max_auto_continuations: int) -> AgentLifecycle:
    return AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path / "state"),
        workspace=InMemoryWorkspace(tmp_path / "workspace", issue_number=24),
        profile_loader=lambda _: SimpleNamespace(base_branch="main"),
        publisher=FakePublisher(),
        attempt_runner=lambda c, p, pr: None,
        control_store=ControlStore(tmp_path),
        completion_store=AttemptCompletionStore(tmp_path),
        sleeper=lambda seconds: None,
        repository="octo/example",
        max_auto_continuations=max_auto_continuations,
    )


def _result_bodies(github: HandoffGitHub) -> list[str]:
    return [body for _, body in github.comments if "Agent Attempt Result:" in body]


# --- operator setting: default, normalization, validation -------------------


def test_handoff_comment_mode_defaults_to_link(tmp_path: Path) -> None:
    config = load_runtime_config(_operator_env(tmp_path))

    assert config.handoff_comment_mode == "link"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("link", "link"), ("verbose", "verbose"), (" Link ", "link"), ("VERBOSE", "verbose")],
)
def test_handoff_comment_mode_accepts_link_or_verbose(tmp_path: Path, raw: str, expected: str) -> None:
    config = load_runtime_config(_operator_env(tmp_path, HANDOFF_COMMENT_MODE=raw))

    assert config.handoff_comment_mode == expected


@pytest.mark.parametrize("raw", ["", "both", "email", "links", "quiet"])
def test_handoff_comment_mode_rejects_anything_else(tmp_path: Path, raw: str) -> None:
    with pytest.raises(ConfigurationError):
        load_runtime_config(_operator_env(tmp_path, HANDOFF_COMMENT_MODE=raw))


@pytest.mark.parametrize("raw", ["link", "verbose"])
def test_handoff_comment_mode_is_not_a_repository_profile_setting(
    tmp_path: Path, raw: str
) -> None:
    profile_path = tmp_path / "docs" / "agents" / "simple-coding-agent-profile.yml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(f"setup: pytest\ncheck: pytest\nhandoff_comment_mode: {raw}\n")

    with pytest.raises(ProfileError):
        load_repository_profile(tmp_path)


def test_publisher_constructor_rejects_an_unknown_mode(tmp_path: Path) -> None:
    workspace, _ = prepared_branch(tmp_path)
    github = HandoffGitHub()

    with pytest.raises(ValueError):
        Publisher(workspace, github, "octo/example", handoff_comment_mode="carrier-pigeon")


# --- link mode: one shared comment, routine handoffs quiet ------------------


def test_link_first_handoff_creates_only_the_shared_comment(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub()

    result = Publisher(workspace, github, "octo/example").publish(handoff_request(prepared.branch))

    assert result.outcome is AttemptOutcome.HANDOFF
    assert result.handoff_posted is True
    assert result.comment_posted is True
    assert result.comment_suppressed is True
    assert len(github.comments) == 1
    assert len(shared_bodies(github)) == 1
    assert _result_bodies(github) == []
    assert "Halfway done." not in github.comments[0][1]


def test_link_successive_handoffs_update_the_same_comment_quietly(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub()
    publisher = Publisher(workspace, github, "octo/example")

    first = publisher.publish(handoff_request(prepared.branch))
    second = publisher.publish(
        handoff_request(prepared.branch, started_at=ATTEMPT_TWO, note_sha=NOTE_TWO, count=1)
    )

    assert first.comment_suppressed is True
    assert second.comment_suppressed is True
    assert second.handoff_posted is True
    assert len(github.comments) == 1
    assert _result_bodies(github) == []
    record = parse_handoff_record(shared_bodies(github)[0])
    assert record is not None
    assert record.attempt_id == ATTEMPT_TWO
    assert record.continuation_count == 1


# --- verbose mode: historical per-attempt comments retained ------------------


def test_verbose_posts_a_result_comment_per_handoff(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub()
    publisher = Publisher(workspace, github, "octo/example", handoff_comment_mode="verbose")

    first = publisher.publish(handoff_request(prepared.branch))
    second = publisher.publish(
        handoff_request(prepared.branch, started_at=ATTEMPT_TWO, note_sha=NOTE_TWO, count=1)
    )

    assert first.comment_suppressed is False
    assert second.comment_suppressed is False
    assert first.handoff_posted is True
    assert second.handoff_posted is True
    assert len(shared_bodies(github)) == 1
    assert len(_result_bodies(github)) == 2
    assert "Halfway done." in _result_bodies(github)[0]
    record = parse_handoff_record(shared_bodies(github)[0])
    assert record is not None
    assert record.attempt_id == ATTEMPT_TWO


# --- mode switching: one record, one budget, no duplication -----------------


def test_switching_from_link_to_verbose_keeps_one_shared_comment(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub()
    store = MemoryHandoffPublicationStore()

    link_result = Publisher(workspace, github, "octo/example", handoff_store=store).publish(
        handoff_request(prepared.branch, count=1)
    )
    verbose_result = Publisher(
        workspace, github, "octo/example", handoff_store=store, handoff_comment_mode="verbose"
    ).publish(
        handoff_request(prepared.branch, started_at=ATTEMPT_TWO, note_sha=NOTE_TWO, count=2)
    )

    assert link_result.comment_suppressed is True
    assert verbose_result.comment_suppressed is False
    assert len(shared_bodies(github)) == 1
    assert len(_result_bodies(github)) == 1
    record = parse_handoff_record(shared_bodies(github)[0])
    assert record is not None
    assert record.continuation_count == 2
    assert record.attempt_id == ATTEMPT_TWO


def test_switching_from_verbose_to_link_keeps_one_shared_comment(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub()
    store = MemoryHandoffPublicationStore()

    Publisher(
        workspace, github, "octo/example", handoff_store=store, handoff_comment_mode="verbose"
    ).publish(handoff_request(prepared.branch, count=1))
    switched = Publisher(workspace, github, "octo/example", handoff_store=store).publish(
        handoff_request(prepared.branch, started_at=ATTEMPT_TWO, note_sha=NOTE_TWO, count=2)
    )

    assert switched.comment_suppressed is True
    assert len(shared_bodies(github)) == 1
    assert len(_result_bodies(github)) == 1
    record = parse_handoff_record(shared_bodies(github)[0])
    assert record is not None
    assert record.continuation_count == 2


def test_retry_after_a_mode_switch_reuses_the_first_intent(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub()
    data_dir = tmp_path / "data"

    first = Publisher(
        workspace, github, "octo/example", handoff_store=HandoffPublicationStore(data_dir)
    ).publish(handoff_request(prepared.branch, count=1))
    assert first.handoff_posted is True

    # After a restart under the other mode the caller recomputes a different
    # standing, but the first persisted intent still wins: no second comment,
    # no incremented count.
    restarted = Publisher(
        workspace,
        github,
        "octo/example",
        handoff_store=HandoffPublicationStore(data_dir),
        handoff_comment_mode="verbose",
    ).publish(handoff_request(prepared.branch, count=5))

    assert restarted.handoff_posted is True
    assert len(shared_bodies(github)) == 1
    assert len(_result_bodies(github)) == 1
    record = parse_handoff_record(shared_bodies(github)[0])
    assert record is not None
    assert record.continuation_count == 1


# --- notifications: explicit, deduplicated, with standing --------------------


def test_link_notified_handoff_posts_with_the_current_standing(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub()
    request = handoff_request(prepared.branch, count=2)
    notified = PublicationRequest(
        issue_number=request.issue_number,
        issue_title=request.issue_title,
        branch=request.branch,
        started_at=request.started_at,
        decision=request.decision,
        check_command=request.check_command,
        check_exit_code=request.check_exit_code,
        review_cycles=request.review_cycles,
        review_findings=request.review_findings,
        details="Halfway done.\n\nAutomatic continuation 2/2 used; the automatic"
        " continuation budget is exhausted and the issue now needs a human requeue.",
        base_branch=request.base_branch,
        note_commit_sha=request.note_commit_sha,
        requeue_source=REQUEUE_SOURCE_AUTOMATIC,
        handoff_continuation_count=2,
        handoff_notify=True,
    )

    result = Publisher(workspace, github, "octo/example").publish(notified)

    assert result.handoff_posted is True
    assert result.comment_suppressed is False
    assert result.comment_posted is True
    assert len(_result_bodies(github)) == 1
    assert "Automatic continuation 2/2 used" in _result_bodies(github)[0]


def test_link_notification_is_deduplicated_on_retry(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub()
    publisher = Publisher(workspace, github, "octo/example")
    base = handoff_request(prepared.branch, count=2)
    notified = PublicationRequest(
        issue_number=base.issue_number,
        issue_title=base.issue_title,
        branch=base.branch,
        started_at=base.started_at,
        decision=base.decision,
        check_command=base.check_command,
        check_exit_code=base.check_exit_code,
        review_cycles=base.review_cycles,
        review_findings=base.review_findings,
        details=base.details,
        base_branch=base.base_branch,
        note_commit_sha=base.note_commit_sha,
        requeue_source=base.requeue_source,
        handoff_continuation_count=base.handoff_continuation_count,
        handoff_notify=True,
    )

    assert publisher.publish(notified).comment_posted is True
    assert publisher.publish(notified).comment_posted is True

    assert len(_result_bodies(github)) == 1


def test_link_failed_shared_record_posts_a_notification_and_retains(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub(fail_create=True)

    result = Publisher(workspace, github, "octo/example").publish(handoff_request(prepared.branch))

    assert result.handoff_posted is False
    assert result.comment_suppressed is False
    assert result.comment_posted is True
    assert len(_result_bodies(github)) == 1
    assert shared_bodies(github) == []


def test_link_legacy_only_handoff_still_posts_its_result(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub()

    result = Publisher(workspace, github, "octo/example").publish(
        handoff_request(prepared.branch, count=None)
    )

    assert result.handoff_posted is True
    assert result.comment_suppressed is False
    assert result.comment_posted is True
    assert len(_result_bodies(github)) == 1
    assert shared_bodies(github) == []


def test_link_normal_completion_reporting_is_unchanged(tmp_path: Path) -> None:
    workspace, prepared = prepared_branch(tmp_path)
    github = HandoffGitHub()
    complete = PublicationRequest(
        issue_number=23,
        issue_title="Implement publication",
        branch=prepared.branch,
        started_at=ATTEMPT_ONE,
        decision=CompletionDecision(
            AttemptOutcome.COMPLETE, False, PublicationPath.NONE, ("checks green",)
        ),
        check_command="pytest",
        check_exit_code=0,
        review_cycles=1,
        review_findings="none",
        details="All done.",
        base_branch="main",
    )

    result = Publisher(workspace, github, "octo/example").publish(complete)

    assert result.outcome is AttemptOutcome.COMPLETE
    assert result.comment_posted is True
    assert result.comment_suppressed is False
    assert len(_result_bodies(github)) == 1


# --- lifecycle: exhaustion notifies, routine stays quiet --------------------


def test_routine_handoff_needs_no_notification(tmp_path: Path) -> None:
    tracker = FakeTracker(issues={24: _issue()}, auto_counts={24: 0})
    lifecycle = _lifecycle(tmp_path, tracker=tracker, max_auto_continuations=2)

    assert lifecycle._handoff_needs_notification(_claim(), None) is False  # noqa: SLF001


def test_exhausted_budget_needs_notification(tmp_path: Path) -> None:
    tracker = FakeTracker(issues={24: _issue()}, auto_counts={24: 1})
    lifecycle = _lifecycle(tmp_path, tracker=tracker, max_auto_continuations=2)
    checkpoint = SimpleNamespace(requeue_source=REQUEUE_SOURCE_AUTOMATIC)

    assert lifecycle._handoff_needs_notification(_claim(), checkpoint) is True  # noqa: SLF001


def test_non_final_automatic_handoff_stays_quiet(tmp_path: Path) -> None:
    tracker = FakeTracker(issues={24: _issue()}, auto_counts={24: 0})
    lifecycle = _lifecycle(tmp_path, tracker=tracker, max_auto_continuations=2)
    checkpoint = SimpleNamespace(requeue_source=REQUEUE_SOURCE_AUTOMATIC)

    assert lifecycle._handoff_needs_notification(_claim(), checkpoint) is False  # noqa: SLF001


def test_zero_budget_never_notifies(tmp_path: Path) -> None:
    tracker = FakeTracker(issues={24: _issue()}, auto_counts={24: 5})
    lifecycle = _lifecycle(tmp_path, tracker=tracker, max_auto_continuations=0)
    checkpoint = SimpleNamespace(requeue_source=REQUEUE_SOURCE_AUTOMATIC)

    assert lifecycle._handoff_needs_notification(_claim(), checkpoint) is False  # noqa: SLF001


def test_unreadable_count_notifies_rather_than_hiding_a_stopped_issue(tmp_path: Path) -> None:
    class UnreadableTracker(FakeTracker):
        def automatic_continuation_count(self, issue):  # noqa: ANN001, ANN202
            raise OSError("tracker unavailable")

    tracker = UnreadableTracker(issues={24: _issue()})
    lifecycle = _lifecycle(tmp_path, tracker=tracker, max_auto_continuations=2)
    checkpoint = SimpleNamespace(requeue_source=REQUEUE_SOURCE_AUTOMATIC)

    assert lifecycle._handoff_needs_notification(_claim(), checkpoint) is True  # noqa: SLF001


# --- confirmation: suppressed never blocks, failure never confirms ----------


def test_suppressed_routine_handoff_counts_as_confirmed() -> None:
    published = FakePublisher(comment_suppressed=True).publish(
        SimpleNamespace(
            decision=SimpleNamespace(outcome=AttemptOutcome.HANDOFF),
        )
    )

    assert published.comment_posted is True
    assert published.handoff_posted is True
    assert _publication_confirmed(published) is True


def test_unconfirmed_shared_record_never_counts_as_confirmed() -> None:
    published = FakePublisher(handoff_posted=False).publish(
        SimpleNamespace(
            decision=SimpleNamespace(outcome=AttemptOutcome.HANDOFF),
        )
    )

    assert _publication_confirmed(published) is False


def test_suppressed_flag_survives_the_publisher_double() -> None:
    assert FakePublisher()._comment_suppressed is False  # noqa: SLF001
    assert FakePublisher(comment_suppressed=True)._comment_suppressed is True  # noqa: SLF001
