"""Automatic continuation of ``round-finished`` issues (issue #202).

``MAX_AUTO_CONTINUATIONS`` (operator ``.env`` only, default ``0``) lets the
agent pick up ``round-finished`` issues on its own, up to N automatic
continuations in a row. A human requeue (re-applying ``ready-for-agent``)
resets the count and grants another N rounds.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.completion import (
    AttemptOutcome,
    CompletionDecision,
    PublicationPath,
)
from simple_coding_agent.config import (
    ConfigurationError,
    ProfileError,
    load_repository_profile,
    load_runtime_config,
)
from simple_coding_agent.control import (
    CommandAcknowledgement,
    ControlStore,
    NextIssueRejectedError,
)
from simple_coding_agent.github_tracker import (
    Assignment,
    GitHubIdentity,
    GitHubTracker,
    IssueComment,
    IssuePage,
    TrackerIssue,
    ineligibility_reason,
)
from simple_coding_agent.lifecycle import AgentLifecycle, AttemptEvidence
from simple_coding_agent.recovery import (
    REQUEUE_SOURCE_AUTOMATIC,
    REQUEUE_SOURCE_HUMAN,
    requeue_source_marker,
)
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace

READY = frozenset({"ready-for-agent"})
ROUND = frozenset({"round-finished"})
BOTH = frozenset({"ready-for-agent", "round-finished"})

_CREATED_BASE = datetime(2026, 9, 20, 13, 0, tzinfo=UTC)


def issue(number: int, **overrides: object) -> TrackerIssue:
    base: dict[str, object] = {
        "id": f"issue-{number}",
        "number": number,
        "title": f"Issue {number}",
        "body": f"Implement {number}.",
        "created_at": _CREATED_BASE + timedelta(seconds=number),
        "state": "OPEN",
        "labels": READY,
        "assignee_logins": (),
        "blocked_by": 0,
        "author_login": "reporter",
    }
    base.update(overrides)
    return TrackerIssue(**base)  # type: ignore[arg-type]


def result_body(source: str) -> str:
    """An own attempt-result comment body stamped with a requeue source."""

    return (
        "## Agent Attempt Result: handoff\n\nHalf done.\n\n"
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->\n"
        f"{requeue_source_marker(source)}"
    )


def comment(body: str, author: str = "agent") -> IssueComment:
    return IssueComment(author_login=author, body=body)


def complete_evidence() -> AttemptEvidence:
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
        note_commit_sha="note-rev",
    )


class ContinuationTransport:
    """Mutable transport double with per-issue comments for continuation tests.

    ``listed`` is served by ``list_issues`` as a single page; ``get_issue``
    serves ``refreshed`` overrides when present, otherwise the live state, so
    tests can change what the claim-time re-read observes. ``comments`` maps
    issue numbers to standing comment lists; ``comment_scripts`` maps issue
    numbers to successive comment lists, one per ``list_issue_comments``
    call, to simulate the count changing between the queue read and the
    claim. Label and assignee mutations apply to the live state, like the
    production transport's writes.
    """

    def __init__(
        self,
        issues: list[TrackerIssue],
        *,
        comments: dict[int, list[IssueComment]] | None = None,
        comment_scripts: dict[int, list[list[IssueComment]]] | None = None,
        refreshed: dict[int, TrackerIssue | None] | None = None,
    ) -> None:
        self.state = {item.number: item for item in issues}
        self.listed = [item.number for item in issues]
        self.comments = {number: list(bodies) for number, bodies in (comments or {}).items()}
        self.comment_scripts = {
            number: [list(script) for script in scripts]
            for number, scripts in (comment_scripts or {}).items()
        }
        self.refreshed = dict(refreshed or {})
        self.assignments: list[tuple[str, str]] = []
        self.removed_labels: list[tuple[str, str]] = []
        self.added_labels: list[tuple[str, str]] = []
        self.removed_assignees: list[tuple[str, str]] = []
        self.comment_reads: list[int] = []

    def list_issues(self, repository: str, cursor: str | None) -> IssuePage:
        assert repository == "octo/example"
        assert cursor is None
        return IssuePage(
            issues=tuple(self.state[number] for number in self.listed if number in self.state),
            next_cursor=None,
        )

    def get_issue(self, repository: str, number: int) -> TrackerIssue | None:
        assert repository == "octo/example"
        if number in self.refreshed:
            return self.refreshed[number]
        return self.state.get(number)

    def viewer(self) -> GitHubIdentity:
        return GitHubIdentity(id="viewer-id", login="agent")

    def _find(self, issue_id: str) -> TrackerIssue:
        for item in self.state.values():
            if item.id == issue_id:
                return item
        raise AssertionError(f"unknown issue id {issue_id!r}")

    def assign_issue(self, issue_id: str, assignee_id: str) -> Assignment:
        self.assignments.append((issue_id, assignee_id))
        found = self._find(issue_id)
        self.state[found.number] = replace(
            found, assignee_logins=found.assignee_logins + ("agent",)
        )
        return Assignment(issue_id=issue_id, assignee_id=assignee_id)

    def remove_assignee(self, issue_id: str, assignee_id: str) -> None:
        self.removed_assignees.append((issue_id, assignee_id))
        found = self._find(issue_id)
        self.state[found.number] = replace(
            found,
            assignee_logins=tuple(login for login in found.assignee_logins if login != "agent"),
        )

    def remove_label(self, issue_id: str, label: str) -> None:
        self.removed_labels.append((issue_id, label))
        found = self._find(issue_id)
        self.state[found.number] = replace(found, labels=found.labels - {label})

    def add_label(self, issue_id: str, label: str) -> None:
        self.added_labels.append((issue_id, label))
        found = self._find(issue_id)
        self.state[found.number] = replace(found, labels=found.labels | {label})

    def list_issue_comments(self, repository: str, issue_number: int) -> tuple[IssueComment, ...]:
        assert repository == "octo/example"
        self.comment_reads.append(issue_number)
        script = self.comment_scripts.get(issue_number)
        if script:
            if len(script) > 1:
                return tuple(script.pop(0))
            return tuple(script[0])
        return tuple(self.comments.get(issue_number, ()))


def make_auto_lifecycle(
    tmp_path: Path,
    issues: dict[int, TrackerIssue],
    *,
    fifo: list[int] | None = None,
    auto_counts: dict[int, int] | None = None,
    budget: int = 2,
    runner=None,
    with_control: bool = True,
):
    """A lifecycle wired for continuation tests, recording observability events."""

    tracker = FakeTracker(
        issues, fifo=fifo if fifo is not None else [], auto_counts=auto_counts
    )
    publisher = FakePublisher()
    events: list[tuple[str, str, str, int | None]] = []

    def event_log(
        event: str,
        detail: str = "",
        level: str = "INFO",
        issue_number: int | None = None,
    ) -> None:
        events.append((event, detail, level, issue_number))

    store = ControlStore(tmp_path) if with_control else None
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=publisher,
        attempt_runner=runner or (lambda claim, profile, prepared: complete_evidence()),
        control_store=store,
        sleeper=lambda seconds: None,
        event_log=event_log,
        max_auto_continuations=budget,
    )
    return lifecycle, tracker, publisher, events, store


def auto_events(events: list[tuple[str, str, str, int | None]]) -> list[tuple[str, str, str, int | None]]:
    return [event for event in events if event[0] == "auto_continuation_claimed"]


# ---------------------------------------------------------------------------
# operator setting: load, validate, refuse in the repository profile
# ---------------------------------------------------------------------------


def _operator_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    return {
        "GITHUB_TOKEN": "github-secret",
        "META_API_KEY": "meta-secret",
        "TARGET_REPO": "octo/example",
        "DATA_DIR": str(tmp_path),
        **overrides,
    }


def test_max_auto_continuations_defaults_to_zero(tmp_path: Path) -> None:
    config = load_runtime_config(_operator_env(tmp_path))

    assert config.max_auto_continuations == 0


@pytest.mark.parametrize(("value", "expected"), [("0", 0), ("1", 1), ("2", 2), (" 3 ", 3)])
def test_max_auto_continuations_parses_non_negative_integers(
    tmp_path: Path, value: str, expected: int
) -> None:
    config = load_runtime_config(_operator_env(tmp_path, MAX_AUTO_CONTINUATIONS=value))

    assert config.max_auto_continuations == expected


@pytest.mark.parametrize("value", ["-1", "-10", "many", "2.5", "", "  ", "2x"])
def test_max_auto_continuations_rejects_invalid_values(tmp_path: Path, value: str) -> None:
    with pytest.raises(ConfigurationError):
        load_runtime_config(_operator_env(tmp_path, MAX_AUTO_CONTINUATIONS=value))


def test_repository_profile_refuses_max_auto_continuations(tmp_path: Path) -> None:
    profile_path = tmp_path / "docs" / "agents" / "simple-coding-agent-profile.yml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("setup: pytest\ncheck: pytest\nmax_auto_continuations: 2\n")

    with pytest.raises(ProfileError):
        load_repository_profile(tmp_path)


# ---------------------------------------------------------------------------
# eligibility: signal, budget, and the visible over-budget reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (READY, False),
        (ROUND, True),
        (BOTH, False),
        (frozenset(), False),
        (frozenset({"round-finished", "agent-auto-merge"}), True),
    ],
)
def test_continuation_signal_is_round_finished_without_ready(
    labels: frozenset[str], expected: bool
) -> None:
    assert issue(7, labels=labels).is_continuation_signal is expected


def test_zero_budget_is_never_eligible_and_reads_no_comments() -> None:
    transport = ContinuationTransport([issue(7, labels=ROUND)])
    tracker = GitHubTracker(transport, "octo/example", max_auto_continuations=0)

    assert tracker.is_auto_continuation_eligible(issue(7, labels=ROUND)) is False
    assert tracker.runnable_queue() == ()
    assert transport.comment_reads == []


def test_within_budget_continuation_is_eligible() -> None:
    assert (
        ineligibility_reason(
            issue(7, labels=ROUND),
            7,
            max_auto_continuations=2,
            auto_continuation_count=1,
        )
        is None
    )


def test_over_budget_continuation_names_the_limit() -> None:
    assert ineligibility_reason(
        issue(7, labels=ROUND),
        7,
        max_auto_continuations=2,
        auto_continuation_count=2,
    ) == "issue #7: automatic continuation limit reached (2/2)"


def test_human_requeue_stays_ordinary_regardless_of_count() -> None:
    # A human requeue carries ``ready-for-agent`` again, so the ordinary rule
    # applies even when the automatic count has reached the budget.
    assert (
        ineligibility_reason(
            issue(7, labels=BOTH),
            7,
            max_auto_continuations=2,
            auto_continuation_count=2,
        )
        is None
    )


def test_tracker_counts_automatic_comments_since_the_last_human_requeue() -> None:
    transport = ContinuationTransport(
        [issue(7, labels=ROUND)],
        comments={
            7: [
                comment(result_body(REQUEUE_SOURCE_HUMAN)),
                comment(result_body(REQUEUE_SOURCE_AUTOMATIC)),
                comment(result_body(REQUEUE_SOURCE_AUTOMATIC)),
            ]
        },
    )
    tracker = GitHubTracker(transport, "octo/example", max_auto_continuations=3)

    assert tracker.automatic_continuation_count(issue(7, labels=ROUND)) == 2


# ---------------------------------------------------------------------------
# runnable queue: continuations first, FIFO within each group
# ---------------------------------------------------------------------------


def test_runnable_queue_orders_continuations_before_ready_fifo() -> None:
    # #1 ready and oldest, #3 continuation older than #2, #4 ready newest.
    ready_old = issue(1)
    continuation_newer = issue(2, labels=ROUND)
    continuation_older = issue(
        3, labels=ROUND, created_at=_CREATED_BASE - timedelta(days=1)
    )
    ready_new = issue(4)
    transport = ContinuationTransport(
        [ready_old, continuation_newer, continuation_older, ready_new]
    )
    tracker = GitHubTracker(transport, "octo/example", max_auto_continuations=2)

    assert [item.number for item in tracker.runnable_queue()] == [3, 2, 1, 4]


def test_runnable_queue_keeps_human_requeues_in_the_ordinary_group() -> None:
    # #5 is a human requeue after a handoff (both labels) created before the
    # continuation #6: it must not jump the queue as a continuation.
    human_requeue = issue(5, labels=BOTH)
    continuation = issue(6, labels=ROUND)
    transport = ContinuationTransport([human_requeue, continuation])
    tracker = GitHubTracker(transport, "octo/example", max_auto_continuations=2)

    assert [item.number for item in tracker.runnable_queue()] == [6, 5]


def test_over_budget_continuation_stays_queued_out_with_its_label() -> None:
    exhausted = issue(7, labels=ROUND)
    ordinary = issue(8)
    transport = ContinuationTransport(
        [exhausted, ordinary],
        comments={7: [comment(result_body(REQUEUE_SOURCE_AUTOMATIC))] * 2},
    )
    tracker = GitHubTracker(transport, "octo/example", max_auto_continuations=2)

    assert [item.number for item in tracker.runnable_queue()] == [8]
    # The queue read changes nothing: the issue keeps waiting labelled.
    assert transport.state[7].labels == ROUND
    assert transport.assignments == []


def test_zero_budget_queue_behaves_as_before() -> None:
    transport = ContinuationTransport([issue(7, labels=ROUND), issue(8)])
    tracker = GitHubTracker(transport, "octo/example", max_auto_continuations=0)

    assert [item.number for item in tracker.runnable_queue()] == [8]
    assert transport.comment_reads == []


# ---------------------------------------------------------------------------
# claim_next: picks in-budget continuations, skips the rest at claim time
# ---------------------------------------------------------------------------


def test_claim_next_picks_an_in_budget_continuation() -> None:
    transport = ContinuationTransport([issue(7, labels=ROUND)])
    tracker = GitHubTracker(transport, "octo/example", max_auto_continuations=2)

    claim = tracker.claim_next()

    assert claim is not None
    assert claim.issue.number == 7
    assert claim.assignment == Assignment(issue_id="issue-7", assignee_id="viewer-id")
    # Claiming removes ``round-finished`` but the claim keeps the snapshot.
    assert transport.removed_labels == [("issue-7", "round-finished")]
    assert claim.issue.labels == ROUND


def test_claim_next_skips_over_budget_and_claims_the_next_issue() -> None:
    transport = ContinuationTransport(
        [issue(7, labels=ROUND), issue(8)],
        comments={7: [comment(result_body(REQUEUE_SOURCE_AUTOMATIC))] * 2},
    )
    tracker = GitHubTracker(transport, "octo/example", max_auto_continuations=2)

    claim = tracker.claim_next()

    assert claim is not None
    assert claim.issue.number == 8
    assert transport.assignments == [("issue-8", "viewer-id")]
    assert ("issue-7", "round-finished") not in transport.removed_labels


def test_claim_next_skips_a_human_relabel_between_queue_read_and_claim() -> None:
    listed = issue(7, labels=ROUND)
    transport = ContinuationTransport(
        [listed], refreshed={7: replace(listed, labels=frozenset())}
    )
    tracker = GitHubTracker(transport, "octo/example", max_auto_continuations=2)

    assert tracker.claim_next() is None
    assert transport.assignments == []


def test_claim_next_skips_when_the_budget_runs_out_between_reads() -> None:
    transport = ContinuationTransport(
        [issue(7, labels=ROUND)],
        comment_scripts={
            7: [
                [comment(result_body(REQUEUE_SOURCE_AUTOMATIC))],
                [
                    comment(result_body(REQUEUE_SOURCE_AUTOMATIC)),
                    comment(result_body(REQUEUE_SOURCE_AUTOMATIC)),
                ],
            ]
        },
    )
    tracker = GitHubTracker(transport, "octo/example", max_auto_continuations=2)

    assert tracker.claim_next() is None
    assert transport.assignments == []
    # One comment read for the queue scan, one for the claim revalidation.
    assert transport.comment_reads == [7, 7]


# ---------------------------------------------------------------------------
# next issue: accepts within budget, rejects past it with the limit named
# ---------------------------------------------------------------------------


def test_next_issue_accepts_a_continuation_within_budget(tmp_path: Path) -> None:
    lifecycle, _, _, _, store = make_auto_lifecycle(
        tmp_path, {7: issue(7, labels=ROUND)}, auto_counts={7: 1}, budget=2
    )

    record = lifecycle.submit_next_issue("req-next", 7)

    assert record.acknowledgement is CommandAcknowledgement.ACCEPTED
    assert store.next_issue_snapshot() == {"request_id": "req-next", "issue_number": 7}


def test_next_issue_rejects_a_continuation_past_budget_naming_the_limit(
    tmp_path: Path,
) -> None:
    lifecycle, _, _, _, store = make_auto_lifecycle(
        tmp_path, {7: issue(7, labels=ROUND)}, auto_counts={7: 2}, budget=2
    )

    with pytest.raises(
        NextIssueRejectedError, match=r"automatic continuation limit reached \(2/2\)"
    ):
        lifecycle.submit_next_issue("req-next", 7)

    assert store.next_issue_snapshot() is None
    assert store.get_command("req-next") is None


def test_next_issue_rejects_a_continuation_at_zero_budget(tmp_path: Path) -> None:
    lifecycle, _, _, _, store = make_auto_lifecycle(
        tmp_path, {7: issue(7, labels=ROUND)}, auto_counts={7: 0}, budget=0
    )

    with pytest.raises(NextIssueRejectedError, match="not labelled `ready-for-agent`"):
        lifecycle.submit_next_issue("req-next", 7)

    assert store.next_issue_snapshot() is None


def test_priority_claim_revalidates_the_budget_before_assigning(
    tmp_path: Path,
) -> None:
    lifecycle, tracker, _, _, store = make_auto_lifecycle(
        tmp_path, {7: issue(7, labels=ROUND)}, auto_counts={7: 1}, budget=2
    )
    lifecycle.submit_next_issue("req-next", 7)
    # The budget runs out after acceptance, before the claim boundary.
    tracker.auto_counts[7] = 2

    assert lifecycle.run_once().status.value == "idle"
    assert tracker.claimed_verified == []

    failed = store.get_command("req-next")
    assert failed is not None
    assert failed.acknowledgement is CommandAcknowledgement.NOT_FULFILLED
    assert "automatic continuation limit reached (2/2)" in failed.detail


def test_priority_claim_revalidates_a_human_relabel_before_assigning(
    tmp_path: Path,
) -> None:
    lifecycle, tracker, _, _, store = make_auto_lifecycle(
        tmp_path, {7: issue(7, labels=ROUND)}, auto_counts={7: 0}, budget=2
    )
    lifecycle.submit_next_issue("req-next", 7)
    # A human removes ``round-finished`` after acceptance: no signal remains.
    tracker.issues[7] = issue(7, labels=frozenset())

    assert lifecycle.run_once().status.value == "idle"
    assert tracker.claimed_verified == []

    failed = store.get_command("req-next")
    assert failed is not None
    assert failed.acknowledgement is CommandAcknowledgement.NOT_FULFILLED


def test_priority_automatic_claim_emits_the_event(tmp_path: Path) -> None:
    lifecycle, tracker, _, events, store = make_auto_lifecycle(
        tmp_path, {7: issue(7, labels=ROUND)}, auto_counts={7: 1}, budget=2
    )
    lifecycle.submit_next_issue("req-next", 7)

    assert lifecycle.run_once().status.value == "attempted"

    assert tracker.claimed_verified == [7]
    assert store.get_command("req-next").acknowledgement is CommandAcknowledgement.COMPLETED  # type: ignore[union-attr]
    claimed = auto_events(events)
    assert len(claimed) == 1
    assert claimed[0][3] == 7
    assert "1/2" in claimed[0][1]


# ---------------------------------------------------------------------------
# FIFO claims: automatic continuations emit the observability event
# ---------------------------------------------------------------------------


def test_fifo_automatic_claim_emits_the_event(tmp_path: Path) -> None:
    lifecycle, tracker, _, events, _ = make_auto_lifecycle(
        tmp_path, {7: issue(7, labels=ROUND)}, fifo=[7], auto_counts={7: 0}, budget=2
    )

    assert lifecycle.run_once().status.value == "attempted"

    claimed = auto_events(events)
    assert len(claimed) == 1
    assert claimed[0][3] == 7
    assert "0/2" in claimed[0][1]


def test_fifo_ordinary_claim_emits_no_event(tmp_path: Path) -> None:
    lifecycle, _, _, events, _ = make_auto_lifecycle(
        tmp_path, {7: issue(7)}, fifo=[7], budget=2
    )

    assert lifecycle.run_once().status.value == "attempted"

    assert auto_events(events) == []


# ---------------------------------------------------------------------------
# handoff result comments: standing lines only with a positive budget
# ---------------------------------------------------------------------------


def test_handoff_reports_zero_used_after_a_human_round(tmp_path: Path) -> None:
    lifecycle, _, publisher, _, _ = make_auto_lifecycle(
        tmp_path,
        {24: issue(24)},
        fifo=[24],
        budget=2,
        runner=lambda claim, profile, prepared: handoff_evidence(),
    )

    assert lifecycle.run_once().status.value == "attempted"

    assert publisher.requests[0].details == (
        "handoff note\n\nAutomatic continuation 0/2 used; the agent will continue automatically."
    )


def test_handoff_reports_progress_after_an_automatic_round(tmp_path: Path) -> None:
    lifecycle, _, publisher, _, _ = make_auto_lifecycle(
        tmp_path,
        {24: issue(24, labels=ROUND)},
        fifo=[24],
        auto_counts={24: 0},
        budget=2,
        runner=lambda claim, profile, prepared: handoff_evidence(),
    )

    assert lifecycle.run_once().status.value == "attempted"

    assert publisher.requests[0].details == (
        "handoff note\n\nAutomatic continuation 1/2 used; the agent will continue automatically."
    )


def test_handoff_using_up_the_budget_asks_for_a_human_requeue(tmp_path: Path) -> None:
    lifecycle, _, publisher, _, _ = make_auto_lifecycle(
        tmp_path,
        {24: issue(24, labels=ROUND)},
        fifo=[24],
        auto_counts={24: 1},
        budget=2,
        runner=lambda claim, profile, prepared: handoff_evidence(),
    )

    assert lifecycle.run_once().status.value == "attempted"

    assert publisher.requests[0].details == (
        "handoff note\n\nAutomatic continuation 2/2 used; the automatic continuation"
        " budget is exhausted and the issue now needs a human requeue (re-apply"
        " `ready-for-agent`) before the agent can continue it."
    )


def test_handoff_after_a_human_requeue_reports_a_reset_count(tmp_path: Path) -> None:
    # The human requeue (both labels at claim time) stamps this handoff
    # ``human``, so the count resets even though two automatic rounds ran
    # before: the comment must not claim the budget is exhausted.
    lifecycle, _, publisher, _, _ = make_auto_lifecycle(
        tmp_path,
        {24: issue(24, labels=BOTH)},
        fifo=[24],
        auto_counts={24: 2},
        budget=2,
        runner=lambda claim, profile, prepared: handoff_evidence(),
    )

    assert lifecycle.run_once().status.value == "attempted"

    assert publisher.requests[0].details == (
        "handoff note\n\nAutomatic continuation 0/2 used; the agent will continue automatically."
    )


def test_handoff_is_silent_at_zero_budget(tmp_path: Path) -> None:
    lifecycle, _, publisher, _, _ = make_auto_lifecycle(
        tmp_path,
        {24: issue(24, labels=ROUND)},
        fifo=[24],
        budget=0,
        runner=lambda claim, profile, prepared: handoff_evidence(),
    )

    assert lifecycle.run_once().status.value == "attempted"

    assert publisher.requests[0].details == "handoff note"


def test_complete_outcome_gets_no_standing_line(tmp_path: Path) -> None:
    lifecycle, _, publisher, _, _ = make_auto_lifecycle(
        tmp_path, {24: issue(24)}, fifo=[24], budget=2
    )

    assert lifecycle.run_once().status.value == "attempted"

    assert publisher.requests[0].details == "Implemented."


# ---------------------------------------------------------------------------
# end to end with N=2 through the real tracker
# ---------------------------------------------------------------------------


def test_n2_hands_off_twice_then_waits_and_resumes_after_human_requeue(
    tmp_path: Path,
) -> None:
    """Two automatic continuations, then waiting, then two more after a requeue."""

    transport = ContinuationTransport([issue(24)])
    tracker = GitHubTracker(transport, "octo/example", max_auto_continuations=2)
    publisher = FakePublisher()
    events: list[tuple[str, str, str, int | None]] = []

    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=publisher,
        attempt_runner=lambda claim, profile, prepared: handoff_evidence(),
        sleeper=lambda seconds: None,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level, issue_number)
        ),
        max_auto_continuations=2,
    )

    progress = (
        "Automatic continuation 0/2 used; the agent will continue automatically.",
        "Automatic continuation 1/2 used; the agent will continue automatically.",
    )
    exhausted = (
        "Automatic continuation 2/2 used; the automatic continuation budget is"
        " exhausted and the issue now needs a human requeue (re-apply"
        " `ready-for-agent`) before the agent can continue it."
    )

    def run_handoff_round(expected_line: str, expected_source: str) -> None:
        from simple_coding_agent.lifecycle import LifecycleStatus

        result = lifecycle.run_once()
        assert result.status is LifecycleStatus.ATTEMPTED
        assert result.outcome is AttemptOutcome.HANDOFF
        assert publisher.requests[-1].details == f"handoff note\n\n{expected_line}"
        assert publisher.requests[-1].requeue_source == expected_source
        # Simulate the posted result comment; the release already moved the
        # labels to ``round-finished``-only through the transport.
        transport.comments.setdefault(24, []).append(
            comment(result_body(expected_source))
        )
        assert transport.state[24].labels == ROUND

    # Round 1: ordinary claim, then the first automatic continuation.
    run_handoff_round(progress[0], REQUEUE_SOURCE_HUMAN)
    assert auto_events(events) == []
    run_handoff_round(progress[1], REQUEUE_SOURCE_AUTOMATIC)
    # Round 3 uses up the budget; round 4 finds nothing to claim.
    run_handoff_round(exhausted, REQUEUE_SOURCE_AUTOMATIC)
    assert [event[1] for event in auto_events(events)] != []
    assert [event[3] for event in auto_events(events)] == [24, 24]

    idle_assignments = list(transport.assignments)
    from simple_coding_agent.lifecycle import LifecycleStatus

    assert lifecycle.run_once().status is LifecycleStatus.IDLE
    assert len(publisher.requests) == 3
    assert transport.assignments == idle_assignments
    assert transport.state[24].labels == ROUND
    assert tracker.claim_next() is None

    # A human requeue resets the count and grants two more automatic rounds.
    transport.state[24] = replace(transport.state[24], labels=BOTH)
    run_handoff_round(progress[0], REQUEUE_SOURCE_HUMAN)
    run_handoff_round(progress[1], REQUEUE_SOURCE_AUTOMATIC)
    run_handoff_round(exhausted, REQUEUE_SOURCE_AUTOMATIC)
    assert lifecycle.run_once().status is LifecycleStatus.IDLE
    assert len(publisher.requests) == 6
    assert transport.state[24].labels == ROUND
    assert [event[3] for event in auto_events(events)] == [24, 24, 24, 24]
