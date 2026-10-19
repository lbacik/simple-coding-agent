"""Conflict requeue from a manually applied ``merge-conflict`` label (issue #269).

The tracer bullet of Conflict requeue: an issue returns to the queue because
the pull request a worker already published for it reports merge conflicts
with the base branch. The conflict attempt that follows only brings that
pull request up to date (clean rebase, setup plus one profile check, push
with ``--force-with-lease``) and does no new work: a real rebase conflict,
or a red check after a clean rebase, runs at most one narrow model
resolution session under the ordinary model limits.

``MAX_CONFLICT_ATTEMPTS`` (operator ``.env`` only, default ``0``) gates the
path: with ``0`` a ``merge-conflict`` issue is never claimed. Conflict
requeues are claimed before automatic continuations, which are claimed before
ordinary issues. Every conflict attempt removes ``merge-conflict`` and the
assignee, never adds ``ready-for-agent``/``round-finished``, and posts a
result comment with the machine-readable conflict-attempt marker, which stays
transparent to the automatic-continuation count.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.command_runner import CommandResult
from simple_coding_agent.completion import AttemptOutcome
from simple_coding_agent.config import (
    ConfigurationError,
    ProfileError,
    load_repository_profile,
    load_runtime_config,
)
from simple_coding_agent.config import RepositoryProfile
from simple_coding_agent.control import CommandAcknowledgement, NextIssueRejectedError
from simple_coding_agent.github_tracker import (
    Assignment,
    GitHubIdentity,
    GitHubTracker,
    IssueComment,
    IssuePage,
    TrackerIssue,
    ineligibility_reason,
)
from simple_coding_agent.conflict_attempt import is_conflict_claim
from simple_coding_agent.lifecycle import AgentLifecycle
from simple_coding_agent.publication import ConflictPublicationRequest, PullRequest
from simple_coding_agent.publication import _conflict_result_comment
from simple_coding_agent.recovery import (
    CONFLICT_OUTCOME_CHECK_FAILED,
    CONFLICT_OUTCOME_ERROR,
    CONFLICT_OUTCOME_REBASE_CONFLICT,
    CONFLICT_OUTCOME_SETUP_FAILED,
    CONFLICT_OUTCOME_SKIPPED,
    CONFLICT_OUTCOME_SUCCESS,
    REQUEUE_SOURCE_AUTOMATIC,
    REQUEUE_SOURCE_HUMAN,
    conflict_attempt_marker,
    count_automatic_continuations,
    is_conflict_attempt,
    parse_conflict_attempt,
    requeue_source_marker,
)
from tests.fakes import (
    FakeCommandRunner,
    FakeConflictExecutor,
    FakePublisher,
    FakeTracker,
    InMemoryWorkspace,
    failed_result,
    ok_result,
)

READY = frozenset({"ready-for-agent"})
ROUND = frozenset({"round-finished"})
CONFLICT = frozenset({"merge-conflict"})
HUMAN = frozenset({"ready-for-human"})

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


def pull_request(number: int = 7, **overrides: object) -> PullRequest:
    base: dict[str, object] = {
        "number": number,
        "url": f"https://example.test/pull/{number}",
        "node_id": f"pr-{number}",
        "draft": False,
        "mergeable": "CONFLICTING",
        "base_ref": "main",
    }
    base.update(overrides)
    return PullRequest(**base)  # type: ignore[arg-type]


def profile() -> RepositoryProfile:
    return RepositoryProfile(
        setup=("setup.sh",),
        check=("pytest",),
        base_branch="main",
        timeout=300,
        setup_timeout=120,
        env={},
    )


def raising_runner(claim: object, profile: object, prepared: object) -> object:
    """The ordinary model seam: conflict attempts must never reach it.

    Conflict resolution runs through ``conflict_model_executor`` instead, so
    the ordinary attempt runner stays untouched by conflict attempts.
    """

    raise AssertionError("a conflict attempt must never call the ordinary model path")


def make_conflict_lifecycle(
    tmp_path: Path,
    issues: dict[int, TrackerIssue],
    *,
    fifo: list[int] | None = None,
    pull_requests: dict[str, object] | None = None,
    command_results: list[CommandResult] | None = None,
    budget: int = 1,
    command_runner: FakeCommandRunner | None = None,
    runner: object | None = None,
    workspace: InMemoryWorkspace | None = None,
    conflict_executor: FakeConflictExecutor | None = None,
):
    """A lifecycle wired for conflict tests, recording observability events."""

    tracker = FakeTracker(issues, fifo=fifo if fifo is not None else [])
    active_workspace = workspace or InMemoryWorkspace(tmp_path / "repo")
    publisher = FakePublisher(pull_requests=pull_requests or {})
    active_runner = command_runner or FakeCommandRunner(list(command_results or []))
    events: list[tuple[str, str, str, int | None]] = []

    def event_log(
        event: str,
        detail: str = "",
        level: str = "INFO",
        issue_number: int | None = None,
    ) -> None:
        events.append((event, detail, level, issue_number))

    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=active_workspace,
        profile_loader=lambda _: profile(),
        publisher=publisher,
        attempt_runner=runner or raising_runner,  # type: ignore[arg-type]
        control_store=None,
        sleeper=lambda seconds: None,
        event_log=event_log,
        max_conflict_attempts=budget,
        command_runner=active_runner,  # type: ignore[arg-type]
        conflict_model_executor=conflict_executor,
    )
    return lifecycle, tracker, publisher, active_workspace, active_runner, events


def run_conflict(tmp_path: Path, number: int = 7, **kwargs: object) -> tuple[object, ...]:
    """Claim one conflict issue through ``run_once`` and return the harness."""

    labels = kwargs.pop("labels", CONFLICT)
    issues = {number: issue(number, labels=labels)}
    harness = make_conflict_lifecycle(tmp_path, issues, fifo=[number], **kwargs)  # type: ignore[arg-type]
    lifecycle = harness[0]
    result = lifecycle.run_once()
    return (result, *harness[1:])


def event_names(events: list[tuple[str, str, str, int | None]]) -> list[str]:
    return [event[0] for event in events]


def result_body(source: str) -> str:
    """An own attempt-result comment body stamped with a requeue source."""

    return (
        "## Agent Attempt Result: handoff\n\nHalf done.\n\n"
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->\n"
        f"{requeue_source_marker(source)}"
    )


def conflict_body(outcome: str = "success", sha: str = "abc123") -> str:
    """A conflict-attempt result comment body carrying both markers."""

    return (
        f"## Conflict Attempt Result: {outcome}\n\nRebased.\n\n"
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->\n"
        f"{conflict_attempt_marker(outcome, sha=sha)}"
    )


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


def test_max_conflict_attempts_defaults_to_zero(tmp_path: Path) -> None:
    config = load_runtime_config(_operator_env(tmp_path))

    assert config.max_conflict_attempts == 0


@pytest.mark.parametrize(("value", "expected"), [("0", 0), ("1", 1), ("2", 2), (" 3 ", 3)])
def test_max_conflict_attempts_parses_non_negative_integers(
    tmp_path: Path, value: str, expected: int
) -> None:
    config = load_runtime_config(_operator_env(tmp_path, MAX_CONFLICT_ATTEMPTS=value))

    assert config.max_conflict_attempts == expected


@pytest.mark.parametrize("value", ["-1", "-10", "many", "2.5", "", "  ", "2x"])
def test_max_conflict_attempts_rejects_invalid_values(tmp_path: Path, value: str) -> None:
    with pytest.raises(ConfigurationError):
        load_runtime_config(_operator_env(tmp_path, MAX_CONFLICT_ATTEMPTS=value))


def test_repository_profile_refuses_max_conflict_attempts(tmp_path: Path) -> None:
    profile_path = tmp_path / "docs" / "agents" / "simple-coding-agent-profile.yml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("setup: pytest\ncheck: pytest\nmax_conflict_attempts: 2\n")

    with pytest.raises(ProfileError):
        load_repository_profile(tmp_path)


# ---------------------------------------------------------------------------
# conflict-attempt marker: round-trip, validation, transparency
# ---------------------------------------------------------------------------


def test_conflict_marker_round_trip_with_reserved_extension_room() -> None:
    marker = conflict_attempt_marker(
        "rebase_conflict",
        sha=None,
        reason=None,
        files=("a.txt", "b.txt"),
        fix_commits=("deadbee",),
    )

    parsed = parse_conflict_attempt(f"header\n{marker}\n")
    assert parsed is not None
    assert parsed["outcome"] == "rebase_conflict"
    assert parsed["sha"] == "none"
    assert parsed["files"] == "a.txt,b.txt"
    assert parsed["fix_commits"] == "deadbee"
    assert is_conflict_attempt(marker) is True


def test_conflict_marker_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        conflict_attempt_marker("bogus", sha="abc")
    with pytest.raises(ValueError):
        conflict_attempt_marker("success", sha="has space")
    assert parse_conflict_attempt("no marker here") is None
    assert is_conflict_attempt(result_body(REQUEUE_SOURCE_AUTOMATIC)) is False


def test_conflict_comments_are_transparent_to_the_continuation_count() -> None:
    newest_first = [
        result_body(REQUEUE_SOURCE_AUTOMATIC),
        conflict_body("success"),
        result_body(REQUEUE_SOURCE_AUTOMATIC),
        conflict_body("skipped"),
        result_body(REQUEUE_SOURCE_AUTOMATIC),
        result_body(REQUEUE_SOURCE_HUMAN),
        result_body(REQUEUE_SOURCE_AUTOMATIC),
    ]

    assert count_automatic_continuations(newest_first) == 3


def test_conflict_comments_neither_count_nor_reset() -> None:
    assert count_automatic_continuations([conflict_body("success")]) == 0
    assert (
        count_automatic_continuations(
            [conflict_body("success"), result_body(REQUEUE_SOURCE_HUMAN)]
        )
        == 0
    )
    # A legacy unmarked own comment still resets, even past conflict rows.
    assert (
        count_automatic_continuations(
            [
                result_body(REQUEUE_SOURCE_AUTOMATIC),
                conflict_body("success"),
                "## Agent Attempt Result: complete\n\nDone.",
            ]
        )
        == 1
    )


# ---------------------------------------------------------------------------
# tracker: queue groups, gating, precedence, claim-time revalidation
# ---------------------------------------------------------------------------


class ConflictTransport:
    """Mutable transport double serving queue reads and comment standing."""

    def __init__(
        self,
        issues: list[TrackerIssue],
        *,
        comments: dict[int, list[IssueComment]] | None = None,
        refreshed: dict[int, TrackerIssue | None] | None = None,
    ) -> None:
        self.state = {item.number: item for item in issues}
        self.listed = [item.number for item in issues]
        self.comments = {number: list(bodies) for number, bodies in (comments or {}).items()}
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
        return tuple(self.comments.get(issue_number, ()))


def test_zero_budget_never_claims_merge_conflict() -> None:
    transport = ConflictTransport([issue(7, labels=CONFLICT)])
    tracker = GitHubTracker(transport, "octo/example", max_conflict_attempts=0)

    assert tracker.is_conflict_eligible(issue(7, labels=CONFLICT)) is False
    assert tracker.runnable_queue() == ()
    assert tracker.claim_next() is None


def test_conflict_group_claimed_before_continuations_and_ordinary() -> None:
    transport = ConflictTransport(
        [
            issue(1, labels=READY),
            issue(2, labels=ROUND),
            issue(3, labels=CONFLICT),
        ]
    )
    tracker = GitHubTracker(
        transport, "octo/example", max_auto_continuations=2, max_conflict_attempts=1
    )

    assert [item.number for item in tracker.runnable_queue()] == [3, 2, 1]


def test_conflict_fifo_within_group() -> None:
    transport = ConflictTransport(
        [issue(9, labels=CONFLICT), issue(4, labels=CONFLICT), issue(1, labels=READY)]
    )
    tracker = GitHubTracker(transport, "octo/example", max_conflict_attempts=1)

    assert [item.number for item in tracker.runnable_queue()] == [4, 9, 1]


def test_ready_for_human_excludes_conflict() -> None:
    labelled = issue(7, labels=CONFLICT | HUMAN)
    transport = ConflictTransport([labelled])
    tracker = GitHubTracker(transport, "octo/example", max_conflict_attempts=2)

    assert tracker.is_conflict_eligible(labelled) is False
    assert tracker.runnable_queue() == ()
    assert tracker.claim_next() is None
    assert (
        ineligibility_reason(labelled, 7, max_conflict_attempts=2)
        == "issue #7 requires human implementation (labelled `ready-for-human`)"
    )


def test_conflict_routing_matches_queue_gating() -> None:
    assert is_conflict_claim(CONFLICT) is True
    assert is_conflict_claim(CONFLICT | READY) is False
    assert is_conflict_claim(CONFLICT | ROUND) is False
    assert is_conflict_claim(CONFLICT | HUMAN) is False
    assert is_conflict_claim(frozenset()) is False


def test_ready_for_agent_takes_precedence_and_clears_label_at_claim() -> None:
    target = issue(7, labels=READY | CONFLICT)
    transport = ConflictTransport([target])
    tracker = GitHubTracker(transport, "octo/example", max_conflict_attempts=1)

    assert [item.number for item in tracker.runnable_queue()] == [7]
    claim = tracker.claim_next()

    assert claim is not None and claim.issue.number == 7
    assert (target.id, "merge-conflict") in transport.removed_labels
    assert (target.id, "round-finished") not in transport.removed_labels


def test_round_finished_takes_precedence_over_conflict() -> None:
    target = issue(7, labels=ROUND | CONFLICT)
    transport = ConflictTransport([target])
    tracker = GitHubTracker(
        transport, "octo/example", max_auto_continuations=2, max_conflict_attempts=1
    )

    assert [item.number for item in tracker.runnable_queue()] == [7]
    claim = tracker.claim_next()

    assert claim is not None and claim.issue.number == 7
    assert (target.id, "round-finished") in transport.removed_labels
    assert (target.id, "merge-conflict") in transport.removed_labels


def test_claim_time_revalidation_skips_relabelled_conflict() -> None:
    queued = issue(7, labels=CONFLICT)
    transport = ConflictTransport(
        [queued], refreshed={7: issue(7, labels=CONFLICT | HUMAN)}
    )
    tracker = GitHubTracker(transport, "octo/example", max_conflict_attempts=1)

    assert [item.number for item in tracker.runnable_queue()] == [7]
    assert tracker.claim_next() is None


def test_claim_time_revalidation_routes_requeued_conflict_to_ordinary() -> None:
    queued = issue(7, labels=CONFLICT)
    transport = ConflictTransport(
        [queued], refreshed={7: issue(7, labels=CONFLICT | READY)}
    )
    tracker = GitHubTracker(transport, "octo/example", max_conflict_attempts=1)

    claim = tracker.claim_next()

    assert claim is not None and claim.issue.number == 7
    assert (queued.id, "merge-conflict") in transport.removed_labels


def test_ineligibility_reason_for_conflict_targets() -> None:
    assert (
        ineligibility_reason(issue(7, labels=CONFLICT), 7, max_conflict_attempts=1)
        is None
    )
    assert (
        ineligibility_reason(issue(7, labels=CONFLICT), 7, max_conflict_attempts=0)
        == "issue #7 is not labelled `ready-for-agent`"
    )
    assert (
        ineligibility_reason(
            issue(7, labels=CONFLICT, body="  "), 7, max_conflict_attempts=1
        )
        == "issue #7 has an empty body"
    )


def test_submit_next_issue_accepts_conflict_within_budget(tmp_path: Path) -> None:
    from simple_coding_agent.control import ControlStore

    target = issue(7, labels=CONFLICT)
    tracker = FakeTracker({7: target})
    lifecycle = AgentLifecycle(
        tracker=tracker,  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: profile(),
        publisher=FakePublisher(),
        attempt_runner=raising_runner,  # type: ignore[arg-type]
        control_store=ControlStore(tmp_path),
        sleeper=lambda seconds: None,
        max_conflict_attempts=1,
        command_runner=FakeCommandRunner([]),  # type: ignore[arg-type]
    )

    record = lifecycle.submit_next_issue("req-1", 7)

    assert record.acknowledgement is CommandAcknowledgement.ACCEPTED


def test_submit_next_issue_rejects_conflict_with_zero_budget(tmp_path: Path) -> None:
    from simple_coding_agent.control import ControlStore

    tracker = FakeTracker({7: issue(7, labels=CONFLICT)})
    lifecycle = AgentLifecycle(
        tracker=tracker,  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: profile(),
        publisher=FakePublisher(),
        attempt_runner=raising_runner,  # type: ignore[arg-type]
        control_store=ControlStore(tmp_path),
        sleeper=lambda seconds: None,
        max_conflict_attempts=0,
        command_runner=FakeCommandRunner([]),  # type: ignore[arg-type]
    )

    with pytest.raises(NextIssueRejectedError):
        lifecycle.submit_next_issue("req-1", 7)


# ---------------------------------------------------------------------------
# lifecycle: entry check skips before any git work
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pr", "reason", "expect_bootstrap"),
    [
        (None, "no_open_pr", False),
        (pull_request(draft=True), "draft", False),
        (pull_request(mergeable="MERGEABLE"), "not_conflicting", False),
        (pull_request(base_ref="other"), "base_mismatch", True),
    ],
)
def test_entry_check_skips_without_touching_the_branch(
    tmp_path: Path, pr: PullRequest | None, reason: str, expect_bootstrap: bool
) -> None:
    pull_requests = {} if pr is None else {"agent/issue-7": pr}
    result, tracker, publisher, workspace, runner, events = run_conflict(
        tmp_path, pull_requests=pull_requests
    )

    assert result.outcome is AttemptOutcome.NO_CHANGES
    assert len(publisher.conflict_requests) == 1
    request = publisher.conflict_requests[0]
    assert isinstance(request, ConflictPublicationRequest)
    assert request.conflict_outcome == CONFLICT_OUTCOME_SKIPPED
    assert request.reason == reason
    assert request.push_branch is False
    # No git work on the published branch: it is never prepared or rebased.
    assert workspace.prepared_bases == []
    assert workspace.bootstrap_bases == (["main"] if expect_bootstrap else [])
    assert runner.calls == []
    # The pull request was re-read before anything else ran.
    assert publisher.pr_lookups == ["agent/issue-7"]
    # Labels and assignee are released without adding queue labels.
    assert tracker.cleanup == [(7, "merge-conflict", "agent-id")]
    names = event_names(events)
    assert "conflict_attempt_claimed" in names
    assert "auto_continuation_claimed" not in names


def test_unknown_mergeability_proceeds_to_the_local_rebase(tmp_path: Path) -> None:
    result, tracker, publisher, workspace, runner, events = run_conflict(
        tmp_path,
        pull_requests={"agent/issue-7": pull_request(mergeable="UNKNOWN")},
        command_results=[ok_result("setup.sh"), ok_result()],
    )

    assert result.outcome is AttemptOutcome.COMPLETE
    assert publisher.conflict_requests[0].push_branch is True


# ---------------------------------------------------------------------------
# lifecycle: fast path outcomes
# ---------------------------------------------------------------------------


def test_clean_rebase_with_green_check_pushes_without_a_model_call(
    tmp_path: Path,
) -> None:
    result, tracker, publisher, workspace, runner, events = run_conflict(
        tmp_path,
        pull_requests={"agent/issue-7": pull_request()},
        command_results=[ok_result("setup.sh"), ok_result()],
    )

    assert result.outcome is AttemptOutcome.COMPLETE
    assert len(publisher.conflict_requests) == 1
    request = publisher.conflict_requests[0]
    assert request.conflict_outcome == CONFLICT_OUTCOME_SUCCESS
    assert request.push_branch is True
    assert request.check_command == "pytest"
    assert request.check_exit_code == 0
    # Setup ran before the single profile check, with their own timeouts.
    assert [call[0] for call in runner.calls] == [("setup.sh",), ("pytest",)]
    assert [call[1] for call in runner.calls] == [120, 300]
    # The claim event fired; the automatic event never does for conflicts.
    names = event_names(events)
    assert "conflict_attempt_claimed" in names
    assert "auto_continuation_claimed" not in names
    assert "attempt_phase_transitioned" in names
    # Release removes the conflict signal and the assignee only.
    assert tracker.cleanup == [(7, "merge-conflict", "agent-id")]


def test_conflicting_rebase_runs_one_session_and_reports_unresolved(
    tmp_path: Path,
) -> None:
    workspace = InMemoryWorkspace(tmp_path / "repo", resolve_after_model=False)
    workspace.prepare_conflicts = ("a.txt", "b.txt")
    calls: list[str] = []
    executor = FakeConflictExecutor(calls=calls)
    executor.after_execute = workspace.note_model_finished
    result, tracker, publisher, active, runner, events = run_conflict(
        tmp_path,
        pull_requests={"agent/issue-7": pull_request()},
        workspace=workspace,
        conflict_executor=executor,
    )

    assert result.outcome is AttemptOutcome.INCOMPLETE
    request = publisher.conflict_requests[0]
    assert request.conflict_outcome == CONFLICT_OUTCOME_REBASE_CONFLICT
    assert request.push_branch is False
    assert request.conflicted_files == ("a.txt", "b.txt")
    # Exactly one model session ran; the narrow prompt carries no
    # full-implementation continuation and no mandatory review instruction.
    assert calls == ["execute"]
    assert executor.captured_prompt is not None
    assert "then continue the implementation" not in executor.captured_prompt
    assert "an attempt with commits but no `code-review` invocation" not in (
        executor.captured_prompt
    )
    # The unresolved rebase was aborted; setup and the check never ran.
    assert active.abort_calls != []
    assert runner.calls == []
    assert tracker.cleanup == [(7, "merge-conflict", "agent-id")]


def test_red_check_runs_one_session_and_reports_when_still_red(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    executor = FakeConflictExecutor(calls=calls)
    result, tracker, publisher, workspace, runner, events = run_conflict(
        tmp_path,
        pull_requests={"agent/issue-7": pull_request()},
        # The pre-model setup/check plus the post-model setup/check rerun.
        command_results=[
            ok_result("setup.sh"),
            failed_result(),
            ok_result("setup.sh"),
            failed_result(),
        ],
        conflict_executor=executor,
    )

    assert result.outcome is AttemptOutcome.INCOMPLETE
    request = publisher.conflict_requests[0]
    assert request.conflict_outcome == CONFLICT_OUTCOME_CHECK_FAILED
    assert request.push_branch is False
    assert request.check_command == "pytest"
    assert request.check_exit_code == 1
    # Exactly one model session ran with the red check output in its prompt.
    assert calls == ["execute"]
    assert executor.captured_prompt is not None
    assert "Failing check output" in executor.captured_prompt
    assert [call[0] for call in runner.calls] == [
        ("setup.sh",),
        ("pytest",),
        ("setup.sh",),
        ("pytest",),
    ]
    assert tracker.cleanup == [(7, "merge-conflict", "agent-id")]


def test_setup_failure_is_an_infrastructure_error_without_a_push(
    tmp_path: Path,
) -> None:
    result, tracker, publisher, workspace, runner, events = run_conflict(
        tmp_path,
        pull_requests={"agent/issue-7": pull_request()},
        command_results=[failed_result("setup.sh")],
    )

    assert result.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    request = publisher.conflict_requests[0]
    assert request.conflict_outcome == CONFLICT_OUTCOME_SETUP_FAILED
    assert request.push_branch is False
    # The check never ran after setup failed.
    assert len(runner.calls) == 1
    assert tracker.cleanup == [(7, "merge-conflict", "agent-id")]


def test_unexpected_error_still_publishes_a_conflict_result(tmp_path: Path) -> None:
    def broken_profile(_: Path) -> RepositoryProfile:
        raise RuntimeError("profile exploded")

    tracker = FakeTracker({7: issue(7, labels=CONFLICT)}, fifo=[7])
    publisher = FakePublisher(pull_requests={"agent/issue-7": pull_request()})
    events: list[tuple[str, str, str, int | None]] = []
    lifecycle = AgentLifecycle(
        tracker=tracker,  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=broken_profile,
        publisher=publisher,
        attempt_runner=raising_runner,  # type: ignore[arg-type]
        control_store=None,
        sleeper=lambda seconds: None,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level, issue_number)
        ),
        max_conflict_attempts=1,
        command_runner=FakeCommandRunner([]),  # type: ignore[arg-type]
    )

    result = lifecycle.run_once()

    assert result.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    assert len(publisher.conflict_requests) == 1
    request = publisher.conflict_requests[0]
    assert request.conflict_outcome == CONFLICT_OUTCOME_ERROR
    assert request.push_branch is False
    assert tracker.cleanup == [(7, "merge-conflict", "agent-id")]


def test_missing_command_runner_is_an_error_not_a_crash(tmp_path: Path) -> None:
    tracker = FakeTracker({7: issue(7, labels=CONFLICT)}, fifo=[7])
    publisher = FakePublisher(pull_requests={"agent/issue-7": pull_request()})
    lifecycle = AgentLifecycle(
        tracker=tracker,  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: profile(),
        publisher=publisher,
        attempt_runner=raising_runner,  # type: ignore[arg-type]
        control_store=None,
        sleeper=lambda seconds: None,
        max_conflict_attempts=1,
        command_runner=None,
    )

    result = lifecycle.run_once()

    assert result.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    assert publisher.conflict_requests[0].conflict_outcome == CONFLICT_OUTCOME_ERROR
    assert tracker.cleanup == [(7, "merge-conflict", "agent-id")]


# ---------------------------------------------------------------------------
# publication: the conflict result comment and its marker
# ---------------------------------------------------------------------------


def _comment_request(**overrides: object) -> ConflictPublicationRequest:
    base: dict[str, object] = {
        "issue_number": 7,
        "issue_title": "Issue 7",
        "branch": "agent/issue-7",
        "started_at": "2026-10-11T10:00:00Z",
        "base_branch": "main",
        "outcome": AttemptOutcome.COMPLETE,
        "conflict_outcome": CONFLICT_OUTCOME_SUCCESS,
        "reason": "",
        "details": "Rebased cleanly.",
        "push_branch": True,
        "pull_request": pull_request(),
        "check_command": "pytest",
        "check_exit_code": 0,
        "conflicted_files": (),
        "auto_merge_opt_in": False,
    }
    base.update(overrides)
    return ConflictPublicationRequest(**base)  # type: ignore[arg-type]


def test_success_comment_carries_attempt_and_conflict_markers_only() -> None:
    body = _conflict_result_comment(
        _comment_request(),
        AttemptOutcome.COMPLETE,
        CONFLICT_OUTCOME_SUCCESS,
        "https://example.test/tree/agent/issue-7",
        pull_request(),
        "Rebased cleanly.",
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->",
        None,
        "abc123",
    )

    assert "## Conflict Attempt Result: success" in body
    assert "<!-- agent-attempt: 2026-10-11T10:00:00Z -->" in body
    parsed = parse_conflict_attempt(body)
    assert parsed is not None
    assert parsed["outcome"] == "success"
    assert parsed["sha"] == "abc123"
    assert "requeue-source" not in body


def test_skipped_comment_names_the_entry_check_reason() -> None:
    body = _conflict_result_comment(
        _comment_request(
            outcome=AttemptOutcome.NO_CHANGES,
            conflict_outcome=CONFLICT_OUTCOME_SKIPPED,
            reason="draft",
            push_branch=False,
        ),
        AttemptOutcome.NO_CHANGES,
        CONFLICT_OUTCOME_SKIPPED,
        None,
        pull_request(draft=True),
        "A draft.",
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->",
        None,
        None,
    )

    assert "## Conflict Attempt Result: skipped" in body
    parsed = parse_conflict_attempt(body)
    assert parsed is not None
    assert parsed["outcome"] == "skipped"
    assert parsed["reason"] == "draft"
    assert parsed["sha"] == "none"
    assert "`agent/issue-7` (not pushed)" in body


def test_failure_comment_lists_conflicted_files_without_pushing() -> None:
    body = _conflict_result_comment(
        _comment_request(
            outcome=AttemptOutcome.INCOMPLETE,
            conflict_outcome=CONFLICT_OUTCOME_REBASE_CONFLICT,
            details="Conflicts.",
            push_branch=False,
            conflicted_files=("a.txt",),
            check_command="not run",
            check_exit_code=None,
        ),
        AttemptOutcome.INCOMPLETE,
        CONFLICT_OUTCOME_REBASE_CONFLICT,
        None,
        pull_request(),
        "Conflicts.",
        "<!-- agent-attempt: 2026-10-11T10:00:00Z -->",
        None,
        None,
    )

    assert "Conflicted files: a.txt" in body
    assert parse_conflict_attempt(body) is not None
    assert parse_conflict_attempt(body)["outcome"] == "rebase_conflict"
