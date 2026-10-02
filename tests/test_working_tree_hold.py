"""A dirty working tree must hold intake without exiting the process.

Regression tests for the ``core.6458`` incident: ``Lifecycle.run_once``
checked ``workspace.is_clean()`` before anything else and raised
``SystemExit(1)`` on a dirty tree — even with intake stopped and no claim
pending. Under ``restart: unless-stopped`` the container then restarted
every few seconds, and because the process exited before serving the
control socket, ``agentctl status`` could not connect.

The new contract:
- cleanliness is checked only when a claim could happen (after the intake
  gate), never while intake is stopped;
- a dirty tree blocks the claim but the process stays up, sleeps the
  normal poll interval, and keeps serving ``agentctl status`` (which
  reports the hold in its ``recovery`` block);
- ``working_tree_dirty`` is logged once per dirty episode (naming the
  first offending paths), not on every poll.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.completion import CompletionDecision, PublicationPath
from simple_coding_agent.control import ControlStore
from simple_coding_agent.github_tracker import Assignment, Claim, TrackerIssue
from simple_coding_agent.lifecycle import AgentLifecycle, AttemptEvidence, LifecycleStatus
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace


def test_dirty_tree_with_intake_stopped_stays_up_and_logs_nothing(tmp_path: Path) -> None:
    """Intake stopped + dirty tree: stay up, sleep, claim nothing, log nothing."""

    lifecycle, tracker, sleeps, events = make_lifecycle(tmp_path, next_claim=claim(24))
    lifecycle.submit_stop("req-stop")
    workspace_of(lifecycle).make_dirty("core.6458", "")

    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.IDLE
    assert tracker.claimed == []
    assert sleeps == [60]
    assert [event for event, _, _ in events] == ["intake_stopped"]
    # The live process survives, so the operator socket keeps working.
    snapshot = lifecycle.control_status("owner/repo")
    assert snapshot["intake"] == "stopped"


def test_dirty_tree_with_intake_running_holds_claim_and_logs_once_with_paths(
    tmp_path: Path,
) -> None:
    """Intake running + dirty tree: no claim, one rate-limited log with paths."""

    lifecycle, tracker, sleeps, events = make_lifecycle(tmp_path, next_claim=claim(24))
    workspace_of(lifecycle).make_dirty("core.6458", "")
    workspace_of(lifecycle).make_dirty("notes/scratch.txt", "wip")

    first = lifecycle.run_once()
    second = lifecycle.run_once()

    assert first.status is LifecycleStatus.IDLE
    assert second.status is LifecycleStatus.IDLE
    assert tracker.claimed == []
    assert sleeps == [60, 60]
    dirty_logs = [detail for event, detail, _ in events if event == "working_tree_dirty"]
    assert len(dirty_logs) == 1
    assert "core.6458" in dirty_logs[0]
    assert "notes/scratch.txt" in dirty_logs[0]


def test_dirty_hold_recovers_and_reclaims_when_the_tree_is_clean(tmp_path: Path) -> None:
    """A cleaned tree resets the hold: the next poll claims and re-dirty logs again."""

    lifecycle, tracker, sleeps, events = make_lifecycle(tmp_path, next_claim=claim(24))
    workspace = workspace_of(lifecycle)
    workspace.make_dirty("core.6458", "")

    assert lifecycle.run_once().status is LifecycleStatus.IDLE
    assert tracker.claimed == []

    workspace.dirty.clear()
    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.ATTEMPTED
    assert tracker.claimed == [24]
    assert any(event == "working_tree_clean" for event, _, _ in events)

    # A fresh dirty episode logs again exactly once.
    workspace.make_dirty("core.6458", "")
    lifecycle.run_once()
    lifecycle.run_once()
    assert len([event for event, _, _ in events if event == "working_tree_dirty"]) == 2


def test_status_reports_dirty_hold_with_paths_while_intake_runs(tmp_path: Path) -> None:
    """The operator-visible state: status shows the dirty-tree hold and repair action."""

    lifecycle, _, _, _ = make_lifecycle(tmp_path, next_claim=claim(24))
    workspace_of(lifecycle).make_dirty("core.6458", "")

    assert lifecycle.run_once().status is LifecycleStatus.IDLE
    snapshot = lifecycle.control_status("owner/repo")

    recovery = snapshot["recovery"]
    assert recovery is not None
    assert recovery["issue_number"] is None
    assert "dirty or untracked" in recovery["hold_reason"]
    assert "core.6458" in recovery["hold_reason"]
    assert recovery["next_action"]


def test_unreadable_tree_holds_without_claiming(tmp_path: Path) -> None:
    """A tree whose state cannot be read fails closed: hold, don't exit, don't claim."""

    lifecycle, tracker, sleeps, events = make_lifecycle(tmp_path, next_claim=claim(24))
    workspace = workspace_of(lifecycle)
    workspace.make_dirty("core.6458", "")
    workspace.break_clean_checks()

    result = lifecycle.run_once()

    assert result.status is LifecycleStatus.IDLE
    assert tracker.claimed == []
    assert sleeps == [60]
    assert len([event for event, _, _ in events if event == "working_tree_dirty"]) == 1


def test_git_dirty_paths_names_untracked_files_first(tmp_path: Path) -> None:
    """The production workspace names the offending paths for the log detail."""

    from tests.fakes import repository_with_main
    from simple_coding_agent.git_workspace import GitWorkspace

    remote, seed = repository_with_main(tmp_path)
    clone = tmp_path / "clone"
    workspace = GitWorkspace(clone, str(remote), token_provider=lambda: "secret-token")
    workspace.prepare_attempt(base_branch="main", issue_number=24)
    (clone / "core.6458").write_text("")
    (clone / "notes").mkdir(exist_ok=True)
    (clone / "notes" / "scratch.txt").write_text("wip")

    assert workspace.is_clean() is False
    assert workspace.dirty_paths() == ("core.6458", "notes/scratch.txt")


def claim(number: int) -> Claim:
    return Claim(issue(number), Assignment(f"issue-{number}", "agent-id"))


def issue(number: int) -> TrackerIssue:
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


def make_lifecycle(
    tmp_path: Path, *, next_claim: Claim | None
) -> tuple[AgentLifecycle, FakeTracker, list[float], list[tuple[str, str, str]]]:
    tracker = FakeTracker(next_claim=next_claim)
    workspace = DirtyHoldWorkspace(tmp_path / "repo", issue_number=24)
    store = ControlStore(tmp_path)
    sleeps: list[float] = []
    events: list[tuple[str, str, str]] = []
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=lambda received_claim, profile, prepared: AttemptEvidence(
            decision=CompletionDecision(None, True, PublicationPath.COMPLETE, ("ready",)),
            check_command="pytest",
            check_exit_code=0,
            review_cycles=1,
            review_findings="all clear",
            details="Implemented the lifecycle.",
        ),
        control_store=store,
        poll_interval=60,
        sleeper=sleeps.append,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail, level)
        ),
    )
    return lifecycle, tracker, sleeps, events


def workspace_of(lifecycle: AgentLifecycle) -> DirtyHoldWorkspace:
    return lifecycle._workspace  # noqa: SLF001 - test seam


class DirtyHoldWorkspace(InMemoryWorkspace):
    """In-memory workspace that can also simulate an unreadable tree."""

    def __init__(self, root: Path, *, issue_number: int = 24) -> None:
        super().__init__(root, issue_number=issue_number)
        self._clean_checks_broken = False

    def break_clean_checks(self) -> None:
        self._clean_checks_broken = True

    def is_clean(self) -> bool:
        if self._clean_checks_broken:
            raise OSError("git status failed")
        return super().is_clean()
