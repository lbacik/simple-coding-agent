"""Last-finished attempt telemetry: report the previous attempt between runs.

Between attempts (and alongside a running one) telemetry reports the last
finished attempt: issue, title, outcome, tokens, turns, finish time and pull
request. The value is updated at finalisation and seeded from disk at
startup so it survives agent restarts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.completion import (
    AttemptOutcome,
    CompletionDecision,
    PublicationPath,
)
from simple_coding_agent.control import ControlStore
from simple_coding_agent.github_tracker import TrackerIssue
from simple_coding_agent.lifecycle import AgentLifecycle, AttemptEvidence
from simple_coding_agent.observability import AttemptArchive
from simple_coding_agent.publication import PullRequest
from simple_coding_agent.telemetry import TelemetryReporter, load_last_finished
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace


_CREATED_BASE = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


def make_reporter(**overrides: object) -> TelemetryReporter:
    values: dict[str, object] = {
        "repository": "owner/repo",
        "agent_version": "0.0-test",
        "sdk_version": "0.2.163-test",
        "cli_version": "2.1.286-test",
        "process_started_at": "2026-01-01T00:00:00Z",
        "hard_ceiling": 1000,
        "turns_max": 50,
        "timeout_seconds": 600,
        "continuation_max": 2,
    }
    values.update(overrides)
    return TelemetryReporter(**values)  # type: ignore[arg-type]


class _StubControl:
    def __init__(self, section: dict | None = None) -> None:
        self._section = dict(
            section or {"control_observed_at": "2026-10-14T12:00:00Z"}
        )

    def telemetry_control_snapshot(self) -> dict:
        return dict(self._section)


def finished_summary(**overrides: object) -> dict:
    summary: dict[str, object] = {
        "attempt_id": "2026-10-14T12:00:00.000001Z",
        "issue_number": 7,
        "issue_title": "Fix the widget",
        "outcome": "complete",
        "finished_at": "2026-10-14T13:00:00Z",
        "budget_used": 812345,
        "turns": 23,
        "pull_request": {"number": 11, "url": "https://example.test/11"},
    }
    summary.update(overrides)
    return summary


def tracked_issue(number: int, **overrides: object) -> TrackerIssue:
    base: dict[str, object] = {
        "id": f"issue-{number}",
        "number": number,
        "title": f"Issue {number}",
        "body": f"Implement {number}.",
        "created_at": _CREATED_BASE + timedelta(seconds=number),
        "state": "OPEN",
        "labels": frozenset({"ready-for-agent"}),
        "assignee_logins": (),
        "blocked_by": 0,
        "author_login": "reporter",
    }
    base.update(overrides)
    return TrackerIssue(**base)  # type: ignore[arg-type]


def complete_attempt_evidence() -> AttemptEvidence:
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


def archive_factory(tmp_path: Path):
    def _factory(issue_number: int, started_at: str) -> AttemptArchive:
        return AttemptArchive(
            tmp_path, issue_number=issue_number, started_at=started_at
        )

    return _factory


def make_lifecycle(
    tmp_path: Path,
    *,
    reporter: TelemetryReporter | None = None,
    publisher: FakePublisher | None = None,
    attempt_runner=None,
    seed_data_dir: bool = True,
) -> tuple[AgentLifecycle, FakeTracker, AttemptStateStore]:
    tracker = FakeTracker({7: tracked_issue(7, title="Fix the widget")}, fifo=[7])
    state = AttemptStateStore(tmp_path)
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=state,
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=publisher
        if publisher is not None
        else FakePublisher(
            pull_request=PullRequest(number=11, url="https://example.test/11")
        ),
        attempt_runner=attempt_runner
        if attempt_runner is not None
        else (lambda claim, profile, prepared: complete_attempt_evidence()),
        control_store=ControlStore(tmp_path),
        repository="owner/repo",
        telemetry=reporter if reporter is not None else make_reporter(),
        sleeper=lambda seconds: None,
        max_auto_continuations=2,
        attempt_archive_factory=archive_factory(tmp_path),
        data_dir=tmp_path if seed_data_dir else None,
    )
    return lifecycle, tracker, state


def write_archive(
    tmp_path: Path,
    issue_number: int,
    started_at: str,
    record: dict,
) -> Path:
    archive = AttemptArchive(
        tmp_path, issue_number=issue_number, started_at=started_at
    )
    archive.write_attempt(record)
    return archive.directory / "attempt.json"


# ---------------------------------------------------------------------------
# reporter: finish_attempt(summary)
# ---------------------------------------------------------------------------


def test_finish_attempt_stores_the_shaped_last_finished() -> None:
    reporter = make_reporter()
    reporter.finish_attempt(finished_summary())

    last = reporter.snapshot(_StubControl())["last_finished"]

    assert last == {
        "attempt_id": "2026-10-14T12:00:00.000001Z",
        "issue_number": 7,
        "issue_title": "Fix the widget",
        "outcome": "complete",
        "finished_at": "2026-10-14T13:00:00Z",
        "budget_used": {"value": 812345, "basis": "measured"},
        "turns": {"value": 23, "basis": "measured"},
        "pull_request": {"number": 11, "url": "https://example.test/11"},
    }


def test_finish_attempt_without_a_summary_keeps_last_finished() -> None:
    reporter = make_reporter()
    reporter.begin_attempt(
        {
            "attempt_id": "2026-10-14T12:00:00.000001Z",
            "issue_number": 7,
        }
    )
    reporter.finish_attempt(finished_summary())
    before = reporter.snapshot(_StubControl())["last_finished"]

    reporter.begin_attempt(
        {
            "attempt_id": "2026-10-14T14:00:00.000001Z",
            "issue_number": 8,
        }
    )
    reporter.finish_attempt()

    snapshot = reporter.snapshot(_StubControl())
    assert snapshot["current_attempt"] is None
    assert snapshot["last_finished"] == before


def test_last_finished_is_reported_alongside_an_active_current_attempt() -> None:
    reporter = make_reporter()
    reporter.finish_attempt(finished_summary())
    reporter.begin_attempt(
        {
            "attempt_id": "2026-10-14T14:00:00.000001Z",
            "issue_number": 8,
        }
    )

    snapshot = reporter.snapshot(_StubControl())

    assert snapshot["current_attempt"] is not None
    assert snapshot["current_attempt"]["issue_number"] == 8
    assert snapshot["last_finished"] is not None
    assert snapshot["last_finished"]["issue_number"] == 7


def test_finish_attempt_with_a_legacy_summary_reports_unknown_title_and_null_pr() -> None:
    reporter = make_reporter()
    reporter.finish_attempt(
        {
            "attempt_id": "2026-10-14T12:00:00.000001Z",
            "issue_number": 7,
            "outcome": "complete",
            "finished_at": "2026-10-14T13:00:00Z",
            "budget_used": 100,
            "turns": 2,
        }
    )

    last = reporter.snapshot(_StubControl())["last_finished"]

    assert last is not None
    assert last["issue_title"] == "unknown"
    assert last["pull_request"] is None
    assert last["budget_used"] == {"value": 100, "basis": "measured"}
    assert last["turns"] == {"value": 2, "basis": "measured"}


def test_finish_attempt_with_missing_figures_reports_them_unavailable() -> None:
    reporter = make_reporter()
    reporter.finish_attempt(finished_summary(budget_used=None, turns=None))

    last = reporter.snapshot(_StubControl())["last_finished"]

    assert last is not None
    assert last["budget_used"] == {
        "value": None,
        "basis": "unavailable",
        "reason": "not reported",
    }
    assert last["turns"]["basis"] == "unavailable"
    # An unavailable figure is never shown as zero.
    assert last["turns"]["value"] is None


def test_malformed_finish_summary_keeps_the_previous_view(
    caplog,
) -> None:
    reporter = make_reporter()
    reporter.finish_attempt(finished_summary())
    before = reporter.snapshot(_StubControl())["last_finished"]

    with caplog.at_level("WARNING", logger="simple_coding_agent.telemetry"):
        reporter.finish_attempt({"issue_number": 7})
        reporter.finish_attempt("not-a-mapping")

    assert reporter.snapshot(_StubControl())["last_finished"] == before
    kept = [
        record
        for record in caplog.records
        if "keeping previous view" in record.getMessage()
    ]
    assert len(kept) == 1


# ---------------------------------------------------------------------------
# lifecycle: finalisation updates last_finished and the archive record
# ---------------------------------------------------------------------------


def test_finalization_updates_last_finished_and_archives_the_pull_request(
    tmp_path: Path,
) -> None:
    reporter = make_reporter()
    state_holder: dict[str, object] = {}

    lifecycle, _, state = make_lifecycle(tmp_path, reporter=reporter)
    # The model boundary records the token budget in the archive; the fake
    # runner stands in for it here through the same archive seam.

    def runner_with_budget(claim, profile, prepared):
        checkpoint = state.read()
        assert checkpoint is not None
        AttemptArchive(
            tmp_path,
            issue_number=claim.issue.number,
            started_at=checkpoint.started_at,
        ).write_attempt(
            {
                "model_usage": {"main": {"input_tokens": 1}},
                "token_budget": {"actual_tokens": 812345, "turns": 23},
            }
        )
        state_holder["attempt_id"] = checkpoint.started_at
        return complete_attempt_evidence()

    lifecycle.set_attempt_runner(runner_with_budget)

    result = lifecycle.run_once()

    assert result.status.value == "attempted"
    snapshot = lifecycle.telemetry_snapshot()
    assert snapshot["current_attempt"] is None
    last = snapshot["last_finished"]
    assert last is not None
    assert last["attempt_id"] == state_holder["attempt_id"]
    assert last["issue_number"] == 7
    assert last["issue_title"] == "Fix the widget"
    assert last["outcome"] == "complete"
    assert isinstance(last["finished_at"], str) and last["finished_at"]
    assert last["budget_used"] == {"value": 812345, "basis": "measured"}
    assert last["turns"] == {"value": 23, "basis": "measured"}
    assert last["pull_request"] == {
        "number": 11,
        "url": "https://example.test/11",
    }

    record = AttemptArchive(
        tmp_path,
        issue_number=7,
        started_at=str(state_holder["attempt_id"]),
    ).read_attempt()
    assert record["pull_request"] == {
        "number": 11,
        "url": "https://example.test/11",
    }
    assert record["outcome"] == "complete"


def test_finalization_without_model_figures_reports_them_unavailable(
    tmp_path: Path,
) -> None:
    reporter = make_reporter()
    lifecycle, _, _ = make_lifecycle(tmp_path, reporter=reporter)

    assert lifecycle.run_once().status.value == "attempted"

    last = lifecycle.telemetry_snapshot()["last_finished"]
    assert last is not None
    assert last["pull_request"] == {
        "number": 11,
        "url": "https://example.test/11",
    }
    assert last["budget_used"]["basis"] == "unavailable"
    assert last["budget_used"]["value"] is None
    assert last["turns"]["basis"] == "unavailable"


# ---------------------------------------------------------------------------
# startup seeding from disk
# ---------------------------------------------------------------------------


def _run_one_finished_attempt(tmp_path: Path) -> dict:
    lifecycle, _, _ = make_lifecycle(tmp_path)
    assert lifecycle.run_once().status.value == "attempted"
    last = lifecycle.telemetry_snapshot()["last_finished"]
    assert last is not None
    return last


def test_restart_seeds_last_finished_from_disk(tmp_path: Path) -> None:
    expected = _run_one_finished_attempt(tmp_path)

    restarted_reporter = make_reporter()
    restarted, _, _ = make_lifecycle(tmp_path, reporter=restarted_reporter)

    assert restarted.telemetry_snapshot()["current_attempt"] is None
    assert restarted.telemetry_snapshot()["last_finished"] == expected


def test_no_history_seeds_null_last_finished(tmp_path: Path) -> None:
    lifecycle, _, _ = make_lifecycle(tmp_path)

    assert lifecycle.telemetry_snapshot()["last_finished"] is None


def test_seed_prefers_the_latest_completed_attempt(tmp_path: Path) -> None:
    write_archive(
        tmp_path,
        7,
        "2026-10-14T12:00:00.000001Z",
        {
            "started_at": "2026-10-14T12:00:00.000001Z",
            "completed_at": "2026-10-14T13:00:00Z",
            "outcome": "complete",
            "issue_title": "Older",
            "token_budget": {"actual_tokens": 100, "turns": 1},
            "pull_request": {"number": 11, "url": "https://example.test/11"},
        },
    )
    write_archive(
        tmp_path,
        8,
        "2026-10-14T14:00:00.000001Z",
        {
            "started_at": "2026-10-14T14:00:00.000001Z",
            "completed_at": "2026-10-14T15:00:00Z",
            "outcome": "incomplete",
            "issue_title": "Newer",
            "token_budget": {"actual_tokens": 200, "turns": 2},
        },
    )

    assert load_last_finished(tmp_path) == {
        "attempt_id": "2026-10-14T14:00:00.000001Z",
        "issue_number": 8,
        "issue_title": "Newer",
        "outcome": "incomplete",
        "finished_at": "2026-10-14T15:00:00Z",
        "budget_used": {"value": 200, "basis": "measured"},
        "turns": {"value": 2, "basis": "measured"},
        "pull_request": None,
    }


def test_seed_from_a_legacy_archive_reports_unknown_title_and_null_pr(
    tmp_path: Path,
) -> None:
    write_archive(
        tmp_path,
        7,
        "2026-10-14T12:00:00.000001Z",
        {
            "started_at": "2026-10-14T12:00:00.000001Z",
            "completed_at": "2026-10-14T13:00:00Z",
            "outcome": "complete",
            "token_budget": {"actual_tokens": 100, "turns": 2},
        },
    )

    last = load_last_finished(tmp_path)

    assert last is not None
    assert last["issue_title"] == "unknown"
    assert last["pull_request"] is None


def test_seed_ignores_attempts_without_an_outcome(tmp_path: Path) -> None:
    write_archive(
        tmp_path,
        7,
        "2026-10-14T12:00:00.000001Z",
        {"model_usage": {}},
    )

    assert load_last_finished(tmp_path) is None


def test_seed_returns_none_without_any_history(tmp_path: Path) -> None:
    assert load_last_finished(tmp_path) is None


def test_seeded_last_finished_survives_alongside_the_next_attempt(
    tmp_path: Path,
) -> None:
    lifecycle, tracker, _ = make_lifecycle(tmp_path)
    assert lifecycle.run_once().status.value == "attempted"
    tracker.issues[9] = tracked_issue(9, title="Next work")
    tracker.fifo.append(9)

    seen: list[dict] = []

    def capturing_runner(claim, profile, prepared):
        seen.append(lifecycle.telemetry_snapshot())
        return complete_attempt_evidence()

    lifecycle.set_attempt_runner(capturing_runner)
    assert lifecycle.run_once().status.value == "attempted"

    assert len(seen) == 1
    assert seen[0]["current_attempt"] is not None
    assert seen[0]["current_attempt"]["issue_number"] == 9
    assert seen[0]["last_finished"] is not None
    assert seen[0]["last_finished"]["issue_number"] == 7
