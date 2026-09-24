"""One-shot next-issue priority (issue #83)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.completion import AttemptOutcome, CompletionDecision, PublicationPath
from simple_coding_agent.control import (
    CommandAcknowledgement,
    ControlStore,
    IntakeState,
    NextIssueRejectedError,
    PayloadMismatchError,
    parse_next_issue,
)
from simple_coding_agent.github_tracker import (
    Assignment,
    Claim,
    GitHubTracker,
    GitHubTrackerError,
    TrackerIssue,
    ineligibility_reason,
)


def issue(number: int, **overrides: object) -> TrackerIssue:
    base: dict[str, object] = {
        "id": f"issue-{number}",
        "number": number,
        "title": f"Issue {number}",
        "body": f"Implement {number}.",
        "created_at": datetime(2026, 9, 20, 13, 0, tzinfo=UTC),
        "state": "OPEN",
        "labels": frozenset({"ready-for-agent"}),
        "assignee_logins": (),
        "blocked_by": 0,
        "author_login": "reporter",
    }
    base.update(overrides)
    return TrackerIssue(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# parse + eligibility reasons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, -3, "0", "-1", "abc", "12x", "", "  ", None, 3.5, True])
def test_parse_next_issue_rejects_non_positive_integers(value: object) -> None:
    with pytest.raises(NextIssueRejectedError):
        parse_next_issue(value)


@pytest.mark.parametrize("value, expected", [(24, 24), ("24", 24), ("  24  ", 24), ("+24", 24)])
def test_parse_next_issue_accepts_positive_integers(value: object, expected: int) -> None:
    assert parse_next_issue(value) == expected


def test_ineligibility_reason_names_each_failure_mode() -> None:
    assert ineligibility_reason(None, 7) == "issue #7 was not found"
    assert "not open" in (ineligibility_reason(issue(1, state="CLOSED"), 1) or "")
    assert "ready-for-agent" in (ineligibility_reason(issue(1, labels=frozenset()), 1) or "")
    assert "assigned" in (ineligibility_reason(issue(1, assignee_logins=("human",)), 1) or "")
    assert "empty body" in (ineligibility_reason(issue(1, body="  \n"), 1) or "")
    assert "blocker" in (ineligibility_reason(issue(1, blocked_by=2), 1) or "")
    assert ineligibility_reason(issue(1), 1) is None


# ---------------------------------------------------------------------------
# store: durable priority independent of stop plans
# ---------------------------------------------------------------------------


def test_store_accepts_priority_without_touching_intake_or_stop_plan(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)

    record = store.submit_next_issue("req-next", 24)

    assert record.kind == "next_issue"
    assert record.acknowledgement is CommandAcknowledgement.ACCEPTED
    assert store.intake_state() is IntakeState.RUNNING
    assert store.pending_command_id() is None
    assert store.next_issue_snapshot() == {"request_id": "req-next", "issue_number": 24}


def test_store_rejects_invalid_numbers_without_state_change(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)

    with pytest.raises(NextIssueRejectedError):
        store.submit_next_issue("req-bad", 0)

    assert store.next_issue_snapshot() is None
    assert store.get_command("req-bad") is None
    assert store.recent_commands() == []


def test_newer_priority_supersedes_only_the_earlier_priority(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop("req-stop", has_active_attempt=True)
    first = store.submit_next_issue("req-next-1", 24)

    second = store.submit_next_issue("req-next-2", 25)

    assert second.acknowledgement is CommandAcknowledgement.ACCEPTED
    assert "req-next-1" in second.detail
    replaced = store.get_command("req-next-1")
    assert replaced is not None
    assert replaced.acknowledgement is CommandAcknowledgement.SUPERSEDED
    assert "req-next-2" in replaced.detail
    assert first.sequence + 1 <= second.sequence
    # The stop plan is untouched: next-issue replacement is independent.
    assert store.pending_command_id() == "req-stop"
    assert store.intake_state() is IntakeState.STOPPING
    assert store.next_issue_snapshot() == {"request_id": "req-next-2", "issue_number": 25}


def test_identical_retry_returns_the_same_record(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    first = store.submit_next_issue("req-next", 24)

    retry = ControlStore(tmp_path).submit_next_issue("req-next", 24)

    assert retry.sequence == first.sequence
    assert retry.acknowledgement is first.acknowledgement
    assert store.next_issue_snapshot() == {"request_id": "req-next", "issue_number": 24}


def test_request_id_reused_with_a_different_payload_is_rejected(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_next_issue("req-1", 24)

    with pytest.raises(PayloadMismatchError):
        store.submit_next_issue("req-1", 25)

    assert store.next_issue_snapshot() == {"request_id": "req-1", "issue_number": 24}


def test_priority_survives_stopped_intake_and_restart(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_next_issue("req-next", 24)
    store.submit_stop("req-stop", has_active_attempt=False)
    assert store.intake_state() is IntakeState.STOPPED

    restarted = ControlStore(tmp_path)

    assert restarted.intake_state() is IntakeState.STOPPED
    assert restarted.next_issue_snapshot() == {"request_id": "req-next", "issue_number": 24}
    record = restarted.get_command("req-next")
    assert record is not None
    assert record.acknowledgement is CommandAcknowledgement.ACCEPTED


def test_resume_keeps_the_priority_while_replacing_the_stop(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_stop("req-stop", has_active_attempt=True)
    store.submit_next_issue("req-next", 24)

    store.submit_resume("req-resume", has_active_attempt=True)

    assert store.intake_state() is IntakeState.RUNNING
    assert store.pending_command_id() is None
    assert store.next_issue_snapshot() == {"request_id": "req-next", "issue_number": 24}
    assert store.get_command("req-next").acknowledgement is CommandAcknowledgement.ACCEPTED  # type: ignore[union-attr]


def test_complete_consumes_the_priority_once(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_next_issue("req-next", 24)

    completed = store.complete_next_issue(24)

    assert completed is not None
    assert completed.acknowledgement is CommandAcknowledgement.COMPLETED
    assert store.next_issue_snapshot() is None
    assert store.get_command("req-next").acknowledgement is CommandAcknowledgement.COMPLETED  # type: ignore[union-attr]


def test_complete_with_a_mismatched_issue_leaves_the_priority_intact(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_next_issue("req-next", 24)

    assert store.complete_next_issue(25) is None
    assert store.next_issue_snapshot() == {"request_id": "req-next", "issue_number": 24}
    assert store.get_command("req-next").acknowledgement is CommandAcknowledgement.ACCEPTED  # type: ignore[union-attr]


def test_fail_marks_not_fulfilled_with_the_reason(tmp_path: Path) -> None:
    store = ControlStore(tmp_path)
    store.submit_next_issue("req-next", 24)

    failed = store.fail_next_issue("issue #24 is already assigned to human")

    assert failed is not None
    assert failed.acknowledgement is CommandAcknowledgement.NOT_FULFILLED
    assert "assigned" in failed.detail
    assert store.next_issue_snapshot() is None


# ---------------------------------------------------------------------------
# lifecycle: acceptance
# ---------------------------------------------------------------------------


def test_submit_next_issue_accepts_an_eligible_target(tmp_path: Path) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, {24: issue(24)})

    record = lifecycle.submit_next_issue("req-next", 24)

    assert record.acknowledgement is CommandAcknowledgement.ACCEPTED
    assert store.next_issue_snapshot() == {"request_id": "req-next", "issue_number": 24}
    assert store.intake_state() is IntakeState.RUNNING


@pytest.mark.parametrize(
    "override",
    [
        {"state": "CLOSED"},
        {"labels": frozenset()},
        {"assignee_logins": ("human",)},
        {"body": "   "},
        {"blocked_by": 1},
    ],
)
def test_submit_next_issue_rejects_each_ineligibility_mode(
    tmp_path: Path, override: dict
) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, {24: issue(24, **override)})

    with pytest.raises(NextIssueRejectedError):
        lifecycle.submit_next_issue("req-next", 24)

    assert store.next_issue_snapshot() is None
    assert store.get_command("req-next") is None


def test_submit_next_issue_rejects_a_missing_issue(tmp_path: Path) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, {})

    with pytest.raises(NextIssueRejectedError, match="not found"):
        lifecycle.submit_next_issue("req-next", 24)

    assert store.get_command("req-next") is None


def test_rejection_leaves_an_existing_priority_intact(tmp_path: Path) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, {24: issue(24)})
    lifecycle.submit_next_issue("req-next-1", 24)
    tracker.issues[25] = issue(25, assignee_logins=("human",))

    with pytest.raises(NextIssueRejectedError):
        lifecycle.submit_next_issue("req-next-2", 25)

    assert store.next_issue_snapshot() == {"request_id": "req-next-1", "issue_number": 24}
    assert store.get_command("req-next-2") is None
    assert store.get_command("req-next-1").acknowledgement is CommandAcknowledgement.ACCEPTED  # type: ignore[union-attr]


def test_unverifiable_target_is_not_acknowledged(tmp_path: Path) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, {24: issue(24)})
    tracker.fetch_errors[24] = GitHubTrackerError("boom")

    with pytest.raises(NextIssueRejectedError, match="could not be verified"):
        lifecycle.submit_next_issue("req-next", 24)

    assert store.next_issue_snapshot() is None
    assert store.get_command("req-next") is None


def test_identical_retry_does_not_reverify_github(tmp_path: Path) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, {24: issue(24)})
    first = lifecycle.submit_next_issue("req-next", 24)
    # Eligibility is lost after acceptance; the retry must still return history.
    tracker.issues[24] = issue(24, assignee_logins=("human",))

    retry = lifecycle.submit_next_issue("req-next", 24)

    assert retry.sequence == first.sequence
    assert retry.acknowledgement is CommandAcknowledgement.ACCEPTED


# ---------------------------------------------------------------------------
# lifecycle: claim boundary
# ---------------------------------------------------------------------------


def test_prioritized_claim_is_rechecked_and_consumed_once(tmp_path: Path) -> None:
    lifecycle, tracker, store = make_lifecycle(
        tmp_path, {24: issue(24), 25: issue(25)}, fifo=[25, 24]
    )
    lifecycle.submit_next_issue("req-next", 24)

    result = lifecycle.run_once()

    assert result.status.value == "attempted"
    assert tracker.claimed_verified == [24]
    assert tracker.fifo_claims == []
    assert store.next_issue_snapshot() is None
    assert store.get_command("req-next").acknowledgement is CommandAcknowledgement.COMPLETED  # type: ignore[union-attr]

    # The one-shot priority is consumed: the next cycle uses FIFO again.
    tracker.fifo = [claim_for(tracker.issues[25])]
    assert lifecycle.run_once().status.value == "attempted"
    assert tracker.fifo_claims == [25]


def test_loss_of_eligibility_falls_back_to_fifo_in_the_same_cycle(
    tmp_path: Path,
) -> None:
    lifecycle, tracker, store = make_lifecycle(
        tmp_path, {24: issue(24), 25: issue(25)}, fifo=[25]
    )
    lifecycle.submit_next_issue("req-next", 24)
    tracker.issues[24] = issue(24, assignee_logins=("human",))

    result = lifecycle.run_once()

    assert result.status.value == "attempted"
    assert tracker.claimed_verified == []
    assert tracker.fifo_claims == [25]
    failed = store.get_command("req-next")
    assert failed is not None
    assert failed.acknowledgement is CommandAcknowledgement.NOT_FULFILLED
    assert "assigned" in failed.detail
    snapshot = lifecycle.control_status("owner/repo")
    assert snapshot["pending_next_issue"] is None
    assert snapshot["commands"]["req-next"]["acknowledgement"] == "not fulfilled"


def test_ambiguous_assignment_never_completes_nor_permits_another_claim(
    tmp_path: Path,
) -> None:
    lifecycle, tracker, store = make_lifecycle(
        tmp_path, {24: issue(24), 25: issue(25)}, fifo=[25]
    )
    lifecycle.submit_next_issue("req-next", 24)
    tracker.assign_errors[24] = GitHubTrackerError("connection reset during assign")

    with pytest.raises(GitHubTrackerError):
        lifecycle.run_once()

    # Never shown as completed; no FIFO fallback; no checkpoint started.
    assert store.get_command("req-next").acknowledgement is CommandAcknowledgement.ACCEPTED  # type: ignore[union-attr]
    assert store.next_issue_snapshot() == {"request_id": "req-next", "issue_number": 24}
    assert tracker.fifo_claims == []
    assert tracker.claimed_verified == []
    assert AttemptStateStore(tmp_path).read() is None

    # After the transport recovers, the same priority still applies.
    del tracker.assign_errors[24]
    tracker.fifo = []
    assert lifecycle.run_once().status.value == "attempted"
    assert tracker.claimed_verified == [24]
    assert store.get_command("req-next").acknowledgement is CommandAcknowledgement.COMPLETED  # type: ignore[union-attr]


def test_self_assignment_after_an_unconfirmed_claim_holds_for_reconciliation(
    tmp_path: Path,
) -> None:
    """A retry that finds the target assigned to us must not fail + FIFO.

    The target was unassigned at acceptance and only this process claims
    through the serialized boundary, so self-assignment means our own
    assignment went through while its reply was lost.
    """

    from simple_coding_agent.control import ControlStoreError

    lifecycle, tracker, store = make_lifecycle(
        tmp_path, {24: issue(24), 25: issue(25)}, fifo=[25]
    )
    lifecycle.submit_next_issue("req-next", 24)
    tracker.issues[24] = issue(24, assignee_logins=("agent",))

    with pytest.raises(ControlStoreError, match="ambiguous"):
        lifecycle.run_once()

    assert store.get_command("req-next").acknowledgement is CommandAcknowledgement.ACCEPTED  # type: ignore[union-attr]
    assert store.next_issue_snapshot() == {"request_id": "req-next", "issue_number": 24}
    assert tracker.fifo_claims == []
    assert AttemptStateStore(tmp_path).read() is None


def test_assignment_to_someone_else_still_fails_over_to_fifo(
    tmp_path: Path,
) -> None:
    lifecycle, tracker, store = make_lifecycle(
        tmp_path, {24: issue(24), 25: issue(25)}, fifo=[25]
    )
    lifecycle.submit_next_issue("req-next", 24)
    tracker.issues[24] = issue(24, assignee_logins=("human",))

    assert lifecycle.run_once().status.value == "attempted"

    assert tracker.fifo_claims == [25]
    assert store.get_command("req-next").acknowledgement is CommandAcknowledgement.NOT_FULFILLED  # type: ignore[union-attr]


def test_priority_waits_through_stopped_intake_and_applies_after_resume(
    tmp_path: Path,
) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, {24: issue(24)}, fifo=[])
    lifecycle.submit_next_issue("req-next", 24)
    lifecycle.submit_stop("req-stop")
    assert store.intake_state() is IntakeState.STOPPED

    assert lifecycle.run_once().status.value == "idle"
    assert tracker.claimed_verified == []
    assert tracker.fifo_claims == []
    assert store.next_issue_snapshot() == {"request_id": "req-next", "issue_number": 24}

    lifecycle.submit_resume("req-resume")
    assert lifecycle.run_once().status.value == "attempted"
    assert tracker.claimed_verified == [24]


def test_priority_does_not_resume_intake_on_its_own(tmp_path: Path) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, {24: issue(24)}, fifo=[])
    lifecycle.submit_stop("req-stop")
    lifecycle.submit_next_issue("req-next", 24)

    assert store.intake_state() is IntakeState.STOPPED
    assert lifecycle.run_once().status.value == "idle"
    assert tracker.claimed_verified == []


def test_priority_survives_restart_before_the_claim(tmp_path: Path) -> None:
    first, tracker, store = make_lifecycle(tmp_path, {24: issue(24)}, fifo=[])
    first.submit_next_issue("req-next", 24)

    second, tracker2, _ = make_lifecycle(tmp_path, {24: issue(24)}, fifo=[])
    assert second.control_status("owner/repo")["pending_next_issue"] == {
        "request_id": "req-next",
        "issue_number": 24,
    }

    assert second.run_once().status.value == "attempted"
    assert tracker2.claimed_verified == [24]


def test_replacement_is_visible_in_status_and_command_lookup(tmp_path: Path) -> None:
    lifecycle, tracker, store = make_lifecycle(tmp_path, {24: issue(24), 25: issue(25)})
    lifecycle.submit_next_issue("req-next-1", 24)
    lifecycle.submit_next_issue("req-next-2", 25)

    snapshot = lifecycle.control_status("owner/repo")

    assert snapshot["pending_next_issue"] == {"request_id": "req-next-2", "issue_number": 25}
    assert snapshot["commands"]["req-next-1"]["acknowledgement"] == "superseded"
    assert "req-next-2" in snapshot["commands"]["req-next-1"]["detail"]
    assert "req-next-1" in snapshot["commands"]["req-next-2"]["detail"]
    assert lifecycle.get_command("req-next-1").acknowledgement is CommandAcknowledgement.SUPERSEDED  # type: ignore[union-attr]


def test_real_tracker_claim_verified_assigns_and_clears_round_finished() -> None:
    transport = _RecordingTransport({1: issue(1, labels=frozenset({"ready-for-agent", "round-finished"}))})
    tracker = GitHubTracker(transport, "octo/example")

    claim = tracker.claim_verified(transport.issues[1])

    assert claim.issue.number == 1
    assert transport.assignments == [("issue-1", "viewer-id")]
    assert transport.removed_labels == [("issue-1", "round-finished")]
    assert tracker.fetch_issue(1) is not None


def test_real_tracker_reports_self_assignment() -> None:
    transport = _RecordingTransport(
        {
            1: issue(1, assignee_logins=("agent",)),
            2: issue(2, assignee_logins=("human",)),
            3: issue(3),
        }
    )
    tracker = GitHubTracker(transport, "octo/example")

    assert tracker.is_self_assigned(transport.issues[1]) is True
    assert tracker.is_self_assigned(transport.issues[2]) is False
    assert tracker.is_self_assigned(transport.issues[3]) is False


# ---------------------------------------------------------------------------
# socket + CLI
# ---------------------------------------------------------------------------


def test_next_issue_over_the_socket_and_status(tmp_path: Path) -> None:
    import json
    import socket as stdlib_socket

    from simple_coding_agent.control_server import ControlServer

    lifecycle, tracker, store = make_lifecycle(tmp_path, {24: issue(24), 25: issue(25)})
    socket_path = tmp_path / "state" / "agentctl.sock"
    server = ControlServer(socket_path, lifecycle)
    server.start()
    try:

        def send(payload: dict) -> dict:
            with stdlib_socket.socket(stdlib_socket.AF_UNIX, stdlib_socket.SOCK_STREAM) as client:
                client.settimeout(5)
                client.connect(str(socket_path))
                client.sendall((json.dumps(payload) + "\n").encode())
                data = b""
                while not data.endswith(b"\n"):
                    chunk = client.recv(65536)
                    if not chunk:
                        break
                    data += chunk
            return json.loads(data.decode())

        reply = send({"op": "next_issue", "request_id": "req-next", "issue": 24})
        assert reply["ok"] is True
        assert reply["kind"] == "next_issue"
        assert reply["acknowledgement"] == "accepted"

        status = send({"op": "status"})
        assert status["ok"] is True
        assert status["status"]["pending_next_issue"] == {
            "request_id": "req-next",
            "issue_number": 24,
        }

        lookup = send({"op": "command", "request_id": "req-next"})
        assert lookup["ok"] is True
        assert lookup["command"]["acknowledgement"] == "accepted"

        bad = send({"op": "next_issue", "request_id": "req-bad", "issue": 25})
        # 25 exists and is eligible here, so this succeeds; use an unknown one.
        assert bad["ok"] is True
        missing = send({"op": "next_issue", "request_id": "req-missing", "issue": 999})
        assert missing["ok"] is False
        # Rejection leaves the newer accepted priority intact.
        status2 = send({"op": "status"})
        assert status2["status"]["pending_next_issue"] == {
            "request_id": "req-bad",
            "issue_number": 25,
        }
    finally:
        server.stop()


def test_agentctl_next_issue_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import socket as stdlib_socket
    import json

    from simple_coding_agent.agentctl import main as agentctl_main
    from simple_coding_agent.control_server import ControlServer

    lifecycle, tracker, store = make_lifecycle(tmp_path, {24: issue(24)})
    socket_path = tmp_path / "state" / "agentctl.sock"
    server = ControlServer(socket_path, lifecycle)
    server.start()
    try:
        code = agentctl_main(["--socket", str(socket_path), "next", "issue", "24"])
        assert code == 0
        out = capsys.readouterr().out
        assert "accepted" in out
        assert "24" in out

        code = agentctl_main(
            ["--socket", str(socket_path), "status"],
        )
        assert code == 0
        status_out = capsys.readouterr().out
        assert "pending_next_issue" in status_out
        assert "#24" in status_out

        code = agentctl_main(["--socket", str(socket_path), "next", "issue", "0"])
        assert code == 1
        err = capsys.readouterr().err
        assert "rejected" in err
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


def claim_for(found: TrackerIssue) -> Claim:
    return Claim(issue=found, assignment=Assignment(issue_id=found.id, assignee_id="viewer-id"))


class FakeTracker:
    def __init__(
        self,
        issues: dict[int, TrackerIssue],
        *,
        fifo: list[int] | None = None,
    ) -> None:
        self.issues = dict(issues)
        self.fifo: list[Claim] = (
            [claim_for(self.issues[number]) for number in fifo]
            if fifo is not None
            else []
        )
        self.fetch_errors: dict[int, Exception] = {}
        self.assign_errors: dict[int, Exception] = {}
        self.claimed_verified: list[int] = []
        self.fifo_claims: list[int] = []
        self.removed_labels: list[tuple[str, str]] = []

    def fetch_issue(self, number: int) -> TrackerIssue | None:
        if number in self.fetch_errors:
            raise self.fetch_errors[number]
        return self.issues.get(number)

    def is_self_assigned(self, found: TrackerIssue) -> bool:
        return "agent" in found.assignee_logins

    def claim_verified(self, found: TrackerIssue) -> Claim:
        if found.number in self.assign_errors:
            raise self.assign_errors[found.number]
        current = self.issues.get(found.number)
        assert current is not None
        from simple_coding_agent.github_tracker import ROUND_FINISHED

        if ROUND_FINISHED in current.labels:
            self.removed_labels.append((current.id, ROUND_FINISHED))
        self.claimed_verified.append(found.number)
        return Claim(issue=current, assignment=Assignment(issue_id=current.id, assignee_id="viewer-id"))

    def claim_next(self) -> Claim | None:
        if not self.fifo:
            return None
        claim = self.fifo.pop(0)
        self.fifo_claims.append(claim.issue.number)
        return claim

    def recover_claim(self, issue_number: int) -> Claim | None:
        found = self.issues.get(issue_number)
        if found is None:
            return None
        return claim_for(found)

    def release_attempt(self, issue_number: int, label: str, assignee_id: str) -> None:
        return None

    def release_handoff(self, issue_number: int, assignee_id: str) -> None:
        return None


class FakeWorkspace:
    working_directory = Path("/repository")

    def is_clean(self) -> bool:
        return True

    def prepare_for_profile_read(self, *, base_branch: str = "main") -> None:
        return None

    def prepare_attempt(self, *, base_branch: str, issue_number: int):
        return type("Prepared", (), {"branch": f"agent/issue-{issue_number}"})()

    def cleanup(self, *, base_branch: str, prepared: object, retain_branch: bool) -> None:
        return None


class FakePublisher:
    def publish(self, request: object):
        return type("Published", (), {"outcome": AttemptOutcome.COMPLETE, "branch_url": "https://example.test/x"})()


def make_lifecycle(tmp_path: Path, issues: dict[int, TrackerIssue], *, fifo: list[int] | None = None):
    from simple_coding_agent.lifecycle import AgentLifecycle

    tracker = FakeTracker(issues, fifo=fifo if fifo is not None else [next(iter(issues))] if issues else [])
    # Default FIFO follows dict order unless overridden.
    if fifo is None and issues:
        tracker.fifo = [claim_for(issues[number]) for number in issues]
    store = ControlStore(tmp_path)
    lifecycle = AgentLifecycle(
        tracker=tracker,  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(tmp_path),
        workspace=FakeWorkspace(),  # type: ignore[arg-type]
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),  # type: ignore[arg-type]
        attempt_runner=lambda received_claim, profile, prepared: _evidence(),
        control_store=store,
        sleeper=lambda seconds: None,
    )
    return lifecycle, tracker, store


def _evidence():
    from simple_coding_agent.lifecycle import AttemptEvidence

    return AttemptEvidence(
        decision=CompletionDecision(AttemptOutcome.COMPLETE, True, PublicationPath.COMPLETE, ("ready",)),
        check_command="pytest",
        check_exit_code=0,
        review_cycles=1,
        review_findings="all clear",
        details="Implemented.",
    )


class _RecordingTransport:
    def __init__(self, issues: dict[int, TrackerIssue]) -> None:
        self.issues = dict(issues)
        self.assignments: list[tuple[str, str]] = []
        self.removed_labels: list[tuple[str, str]] = []

    def get_issue(self, repository: str, number: int) -> TrackerIssue | None:
        return self.issues.get(number)

    def viewer(self):  # type: ignore[no-untyped-def]
        from simple_coding_agent.github_tracker import GitHubIdentity

        return GitHubIdentity(id="viewer-id", login="agent")

    def assign_issue(self, issue_id: str, assignee_id: str) -> Assignment:
        self.assignments.append((issue_id, assignee_id))
        return Assignment(issue_id=issue_id, assignee_id=assignee_id)

    def remove_label(self, issue_id: str, label: str) -> None:
        self.removed_labels.append((issue_id, label))
