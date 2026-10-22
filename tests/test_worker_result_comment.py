"""Worker attribution on attempt-result comments (issue #301).

With ``WORKER_ID`` set, every attempt-result comment (ordinary, conflict,
and recovery-release) shows the worker id and carries the hidden
``agent-worker`` marker. Without it the comment is byte-for-byte the legacy
body, and existing marker parsing and continuation counting ignore the new
marker.
"""

from dataclasses import replace

from simple_coding_agent.completion import (
    AttemptOutcome,
    CompletionDecision,
    PublicationPath,
)
from simple_coding_agent.publication import (
    ConflictPublicationRequest,
    PublicationRequest,
    _conflict_result_comment,
    _result_comment,
)
from simple_coding_agent.recovery import (
    REQUEUE_SOURCE_AUTOMATIC,
    REQUEUE_SOURCE_HUMAN,
    attempt_marker,
    count_automatic_continuations,
    is_generated_result_comment,
    parse_agent_worker,
    parse_comment_attempt_id,
    parse_requeue_source,
    release_comment_body,
    requeue_source_marker,
    strip_result_bookkeeping,
    worker_blocks,
    worker_line,
    worker_marker,
)

WORKER = "worker-1"
ATTEMPT = "2026-10-22T10:00:00Z"


def _ordinary_request(**overrides) -> PublicationRequest:
    return PublicationRequest(
        23,
        "Implement publication",
        "agent/issue-23",
        ATTEMPT,
        CompletionDecision(AttemptOutcome.COMPLETE, True, PublicationPath.COMPLETE, ("ready",)),
        "pytest",
        0,
        0,
        "all clear",
        "Implemented the boundary.",
        "main",
        **overrides,
    )


def _conflict_request(**overrides) -> ConflictPublicationRequest:
    return ConflictPublicationRequest(
        7,
        "Issue 7",
        "agent/issue-7",
        ATTEMPT,
        "main",
        AttemptOutcome.INCOMPLETE,
        "check_failed",
        "",
        "The profile check did not pass.",
        False,
        None,
        **overrides,
    )


def _ordinary_comment(request, outcome=AttemptOutcome.COMPLETE) -> str:
    return _result_comment(
        request,
        outcome,
        "https://example.test/tree/agent/issue-23",
        None,
        "Implemented the boundary.",
        attempt_marker(ATTEMPT),
    )


def _conflict_comment(request) -> str:
    return _conflict_result_comment(
        request,
        AttemptOutcome.INCOMPLETE,
        "check_failed",
        None,
        None,
        "The profile check did not pass.",
        attempt_marker(ATTEMPT),
        None,
        None,
    )


def test_worker_helpers_round_trip() -> None:
    assert worker_marker(WORKER) == "<!-- agent-worker: worker-1 -->"
    assert worker_line(WORKER) == "Worker: worker-1"
    assert parse_agent_worker(worker_marker(WORKER)) == WORKER
    assert parse_agent_worker("no markers here") is None
    assert parse_agent_worker("") is None
    assert worker_blocks("") == ("", "")
    assert worker_blocks("   ") == ("", "")
    line_block, marker_block = worker_blocks(WORKER)
    assert line_block == "Worker: worker-1\n"
    assert marker_block == "<!-- agent-worker: worker-1 -->\n"


def test_ordinary_comment_without_worker_has_no_attribution() -> None:
    body = _ordinary_comment(_ordinary_request())

    assert "Worker:" not in body
    assert "agent-worker" not in body
    assert attempt_marker(ATTEMPT) in body
    assert requeue_source_marker(REQUEUE_SOURCE_HUMAN) in body


def test_ordinary_comment_with_worker_carries_line_and_marker() -> None:
    plain = _ordinary_comment(_ordinary_request())
    body = _ordinary_comment(replace(_ordinary_request(), worker_id=WORKER))

    assert "\nWorker: worker-1\n" in body
    assert "<!-- agent-worker: worker-1 -->" in body
    assert parse_agent_worker(body) == WORKER
    # Removing the attribution restores the legacy body byte-for-byte.
    assert (
        body.replace("Worker: worker-1\n", "").replace(
            "<!-- agent-worker: worker-1 -->\n", ""
        )
        == plain
    )


def test_ordinary_comment_worker_on_all_outcomes() -> None:
    for outcome in (
        AttemptOutcome.COMPLETE,
        AttemptOutcome.INCOMPLETE,
        AttemptOutcome.HANDOFF,
        AttemptOutcome.INFRASTRUCTURE_ERROR,
    ):
        body = _ordinary_comment(replace(_ordinary_request(), worker_id=WORKER), outcome)
        assert f"## Agent Attempt Result: {outcome.value}" in body
        assert "Worker: worker-1" in body
        assert "<!-- agent-worker: worker-1 -->" in body


def test_conflict_comment_with_worker_carries_line_and_marker() -> None:
    plain = _conflict_comment(_conflict_request())
    assert "agent-worker" not in plain

    body = _conflict_comment(replace(_conflict_request(), worker_id=WORKER))

    assert "Worker: worker-1" in body
    assert "<!-- agent-worker: worker-1 -->" in body
    assert (
        body.replace("Worker: worker-1\n", "").replace(
            "<!-- agent-worker: worker-1 -->\n", ""
        )
        == plain
    )


def test_release_comment_body_worker() -> None:
    plain = release_comment_body(
        issue_number=23,
        attempt_id=ATTEMPT,
        actual_outcome="incomplete",
        failure_reason="interrupted",
        saved_at="/tmp/work",
        branch="agent/issue-23",
    )
    assert "agent-worker" not in plain

    body = release_comment_body(
        issue_number=23,
        attempt_id=ATTEMPT,
        actual_outcome="incomplete",
        failure_reason="interrupted",
        saved_at="/tmp/work",
        branch="agent/issue-23",
        worker_id=WORKER,
    )

    assert "Worker: worker-1" in body
    assert "<!-- agent-worker: worker-1 -->" in body
    assert (
        body.replace("Worker: worker-1\n", "").replace(
            "<!-- agent-worker: worker-1 -->\n", ""
        )
        == plain
    )


def test_existing_parsers_ignore_worker_marker() -> None:
    body = _ordinary_comment(replace(_ordinary_request(), worker_id=WORKER))
    assert parse_comment_attempt_id(body) == ATTEMPT
    assert parse_requeue_source(body) == REQUEUE_SOURCE_HUMAN
    assert is_generated_result_comment(body) is True
    # The worker marker alone never identifies a generated result.
    assert is_generated_result_comment(worker_marker(WORKER)) is False
    # Continuation counting only reads requeue-source markers.
    bodies = [
        body,  # human requeue with worker attribution stops the row
        _ordinary_comment(
            replace(
                _ordinary_request(),
                requeue_source=REQUEUE_SOURCE_AUTOMATIC,
                worker_id=WORKER,
            )
        ),
    ]
    assert count_automatic_continuations(bodies) == 0
    assert count_automatic_continuations(bodies[1:]) == 1


def test_bookkeeping_strip_removes_worker_attribution() -> None:
    body = _ordinary_comment(replace(_ordinary_request(), worker_id=WORKER))
    stripped = strip_result_bookkeeping(body)

    assert "Worker:" not in stripped
    assert "agent-worker" not in stripped
    assert "agent-attempt" not in stripped
