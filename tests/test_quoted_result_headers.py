"""Quoted result headers and markers never change control flow (issue #393).

A comment that merely *mentions* an attempt-result header or a hidden
machine marker — in prose, a list item, an inline code span, a blockquote,
or a fenced code block — is human discussion, not a generated result. Only
a body that *starts* with the header (after optional leading whitespace)
and carries the matching hidden marker counts as a generated result,
handoff record, or budget signal.
"""

from __future__ import annotations

from datetime import UTC, datetime

from simple_coding_agent.completion import (
    AttemptOutcome,
    CompletionDecision,
    PublicationPath,
)
from simple_coding_agent.github_tracker import TrackerIssue
from simple_coding_agent.handoff_record import (
    HANDOFF_RECORD_VERSION,
    HandoffRecord,
    build_handoff_record_body,
    handoff_record_marker,
    parse_handoff_record,
)
from simple_coding_agent.lifecycle import _continuation_expected
from simple_coding_agent.publication import (
    ConflictPublicationRequest,
    PublicationRequest,
    _conflict_result_comment,
    _result_comment,
)
from simple_coding_agent.recovery import (
    AmbiguousContinuationError,
    REQUEUE_SOURCE_AUTOMATIC,
    REQUEUE_SOURCE_HUMAN,
    attempt_marker,
    conflict_attempt_marker,
    count_automatic_continuations,
    count_conflict_failures,
    filter_prompt_comments,
    is_attempt_result_comment,
    is_conflict_attempt,
    is_conflict_escalation,
    is_conflict_failure,
    is_conflict_result_comment,
    is_generated_handoff_result,
    is_generated_result_comment,
    is_handoff_record,
    release_comment_body,
    resolve_automatic_continuation_count,
    requeue_source_marker,
)

ATTEMPT = "2026-11-10T09:00:00Z"
ATTEMPT_NEWER = "2026-11-10T10:00:00Z"
REPO = "octo/example"


def _issue(**overrides) -> TrackerIssue:
    return TrackerIssue(
        id="issue-393",
        number=393,
        title="Fix the parser",
        body="Fix the parser.",
        created_at=datetime(2026, 11, 10, tzinfo=UTC),
        state="OPEN",
        labels=frozenset({"bug", "ready-for-agent"}),
        assignee_logins=(),
        blocked_by=0,
        author_login="reporter",
    )


def _ordinary_request(**overrides) -> PublicationRequest:
    return PublicationRequest(
        393,
        "Fix the parser",
        "agent/issue-393",
        ATTEMPT,
        CompletionDecision(AttemptOutcome.HANDOFF, True, PublicationPath.PARTIAL, ("ready",)),
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
        393,
        "Fix the parser",
        "agent/issue-393",
        ATTEMPT,
        "main",
        AttemptOutcome.INCOMPLETE,
        "rebase_conflict",
        "",
        "The rebase conflicted.",
        False,
        None,
        **overrides,
    )


def _published(outcome: AttemptOutcome, **overrides) -> str:
    """A real published ordinary result: header first, marker, branch link."""
    return _result_comment(
        _ordinary_request(**overrides),
        outcome,
        "https://example.test/tree/agent/issue-393",
        None,
        "Implemented the boundary.",
        attempt_marker(ATTEMPT),
    )


def _quoting_variants(header: str) -> list[str]:
    return [
        f"While describing the code, note the header `{header}` in passing.",
        f"- the header (`{header}`) names the kind",
        f"Use the `{header}` header for results.",
        f"> {header}\n>\n> quoted from the docs",
        f"```\n{header}\n```",
        f"Some prose\n\n{header}\n\nquoted mid-body, not at the top",
        f"   \nThe manual shows:\n\n    {header}",
    ]


# ---------------------------------------------------------------------------
# _continuation_expected: quoting never continues; real results still do
# ---------------------------------------------------------------------------


def test_quoted_handoff_header_never_continues() -> None:
    issue = _issue()
    for quoting in _quoting_variants("## Agent Attempt Result: handoff"):
        assert _continuation_expected(issue, lambda unused, body=quoting: (body,)) is False


def test_quoted_incomplete_header_never_continues() -> None:
    issue = _issue()
    for quoting in _quoting_variants("## Agent Attempt Result: incomplete"):
        assert _continuation_expected(issue, lambda unused, body=quoting: (body,)) is False


def test_issue_393_regression_quoted_brief_does_not_continue() -> None:
    """The exact #393 shape: a never-attempted issue whose only comment quotes
    the handoff header inside an inline code span proceeds past the check."""
    issue = _issue()
    brief = (
        "`_result_comment` / `_post_result`: the header "
        "(`## Agent Attempt Result: handoff`), published with the branch link."
    )
    assert _continuation_expected(issue, lambda unused: (brief,)) is False


def test_header_at_top_without_marker_does_not_continue() -> None:
    issue = _issue()
    body = "## Agent Attempt Result: handoff\n\nHalf done.\n"
    assert _continuation_expected(issue, lambda unused: (body,)) is False


def test_real_published_handoff_still_continues() -> None:
    issue = _issue()
    assert _continuation_expected(issue, lambda unused: (_published(AttemptOutcome.HANDOFF),)) is True


def test_real_published_incomplete_with_branch_still_continues() -> None:
    issue = _issue()
    assert _continuation_expected(issue, lambda unused: (_published(AttemptOutcome.INCOMPLETE),)) is True


def test_incomplete_without_branch_still_excluded() -> None:
    issue = _issue()
    body = _result_comment(
        _ordinary_request(),
        AttemptOutcome.INCOMPLETE,
        None,
        None,
        "Setup failed.",
        attempt_marker(ATTEMPT),
    )
    assert "no branch created" in body
    assert _continuation_expected(issue, lambda unused: (body,)) is False


def test_continuation_missing_detail_still_excluded() -> None:
    issue = _issue()
    body = _published(AttemptOutcome.HANDOFF) + "\nwas not found on origin"
    assert _continuation_expected(issue, lambda unused: (body,)) is False


def test_round_finished_label_still_continues_without_comments() -> None:
    issue = TrackerIssue(
        id="issue-393",
        number=393,
        title="Fix the parser",
        body="Fix the parser.",
        created_at=datetime(2026, 11, 10, tzinfo=UTC),
        state="OPEN",
        labels=frozenset({"bug", "round-finished"}),
        assignee_logins=(),
        blocked_by=0,
        author_login="reporter",
    )
    assert _continuation_expected(issue, lambda unused: ()) is True


# ---------------------------------------------------------------------------
# prompt filtering: quoting passes through; generated results still handled
# ---------------------------------------------------------------------------


def test_prompt_filter_keeps_quoted_headers_verbatim() -> None:
    for quoting in _quoting_variants("## Agent Attempt Result: handoff"):
        assert filter_prompt_comments((quoting,)) == (quoting,)


def test_prompt_filter_keeps_quoted_marker_verbatim() -> None:
    body = (
        "See `<!-- agent-attempt: 2026-11-10T09:00:00Z -->` for the marker format, "
        "and `<!-- requeue-source: automatic -->` for the source."
    )
    assert filter_prompt_comments((body,)) == (body,)


def test_prompt_filter_still_drops_generated_handoff() -> None:
    assert filter_prompt_comments((_published(AttemptOutcome.HANDOFF),)) == ()


def test_prompt_filter_still_strips_other_generated_results() -> None:
    body = _published(AttemptOutcome.COMPLETE)
    (stripped,) = filter_prompt_comments((body,))
    assert stripped != body
    assert "Implemented the boundary." in stripped
    assert "agent-attempt" not in stripped


def test_legacy_markerless_header_still_filtered_but_not_a_continuation() -> None:
    body = "## Agent Attempt Result: handoff\n\nHalf done.\n"
    assert filter_prompt_comments((body,)) == ()
    assert _continuation_expected(_issue(), lambda unused: (body,)) is False


# ---------------------------------------------------------------------------
# shared handoff record: quoting is not a record; real damage still raises
# ---------------------------------------------------------------------------


def _record(marker_attempt: str = ATTEMPT) -> HandoffRecord:
    return HandoffRecord(
        version=HANDOFF_RECORD_VERSION,
        repository=REPO,
        issue_number=393,
        attempt_id=marker_attempt,
        requeue_source=REQUEUE_SOURCE_AUTOMATIC,
        continuation_count=2,
        branch="agent/issue-393",
        branch_sha="b" * 40,
        note_sha="c" * 40,
    )


def _shared() -> str:
    return build_handoff_record_body(_record())


def test_quoted_handoff_record_marker_is_not_a_record() -> None:
    marker = handoff_record_marker(_record())
    for quoting in (
        f"See `{marker}` for the format.",
        f"- marker: `{marker}`",
        f"```\n{marker}\n```",
        f"> {marker}",
        f"Notes\n\n{marker}\n\nquoted mid-body",
    ):
        assert is_handoff_record(quoting) is False
        assert parse_handoff_record(quoting) is None


def test_quoted_record_marker_neither_counts_nor_raises() -> None:
    marker = handoff_record_marker(_record())
    quoting = f"The shared comment carries `{marker}` inline."
    assert resolve_automatic_continuation_count((quoting, _shared())) == 2


def test_real_record_still_recognised_and_parsed() -> None:
    body = _shared()
    assert is_handoff_record(body) is True
    assert parse_handoff_record(body) is not None


def test_malformed_real_record_still_raises() -> None:
    malformed = (
        "## Handoff note\n\nCurrent note.\n\n"
        "<!-- handoff-record: v=999 repo=octo/example issue=393 "
        f"attempt={ATTEMPT} source=automatic count=1 "
        "branch=agent/issue-393 branch_sha=" + "b" * 40 + " note_sha=" + "c" * 40 + " -->"
    )
    assert is_handoff_record(malformed) is True
    try:
        resolve_automatic_continuation_count((malformed,))
    except AmbiguousContinuationError:
        pass
    else:
        raise AssertionError("malformed record must raise")


def test_two_real_records_still_raise() -> None:
    try:
        resolve_automatic_continuation_count((_shared(), _shared()))
    except AmbiguousContinuationError:
        pass
    else:
        raise AssertionError("duplicate records must raise")


# ---------------------------------------------------------------------------
# budget counting: quoted markers neither count nor hide the row
# ---------------------------------------------------------------------------


def _auto_result(attempt: str) -> str:
    return (
        "## Agent Attempt Result: handoff\n\nHalf done.\n\n"
        f"{attempt_marker(attempt)}\n"
        f"{requeue_source_marker(REQUEUE_SOURCE_AUTOMATIC)}"
    )


def test_quoted_automatic_marker_neither_counts_nor_skips() -> None:
    quoting = (
        "The `<!-- requeue-source: automatic -->` marker "
        "counts automatic requeues."
    )
    assert count_automatic_continuations((quoting,)) == 0
    # Not skipped as transparent: it stops the row like any ordinary comment.
    assert count_automatic_continuations((_auto_result(ATTEMPT_NEWER), quoting, _auto_result(ATTEMPT))) == 1
    # A quoted marker in the newest comment hides older automatics.
    assert count_automatic_continuations((quoting, _auto_result(ATTEMPT))) == 0


def test_quoted_conflict_marker_neither_counts_nor_skips() -> None:
    quoting = f"See `{conflict_attempt_marker('rebase_conflict')}` for the format."
    assert count_automatic_continuations((_auto_result(ATTEMPT_NEWER), quoting)) == 1
    failure = (
        "## Conflict Attempt Result: rebase_conflict\n\n"
        f"{attempt_marker(ATTEMPT)}\n"
        f"{conflict_attempt_marker('rebase_conflict')}"
    )
    assert count_conflict_failures((failure, quoting)) == 1
    assert count_conflict_failures((quoting, failure)) == 0


def test_real_histories_count_unchanged() -> None:
    assert (
        count_automatic_continuations((_auto_result(ATTEMPT_NEWER), _auto_result(ATTEMPT))) == 2
    )
    human = (
        "## Agent Attempt Result: handoff\n\nHalf done.\n\n"
        f"{attempt_marker(ATTEMPT)}\n"
        f"{requeue_source_marker(REQUEUE_SOURCE_HUMAN)}"
    )
    assert count_automatic_continuations((_auto_result(ATTEMPT_NEWER), human)) == 1


# ---------------------------------------------------------------------------
# every rendered body is recognised by its own predicate
# ---------------------------------------------------------------------------


def test_every_ordinary_outcome_recognised() -> None:
    for outcome in AttemptOutcome:
        body = _published(outcome)
        assert is_generated_result_comment(body) is True
        assert is_attempt_result_comment(body) is True
        assert is_attempt_result_comment(body, kinds=("handoff", "incomplete")) == (
            outcome in (AttemptOutcome.HANDOFF, AttemptOutcome.INCOMPLETE)
        )
        assert is_generated_handoff_result(body) is (outcome is AttemptOutcome.HANDOFF)


def test_conflict_result_recognised() -> None:
    body = _conflict_result_comment(
        _conflict_request(),
        AttemptOutcome.INCOMPLETE,
        "rebase_conflict",
        None,
        None,
        "The rebase conflicted.",
        attempt_marker(ATTEMPT),
        None,
        None,
    )
    assert is_generated_result_comment(body) is True
    assert is_conflict_result_comment(body) is True
    assert is_conflict_attempt(body) is True
    assert is_conflict_failure(body) is True
    assert is_conflict_escalation(body) is False
    # Conflict results stay transparent to the automatic-continuation budget.
    assert count_automatic_continuations((body, _auto_result(ATTEMPT))) == 1


def test_conflict_escalation_recognised() -> None:
    body = _conflict_result_comment(
        _conflict_request(escalate=True, conflict_budget=3, conflict_index=3),
        AttemptOutcome.INCOMPLETE,
        "rebase_conflict",
        None,
        None,
        "The rebase conflicted.",
        attempt_marker(ATTEMPT),
        None,
        None,
    )
    assert is_generated_result_comment(body) is True
    assert is_conflict_result_comment(body) is True
    assert is_conflict_escalation(body) is True
    assert is_conflict_failure(body) is False


def test_recovery_release_comment_recognised() -> None:
    body = release_comment_body(
        issue_number=393,
        attempt_id=ATTEMPT,
        actual_outcome="infrastructure_error",
        failure_reason="boom",
        saved_at="/tmp/saved",
        branch="agent/issue-393",
        requeue_source=REQUEUE_SOURCE_HUMAN,
    )
    assert is_generated_result_comment(body) is True
    assert is_attempt_result_comment(body) is True
    assert is_generated_handoff_result(body) is False


def test_shared_record_body_recognised() -> None:
    body = _shared()
    assert is_handoff_record(body) is True
    parsed = parse_handoff_record(body)
    assert parsed is not None
    assert parsed.issue_number == 393
    assert parsed.attempt_id == ATTEMPT
