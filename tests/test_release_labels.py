"""Unit tests for the PR release-label rule in ``scripts/release_labels.py``."""

from __future__ import annotations

import pytest

from release_labels import COPIED_LABELS, labels_to_add, linked_issue_number

BODY = "Closes #123\n\n## Summary\nDone."


@pytest.mark.parametrize("label", ["bug", "enhancement", "documentation"])
def test_copies_categorising_label_from_linked_issue(label: str) -> None:
    assert labels_to_add(BODY, "agent/issue-123", [label]) == [label]


def test_copies_several_labels_sorted() -> None:
    assert labels_to_add(BODY, "agent/issue-123", ["documentation", "bug"]) == [
        "bug",
        "documentation",
    ]


def test_ignores_non_categorising_issue_labels() -> None:
    assert (
        labels_to_add(
            BODY,
            "agent/issue-123",
            ["needs-triage", "ready-for-agent", "wontfix", "agent-auto-merge"],
        )
        == []
    )


def test_copies_only_categorising_labels_from_mixed_issue_labels() -> None:
    assert labels_to_add(BODY, "agent/issue-123", ["bug", "needs-triage"]) == ["bug"]


def test_deduplicates_repeated_issue_labels() -> None:
    assert labels_to_add(BODY, "agent/issue-123", ["bug", "bug"]) == ["bug"]


def test_docs_branch_adds_documentation_without_linked_issue() -> None:
    assert labels_to_add("No linked issue here.", "docs/foo", []) == ["documentation"]


def test_docs_branch_adds_documentation_alongside_copied_labels() -> None:
    assert labels_to_add(BODY, "docs/foo", ["bug"]) == ["bug", "documentation"]


def test_non_docs_branch_without_closes_adds_nothing() -> None:
    assert (
        labels_to_add(
            "No linked issue here.",
            "agent/issue-123",
            ["bug", "enhancement", "documentation"],
        )
        == []
    )
    assert labels_to_add("", "agent/issue-123", ["bug"]) == []
    assert labels_to_add(None, "agent/issue-123", ["bug"]) == []


def test_result_only_ever_adds_categorising_labels() -> None:
    """The rule returns labels to add; it never asks to remove any label."""
    cases = [
        (BODY, "agent/issue-123", ["bug", "needs-triage"]),
        (BODY, "docs/foo", ["enhancement", "wontfix"]),
        ("Fixes #123", "agent/issue-123", ["bug"]),
        ("No linked issue.", "agent/issue-123", ["bug"]),
        ("No linked issue.", "docs/foo", ["ready-for-human"]),
    ]
    for pr_body, branch, issue_labels in cases:
        result = labels_to_add(pr_body, branch, issue_labels)
        assert isinstance(result, list)
        assert set(result) <= set(COPIED_LABELS)


def test_does_not_mutate_issue_labels() -> None:
    issue_labels = ["bug", "needs-triage"]
    labels_to_add(BODY, "agent/issue-123", issue_labels)
    assert issue_labels == ["bug", "needs-triage"]


def test_linked_issue_number_reads_closes_reference() -> None:
    assert linked_issue_number(BODY) == 123


def test_linked_issue_number_is_case_insensitive() -> None:
    assert linked_issue_number("closes #42") == 42
    assert linked_issue_number("CLOSES #7") == 7


def test_linked_issue_number_accepts_colon_form() -> None:
    assert linked_issue_number("Closes: #9") == 9


@pytest.mark.parametrize(
    "body",
    [
        "Fixes #123",
        "Resolves #123",
        "Fixed #123",
        "Closed #123",
        "Closes lbacik/repo#123",
        "No linked issue here.",
        "",
    ],
)
def test_other_keywords_and_missing_reference_link_nothing(body: str | None) -> None:
    assert linked_issue_number(body) is None
    assert labels_to_add(body, "agent/issue-123", ["bug"]) == []


def test_first_closes_reference_wins() -> None:
    assert linked_issue_number("Closes #1\n\nCloses #2") == 1


def test_linked_issue_number_handles_empty_body() -> None:
    assert linked_issue_number("") is None
    assert linked_issue_number(None) is None
