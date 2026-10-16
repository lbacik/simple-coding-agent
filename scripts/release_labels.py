"""Release-note labels for pull requests.

Generated release notes categorise changes by **PR** labels, while this repo
triages work with **issue** labels. This module holds the single rule that
derives which labels to add to a PR: it copies the categorising labels from
the issue the PR closes, plus a branch-name rule for documentation PRs.

The rule lives here, outside ``simple_coding_agent``, because it is a
repository convention, not product behaviour. Both the ``release-labels``
workflow and the release backfill script import this same function so the
rule exists exactly once.

The function only ever returns labels to *add*; it never asks to remove any.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

#: Issue labels copied onto the PR that closes the issue. These feed the
#: generated release-note categories; every other issue label is ignored.
COPIED_LABELS: frozenset[str] = frozenset({"bug", "enhancement", "documentation"})

#: PR head branches with this prefix are documentation changes.
DOCS_BRANCH_PREFIX = "docs/"

#: Only the ``Closes #N`` form (the form the agent writes) links a PR to its
#: issue. Other closing keywords (``Fixes``, ``Resolves``, ...) do not link,
#: and when several ``Closes`` references appear the first one wins.
_CLOSES_RE = re.compile(r"\bcloses\s*:?\s+#(\d+)", re.IGNORECASE)


def linked_issue_number(pr_body: str | None) -> int | None:
    """Return the issue number from the first ``Closes #N`` reference.

    Return ``None`` when the PR body names no linked issue.
    """
    if not pr_body:
        return None
    match = _CLOSES_RE.search(pr_body)
    return int(match.group(1)) if match else None


def labels_to_add(
    pr_body: str | None,
    head_branch: str | None,
    issue_labels: Iterable[str] | None,
) -> list[str]:
    """Return the sorted labels to add to a PR, never removals.

    ``issue_labels`` are the labels of the issue named by ``Closes #N`` in
    ``pr_body``; only ``bug``, ``enhancement`` and ``documentation`` are
    copied. PRs whose head branch starts with ``docs/`` additionally gain
    ``documentation``. With no ``Closes #N`` reference only the branch rule
    can add anything.
    """
    to_add: set[str] = set()
    if linked_issue_number(pr_body) is not None:
        to_add.update(label for label in (issue_labels or ()) if label in COPIED_LABELS)
    if (head_branch or "").startswith(DOCS_BRANCH_PREFIX):
        to_add.add("documentation")
    return sorted(to_add)
