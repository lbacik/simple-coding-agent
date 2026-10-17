"""Unit tests for the pure planning in ``scripts/backfill_releases.py``.

Everything here runs against fakes (plain tags, releases and PRs); no test
touches the network or shells out to ``gh``.
"""

from __future__ import annotations

import release_labels
from backfill_releases import (
    BackfillPlan,
    PullRequest,
    ReleaseInfo,
    categorize,
    count_categories,
    labels_to_add,
    plan_backfill,
    plan_is_empty,
    final_labels,
    notes_have_categories,
    render_notes,
    sort_tags_semver,
    strip_agent_prefix,
)


def make_pr(number: int, **kwargs: object) -> PullRequest:
    defaults: dict[str, object] = {
        "title": "Some change",
        "body": None,
        "head_branch": "agent/issue-1",
        "labels": (),
        "issue_labels": None,
    }
    defaults.update(kwargs)
    return PullRequest(number=number, **defaults)  # type: ignore[arg-type]


def applied_state(plan: BackfillPlan, prs: list[PullRequest]) -> list[PullRequest]:
    """Return the PR list as it looks after ``plan`` has been applied."""
    added = {op.number: op.add for op in plan.label_ops}
    retitled = {op.number: op.new_title for op in plan.retitle_ops}
    return [
        PullRequest(
            number=pr.number,
            title=retitled.get(pr.number, pr.title),
            body=pr.body,
            head_branch=pr.head_branch,
            labels=tuple(sorted(set(pr.labels) | set(added.get(pr.number, ())))),
            issue_labels=pr.issue_labels,
        )
        for pr in prs
    ]


def test_imports_shared_label_rule_instead_of_copying_it() -> None:
    assert labels_to_add is release_labels.labels_to_add


def test_sorts_tags_in_numeric_semver_order() -> None:
    assert sort_tags_semver(["v0.10.0", "v0.9.0", "v0.2.0", "v0.1.0"]) == [
        "v0.1.0",
        "v0.2.0",
        "v0.9.0",
        "v0.10.0",
    ]


def test_previous_tag_chain_omitted_only_for_first_tag() -> None:
    plan = plan_backfill(["v0.3.0", "v0.1.0", "v0.2.0"], [], [])
    assert [(op.tag, op.previous_tag) for op in plan.release_ops] == [
        ("v0.1.0", None),
        ("v0.2.0", "v0.1.0"),
        ("v0.3.0", "v0.2.0"),
    ]


def test_previous_tag_points_at_preceding_tag_when_middle_release_exists() -> None:
    releases = [ReleaseInfo("v0.2.0")]
    plan = plan_backfill(["v0.1.0", "v0.2.0", "v0.3.0"], releases, [])
    assert [(op.tag, op.previous_tag) for op in plan.release_ops] == [
        ("v0.1.0", None),
        ("v0.3.0", "v0.2.0"),
    ]


def test_skips_tags_that_already_have_a_release() -> None:
    releases = [ReleaseInfo("v0.1.0"), ReleaseInfo("v0.2.0", is_latest=True)]
    plan = plan_backfill(["v0.1.0", "v0.2.0", "v0.3.0"], releases, [])
    assert [op.tag for op in plan.release_ops] == ["v0.3.0"]
    assert plan.release_ops[0].previous_tag == "v0.2.0"


def test_strips_only_a_leading_agent_prefix() -> None:
    assert strip_agent_prefix("agent: Do the thing") == "Do the thing"
    assert strip_agent_prefix("Do the agent: thing") == "Do the agent: thing"
    assert strip_agent_prefix("Agent: Do the thing") == "Agent: Do the thing"
    assert strip_agent_prefix("agent:agent: nested") == "agent:agent: nested"
    assert strip_agent_prefix("agent: agent: nested") == "agent: nested"
    assert strip_agent_prefix("Do the thing") == "Do the thing"


def test_retitles_every_merged_pr_with_the_prefix() -> None:
    prs = [
        make_pr(1, title="agent: First"),
        make_pr(2, title="Second agent: kept"),
        make_pr(3, title="agent: Third"),
    ]
    plan = plan_backfill([], [], prs)
    assert [(op.number, op.new_title) for op in plan.retitle_ops] == [
        (1, "First"),
        (3, "Third"),
    ]


def test_labels_come_from_the_shared_rule_and_are_only_added() -> None:
    prs = [
        make_pr(
            1,
            body="Closes #10",
            head_branch="agent/issue-10",
            labels=(),
            issue_labels=("bug", "needs-triage"),
        ),
        make_pr(
            2,
            body="Closes #11",
            head_branch="agent/issue-11",
            labels=("enhancement",),
            issue_labels=("enhancement",),
        ),
        make_pr(3, body="No linked issue.", head_branch="docs/update"),
    ]
    plan = plan_backfill([], [], prs)
    assert [(op.number, op.add) for op in plan.label_ops] == [
        (1, ("bug",)),
        (3, ("documentation",)),
    ]


def test_skips_prs_that_already_carry_the_labels_and_clean_title() -> None:
    prs = [
        make_pr(
            1,
            title="Fixed already",
            body="Closes #10",
            labels=("bug",),
            issue_labels=("bug",),
        )
    ]
    plan = plan_backfill(["v0.1.0"], [ReleaseInfo("v0.1.0", is_latest=True)], prs)
    assert plan_is_empty(plan)


def test_latest_is_highest_semver_tag() -> None:
    plan = plan_backfill(["v0.9.0", "v0.10.0", "v0.1.0"], [], [])
    assert plan.mark_latest == "v0.10.0"


def test_latest_covers_tags_whose_release_already_exists() -> None:
    releases = [ReleaseInfo("v0.5.0")]
    plan = plan_backfill(["v0.4.0", "v0.5.0"], releases, [])
    assert plan.mark_latest == "v0.5.0"
    assert [op.tag for op in plan.release_ops] == ["v0.4.0"]


def test_no_latest_mark_when_highest_is_already_latest() -> None:
    releases = [ReleaseInfo("v0.4.0"), ReleaseInfo("v0.5.0", is_latest=True)]
    plan = plan_backfill(["v0.4.0", "v0.5.0"], releases, [])
    assert plan.mark_latest is None
    assert plan_is_empty(plan)


def test_replanning_against_applied_state_yields_an_empty_plan() -> None:
    tags = ["v0.1.0", "v0.2.0"]
    prs = [
        make_pr(
            1,
            title="agent: First",
            body="Closes #10",
            labels=(),
            issue_labels=("bug",),
        ),
        make_pr(2, title="agent: Second", body="Nothing linked."),
    ]
    first = plan_backfill(tags, [], prs)
    assert not plan_is_empty(first)

    applied_releases = [
        ReleaseInfo(op.tag, is_latest=(op.tag == first.mark_latest))
        for op in first.release_ops
    ]
    second = plan_backfill(tags, applied_releases, applied_state(first, prs))
    assert plan_is_empty(second)


def test_categorize_matches_release_yml() -> None:
    assert categorize(("enhancement",)) == "Features"
    assert categorize(("bug",)) == "Fixes"
    assert categorize(()) == "Other changes"
    assert categorize(("documentation",)) == "hidden"
    assert categorize(("skip-changelog",)) == "hidden"
    assert categorize(("bug", "documentation")) == "hidden"
    assert categorize(("enhancement", "bug")) == "Features"


def test_count_categories_uses_final_labels() -> None:
    prs = [
        make_pr(1, body="Closes #1", labels=(), issue_labels=("enhancement",)),
        make_pr(2, labels=("bug",)),
        make_pr(3, labels=()),
        make_pr(4, labels=("documentation",)),
    ]
    plan = plan_backfill([], [], prs)
    assert count_categories(prs, plan) == {
        "Features": 1,
        "Fixes": 1,
        "Other changes": 1,
        "hidden": 1,
    }


def test_empty_plan_for_no_tags_no_releases_no_prs() -> None:
    assert plan_is_empty(plan_backfill([], [], []))


def _flat(*prs: tuple[int, str]) -> str:
    lines = [f"* {t} by @u in https://github.com/o/r/pull/{n}" for n, t in prs]
    return (
        "## What's Changed\n"
        + "\n".join(lines)
        + "\n\n\n**Full Changelog**: https://github.com/o/r/compare/v0.1.0...v0.2.0"
    )


def test_render_notes_groups_prs_by_final_label() -> None:
    body = _flat((1, "Add a thing"), (2, "Fix a thing"), (3, "Tidy"), (4, "Docs"))
    labels = {1: ("enhancement",), 2: ("bug",), 3: (), 4: ("documentation",)}
    assert render_notes(body, labels) == (
        "## What's Changed\n"
        "### Features\n"
        "* Add a thing by @u in https://github.com/o/r/pull/1\n"
        "\n"
        "### Fixes\n"
        "* Fix a thing by @u in https://github.com/o/r/pull/2\n"
        "\n"
        "### Other changes\n"
        "* Tidy by @u in https://github.com/o/r/pull/3\n"
        "\n"
        "**Full Changelog**: https://github.com/o/r/compare/v0.1.0...v0.2.0\n"
    )


def test_render_notes_strips_the_agent_prefix_and_omits_empty_sections() -> None:
    out = render_notes(_flat((1, "agent: Fix it")), {1: ("bug",)})
    assert "* Fix it by @u" in out
    assert "agent:" not in out
    assert "### Features" not in out
    assert "### Other changes" not in out


def test_render_notes_leaves_a_body_without_prs_alone() -> None:
    assert render_notes("## What's Changed\n", {}) == "## What's Changed\n"


def test_final_labels_include_the_planned_additions() -> None:
    prs = [make_pr(1, labels=("x",)), make_pr(2)]
    plan = plan_backfill([], [], prs)
    assert final_labels(prs, plan)[1] == ("x",)


def test_notes_have_categories_needs_features_and_fixes_somewhere() -> None:
    assert notes_have_categories(["### Features\n", "### Fixes\n"]) is True
    assert notes_have_categories(["### Fixes\n"]) is False
    assert notes_have_categories(["## What's Changed\n"]) is False
