"""One-off backfill: give every historical ``v*`` tag a GitHub Release.

The maintainer runs this once, locally, authenticated via ``gh``::

    python scripts/backfill_releases.py --dry-run
    python scripts/backfill_releases.py

Steps, in this order:

1. Label every merged PR with the labels returned by the shared rule in
   :mod:`release_labels` (imported, not copied). Labels are only ever added.
2. Strip a leading ``agent: `` from the title of every merged PR.
3. Create a Release for each ``v*`` tag without one, in ascending semver
   order, with generated notes and an explicit ``previous_tag_name`` (omitted
   only for the first tag), ``make_latest: false``, neither draft nor
   prerelease. Afterwards mark the highest-semver tag's Release as latest.

The run is idempotent: a second run plans nothing. ``--dry-run`` changes
nothing and prints the plan, category counts, and a generate-notes probe
which aborts when the ``Features`` / ``Fixes`` categories are missing.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running as ``python scripts/backfill_releases.py`` while still
# importing the sibling ``release_labels`` module the same way the workflow
# and the tests do.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from release_labels import labels_to_add  # noqa: E402

#: Only a leading occurrence of this prefix is stripped from PR titles.
AGENT_PREFIX = "agent: "

#: PR labels that exclude a change from the generated release notes entirely.
EXCLUDED_LABELS = frozenset({"documentation", "skip-changelog"})

_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


@dataclass(frozen=True)
class PullRequest:
    """A merged PR plus the labels of the issue it closes (if any)."""

    number: int
    title: str
    body: str | None
    head_branch: str
    labels: tuple[str, ...] = ()
    issue_labels: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ReleaseInfo:
    """A GitHub Release that already exists."""

    tag: str
    is_latest: bool = False


@dataclass(frozen=True)
class LabelOp:
    number: int
    add: tuple[str, ...]


@dataclass(frozen=True)
class RetitleOp:
    number: int
    old_title: str
    new_title: str


@dataclass(frozen=True)
class ReleaseOp:
    tag: str
    previous_tag: str | None


@dataclass(frozen=True)
class BackfillPlan:
    label_ops: tuple[LabelOp, ...] = ()
    retitle_ops: tuple[RetitleOp, ...] = ()
    release_ops: tuple[ReleaseOp, ...] = ()
    mark_latest: str | None = None


def semver_key(tag: str) -> tuple[int, int, int]:
    """Return a numeric sort key so ``v0.10.0`` sorts after ``v0.9.0``."""
    match = _TAG_RE.match(tag)
    if match is None:
        return (-1, -1, -1)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def sort_tags_semver(tags: list[str]) -> list[str]:
    """Return ``v*`` tags in ascending numeric order."""
    return sorted(tags, key=semver_key)


def strip_agent_prefix(title: str) -> str:
    """Strip one leading ``agent: `` prefix; leave any other title as is."""
    if title.startswith(AGENT_PREFIX):
        return title[len(AGENT_PREFIX) :]
    return title


def plan_backfill(
    tags: list[str],
    releases: list[ReleaseInfo],
    prs: list[PullRequest],
) -> BackfillPlan:
    """Plan label, retitle and Release work. Pure: no I/O, no ``gh`` calls."""
    label_ops: list[LabelOp] = []
    for pr in prs:
        wanted = labels_to_add(pr.body, pr.head_branch, pr.issue_labels)
        missing = tuple(label for label in wanted if label not in pr.labels)
        if missing:
            label_ops.append(LabelOp(pr.number, missing))

    retitle_ops: list[RetitleOp] = []
    for pr in prs:
        new_title = strip_agent_prefix(pr.title)
        if new_title != pr.title:
            retitle_ops.append(RetitleOp(pr.number, pr.title, new_title))

    ordered = sort_tags_semver(tags)
    existing = {release.tag for release in releases}
    release_ops: list[ReleaseOp] = []
    for index, tag in enumerate(ordered):
        if tag in existing:
            continue
        previous = ordered[index - 1] if index > 0 else None
        release_ops.append(ReleaseOp(tag, previous))

    mark_latest: str | None = None
    if ordered:
        highest = ordered[-1]
        current = next((r.tag for r in releases if r.is_latest), None)
        if current != highest:
            mark_latest = highest

    return BackfillPlan(
        label_ops=tuple(label_ops),
        retitle_ops=tuple(retitle_ops),
        release_ops=tuple(release_ops),
        mark_latest=mark_latest,
    )


def plan_is_empty(plan: BackfillPlan) -> bool:
    """Return True when a re-plan against applied state would do nothing."""
    return (
        not plan.label_ops
        and not plan.retitle_ops
        and not plan.release_ops
        and plan.mark_latest is None
    )


def categorize(labels: list[str] | tuple[str, ...]) -> str:
    """Classify a PR's final labels like ``.github/release.yml`` does."""
    names = set(labels)
    if names & EXCLUDED_LABELS:
        return "hidden"
    if "enhancement" in names:
        return "Features"
    if "bug" in names:
        return "Fixes"
    return "Other changes"


def count_categories(
    prs: list[PullRequest], plan: BackfillPlan
) -> dict[str, int]:
    """Count PRs per notes category after the planned label additions."""
    added = {op.number: op.add for op in plan.label_ops}
    counts = {"Features": 0, "Fixes": 0, "Other changes": 0, "hidden": 0}
    for pr in prs:
        final = sorted(set(pr.labels) | set(added.get(pr.number, ())))
        counts[categorize(final)] += 1
    return counts


def probe_target(tags: list[str]) -> tuple[str, str | None]:
    """Pick one old tag (and its predecessor) for the generate-notes probe."""
    ordered = sort_tags_semver(tags)
    if len(ordered) >= 2:
        return (ordered[1], ordered[0])
    return (ordered[0], None)


def probe_ok(notes_body: str) -> bool:
    """Return True when generated notes carry the Features/Fixes categories."""
    return "Features" in notes_body and "Fixes" in notes_body


class GitHub:
    """Thin ``gh`` seam: fetches state and applies a plan. Only I/O lives here."""

    def __init__(self, repo: str | None = None) -> None:
        self.repo = repo

    def _gh(self, *args: str, input_text: str | None = None) -> str:
        cmd = ["gh", *args]
        if self.repo:
            cmd += ["-R", self.repo]
        result = subprocess.run(
            cmd, capture_output=True, text=True, input=input_text, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    def _api(self, *args: str) -> str:
        return self._gh("api", *args)

    def list_tags(self) -> list[str]:
        out = self._api("repos/{owner}/{repo}/tags", "--paginate", "--jq", ".[].name")
        # ``gh api`` expands {owner}/{repo} to the current repository.
        tags = json.loads(out) if out.startswith("[") else out.splitlines()
        return [
            t for t in tags if isinstance(t, str) and _TAG_RE.match(t) is not None
        ]

    def list_releases(self) -> list[ReleaseInfo]:
        out = self._gh(
            "release", "list", "--limit", "100", "--json", "tagName,isLatest"
        )
        return [
            ReleaseInfo(tag=item["tagName"], is_latest=item.get("isLatest", False))
            for item in json.loads(out or "[]")
        ]

    def _issue_labels(self, repo: str, number: int) -> tuple[str, ...]:
        try:
            out = self._api(
                f"repos/{repo}/issues/{number}", "--jq", "[.labels[].name]"
            )
        except RuntimeError:
            return ()
        try:
            names = json.loads(out or "[]")
        except json.JSONDecodeError:
            return ()
        return tuple(n for n in names if isinstance(n, str))

    def list_merged_prs(self) -> list[PullRequest]:
        from release_labels import linked_issue_number

        out = self._gh(
            "pr",
            "list",
            "--state",
            "merged",
            "--limit",
            "1000",
            "--json",
            "number,title,body,headRefName,labels",
        )
        repo = self.repo or self._gh("repo", "view", "--json", "nameWithOwner",
                                     "--jq", ".nameWithOwner")
        prs: list[PullRequest] = []
        for item in json.loads(out or "[]"):
            number = item["number"]
            linked = linked_issue_number(item.get("body") or "")
            issue_labels: tuple[str, ...] | None = None
            if linked is not None:
                issue_labels = self._issue_labels(repo, linked)
            prs.append(
                PullRequest(
                    number=number,
                    title=item.get("title") or "",
                    body=item.get("body"),
                    head_branch=item.get("headRefName") or "",
                    labels=tuple(l["name"] for l in item.get("labels") or []),
                    issue_labels=issue_labels,
                )
            )
        return prs

    def generate_notes(self, tag: str, previous: str | None) -> str:
        args = [
            "repos/{owner}/{repo}/releases/generate-notes",
            "-X",
            "POST",
            "-f",
            f"tag_name={tag}",
            "--jq",
            ".body",
        ]
        if previous is not None:
            args += ["-f", f"previous_tag_name={previous}"]
        return self._api(*args)

    def add_labels(self, number: int, labels: tuple[str, ...]) -> None:
        self._gh("pr", "edit", str(number), "--add-label", ",".join(labels))

    def retitle(self, number: int, new_title: str) -> None:
        self._gh("pr", "edit", str(number), "--title", new_title)

    def create_release(self, tag: str, previous: str | None) -> None:
        args = [
            "release",
            "create",
            tag,
            "--title",
            tag,
            "--generate-notes",
            "--latest=false",
        ]
        if previous is not None:
            args += ["--notes-start-tag", previous]
        self._gh(*args)

    def mark_latest(self, tag: str) -> None:
        self._gh("release", "edit", tag, "--latest")


def print_plan(plan: BackfillPlan, counts: dict[str, int]) -> None:
    print("Label changes:")
    if plan.label_ops:
        for op in plan.label_ops:
            print(f"  #{op.number}: +{', +'.join(op.add)}")
    else:
        print("  (none)")
    print("Title changes:")
    if plan.retitle_ops:
        for op in plan.retitle_ops:
            print(f"  #{op.number}: {op.old_title!r} -> {op.new_title!r}")
    else:
        print("  (none)")
    print("Releases to create:")
    if plan.release_ops:
        for op in plan.release_ops:
            previous = op.previous_tag if op.previous_tag is not None else "none"
            print(f"  {op.tag} (previous_tag_name: {previous})")
    else:
        print("  (none)")
    print(f"Mark latest: {plan.mark_latest if plan.mark_latest else '(none)'}")
    print(
        "Category counts: "
        + ", ".join(f"{name} {counts[name]}" for name in
                    ("Features", "Fixes", "Other changes", "hidden"))
    )


def fetch_probe(gh: GitHub, tags: list[str]) -> tuple[str, str | None, str]:
    """Generate notes for one old tag; return the tag, previous tag and body."""
    tag, previous = probe_target(tags)
    return (tag, previous, gh.generate_notes(tag, previous))


def report_probe(tag: str, previous: str | None, body: str) -> bool:
    """Print the probe outcome; return False unless categories apply."""
    where = f"{tag} (previous_tag_name: {previous if previous else 'none'})"
    if not probe_ok(body):
        print(f"Probe FAILED for {where}: no Features/Fixes categories.", file=sys.stderr)
        print("Is .github/release.yml on main? Aborting.", file=sys.stderr)
        return False
    print(f"Probe OK for {where}: notes carry Features/Fixes categories.")
    return True


def apply_plan(gh: GitHub, plan: BackfillPlan) -> None:
    for op in plan.label_ops:
        print(f"Adding {', '.join(op.add)} to #{op.number}.")
        gh.add_labels(op.number, op.add)
    for op in plan.retitle_ops:
        print(f"Retitling #{op.number} to {op.new_title!r}.")
        gh.retitle(op.number, op.new_title)
    for op in plan.release_ops:
        print(f"Creating release {op.tag}.")
        gh.create_release(op.tag, op.previous_tag)
    if plan.mark_latest is not None:
        print(f"Marking {plan.mark_latest} as latest.")
        gh.mark_latest(plan.mark_latest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill Releases for v* tags.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan only.")
    parser.add_argument("--repo", default=None, help="OWNER/REPO (default: current).")
    args = parser.parse_args(argv)

    gh = GitHub(repo=args.repo)
    tags = gh.list_tags()
    releases = gh.list_releases()
    prs = gh.list_merged_prs()
    plan = plan_backfill(tags, releases, prs)
    counts = count_categories(prs, plan)

    if args.dry_run:
        print_plan(plan, counts)
        if tags:
            tag, previous, body = fetch_probe(gh, tags)
            if not report_probe(tag, previous, body):
                raise SystemExit(1)
        else:
            print("Probe skipped: no v* tags.")
        return 0

    if tags:
        # Prove the categories apply before mutating anything.
        tag, previous, body = fetch_probe(gh, tags)
        if not report_probe(tag, previous, body):
            return 1
    if plan_is_empty(plan):
        print("Nothing to do.")
        return 0
    apply_plan(gh, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
