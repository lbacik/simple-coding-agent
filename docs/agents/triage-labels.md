# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role, use the corresponding label string from this table.

## `round-finished` continuation

`round-finished` is not a triage role: the agent applies it when a handoff
releases the issue (removing `ready-for-agent` and the assignment). With the
default `MAX_AUTO_CONTINUATIONS=0`, the issue waits there until a human
re-applies `ready-for-agent` to requeue it. With `MAX_AUTO_CONTINUATIONS=N`,
the agent claims a `round-finished` issue on its own — ahead of new
`ready-for-agent` work — up to N automatic continuations in a row; the
handoff that uses up the budget says a human requeue is needed. Re-applying
`ready-for-agent` resets the count and grants another N rounds, while
removing `round-finished` stops automatic continuation at any time.

## `merge-conflict` requeue

`merge-conflict` returns an issue to the queue because the pull request a
worker already published for it reports merge conflicts with the base
branch. It is not a triage role: the conflict attempt that follows only
brings that pull request up to date and does no new work — a clean rebase
plus setup and one profile check pushes directly, while a real rebase
conflict (or a red check after a clean rebase) runs at most one narrow
model resolution session under the ordinary model limits, never invoking
`code-review`.

With `MAX_CONFLICT_ATTEMPTS` above `0`, workers detect the conflict on
their own: every poll reads the repository's open pull requests in one
paged query and adds `merge-conflict` to an otherwise ordinary issue
(open, unassigned, non-empty body, no open blockers, none of
`ready-for-agent`, `round-finished`, `ready-for-human`, `merge-conflict`)
whose `agent/issue-N` branch reports `CONFLICTING`, so the issue is
eligible in that same poll. Each detection emits a
`conflict_requeue_detected` event naming the issue, the pull request, and
the head sha. A human may also apply the label manually with the same
effect.

With the default `MAX_CONFLICT_ATTEMPTS=0`, the label is not part of the
queue, no pull-request query runs, and a labelled issue waits untouched.
With the setting above `0`, a `merge-conflict` issue that meets
the base queue rules is claimed before automatic continuations and
ordinary work; an issue labelled `ready-for-human` is excluded. If the
issue also carries `ready-for-agent` or `round-finished`, those take
precedence and the stale `merge-conflict` signal is removed at the claim.

A pull request whose recorded head differs from its branch tip is never
labelled: GitHub stopped syncing the pull request with its branch, so its
`CONFLICTING` state cannot be trusted. The detector reads the branch tip
in the same paged query (no extra call per pull request) and, on a
mismatch, skips the label and emits a `conflict_requeue_skipped` WARNING
naming the issue, the pull request, and both shas. It also posts one
notice comment on the issue — naming both shas, linking the pull request,
and advising never to close and reopen it (GitHub refuses the reopen with
`422 ... branch was force-pushed or recreated`) but to open a replacement
pull request from the same `agent/issue-N` branch instead. The notice
carries a hidden `<!-- pr-head-stale: ... -->` marker (pull request,
recorded head, branch tip); a later poll posts again only when the state
changes, never every poll. A pull request with an unknown tip (for
example a deleted branch) is handled as today.

Every conflict attempt removes `merge-conflict` and the assignee, never
adds `ready-for-agent` or `round-finished`, and posts a result comment
stamped with a `<!-- conflict-attempt: ... -->` marker carrying the
outcome (`success`, `skipped`, or a failure kind), the pushed sha, the
conflicted files, files the resolution changed outside them, and fix
commits. Every conflict result carries its `k/N` standing against
`MAX_CONFLICT_ATTEMPTS`.
Those comments are transparent to the automatic-continuation count: they
neither consume nor reset the budget.

A green push also counts as a failure (`no_change`) when it changed
nothing on the remote branch: the pushed sha equals the pull request's
recorded head, the remote revision observed at claim time, or the sha the
previous conflict result already published (a repeated identical push).
N consecutive failures on one issue (including a push that changes
nothing) escalate to you: the attempt whose failure reaches the budget
adds `ready-for-human` — the first label the agent applies itself — and
its own result comment carries the handoff (the failure tail or reason, a
pull-request link, and the hand-back line) plus the hidden
`<!-- conflict-escalation -->` marker. While `ready-for-human` is present
the issue is out of scope: the detector will not relabel it and the queue
will not claim it, even with a hand-applied `merge-conflict`, and the
agent never removes the label. Remove `ready-for-human` to hand the issue
back: the escalation comment stops the streak, so the next poll requeues
with the full budget and N more failures are needed to escalate again.

## `agent-auto-merge` opt-in

The `agent-auto-merge` label opts one implementation issue into pull-request
auto-merge when the operator runs with `PR_AUTO_MERGE=label`. It is not a
triage role and never makes an issue eligible on its own: the issue still
needs `ready-for-agent` (and the other queue rules) to be claimed.

Like `ready-for-agent`, it sits on the collaborator trust boundary: only
repository collaborators with **write access** may apply it, because an
auto-merged PR lands on the base branch with no human review unless branch
protection requires one. The label counts only when the issue carries it at
claim time, so adding it while an attempt runs (or after it finishes) never
enables that attempt's merge.
