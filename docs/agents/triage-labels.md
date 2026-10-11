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
