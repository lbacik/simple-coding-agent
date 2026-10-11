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
