# Bound conflict attempts per issue, then hand to a human with `ready-for-human`

A conflict attempt can fail repeatedly on the same issue — the rebase still
conflicts, the check stays red, the push fails, or the push lands but changes
nothing on the remote branch (`no_change`). We decided the operator grants up
to N consecutive failed conflict attempts per issue (`MAX_CONFLICT_ATTEMPTS`,
default `0`), and the attempt whose failure reaches the budget hands the issue
to a human instead of requeueing silently: its own result comment carries the
handoff (the failure tail or reason in a collapsed section, a link to the pull
request, and a line telling the human to remove `ready-for-human` to grant N
more attempts), and the issue gains the `ready-for-human` label. This is the
first use of `ready-for-human` by agent code: until now only humans applied
it. The agent never removes it — a human hands the issue back by removing it,
which requeues with the full budget.

The count is derived from the conflict-attempt markers in the agent's own
result comments (its own login, plus pool-peer logins where configured), so
there is no `/data` state and no counter label to drift: reading newest first,
consecutive failures count, and the streak stops at a success (a pushed sha),
an ordinary attempt result, a `skipped` result, or the latest escalation
comment. Every conflict-attempt result reports where the issue stands
(`k/N`). The budget is enforced at escalation time, never at claim time, so
lowering N below a running streak escalates on the next failure, and an issue
carrying `ready-for-human` is neither labelled by the detector nor claimed
from `merge-conflict`, even with a hand-applied signal.

## Considered Options

- **Requeue conflicts without any limit.** Rejected: an issue whose pull
  request never converges would rebase forever with no human looking at the
  failure. The budget bounds the unattended stretch; the attempt that uses it
  up says a human is needed.
- **Keep the counter in `/data` or on a label.** Rejected: durable local
  state can disagree with the visible comment history after restarts and
  operator moves, and a label is one more thing to add, read, and remove.
  Deriving the streak from the comments keeps the history itself
  authoritative, like the automatic-continuation count (ADR 0005).
- **Enforce the budget at claim time (refuse the N+1th claim).** Rejected:
  the detector would need the comment read on every poll, and a lowered
  budget would silently strand an already-queued issue. Enforcing at
  escalation keeps claiming cheap and makes a lowered budget take effect on
  the next failure.
- **Have the agent remove `ready-for-human` when retrying.** Rejected: the
  label is the human's gate. Only a human removal hands the issue back; the
  agent treats a labelled issue as out of scope for detection and claiming.

## Consequences

- Operators opt in with `MAX_CONFLICT_ATTEMPTS=N`; `0` (the default)
  disables the path with no extra GitHub calls and byte-identical behaviour.
- With `N > 0`, the Nth consecutive failure escalates exactly once: one
  `conflict_escalated` event, one `ready-for-human` label (created when
  missing), and one escalation comment carrying the hidden
  `<!-- conflict-escalation -->` marker that stops the streak.
- Removing `ready-for-human` requeues with the full budget; N more failures
  are needed to escalate again.
- Like the other budgets, the setting is operator-only: it lives in `.env`,
  never in the repository profile (an unknown profile key is refused), so an
  agent-written PR cannot raise its own conflict budget (ADR 0004).
