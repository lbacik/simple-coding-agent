# Let the agent continue a handoff on its own within an operator-set budget

A `handoff` outcome releases the issue labelled `round-finished`: until now,
continuing always required a human decision — review the handoff note and
re-apply `ready-for-agent` to requeue, or leave the issue labelled to end the
sequence. We decided the operator may instead grant the agent up to N
automatic continuations in a row (`MAX_AUTO_CONTINUATIONS`, default `0`),
so a started branch does not wait behind new work while a human requeues
every round. A human requeue resets the count and grants another N rounds;
the default `0` keeps today's behaviour exactly.

Automatic continuation reuses the existing claim path unchanged: the issue
must be open and unassigned, have a non-empty body and no open native
blockers (the same rules as the runnable queue), plus carry only
`round-finished` (never alongside `ready-for-agent`, which stays an ordinary
human requeue) with a count below the budget. Claiming assigns, removes
`round-finished`, and restores the branch with the handoff note in the
starting prompt. Eligible continuations are claimed before ordinary
`ready-for-agent` issues, FIFO by creation date within each group. The
operator `next issue` command accepts a `round-finished` target within
budget and rejects an over-budget one naming the limit (`k/N`); every
handoff result comment reports where the issue stands (`k/N`, and whether a
human requeue is now needed).

The brakes that already exist stay in place, which is why a bounded loop is
safe to run unattended. A handoff with zero commits becomes `no_changes`
and never re-adds `round-finished`, so an issue that produces nothing stops
instead of looping. A human can stop automatic continuation at any time by
removing `round-finished`. The count itself is derived from the
requeue-source markers in the agent's own result comments (see #201), so a
human-stamped result always resets it — the agent cannot drift past the
budget by relabelling, only a human requeue grants more rounds.

## Considered Options

- **Keep requiring a human requeue after every handoff.** Rejected: each
  round-trip waits on a human for a decision that is usually "keep going",
  and a started branch with a restored context queues behind fresh work.
  The default `0` preserves this behaviour for operators who want it.
- **Continue handoffs without any limit once enabled.** Rejected: an issue
  whose work never converges would hand off forever with no human looking
  at the notes. The budget bounds the unattended stretch; the handoff that
  uses it up says a human requeue is needed.
- **Allow the limit in the repository profile.** Rejected for the same
  reason as ADR 0004: the profile lives in the target repository, so an
  agent-written PR could raise its own round limit. `MAX_AUTO_CONTINUATIONS`
  is read only from the operator's `.env`, and the profile keeps rejecting
  unknown keys. A non-negative integer is required; anything else fails at
  startup with a configuration error.

## Consequences

- Operators opt in with `MAX_AUTO_CONTINUATIONS=N`; `0` (the default)
  disables the path with no extra GitHub calls and byte-identical comments.
- With `N > 0`, `round-finished` issues within budget join the runnable
  queue ahead of new work, each claim emits an `auto_continuation_claimed`
  event, and each handoff comment carries the `k/N` standing line.
- A human requeue (re-applying `ready-for-agent`) resets the count; removing
  `round-finished` stops the sequence at any time.
- No new token scope is needed: eligibility reads reuse the issue and
  comment reads the tracker already performs.

## Shared handoff record as the accounting authority (issue #283)

The decision above derived all automatic-continuation standing from the
per-attempt result comments, so a routine handoff without its own result
comment left no verifiable standing. The single shared handoff comment
(issue #282) is now authoritative instead: it carries the
`continuation_count` for its own attempt, and ordinary non-handoff results
newer than that attempt reconcile against it in attempt-ID order, so a
newer failure or human requeue cannot be hidden by older handoff metadata.
An overlapping legacy result for the same attempt counts once; conflict
attempts stay transparent (they neither consume nor reset the row); a
human requeue resets the standing from the existing claim-time signal.

The publication's intended count is computed once and retried with the same
value: the first persisted intent for an attempt always wins, so
confirming or replaying the same attempt never consumes the budget twice.
Issues without shared metadata migrate from the trusted historical result
comments using the previous semantics, and historical comments stay
intact. Missing, deleted, malformed, or unsupported metadata never
silently grants a fresh budget: accounting falls back to verifiable legacy
evidence, and anything still unverifiable (duplicate or unreadable shared
records, or a record naming a different issue or repository) retains the
issue for operator repair instead of granting more rounds. Recovery
confirms a published handoff from the shared record and verified revisions
even with no individual result comment, and an older retained checkpoint
never overwrites a newer shared record.
