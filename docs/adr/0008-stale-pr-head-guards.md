# Bound the conflict loop when a pull request head goes stale

A pull request whose recorded head (`headRefOid`) no longer follows its
branch tip desyncs the conflict path: the detector sees `CONFLICTING` and
requeues, the attempt rebases cleanly and force-pushes a revision the
remote branch already holds, and publication reports `success` because the
pushed sha differs from the stale recorded head. Each `success` resets the
conflict failure streak, so `MAX_CONFLICT_ATTEMPTS` (ADR 0007) never
escalates and the loop runs unbounded. We decided the agent treats a stale
head as untrustworthy at every layer, so an external sync glitch cannot
spam an issue indefinitely.

The detector never labels a pull request whose recorded head differs from
its branch tip (read in the same paged query, so no extra call per pull
request); it emits a `conflict_requeue_skipped` WARNING with reason
`pr_head_stale` and both shas instead. A pull request with an unknown tip
(a deleted branch, or a response without the field) falls back to today's
behaviour. The detector also posts one notice comment per stale state on
the issue — naming both shas, linking the pull request, carrying a hidden
`<!-- pr-head-stale: ... -->` marker (pull request, recorded head, branch
tip), and advising never to close and reopen the pull request (GitHub
refuses the reopen with `422 ... branch was force-pushed or recreated`)
but to open a replacement from the same `agent/issue-N` branch instead. A
later poll posts again only when the state changes; the notice is
transparent to the conflict streak and the automatic-continuation count,
like conflict results, and only markers from trusted agent logins suppress
it. No `ready-for-human` or other label is applied for a stale head.

Publication reports a green push as `no_change` (a failure toward the
budget, as today) when the pushed sha equals the remote branch revision
recorded at claim — the claim lease — in addition to the existing pull
request head comparison, and when it equals the sha the immediately
preceding conflict result already published (a repeated identical push).
Without a recorded lease the existing comparison decides exactly as
before. The conflict-attempt entry check likewise ends `skipped` with a
`stale_head` reason when a recorded claim lease differs from the pull
request's recorded head, before any rebase or push.

## Considered Options

- **Trust the pull request head everywhere.** Rejected: it is exactly what
  desynced. The remote branch revision observed at claim is the only push
  baseline the agent controls, and the branch tip read alongside the pull
  request is the only freshness signal that needs no extra call.
- **Repair the stale pull request (close/reopen, recreate, or push an
  empty commit).** Rejected: closing risks the unrecoverable `422`
  reopen refusal, recreating or pushing changes what a human owns, and any
  of them acts on a pull request whose state GitHub itself lost.
- **Wait for the head to resync after a push.** Rejected: the observed
  pull request never recovered on its own, so polling would stall every
  conflict attempt on an event that may never arrive.
- **Escalate stale heads with `ready-for-human`.** Rejected: the label
  gates the queue and the budget; a desync needs a human note, not a gate,
  and the notice comment is the only signal.

## Consequences

- An external sync loss requeues nothing, pushes nothing new, and posts at
  most one comment per stale state; the existing budget then bounds any
  residual repetition (three `no_change` results escalate with
  `ready-for-human` as before).
- The budget semantics of ADR 0007 are unchanged: `no_change` already
  counted as a failure, and the two extra downgrade rules only decide what
  counts as `no_change`. Enforcement stays at escalation time.
- No extra per-pull-request API calls: the tip rides in the detector's
  existing paged query, and the notice dedup reads comments only on stale
  polls (plus at most one lazy viewer lookup per poll).
