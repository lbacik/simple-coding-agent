# Record an attempt's verdict when publication decides it

The publisher decides an attempt's verdict: `complete` only after the pull
request exists (a failed pull request call becomes `infrastructure_error`),
and every other outcome from the evidence it publishes. The engine used to
record that verdict only at the very end of finalization, in the same step
that wrote the ledger and emitted `outcome_recorded`. When finalization
failed at its boundary (result comment unconfirmed, issue release or
cleanup failing), the Retained Attempt kept no verdict at all, and
`recovery release` re-inferred one from the workspace. That inference can
answer `handoff`, `incomplete` or `infrastructure_error` but never
`complete`, so an attempt that had already opened its pull request was
released as `incomplete`: a command changed the recorded verdict, which the
operator contract forbids (§14).

We decided:

1. **A verdict is recorded when the publisher returns it**, before the
   finalization boundary. The engine durably writes the outcome and a short
   reason into the attempt checkpoint (`verdict`), and the outcome into the
   attempt's archive record, before it checks the result comment, releases
   the issue or cleans up. A recorded verdict survives a failed
   finalization and a process restart. Only the first recording counts: a
   later publication of the same attempt never overwrites it.
2. **This is consistent with #84** ("an unpublished branch is never reported
   as a success"): `complete` is only ever decided after the pull request
   exists, so a recorded `complete` is never an unpublished branch.
   `recovery release` still posts a release comment that names the recorded
   verdict and where the operator secured the work, does a plain release
   (never `round-finished`) and retains the branch.
3. **Inference remains the fallback only when no verdict was recorded**:
   attempts interrupted before publication returned (a crash during setup,
   the model run or the push) and checkpoints written before this change.
   `recovery release` and `recovery retry` keep today's inference there,
   unchanged. With a recorded verdict, `recovery release` finalizes with it,
   and `recovery retry` and startup reconciliation after a restart
   republish towards it and finalize with it. When safe republication
   cannot reach it — a recorded `handoff` whose note can no longer be
   recovered, a recorded `complete` the checkpoint no longer shows as
   publishing (republishing would push the branch first), or a publisher
   answering another outcome —
   retry answers `not_fulfilled`, startup reconciliation emits
   `recorded_verdict_held`, and the attempt stays retained for the
   operator.
4. **`outcome_recorded` moves to the recording moment.** It is emitted once
   per attempt, right after the verdict is recorded, and never again at the
   final commit or after a restart. An attempt that reaches finalization
   without a recorded verdict still emits it exactly once, at finalization.
   The Contract Event order stays `claim_opened … outcome_recorded …
   attempt_finished`. The conformance harness's `after-outcome` hold, which
   waits right after `outcome_recorded`, now takes effect before the result
   comment is confirmed.

## Considered Options

- **Teach `recovery release` to infer `complete` from an open pull
  request.** Rejected: it would still re-infer a verdict instead of keeping
  the one publication decided, and the workspace and remote can drift after
  the fact (a pull request closed by a human, a branch force-pushed).
- **Record the verdict in the completion ledger.** Rejected: the ledger
  marks the durable completion boundary, and recovery reads a ledger entry
  as "already finalized". A verdict recorded before the boundary must not
  look like a finished attempt.
- **Keep emitting `outcome_recorded` at finalization.** Rejected: the event
  would then lag the recorded verdict, and a Retained Attempt carrying a
  verdict would never announce it until an operator acted.

## Consequences

- `attempts show` reports the outcome of an active or retained attempt as
  soon as publication decides it; `last_finished` still changes only at
  finalization (the archive gains no `completed_at` until then).
- A Retained Attempt held by a failed Operator Handoff now carries the
  verdict its publication recorded, so `recovery retry` finalizes it with
  that verdict instead of re-inferring `handoff` from a note found later.
- A retained conflict attempt with a recorded verdict other than
  `infrastructure_error` is no longer republished as a conflict error after
  a restart; it stays retained for `recovery retry` or `recovery release`.
