# Size the soft threshold as a reserve of turns, and let sealing preserve work during a token handoff

The soft threshold used to sit a fixed share of the token budget below the hard ceiling (`SOFT_THRESHOLD_PERCENTAGE=0.2`, 800K of 4M tokens). Every turn re-counts the whole main context, so the cost of a turn grows with the context: on #155, with a ~148K context, 800K covered about five turns. The model spent them on orientation commands and hit the hard ceiling before writing its note (#159). We decided that the soft threshold is the earlier of two rules: the existing percentage, and a Handoff reserve of `HANDOFF_RESERVE_TURNS` turns (default 6) at the main thread's latest context size, plus the turn already in flight (`soft = min(max·(1−pct), max − (reserve_turns + 1)·main_context)`). We also decided that after a token soft threshold the model commits only the handoff note. It writes the note with `last_work_commit` set to the HEAD the runtime gives it, and leaves any uncommitted work to sealing (ADR 0001), which commits it on top of the note. Together with injecting the git state the model would otherwise look up, this reduces the handoff to one `Write` and one `Bash`.

## Considered Options

- **Replace the percentage with the reserve.** Rejected: the ledger's estimates are hand-tuned and can err (ADR 0002). The percentage stays as a margin against that, and operators can set it to 0.
- **A separate predictive pre-turn check.** Rejected: the runtime has no hook before a turn, only when a response is counted and in PreToolUse. Counting the in-flight turn into the reserve gives the same guarantee with one comparison.
- **The model commits its work, then writes and commits the note (three tool calls).** Rejected for the token path: it costs a turn the reserve may not have, and sealing already guarantees uncommitted work is kept. Operator and turn/time-limit handoffs are not short of tokens and keep that flow.
- **The runtime fills in or relaxes `last_work_commit`.** Rejected: the note validator's check that the note describes the work beneath it is what makes a note trustworthy.
- **Deny read-only orientation commands after the soft threshold.** Rejected: a denied tool call costs a full turn too, so it saves nothing and invites retries. The instruction tells the model not to run them instead.

## Consequences

- After a token handoff, the uncommitted work lands in a seal commit above the note, under the generic seal message rather than a descriptive one. The validator already accepts this order.
- With a very large context, the reserve can exceed the remaining budget, and the soft threshold then fires immediately. That is intended, not a configuration error.
