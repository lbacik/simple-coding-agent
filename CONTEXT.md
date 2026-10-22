# Unattended Issue Implementation

The agent attempts eligible repository issues and preserves either a proposed solution or an explanation of incomplete work for human follow-up.

## Language

**Implementation issue**:
A repository issue describing work offered to the agent, including the criteria its solution must satisfy. Planning questions on the wayfinding map are not implementation issues.
_Avoid_: Prompt, job

**Implementation attempt**:
One bounded effort to solve an implementation issue, together with its changes, evidence, and outcome. An issue and an attempt are distinct: an unsuccessful attempt does not close the issue.
_Avoid_: Issue, model turn

**Complete implementation**:
A proposed solution that satisfies the implementation issue's acceptance criteria and the configured repository checks. Completion makes the solution eligible for a pull request; it does not mean the solution has been merged.
_Avoid_: Successful model response, merged issue

**Auto-merge**:
The orchestrator-owned follow-up that asks GitHub to merge a complete implementation's pull request when the operator sets `PR_AUTO_MERGE=true`. Auto-merge is distinct from completion: it never decides whether the solution is correct, only whether GitHub should merge an already-published PR (natively once branch protection passes, or directly when the PR is already mergeable). A failed merge step is reported in the result comment and leaves the attempt `complete`. With `PR_AUTO_MERGE=label` the follow-up runs only for issues explicitly opted in with the `agent-auto-merge` label at claim time; an unopted issue still gets its PR, and the result comment reports that auto-merge was skipped for lack of opt-in.
_Avoid_: Merged issue, human merge

**Eligible issue**:
An implementation issue that the agent may claim: OPEN, unassigned, with a non-empty body, and no open native GitHub blockers, plus exactly one queue signal — `ready-for-agent` (ordinary work), only `round-finished` (an automatic continuation), or only `merge-conflict` (a conflict requeue, and only when the operator sets `MAX_CONFLICT_ATTEMPTS` above `0`). An issue labelled `ready-for-human` is never eligible.
_Avoid_: Available issue, queued issue

**Runnable queue**:
The ordered set of eligible issues, from which the agent picks the next issue to attempt: conflict requeues first, then automatic continuations, then ordinary `ready-for-agent` work, FIFO by creation date within each group.
_Avoid_: Backlog, work queue, job queue

**Claim**:
Reserving an eligible issue by assigning the agent's GitHub identity as the issue assignee, after verifying the assignee is still empty. A claimed issue is no longer eligible for other sessions.
_Avoid_: Lock, reservation

**Automatic continuation**:
An implementation attempt claimed from `round-finished` without a human requeue: the issue carried only `round-finished` (no `ready-for-agent`) at claim time, so the agent continues the previous round on its own. Its standing is carried by the single shared handoff comment (reconciled with newer ordinary results in attempt order, so a routine handoff needs no individual result comment); issues without shared metadata count the requeue source recorded in each attempt-result comment instead. A human requeue resets the count; ambiguous shared metadata retains the issue for operator repair instead of granting more rounds.
_Avoid_: Auto-retry, self requeue

**Conflict requeue**:
Returning an issue to the queue because the pull request a worker already published for it reports merge conflicts with the base branch. The signal is a `merge-conflict` label on an otherwise ordinary issue (no `ready-for-agent`, `round-finished`, or `ready-for-human`); with `MAX_CONFLICT_ATTEMPTS` above `0` workers apply it themselves — every poll reads the repository's open pull requests and labels the issue behind an `agent/issue-N` branch reporting `CONFLICTING`, so it is eligible in that same poll — and a human may also apply it manually. With `MAX_CONFLICT_ATTEMPTS=0` (the default) the signal is not part of the queue at all and no pull-request query runs. An issue carrying `ready-for-agent` or `round-finished` alongside `merge-conflict` takes that path instead, and the stale conflict signal is removed at the claim.
_Avoid_: Conflict retry, auto-rebase

**Conflict attempt**:
The attempt that follows a conflict requeue: it only brings the published pull request up to date and does no new work. After an entry check (an open, non-draft, `CONFLICTING` pull request against the profile's base branch; anything else ends `skipped`), it rebases the published branch onto the base. A clean rebase plus setup and one profile check force-pushes directly, re-enabling auto-merge per `PR_AUTO_MERGE`; a real rebase conflict, or a red check after a clean rebase, runs at most one narrow model resolution session under the ordinary model limits (context-only issue material, conflicted files, base changes, and the review-free gate — never `code-review`), which pushes only when the commit count holds, fixes land solely on top, and verification passes. Every conflict attempt removes `merge-conflict` and the assignee, never adds a queue label or a Handoff note, and posts a result comment stamped with a `<!-- conflict-attempt: ... -->` marker carrying the outcome, the pushed sha, the conflicted files, files changed outside them, and fix commits. Every conflict result carries its `k/N` standing against `MAX_CONFLICT_ATTEMPTS`; N consecutive failures (including `no_change`) escalate with `ready-for-human` — the first label the agent applies itself — and removing that label hands the issue back with the full budget.
_Avoid_: Implementation attempt, ordinary model run

**Requeue source**:
How one implementation attempt was requeued: **human** (the issue carried `ready-for-agent` at claim time), **automatic** (it carried only `round-finished` at claim time), or **conflict** (it carried `merge-conflict` at claim time). The source is decided once from the claim snapshot and recorded as a hidden `<!-- requeue-source: ... -->` marker in the attempt-result comment, next to the `<!-- agent-attempt: ... -->` marker, and in the shared handoff comment's hidden record. The automatic-continuation count is authoritative in that shared record when present (newer ordinary results reconcile on top in attempt-ID order; an overlapping legacy result for the same attempt counts once); without shared metadata it is the number of automatic-requeue result comments in a row from the newest, up to the most recent human-requeue result; a human requeue (including a legacy unmarked result) resets it to zero. Conflict-attempt result comments are transparent to that count: they are skipped without counting and without stopping the row. Ordinary conflict results carry no requeue-source marker; only the failure comment from recovery of a retained conflict attempt after a restart records the `conflict` source.
_Avoid_: Claim reason, retry count

**Attempt outcome**:
The structured result of an implementation attempt. Exactly one of: `complete` (all acceptance criteria met, checks pass, review clear, PR created), `incomplete` (work performed but criteria not met), `infrastructure_error` (failure unrelated to implementation logic), `no_changes` (skill loop finished with zero commits), or `handoff` (work paused cooperatively near a resource limit, with progress committed and published as a Handoff note for human-approved continuation; a zero-commit handoff downgrades to `no_changes`).
_Avoid_: Status, result code, exit status

**Handoff note**:
The durable record of a `handoff` outcome: a committed file at a fixed per-issue path (`.agent/handoff/<issue-number>.md`) describing remaining work, plus a verbatim copy posted as an issue comment. A continued attempt's starting prompt excludes every historical handoff result comment (including its note payload) and instructs the model to read the current note file on the resumed branch before implementing: the note is consumed from the local file, never digested from the comment history. The shared handoff comment is likewise excluded from the prompt; it is machine bookkeeping for accounting and recovery, never model guidance. Each cooperative handoff writes a self-contained replacement note covering current state, still-relevant decisions and constraints, unresolved review findings, remaining work, and applicable evidence; resolved work and obsolete instructions do not remain actionable requirements. Its `started_at` field is the attempt id (the attempt checkpoint's `started_at`, the same value as the `<!-- agent-attempt: ... -->` marker): an informational value owned by the driving process, never a gate on the handoff.
_Avoid_: Progress note, status update, summary

**Emergency handoff note**:
A Handoff note written by the runtime, not the model, when the model reaches a limit without having written one. It states its own provenance and carries only what the runtime can reconstruct (changes, activity, budget facts); it never claims to know what work remains. When an earlier round left a complete note, it preserves that note at its immutable revision as older context, clearly distinguished from the current runtime evidence; a missing or unreadable previous note is said explicitly.
_Avoid_: Fallback note, synthetic note

**Attempt workspace**:
The working tree in which one implementation attempt makes its changes, prepared from the verified base and holding the attempt's branch. It belongs to a single attempt at a time; after the attempt it returns to the base without carrying any of the attempt's changes with it.
_Avoid_: Repo, checkout, sandbox

**Sealing**:
Turning every uncommitted change in the attempt workspace into a commit attributed to that attempt (issue, attempt ID, and the reason the model stopped). An attempt is sealed after every model execution, whatever its status, so no work the model wrote is ever lost or leaks onto the base. A workspace mid-way through an unresolved rebase or merge is never sealed: its conflicts are aborted instead.
_Avoid_: Preserving dirty work, safety-net commit, checkpoint (the attempt checkpoint is a different thing)

**Token budget**:
The limit on the total tokens one implementation attempt may process and generate, counting every token category unweighted and including the tokens spent by its subagents. It measures the amount of work in the attempt, not its price; a USD limit exists only as a looser backstop.
_Avoid_: Cost budget, spend limit

**Soft threshold**:
The point in the token budget past which the model must end the attempt with a handoff, placed early enough to leave a Handoff reserve before the Hard ceiling.
_Avoid_: Cost threshold, warning limit

**Handoff reserve**:
The part of the token budget kept back for the handoff itself: the number of turns guaranteed after the turn that crosses the soft threshold (`HANDOFF_RESERVE_TURNS`, default 6), sized at the main thread's latest context size, since every turn re-counts that whole context. It pulls the effective soft threshold earlier when the context is large.
_Avoid_: Buffer, headroom, safety margin, turn cap

**Hard ceiling**:
The point at which the attempt is stopped regardless of the model, whether or not it has handed off.
_Avoid_: Budget cap, kill limit

**Pinned runtime**:
The single approved agent-SDK release, together with the Claude Code CLI shipped inside it, that every model execution must run on. Pinning the SDK pins the CLI; the CLI never has a version of its own to keep in step. Provenance verification confirms at startup that the CLI about to run is exactly the one this release ships, and refuses to start otherwise.
_Avoid_: Runtime pair, SDK and CLI pins, the CLI on PATH

**Dashboard**:
A separate operator view over the managed instances on one Docker host. It observes instances and, from its second version, relays operator commands to them; it never claims issues or takes part in an implementation attempt.
_Avoid_: Control panel, monitor, fleet manager

**Managed instance**:
One agent container that a dashboard discovers and observes, serving exactly one target repository. It is identified by its container name, so recreating the container keeps the same managed instance; a container started for a one-off command is never a managed instance.
_Avoid_: Agent, worker, node

**Telemetry snapshot**:
A point-in-time, read-only report a managed instance gives about itself: its control state, the current (or last finished) implementation attempt's token, budget and context figures, and its most recent events. The agent owns and versions it; a dashboard only reads it and tolerates versions it does not fully know.
_Avoid_: Status, metrics, heartbeat

**Figure basis**:
How a figure in a telemetry snapshot was obtained: `measured` (reported by the provider), `estimated` (computed by the agent), `configured` (set by the operator), `sdk_assumed` (a default the runtime assumes for an unknown model), or `unavailable` (no trustworthy source). An unavailable figure is never shown as zero.
_Avoid_: Provenance (reserved for provenance verification of the pinned runtime), source, confidence
