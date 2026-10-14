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
An implementation issue that the agent may claim: labelled `ready-for-agent`, OPEN, unassigned, with a non-empty body, and no open native GitHub blockers.
_Avoid_: Available issue, queued issue

**Runnable queue**:
The ordered set of eligible issues, sorted FIFO by creation date, from which the agent picks the next issue to attempt.
_Avoid_: Backlog, work queue, job queue

**Claim**:
Reserving an eligible issue by assigning the agent's GitHub identity as the issue assignee, after verifying the assignee is still empty. A claimed issue is no longer eligible for other sessions.
_Avoid_: Lock, reservation

**Automatic continuation**:
An implementation attempt claimed from `round-finished` without a human requeue: the issue carried only `round-finished` (no `ready-for-agent`) at claim time, so the agent continues the previous round on its own. It is counted by the requeue source recorded in each attempt-result comment.
_Avoid_: Auto-retry, self requeue

**Requeue source**:
How one implementation attempt was requeued: **human** (the issue carried `ready-for-agent` at claim time) or **automatic** (it carried only `round-finished` at claim time). The source is decided once from the claim snapshot and recorded as a hidden `<!-- requeue-source: ... -->` marker in the attempt-result comment, next to the `<!-- agent-attempt: ... -->` marker. The automatic-continuation count is the number of automatic-requeue result comments in a row from the newest, up to the most recent human-requeue result; a human requeue (including a legacy unmarked result) resets it to zero.
_Avoid_: Claim reason, retry count

**Attempt outcome**:
The structured result of an implementation attempt. Exactly one of: `complete` (all acceptance criteria met, checks pass, review clear, PR created), `incomplete` (work performed but criteria not met), `infrastructure_error` (failure unrelated to implementation logic), `no_changes` (skill loop finished with zero commits), or `handoff` (work paused cooperatively near a resource limit, with progress committed and published as a Handoff note for human-approved continuation; a zero-commit handoff downgrades to `no_changes`).
_Avoid_: Status, result code, exit status

**Handoff note**:
The durable record of a `handoff` outcome: a committed file at a fixed per-issue path (`.agent/handoff/<issue-number>.md`) describing remaining work, plus a verbatim copy posted as an issue comment. The comment copy is what a continuation's starting prompt is enriched with; the file itself is only pointed at, never digested, since it is already reachable on the resumed branch. Its `started_at` field is the attempt id (the attempt checkpoint's `started_at`, the same value as the `<!-- agent-attempt: ... -->` marker): an informational value owned by the driving process, never a gate on the handoff.
_Avoid_: Progress note, status update, summary

**Emergency handoff note**:
A Handoff note written by the runtime, not the model, when the model reaches a limit without having written one. It states its own provenance and carries only what the runtime can reconstruct (changes, activity, budget facts); it never claims to know what work remains.
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
