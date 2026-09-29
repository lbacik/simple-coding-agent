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

**Eligible issue**:
An implementation issue that the agent may claim: labelled `ready-for-agent`, OPEN, unassigned, with a non-empty body, and no open native GitHub blockers.
_Avoid_: Available issue, queued issue

**Runnable queue**:
The ordered set of eligible issues, sorted FIFO by creation date, from which the agent picks the next issue to attempt.
_Avoid_: Backlog, work queue, job queue

**Claim**:
Reserving an eligible issue by assigning the agent's GitHub identity as the issue assignee, after verifying the assignee is still empty. A claimed issue is no longer eligible for other sessions.
_Avoid_: Lock, reservation

**Attempt outcome**:
The structured result of an implementation attempt. Exactly one of: `complete` (all acceptance criteria met, checks pass, review clear, PR created), `incomplete` (work performed but criteria not met), `infrastructure_error` (failure unrelated to implementation logic), `no_changes` (skill loop finished with zero commits), or `handoff` (work paused cooperatively near a resource limit, with progress committed and published as a Handoff note for human-approved continuation; a zero-commit handoff downgrades to `no_changes`).
_Avoid_: Status, result code, exit status

**Handoff note**:
The durable record of a `handoff` outcome: a committed file at a fixed per-issue path (`.agent/handoff/<issue-number>.md`) describing remaining work, plus a verbatim copy posted as an issue comment. The comment copy is what a continuation's starting prompt is enriched with; the file itself is only pointed at, never digested, since it is already reachable on the resumed branch.
_Avoid_: Progress note, status update, summary

**Attempt workspace**:
The working tree in which one implementation attempt makes its changes, prepared from the verified base and holding the attempt's branch. It belongs to a single attempt at a time; after the attempt it returns to the base without carrying any of the attempt's changes with it.
_Avoid_: Repo, checkout, sandbox

**Sealing**:
Turning every uncommitted change in the attempt workspace into a commit attributed to that attempt (issue, attempt ID, and the reason the model stopped). An attempt is sealed after every model execution, whatever its status, so no work the model wrote is ever lost or leaks onto the base. A workspace mid-way through an unresolved rebase or merge is never sealed: its conflicts are aborted instead.
_Avoid_: Preserving dirty work, safety-net commit, checkpoint (the attempt checkpoint is a different thing)
