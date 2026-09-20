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
The structured result of an implementation attempt. Exactly one of: `complete` (all acceptance criteria met, checks pass, review clear, PR created), `incomplete` (work performed but criteria not met), `infrastructure_error` (failure unrelated to implementation logic), or `no_changes` (skill loop finished with zero commits).
_Avoid_: Status, result code, exit status
