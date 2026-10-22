---
name: handoff
description: "Pause an implementation attempt cooperatively near a resource limit, preserving progress for human-approved continuation."
---

Invoke this skill only when instructed to do so (a token soft-threshold
notice, a turn/time limit follow-up prompt, or an operator `handoff now`
request delivered by the driving process). It is the only supported way
to end an attempt early while keeping the work usable by a continuation.

The driving process owns remote publication (push, PR, comments, labels).
This skill only commits locally. Never run `git push`, `gh pr create`, or any
other publishing command.

## Token soft-threshold handoff (two tool calls)

When the handoff was triggered by a token soft-threshold notice, the
instruction already supplies the current branch state: the branch, the base
revision, the HEAD SHA, `git status --short`, `git log --oneline
<base>..HEAD`, `git diff --stat <base>`, and a pre-filled note header whose
`last_work_commit` is HEAD at injection time. Trust that state; do not run
diagnostic commands to re-derive it.

The handoff then takes at most two tool calls, and work is not committed in
this case:

1. Create or edit `.agent/handoff/<issue-number>.md` with the `Write` tool
   (use `Edit` to update an existing note), using the pre-filled header
   verbatim, and fill in Summary, Remaining work and Evidence. Bash cannot
   create or modify files while the handoff is pending, so always use
   `Write`/`Edit` for the note.
2. In one `Bash` call, run
   `git add .agent/handoff/<issue-number>.md && git commit -m "Handoff note: issue #<issue-number>"`.
3. Stop.

If a Handoff note already exists at `.agent/handoff/<issue-number>.md`,
read it first: your note replaces it as the single continuation record, so
carry forward everything still relevant to continuation — current state,
still-applicable decisions and constraints, unresolved review findings,
remaining work, and evidence. Resolved work and obsolete instructions stay
behind; the next attempt reads only your note, never the old comments.

Do not commit work yourself on this path: sealing commits any uncommitted
work after the attempt, so the branch ends as the earlier work commits, then
the note commit, then (if there was uncommitted work) the seal commit.

## Steps (operator and turn/time-limit handoffs)

This is the documented path for operator `handoff now` requests and for
turn/time-limit follow-ups. (Token soft-threshold handoffs use the two-call
path above instead and skip step 1.)

1. **Preserve outstanding work.** If there is any dirty or untracked change
   in the working directory, commit it now, as one or more ordinary work
   commits with descriptive messages. Do not combine this with the handoff
   note commit (step 3) — the note commit must be separate and must be the
   last commit on the branch.

2. **Write the handoff note** at `.agent/handoff/<issue-number>.md` with the
   `Write` tool (use `Edit` to update an existing note),
   overwriting any note already there (its history remains reachable through
   Git). Bash cannot create or modify files while the handoff is pending, so
   always use `Write`/`Edit` for the note, then commit it with `git add` +
   `git commit` in Bash. When a note is already present, read it first: your note replaces
   it as the single continuation record, so it must be self-contained —
   current state, still-relevant decisions and constraints, unresolved
   review findings, remaining work, and applicable evidence. Resolved work
   and obsolete instructions do not remain actionable requirements. Use this
   structure:

   ```markdown
   # Handoff note: issue #<issue-number>

   - issue: <issue-number>
   - started_at: <the attempt id given in the handoff instruction; write it
     verbatim. If no attempt id was given, omit this line>
   - reason: <token_soft_threshold | token_hard_ceiling | cost_hard_limit | turn_limit | time_limit | operator_request>
   - last_work_commit: <SHA of the last commit from step 1, or the branch's
     existing HEAD if step 1 committed nothing>

   ## Summary

   <What was accomplished this attempt, in a few sentences.>

   ## Remaining work

   <A concrete, actionable list of what is left to do.>

   ## Known issues

   <Optional: anything broken, flaky, or suspicious that a continuation
   should know about. Omit this section if there is nothing to report.>

   ## Evidence

   <References a continuation can check: test names, file paths, command
   output locations.>
   ```

3. **Commit the note separately**, as the final commit on the branch, with a
   commit message whose first line is exactly:

   ```
   Handoff note: issue #<issue-number>
   ```

   This exact subject line is how the driving process recognizes the note
   commit and distinguishes it from preserved work.

4. **Stop.** Do not run further checks, do not continue implementing, and do
   not attempt to push or open a pull request. The driving process publishes
   the note, posts it as an issue comment, and manages the continuation
   labels.

If there is no code change to preserve (no dirty/untracked changes and no commits
already on the branch beyond its base), still write and commit the note —
the handoff note itself preserves progress (such as investigation findings
and implementation plans), and the driving process will publish the branch and note
for human evaluation.

When the driving process delivered an operator `handoff now` request, use
`reason: operator_request` in the note. The request reaches you at a safe
boundary (or one follow-up prompt) and your handoff work still ends at the
attempt's original deadline: preserve work in ordinary commits, write and
commit the note separately as the final commit, then stop as usual.
