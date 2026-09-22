---
name: handoff
description: "Pause an implementation attempt cooperatively near a resource limit, preserving progress for human-approved continuation."
---

Invoke this skill only when instructed to do so (a cost soft-threshold
notice, or a turn/time limit follow-up prompt). It is the only supported way
to end an attempt early while keeping the work usable by a continuation.

The driving process owns remote publication (push, PR, comments, labels).
This skill only commits locally. Never run `git push`, `gh pr create`, or any
other publishing command.

## Steps

1. **Preserve outstanding work.** If there is any dirty or untracked change
   in the working directory, commit it now, as one or more ordinary work
   commits with descriptive messages. Do not combine this with the handoff
   note commit (step 3) — the note commit must be separate and must be the
   last commit on the branch.

2. **Write the handoff note** at `.agent/handoff/<issue-number>.md`,
   overwriting any note already there (its history remains reachable through
   Git). Use this structure:

   ```markdown
   # Handoff note: issue #<issue-number>

   - issue: <issue-number>
   - started_at: <UTC ISO-8601 timestamp, e.g. from `date -u +%Y-%m-%dT%H:%M:%SZ`>
   - reason: <cost_soft_threshold | cost_hard_limit | turn_limit | time_limit>
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
