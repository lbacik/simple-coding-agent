# Operator Onboarding Guide

This guide covers target-profile onboarding, credential configuration,
trust policies, operational monitoring, and recovery procedures for the
simple-coding-agent.

## Target profile onboarding

Every target repository must supply a profile at
`docs/agents/simple-coding-agent-profile.yml`.  See
[`examples/simple-coding-agent-profile.yml`](../examples/simple-coding-agent-profile.yml)
for a reference.  Required fields are `setup` (one command or a list) and
`check` (the same shape).  Optional fields: `base_branch` (default `main`),
`timeout` (seconds, default 300), `setup_timeout` (seconds, default 120),
and `env` (a string→string map).

The profile must be in the default branch before a `ready-for-agent` issue is
claimed.  A missing or unparseable profile is an `infrastructure_error`; the
agent posts a result comment and removes its assignee.

Set `PROFILE_PATH` to load the profile from a path outside the target
repository instead, such as alongside the agent's own `Dockerfile`.  When
unset or empty, the default in-repo path above is used.

Required tracker context for skills to function correctly:
- The repository's `CONTEXT.md` describes the project domain vocabulary.
- `docs/agents/issue-tracker.md` explains how the agent finds and claims issues.
- `docs/agents/triage-labels.md` explains the label lifecycle.
- `docs/agents/domain.md` links to Architecture Decision Records.

Keep all four files current.  The agent passes the issue body verbatim to the
model; content in these files shapes the model's understanding of the domain.

## Required GitHub PAT permissions

Use a fine-grained Personal Access Token scoped to the single target
repository with exactly:
- `contents: write` — push branches
- `pull-requests: write` — open pull requests
- `issues: write` — add comments, set and remove labels and assignments

Do not use a classic token or a token with broader scope.  Store the token in
`GITHUB_TOKEN` inside the `.env` file (never in source control).

## `ready-for-agent` trust policy

The `ready-for-agent` label is the trust boundary.  Only repository
collaborators with **write access** may apply it.  Once applied, the agent
treats the issue body as an implementation specification and passes it verbatim
to the model, which runs with `bypassPermissions`.

Policy requirements:
- Review every issue body before applying `ready-for-agent`.
- Never apply the label to an issue you did not write or have not read in full.
- Treat the label as equivalent to allowing arbitrary code execution inside
  the repository clone.

## Project settings opt-in and residual risks

`AGENT_TRUST_PROJECT_SETTINGS` defaults to `false`.  When false, only your
user settings (`~/.claude/`) load; no repository-supplied configuration runs.

Setting it to `true` enables `setting_sources: [user, project]` with
`disableAllHooks: true` and `strict_mcp_config: true`.  Even with hooks
disabled, three repository-supplied inputs remain active:
- `apiKeyHelper` — executes a subprocess to obtain an API key.
- `env` entries in project settings — extend the model's environment.
- Project-skill `allowed-tools` — expands the tool allow-list.

Enable project settings only for a repository you control.  When in doubt,
leave the default `false`.

## Log retention

Attempt evidence is retained at `$DATA_DIR/logs/<issue>/<timestamp>/` and
is never automatically removed.  The directory contains:
- `outcome.json` — structured attempt result and token metadata.
- `setup.log`, `baseline_check.log`, `final_check.log` — command output.
- `sdk_transcript/` — available SDK skill-event and token data.

All log output is credential-redacted at write time.  Any SDK dollar
estimate in `outcome.json` is explicitly non-authoritative; rely on
Meta's billing dashboard for accurate cost data.

Establish a retention policy appropriate for your compliance environment.
The agent does not delete log directories.

## Provider spending caps

Configure spending limits in Meta's dashboard.  The agent sets
`max_budget_usd=5` as a client-side SDK circuit breaker; this is not
authoritative billing.  Override it with `MAX_BUDGET_USD` (a positive
integer number of dollars) and the turn cap with `MAX_TURNS` (default 60).
The only model used is `muse-spark-1.3-contributor`; no fallback is
configured.

Before the hard ceiling, the agent tries a cooperative `handoff`: once
estimated spend crosses `soft = max_budget_usd - max_budget_usd *
SOFT_THRESHOLD_PERCENTAGE` (default `SOFT_THRESHOLD_PERCENTAGE=0.2`, i.e. 80%
of budget), it asks the model to commit its progress and stop. This estimate
is derived from streamed token counts, not the SDK's authoritative
`total_cost_usd` (only available once the attempt ends), so treat the
threshold as approximate. Once the soft threshold is crossed, tool enforcement
blocks general file modifications and arbitrary execution, only allowing the
model to write the handoff note (`.agent/handoff/<issue-number>.md`) and execute
read/inspection or git wrap-up commands (e.g. `git status`, `git add`, `git commit`).
Turns and `MODEL_TIMEOUT` get no soft threshold: reaching either limit gets
one best-effort same-client handoff follow-up instead.

If the hard ceiling (`max_budget_usd`) is reached, or the model limit is hit
without a cooperative handoff note, the agent performs an **emergency handoff**:
any uncommitted dirty work is automatically committed, an emergency handoff note
documenting the limit (`cost_hard_limit` or `model_limit`) and touched files is
created and committed, and the outcome is recorded as `handoff` so that no work
is lost. If zero work was accomplished before hitting the limit, the attempt
downgrades cleanly to `no_changes`.

A `handoff` outcome pushes the attempt branch (no pull request), posts the
handoff note as an issue comment, removes `ready-for-agent`, and adds
`round-finished`. **Continuation requires a human decision**: review the
note and comment, then re-apply `ready-for-agent` to requeue the issue (this
also clears `round-finished`), or leave it labelled `round-finished` to end
the attempt sequence.

Independent of the handoff path, any dirty or untracked change left in the
working tree once the model reports success is also committed automatically
before commits are counted and the final check runs. Without this, a model
that describes further edits (e.g. in response to a code review) but stops
before actually running `git commit` would have those edits silently left
out of the pushed branch and the pull request, even though the final check
observed them on disk and the attempt still gets recorded as `complete`.

## Non-destructive workspace management

The agent adheres strictly to non-destructive cleanup:
- The agent **never** deletes uncommitted files or removes branches after an attempt
  ends (`git reset --hard`, `git clean -fd`, and `git branch -D` are not run on attempt cleanup).
- Attempt branches and untracked artifacts remain intact for human review and debugging.
- **Clean workspace precondition**: Before starting or claiming an attempt, the agent
  inspects the repository working tree (`git status --porcelain`). If the workspace
  contains uncommitted or untracked changes, the agent:
  1. Emits a warning log (`working_tree_dirty`).
  2. Releases the claim, removes the `ready-for-agent` label from the issue, and posts
     a comment indicating that uncommitted changes exist in the workspace.
  3. Halts execution with exit code 1.
  Cleaning the working tree is strictly the responsibility of human operators.

## Per-instance operator control (`agentctl`)

Each running agent container accepts operator commands through an installed
control CLI. Container selection is the instance selector: the CLI has no
instance flag, so opening the wrong container controls the wrong instance.

1. Open a shell in the intended **running** container (for example, through
   OrbStack) and verify its identity before any mutating command:

   ```shell
   /usr/local/bin/agentctl status
   ```

   The first line of every response names the `TARGET_REPO` configured for
   that container (status, command acceptance, and request-ID lookup all
   display it). If the repository is not the one you intend to control,
   leave that container and open the correct one.

2. Submit a command from the same shell, e.g.
   `/usr/local/bin/agentctl stop`. The CLI prints a unique request ID
   before submission; if the connection drops before the reply arrives,
   retry the identical payload with
   `agentctl stop --request-id <id>` and look it up later with
   `agentctl command <id>`.

Always use the absolute path `/usr/local/bin/agentctl`. The image prepends
the target checkout's virtual environment to `PATH`, so a bare `agentctl`
could resolve to the target repository's package instead of the running
agent's installed entry point.

Never start a second agent process or a one-shot Compose container
(`docker compose run`) to issue a command: commands are accepted only by the
live agent inside its own running container, and a second process would
compete for the same persistent state. Likewise, never read or edit
`$DATA_DIR/state/control.sqlite3` directly; the CLI talks only to the live
process over its private socket.

### Runtime socket and persistence

The live agent binds `/run/simple-coding-agent/control.sock` with mode
`0600` inside `/run/simple-coding-agent` (owner `agent`, mode `0700`).
Only the `agent` OS user and container root can reach it. The directory and
socket are container-local runtime state: no bind mount, published port, or
shared socket volume exposes control to another instance, and there is no
network listener.

Accepted commands, request IDs, acknowledgements, and intake state persist
on the instance's `/data` volume and survive container replacement; the new
container recreates the runtime directory and socket on startup. While
startup reconciliation is still running, live status reports `recovering`.
When the agent process cannot be reached at all, the CLI reports a
connection error rather than a saved snapshot.

### Retained-attempt recovery hold

When an attempt cannot finish its completion boundary (confirmed result
comment, issue release, local cleanup, and accounting), the agent preserves
the branch, checkpoint, and working tree and holds intake: `status` shows a
`recovery` block with the attempt identity, phase, confirmed publication
progress, branch/workspace/checkpoint, hold reason, and the next operator
action. `resume` cannot bypass the hold, and no new issue is claimed until
an operator clears it:

- `/usr/local/bin/agentctl recovery retry <attempt-id>` rechecks local and
  remote evidence and retries only the unconfirmed publication, release,
  cleanup, and accounting steps. It never reruns the model. Ambiguous
  remote state, conflicting human changes, and persistent failures keep the
  hold.
- `/usr/local/bin/agentctl recovery release <attempt-id> --saved-at
  <path-or-url>` abandons the attempt after you secured its work elsewhere.
  It confirms a deduplicated issue comment stating the actual outcome and
  the saved-work reference before changing labels or assignee, and it never
  claims a successful handoff.

Both commands print a durable request ID and accept `--request-id` for an
identical retry after an ambiguous connection loss, like `stop`.

A related hold covers unexplained dirty or untracked work with no attempt
identity behind it. It also blocks intake and `resume`, and `status` reports
it with the repair instruction — but no `recovery` subcommand can clear it:
inspect the working tree and repair or remove the unexplained changes
manually.

### Operator-requested handoff finalization

`agentctl handoff now` asks the active attempt to hand off to you with a
published `operator_request` note. The accepted command keeps its original
deadlines across restarts: the model has 240 seconds from acceptance to begin
the handoff skill, publication has 120 seconds after the valid local handoff,
and the whole command has 360 seconds from acceptance — `status` shows all
three in the `handoff_deadlines` line. Completion waits for publication,
release, and cleanup; anything short of that (including a missing invocation,
an invalid note, or an expired deadline) publishes the actual attempt outcome
and marks the command `not fulfilled` with its reason.

A failed handoff holds like a retained attempt above: the branch, checkpoint,
and working tree stay preserved, later claims stay blocked, and `recovery
retry` (replays only unconfirmed safe steps, never the model) or `recovery
release --saved-at` (after you secured the work) clears it without duplicate
remote effects.

## Persisted error-guard recovery

The agent tracks consecutive `infrastructure_error` outcomes in
`$DATA_DIR/state/consecutive_errors.json`.  At `MAX_CONSECUTIVE_ERRORS`
(default 3) the process exits with status 1.

A restart alone does **not** reset the guard.  To recover:
1. Diagnose and fix the underlying infrastructure problem (credentials,
   network, disk, or broken provenance).
2. Either remove `$DATA_DIR/state/consecutive_errors.json` to reset the
   counter, or allow the next successful outcome to reset it automatically.
3. Restart the agent.

An operator's auto-restart policy (e.g. `restart: unless-stopped` in Compose)
cannot silently defeat this guard because the counter is persisted across
restarts.
