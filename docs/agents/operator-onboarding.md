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
`max_budget_usd=20` as a client-side SDK circuit breaker; this is not
authoritative billing.  Override it with `MAX_BUDGET_USD` (a positive
integer number of dollars) and the turn cap with `MAX_TURNS` (default 60).
The only model used is `muse-spark-1.3-contributor`; no fallback is
configured.

USD figures for the Meta backend come from the container's managed-settings
`modelPricing` entry (`docker/managed-settings.json`, installed at
`/etc/claude-code/managed-settings.json`), which prices
`muse-spark-1.3-contributor` at Meta's published rates instead of the CLI's
default-model rates.  They are still estimates for observability, not a
measured bill: rely on Meta's billing dashboard for accurate cost data.

Before the hard ceiling, the agent tries a cooperative `handoff`: once the
token budget crosses the effective soft threshold, it asks the model to
commit its progress and stop. The threshold is the earlier of two rules,
computed from the main thread's latest context size (defaults
`MAX_BUDGET_TOKENS=4000000`, `SOFT_THRESHOLD_PERCENTAGE=0.2`,
`HANDOFF_RESERVE_TURNS=6`):

`soft = min(MAX_BUDGET_TOKENS * (1 - SOFT_THRESHOLD_PERCENTAGE), MAX_BUDGET_TOKENS - (HANDOFF_RESERVE_TURNS + 1) * main_context_tokens)`

`SOFT_THRESHOLD_PERCENTAGE` (0-1, exclusive) sets the fixed share of the
budget (3,200,000 tokens by default); `HANDOFF_RESERVE_TURNS` (integer 1-20)
guarantees that many turns after the crossing turn at the current context
size (with a 150K context the reserve rule sets the threshold to 2,950,000
tokens, leaving at least six turns before the hard ceiling). The threshold
is evaluated on every counted response and in the `PreToolUse` hook, and
once crossed it stays crossed for the attempt, even if a compaction later
shrinks the context. A huge context whose reserve exceeds the remaining
budget trips the threshold on the next check. The token budget comes from
a per-attempt
ledger: each streamed response is first counted as a local estimate and is
replaced by the usage the backend reports (`message_delta` events) when that
arrives, so the threshold still works when the backend reports no usage; the
`token_estimate_degraded` warning marks that case. `max_budget_usd` (default
`20`) is only the SDK's client-side USD backstop and does not drive the soft
threshold. Once the soft threshold is crossed, tool enforcement
blocks general file modifications and arbitrary execution, only allowing the
model to write the handoff note (`.agent/handoff/<issue-number>.md`) and execute
read/inspection or git wrap-up commands (e.g. `git status`, `git add`, `git commit`).
Turns and `MODEL_TIMEOUT` get no soft threshold: reaching either limit gets
one best-effort same-client handoff follow-up instead.

If the token hard ceiling (`MAX_BUDGET_TOKENS` tokens, default 4000000) is
reached, the attempt is stopped unconditionally, regardless of the model and
whether or not it has handed off: every tool call is denied (the `handoff`
skill and its git commands included), the stream is interrupted and drained,
and no follow-up prompt of any kind is issued. The attempt is classified as a
model limit and the agent performs an **emergency handoff**: any uncommitted
dirty work is automatically committed, an emergency handoff note with
`reason: token_hard_ceiling` is created and committed, and the outcome is
recorded as `handoff` so that no work is lost. If zero work was accomplished
before hitting the ceiling, the attempt downgrades cleanly to `no_changes`.
The SDK's client-side USD backstop (`max_budget_usd`, default `20`) only
applies while the token budget is still below the ceiling: if it fires first,
the model gets one best-effort same-client handoff follow-up and the
emergency note carries `reason: cost_hard_limit`. A turn/time limit reached
without a cooperative handoff note likewise ends in an emergency handoff
(`turn_limit`).

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

## Model backend settings

The model backend is configured through operator environment settings that
default to Meta, so an existing `.env` with only `META_API_KEY` keeps working
with no edits:

- `MODEL_BASE_URL` (default `https://api.meta.ai`) is passed as
  `ANTHROPIC_BASE_URL`. An explicitly empty value leaves it unset, which
  selects the Anthropic API directly.
- `MODEL_API_KEY` holds the backend credential and falls back to
  `META_API_KEY` when unset. When both are set, `MODEL_API_KEY` wins. One of
  the two must be set.
- `MODEL_AUTH_MODE` selects how the credential is sent: `auth_token` sends it
  as `ANTHROPIC_AUTH_TOKEN` (Bearer, the Meta behaviour); `api_key` sends it
  as `ANTHROPIC_API_KEY` (x-api-key, the Anthropic API).
- `MODEL_STREAM_IDLE_TIMEOUT_MS` (default `60000`) is passed as
  `CLAUDE_STREAM_IDLE_TIMEOUT_MS`, the stalled-stream control.

The resolved credential is redacted from logs everywhere `META_API_KEY` was,
and profile commands never see `META_API_KEY`, `MODEL_API_KEY`,
`ANTHROPIC_API_KEY`, or `ANTHROPIC_AUTH_TOKEN`.

## Meta through the LiteLLM gateway

Meta's prompt cache lives on each backend replica and is reused reliably only
when requests carry an affinity key, `prompt_cache_key`. Meta's
Anthropic-compatible `/v1/messages` endpoint is the only protocol the CLI
speaks, and it rejects that parameter, so direct traffic gets close to 0%
cache reuse (#169). A LiteLLM gateway accepts the CLI's Messages requests,
translates them to Meta's Responses API, and derives `prompt_cache_key` from
the session ID the CLI already sends. Measured hit rates were 89–99% in the
probes and 86% across all threads of a real attempt (#171). The decision is
recorded in
[ADR 0003](../adr/0003-litellm-gateway-for-meta-prompt-caching.md).

The gateway is separate infrastructure (its own compose project). The agent
needs no extra dependency, only the backend settings above:

```env
MODEL_BASE_URL=http://litellm:4000
MODEL_API_KEY=<LITELLM_MASTER_KEY>
# MODEL_AUTH_MODE=auth_token (default, Bearer)
# MODEL_NAME=muse-spark-1.3-contributor (unchanged)
```

- `META_API_KEY` then lives only in the gateway's configuration. Remove it
  from the agent's `.env`.
- `MODEL_NAME` must match the gateway's `model_name` entry. Keeping it
  unchanged keeps the observed-model check and the pricing in
  `docker/managed-settings.json` working.
- The default `docker-compose.yml` uses the project's own network, where
  `litellm` does not resolve. Attach the agent to the network it shares with
  the gateway with the overlay file:

  ```shell
  docker compose -f docker-compose.yml -f docker-compose.litellm.yml up -d
  ```

  The overlay joins the external network named by `LITELLM_NETWORK`
  (default `main`). Set it in `.env` if the gateway uses another network.
  Plain `docker compose up` ignores the overlay and connects to Meta
  directly, as before.
- Check the cache hit rate in `token_budget_reconciled.prompt_cache` after
  the first attempt through the gateway.
- **Re-run the prompt-cache probe after every LiteLLM version bump.** The
  `prompt_cache_key` derivation is LiteLLM behaviour, not a contract, and a
  regression would silently drop the hit rate back to zero. The probe is
  `scripts/probe_prompt_cache.py` on the `codex/muse-cache-investigation-169`
  branch.
- Meta's `request-id` is not propagated through LiteLLM. Support requests to
  Meta need the upstream request IDs from the gateway's own logs.

## Alternative backend for testing

To run the agent against the Anthropic API directly instead of Meta:

```env
MODEL_BASE_URL=
MODEL_AUTH_MODE=api_key
MODEL_API_KEY=sk-ant-...
MODEL_NAME=claude-sonnet-5-5
```

- `MODEL_NAME` must be the full model ID (for example
  `claude-sonnet-5-5`), not an alias like `sonnet`. The strict
  observed-model check compares the model reported by the API with
  `MODEL_NAME` and treats any mismatch as an infrastructure error.
- `docker/managed-settings.json` only overrides pricing for the Meta model.
  Anthropic models use the CLI's built-in price table on purpose, so
  `total_cost_usd` and `MAX_BUDGET_USD` reflect Anthropic rates. Those rates
  are several times higher than Meta's.
- The token estimator constants (`token_ledger.py`, e.g.
  `THINKING_ALLOWANCE_TOKENS`) are tuned for Meta. Estimates may be less
  accurate on other backends, but measured usage still replaces them once the
  backend reports it.

## Non-destructive workspace management

The agent adheres strictly to non-destructive cleanup:
- The agent **never** deletes uncommitted files or removes branches after an attempt
  ends (`git reset --hard`, `git clean -fd`, and `git branch -D` are not run on attempt cleanup).
- Attempt branches and untracked artifacts remain intact for human review and debugging.
- **Clean workspace precondition**: When intake is running and a claim could
  happen, the agent inspects the repository working tree
  (`git status --porcelain`). If the workspace contains uncommitted or
  untracked changes, the agent blocks the claim but keeps running:
  1. Emits a warning log (`working_tree_dirty`) once per dirty episode,
     naming the first offending paths (a follow-up `working_tree_clean`
     is logged when the tree reads clean again).
  2. Sleeps the normal poll interval and keeps serving `agentctl status`,
     whose `recovery` block reports the hold with the offending paths and
     the repair instruction — no exit, no restart loop, no label changes.
  A stopped instance never inspects the tree and stays quiet.
  Cleaning the working tree is strictly the responsibility of human operators.
  Keep stray files (notably core dumps — the compose file disables them via
  `ulimits: core: 0`) out of the clone: any untracked file blocks intake.

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

`agentctl status` shows the guard as a `consecutive errors` line with the
current count, the limit, and the last attempt and success timestamps. When
the store cannot be read, status reports the counter as unavailable instead
of failing the whole snapshot.

A restart alone does **not** reset the guard.  To recover:
1. Diagnose and fix the underlying infrastructure problem (credentials,
   network, disk, or broken provenance).
2. Reset the counter through the live process (never by editing the state
   file, which would race the agent):

   ```shell
   /usr/local/bin/agentctl errors reset
   ```

   The command prints a durable request ID, accepts `--request-id` for an
   identical retry after an ambiguous connection loss, is recorded so
   `agentctl command <id>` reports it, and logs a `consecutive_errors_reset`
   event with the previous count. It applies immediately even with an
   active attempt. (`consecutive-errors reset` is an accepted alias.)
   A non-infrastructure outcome also resets the counter automatically.
3. Restart the agent if it already exited.

An operator's auto-restart policy (e.g. `restart: unless-stopped` in Compose)
cannot silently defeat this guard because the counter is persisted across
restarts: with a restart policy the restarted process is reachable, so run
the reset before its next attempt ends.
