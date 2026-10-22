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
`env` (a string→string map), and `model_instructions` (a free-form string of
repository-provided guidance for the model; omit it or leave it blank when
there is nothing to add — a non-string value is rejected).

The `check` commands are the completion gate and are also shown to the model
in its starting prompt, in profile order, alongside `model_instructions`
(when non-blank) and the gate's environment-variable names (never values).
Operator credentials and profile `env` values are redacted from that listing.

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

## Pull-request auto-merge (`PR_AUTO_MERGE`)

By default (`PR_AUTO_MERGE=false`) the agent only opens a pull request for a
complete implementation and leaves merging to a human.  Set
`PR_AUTO_MERGE=true` to let the orchestrator (never the model) ask GitHub to
merge that PR right after creating it, or `PR_AUTO_MERGE=label` to merge
only issues explicitly opted in with the `agent-auto-merge` label (see the
trust policy below).  Any other value fails at startup with a configuration
error.  `PR_MERGE_METHOD` (`merge`, `squash`, or `rebase`; default `merge`)
selects the method for both paths:

1. It first enables GitHub's native auto-merge on the PR, so GitHub merges
   it once the base branch's protection rules are met.  Deciding when it is
   safe to merge stays with the repository's branch protection.
2. If GitHub refuses because the PR is already mergeable (nothing is
   pending), it merges the PR directly with the same method.

Merging is a follow-up to publication and never redefines completion: an
attempt whose PR was created stays `complete` even if the merge step fails,
and the issue result comment reports what the step did (`auto-merge enabled
(<method>)`, `merged`, or `auto-merge failed: <reason>`).  The existing
`Closes #N` in the PR body closes the issue when the PR merges.

In `label` mode the merge step runs only when the implementation issue
carried `agent-auto-merge` at claim time, and goes through the same two
paths with the same `PR_MERGE_METHOD`.  A complete implementation of an
issue without the label still gets its PR but no merge-related GitHub calls,
and the result comment reports `auto-merge skipped (issue not labelled
`agent-auto-merge` at claim time)`.  A label added while the attempt runs,
or after it finishes, never enables that attempt's merge — including when a
crashed attempt is recovered, which keeps the claim-time decision.

Prerequisites:

- Turn on **Allow auto-merge** in the target repository's settings;
  otherwise every merge step fails and is reported in the result comment.
- Configure branch protection on the base branch with the required status
  checks (and required reviews if a human must still approve): without
  protection, auto-merged PRs land with no human review.
- No new token scope is needed: the existing PAT scopes (`contents: write`,
  `pull-requests: write`) are enough; issue comments already use
  `issues: write`.

Auto-merge is operator-only on purpose: the repository profile must not
carry it (an unknown profile key is refused), because the profile lives in
the target repository and an auto-merged PR could otherwise edit the profile
to grant itself more trust.  See
[ADR 0004](../adr/0004-pr-auto-merge-operator-setting.md).

## Conflict requeue (`MAX_CONFLICT_ATTEMPTS`)

When a published pull request falls behind the base branch and reports
merge conflicts, the issue behind it can come back to the queue on its
own. With `MAX_CONFLICT_ATTEMPTS` above `0` (default `0`, which disables
the path entirely — no pull-request query runs), every poll reads the
repository's open pull requests in one paged query and adds the
`merge-conflict` label to an otherwise ordinary issue (open, unassigned,
non-empty body, no open blockers, none of `ready-for-agent`,
`round-finished`, `ready-for-human`, `merge-conflict`) whose
`agent/issue-N` branch reports `CONFLICTING`. The issue is eligible in
that same poll, claimed before automatic continuations and ordinary
work, and each detection emits a `conflict_requeue_detected` event naming
the issue, the pull request, and the head sha. You may also apply the
label manually with the same effect.

The conflict attempt that follows only brings the published pull request
up to date — no model call runs, no new work is done. It rebases the
published branch onto the base, runs setup plus one profile check, and
force-pushes only when both are green, re-enabling auto-merge per
`PR_AUTO_MERGE`. Every conflict attempt removes `merge-conflict` and the
assignee, never adds a queue label, and posts a result comment stamped
with a `<!-- conflict-attempt: ... -->` marker. If the entry check finds
anything other than an open, non-draft, `CONFLICTING` pull request
against the profile's base branch, the attempt ends `skipped`.

The same setting bounds the attempts: N consecutive failures on one issue
(an unresolved conflict, a red check, an infrastructure error, or a push
that leaves the remote head unchanged) escalate instead of requeueing.
The attempt whose failure reaches the budget posts the handoff in its own
result comment — the `k/N` standing, the failure tail or reason in a
collapsed section, a pull-request link, and a line telling you to remove
`ready-for-human` to grant N more attempts — and labels the issue
`ready-for-human` (created when missing), emitting a `conflict_escalated`
event. Every conflict result carries its `k/N` standing. While the label
is present the detector will not relabel the issue and the queue will not
claim it, even with a hand-applied `merge-conflict`; the agent never
removes the label. Remove `ready-for-human` to hand the issue back: the
escalation comment stops the streak, so the next poll requeues with the
full budget and N more failures are needed to escalate again. The budget
is enforced at escalation, never at claim time. See
[ADR 0007](../adr/0007-conflict-attempt-budget.md).

Like the other budgets, the setting is operator-only on purpose: it lives
in `.env`, never in the repository profile (an unknown profile key is
refused), so an agent-written PR cannot raise its own conflict budget.

## Worker pool with different logins (`PEER_LOGINS`)

When several instances with different GitHub logins work one queue, set
`PEER_LOGINS` to the comma- or space-separated logins of the other workers
(empty by default, which keeps today's single-worker behaviour). The
worker's own login may appear in the list and is ignored, so one value
serves every worker. A shared account needs no peers; peers are for
different logins.

With peers configured, one set — the worker's own login plus its peers —
drives the cross-worker trust: a peer's attempt-result and handoff-note
comments enter a continuation's starting prompt exactly like the worker's
own; a peer's handoff/incomplete result comment counts as the historical
continuation signal (so `continuation_branch_missing` is reported across
workers); and the automatic-continuation streak counts result comments of
all workers, newest first, stopping at the latest human requeue whichever
worker wrote it — `MAX_AUTO_CONTINUATIONS` stays a per-issue budget, and
the conflict-attempt budget uses the same set.

Unchanged: a pull request found by branch name is adopted without an author
check; any worker may claim a `round-finished` issue another worker handed
off (no worker affinity); `is_self_assigned`, `recover_claim`,
`release_attempt`, and `release_handoff` stay per worker. Human
collaborators other than the issue author are never trusted.

## Shared-login worker pool (`WORKER_ID`)

By default a worker claims by GitHub login alone, which assumes one process
per login. To run several workers on one login, run one container per worker
— each with its own `DATA_DIR` (which holds that worker's durable checkpoint)
— and set a distinct `WORKER_ID` per worker. The value must use `[a-z0-9-]`,
at most 32 characters (the `claimed-by:<worker-id>` marker label limit is 50
characters); anything else fails at startup with a configuration error. When
`WORKER_ID` is unset, behaviour is exactly as before and no `claimed-by:*`
label is ever written.

With `WORKER_ID` set, a claimant is a (login, worker id) pair. After
assigning, the worker adds its `claimed-by:<worker-id>` marker label (created
in the repository when missing, like the other worker-applied labels), then
re-reads after the settle delay and yields when it sees another assignee or
another worker's marker. Yielding removes only the loser's own marker and
leaves the shared assignment while the winner's marker exists; when both
racers yield, the last one out also removes the now-unmarked assignment, so
the issue returns to the runnable queue for a later poll. An issue carrying
any `claimed-by:*` label is never eligible on any claim path. Release is
ordered per remote re-read: queue labels first, then the worker's own marker,
then the login's assignment only when no `claimed-by:*` marker remains.

A login assigned with only another worker's marker counts as claimed by
another assignee everywhere: priority targets for it end `not fulfilled` and
fall back to FIFO, and recovery retry holds for operator repair naming the
blocking marker. Restart recovery removes the worker's own marker as part of
release.

A lost `/data` directory or a changed `WORKER_ID` leaves the marker and the
assignment for a human — there is no startup sweep and no TTL. To repair by
hand, remove the stale label and, if the issue should requeue, the assignment:

```shell
gh issue edit <number> --remove-label "claimed-by:<old-worker-id>"
gh issue edit <number> --remove-assignee <login>
```

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

The `agent-auto-merge` label sits on the same trust boundary: only
collaborators with **write access** may apply it.  Once applied (with
`PR_AUTO_MERGE=label` configured), the agent merges the complete
implementation's PR with no human review unless branch protection requires
one, so review the issue — and the risk of landing its solution unreviewed —
before opting it in.  The label counts only when the issue carries it at
claim time; adding it mid-attempt never enables that attempt's merge.

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
`round-finished`. **By default, continuation requires a human decision**:
review the note and comment, then re-apply `ready-for-agent` to requeue the
issue (this also clears `round-finished`), or leave it labelled
`round-finished` to end the attempt sequence.

A continued attempt resumes from the Handoff note file on its branch, not
from the issue's comment history: the starting prompt excludes every
historical handoff result comment (including its note payload) and strips
machine bookkeeping from the generated result comments that remain, while
ordinary human guidance still reaches the model verbatim in order. Filtering
is presentation-only — continuation detection, budget counting, and
publication recovery still read the full history. Each cooperative handoff
replaces the previous note with a self-contained record (current state,
still-relevant decisions and constraints, unresolved review findings,
remaining work, evidence), so a continuation never has to reconcile obsolete
instructions from older rounds.

`MAX_AUTO_CONTINUATIONS` (default `0`) optionally lets the agent continue a
`round-finished` issue on its own, up to N automatic continuations in a row,
claimed ahead of new `ready-for-agent` work. A human requeue resets the
count and grants another N rounds; removing `round-finished` stops the
sequence at any time. Every handoff result comment then reports where the
issue stands (`k/N used`, and whether a human requeue is now needed), and
each automatic claim emits an `auto_continuation_claimed` event. A routine
handoff needs no individual result comment: the single shared handoff
comment carries the standing, reconciled with newer ordinary results, and
the published count is computed once and retried unchanged. `HANDOFF_COMMENT_MODE`
(default `link`) selects this one-comment experience: the first handoff
creates the short shared comment and later routine handoffs update its
hidden record without posting an individual result comment or copying the
note body into GitHub, while the visible branch link keeps showing the
latest note. Set `HANDOFF_COMMENT_MODE=verbose` to retain the historical
per-attempt handoff result comments (including the note body) for
troubleshooting. Both modes use identical shared bookkeeping and
continuation context, both post explicit deduplicated notifications when
human action is needed (an exhausted automatic-continuation budget, an
actionable failure, or an unconfirmed shared record — always with the
current standing), and both keep normal completion reporting; a suppressed
routine comment never blocks release once the shared record and revisions
are confirmed, and a failed or ambiguous publication is never reported as
confirmed. Switching modes between attempts creates no second shared
comment, resets no budget, duplicates no recorded attempt, and leaves
existing comments intact. Like the continuation budget, the setting is
operator-only: it lives in `.env`, never in the repository profile (an
unknown profile key is refused), so an agent-written PR cannot change it.
Older issues
without shared metadata count their historical result comments instead;
duplicate, unreadable, or mismatched shared metadata never grants a fresh
budget — the issue stays queued out for operator repair. The setting
is operator-only on purpose: it lives in `.env`, never in the repository
profile (an unknown profile key is refused), so an agent-written PR cannot
raise its own round limit. See
[ADR 0005](../adr/0005-automatic-continuation-budget.md).

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

### Fleet dashboard discovery

Every container built from the agent image carries the label
`io.github.lbacik.simple-coding-agent.role="agent"`. The label is set in the
image (`Dockerfile`), not in compose, so any container built from the image
is discoverable without operator configuration — visible with `docker
inspect` on the container. A fleet dashboard lists managed instances with:

```shell
docker ps --filter "label=io.github.lbacik.simple-coding-agent.role=agent"
```

and reads one instance's telemetry snapshot with:

```shell
docker exec --user agent <container-id> /usr/local/bin/agentctl telemetry --json
```

No port is published and no token is needed. The snapshot contract (schema
v1) is published in [telemetry-snapshot.md](telemetry-snapshot.md); its
synthetic examples live under `examples/telemetry/`.

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
  cleanup, and accounting steps. It never reruns the model. A published
  handoff is confirmed from the shared handoff record and verified
  revisions even when its individual result comment is missing, and the
  retry resumes the outstanding release and cleanup without changing the
  confirmed outcome. Ambiguous
  remote state (including an unreadable shared handoff record, or a newer
  record that supersedes the retained attempt), conflicting human changes,
  and persistent failures keep the
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
