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
authoritative billing.  The only model used is `muse-spark-1.3-contributor`;
no fallback is configured.  Exceeded budget appears as a model-limit outcome
(incomplete attempt), not an infrastructure error.

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
