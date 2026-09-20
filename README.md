# Simple Coding Agent

An unattended Python agent that attempts eligible GitHub implementation issues
for one configured repository. This initial slice provides the validated runtime
configuration and target-repository profile boundary; it does not start an SDK
session, call GitHub, or publish changes.

## Operator environment

Copy `.env.example` and set `GITHUB_TOKEN`, `META_API_KEY`, and `TARGET_REPO`
(`owner/repo`). `DATA_DIR` defaults to `~/.simple-coding-agent/`; its default
clone location is `$DATA_DIR/repo/`. Other defaults are `POLL_INTERVAL=60`,
`LOG_LEVEL=INFO`, `MODEL_TIMEOUT=3600`, `PUBLISH_TIMEOUT=120`,
`MAX_RETRIES=3`, and `MAX_CONSECUTIVE_ERRORS=3`.

Use a fine-grained GitHub PAT limited to the configured repository with only
`contents:write`, `pull-requests:write`, and `issues:write`. Configure provider
spending controls in Meta: SDK `max_budget_usd=5` is only a circuit breaker.
The agent uses only `muse-spark-1.3-contributor`; no fallback model is set.
The runtime pair is pinned to `claude-agent-sdk==0.2.156` and
`@anthropic-ai/claude-code@2.1.276`.

`AGENT_TRUST_PROJECT_SETTINGS` is operator-only and defaults to `false`. When
false, only user settings load. When true, hooks are disabled and strict MCP
configuration is used, but a target repository's `apiKeyHelper`, settings
`env`, and project-skill `allowed-tools` still execute without a trust dialog.
Treat that as residual exposure and enable the option only for a trusted
repository.

Review findings of severity `must-fix` block by default. Set
`REVIEW_BLOCKING_SEVERITIES` explicitly to an empty value to select the
test-only review gate; `suggestion` remains advisory unless explicitly added.

## Repository profile

The target repository supplies
`docs/agents/simple-coding-agent-profile.yml`. See
[`examples/simple-coding-agent-profile.yml`](examples/simple-coding-agent-profile.yml).
`setup` and `check` are required, each as a command string or non-empty list of
command strings. `base_branch` defaults to `main`, `timeout` to 300 seconds,
and `setup_timeout` to 120 seconds. `env` maps environment-variable names to
string values. Missing or invalid profiles are infrastructure errors for the
lifecycle, never silent success.

## Development

```shell
python3 -m pip install -e '.[dev]'
python3 -m pytest
```
