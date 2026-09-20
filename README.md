# Simple Coding Agent

An unattended Python agent that attempts eligible GitHub implementation issues
for one configured repository. It provides validated runtime configuration,
the target-repository profile boundary, and a model-execution boundary. The
driving process still owns GitHub calls and all publication.

## GitHub issue trust boundary

The tracker selects only OPEN issues in `TARGET_REPO` that are labelled
`ready-for-agent`, unassigned, have a non-empty body, and have no open native
GitHub blockers. It orders eligible issues FIFO by creation date and rechecks
eligibility immediately before assigning the authenticated GitHub identity.
That check-then-set claim assumes one process; it is not a distributed lock.

Applying `ready-for-agent` is the human trust boundary: only repository
collaborators with write access may apply it. The issue body is passed to the
agent verbatim; the tracker intentionally does not infer dependencies from
prose or filter issue text for prompt injection.

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

## Model execution

The execution boundary dispatches `/implement` with the issue body as its
specification through `ClaudeSDKClient`. It uses the pinned upstream
`implement`, `tdd`, `code-review`, and `codebase-design` skills, with
`max_turns=60`, `max_budget_usd=5`, and `MODEL_TIMEOUT`. The dollar limit is a
non-authoritative circuit breaker for Meta. A timeout interrupts the client and
drains the stream before teardown; `CLAUDE_STREAM_IDLE_TIMEOUT_MS=60000` is a
separate stalled-stream control.

The model runs with `bypassPermissions`, but an SDK `PreToolUse` guard denies
model-side `git push`, `gh pr merge`, and `gh issue close`, including compound
shell commands and `git -C`. It records `Skill` pre/post events with the skill
name, subagent ID, and timestamp. Model mismatch, abort, and timeout are
infrastructure errors; `max_turns_exceeded` is incomplete. The pinned SDK's
observed `ResultMessage` fields are `is_error`, `model_usage`, and
`stop_reason`; this differs from the current SDK reference field names.

`tests/test_model_execution_smoke.py` is opt-in and makes a paid request only
when `RUN_MODEL_SMOKE=1` is set. It requires `MODEL_SMOKE_REPOSITORY` (the
disposable fixture path) and `MODEL_SMOKE_ISSUE_BODY`, in addition to the
normal operator environment.

## Repository profile

The target repository supplies
`docs/agents/simple-coding-agent-profile.yml`. See
[`examples/simple-coding-agent-profile.yml`](examples/simple-coding-agent-profile.yml).
`setup` and `check` are required, each as a command string or non-empty list of
command strings. `base_branch` defaults to `main`, `timeout` to 300 seconds,
and `setup_timeout` to 120 seconds. `env` maps environment-variable names to
string values. Missing or invalid profiles are infrastructure errors for the
lifecycle, never silent success.

## Durable attempt state

While an implementation attempt is active, its checkpoint is stored at
`$DATA_DIR/state/attempt.json`. It contains the issue number, branch, current
lifecycle phase, and UTC `started_at`/`updated_at` timestamps. Checkpoints are
written by replacing a temporary file, so an interrupted write leaves the last
valid checkpoint intact. The checkpoint is removed only after lifecycle cleanup;
an absent file means there is no active attempt, while malformed content is an
infrastructure error. `started_at` remains stable for the whole attempt and is
the attempt-specific key used when deduplicating its result comment.

## Development

```shell
python3 -m pip install -e '.[dev]'
python3 -m pytest
```
