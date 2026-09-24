# Simple Coding Agent

An unattended Python agent that attempts eligible GitHub implementation issues
for one configured repository. It provides validated runtime configuration,
the target-repository profile boundary, and a model-execution boundary. The
driving process owns GitHub calls, publication, cleanup, and polling.

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

## Local installation

Install Python dependencies and the package in editable mode, then install the
pinned upstream skill bundle:

```shell
python3 -m pip install -e '.[dev]'
bash scripts/install-skills.sh
```

Copy `.env.example` to `.env` and fill in your credentials:

```shell
cp .env.example .env
# edit .env: set GITHUB_TOKEN, META_API_KEY, TARGET_REPO
```

Start the single-process lifecycle with:

```shell
python -m simple_coding_agent
```

It claims one issue at a time. On every terminal outcome, it posts the result,
removes only its `ready-for-agent` label and assignment, then removes the
checkpoint. If the queue is empty, it sleeps for `POLL_INTERVAL` before polling
again.

## Docker Compose operation

Build the image and start the long-running service (persistent data under the
`agent_data` named volume):

```shell
cp .env.example .env
# edit .env: set GITHUB_TOKEN, META_API_KEY, TARGET_REPO
docker compose up --build
```

The service runs as a single-instance, sequential-worker process.  Do not
scale it beyond one replica.  The `agent_data` volume preserves the repository
clone, state files, and logs through container replacement.  Installed skills
are baked into the image at build time and are therefore also preserved.

For a one-shot run (useful for testing or manual invocation):

```shell
docker compose run --rm agent
```

Lifecycle and persistence semantics are the same for both launch paths.
`restart: unless-stopped` in `docker-compose.yml` ensures the service
recovers from transient failures, but the persisted `consecutive_errors.json`
guard still trips at `MAX_CONSECUTIVE_ERRORS`; a restart alone does not reset
it.  See [docs/agents/operator-onboarding.md](docs/agents/operator-onboarding.md)
for recovery procedures and full onboarding guidance.

### Controlling the running instance

Open a shell in the intended running container and use the installed control
CLI at its absolute path, checking `TARGET_REPO` before any mutating command:

```shell
/usr/local/bin/agentctl status
/usr/local/bin/agentctl stop
```

Container selection is the instance selector. The control socket is
container-local (`/run/simple-coding-agent/control.sock`, owner `agent`,
modes `0700`/`0600`); accepted commands persist on that instance's `/data`
volume through container replacement. Never issue commands from a second
agent process or a one-shot `docker compose run` container. Full procedures
are in [docs/agents/operator-onboarding.md](docs/agents/operator-onboarding.md).

## Operating limits and evidence

Transient GitHub and push operations use `MAX_RETRIES` attempts with exponential
backoff (2 seconds, capped at 30 seconds). GitHub `Retry-After` values are
honoured up to five minutes. `PUBLISH_TIMEOUT` is a single deadline covering
the push and pull-request path. Model implementation work is never replayed as
a retry; an exhausted transient failure is an `infrastructure_error`.

The process persists `$DATA_DIR/state/consecutive_errors.json`. Each distinct
attempt that ends in `infrastructure_error` increments its count; every other
outcome resets it. The attempt identity makes startup reconciliation idempotent.
At `MAX_CONSECUTIVE_ERRORS` the process emits a structured error and exits with
status 1. A restart alone does not reset this guard. After correcting the
underlying infrastructure problem, an operator may remove that state file to
reset the guard, or allow a later non-infrastructure outcome to reset it.

Before model execution, the agent verifies `agent-installer list --json` for
the pinned upstream skill commit, installed skill hashes, and valid HOME links,
plus the exact pinned SDK and CLI versions. Any mismatch blocks model execution.
The process emits credential-redacted JSON Lines with `timestamp`, `level`,
`event`, `issue_number`, `phase`, and `detail`. Attempt evidence is retained
without automatic cleanup under `$DATA_DIR/logs/<issue>/<timestamp>/`, including
outcome metadata and separate setup, baseline-check, and final-check output.
Available SDK skill provenance and token metadata are archived too; any SDK
dollar estimate is explicitly non-authoritative.

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
infrastructure errors; `max_turns_exceeded` is reported to the completion
evaluator as a model-limit status, which maps to an incomplete attempt. The pinned SDK's
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

Set `PROFILE_PATH` to load the profile from a path outside the target
repository instead, such as alongside the agent's own `Dockerfile`. When unset
or empty, the default in-repo path above is used.

## Durable attempt state

While an implementation attempt is active, its checkpoint is stored at
`$DATA_DIR/state/attempt.json`. It contains the issue number, branch, current
lifecycle phase, and UTC `started_at`/`updated_at` timestamps. Checkpoints are
written by replacing a temporary file, so an interrupted write leaves the last
valid checkpoint intact. The checkpoint is removed only after lifecycle cleanup;
an absent file means there is no active attempt, while malformed content is an
infrastructure error. `started_at` remains stable for the whole attempt and is
the attempt-specific key used when deduplicating its result comment.

At process startup, before a new issue can be claimed, the lifecycle reconciles
one active checkpoint. A checkpoint from `claimed` or `setup` becomes an
`infrastructure_error` with the standard comment and cleanup. A
`model_running` checkpoint never resumes the SDK session: committed work is
pushed as incomplete work without a pull request; no commits becomes an
infrastructure error. `pushing` and `publishing` retry the idempotent
publication path, which verifies the branch and existing pull request before
writing a result comment. Cleanup remains ordered as comment, queue label,
agent assignee, then checkpoint deletion, so a later restart can finish an
interrupted cleanup without duplicating a result.

## Development

```shell
python3 -m pip install -e '.[dev]'
python3 -m pytest
```
