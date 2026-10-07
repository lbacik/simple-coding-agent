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
The runtime pair is pinned to `claude-agent-sdk==0.2.163` and
`@anthropic-ai/claude-code@2.1.286`.

The model backend is configured through operator settings that default to
Meta, so an existing `.env` with only `META_API_KEY` keeps working with no
edits: `MODEL_BASE_URL` (default `https://api.meta.ai`, passed as
`ANTHROPIC_BASE_URL`; an explicitly empty value leaves it unset),
`MODEL_API_KEY` (falls back to `META_API_KEY`; when both are set,
`MODEL_API_KEY` wins — one of the two must be set), `MODEL_AUTH_MODE`
(`auth_token` sends the credential as `ANTHROPIC_AUTH_TOKEN`, the Meta
behaviour; `api_key` sends it as `ANTHROPIC_API_KEY` for the Anthropic API),
and `MODEL_STREAM_IDLE_TIMEOUT_MS` (default `60000`).

The image ships a managed-settings file (`docker/managed-settings.json`,
installed at `/etc/claude-code/managed-settings.json`) with a `modelPricing`
override for `muse-spark-1.3-contributor` at Meta's published rates ($1.25
input, $0.15 cached input, $4.25 output per million tokens), so the SDK's
`total_cost_usd` and the `max_budget_usd` backstop reflect Meta's rates
instead of the CLI's default-model rates (about 4x higher). The override row
keys (`input`, `output`, `cacheRead`, `cacheWrite`) are USD-per-million-token
rates, verified against the pinned CLI 2.1.286 bundle (its pricing-row
compiler maps exactly these four keys onto the internal per-token costs) and
the [`modelPricing` settings reference](https://code.claude.com/docs/en/settings-reference).
Meta publishes no cache-write rate and reports zero cache-creation tokens, so
`cacheWrite` is set equal to the input rate. Anthropic models have no entry
and keep the CLI price table.

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

To route Meta traffic through the LiteLLM gateway (restores prompt caching),
add the network overlay and set the gateway backend in `.env`; see
[Meta through the LiteLLM gateway](docs/agents/operator-onboarding.md#meta-through-the-litellm-gateway):

```shell
docker compose -f docker-compose.yml -f docker-compose.litellm.yml up -d
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
drains the stream before teardown; `MODEL_STREAM_IDLE_TIMEOUT_MS` (default
`60000`, passed as `CLAUDE_STREAM_IDLE_TIMEOUT_MS`) is a
separate stalled-stream control.

The model runs with `bypassPermissions`, but an SDK `PreToolUse` guard denies
model-side `git push`, `gh pr merge`, and `gh issue close`, including compound
shell commands and `git -C`. It records `Skill` pre/post events with the skill
name, subagent ID, and timestamp. Model mismatch, abort, and timeout are
infrastructure errors; `max_turns_exceeded` is reported to the completion
evaluator as a model-limit status, which maps to an incomplete attempt. The pinned SDK's
observed `ResultMessage` fields are `is_error`, `model_usage`, and
`stop_reason`; this differs from the current SDK reference field names.

### Alternative backend for testing

To run the agent against the Anthropic API directly instead of Meta:

```env
MODEL_BASE_URL=
MODEL_AUTH_MODE=api_key
MODEL_API_KEY=sk-ant-...
MODEL_NAME=claude-sonnet-5-5
```

`MODEL_NAME` must be the full model ID (for example `claude-sonnet-5-5`),
not an alias like `sonnet`: the strict observed-model check compares the
model reported by the API with `MODEL_NAME` and treats any mismatch as an
infrastructure error. `docker/managed-settings.json` only overrides pricing
for the Meta model; Anthropic models use the CLI's built-in price table on
purpose, so `total_cost_usd` and `MAX_BUDGET_USD` reflect Anthropic rates,
which are several times higher than Meta's. The token estimator constants
(`token_ledger.py`, e.g. `THINKING_ALLOWANCE_TOKENS`) are tuned for Meta, so
estimates may be less accurate on other backends — but measured usage still
replaces the estimates once the backend reports it.

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

## Releasing

1. Bump `version` in `pyproject.toml` and merge the change to `main`.
2. Tag that commit `v<version>` (for example `v0.3.0`) and push the tag.

CI runs `scripts/check_tag_version.py` on every `v*` tag push and fails when the
tag is not exactly `v` + the `pyproject.toml` version, naming both values. The
agent never creates or pushes tags. The running version is reported in the
startup `provenance_verified` event, each attempt's `attempt.json`
(`agent_version`) and `agentctl status`.

## Development

```shell
python3 -m pip install -e '.[dev]'
python3 -m pytest
```
