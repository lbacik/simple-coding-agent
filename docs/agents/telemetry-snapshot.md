# Telemetry snapshot contract (schema v1)

The agent owns a versioned, read-only **telemetry snapshot**: a point-in-time
report a managed instance gives about itself — its control state, the current
(or last finished) implementation attempt's token, budget and context figures,
and its most recent events. A dashboard only reads it and tolerates versions
it does not fully know. Terms **telemetry snapshot** and **figure basis** are
defined in `CONTEXT.md`; the transport decision (serve through `agentctl`
over the control socket, not HTTP) is
[ADR 0006](../adr/0006-telemetry-through-agentctl-not-http.md). The shape was
decided in issue #224 and the command mechanics in issue #225.

Scope note: this document publishes the contract so a dashboard can build and
test against it. The live `telemetry` socket operation and `agentctl
telemetry` subcommand are a separate agent-side change; until they land, the
synthetic examples under `examples/telemetry/` are the conformance target.

## Top-level fields (schema version 1)

```jsonc
{
  "schema_version": 1,
  "generated_at": "2026-09-30T08:01:00Z",  // when this reply was serialised
  "sequence": 412,                          // +1 per published telemetry update; resets with the process
  "identity": {
    "repository": "owner/name",             // TARGET_REPO
    "agent_version": "0.4.5",
    "sdk_version": "0.2.163",               // from provenance evidence
    "cli_version": "2.1.286",
    "process_started_at": "2026-09-30T08:00:00Z"
  },
  "control": { "..." },                     // the `agentctl status` payload (same builder)
  "current_attempt": { "..." } | null,      // null = no attempt is active
  "last_finished": { "..." } | null,        // null = no finished attempt known
  "recent_events": [ "..." ],
  "truncated": false
}
```

- `control` reuses the `agentctl status` payload builder verbatim, so `status`
  and `telemetry` cannot drift (`agent_version`, `repository`, `intake`,
  `recovering`, `active_attempt` including `issue_title`, `pending_command`,
  `pending_next_issue`, `stop_plan`, `recovery`, `handoff` with phase and
  deadlines, `consecutive_errors`, `commands`), plus an additive
  `control_observed_at` timestamp saying when that section was built.
- `current_attempt` is present from claim until the attempt is finalised;
  after model execution ends its figures stay frozen at their final values.
  Its fields: attempt/issue identity (`attempt_id`, `issue_number`,
  `observed_at`), `budget` (used/measured/estimated split, hard ceiling, soft
  threshold with rule and crossing, hard-ceiling flag), `tokens_by_category`
  (measured input/cache-read/cache-creation/output), `turns` with `max`,
  `continuation` (`index`, `max`), `cache.main_hit_rate` with `counted`
  responses, `context` (latest and peak main-thread context, window,
  auto-compact threshold, compactions, up to 20 running-first subagents),
  `provider_rate_limit` (always `unavailable` in v1), and
  `elapsed_seconds`/`timeout_seconds`. USD cost is intentionally out of v1.
- `last_finished` (`attempt_id`, `issue_number`, `issue_title`, `outcome`,
  `finished_at`, `budget_used`, `turns`, `pull_request: {number, url} |
  null`) is held in memory, updated at finalisation, and seeded at startup
  from disk, so it survives restarts. It is shown alongside `current_attempt`
  too. Its `budget_used` and `turns` figures use the `measured` basis (or
  `unavailable` when no token budget was reported); archives written before
  the title and pull-request fields existed seed an `"unknown"` title and a
  null pull request.
- `recent_events` is the last up to 50 redacted log events, newest last, each
  `{timestamp, level, event, issue_number, phase, text}` with `text` truncated
  to 300 characters. High-volume `limits_checked` and `tool_call` events are
  excluded. The buffer is process-local and starts empty after a restart.
- `generated_at` (serve time) together with `sequence` and
  `current_attempt.observed_at` (publication time) lets the dashboard tell
  "agent answered" from "telemetry moving". A restart is visible as a new
  `process_started_at` with the sequence reset.

## Figure object and basis

Every numeric figure in `current_attempt` / `last_finished` is an object:

```jsonc
{ "value": 12345, "basis": "measured" }
{ "value": null, "basis": "unavailable", "reason": "not exposed through the gateway" }
```

`basis` is one of `measured` (reported by the provider), `estimated`
(computed by the agent), `configured` (set by the operator), `sdk_assumed` (a
default the runtime assumes for an unknown model), or `unavailable` (no
trustworthy source). An unavailable figure is never shown as zero:
`unavailable` always has `value: null` and may carry a short `reason`. Plain
fields (ids, timestamps, enums, booleans) are bare values, not figures.

## Missing field vs null section

- A **missing** field means unavailable: the agent had no trustworthy source
  for it. Dashboards render it as unknown.
- A **null section** (`current_attempt`, `last_finished`) means not
  applicable, not "unavailable": `current_attempt: null` is an idle agent,
  `last_finished: null` is no finished attempt known. Likewise a null
  `pull_request` inside `last_finished` means no PR, and null
  `comment_confirmed`/`pull_request` inside `control.recovery.publication`
  or null `workspace_dirty` inside `control.recovery` mean "not observed
  yet in this process": while an attempt is publishing or held in recovery,
  telemetry carries the publication progress and workspace-dirty state the
  lifecycle itself last observed (no network or git I/O on the telemetry
  path), and `agentctl status` keeps its live lookups for the same fields.

## Versioning rules

`schema_version` is an integer **major**. Additive changes (new fields, new
enum values for `basis`, `phase`, or events) do **not** bump it; removing,
renaming, or changing the meaning of a field does. Dashboard rules:

- Ignore unknown fields.
- Render an unknown `basis` value as `estimated`.
- A `schema_version` higher than supported shows the instance as "unsupported
  schema vN" using only the stable fields below.

Stable across all versions: `schema_version`, `generated_at`,
`identity.repository`, `identity.agent_version`.

## Size bound, trimming, and `truncated`

The serialised snapshot is capped at **256 KiB** (a quarter of the control
socket's 1 MiB reply limit). Over the cap the agent trims `recent_events`
(oldest first), then `context.subagents`, and sets `truncated: true`.
Typical size is a few tens of KiB.

## `agentctl telemetry --json`

The dashboard reads one instance with (no TTY, 5 s dashboard-side timeout,
at most one in-flight call per instance):

```shell
docker exec --user agent <container-id> /usr/local/bin/agentctl telemetry --json
```

`agentctl telemetry` uses a 3 s connect+read timeout so it always answers
inside that budget. On success it prints the bare snapshot to stdout and
exits 0. In v1 there is no human format (operators keep using `status`), and
`status` does not get `--json`. Telemetry is always on with no setting: the
operation is read-only on the existing container-local socket.

On failure it prints `{"ok": false, "error": {"code", "message"}}` to
**stdout** (human text may also go to stderr):

| `code` | meaning | exit |
|---|---|---|
| `agent_unreachable` | socket missing, connection refused, or timeout | 2 |
| `unsupported_operation` | agent answered "Unknown operation" | 1 |
| `snapshot_failed` | agent answered but could not build the snapshot, or the reply was unparsable/oversized | 1 |

Version skew: `agentctl` ships in the same image as the agent, so an older
instance shows up as argument parsing rejecting the `telemetry` subcommand
(exit 2, usage on stderr, **empty stdout**). Dashboard rule: stdout that is
not a JSON object means "telemetry not supported by this agent version"; a
JSON error object is handled by its `code`.

## Discovery label

Every container built from the agent image carries the label
`io.github.lbacik.simple-coding-agent.role="agent"`, set in the image (not in
compose), so a dashboard discovers managed instances without operator
configuration:

```shell
docker ps --filter "label=io.github.lbacik.simple-coding-agent.role=agent"
docker inspect -f '{{ index .Config.Labels "io.github.lbacik.simple-coding-agent.role" }}' <container-id>
```

## Synthetic examples

`examples/telemetry/` holds synthetic schema-v1 snapshots a dashboard can
build and test against before any agent serves telemetry:

- `snapshot-idle-with-last-finished.json` — idle agent, no active attempt,
  last finished attempt present.
- `snapshot-attempt-in-progress.json` — attempt in progress with a partly
  estimated budget and running subagents.
- `snapshot-soft-threshold-handoff.json` — soft threshold crossed with
  handoff in progress.
- `snapshot-recovery-hold.json` — recovery hold with its next action.
- `snapshot-truncated.json` — trimmed snapshot with `truncated: true`.
- `error-agent-unreachable.json`, `error-unsupported-operation.json`,
  `error-snapshot-failed.json` — one machine-readable error reply per code.

The examples are synthetic: no data was captured from live instances. The
dashboard repository is public and copies them. `tests/test_telemetry_examples.py`
parses every example and checks the stable fields plus the figure invariant
(`value: null` exactly when `basis` is `unavailable`).
