# Live token usage probe (Meta backend)

PROTOTYPE, disposable. Asset for ticket #129 (map #123, origin #119). It checks, against a live backend, the open points in `docs/research/live-token-sources.md` (branch `research/live-token-sources`).

- Backend: `muse-spark-1.3-contributor` at `https://api.meta.ai`.
- SDK: `claude-agent-sdk==0.2.156`, with the bundled CLI 2.1.276.
- Date: 2026-09-30.
- Runs: two, each a 6-response main loop (Read, Bash, Agent → one `line-counter` subagent, TaskOutput, final answer).
- CLI-priced cost: $0.52 and $0.51.

Run: `PROBE_OUT=runN uv run python prototype-live-token-probe/probe.py`. Credentials are read from `prototype-meta-skills-probe/.env`.

- `run1/` logs PostToolUse only.
- `run2/` also logs PreToolUse.

Each run directory holds `events.jsonl` (timeline), `summary.json`, and the copied CLI transcripts.

## Findings

### 1. `StreamEvent` usage

- `message_start.message.usage` is `{"input_tokens": 0, "output_tokens": 0}` on every response.
- `message_delta.usage` is **non-zero** on every response. There is exactly one `message_delta` per response, and it always has a `stop_reason`. It carries `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens` and `output_tokens_details.thinking_tokens`.
- No stream events were emitted for the subagent: `parent_tool_use_id` was null on all of them.

**Whether `input_tokens` includes cached tokens: not observable in these runs.** Meta reported `cache_read_input_tokens: 0` on every response of both short runs. The earlier 22-turn probe (`prototype/meta-skills-probe`, `full-run.log`) settles it indirectly. There, `inputTokens` (93047) is less than `cacheReadInputTokens` (316575), which is only possible if `input_tokens` **excludes** cached tokens (Anthropic semantics). Cache reads on Meta evidently need longer or repeated prefixes than a 6-response run produces.

### 2. `get_context_usage()["apiUsage"]` mid-run

- It works when called from inside a PostToolUse hook callback: no deadlock, and it returned in milliseconds.
- It returns the **last main-loop response's** usage. That is per-response, not cumulative, and it matches that response's `message_delta` exactly.
- While the subagent runs, it keeps showing the main loop's last response.
- `totalTokens` equalled `apiUsage.input_tokens` at every sample.

### 3. Main transcript vs hooks: **not usable for live counting**

- Transcript entries do carry the merged usage and a non-null `stop_reason`. This holds for the main transcript and, contrary to the research note, for the subagent transcript too.
- **They are not on disk when the PostToolUse for that same response fires.**
  - At the first PostToolUse the transcript held 0 assistant entries.
  - At each later PostToolUse it held only the entries of *earlier* responses.
  - The entry for the current `tool_use` was never found, in either run.
- The transcript lags by at least one response, even though the entries' own `timestamp` fields predate the hook.

### 4. Reconciliation

| | run1 | run2 |
|---|---|---|
| Σ `message_delta` (main loop): input / output | 87401 / 2172 | 87245 / 2005 |
| `ResultMessage.usage`: input / output | 87401 / 2172 | 87245 / 2005 |
| `model_usage`: input / output | 90528 / 2756 | 90345 / 2416 |
| Subagent share (`model_usage` − Σ delta) | 3127 / 584 | 3100 / 411 |
| Σ subagent transcript entries | 3127 / 584 | 3100 / 411 |
| `task_notification.usage.total_tokens` | 2084 | 1901 |
| `total_cost_usd` | 0.52154 | 0.512125 |

- The summed deltas equal `ResultMessage.usage` **exactly**.
- Main loop plus subagent (from its transcript) equals `model_usage` **exactly**. No auxiliary calls showed up.
- `task_notification.total_tokens` is the subagent's **last response** (input + output: 1711 + 373 = 2084, 1686 + 215 = 1901). It is not its cumulative spend, so it undercounts subagent input by roughly the number of subagent responses.
- The cost matches the CLI's unknown-model default rate: 90528 × $5 + 2756 × $25 per million tokens = $0.52154.

### 5. Ordering: `message_delta` arrives **after** PreToolUse

Run2 timeline, main loop, for each tool call:

`AssistantMessage(tool_use)` → **PreToolUse** (+2..9 ms) → **`message_delta`** (+2 ms) → tool runs → PostToolUse.

- A PreToolUse guard therefore sees usage only up to the **previous** response, never the response that requested the tool.
- By PostToolUse, the current response's delta has arrived.

Other observations:

- The `Agent` tool returned immediately: the subagent ran in the background and the model waited on it with `TaskOutput`.
- One main-loop response streamed for about 32 s (from 35.3 s to 67.9 s) with no usage visible until its `message_delta`.
