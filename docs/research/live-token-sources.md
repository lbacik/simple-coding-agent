# Live token sources with the Meta backend

Research for #124 (map #123, origin #119). The question: with `claude-agent-sdk==0.2.156` and the Meta backend (`muse-spark-1.3-contributor` at `https://api.meta.ai`), where can the agent see token usage **during** an attempt, given that `AssistantMessage.usage` is all zeros?

No paid backend was called for this note. Sources:

- **SDK**: `claude_agent_sdk` 0.2.156 source in the project venv (`types.py`, `client.py`, `_internal/message_parser.py`, `_internal/query.py`, `_internal/transport/subprocess_cli.py`).
- **CLI**: the Claude Code CLI bundled with the SDK (`_bundled/claude`, `__cli_version__ = "2.1.276"`). It is minified. Quotes below use its minified identifiers, found with `strings` and searched by literal text.
- **Docs**: the official Anthropic pages [Track cost and usage](https://code.claude.com/docs/en/agent-sdk/cost-tracking) and [Settings reference: `modelPricing`](https://code.claude.com/docs/en/settings-reference).
- **Meta evidence in the repo**: `prototype-meta-skills-probe/full-run.log` on branch `prototype/meta-skills-probe`, a real Meta run of `muse-spark-1.3-contributor`, and the issue #119 logs.
- **Anthropic-backend transcripts**: local `~/.claude/projects/**.jsonl` files from Claude Code 2.1.278-2.1.285. These are interactive sessions, not SDK sessions, and are used only to show when the CLI writes each transcript entry.

## TL;DR

| Source | Live during the attempt? | Plausibly non-zero on Meta? | Confidence |
|---|---|---|---|
| `AssistantMessage.usage` (what we use today) | yes | **no**: it is the raw `message_start` usage, and Meta sends zeros there | confirmed (logs + CLI source + docs) |
| `StreamEvent` `message_delta.usage` (`include_partial_messages=True`) | yes, once per API response | **yes, very likely** | strong inference; needs a live check |
| Main-session transcript JSONL (`transcript_path`) | yes, per API response | **yes, very likely** (holds the merged `message_start` + `message_delta` usage) | strong inference; needs a live check |
| Subagent transcript JSONL (`agent_transcript_path`) | yes | **mostly no** (most entries are written before `message_delta`) | observed on Anthropic transcripts |
| `get_context_usage()` → `apiUsage` | yes, on demand (public SDK method) | likely yes, but only for the **last** main-loop response, not cumulative | CLI source; needs a live check |
| `get_usage` / `get_session_cost` control requests | yes, on demand | **yes**: they read the same cost ledger as `total_cost_usd` | CLI source; not exposed by the Python SDK (private `_send_control_request` only) |
| Hook payloads (PreToolUse/PostToolUse/Stop/SubagentStop) | yes | **no usage fields at all**; they only give `transcript_path` / `agent_transcript_path` | confirmed (SDK types + CLI schema) |
| `TaskProgressMessage.usage.total_tokens` | yes (subagents) | **no**: near-zero on Meta (10, 49), since it is built from the zero per-message usage plus a small client-side estimate | confirmed (Meta probe log) |
| `TaskNotificationMessage.usage.total_tokens` | at subagent end | yes (11016, 11832 on Meta) | confirmed (Meta probe log) |
| `ResultMessage.usage` / `model_usage` / `total_cost_usd` | **no**: terminal only | yes | confirmed (Meta probe log) |

**Main finding.** The CLI itself has real per-response token counts on Meta; otherwise `total_cost_usd` would be zero, and it is not. Those counts reach the CLI in the **`message_delta`** stream event. They are merged into the in-memory message after the SDK-facing `AssistantMessage` has already been serialized. The cheapest live source is therefore `include_partial_messages=True` plus reading `StreamEvent.event["usage"]` when `event["type"] == "message_delta"`.

**Secondary finding.** For an unknown model name the CLI prices tokens at the **default model's rate**. On this bundle that works out to $5 input, $25 output, $0.50 cache read and $6.25 cache write per million tokens. The Meta probe's `costUSD` matches this to the last digit. Meta's published rates are about 4x lower (see [Pricing of unknown models](#5-how-the-cli-prices-an-unknown-model-for-total_cost_usd)). So the "real cost" in #119 (`attempt.json`) is a CLI estimate, not Meta's bill, and the `max_budget_usd` hard ceiling is enforced against that same estimate.

---

## 1. `StreamEvent` partial messages (`message_start` / `message_delta`)

**How the stream is built (CLI 2.1.276).** In the CLI's streaming loop, the `message_start` handler does two things:

- It stores the raw API message: `jg=Hi.message`.
- It merges its usage into an accumulator: `ac=Hse(ac,Hi.message?.usage)`.

At every `content_block_stop` the CLI builds and **yields** an assistant message as `ml={message:{...jg,content:ed},...,type:"assistant",...}`. That message carries the **raw `message_start` usage**.

Only later, on `message_delta`, does it merge the delta usage (`ac=Hse(ac,Hi.usage)`) and mutate the messages it already yielded:

```js
for(let Lp of wh) if(Lp.message.usage=ac, Lp.message.stop_reason=Mp, ...)
```

The same handler credits the cost ledger:

```js
AP+=H5(ej(he,ac),ac,h.model,...)
```

This runs once per response, when `stop_reason` is non-null. If no `message_delta` carried a stop reason, the `message_stop` handler credits it instead.

The official docs describe the same behaviour: "Claude Code builds each assistant message from the usage the API reported when the response began … The API reports the real output count at the end of the response … To watch a response's output count grow while it streams, set `include_partial_messages` … and read `usage` from each `message_delta` stream event" ([cost tracking](https://code.claude.com/docs/en/agent-sdk/cost-tracking#read-output-tokens-from-the-result-message)).

**Raw events are forwarded as-is.** The loop yields `{type:"stream_event",event:Hi}` for every raw API event, `message_start` and `message_delta` included. The SDK passes the event through unchanged: `StreamEvent(uuid, session_id, event, parent_tool_use_id)` in `message_parser.py`, where `event` is typed "The raw Anthropic API stream event". The option maps to the `--include-partial-messages` flag (`subprocess_cli.py:687`).

**Merge semantics (`Hse`).** Two rules matter here:

- `input_tokens`, `cache_read_input_tokens` and `cache_creation_input_tokens` are taken from the delta only when they are `> 0`; otherwise the `message_start` value is kept.
- `output_tokens` is taken from the delta whenever it is present.

So a backend that sends zeros in `message_start` and real numbers in `message_delta` gets correct totals in the CLI.

**Why this is very likely populated on Meta:**

- The #119 log shows `model_usage_shape usage={'input_tokens': 0, 'output_tokens': 0}`. It has exactly two keys. An Anthropic `message_start` usage also has cache and `service_tier` keys, and the CLI's merged `ac` object has more still (`output_tokens_details`, `server_tool_use`, `cache_creation`, `inference_geo`, ...). So the SDK received Meta's raw `message_start` usage, not the merged object.
- The same runs still produce non-zero `model_usage` and `total_cost_usd`. The Meta probe shows `inputTokens: 93047, outputTokens: 7002, cacheReadInputTokens: 316575`. These are accumulated from `ac` in `H5`. With `message_start` at zero, the counts can only have come from `message_delta`. The one exception is a non-streaming fallback path, which is unlikely to be used on every turn.
- The CLI also has a telemetry event `tengu_message_delta_usage_missing` for deltas without usage. That confirms `message_delta.usage` is the CLI's canonical source for final output counts.

**Caveats:**

- Subagent stream events are **not** forwarded. Both `stream_event` emission sites in the bundle hard-code `parent_tool_use_id:null`, so this source only covers main-loop requests. Subagent spend must come from somewhere else (see §6).
- Partial messages add a large number of events. The consumer loop has to skip them cheaply.
- There is one `message_delta` with a stop reason per API response. Count it once per response. `message_start.message.id` gives the response id if deduplication is needed.

**Live check needed.** Run one short Meta query with `include_partial_messages=True` and log each raw `message_start.message.usage` and `message_delta.usage`. Confirm that the delta carries non-zero `input_tokens`, `cache_read_input_tokens` and `output_tokens`. Also check whether `input_tokens` includes or excludes cached tokens. The earlier research (`docs/research/meta-sdk-compatibility.md` on `research/meta-sdk-compatibility`) flags this as unverified.

## 2. CLI session transcript JSONL

**Where it is.** Every hook input carries `transcript_path` (`BaseHookInput` in `types.py`). `SubagentStopHookInput` adds `agent_transcript_path`. Main-session transcripts are at `~/.claude/projects/<cwd-slug>/<session_id>.jsonl`. Subagent transcripts are at `<session_id>/subagents/agent-<id>.jsonl`.

**What it holds.** Assistant entries are `{"type":"assistant","message":{..., "usage":{...}, "stop_reason":...}, "requestId", "isSidechain", ...}`. There is one entry per content block, and entries from the same API response share `message.id`.

**When it is written (observed, Anthropic backend, CLI 2.1.281-2.1.285):**

- **Main session**: every assistant entry sampled (531 of 531) has a non-null `stop_reason` and the merged `Hse` shape (`output_tokens_details`, `server_tool_use`, `cache_creation`, ...). So main-loop entries are written **after** `message_delta` and carry the final merged usage.
- **Subagent transcripts**: most entries (230 of 268 sampled) have `stop_reason: null` and the raw `message_start` shape with placeholder `output_tokens` (for example 4 or 16). Only the final entries are merged. Subagent entries are mostly written at yield time.

**On Meta:**

- Main-session transcript: very likely non-zero, because it holds the same merged `ac` that feeds `total_cost_usd`.
- Subagent transcripts: most entries would show zeros.

**Costs of using it.** The agent would have to tail a file and parse JSONL, dedupe by `message.id`, and cope with partially written lines. The transcript is also a CLI-internal format, not an SDK contract. It is a workable fallback, not the first choice.

**Live check needed.** After a short Meta run, read the main transcript and confirm that assistant entries carry non-zero usage with a non-null `stop_reason`. The container keeps it under the agent user's `~/.claude/projects/`. Also confirm that the entry appears before the next `PostToolUse` hook fires.

## 3. Hook payloads

In the SDK 0.2.156 TypedDicts (`types.py:299-420`) and the CLI 2.1.276 zod schemas, **no hook input carries token usage or cost**:

- **Base**: `session_id`, `transcript_path`, `cwd`, `prompt_id`, `permission_mode`, `agent_id`, `agent_type`.
- **PostToolUse**: adds `tool_name`, `tool_input`, `tool_response`, `tool_use_id`, and (CLI only) `duration_ms` and `mcp_server`.
- **Stop / SubagentStop**: add `stop_hook_active`, `last_assistant_message`, `background_tasks` and `session_crons`. SubagentStop also has `agent_id`, `agent_transcript_path` and `agent_type`.

So hooks are only useful as a **trigger** plus a **path** to the transcript (§2). The `include_hook_events` option only echoes these same payloads into the message stream.

## 4. Exact `ResultMessage.usage` / `model_usage` fields on this backend

From the Meta probe run (`prototype-meta-skills-probe/full-run.log`, `muse-spark-1.3-contributor`, 22 turns, two `code-review` subagents):

```json
"total_cost_usd": 0.7985724999999998,
"usage": {
  "input_tokens": 60547, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 295613,
  "output_tokens": 5325, "output_tokens_details": {"thinking_tokens": 1588},
  "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
  "service_tier": "standard",
  "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
  "inference_geo": "", "iterations": [], "speed": "standard"
},
"model_usage": {
  "muse-spark-1.3-contributor": {
    "inputTokens": 93047, "outputTokens": 7002, "cacheReadInputTokens": 316575,
    "cacheCreationInputTokens": 0, "webSearchRequests": 0, "costUSD": 0.7985724999999998,
    "contextWindow": 200000, "maxOutputTokens": 32000, "thinkingTokens": 2338,
    "canonicalModel": "muse-spark-1.3-contributor", "provider": "firstParty", "costBasis": "unknown"
  }
}
```

- `usage` is **main loop only**. The CLI schema describes it as "MAIN AGENT LOOP ONLY — excludes Task subagent, sidechain, and auxiliary model calls … Prefer modelUsage". The docs table agrees. Here `usage.input_tokens` is 60547 against `model_usage` 93047; the difference is subagent and auxiliary calls.
- `model_usage` covers the whole tree: main loop, subagents and compaction. The SDK's `ModelUsage` TypedDict omits `thinkingTokens` and `costBasis`, but they arrive in the raw dict.
- `cacheCreationInputTokens` is always 0 on Meta. This matches #119: Meta reports cache reads but no cache writes.
- All three are **terminal only**. They arrive on the `ResultMessage`, too late to drive a soft threshold. That agrees with the comment above `_ESTIMATED_USD_PER_MILLION_TOKENS`.

## 5. How the CLI prices an unknown model for `total_cost_usd`

CLI 2.1.276 prices each response with `ej(model, usage)`:

1. **Managed `modelPricing` table.** If the organization's managed-settings `modelPricing` table (`Lc()`) matches the model, that rate is used. It is only read from managed or policy sources (`Pj`: `helper`/`plist`/`hklm`/`file`); user and project settings are ignored. The docs agree: "Managed settings only".
2. **Built-in lookup (`oEn`).** Otherwise the CLI tries, in order:
   - the built-in table `Ghe`, keyed by canonical short name;
   - `additionalModelCostsCache`;
   - failing both, it calls `a_e()`, which logs `tengu_unknown_model_cost` and sets `costLedger.hasUnknownModelCost()`, and returns `Ghe[ze(kl())] ?? IXe`. That is **the default main-loop model's rate, or the constant `IXe = Zfe`** = `{inputTokens: 5, outputTokens: 25, promptCacheWriteTokens: 6.25, promptCacheWrite1hTokens: 10, promptCacheReadTokens: 0.5, webSearchRequests: 0.01}` USD per million tokens.
3. **Reporting.** `modelUsage[m].costBasis` is set to `"unknown"`. The CLI schema text reads: "no pricing row and no built-in price matched the model ID, so costUSD is a guess at the default model's rate". `/cost` text (`get_session_cost`) appends "costs may be inaccurate due to usage of unknown models".

**Check against the Meta probe:** 93047 × $5 + 7002 × $25 + 316575 × $0.50, all per million tokens, is **$0.7985725**. That equals the reported `costUSD` exactly. So on this deployment `muse-spark-1.3-contributor` is priced at $5 input / $25 output / $0.50 cache read.

**Consequences:**

- Meta's published rates are $1.25 input, $0.15 cached input and $4.25 output per million. The earlier research cites [Meta pricing](https://dev.meta.ai/docs/pricing-rate-limits). At those rates the probe would have cost about $0.19, roughly 4x less. The "real cost" in #119 (`attempt.json`, taken from `ResultMessage.model_usage`) is the same CLI estimate. The $20 hard ceiling (`max_budget_usd`) is enforced against it too.
- Our own estimator (`_CATEGORY_USD_PER_MILLION_TOKENS`: $3 / $15 / $0.30 / $3.75) uses yet another rate. It will not match the CLI's `total_cost_usd` even with perfect token counts.
- A `modelPricing` override for `muse-spark-1.3-contributor` would align both the CLI's `total_cost_usd` and the `max_budget_usd` ceiling with Meta's rates. It would go in a managed-settings file in the container (for example `/etc/claude-code/managed-settings.json`); the TS SDK's `managedSettings` option does the same, but it does not exist in Python SDK 0.2.156. This is outside #124's scope and is worth a separate decision.

## 6. Subagent turns and their usage

- **Visibility.** Subagent tool calls arrive as `AssistantMessage` / `UserMessage` with `parent_tool_use_id` set to the spawning Agent/Task `tool_use` id. Text and thinking blocks are included only with `forward_subagent_text=True`. Their `usage` is the same raw `message_start` usage, so it is zero on Meta. Hooks attribute subagent tool calls through `agent_id` (already used by `_hook_agent_id` / `_is_main_thread`).
- **No partial events.** Subagent `StreamEvent`s are not emitted: `parent_tool_use_id:null` is hard-coded at both emission sites.
- **`TaskProgressMessage.usage.total_tokens`.** This is computed as `latestInputTokens + cumulativeOutputTokens + streamedTokenEstimate`, where the first two come from each subagent assistant message's usage as seen at yield time. On Meta it is essentially just the client-side streamed-text estimate. The probe shows `total_tokens: 10`, `10`, `49` for subagents that later reported about 11k. **Not usable** as a live cost source.
- **`TaskNotificationMessage.usage.total_tokens`.** This is real on Meta (11016 and 11832 in the probe), but it only arrives when the subagent finishes. It is a single total (input + output), so it cannot be priced by category.
- **Subagent transcripts** (`agent_transcript_path`): mostly pre-`message_delta` entries (§2), so mostly zeros on Meta.
- **Totals.** Subagent spend does reach `model_usage`, `total_cost_usd` and the live cost ledger (§7). It is excluded from `ResultMessage.usage`.

## 7. Other live sources found in the CLI

- **`ClaudeSDKClient.get_context_usage()`** is a public SDK method, a control request, and works mid-run. Its `apiUsage` field (`{input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens}`) is taken by `dfe()` from the **last assistant message in the in-memory main-loop history**. That object is the one `message_delta` mutated, so it very likely holds non-zero values on Meta. It is **per-response, not cumulative**, and `null` when all input counts are zero. Polling it once per `PostToolUse` would give the current context size. That is exactly the quantity that made attempts 3 and 4 in #119 cost about $0.26 per turn. Its `totalTokens` is the CLI's own context estimate and does not depend on backend usage.
- **`get_usage` control request.** It returns "session cost/usage totals plus claude.ai plan rate-limit utilization", including `session.total_cost_usd` and `session.model_usage`, and supports `skip_behaviors: true`. It is marked experimental (TS: `usage_EXPERIMENTAL_MAY_CHANGE_DO_NOT_RELY_ON_THIS_API_YET`).
- **`get_session_cost` control request.** It returns the formatted `/cost` text built from `costLedger.totalCostUSD()`, the same ledger `H5` credits on each `message_delta`, and would parse into a live `total_cost_usd`.

Neither `get_usage` nor `get_session_cost` is exposed by Python SDK 0.2.156. Both could be sent through the private `client._query._send_control_request({"subtype": "get_usage", "skip_behaviors": True})`, which is fragile and should be avoided. They are the only mid-run sources that give the CLI's authoritative, whole-tree (subagents included) cost figure.

## What needs a live check against Meta

One short run against the Meta backend, capped at a few turns and with one subagent, would settle all open points. Log:

1. Each raw `StreamEvent` whose `event.type` is `message_start` or `message_delta` (with `include_partial_messages=True`). Record the `usage` dicts, and whether `input_tokens` includes cached tokens (compare with `cache_read_input_tokens`).
2. `get_context_usage()["apiUsage"]` after each `PostToolUse`.
3. The main transcript's assistant entries (via `transcript_path`): their usage and `stop_reason`, and whether each entry is on disk before the next `PostToolUse`.
4. Optionally, the private `get_usage` / `get_session_cost` control responses mid-run. Compare them with the final `ResultMessage.total_cost_usd`.
5. Whether the sum of `message_delta` usage over main-loop responses matches `ResultMessage.usage`, and whether main-loop plus the subagent `task_notification` totals approach `model_usage`.

## Implications for #119 (for the follow-up tickets, not decided here)

- The first-choice live source is `message_delta.usage` via `include_partial_messages=True`. It is public, documented and per response, and it lets the token-based estimator run on Meta for main-loop spend. Subagent spend still needs the turn-based fallback or `task_notification` totals.
- `get_context_usage()["apiUsage"]` is a public, lower-volume alternative for polling the main-loop context size.
- Whichever source is chosen, the estimator's rate table should match the rate the CLI actually charges against `max_budget_usd`. On this deployment that is the unknown-model default ($5 / $25 / $0.50), unless a managed `modelPricing` row is added. Otherwise the soft threshold and the hard ceiling are measured in different currencies.
