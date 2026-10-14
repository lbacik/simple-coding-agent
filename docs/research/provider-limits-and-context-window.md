# Provider limits and context-window data available at runtime

Research for #223 (map #222, agent fleet dashboard). Date: 2026-10-14. Question: which limit and context-window data do the Meta model API, the LiteLLM gateway and the pinned Claude Agent SDK expose at runtime, and which of it is reliable versus an estimate? No paid backend was called and no credentials were read. Items marked **unverified** need a live check.

Sources (primary only):

- Meta docs: [Pricing and rate limits](https://dev.meta.ai/docs/pricing-rate-limits), [Messages protocol](https://dev.meta.ai/docs/protocols/messages), [Models](https://dev.meta.ai/docs/models).
- LiteLLM docs: [Response headers](https://docs.litellm.ai/docs/proxy/response_headers), [Model management](https://docs.litellm.ai/docs/proxy/model_management), [Virtual keys](https://docs.litellm.ai/docs/proxy/virtual_keys). Gateway image pinned by ADR 0003 (`v1.104.0`).
- Claude Agent SDK / Claude Code docs: [Track cost and usage](https://code.claude.com/docs/en/agent-sdk/cost-tracking), [Python SDK reference](https://code.claude.com/docs/en/agent-sdk/python), [Model configuration](https://code.claude.com/docs/en/model-config), [Environment variables](https://code.claude.com/docs/en/env-vars).
- SDK source: wheel `claude_agent_sdk==0.2.163` (the pin in `pyproject.toml` on `main`; `_cli_version.py` = `2.1.286`): `types.py`, `client.py`, `_internal/message_parser.py`, and the bundled CLI binary (strings search).
- Repo: `simple_coding_agent/token_ledger.py`, `simple_coding_agent/model_execution.py`, `docker/managed-settings.json`, `docs/adr/0003-litellm-gateway-for-meta-prompt-caching.md` (all on `main`); `docs/research/live-token-sources.md` on branch `research/live-token-sources` (earlier evidence from SDK 0.2.156 / CLI 2.1.276 and a real Meta probe log).

## TL;DR

| Datum | Source | Live? | Reliability for a dashboard |
|---|---|---|---|
| Meta rate limits (RPM/TPM per team) | Meta docs | static | Documented constants; the account's real quota can differ. |
| `x-ratelimit-{limit,remaining}-{tokens,requests}` | Meta response headers | per response | Reliable on Meta, but per **team** (shared by all agent instances), and the CLI does not surface response headers to the SDK. See "Header access gap". |
| Meta usage endpoint | none documented | - | Does not exist in the docs. |
| Meta context window | Meta Models page: 1,048,576 tokens for all Muse Spark versions | static | Documented. The Contributor variant is not listed separately (**unverified** for `muse-spark-1.3-contributor`). Max output tokens: not documented. |
| Meta `GET /v1/models`, `/v1/models/{model}` | Meta Models page | on demand | Only `created` is documented; no window field. |
| `POST /v1/messages/count_tokens` | Meta Messages protocol | on demand | Documented, returns `{input_tokens}`. |
| Usage fields in responses | Meta Messages protocol | per response | `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `output_tokens_details.thinking_tokens`. Measured. No cache-creation counter. |
| SDK `ResultMessage.model_usage[m].contextWindow` | SDK | terminal | **Not reliable for Meta**: a guess for an unknown model ID (see below). |
| SDK `get_context_usage()` (`totalTokens`, `maxTokens`, `percentage`) | SDK | on demand | `totalTokens` is the CLI's own context count; `maxTokens` is the CLI's assumed window, not Meta's. |
| SDK `RateLimitEvent` | SDK | on change | **Not applicable to Meta**: models claude.ai subscription windows only. |
| SDK compaction signal | `SystemMessage` subtype `compact_boundary`, `PreCompact` hook | event | Reliable as a signal; payload shape only partly typed. |
| SDK usage during the attempt | `message_delta` stream events, `get_context_usage()` | live | Measured, see `live-token-sources.md`; `AssistantMessage.usage` is zeros on Meta. |
| `total_cost_usd` / `costUSD` | SDK | terminal | Client-side **estimate**, even per the official docs. |
| LiteLLM `x-litellm-*` cost and call headers | LiteLLM docs | per response | LiteLLM's own accounting; the CLI cannot expose them to the agent. |
| LiteLLM `GET /model/info`, `/key/info` | LiteLLM docs | on demand | Usable out-of-band by a dashboard; token limits only if configured in `model_info` (**unverified**). |

## 1. Meta model API

### Rate limits (documented)

From [Pricing and rate limits](https://dev.meta.ai/docs/pricing-rate-limits):

- "Rate limits apply per team, not per API key." Each model has its own limits.
- Muse Spark 1.1, 1.2, 1.3 (and Muse Wonder 1.0): 3,000 RPM and 4,000,000 TPM.
- Muse Spark 1.2 and 1.3 **Contributor** variants: **150 RPM and 3,000,000 TPM**. The repo's backend is `muse-spark-1.3-contributor`, so the 150 RPM figure is the relevant one.
- Token counts include only input and output tokens; injected steering tokens are excluded from `usage` and from token counting.
- Over a limit the API rejects the request (HTTP 429, `rate_limit_error` per the Messages page). Retry guidance: exponential backoff with jitter from about 500 ms; honor `Retry-After` when present (documented explicitly for background submissions).

### Rate-limit response headers (documented)

On successful token-based responses: `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-tokens`, `x-ratelimit-limit-requests`, `x-ratelimit-remaining-requests`. The docs give no reset-time header and no window length. Because limits are per team, a value read by one agent instance reflects the traffic of every instance on the team, so it is a fleet-level number, not an instance-level one.

### Usage endpoint

None is documented. The Messages and Pricing pages describe no endpoint that returns consumed quota or spend. Per-attempt spend has to come from the response `usage` fields.

### Context window

[Models](https://dev.meta.ai/docs/models): "all Muse Spark versions have a 1,048,576-token context window." Max output tokens are not stated on the Models, Messages or Pricing pages. `GET /v1/models` documents only a `created` field, so the window cannot be discovered at runtime from Meta; it has to be configured. Overflow behavior (which error code a too-long prompt returns) is not documented.

### Measured fields

`POST /v1/messages` responses carry `input_tokens`, `output_tokens`, `cache_read_input_tokens` and `output_tokens_details.thinking_tokens` ([Messages protocol](https://dev.meta.ai/docs/protocols/messages)). `POST /v1/messages/count_tokens` returns `{ "input_tokens": n }`. These are provider-measured and the only fully reliable token numbers. Per `live-token-sources.md`, Meta sends zeros in `message_start` and real numbers in `message_delta`.

## 2. LiteLLM gateway (ADR 0003)

Everything below comes from the LiteLLM docs for the proxy; the pinned `v1.104.0` image was not run, so header presence on `/v1/messages` (the route the CLI uses) is **unverified**.

- **Rate-limit headers.** LiteLLM can return `x-ratelimit-{limit,remaining}-{requests,tokens}` and `x-ratelimit-reset-{requests,tokens}`. Per the docs, if the key has rate limits it returns the key's remaining limits; otherwise it returns the backend provider's values, or `None` when the provider sends none. So behind the gateway the values describe the LiteLLM key if one is limited, not Meta's team quota. Provider headers are also forwarded with an `llm_provider-` prefix (Meta-specific names **unverified**).
- **Cost headers** (LiteLLM's own price table, not Meta's invoice): `x-litellm-response-cost`, component costs (non-streaming only), `x-litellm-key-spend`.
- **Other headers:** `x-litellm-response-duration-ms`, `x-litellm-overhead-duration-ms`, `x-litellm-attempted-retries`, `x-litellm-attempted-fallbacks`, `x-litellm-call-id`, `x-litellm-model-id`, `x-litellm-model-group`, `x-litellm-version`.
- **Out-of-band endpoints a dashboard could call:** `GET /model/info` ("full model list with API keys masked", with `model_info` passthrough "alongside the mapped cost and context data") and `/key/info` (sample fields `spend`, `expires`, `models`, `aliases`, `config`). The docs reviewed do not enumerate the token-limit field names in the `/model/info` response, so a context-window value from LiteLLM is **unverified** and, for a custom model name such as `muse-spark-1.3-contributor`, is whatever the operator configured.
- **Request id.** ADR 0003 already records that Meta's `request-id` does not reach the agent through the gateway.
- **Not an authority for the context window.** LiteLLM does not know Meta's window unless configured.

## 3. Pinned Claude Agent SDK (`claude-agent-sdk==0.2.163`, CLI 2.1.286)

### Usage and cost

- `AssistantMessage.usage`: zeros on Meta (`live-token-sources.md`, measured on 0.2.156; the Anthropic docs say per-step `output_tokens` is a placeholder). `TokenLedger` therefore reads `message_delta` stream events (`include_partial_messages=True`, set in `model_execution.py`) and replaces its estimate with the reported categories.
- `ResultMessage.usage` covers the main loop only; `model_usage` and `total_cost_usd` cover subagents too ([cost tracking](https://code.claude.com/docs/en/agent-sdk/cost-tracking)). All terminal only.
- Official warning: `total_cost_usd` and `costUSD` "are client-side estimates, not authoritative billing data". The repo mitigates this with `docker/managed-settings.json` `modelPricing` ($1.25 input, $4.25 output, $0.15 cache read, $1.25 cache write per million for `muse-spark-1.3-contributor`). The docs say `costBasis` is then `managed`; on 2.1.286 this is **unverified** (it was `unknown` on 2.1.276 before the override).

### Context-window size: the SDK value is not Meta's

- `ModelUsage.contextWindow` and `get_context_usage()["maxTokens"]/"rawMaxTokens"` come from the CLI's own model table, not from the provider. In the real Meta probe log (CLI 2.1.276, quoted in `live-token-sources.md`) the entry for `muse-spark-1.3-contributor` reads `contextWindow: 200000`, `maxOutputTokens: 32000`, `costBasis: "unknown"`. Meta documents 1,048,576. The Claude Code docs confirm the mechanism: for an unrecognized ID the CLI "compacts at the context window Claude Code assumes for the ID", and the default max output is 32,000 for unknown IDs ([env vars](https://code.claude.com/docs/en/env-vars), [model config](https://code.claude.com/docs/en/model-config#correct-the-window-for-a-gateway-or-custom-model-id)).
- Overrides exist: `CLAUDE_CODE_MAX_CONTEXT_TOKENS` (declare the real window for a gateway/custom ID), `CLAUDE_CODE_AUTO_COMPACT_WINDOW` (100,000 to 1,000,000, capped at the model window), `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`, `CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1`, `CLAUDE_CODE_MAX_OUTPUT_TOKENS`. A grep of `simple_coding_agent/`, `docker/`, `Dockerfile`, both compose files and `.env.example` on `main` found none of them set. So the CLI's assumed window (and thus the auto-compaction point and any "percent of window") is its guess for an unknown ID. Whether 2.1.286 still reports 200000 for this ID is **unverified**.
- Consequence for the snapshot: report `context_window` as a configured/declared value with a `source` field (`configured`, `sdk_assumed`), never present the SDK value as the provider's.

### `get_context_usage()`

Public `ClaudeSDKClient` method, on demand. `ContextUsageResponse` has `totalTokens` ("session's current context usage"), `maxTokens`, `rawMaxTokens`, `percentage`, `model`, `isAutoCompactEnabled`, optional `autoCompactThreshold`, and per-category breakdowns. The Python reference says Claude Code computes it with "several token-counting API requests that don't appear in the message stream". Against Meta these go to the gateway or Meta's `count_tokens`; their cost and effect on the 150 RPM Contributor limit per poll is **unverified**, so poll sparingly. It does not report subagent contexts.

### Rate-limit events

`RateLimitEvent` / `RateLimitInfo` (`types.py`, parsed in `message_parser.py` from `rate_limit_event`) have `status` (`allowed`, `allowed_warning`, `rejected`), `resets_at`, `utilization` and `rate_limit_type` limited to `five_hour`, `seven_day`, `seven_day_opus`, `seven_day_sonnet`, `overage`. The bundled CLI strings show it derives from `anthropic-ratelimit-unified-*` headers, which are claude.ai subscription headers. Meta sends `x-ratelimit-*` instead. Conclusion: **no usable rate-limit event on the Meta backend**. `model_execution.py` does not handle `RateLimitEvent` today (no references found).

### Header access gap

Neither the SDK nor the CLI stream exposes raw HTTP response headers of the model backend; the SDK passes through the CLI's JSON stream only. So Meta's or LiteLLM's `x-ratelimit-*` and `x-litellm-*` headers cannot be read inside the agent process without sitting in the request path (for example a proxy) or reading gateway logs. The gateway side can be queried out-of-band (LiteLLM endpoints above). This is an inference from the typed SDK surface, not a documented statement.

### Compaction signals

- `model_execution.py` handles `SystemMessage` with subtype `compact_boundary` and logs `context_compacted`; its comment states SDK 0.2.163 has no typed message for it and `compact_metadata` is unverified.
- The bundled CLI 2.1.286 declares the schema: `compact_metadata: {trigger: "manual"|"auto", pre_tokens: int, post_tokens?: int, cumulative_dropped_tokens?: int (marked @internal)}`. This settles the shape at schema level (CLI binary, not a documented contract). A runtime check on Meta is still needed.
- The SDK types a `PreCompact` hook input (`trigger`: `manual`|`auto`, `custom_instructions`), usable as an earlier trigger.
- `token_ledger.py` re-anchors main-thread context at the next measured response, so compaction does not change budget accounting. For a dashboard, `pre_tokens`/`post_tokens` would give a measured before/after context size.

## 4. Reliable versus estimate

| Reliable (provider- or CLI-measured) | Estimate or guess |
|---|---|
| Provider `usage` per response (via `message_delta`): input, cache read, output, thinking | `TokenLedger` estimates for responses without usage and for subagent threads (chars-per-token constants, system bases) |
| `count_tokens` exact input size (on request) | `total_cost_usd` / `costUSD` (client-side; priced by `managed-settings.json` or default-model guess) |
| `model_usage` token counts (terminal, whole tree) | `contextWindow` / `maxTokens` from the SDK for the unknown Meta model ID (200000 observed vs 1,048,576 documented) |
| Compaction occurrence (`compact_boundary`) and `pre_tokens` per the CLI schema | `TaskProgressMessage.usage.total_tokens` on Meta (near zero) |
| Meta `x-ratelimit-*` values (team-wide; not readable by the CLI) | LiteLLM cost headers (LiteLLM's price table, not Meta's invoice) |

The ledger's reconciliation mode (`measured` / `mixed` / `estimated`, in `token_ledger.py`) is already the right per-figure flag for the snapshot.

## 5. Implications for the telemetry snapshot (not decided here)

1. Add explicit provenance per figure (`measured`, `estimated`, `configured`); do not collapse them.
2. Treat the context window as configuration: Meta documents 1,048,576, the CLI assumes another value unless `CLAUDE_CODE_MAX_CONTEXT_TOKENS` is set. Whether to set it (so auto-compaction and `percentage` match the real window) changes agent behavior and is a separate decision.
3. Provider rate-limit headroom is not available from inside the agent. Options: read LiteLLM `/key/info` or `/model/info` out-of-band, or mark it "unavailable" in v1.
4. Subagent contexts have no live measured source on Meta; the ledger value is an estimate.
5. Label cost as "estimated".

## Open live checks

1. Log `ResultMessage.model_usage[...]["contextWindow"]`, `maxOutputTokens`, `costBasis` and `get_context_usage()` (`maxTokens`, `rawMaxTokens`, `autoCompactThreshold`) on SDK 0.2.163 against Meta, with and without `CLAUDE_CODE_MAX_CONTEXT_TOKENS`.
2. Confirm Meta's window for the Contributor variant equals 1,048,576.
3. Capture raw response headers from the gateway for `/v1/messages` (`x-ratelimit-*`, `llm_provider-*`, `x-litellm-*`) with plain `curl`.
4. Confirm `compact_metadata` fields at runtime from a forced `/compact`.
5. Measure the request cost of `get_context_usage()` against the 150 RPM Contributor limit.

## Live check results (2026-10-14)

Run for "Run the live checks from the provider-limits research" (map "Agent fleet dashboard: map"). Each check ran in a throwaway container from the agent image (`claude-agent-sdk==0.2.163`, CLI 2.1.286, `/etc/claude-code/managed-settings.json` present), never inside a running agent. The SDK checks sent the CLI's traffic through an in-container logging proxy that recorded paths, response headers and usage, never request headers. Scripts: `docs/research/provider-limits-live-checks/sdk_probe.py` and `gateway_probe.py`. Total spend about $1.7, mostly the two large window probes.

### 1. SDK window, output and cost fields

| Field | Default (no override) | `CLAUDE_CODE_MAX_CONTEXT_TOKENS=1048576` |
|---|---|---|
| `model_usage[m].contextWindow` | 200000 | 1048576 |
| `model_usage[m].maxOutputTokens` | 32000 | 32000 |
| `model_usage[m].costBasis` | `managed` | `managed` |
| `get_context_usage()` `maxTokens` / `rawMaxTokens` | 200000 / 200000 | 1048576 / 1048576 |
| `autoCompactThreshold` | 167000 | 1015576 |
| `isAutoCompactEnabled` | true | true |

- The CLI 2.1.286 still assumes a 200,000-token window for `muse-spark-1.3-contributor`; the override changes every window figure, and auto-compaction moves with it (threshold = window minus a 33,000-token "Autocompact buffer" category).
- `costBasis` is `managed`, and `total_cost_usd` matches the `managed-settings.json` prices exactly (16,053 input × $1.25/M + 44 output × $4.25/M = $0.02025325). Still an estimate of Meta's invoice, but a deterministic one.
- `get_context_usage()` also returns keys the SDK `TypedDict` does not declare: `agents`, `apiUsage`, `autocompactSource`, `gridRows`, `mcpTools`, `messageBreakdown`, `skills`, `slashCommands`.

### 2. Meta context window for the Contributor variant

Confirmed at 1,048,576 within the probe's resolution. `GET /v1/models/muse-spark-1.3-contributor` returns only `{id, object, created: 0, owned_by: "meta"}`, so the window was bounded empirically with `max_tokens: 16`:

| Input (Meta `usage.input_tokens` or `count_tokens`) | Result |
|---|---|
| 249,324 | 200, accepted |
| 1,037,161 | 200, accepted |
| 1,052,279 (`count_tokens`) | 400 |
| 1,097,153 (`count_tokens`) | 400 |

The window (input plus `max_tokens`) lies between 1,037,177 and 1,052,295 tokens, consistent with 1,048,576. Overflow is **not** reported as a context-length error: Meta returns a generic `400 invalid_request_error` "The request contains invalid parameters. Check the request body for any errors or inconsistencies." without `x-ratelimit-*` headers. A caller cannot tell an overflow from any other invalid request by the message. Meta also rejects `max_tokens` below 16.

### 3. Rate-limit and gateway response headers

- **Direct to Meta** (`api.meta.ai`): every `/v1/messages` and `/v1/messages/count_tokens` response carries `x-ratelimit-limit-requests: 150`, `x-ratelimit-remaining-requests`, `x-ratelimit-limit-tokens: 3000000`, `x-ratelimit-remaining-tokens` and `x-request-id`. The remaining counters lag and are team-wide (other instances shared the team during the run), so they describe fleet headroom, not the instance's.
- **Through the LiteLLM gateway** (`v1.104.0`, the model routed as `openai/muse-spark-1.3-contributor`): `/v1/messages`, streaming or not, returns **no** `x-ratelimit-*` and **no** `llm_provider-*` headers; only `x-litellm-*` (`call-id`, `model-id`, `model-name`, `model-group`, `version`, `response-duration-ms`, `overhead-duration-ms`, `attempted-retries`, `attempted-fallbacks`) and cost headers that are all `0.0` because no price is configured for the model. `count_tokens` returns no `x-litellm-*` headers at all.
- **Gateway endpoints**: `GET /model/info` answers, but every limit and cost field in `model_info` is `null` (`max_tokens`, `max_input_tokens`, `max_output_tokens`, the cost fields); `GET /key/info` returns 404 with the instance's key.
- Conclusion: provider rate-limit headroom cannot appear in the v1 snapshot from either the agent or the gateway as deployed. Only a direct-to-Meta instance sees the headers on the wire, and the CLI does not expose them to the agent.

### 4. `compact_metadata` at runtime

A forced `/compact` produced, in order: `SystemMessage` `status` (`status: "compacting"`), one `/v1/messages` call (17,389 input, 2,783 output tokens), `status` (`compact_result: "success"`), a fresh `init`, then `compact_boundary` with

```json
{"trigger": "manual", "pre_tokens": 16098, "post_tokens": 3102, "cumulative_dropped_tokens": 12996, "duration_ms": 11548}
```

and `logical_parent_uuid` next to it. This matches the CLI schema plus one field the schema did not list (`duration_ms`). The compaction call's tokens appear in the following `ResultMessage.model_usage` (cumulative per session) while that result's `usage` is all zeros and `num_turns` is 0.

### 5. Request cost of `get_context_usage()`

| Call | Requests | Latency |
|---|---|---|
| First call in a session | 5 × `POST /v1/messages/count_tokens` | ~1.7 s |
| Any later call, including after new turns (3 turns measured) | 0 | ~10 ms |
| First call after a compaction | 2 × `count_tokens` | ~1.3 s |

After new turns `totalTokens` follows the latest response's `input_tokens` without new requests, so the CLI recomputes only cached categories when they change. The `count_tokens` calls lowered `x-ratelimit-remaining-requests` but not `x-ratelimit-remaining-tokens`, so they count against the 150 RPM request limit only. Polling every 5 s therefore costs about 5 requests per session plus 2 per compaction: negligible against 150 RPM. The values it adds over the ledger are the CLI's window, `autoCompactThreshold` and the category breakdown; `totalTokens` equals the main thread's last measured input.

### Side observation

In one session the CLI sent an extra `/v1/messages` call (1 message, no tools, 797 input, 2,814 output tokens of which 2,797 thinking, ~20 s) that finished after the session's `ResultMessage` and was not part of its `model_usage`. It did not recur in the second session. Its purpose was not identified; it is noted only because such calls spend team quota outside the attempt's accounting.
