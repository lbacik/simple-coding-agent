# Meta and Claude Agent SDK compatibility contract

Research date: 2026-09-19. Resolves the documentary question in [Establish the Meta and Agent SDK compatibility contract](https://github.com/lbacik/simple-coding-agent/issues/2). No credentials were read, no model requests were sent, and no runtime compatibility has been demonstrated.

## Recommendation

Keep Python Claude Agent SDK with `muse-spark-1.3` as the MVP candidate, conditional on the separately tracked live prototype. Meta explicitly documents this integration. It is not a replacement of the Claude Code runtime: the Python package drives that runtime. Treat the provider configuration, installed skills, SDK version, and runtime version as one compatibility unit. [Meta integration](https://dev.meta.ai/docs/agent-frameworks), [SDK repository](https://github.com/anthropics/claude-agent-sdk-python)

## Configuration contract

Use `ClaudeAgentOptions.model="muse-spark-1.3"` and the following child-process environment:

| Variable | Value |
|---|---|
| `ANTHROPIC_BASE_URL` | `https://api.meta.ai` (no `/v1`) |
| `ANTHROPIC_AUTH_TOKEN` | Runtime-injected Meta Model API key |
| `ANTHROPIC_MODEL` | `muse-spark-1.3` |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | `muse-spark-1.3` |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | `muse-spark-1.3` |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | `muse-spark-1.3` |
| `CLAUDE_CODE_SUBAGENT_MODEL` | `muse-spark-1.3` |

The SDK adds `/v1/messages`; bearer authentication requires `ANTHROPIC_AUTH_TOKEN`, rather than the `x-api-key` mechanism. Alias overrides cover tier-selected subagents and lightweight background work. Meta recommends terminal-result detection, overall and idle deadlines, sandboxing, and a scrubbed environment. [Meta integration](https://dev.meta.ai/docs/agent-frameworks)

Our proposed contract additionally removes conflicting Anthropic/provider credentials, disallows model fallback, explicitly configures every custom subagent, and records only non-secret configuration. These are application decisions, not claims that upstream enforces them. The prototype must detect unexpected model IDs or destinations.

## Version candidate

Pin `claude-agent-sdk==0.2.156`, published 2026-09-18, source commit `e9af0778559032afca55ac200608c24f18af86ca`. Its bundled CLI version is `2.1.276`. This is a reproducible **candidate**, not a known-good Meta certification. Sources: [release](https://github.com/anthropics/claude-agent-sdk-python/releases/tag/v0.2.156), [immutable CLI version](https://github.com/anthropics/claude-agent-sdk-python/blob/e9af0778559032afca55ac200608c24f18af86ca/src/claude_agent_sdk/_cli_version.py), [immutable options/types](https://github.com/anthropics/claude-agent-sdk-python/blob/e9af0778559032afca55ac200608c24f18af86ca/src/claude_agent_sdk/types.py).

At prototype build time also lock the Python dependency graph, platform-specific wheel hashes, base image digest, installer version, and skills commit. Record the actual CLI version from the built artifact; do not silently replace it with a separately installed CLI. Docker architecture support and wheel availability must be verified for each supported local/server platform.

## Documented boundaries

The Meta Messages adapter documents custom tool calls/results, Anthropic SSE streaming, JSON-schema output, and replayable encrypted reasoning. It is stateless: history belongs to the client. Unsupported requests include named `tool_choice`, `thinking.type="disabled"`, `stop_sequences`, `top_k`, `container`, `inference_geo`, and unknown top-level fields. `thinking.type="enabled"` budgets do not select reasoning effort; prefer documented adaptive thinking and `output_config.effort` values. Input counting is available. These facts establish a wire-level foundation, not complete compatibility with every runtime feature. [Messages protocol](https://dev.meta.ai/docs/protocols/messages)

| Capability | Evidence and remaining requirement |
|---|---|
| File reads, edits, shell tests, multi-step loop | Explicit Meta framework use case; demonstrate an actual edit and independent passing test. |
| Main and delegated calls | Documented routing configuration; force one subagent and inspect observed model metadata. |
| Streaming and reasoning replay | Protocol support is documented; test multiple tool turns and resume without unsupported-block/parser failures. |
| Skills | SDK feature documented below; Meta-specific adherence remains unproven. |
| Structured final report | SDK feature documented below; validate exact schema and failure behavior on Meta. |
| Resume/context handling | SDK session support exists; provider-specific resume and compaction behavior are unverified. |
| Cancellation | SDK interrupt exists; prove process/container cleanup separately. |
| Dollar limits | SDK estimates are not authoritative; independent Meta accounting required. |

### Skills

Skills are filesystem artifacts discovered from selected settings sources. Explicit `/<name>` dispatch is supported. The `skills` option controls enabled names; an explicit `tools` list must retain `Skill`. Loading project settings also loads other project configuration, so do not copy Meta's `setting_sources=[]` example and expect ordinary filesystem skills to appear. Choose a controlled user source or explicit plugin, or deliberately enable audited project sources. [SDK skills](https://code.claude.com/docs/en/agent-sdk/skills)

The source and installation audit owns the actual names, dependency closure, and unattended suitability of mattpocock skills. This report does **not** assume `/implement` exists upstream. A skill invoking another skill is model-directed behavior, not a deterministic function dependency. The prototype must use the audited entrypoint and installed dependency set, rather than substitute a successful toy skill and declare the production chain validated.

### Structured results and terminal events

`output_format` can request a JSON schema and the SDK provides `structured_output`; validation retries can terminate with `error_max_structured_output_retries`. A syntactically valid report cannot prove the task meets its acceptance criteria. [Structured outputs](https://code.claude.com/docs/en/agent-sdk/structured-outputs)

Application proposal: require one correlated terminal `ResultMessage`, an acceptable subtype, no fatal error, validated report, and independent repository checks. Stream exhaustion, process exit zero, or an agent's success statement is insufficient. Preserve partial evidence when no terminal result arrives. Do not publish success following limit exhaustion or cancellation.

### Sessions, context, and cancellation

SDK sessions can resume using a session ID; resuming requires retaining the corresponding session data, not only storing an ID. [SDK sessions](https://code.claude.com/docs/en/agent-sdk/sessions)

`ClaudeSDKClient.interrupt()` is available for streaming-mode interaction. Buffered messages remain and must be drained before receiving a new response. Terminal reasons distinguish aborted streaming/tools. [Python SDK reference](https://code.claude.com/docs/en/agent-sdk/python)

Proposed MVP posture: a new session for each issue; no cross-issue context. Persist transcripts separately from durable queue state. A restart may preserve partial work without silently resuming a possibly non-idempotent command. Resume is accepted only after the prototype verifies the exact storage layout. Treat context overflow/compaction failures as unsuccessful attempts. Documentation reviewed here does not establish Meta-specific compaction thresholds, auxiliary-call compatibility, or that interruption cancels provider billing. Do not claim the model's nominal context capacity automatically becomes the runtime's usable context capacity.

## Cost and limits

Anthropic explicitly describes SDK dollar totals as client-side estimates based on a bundled table, potentially wrong for unknown models. Python exposes per-step usage and per-model result usage; resumed invocations report their own costs. Deduplicate repeated assistant fragments by message identity rather than summing every emitted fragment. [Cost tracking](https://code.claude.com/docs/en/agent-sdk/cost-tracking)

The pinned Python `ClaudeAgentOptions` exposes `max_budget_usd` but no first-class `model_pricing` option. Do not treat that limit as an accurate Meta spending cap or assume the bundled table knows Muse Spark. A settings-based pricing override, if explored, needs a separate verified contract. [Pinned Python types](https://github.com/anthropics/claude-agent-sdk-python/blob/e9af0778559032afca55ac200608c24f18af86ca/src/claude_agent_sdk/types.py)

Meta's current standard model rates per million tokens are $1.25 input, $0.15 cached input, and $4.25 output. Built-in web search adds $2.50 per 1,000 queries. Keep the standard `muse-spark-1.3` identifier; do not silently substitute a contributor model. Version the price configuration. [Meta pricing](https://dev.meta.ai/docs/pricing-rate-limits)

Reasoning tokens are billed within output tokens, so do not add reasoning usage twice. [Meta reasoning](https://dev.meta.ai/docs/reasoning)

Application proposal: maintain an independent ledger with normalized uncached-input, cached-input, and total-output counts, including subagent/auxiliary work. Verify whether raw `input_tokens` includes cached tokens before using a subtraction formula. Compare estimates with Meta's account usage during the probe. Missing usage after an interrupted stream means **unknown cost**, not zero. Timeouts and turn caps bound work but are not hard dollar ceilings. Hard spending guarantees require a verified provider/gateway enforcement mechanism or conservative admission reservations that include in-flight requests; final-result accounting alone is insufficient. Reject new work when the ledger is uncertain beyond the agreed allowance.

## Error classification proposal

Meta documents backoff/jitter for 429, 500, and 503, including `Retry-After`; stream failures may occur after HTTP success. Authentication, permission, absent model, invalid request, and billing problems need configuration/operator action. Non-streaming 504 requires changing request behavior rather than blindly repeating it. [Meta errors](https://dev.meta.ai/docs/error-handling)

Normalize SDK exceptions, result errors, and missing-terminal failures into application categories. Retry only evidenced transient transport/provider failures, within one bounded retry budget; account for any runtime-internal retries. Never repeat the whole issue blindly after a stream failure: local tools may already have changed files. Test failure, unmet acceptance criteria, missing requirements, denied tools, exhausted budgets, or invalid final reports are attempt outcomes, not transient infrastructure retries. Unknown errors fail closed for operator classification. Probe the actual SDK error shape; do not assume HTTP headers survive every runtime abstraction.

## Smallest acceptance probe

This is a proposed experiment for the separate prototype ticket, not implementation work performed by this report.

1. Build the locked candidate image with the audited skills, controlled settings and a disposable fixture repository. Keep publication credentials outside it. Record hashes and non-secret config.
2. Submit a tiny, fully specified issue through the real entrypoint. Require it to invoke the intended testing skill, add a failing behavior test, implement the change, run tests, and produce the small final schema. Collect tool events and independently rerun the test outside the agent loop.
3. Force one inexpensive subagent. Capture model IDs/destinations for main, delegated, and any observed auxiliary requests without logging keys. Any fallback to Claude or non-Meta inference fails routing acceptance.
4. In the same fixture, resume a completed session with a follow-up that requires earlier context. Verify tool-turn reasoning blocks survive replay. Trigger a bounded context-management case if automatic compaction will be enabled; otherwise document that untested feature as excluded.
5. Interrupt a deliberately long local tool command. Verify a bounded shutdown, no orphan worker/tool processes, no successful outcome, and preserved partial evidence. Separately check overall/idle watchdogs.
6. Inject transient and permanent failures through a local mock transport or proxy, avoiding paid error experiments. Verify retry classification, limits, missing-terminal handling, malformed structured reports, and no duplicate publication intent.
7. Reconcile step/result/subagent usage and the independent price calculation against provider usage. Demonstrate limit behavior and explicitly quantify overshoot/unknown-usage handling.

Pass requires all mandatory checks, immutable image/version records, a sanitized event log, fixture diff, independent test output, and an explicit list of excluded features. A basic successful model response alone does not pass. If full skill execution, mandatory routing, structured reporting, or bounded shutdown fails, keep the runtime decision provisional. Resolve a concrete workaround and rerun its failing case before implementation starts; do not silently switch models or build a new raw-API agent loop.
