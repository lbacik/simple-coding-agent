# Resource-aware handoff: SDK contract

Research for [Establish the SDK contract for resource-aware handoff](https://github.com/lbacik/simple-coding-agent/issues/42), 2026-09-21. This note establishes feasible mechanisms and remaining experiments, not product policy. No model invocation was made.

## Evidence identity

- Repository baseline: `e125bd79495ccd24b07a0576bb303b71005fab1e`.
- Installed Python package metadata: `claude-agent-sdk==0.2.156`; upstream tag resolves to `39645355de8518711c60304c8370b2838d338191`. Inspected installed `types.py`, `client.py`, `message_parser.py`, and `subprocess_cli.py`; citations below point to the corresponding pinned upstream files.
- Executing the installed package's bundled binary with `--version` reports **2.1.276**, matching the repository's intended pair. The developer machine's PATH `claude --version` reports **2.1.278**. Neither command invokes a model.
- [Transport discovery](https://github.com/anthropics/claude-agent-sdk-python/blob/39645355de8518711c60304c8370b2838d338191/src/claude_agent_sdk/_internal/transport/subprocess_cli.py) prefers the bundled binary unless `cli_path` is supplied. Repository [provenance verification](https://github.com/lbacik/simple-coding-agent/blob/e125bd79495ccd24b07a0576bb303b71005fab1e/simple_coding_agent/provenance.py) checks PATH `claude`, while [execution](https://github.com/lbacik/simple-coding-agent/blob/e125bd79495ccd24b07a0576bb303b71005fab1e/simple_coding_agent/model_execution.py) supplies no `cli_path`. Therefore the verified executable and executed executable need not be identical. This local result is not verification of the deployed container.

## Signals: available does not mean equivalent to the hard ceiling

Source-confirmed in the pinned [SDK types](https://github.com/anthropics/claude-agent-sdk-python/blob/39645355de8518711c60304c8370b2838d338191/src/claude_agent_sdk/types.py) and [parser](https://github.com/anthropics/claude-agent-sdk-python/blob/39645355de8518711c60304c8370b2838d338191/src/claude_agent_sdk/_internal/message_parser.py):

| Surface | Available evidence | Limitation |
| --- | --- | --- |
| `AssistantMessage` | `message_id`, `usage`, `parent_tool_use_id`, `stop_reason` | Optional fields; multiple messages can belong to one response. Counting objects or tool hooks is not counting turns. |
| `StreamEvent` | Raw API event and parent tool ID | Requires `include_partial_messages=True`; current executor does not enable it. |
| Task progress/completion | `total_tokens`, `tool_uses`, `duration_ms`, task identity | No typed dollar total or token price categories; do not add cumulative task snapshots repeatedly. |
| `ResultMessage` | `num_turns`, `total_cost_usd`, `model_usage`, `usage`, `subtype`, `terminal_reason`, `stop_reason` | Terminal evidence arrives too late to initiate a pre-limit handoff. |
| `get_context_usage()` | Context categories and optional `apiUsage` dictionary | Context occupancy is not spent turns or dollars. Optional dictionary lacks a sufficiently precise typed pricing contract for a controller. |

The [official cost guide](https://code.claude.com/docs/en/agent-sdk/cost-tracking) says assistant input/cache usage must be deduplicated by message ID; its output count is a placeholder. Partial `message_delta` usage provides evolving output counts. Result `usage` excludes subagents; `model_usage` and total cost include them. Costs are client estimates, not invoices. Streaming results contain cumulative cost, so retain the latest total rather than summing results. Fresh query calls account separately, including when resuming conversation history. These facts prevent treating a sum of ordinary assistant messages as an exact whole-tree dollar meter.

The [official loop guide](https://code.claude.com/docs/en/agent-sdk/agent-loop#turns-and-budget) defines the hard turn limit in tool-use round trips. A same-process streaming follow-up starts a new turn allowance, but retains accumulated dollar spend; subagents share the dollar cap. Enforcement described there requires CLI 2.1.217+, older than the installed bundle. This is documented behavior, not a measured Meta execution result. Unique main-agent response IDs containing tool calls are a candidate soft-turn signal; equivalence under retries, compaction, nested work, and interruption still needs a trace comparison against terminal counters.

## Control and continuity

The pinned [client](https://github.com/anthropics/claude-agent-sdk-python/blob/39645355de8518711c60304c8370b2838d338191/src/claude_agent_sdk/client.py) implements both `interrupt()` and subsequent `query()` inputs on an existing transport. It supports receiving another response after submitting another prompt. `resume` loads a stored conversation via options. History continuity and budget continuity are distinct contracts: a new process must not silently grant another full implementation-attempt allowance.

The pinned hook schemas support `additionalContext` on `PreToolUse` and `PostToolUse`, with `agent_id` attribution for subagent tool hooks. The [hook documentation](https://code.claude.com/docs/en/agent-sdk/hooks) distinguishes model context from `systemMessage`, which is user-facing. Returning `continue_=False` stops execution; it does not ask the model to finish a document. A hook can steer the next model step, but cannot guarantee instruction compliance, reserve compute, or fire during a response that makes no tool call.

Feasible candidates, subject to explicit policy and validation:

1. **Cooperative handoff in the current loop.** Detect a soft threshold, inject a one-time handoff instruction, constrain subsequent tool activity, and retain the original ceilings. Validate hook timing and remaining headroom; concurrent subagents may still be working.
2. **Interrupt, drain, then request handoff on the same client.** Retains conversation context and permits a separate handoff prompt. Track the combined attempt turn allowance outside the SDK because the new input gets its own turn allowance. The cumulative dollar ceiling leaves only remaining headroom. Verify tool cancellation and subagent quiescence before saving final progress.
3. **Fresh/resumed client for handoff.** Possible API shape, but requires explicit remaining-budget allocation and durable attempt accounting. A fresh session plus branch/note can also support later human-authorized continuation without retaining a CLI transcript. Do not confuse either approach with automatic budget renewal within the same attempt.

Current [executor](https://github.com/lbacik/simple-coding-agent/blob/e125bd79495ccd24b07a0576bb303b71005fab1e/simple_coding_agent/model_execution.py) returns at the first terminal result and only interrupts on timeout. It discards incremental usage, session IDs and several terminal fields. Its comment claiming SDK 0.2.156 lacks `terminal_reason`/`total_cost_usd` conflicts with the installed source; the handoff specification should not preserve that assumption. Its SDK protocol would need follow-up-query support for candidate 2.

## Meta boundary and skill provenance

[Meta's official overview](https://dev.meta.ai/docs/overview) documents Anthropic-compatible access and the `muse-spark-1.3-contributor` model used by the repository. The executor directs Anthropic requests to `https://api.meta.ai` and forces the configured model for primary and subagent defaults. Protocol compatibility does not establish that the CLI's local price table recognizes that model or matches Meta billing. This investigation establishes neither exact Meta dollar accounting nor a guaranteed dollar-triggered handoff. Inspect actual usage fields and pricing identity in a bounded probe before making that promise.

The repository [installer](https://github.com/lbacik/simple-coding-agent/blob/e125bd79495ccd24b07a0576bb303b71005fab1e/scripts/install-skills.sh) pins installer 0.6.0 and upstream commit `c55ee46073ed923f86ce59a5eb3b6d895095d1b7`, installing only `implement`, `tdd`, `code-review`, and `codebase-design`. Runtime skill selection and provenance require the same four names. The pinned upstream **does** contain [handoff](https://github.com/mattpocock/skills/blob/c55ee46073ed923f86ce59a5eb3b6d895095d1b7/skills/productivity/handoff/SKILL.md), but it is not installed here. It disables model-initiated invocation and asks for a conversation summary in the OS temporary directory, with suggested skills and references. It does not prescribe commits, issue publication, or the continuation contract. Adding its name alone does not satisfy the requested flow. Installation, invocation method, provenance, and durable artifact capture need an explicit implementation design; either keep the upstream artifact intact and adapt orchestration or define provenance for a project-owned artifact.

## What remains to validate

No paid probe was performed. A small approved disposable-worktree experiment is recommended before accepting exact threshold guarantees:

- Record the actual SDK and executed CLI identities, non-secret stream messages and hook order; compare deduplicated primary tool-use responses with `num_turns` at normal completion and the hard ceiling.
- Include one subagent; reconcile its events with final whole-tree accounting and determine whether available incremental signals cover every charged request.
- Trigger cooperative handoff and interrupt/follow-up separately; verify context retention, terminal reason fields, active-task termination, filesystem quiescence, and remaining budget behavior.
- Observe the Meta model's price identity and usage completeness. Treat missing or unrecognized pricing as unknown, not zero spend. Test a second same-client input and a fresh resumed client to confirm accounting boundaries.

Mocks can validate controller decisions and publication recovery without model spend, but cannot establish these provider/CLI behaviors. The remaining product decisions are reserve size, allowed handoff tools, fallback when handoff fails, durability before cleanup, human-visible state, and the branch/note continuation contract. This research resolves available mechanisms and uncertainty; it does not choose those policies or implement them.
