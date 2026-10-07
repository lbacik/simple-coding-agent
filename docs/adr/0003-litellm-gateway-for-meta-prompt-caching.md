# Route Meta model traffic through a separately deployed LiteLLM gateway

Since 2026-09-30, attempts on the Meta backend (`muse-spark-1.3-contributor`) get close to 0% prompt-cache reuse, down from 79% on 2026-09-21 (#169). Meta's cache lives on each backend replica and is reused reliably only when requests carry an affinity key, `prompt_cache_key`. The Claude Code CLI, which the Agent SDK runs, speaks only the Anthropic Messages protocol, and Meta's `/v1/messages` rejects that parameter (`unknown parameter 'prompt_cache_key'`). LiteLLM accepts Messages requests, translates them to Meta's Responses API, and derives `prompt_cache_key` from the `session_id` in the CLI's `metadata.user_id`. Measured on 2026-10-06, that brought cache reuse to 99.47% (probe `minimal`), 89.35% (probe `read-loop`) and 86.3% across all threads of a real `/implement` attempt (92.65% on the main thread). We decided to run LiteLLM as separate infrastructure, its own compose project on a Docker network shared with the agent, and point the agent at it with the existing backend settings (`MODEL_BASE_URL=http://litellm:4000`, `MODEL_API_KEY=<LITELLM_MASTER_KEY>`). The agent's own code and dependencies stay unchanged.

## Considered Options

- **Keep calling Meta's Messages endpoint directly.** Rejected: there is no way to send the affinity key, so near-zero cache reuse persists until Meta supports `prompt_cache_key` on `/v1/messages`. This option stays as the default deployment, and the gateway can be dropped once Meta adds that support.
- **An in-process proxy started by the agent for the Meta backend.** Rejected: it would add `litellm` and its dependency tree to the agent's pinned runtime, and tie gateway upgrades to agent releases. The HTTP hop exists either way, because the CLI is a subprocess that talks HTTP to `ANTHROPIC_BASE_URL`.
- **Our own minimal Messages→Responses translator.** Rejected: the translation covers streaming, tool use, reasoning and usage mapping (including `cache_read_input_tokens`, which cache observability relies on). That work is already done and tested in LiteLLM.

## Consequences

- The default `docker compose up` still connects directly to Meta. The gateway is opt-in through `docker-compose.litellm.yml`, which attaches the agent to the external network (`LITELLM_NETWORK`, default `main`).
- `META_API_KEY` lives only in the gateway. The agent holds the LiteLLM master key as `MODEL_API_KEY`.
- The model name stays the same, so the observed-model check, `docker/managed-settings.json` pricing and `token_budget_reconciled.prompt_cache` keep working unchanged.
- `prompt_cache_key` derivation is LiteLLM behaviour, not a documented contract, so the cache probe must be re-run after every LiteLLM version bump. The image is pinned (`ghcr.io/berriai/litellm:v1.104.0`) for this reason.
- Meta's `request-id` does not reach the agent through LiteLLM. Support tickets need the gateway's own logs.
- The gateway is a new point of failure. To the agent, an unreachable gateway looks the same as an unreachable model backend.
