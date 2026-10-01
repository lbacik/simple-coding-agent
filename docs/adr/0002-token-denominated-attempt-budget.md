# Budget each attempt in tokens, estimated locally and replaced by measurement

The agent used to budget an implementation attempt in USD, from `AssistantMessage.usage` priced with a hard-coded rate table, and fell back to a flat cost per turn when usage was zero. On the Meta backend (`muse-spark-1.3-contributor`), live usage is always zero, so the estimate ran about 3x low, the soft threshold never fired, and attempts ended at the hard ceiling without a model-written handoff note (#119). USD is not even measurable there, because the Claude Code CLI prices a non-Anthropic model from its own table. We decided that the token budget counts the unweighted sum of all four token categories, subagents included. It is fed by a per-thread ledger that counts each response as a content-based estimate the moment it appears, and replaces the estimate with reported usage when the backend sends any. The agent enforces both the soft threshold and the hard ceiling on that ledger, and keeps the SDK's `max_budget_usd` only as a looser backstop. The full reasoning lives on map #123.

## Considered Options

- **Keep USD and fix the fallback** (a more conservative per-turn cost). Rejected: a per-turn cost is unrelated to how much context a turn carries, and USD for Meta is not a measured bill.
- **Per-backend profiles** (declare per model whether usage is reported). Rejected: one mechanism that detects missing usage per response, and logs it, covers every backend without a registry.
- **Weight tokens by price.** Rejected: caching differs across backends (near zero on Meta, dominant on Anthropic), so the same work would get a different budget.
- **Use only `ResultMessage.model_usage`.** Rejected: it is authoritative but arrives at the end of the attempt, too late to drive the soft threshold. It is used for reconciliation instead.

## Consequences

- The estimate's error is recorded per attempt (`token_budget` in `attempt.json`), and the constants are tuned by hand from those records. Nothing is carried over between attempts automatically.
- Base-context constants err high (15K tokens per subagent by default), because an overestimate only moves the soft threshold earlier.
