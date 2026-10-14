# Synthetic telemetry examples (schema v1)

These files are **synthetic**: invented repositories (`example-owner/...`),
invented attempt ids, and `example.invalid` URLs. No data was captured from
live instances. The dashboard repository is public and copies them.

- `snapshot-idle-with-last-finished.json` — idle agent, no active attempt,
  last finished attempt present.
- `snapshot-attempt-in-progress.json` — attempt in progress with a partly
  estimated budget and running subagents.
- `snapshot-soft-threshold-handoff.json` — soft threshold crossed (reserve
  rule) with an operator handoff in progress.
- `snapshot-recovery-hold.json` — recovery hold with its next action; the
  snapshot performs no network I/O so publication progress is "not known
  yet".
- `snapshot-truncated.json` — trimmed snapshot with `truncated: true`
  (events trimmed oldest-first, subagents trimmed).
- `error-agent-unreachable.json`, `error-unsupported-operation.json`,
  `error-snapshot-failed.json` — one machine-readable `agentctl telemetry`
  error reply per code, as printed to stdout.

Contract: `docs/agents/telemetry-snapshot.md`. Conformance test:
`tests/test_telemetry_examples.py`.
