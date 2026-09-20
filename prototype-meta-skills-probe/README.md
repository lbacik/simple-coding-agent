# PROTOTYPE: Meta + skills execution probe

Disposable artifact for [ticket #8](https://github.com/lbacik/simple-coding-agent/issues/8)
on the [MVP map](https://github.com/lbacik/simple-coding-agent/issues/1). Lives
on the throwaway `prototype/meta-skills-probe` branch; nothing here ships.

## What this checks (reduced-core scope, agreed with the human)

- `claude-agent-sdk==0.2.156` actually drives `@anthropic-ai/claude-code@2.1.276`
  (the CLI version that SDK release expects) against Meta's `muse-spark-1.3`
  through the [documented](https://dev.meta.ai/docs/agent-frameworks) env
  contract from the [compatibility research](https://github.com/lbacik/simple-coding-agent/issues/2).
- The pinned skill bundle (`mattpocock/skills@c55ee46073ed923f86ce59a5eb3b6d895095d1b7`,
  installed by `agent-installer@0.6.0` per the
  [installer research](https://github.com/lbacik/simple-coding-agent/issues/3))
  is discoverable and `/implement` actually dispatches it.
- The agent edits the fixture, runs its own test, and the result is verified
  independently (this script reruns `pytest` outside the agent loop and diffs
  the fixture's git history).
- Every observed `AssistantMessage.model` matches the requested model (no
  silent fallback/substitution).
- Which `code-review` actually ran is visible in the transcript (the upstream
  skill vs. the CLI's own bundled one), by inspecting the `Skill` tool_use
  blocks and the diff both reviewers were shown.
- The terminal `ResultMessage` is inspected directly (`is_error`, `usage`,
  `model_usage`, `structured_output`), not just the agent's own narration.

## What this deliberately excludes

Per the reduced-scope decision on ticket #8: session resume/context replay,
mid-run interruption/cancellation, injected transient/permanent transport
failures, and reconciliation against Meta's own account usage dashboard. The
[full 7-step probe](https://github.com/lbacik/simple-coding-agent/issues/2)
describes these; they remain open follow-up if this reduced probe passes.

## Running it

1. `cp .env.example .env` and put a real Meta Model API key in it. `.env` is
   gitignored; never commit it once it holds a real value.
2. `./run.sh` — builds the image (no model calls at build time beyond the
   `agent-installer` GitHub fetch) and runs the probe container once.
3. Read `probe-report.json` (redacted env, observed vs. requested model,
   every `Skill` tool_use, the terminal result, the independent pytest run,
   and the fixture's git diff) and `probe-transcript.log` (full message
   stream) next to this README.

## Verdict

Two independent live runs against Meta's API, both `2026-09-20`. Full stdout
(including cost/usage) of both is in `full-run.log`.

- **Model actually used:** `muse-spark-1.3-contributor`, not the pinned
  `muse-spark-1.3` candidate from
  [issue #2](https://github.com/lbacik/simple-coding-agent/issues/2). This
  was a deliberate, human-directed deviation for this run, not an agent
  substitution — flagged explicitly because issue #2 says not to silently
  substitute a contributor model. **Follow-up needed:** rerun with
  `muse-spark-1.3` before treating the standard identifier as verified, or
  update issue #2's contract if the contributor model is the intended target.
- **Result:** PASS on every reduced-core check.
  - `/implement` dispatched, which called the `tdd` skill (red confirmed,
    fix applied, green), then `code-review`, matching the upstream skill's
    documented Standards+Spec parallel-subagent design exactly (two
    background tasks named "Run Standards review" / "Run Spec review", each
    with its own token usage) — this is the audited upstream skill winning
    the collision, not a generic bundled review.
  - Fixture actually edited (`calc.py`: `a - b` → `a + b`), committed
    (`63e1cb3`, then `06bfd90` on the second run), and independently rerun
    with `pytest` outside the agent loop: `1 passed`.
  - `routing_matches_requested: true` both times — every observed
    `AssistantMessage.model` and the `ResultMessage.model_usage` key matched
    the requested model, no fallback.
  - Terminal `ResultMessage.is_error` was `false`, `stop_reason` `end_turn`,
    `structured_output` was not requested/exercised here (excluded from this
    reduced probe).
  - One non-fatal surprise: the CLI printed
    `[claude-code:unrecognized_model] {"model":"muse-spark-1.3-contributor","query_source":"sdk"}`
    to stderr on both runs before doing anything else. It did not block the
    session. Worth re-checking with the standard `muse-spark-1.3` id.
  - Cost/usage was present and consistent in both `ResultMessage.usage` and
    `model_usage[model]` (~$0.72 and ~$0.80 for the two runs;
    `costBasis: "unknown"` — Meta-side accounting is not yet independently
    reconciled, which is explicitly out of scope for this reduced probe).
- **Accepted / rejected:** _pending discussion with the human_
- **Follow-up:** rerun with `muse-spark-1.3` (not `-contributor`); the
  excluded full-7-step checks (resume, interrupt, error injection, provider
  usage reconciliation) remain open per the reduced-scope decision.
