# Local token estimator prototype

PROTOTYPE, disposable. Asset for ticket #127 "Design the local token estimator" (map #123, origin #119).

- `index.html`: open it by double-click. A pure `TokenEstimator` module (the part worth lifting into `simple_coding_agent/model_execution.py`) and a throwaway page with six guided walkthroughs plus free play.
- `check.js`: `node prototype-token-estimator/check.js` replays every walkthrough headlessly and prints the reconciliation.
- `extract.py`: turns a CLI transcript JSONL into the per-response timeline (visible chars, usage) embedded in `index.html`.

The embedded data holds only character counts and token usage, no content:

- #109 attempts started 2026-09-30T07:58Z and 09:13Z: 77 main-loop responses each, from the two surviving CLI transcripts.
- #129 probe run 2: the main loop plus one subagent.

## Model

Each model response is counted as soon as its `AssistantMessage` is seen, first as an estimate:

- the thread's context + visible output / 3.0 + a thinking allowance of 150 tokens;
- tool results add chars / 4.0 + 60 framing tokens to the thread's context.

When usage is reported for that response (`message_delta`, or a non-zero `AssistantMessage.usage`), it is merged per category by max and replaces the estimate. The thread's context is then re-anchored on the measured value.

A response that settles without usage (the next event on its thread arrives first) keeps its estimate. The first such response logs a WARNING.

At `ResultMessage`, the count is reconciled with `model_usage`. The unreported part is the total minus the measured part.

## Replay results (`check.js`)

| scenario | counted vs actual | soft threshold crossed at |
|---|---|---|
| #109 07:58, main reported | +0.0% | response 49 (truth 49) |
| #109 07:58, nothing reported | −0.5% | 49 (truth 49) |
| #109 09:13, nothing reported | +2.5% | 53 (truth 54) |
| probe, subagent unreported | subagent part −0.8% | — |
| synthetic: 6 subagents, real base 15K, assumed 1.4K | −49% | not crossed (truth: response 114) |
| synthetic Anthropic, repeated per-block usage | +0.0% (counted once) | — |
