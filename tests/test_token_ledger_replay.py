"""Replay the prototype timelines against the ledger.

The fixture holds the per-response timelines embedded in
``prototype-token-estimator/index.html`` (branch ``prototype/token-estimator``):
character counts and token usage only, no content. Each scenario mirrors one
walkthrough from the prototype's ``check.js``:

* #109 attempt started 07:58, main loop reported;
* the same attempt with nothing reported (the zero-usage case from #119);
* #109 attempt started 09:13, nothing reported;
* #129 probe run 2, main loop reported and the subagent unreported;
* synthetic Anthropic shape with usage repeated per content block.

The probe scenario uses the probe-calibrated subagent base (1.4K for its
tool-light custom agent, as the prototype did): the replay verifies the
ledger's arithmetic, while the conservative 15K production default is pinned
by ``test_token_ledger.py``.
"""

import json
from pathlib import Path

import pytest

from simple_coding_agent.token_ledger import TokenLedger

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "token_ledger_replay.json"
RECORDINGS = json.loads(FIXTURE_PATH.read_text())["recordings"]

SOFT_THRESHOLD_TOKENS = 3_200_000
HARD_CEILING_TOKENS = 4_000_000

#: Probe-calibrated subagent base: the #129 probe's tool-light custom agent
#: measured ~1.4K (see #127). The production default errs high instead.
PROBE_SUBAGENT_BASE_TOKENS = 1_400


def _usage_of(row: list) -> dict[str, int]:
    return {
        "input_tokens": row[4],
        "cache_read_input_tokens": row[5],
        "cache_creation_input_tokens": row[6],
        "output_tokens": row[7],
    }


def _replay(
    recording: list,
    *,
    main_live: bool,
    sub_live: bool = False,
    ledger: TokenLedger | None = None,
) -> dict:
    """Feed recording rows into a ledger, tracking truth and the crossing."""

    active = ledger if ledger is not None else TokenLedger()
    truth = 0
    responses_seen = 0
    crossed_at_response: int | None = None
    crossed_reading: float | None = None
    model_usage = {
        "input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "output_tokens": 0,
    }
    for row in recording:
        if row[0] == "s":
            _, thread, kind, prompt_chars = row
            active.start_thread(thread, kind=kind, prompt_chars=prompt_chars)
        elif row[0] == "r":
            _, thread, response_id, visible_chars = row[:4]
            usage = _usage_of(row)
            for category, value in usage.items():
                model_usage[category] += value
            truth += sum(usage.values())
            responses_seen += 1
            active.observe_response(thread, response_id, visible_chars)
            live = main_live if thread == "main" else sub_live
            if live:
                active.observe_usage(thread, response_id, usage)
        elif row[0] == "i":
            _, thread, chars = row
            active.observe_input(thread, chars)
        if crossed_at_response is None and active.budget_tokens >= SOFT_THRESHOLD_TOKENS:
            crossed_at_response = responses_seen
            crossed_reading = active.budget_tokens
    reconciled = active.reconcile({"replay-model": model_usage})
    assert reconciled.actual_tokens == truth
    return {
        "error": reconciled.error_ratio,
        "crossed_at_response": crossed_at_response,
        "crossed_reading": crossed_reading,
        "reconciled": reconciled,
    }


def test_0758_with_main_loop_reported_matches_prototype() -> None:
    replayed = _replay(RECORDINGS["issue109_0758"], main_live=True)

    assert replayed["error"] is not None
    assert abs(replayed["error"]) <= 0.005
    assert replayed["crossed_at_response"] == 49


def test_0758_with_nothing_reported_matches_prototype() -> None:
    replayed = _replay(RECORDINGS["issue109_0758"], main_live=False)

    assert replayed["error"] is not None
    assert abs(replayed["error"]) <= 0.01
    assert replayed["crossed_at_response"] == 49


def test_0913_with_nothing_reported_matches_prototype() -> None:
    replayed = _replay(RECORDINGS["issue109_0913"], main_live=False)

    assert replayed["error"] is not None
    assert abs(replayed["error"]) <= 0.03
    assert replayed["crossed_at_response"] in (53, 54)
    assert replayed["crossed_reading"] is not None
    assert replayed["crossed_reading"] < HARD_CEILING_TOKENS


def test_probe_with_subagent_unreported_matches_prototype() -> None:
    ledger = TokenLedger(subagent_system_base_tokens=PROBE_SUBAGENT_BASE_TOKENS)
    replayed = _replay(RECORDINGS["probe129_run2"], main_live=True, ledger=ledger)

    unreported = replayed["reconciled"]
    assert unreported.mode == "mixed"
    assert unreported.unreported_error_ratio is not None
    assert abs(unreported.unreported_error_ratio) <= 0.02


def test_repeated_per_block_usage_is_counted_once() -> None:
    ledger = TokenLedger()
    ledger.start_thread("main", kind="main", prompt_chars=26_000)
    truth = 0

    def respond(response_id: str, usage_input: int, cache_read: int, output: int) -> None:
        nonlocal truth
        start = {
            "input_tokens": usage_input,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": 0,
            "output_tokens": 1,
        }
        final = {**start, "output_tokens": output}
        truth += sum(final.values())
        ledger.observe_response("main", response_id, visible_chars=900)
        for _ in range(3):
            ledger.observe_usage("main", response_id, start)
        ledger.observe_usage("main", response_id, final)

    respond("a1", 3_000, 29_000, 420)
    ledger.observe_input("main", chars=6_000)
    respond("a2", 1_600, 32_000, 380)

    reconciled = ledger.reconcile({"replay-model": {
        "input_tokens": 4_600,
        "cache_read_input_tokens": 61_000,
        "cache_creation_input_tokens": 0,
        "output_tokens": 800,
    }})

    assert reconciled.actual_tokens == truth
    assert reconciled.error_ratio == pytest.approx(0.0, abs=1e-12)
