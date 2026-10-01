"""Unit tests for the pure per-attempt token ledger.

The ledger counts an attempt's tokens per thread (estimate first, replaced
by reported usage). It does no I/O and no logging: the executor feeds it
events and reads it.
"""

from simple_coding_agent.token_ledger import (
    DEFAULT_SUBAGENT_SYSTEM_BASE_TOKENS,
    TokenLedger,
    subagent_system_base,
)


def test_main_thread_starts_at_system_base_plus_prompt() -> None:
    ledger = TokenLedger()

    ledger.start_thread("main", kind="main", prompt_chars=52_265)

    assert ledger.context_tokens("main") == 19_000 + 52_265 / 4.0


def test_response_is_counted_as_an_estimate_and_grows_the_context() -> None:
    ledger = TokenLedger()
    ledger.start_thread("main", kind="main", prompt_chars=0)

    ledger.observe_response("main", "r1", visible_chars=120)

    estimate = 19_000 + 120 / 3.0 + 150
    assert ledger.budget_tokens == estimate
    assert ledger.measured_tokens == 0
    assert ledger.estimated_tokens == estimate
    assert ledger.context_tokens("main") == estimate


def test_tool_result_grows_the_thread_context() -> None:
    ledger = TokenLedger()
    ledger.start_thread("main", kind="main", prompt_chars=0)

    ledger.observe_input("main", chars=1_048)

    assert ledger.context_tokens("main") == 19_000 + 1_048 / 4.0 + 60
    assert ledger.budget_tokens == 0


def test_reported_usage_replaces_the_estimate_and_reanchors_the_context() -> None:
    ledger = TokenLedger()
    ledger.start_thread("main", kind="main", prompt_chars=0)
    ledger.observe_response("main", "r1", visible_chars=120)
    ledger.observe_input("main", chars=1_048)

    ledger.observe_usage(
        "main",
        "r1",
        {
            "input_tokens": 1_000,
            "cache_read_input_tokens": 2_000,
            "cache_creation_input_tokens": 500,
            "output_tokens": 300,
        },
    )

    assert ledger.measured_tokens == 3_800
    assert ledger.estimated_tokens == 0
    assert ledger.budget_tokens == 3_800
    assert ledger.context_tokens("main") == 3_800


def test_repeated_report_merges_by_max_and_never_adds() -> None:
    ledger = TokenLedger()
    ledger.start_thread("main", kind="main", prompt_chars=0)
    ledger.observe_response("main", "r1", visible_chars=900)
    start_usage = {
        "input_tokens": 3_000,
        "cache_read_input_tokens": 29_000,
        "cache_creation_input_tokens": 0,
        "output_tokens": 1,
    }
    final_usage = {**start_usage, "output_tokens": 420}

    ledger.observe_usage("main", "r1", start_usage)
    ledger.observe_usage("main", "r1", start_usage)
    ledger.observe_usage("main", "r1", final_usage)

    assert ledger.budget_tokens == 3_000 + 29_000 + 420


def test_all_zero_usage_is_ignored_like_a_missing_report() -> None:
    ledger = TokenLedger()
    ledger.start_thread("main", kind="main", prompt_chars=0)
    estimate = ledger.observe_response("main", "r1", visible_chars=120)

    ledger.observe_usage(
        "main",
        "r1",
        {"input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0},
    )

    assert ledger.measured_tokens == 0
    assert ledger.estimated_tokens == estimate


def test_repeat_assistant_message_for_one_response_is_counted_once() -> None:
    ledger = TokenLedger()
    ledger.start_thread("main", kind="main", prompt_chars=0)

    first = ledger.observe_response("main", "r1", visible_chars=900)
    second = ledger.observe_response("main", "r1", visible_chars=900)

    assert second == first
    assert ledger.budget_tokens == first


def test_subagent_thread_is_keyed_by_parent_tool_use_id() -> None:
    ledger = TokenLedger()
    ledger.start_thread("main", kind="main", prompt_chars=0)
    ledger.start_thread("toolu_123", kind="subagent", prompt_chars=65, subagent_type="line-counter")

    ledger.observe_response("toolu_123", "s1", visible_chars=117)
    ledger.observe_input("toolu_123", chars=48)
    ledger.observe_response("toolu_123", "s2", visible_chars=150)

    assert set(ledger.thread_contexts()) == {"main", "toolu_123"}
    assert ledger.context_tokens("main") == 19_000
    assert ledger.measured_tokens == 0
    assert ledger.estimated_tokens > 0


def test_subagent_base_is_conservative_by_default_for_known_and_unknown_types() -> None:
    assert DEFAULT_SUBAGENT_SYSTEM_BASE_TOKENS == 15_000
    assert subagent_system_base("Explore") == 15_000
    assert subagent_system_base("general-purpose") == 15_000
    assert subagent_system_base("line-counter") == 15_000
    assert subagent_system_base(None) == 15_000

    ledger = TokenLedger()
    ledger.start_thread("main", kind="main", prompt_chars=0)
    ledger.start_thread("toolu_1", kind="subagent", prompt_chars=0, subagent_type="Explore")

    assert ledger.context_tokens("toolu_1") == 15_000


def test_ledger_accepts_per_type_subagent_base_overrides() -> None:
    ledger = TokenLedger(subagent_bases_by_type={"custom": 1_400})
    ledger.start_thread("toolu_1", kind="subagent", prompt_chars=0, subagent_type="custom")
    ledger.start_thread("toolu_2", kind="subagent", prompt_chars=0, subagent_type="other")

    assert ledger.context_tokens("toolu_1") == 1_400
    assert ledger.context_tokens("toolu_2") == 15_000

def test_first_response_that_settles_without_usage_is_reported_once() -> None:
    ledger = TokenLedger()
    ledger.start_thread("main", kind="main", prompt_chars=0)
    ledger.start_thread("toolu_9", kind="subagent", prompt_chars=0, subagent_type="Explore")
    ledger.observe_response("main", "r1", visible_chars=100)

    assert ledger.first_settled_without_usage is None

    ledger.observe_input("main", chars=10)
    notice = ledger.first_settled_without_usage
    assert notice is not None
    assert notice.thread == "main"
    assert notice.subagent_type is None

    ledger.observe_response("toolu_9", "s1", visible_chars=100)
    ledger.observe_response("toolu_9", "s2", visible_chars=100)

    settled_again = ledger.first_settled_without_usage
    assert settled_again is notice


def test_reported_response_settling_first_leaves_no_degraded_notice() -> None:
    ledger = TokenLedger()
    ledger.start_thread("main", kind="main", prompt_chars=0)
    ledger.observe_response("main", "r1", visible_chars=100)
    ledger.observe_usage(
        "main",
        "r1",
        {"input_tokens": 500, "cache_read_input_tokens": 0, "output_tokens": 50},
    )
    ledger.observe_response("main", "r2", visible_chars=100)

    assert ledger.first_settled_without_usage is None


def test_peak_main_context_tracks_the_highest_held_context() -> None:
    ledger = TokenLedger()
    ledger.start_thread("main", kind="main", prompt_chars=0)
    base = ledger.peak_main_context_tokens
    assert base == 19_000

    ledger.observe_response("main", "r1", visible_chars=300)
    grown = ledger.peak_main_context_tokens
    assert grown == 19_000 + 300 / 3.0 + 150

    ledger.observe_usage(
        "main",
        "r1",
        {"input_tokens": 1_000, "cache_read_input_tokens": 0, "output_tokens": 100},
    )

    assert ledger.peak_main_context_tokens == grown
    assert ledger.context_tokens("main") == 1_100


def test_subagent_context_does_not_move_the_main_peak() -> None:
    ledger = TokenLedger()
    ledger.start_thread("main", kind="main", prompt_chars=0)
    ledger.start_thread("toolu_1", kind="subagent", prompt_chars=0, subagent_type="Explore")
    ledger.observe_response("toolu_1", "s1", visible_chars=300)

    assert ledger.peak_main_context_tokens == 19_000


def test_reconcile_reports_mode_estimate_actual_and_unreported_part() -> None:
    ledger = TokenLedger()
    ledger.start_thread("main", kind="main", prompt_chars=0)
    ledger.start_thread("toolu_1", kind="subagent", prompt_chars=0, subagent_type="Explore")
    ledger.observe_response("main", "r1", visible_chars=300)
    ledger.observe_usage(
        "main",
        "r1",
        {"input_tokens": 1_000, "cache_read_input_tokens": 0, "output_tokens": 100},
    )
    ledger.observe_response("toolu_1", "s1", visible_chars=150)
    sub_estimate = 15_000 + 150 / 3.0 + 150

    result = ledger.reconcile({"m": {"input_tokens": 1_000, "output_tokens": 100 + 500}})

    assert result.mode == "mixed"
    assert result.estimated_tokens == 1_100 + sub_estimate
    assert result.actual_tokens == 1_600
    assert result.error_ratio == (1_100 + sub_estimate - 1_600) / 1_600
    assert result.unreported_estimated_tokens == sub_estimate
    assert result.unreported_actual_tokens == 500
    assert result.unreported_error_ratio == (sub_estimate - 500) / 500


def test_reconcile_without_model_usage_leaves_actuals_and_errors_empty() -> None:
    ledger = TokenLedger()
    ledger.start_thread("main", kind="main", prompt_chars=0)
    ledger.observe_response("main", "r1", visible_chars=300)

    result = ledger.reconcile(None)

    assert result.mode == "estimated"
    assert result.estimated_tokens == ledger.budget_tokens
    assert result.actual_tokens is None
    assert result.error_ratio is None
    assert result.unreported_actual_tokens is None
    assert result.unreported_error_ratio is None


def test_reconcile_accepts_camel_case_model_usage_without_thinking() -> None:
    ledger = TokenLedger()
    ledger.start_thread("main", kind="main", prompt_chars=0)
    ledger.observe_response("main", "r1", visible_chars=0)
    ledger.observe_usage(
        "main",
        "r1",
        {
            "inputTokens": 1_000,
            "cacheReadInputTokens": 2_000,
            "cacheCreationInputTokens": 0,
            "outputTokens": 100,
            "thinkingTokens": 90,
            "costUSD": 0.01,
        },
    )

    result = ledger.reconcile(
        {
            "m": {
                "inputTokens": 1_000,
                "cacheReadInputTokens": 2_000,
                "cacheCreationInputTokens": 0,
                "outputTokens": 100,
                "thinkingTokens": 90,
            }
        }
    )

    assert result.mode == "measured"
    assert result.actual_tokens == 3_100
    assert result.error_ratio == 0




def test_executor_readings_cover_thread_presence_counts_and_settled_ids() -> None:
    ledger = TokenLedger()

    assert ledger.has_thread("main") is False

    ledger.start_thread("main", kind="main", prompt_chars=0)
    ledger.observe_response("main", "r1", visible_chars=0)
    ledger.observe_response("main", "r1", visible_chars=999)

    assert ledger.has_thread("main") is True
    assert ledger.has_thread("toolu-1") is False
    # The repeat for the same id is not counted twice.
    assert ledger.response_count("main") == 1
    # Nothing settled yet: only the next event on the thread settles r1.
    assert ledger.settled_response_ids("main") == ()

    ledger.observe_response("main", "r2", visible_chars=0)

    assert ledger.response_count("main") == 2
    assert ledger.settled_response_ids("main") == ("r1",)
