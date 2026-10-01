"""Unit tests for the pure per-attempt token ledger.

The ledger counts an attempt's tokens per thread (estimate first, replaced
by reported usage). It does no I/O and no logging: the executor feeds it
events and reads it.
"""

from simple_coding_agent.token_ledger import TokenLedger


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
