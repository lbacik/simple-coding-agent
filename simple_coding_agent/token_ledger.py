"""Per-attempt token ledger: estimate first, replaced by reported usage.

One ledger lives for one implementation attempt, with one thread per
conversation: ``main`` for the main loop plus one per subagent, keyed by
``parent_tool_use_id``.

The model (decided in #127 from the ``prototype/token-estimator`` replay):

* A response is counted on its first ``AssistantMessage`` as an estimate:
  the thread's context plus ``visible output chars / OUTPUT_CHARS_PER_TOKEN``
  plus a thinking allowance. The output estimate is added to the thread's
  context.
* Reported usage (a ``message_delta`` stream event, or a non-zero
  ``AssistantMessage.usage``) is merged into the response per category by
  max, keyed by response id, never added. Once a response has usage it
  counts as measured, and the thread's context is re-anchored to the
  measured ``input + cache_read + cache_creation + output``.
* A tool result or user message adds ``chars / INPUT_CHARS_PER_TOKEN``
  plus framing tokens to the thread's context.
* A new thread starts at a system base plus ``prompt chars /
  INPUT_CHARS_PER_TOKEN``. The subagent base is per subagent type.
* A response settles when the next event on its thread arrives. The first
  response per attempt that settles without reported usage is reported via
  :attr:`TokenLedger.first_settled_without_usage` so the executor can log
  ``token_estimate_degraded``. The ledger captures a
  :class:`SettledReading` for each settled response at settle time, before
  the event that settled it is applied, so ``limits_checked`` can report
  the settled response's own figures instead of the live ledger state
  (which already carries the next response's estimate).
* A response cut off by an interrupt (for example the hard ceiling firing
  right after its first ``AssistantMessage``, before usage arrives) is
  flagged via :meth:`TokenLedger.mark_interrupted`. Interrupted estimates
  stay in ``budget_tokens``/``estimated_tokens`` (enforcement is
  unchanged) but are reported separately and excluded from the
  reconciliation ``mode``/``error_ratio``, which cover completed
  responses only.

Compaction needs no handling here: the main thread re-anchors to the
measured context at its next reported response, and estimated threads are
deliberately not reset (their overestimation can only move the soft
threshold earlier).

The module is pure: no I/O, no logging, and no SDK imports. Usage arrives
as plain mappings, so both the snake_case CLI usage shape and the
camelCase ``ResultMessage.model_usage`` shape are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


# ---------------------------------------------------------------------------
# Constants (kept in one place; fitted on the #109 transcripts, see #127)
# ---------------------------------------------------------------------------

#: Tool-result / prompt characters per token (observed IQR 3.6-4.2).
INPUT_CHARS_PER_TOKEN = 4.0
#: Visible-output characters per token (summed over an attempt).
OUTPUT_CHARS_PER_TOKEN = 3.0
#: Framing tokens added per tool result / user message.
FRAMING_TOKENS_PER_MESSAGE = 60
#: Thinking allowance per response (Meta thinking is opaque; mean 150-245).
THINKING_ALLOWANCE_TOKENS = 150
#: Main-loop system base context in tokens.
MAIN_SYSTEM_BASE_TOKENS = 19_000
#: Conservative subagent system base in tokens (see below).
DEFAULT_SUBAGENT_SYSTEM_BASE_TOKENS = 15_000
#: Subagent system base per subagent type. Unknown types fall back to the
#: conservative default: erring high moves the soft threshold earlier,
#: never later. The probe in #129 measured ~1.4K for its tool-light custom
#: agent, which is why the default errs high instead of fitting it.
SUBAGENT_SYSTEM_BASE_TOKENS_BY_TYPE = {
    "Explore": 15_000,
    "general-purpose": 15_000,
}

#: Token categories counted by the budget, unweighted. ``thinkingTokens``
#: is not a category: thinking is already inside ``output_tokens``.
USAGE_CATEGORIES = (
    "input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "output_tokens",
)

_CAMEL_CASE_ALIASES = {
    "input_tokens": "inputTokens",
    "cache_read_input_tokens": "cacheReadInputTokens",
    "cache_creation_input_tokens": "cacheCreationInputTokens",
    "output_tokens": "outputTokens",
}


def _category_value(usage: Mapping[str, Any], category: str) -> int:
    value = usage.get(category, None)
    if value is None:
        value = usage.get(_CAMEL_CASE_ALIASES[category], 0)
    return int(value or 0)


def usage_by_category(usage: Mapping[str, Any]) -> dict[str, int]:
    """Per-category token values of one usage report (either naming shape)."""

    return {category: _category_value(usage, category) for category in USAGE_CATEGORIES}


def subagent_system_base(subagent_type: str | None) -> int:
    """Return the system base for a subagent type (default: conservative)."""

    return SUBAGENT_SYSTEM_BASE_TOKENS_BY_TYPE.get(
        subagent_type or "", DEFAULT_SUBAGENT_SYSTEM_BASE_TOKENS
    )


def usage_total(usage: Mapping[str, Any]) -> int:
    """Sum the four budget categories of a usage mapping, unweighted."""

    return sum(usage_by_category(usage).values())


def _total_input_of(usage: Mapping[str, int]) -> int:
    """One measured response's total input (output excluded)."""

    return usage["input_tokens"] + usage["cache_read_input_tokens"] + usage["cache_creation_input_tokens"]


def hit_rate_of(totals: Mapping[str, int]) -> float | None:
    """Cache-read share of summed input; ``None`` when the denominator is 0."""

    denominator = (
        totals["input_tokens"]
        + totals["cache_read_input_tokens"]
        + totals["cache_creation_input_tokens"]
    )
    if denominator == 0:
        return None
    return totals["cache_read_input_tokens"] / denominator


@dataclass(frozen=True)
class DegradedNotice:
    """The first response that settled without reported usage."""

    thread: str
    subagent_type: str | None
    response_id: str


@dataclass(frozen=True)
class MainCacheStats:
    """Prompt-cache effectiveness over an attempt's measured main-thread responses."""

    hit_rate: float | None
    miss_tokens: int
    counted: int
    excluded: int


@dataclass(frozen=True)
class ResponseCacheMetrics:
    """One measured response's own cache figures (``miss_tokens`` is ``None`` when excluded)."""

    input_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    hit_rate: float | None
    miss_tokens: int | None


@dataclass(frozen=True)
class Reconciliation:
    """Budget count checked against the authoritative ``model_usage`` total.

    ``estimated_tokens`` is the full ledger total, including interrupted
    in-flight estimates (enforcement is unchanged). ``mode``,
    ``error_ratio`` and the ``unreported_*`` fields cover completed
    responses only: interrupted estimates are excluded and reported via
    ``interrupted_estimated_tokens``.
    """

    mode: Literal["measured", "mixed", "estimated"]
    estimated_tokens: float
    actual_tokens: int | None
    error_ratio: float | None
    unreported_estimated_tokens: float
    unreported_actual_tokens: int | None
    unreported_error_ratio: float | None
    interrupted_estimated_tokens: float


@dataclass(frozen=True)
class SettledReading:
    """Ledger aggregates captured when one response settled.

    Taken before the event that settled it is applied, so the figures are
    the settled response's own: they exclude the next response's estimate
    (when the settler is the next response's first ``AssistantMessage``)
    and any tool-input growth (when the settler is a tool result).
    """

    budget_tokens: float
    measured_tokens: float
    estimated_tokens: float
    main_context_tokens: float
    main_response_count: int


@dataclass
class _Response:
    """One counted response: an estimate until usage replaces it."""

    id: str
    estimate: float
    output_estimate: float
    usage: dict[str, int] | None = None
    settled: bool = False
    compacted_before: bool = False
    interrupted: bool = False

    def counted(self) -> float:
        """Tokens this response contributes to the budget reading."""

        if self.usage is None:
            return self.estimate
        return float(sum(self.usage[category] for category in USAGE_CATEGORIES))


@dataclass
class _Thread:
    """One conversation: the main loop or one subagent."""

    id: str
    kind: Literal["main", "subagent"]
    subagent_type: str | None
    context: float
    responses: list[_Response] = field(default_factory=list)


class TokenLedger:
    """Counts one attempt's tokens per thread (estimate first, then measured)."""

    def __init__(
        self,
        *,
        main_system_base: float = MAIN_SYSTEM_BASE_TOKENS,
        subagent_system_base_tokens: float = DEFAULT_SUBAGENT_SYSTEM_BASE_TOKENS,
        subagent_bases_by_type: Mapping[str, float] | None = None,
    ) -> None:
        self._main_system_base = main_system_base
        self._subagent_system_base = subagent_system_base_tokens
        if subagent_bases_by_type is None:
            self._subagent_bases_by_type = dict(SUBAGENT_SYSTEM_BASE_TOKENS_BY_TYPE)
        else:
            self._subagent_bases_by_type = dict(subagent_bases_by_type)
        self._threads: dict[str, _Thread] = {}
        self._first_settled_without_usage: DegradedNotice | None = None
        self._settled_readings: dict[tuple[str, str], SettledReading] = {}
        self._peak_main_context_tokens = 0.0
        self._compaction_pending: set[str] = set()

    # -- events ----------------------------------------------------------

    def start_thread(
        self,
        thread_id: str,
        *,
        kind: Literal["main", "subagent"],
        prompt_chars: int,
        subagent_type: str | None = None,
    ) -> None:
        """Start a thread at its system base plus ``prompt_chars``."""

        if thread_id in self._threads:
            raise ValueError(f"thread {thread_id!r} already started")
        if kind == "main":
            base = self._main_system_base
        elif kind == "subagent":
            base = self._subagent_bases_by_type.get(subagent_type or "", self._subagent_system_base)
        else:
            raise ValueError(f"unknown thread kind {kind!r}")
        thread = _Thread(
            id=thread_id,
            kind=kind,
            subagent_type=subagent_type if kind == "subagent" else None,
            context=base + prompt_chars / INPUT_CHARS_PER_TOKEN,
        )
        self._threads[thread_id] = thread
        self._compaction_pending.discard(thread_id)
        self._track_peak(thread)

    def observe_response(self, thread_id: str, response_id: str, visible_chars: int) -> float:
        """Count a response on its first ``AssistantMessage`` as an estimate.

        Repeats for the same response id (one ``AssistantMessage`` per
        content block) are ignored: each response is counted once.
        """

        thread = self._threads[thread_id]
        for response in thread.responses:
            if response.id == response_id:
                return response.counted()
        self._settle(thread)
        output_estimate = visible_chars / OUTPUT_CHARS_PER_TOKEN + THINKING_ALLOWANCE_TOKENS
        estimate = thread.context + output_estimate
        thread.responses.append(
            _Response(
                id=response_id,
                estimate=estimate,
                output_estimate=output_estimate,
                compacted_before=thread_id in self._compaction_pending,
            )
        )
        self._compaction_pending.discard(thread_id)
        thread.context += output_estimate
        self._track_peak(thread)
        return estimate

    def observe_usage(
        self, thread_id: str, response_id: str, usage: Mapping[str, Any] | None
    ) -> None:
        """Merge reported usage into a response per category by max.

        All-zero usage carries no measurement and is ignored, like a
        missing report. The thread's context is re-anchored to the measured
        ``input + cache_read + cache_creation + output``.
        """

        if usage is None:
            return
        merged = {category: _category_value(usage, category) for category in USAGE_CATEGORIES}
        if sum(merged.values()) == 0:
            return
        thread = self._threads[thread_id]
        response = self._response_for_usage(thread, response_id)
        if response is None:
            return
        if response.usage is None:
            response.usage = merged
        else:
            for category in USAGE_CATEGORIES:
                response.usage[category] = max(response.usage[category], merged[category])
        thread.context = (
            response.usage["input_tokens"]
            + response.usage["cache_read_input_tokens"]
            + response.usage["cache_creation_input_tokens"]
            + response.usage["output_tokens"]
        )
        self._track_peak(thread)

    def observe_input(self, thread_id: str, chars: int) -> None:
        """Grow a thread's context with a tool result or user message."""

        thread = self._threads[thread_id]
        self._settle(thread)
        thread.context += chars / INPUT_CHARS_PER_TOKEN + FRAMING_TOKENS_PER_MESSAGE
        self._track_peak(thread)

    def notify_compaction(self, thread_id: str) -> None:
        """Record a context compaction on a thread (pure: state only, no I/O).

        The next response counted on that thread is excluded from
        cache-miss accounting: its prefix was rewritten, so the previous
        response's input is no longer the expected cache read.
        """

        self._compaction_pending.add(thread_id)

    def reconcile(
        self, model_usage: Mapping[str, Mapping[str, Any]] | None
    ) -> Reconciliation:
        """Check the count against the full ``model_usage`` (all models).

        With no ``model_usage`` every actual and error is ``None``.
        Interrupted in-flight estimates stay in ``estimated_tokens`` but
        are excluded from ``mode`` and the error ratios, which cover
        completed responses only.
        """

        for thread in self._threads.values():
            self._settle(thread)
        estimated_total = self.budget_tokens
        measured = self.measured_tokens
        interrupted = self.interrupted_estimated_tokens
        covered_estimated = estimated_total - interrupted
        covered_unreported = self.estimated_tokens - interrupted
        if covered_unreported > 0:
            mode: Literal["measured", "mixed", "estimated"] = (
                "mixed" if measured > 0 else "estimated"
            )
        elif measured > 0 or interrupted == 0:
            mode = "measured"
        else:
            mode = "estimated"
        if model_usage is None:
            return Reconciliation(
                mode=mode,
                estimated_tokens=estimated_total,
                actual_tokens=None,
                error_ratio=None,
                unreported_estimated_tokens=covered_unreported,
                unreported_actual_tokens=None,
                unreported_error_ratio=None,
                interrupted_estimated_tokens=interrupted,
            )
        actual = sum(usage_total(usage) for usage in model_usage.values())
        unreported_actual = actual - int(measured)
        return Reconciliation(
            mode=mode,
            estimated_tokens=estimated_total,
            actual_tokens=actual,
            error_ratio=(covered_estimated - actual) / actual if actual > 0 else None,
            unreported_estimated_tokens=covered_unreported,
            unreported_actual_tokens=unreported_actual,
            unreported_error_ratio=(
                (covered_unreported - unreported_actual) / unreported_actual
                if unreported_actual > 0
                else None
            ),
            interrupted_estimated_tokens=interrupted,
        )

    # -- readings ---------------------------------------------------------

    @property
    def budget_tokens(self) -> float:
        """Measured plus estimates, including the in-flight response."""

        return self.measured_tokens + self.estimated_tokens

    @property
    def measured_tokens(self) -> float:
        """Tokens of responses with reported usage."""

        return sum(
            response.counted()
            for thread in self._threads.values()
            for response in thread.responses
            if response.usage is not None
        )

    @property
    def estimated_tokens(self) -> float:
        """Tokens of responses without reported usage (in-flight included)."""

        return sum(
            response.counted()
            for thread in self._threads.values()
            for response in thread.responses
            if response.usage is None
        )

    @property
    def interrupted_estimated_tokens(self) -> float:
        """Tokens of in-flight responses cut off by an interrupt.

        A subset of :attr:`estimated_tokens`: responses flagged via
        :meth:`mark_interrupted` that still have no reported usage. Once
        usage arrives for such a response it counts as measured again.
        """

        return sum(
            response.counted()
            for thread in self._threads.values()
            for response in thread.responses
            if response.usage is None and response.interrupted
        )

    def mark_interrupted(self) -> int:
        """Flag in-flight responses without usage as cut off by an interrupt.

        Only unsettled responses qualify: settled-but-unmeasured responses
        completed normally (their usage report is merely missing) and stay
        in the ordinary unreported bucket. Returns the number flagged.
        """

        flagged = 0
        for thread in self._threads.values():
            for response in thread.responses:
                if response.usage is None and not response.settled and not response.interrupted:
                    response.interrupted = True
                    flagged += 1
        return flagged

    @property
    def measured_by_category(self) -> dict[str, int]:
        """Measured totals per category over all threads (estimates excluded)."""

        totals = {category: 0 for category in USAGE_CATEGORIES}
        for thread in self._threads.values():
            for response in thread.responses:
                if response.usage is not None:
                    for category in USAGE_CATEGORIES:
                        totals[category] += response.usage[category]
        return totals

    def main_measured_by_category(self, limit: int | None = None) -> dict[str, int]:
        """Measured main-thread totals per category over the first ``limit`` responses."""

        totals = {category: 0 for category in USAGE_CATEGORIES}
        for response in self._measured_main_responses(limit):
            assert response.usage is not None
            for category in USAGE_CATEGORIES:
                totals[category] += response.usage[category]
        return totals

    def main_hit_rate(self, limit: int | None = None) -> float | None:
        """Share of main-thread input served from the prompt cache.

        ``cache_read / (input + cache_read + cache_creation)`` summed over
        the first ``limit`` measured main-thread responses (all of them when
        ``limit`` is None). ``None`` when the denominator is 0.
        """

        return hit_rate_of(self.main_measured_by_category(limit))

    def main_cache_stats(self, limit: int | None = None) -> MainCacheStats:
        """Hit rate and miss accounting over the first ``limit`` measured main responses."""

        hit_rate = self.main_hit_rate(limit)
        miss_tokens = 0
        counted = 0
        excluded = 0
        thread = self._threads.get("main")
        if thread is not None:
            measured_ids = {
                id(response) for response in self._measured_main_responses(limit)
            }
            for index, response in enumerate(thread.responses):
                if id(response) not in measured_ids:
                    continue
                miss = self._main_miss_at(thread.responses, index)
                if miss is None:
                    excluded += 1
                else:
                    counted += 1
                    miss_tokens += miss
        return MainCacheStats(
            hit_rate=hit_rate, miss_tokens=miss_tokens, counted=counted, excluded=excluded
        )

    def response_cache_metrics(
        self, thread_id: str, response_id: str
    ) -> ResponseCacheMetrics | None:
        """One response's own cache figures, or ``None`` when it is unmeasured.

        ``miss_tokens`` is ``None`` (excluded) for non-main threads and for
        main-thread responses whose predecessor is missing, estimated, or
        separated by a compaction. Exclusion never affects the hit rate.
        """

        thread = self._threads.get(thread_id)
        if thread is None:
            return None
        index = next(
            (i for i, response in enumerate(thread.responses) if response.id == response_id),
            None,
        )
        if index is None:
            return None
        response = thread.responses[index]
        if response.usage is None:
            return None
        miss = self._main_miss_at(thread.responses, index) if thread_id == "main" else None
        return ResponseCacheMetrics(
            input_tokens=response.usage["input_tokens"],
            cache_read_tokens=response.usage["cache_read_input_tokens"],
            cache_creation_tokens=response.usage["cache_creation_input_tokens"],
            hit_rate=hit_rate_of(response.usage),
            miss_tokens=miss,
        )

    @property
    def main_measured_settled_count(self) -> int:
        """Settled measured responses on the main thread (warning trigger)."""

        thread = self._threads.get("main")
        if thread is None:
            return 0
        return sum(
            1 for response in thread.responses if response.settled and response.usage is not None
        )

    def context_tokens(self, thread_id: str) -> float:
        """Current context of one thread (measured where reported)."""

        return self._threads[thread_id].context

    def thread_contexts(self) -> dict[str, float]:
        """Current context per thread."""

        return {thread_id: thread.context for thread_id, thread in self._threads.items()}

    @property
    def peak_main_context_tokens(self) -> float:
        """Highest main-thread context held during the attempt."""

        return self._peak_main_context_tokens

    @property
    def first_settled_without_usage(self) -> DegradedNotice | None:
        """First response that settled without reported usage, if any."""

        return self._first_settled_without_usage

    def has_thread(self, thread_id: str) -> bool:
        """Whether a thread was started (the executor starts them lazily)."""

        return thread_id in self._threads

    def response_count(self, thread_id: str) -> int:
        """Number of responses counted on a thread, in-flight included."""

        return len(self._threads[thread_id].responses)

    def settled_response_ids(self, thread_id: str) -> tuple[str, ...]:
        """Ids of a thread's settled responses, in counting order."""

        return tuple(
            response.id for response in self._threads[thread_id].responses if response.settled
        )

    def settled_reading(self, thread_id: str, response_id: str) -> SettledReading | None:
        """Aggregates captured when one response settled, if it has settled."""

        return self._settled_readings.get((thread_id, response_id))

    # -- internals --------------------------------------------------------

    def _settle(self, thread: _Thread) -> None:
        """Settle a thread's in-flight response as the next event arrives.

        The reading is captured before the settling event is applied: at
        this point the ledger still holds the settled response's own
        state, without the next response's estimate or tool-input growth.
        """

        if not thread.responses:
            return
        response = thread.responses[-1]
        if response.settled:
            return
        self._settled_readings[(thread.id, response.id)] = self._current_reading()
        response.settled = True
        if response.usage is None and self._first_settled_without_usage is None:
            self._first_settled_without_usage = DegradedNotice(
                thread=thread.id,
                subagent_type=thread.subagent_type,
                response_id=response.id,
            )

    def _current_reading(self) -> SettledReading:
        """Snapshot the current aggregates (the settled response's own state)."""

        main_thread = self._threads.get("main")
        return SettledReading(
            budget_tokens=self.budget_tokens,
            measured_tokens=self.measured_tokens,
            estimated_tokens=self.estimated_tokens,
            main_context_tokens=main_thread.context if main_thread is not None else 0.0,
            main_response_count=len(main_thread.responses) if main_thread is not None else 0,
        )

    def _response_for_usage(self, thread: _Thread, response_id: str) -> _Response | None:
        for response in thread.responses:
            if response.id == response_id:
                return response
        if thread.responses:
            return thread.responses[-1]
        return None

    def _measured_main_responses(self, limit: int | None) -> list[_Response]:
        """Measured main-thread responses in counting order (first ``limit``)."""

        thread = self._threads.get("main")
        if thread is None:
            return []
        measured = [response for response in thread.responses if response.usage is not None]
        return measured if limit is None else measured[:limit]

    @staticmethod
    def _main_miss_at(responses: list[_Response], index: int) -> int | None:
        """Cache-miss tokens of ``responses[index]``; ``None`` means excluded.

        The expected prefix is the previous main-thread response's total
        input (its output excluded: Meta's thinking is opaque and may not be
        cached). The first response, a response after a compaction, and a
        response after an estimated predecessor are excluded.
        """

        response = responses[index]
        if response.usage is None or index == 0 or response.compacted_before:
            return None
        previous = responses[index - 1]
        if previous.usage is None:
            return None
        return max(0, _total_input_of(previous.usage) - response.usage["cache_read_input_tokens"])

    def _track_peak(self, thread: _Thread) -> None:
        if thread.kind == "main" and thread.context > self._peak_main_context_tokens:
            self._peak_main_context_tokens = thread.context
