"""Claude SDK boundary for one bounded implementation attempt."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
import json
from pathlib import Path
import shlex
from typing import Any, Protocol

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    TextBlock,
)

from simple_coding_agent.config import RuntimeConfig


_SKILLS = ["implement", "tdd", "code-review", "codebase-design", "handoff"]
_META_BASE_URL = "https://api.meta.ai"

# SDK 0.2.156 reports authoritative `total_cost_usd` only on the terminal
# ResultMessage; there is no live cost feed mid-stream. This blended rate
# converts tokens observed on each AssistantMessage into an estimate that is
# precise enough to trigger a best-effort cooperative handoff near the
# budget ceiling, but it is never an exact accounting guarantee.
_ESTIMATED_USD_PER_MILLION_TOKENS = 15.0
_USAGE_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)
_CATEGORY_USD_PER_MILLION_TOKENS: Mapping[str, float] = {
    "cache_read_input_tokens": 0.30,
    "cache_creation_input_tokens": 3.75,
    "input_tokens": 3.00,
    "output_tokens": 15.00,
}

_HANDOFF_FOLLOWUP_TIMEOUT = 300

_OPERATOR_HANDOFF_DELIVERY_WINDOW_SECONDS = 60
_OPERATOR_HANDOFF_MODEL_DEADLINE_SECONDS = 240

_COST_HANDOFF_INSTRUCTION = (
    "This attempt is approaching its cost budget. Invoke the `handoff` skill now "
    "to preserve your progress cooperatively instead of continuing further work."
)
_OPERATOR_HANDOFF_INSTRUCTION = (
    "The operator requested a handoff (`agentctl handoff now`). Invoke the"
    " `handoff` skill now to preserve your progress cooperatively instead of"
    " continuing further work: commit any outstanding work in ordinary work"
    " commits, then write `.agent/handoff/<issue-number>.md` with"
    " `reason: operator_request` and commit it separately as the final commit."
    " Do not start another implementation step."
)
_LIMIT_HANDOFF_FOLLOWUP_PROMPT = (
    "Execution reached its turn or time limit. Invoke the `handoff` skill now to "
    "preserve your progress: commit any outstanding work, then write and commit "
    "the handoff note. Do not attempt further implementation work."
)
_OPERATOR_HANDOFF_FOLLOWUP_PROMPT = (
    "The operator requested a handoff (`agentctl handoff now`) and the safe"
    " boundary was missed. Invoke the `handoff` skill now to preserve your"
    " progress: commit any outstanding work in ordinary work commits, then write"
    " `.agent/handoff/<issue-number>.md` with `reason: operator_request` and"
    " commit it separately as the final commit. Do not attempt further"
    " implementation work."
)


class _CostEstimator:
    """Accumulate an approximate USD spend from streamed token usage or turn count."""

    def __init__(
        self,
        rate_per_million_tokens: float = _ESTIMATED_USD_PER_MILLION_TOKENS,
        category_rates: Mapping[str, float] | None = None,
        fallback_turn_cost: float = 0.05,
    ) -> None:
        self._rate = rate_per_million_tokens
        self._category_rates = dict(category_rates or _CATEGORY_USD_PER_MILLION_TOKENS)
        self._fallback_turn_cost = fallback_turn_cost
        self._tokens = 0
        self._estimated_cost = 0.0
        self._has_positive_usage = False
        self._turns = 0

    @property
    def has_positive_usage(self) -> bool:
        return self._has_positive_usage

    @property
    def turns(self) -> int:
        return self._turns

    def observe(self, usage: Any) -> float:
        has_tokens = False
        if isinstance(usage, Mapping):
            has_cache = any(
                isinstance(usage.get(f), int) and usage.get(f) > 0
                for f in ("cache_read_input_tokens", "cache_creation_input_tokens")
            )
            for field in _USAGE_TOKEN_FIELDS:
                value = usage.get(field)
                if isinstance(value, int) and value > 0:
                    has_tokens = True
                    self._tokens += value
                    rate = self._category_rates.get(field, self._rate) if has_cache else self._rate
                    self._estimated_cost += (value / 1_000_000) * rate
        if has_tokens:
            self._has_positive_usage = True
            self._turns += 1
        return self.estimated_cost_usd

    def observe_turn(self, estimated_turn_cost_usd: float) -> float:
        self._turns += 1
        self._estimated_cost += estimated_turn_cost_usd
        return self.estimated_cost_usd

    @property
    def estimated_cost_usd(self) -> float:
        return self._estimated_cost


class ModelExecutionStatus(StrEnum):
    """Status of the SDK stream; the evaluator maps it to an attempt outcome."""

    SUCCEEDED = "succeeded"
    MODEL_LIMIT_REACHED = "model_limit_reached"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    HANDOFF_REQUESTED = "handoff_requested"
    OPERATOR_HANDOFF_EXPIRED = "operator_handoff_expired"


@dataclass(frozen=True)
class OperatorHandoff:
    """An operator ``handoff now`` request the model must observe.

    ``accepted_at`` is the wall-clock acceptance time of the durable command
    commit; delivery and model-deadline windows are measured from it.
    """

    request_id: str
    accepted_at: datetime


OperatorHandoffProvider = Callable[[], "OperatorHandoff | None"]
OperatorHandoffReporter = Callable[[str, str], None]


@dataclass(frozen=True)
class SkillEvent:
    """An auditable invocation of an installed skill."""

    phase: str
    name: str
    agent_id: str | None
    timestamp: str


@dataclass(frozen=True)
class ModelExecution:
    """Structured execution evidence; publication remains outside this boundary."""

    status: ModelExecutionStatus
    explanation: str
    stop_reason: str | None
    model_usage: Mapping[str, Any] | None
    observed_models: tuple[str, ...]
    skill_events: tuple[SkillEvent, ...]
    terminal_reason: str | None = None


class SDKClient(Protocol):
    """The subset of streaming client behaviour used by this boundary."""

    async def __aenter__(self) -> SDKClient: ...

    async def __aexit__(self, *arguments: object) -> None: ...

    async def connect(self, prompt: str) -> None: ...

    async def interrupt(self) -> None: ...

    async def query(self, prompt: str) -> None: ...

    def receive_response(self) -> Any: ...


class EvidenceWriter(Protocol):
    """Durable per-attempt storage; the screen log gets a reference, not the payload."""

    def write_text(self, name: str, content: str) -> Path: ...


ClientFactory = Callable[[ClaudeAgentOptions], SDKClient]
Clock = Callable[[], datetime]
EventLog = Callable[..., None]

_MAX_LOG_DETAIL = 2000


class ModelExecutor:
    """Run the approved upstream skills and retain terminal stream evidence."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        client_factory: ClientFactory = ClaudeSDKClient,
        clock: Clock = lambda: datetime.now(UTC),
        event_log: EventLog = lambda event, detail="", issue_number=None: None,
    ) -> None:
        self._config = config
        self._client_factory = client_factory
        self._clock = clock
        self._event_log = event_log
        self._skill_events: list[SkillEvent] = []
        self._review_count = 0
        self._issue_number: int | None = None
        self._archive: EvidenceWriter | None = None
        self._event_sequence = 0
        self._cost_estimator = _CostEstimator(
            fallback_turn_cost=self._config.max_budget_usd / max(self._config.max_turns, 1)
        )
        self._started_at = self._clock()
        self._soft_threshold_crossed = False
        self._handoff_context_delivered = False
        self._usage_shape_logged = False
        self._cost_estimator_observed_on_assistant = False
        self._operator_handoff_provider: OperatorHandoffProvider | None = None
        self._operator_handoff_reporter: OperatorHandoffReporter | None = None
        self._operator_handoff: OperatorHandoff | None = None
        self._operator_context_delivered = False
        self._operator_fallback_used = False
        self._operator_begun = False
        self._operator_deadline_expired = False

    def set_operator_handoff_provider(
        self, provider: OperatorHandoffProvider | None
    ) -> None:
        """Observe ``handoff now`` requests accepted while the model runs.

        The provider is polled at every safe SDK boundary (after an
        in-flight tool finishes) and while waiting for stream messages, so a
        request accepted during setup or mid-execution is picked up without
        interrupting an in-flight tool early. ``None`` clears the provider.
        """

        self._operator_handoff_provider = provider

    def set_operator_handoff_reporter(
        self, reporter: OperatorHandoffReporter | None
    ) -> None:
        """Report ``(event, request_id)`` for ``delivered`` and ``begun``.

        ``delivered`` fires once the handoff instruction reaches the model;
        ``begun`` fires once invocation of the project-owned ``handoff``
        skill is observed (delivery alone never counts as begun). ``None``
        clears the reporter.
        """

        self._operator_handoff_reporter = reporter

    @property
    def operator_request_id(self) -> str | None:
        """The operator ``handoff now`` request latched for this execution, if any."""

        latched = self._operator_handoff
        return latched.request_id if latched is not None else None

    def _poll_operator_handoff(self) -> None:
        """Latch the newest operator handoff request that may still be served.

        Before the instruction is delivered the newest pending request wins
        (an earlier one may already have been superseded); once delivered or
        begun the latched request is pinned so a replacement cannot undo the
        handoff mid-flight. Provider failures never disturb the stream.
        """

        provider = self._operator_handoff_provider
        if provider is None:
            return
        try:
            current = provider()
        except Exception:
            return
        if current is None:
            return
        latched = self._operator_handoff
        if latched is not None and (
            latched.request_id == current.request_id
            or self._operator_context_delivered
            or self._operator_begun
        ):
            return
        self._operator_handoff = current

    def _report_operator(self, event: str) -> None:
        """Report ``delivered``/``begun`` without ever breaking execution."""

        reporter = self._operator_handoff_reporter
        latched = self._operator_handoff
        if reporter is None or latched is None:
            return
        try:
            reporter(event, latched.request_id)
        except Exception:
            self._log(
                "operator_handoff_report_failed",
                f"event={event}; request={latched.request_id}",
            )

    def _operator_due_action(self) -> str | None:
        """Return ``fallback``/``expire`` when an operator boundary is due now."""

        self._poll_operator_handoff()
        latched = self._operator_handoff
        if latched is None or self._operator_deadline_expired:
            return None
        elapsed = (self._clock() - latched.accepted_at).total_seconds()
        if elapsed >= _OPERATOR_HANDOFF_MODEL_DEADLINE_SECONDS:
            return "expire"
        if (
            not self._operator_fallback_used
            and not self._operator_context_delivered
            and not self._operator_begun
            and elapsed >= _OPERATOR_HANDOFF_DELIVERY_WINDOW_SECONDS
        ):
            return "fallback"
        return None

    def _operator_stream_wait(self) -> float | None:
        """How long the stream may wait before the next operator boundary.

        ``None`` waits indefinitely (no provider is watching). Otherwise the
        wait ends at the next due action — the single fallback or the model
        deadline — or at the delivery-window cadence that re-polls the
        provider, so a request accepted mid-stream is picked up within 60
        seconds even on an idle stream.
        """

        self._poll_operator_handoff()
        latched = self._operator_handoff
        if latched is None:
            if self._operator_handoff_provider is None:
                return None
            return float(_OPERATOR_HANDOFF_DELIVERY_WINDOW_SECONDS)
        elapsed = (self._clock() - latched.accepted_at).total_seconds()
        targets = [_OPERATOR_HANDOFF_MODEL_DEADLINE_SECONDS - elapsed]
        if (
            not self._operator_fallback_used
            and not self._operator_context_delivered
            and not self._operator_begun
        ):
            targets.append(_OPERATOR_HANDOFF_DELIVERY_WINDOW_SECONDS - elapsed)
        return max(min(targets), 0.0)

    def _operator_fallback_blocked(self) -> bool:
        """Whether the single fallback query would bypass the hard cost limit."""

        return (
            self._cost_estimator.estimated_cost_usd >= self._config.max_budget_usd
        )

    async def _run_operator_fallback(
        self, client: SDKClient, observed_models: list[str]
    ) -> tuple[ResultMessage | None, tuple[str, ...]] | None:
        """Interrupt, drain, and issue the single follow-up handoff query.

        Returns the follow-up terminal result (and merged models) to
        propagate, or ``None`` to keep reading the original stream. A blocked
        fallback (hard cost limit) consumes the single attempt without
        querying: the stream's own terminal result then decides the outcome.
        """

        latched = self._operator_handoff
        self._operator_fallback_used = True
        if latched is None:
            return None
        if self._operator_fallback_blocked():
            self._log(
                "operator_handoff_fallback_blocked",
                f"request={latched.request_id}; estimated cost reached the hard"
                " cost limit, so no follow-up query is issued and the stream's"
                " own terminal result decides the outcome",
            )
            return None
        self._operator_context_delivered = True
        self._report_operator("delivered")
        self._log(
            "operator_handoff_fallback",
            f"request={latched.request_id}; the safe boundary was missed, so the"
            " single follow-up handoff prompt is issued",
        )
        await client.interrupt()
        await self._drain(client)
        try:
            await client.query(_OPERATOR_HANDOFF_FOLLOWUP_PROMPT)
            followup_terminal, followup_models = await self._receive_terminal(client)
        except Exception:
            return None
        return followup_terminal, tuple([*observed_models, *followup_models])

    @property
    def skill_events(self) -> tuple[SkillEvent, ...]:
        """Return the skill provenance recorded during this execution."""

        return tuple(self._skill_events)

    def _log(self, event: str, detail: str = "") -> None:
        self._event_log(event, detail, issue_number=self._issue_number)

    def _archive_evidence(self, kind: str, content: str, *, extension: str) -> str | None:
        """Write full evidence to the attempt archive; return a reference or None.

        The reference is the bare filename: every archived file, including
        agent_output.json, lives in the same attempt directory, so a
        relative-to-itself path is just the name.
        """

        if self._archive is None:
            return None
        self._event_sequence += 1
        filename = f"{self._event_sequence:04d}_{kind}.{extension}"
        self._archive.write_text(filename, content)
        return filename

    async def execute(
        self,
        *,
        issue_body: str,
        working_directory: Path,
        issue_number: int | None = None,
        archive: EvidenceWriter | None = None,
    ) -> ModelExecution:
        """Dispatch ``/implement`` and classify its one terminal SDK result."""

        self._skill_events.clear()
        self._review_count = 0
        self._issue_number = issue_number
        self._archive = archive
        self._event_sequence = 0
        self._started_at = self._clock()
        self._cost_estimator = _CostEstimator(
            fallback_turn_cost=self._config.max_budget_usd / max(self._config.max_turns, 1)
        )
        self._soft_threshold_crossed = False
        self._handoff_context_delivered = False
        self._usage_shape_logged = False
        self._cost_estimator_observed_on_assistant = False
        self._operator_handoff = None
        self._operator_context_delivered = False
        self._operator_fallback_used = False
        self._operator_begun = False
        self._operator_deadline_expired = False
        soft_threshold = self._config.max_budget_usd * (1 - self._config.soft_threshold_percentage)
        self._log(
            "model_execution_started",
            f"issue_body_length={len(issue_body)}; limits: "
            f"max_budget_usd={self._config.max_budget_usd:.4f}; "
            f"soft_threshold_usd={soft_threshold:.4f}; "
            f"max_turns={self._config.max_turns}; "
            f"timeout_seconds={self._config.model_timeout}",
        )
        client = self._client_factory(self._options(working_directory))
        try:
            async with client:
                await client.connect(f"/implement\n\n{issue_body}")
                try:
                    async with asyncio.timeout(self._config.model_timeout):
                        terminal, observed_models = await self._receive_terminal(client)
                except TimeoutError:
                    await client.interrupt()
                    await self._drain(client)
                    followup = await self._attempt_handoff_followup(client, ())
                    if followup is not None:
                        return followup
                    return self._evidence(
                        ModelExecutionStatus.INFRASTRUCTURE_ERROR,
                        "Model execution exceeded MODEL_TIMEOUT and was interrupted.",
                        None,
                        None,
                        (),
                    )
                if terminal is not None and self._is_model_limit(terminal, observed_models):
                    followup = await self._attempt_handoff_followup(client, observed_models)
                    if followup is not None:
                        return followup
        except Exception as error:
            return self._evidence(
                ModelExecutionStatus.INFRASTRUCTURE_ERROR,
                f"Claude SDK execution failed: {type(error).__name__}.",
                None,
                None,
                (),
            )

        if terminal is None:
            if self._operator_deadline_expired and self._operator_handoff is not None:
                return self._evidence(
                    ModelExecutionStatus.OPERATOR_HANDOFF_EXPIRED,
                    f"Operator handoff {self._operator_handoff.request_id} reached"
                    f" its {_OPERATOR_HANDOFF_MODEL_DEADLINE_SECONDS}-second model"
                    " deadline without the model invoking the handoff skill.",
                    None,
                    None,
                    observed_models,
                )
            return self._evidence(
                ModelExecutionStatus.INFRASTRUCTURE_ERROR,
                "Claude SDK stream ended without a terminal ResultMessage.",
                None,
                None,
                observed_models,
            )
        return self._classify(terminal, observed_models)

    def _is_model_limit(
        self, terminal: ResultMessage, observed_models: tuple[str, ...]
    ) -> bool:
        """Mirror ``_classify``'s MODEL_LIMIT_REACHED predicate ahead of time.

        Every path that would otherwise classify as MODEL_LIMIT_REACHED
        (turns exceeded, hard budget ceiling, or a bare SDK error result)
        deserves one cooperative handoff attempt before that classification
        is finalised; a model/config mismatch or a plain
        timeout/aborted_* stop reason is an infrastructure problem, not a
        limit the model can hand off from, so those are excluded.
        """

        model_usage = getattr(terminal, "model_usage", None)
        result_models = tuple(model_usage.keys()) if isinstance(model_usage, Mapping) else ()
        all_models = tuple(dict.fromkeys((*observed_models, *result_models)))
        if any(model != self._config.model for model in all_models):
            return False
        stop_reason = getattr(terminal, "stop_reason", None)
        terminal_reason = getattr(terminal, "terminal_reason", None)
        subtype = getattr(terminal, "subtype", None)
        total_cost_usd = getattr(terminal, "total_cost_usd", None)
        if (
            stop_reason == "timeout"
            or (isinstance(stop_reason, str) and stop_reason.startswith("aborted_"))
            or terminal_reason in ("aborted_streaming", "aborted_tools")
        ):
            return False
        if (
            terminal_reason in ("max_turns", "budget_exhausted")
            or subtype in ("error_max_turns", "error_max_budget_usd")
            or stop_reason in ("max_turns_exceeded", "max_budget_usd_exceeded")
            or (
                getattr(terminal, "is_error", False)
                and total_cost_usd is not None
                and total_cost_usd >= self._config.max_budget_usd
            )
        ):
            return True
        return bool(getattr(terminal, "is_error", False))

    async def _attempt_handoff_followup(
        self, client: SDKClient, observed_models: tuple[str, ...]
    ) -> ModelExecution | None:
        """Best-effort same-client follow-up after a turns/timeout limit.

        Returns a HANDOFF_REQUESTED evidence only when the model actually
        invoked the handoff skill during the follow-up; otherwise returns
        None so the caller falls back to its ordinary limit classification.
        This never retries: a follow-up that fails or times out is itself
        evidence that handoff is not achievable right now.
        """

        try:
            await client.query(_LIMIT_HANDOFF_FOLLOWUP_PROMPT)
            async with asyncio.timeout(_HANDOFF_FOLLOWUP_TIMEOUT):
                followup_terminal, followup_models = await self._receive_terminal(client)
        except Exception:
            return None
        if followup_terminal is None:
            return None
        all_models = tuple(dict.fromkeys((*observed_models, *followup_models)))
        if not any(event.name == "handoff" for event in self._skill_events):
            return None
        return self._evidence(
            ModelExecutionStatus.HANDOFF_REQUESTED,
            "Model invoked the handoff skill after reaching a turns/timeout limit.",
            followup_terminal.stop_reason,
            getattr(followup_terminal, "model_usage", None),
            all_models,
        )

    def _options(self, working_directory: Path) -> ClaudeAgentOptions:
        settings_sources = ["user"]
        if self._config.agent_trust_project_settings:
            settings_sources.append("project")
        return ClaudeAgentOptions(
            model=self._config.model,
            fallback_model=None,
            permission_mode="bypassPermissions",
            max_turns=self._config.max_turns,
            max_budget_usd=self._config.max_budget_usd,
            cwd=working_directory,
            env=_meta_environment(self._config.meta_api_key, self._config.model),
            setting_sources=settings_sources,
            strict_mcp_config=self._config.agent_trust_project_settings,
            extra_args={"disable-all-hooks": None}
            if self._config.agent_trust_project_settings
            else {},
            skills=_SKILLS,
            hooks={
                "PreToolUse": [
                    HookMatcher(hooks=[self._guard_and_record_pre_tool_use])
                ],
                "PostToolUse": [HookMatcher(hooks=[self._record_post_tool_use])],
            },
        )

    async def _receive_terminal(
        self, client: SDKClient
    ) -> tuple[ResultMessage | None, tuple[str, ...]]:
        observed_models: list[str] = []
        stream = client.receive_response().__aiter__()
        while True:
            action = self._operator_due_action()
            if action == "fallback":
                outcome = await self._run_operator_fallback(client, observed_models)
                if outcome is not None:
                    return outcome
                continue
            if action == "expire":
                await client.interrupt()
                await self._drain(client)
                self._operator_deadline_expired = True
                self._log(
                    "operator_handoff_model_deadline_expired",
                    f"request={self._operator_handoff.request_id if self._operator_handoff else None}; "
                    f"deadline_seconds={_OPERATOR_HANDOFF_MODEL_DEADLINE_SECONDS}",
                )
                return None, tuple(observed_models)
            wait = self._operator_stream_wait()
            try:
                if wait is None:
                    message = await stream.__anext__()
                else:
                    async with asyncio.timeout(wait):
                        message = await stream.__anext__()
            except StopAsyncIteration:
                return None, tuple(observed_models)
            except TimeoutError:
                # Awaited longer than the next operator boundary without a
                # message; loop back so the due action runs above.
                continue
            if isinstance(message, ResultMessage) or _looks_like_result(message):
                return message, tuple(observed_models)
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text.strip():
                        self._log("model_response", self._model_response_detail(block.text))
                self._observe_cost(message.usage)
            model = getattr(message, "model", None)
            if isinstance(model, str):
                observed_models.append(model)

    def _observe_cost(self, usage: Any) -> None:
        """Latch a one-time soft-threshold crossing from estimated cumulative cost."""

        self._cost_estimator_observed_on_assistant = True
        if not self._usage_shape_logged:
            self._usage_shape_logged = True
            self._log("model_usage_shape", f"usage={usage!r}")
        self._cost_estimator.observe(usage)
        if self._cost_estimator.has_positive_usage:
            self._check_limits()

    def _elapsed_seconds(self) -> int:
        delta = (self._clock() - self._started_at).total_seconds()
        return max(0, int(delta))

    def _limits_detail(self) -> str:
        soft_threshold = self._config.max_budget_usd * (1 - self._config.soft_threshold_percentage)
        return (
            f"estimated_cost_usd={self._cost_estimator.estimated_cost_usd:.4f}; "
            f"soft_threshold_usd={soft_threshold:.4f}; "
            f"max_budget_usd={self._config.max_budget_usd:.4f}; "
            f"turns={self._cost_estimator.turns}; "
            f"max_turns={self._config.max_turns}; "
            f"elapsed_seconds={self._elapsed_seconds()}; "
            f"timeout_seconds={self._config.model_timeout}"
        )

    def _check_limits(self) -> None:
        self._log("limits_checked", self._limits_detail())
        soft_threshold = self._config.max_budget_usd * (1 - self._config.soft_threshold_percentage)
        if not self._soft_threshold_crossed and self._cost_estimator.estimated_cost_usd >= soft_threshold:
            self._soft_threshold_crossed = True
            self._log(
                "cost_soft_threshold_crossed",
                f"estimated_cost_usd={self._cost_estimator.estimated_cost_usd:.4f}; "
                f"soft_threshold_usd={soft_threshold:.4f}",
            )

    async def _drain(self, client: SDKClient) -> None:
        """Consume buffered messages after interrupting before client teardown."""

        try:
            async with asyncio.timeout(10):
                async for _ in client.receive_response():
                    pass
        except Exception:
            # Client teardown still runs; drain evidence is secondary to avoiding an orphan.
            return

    def _classify(
        self, terminal: ResultMessage, observed_models: tuple[str, ...]
    ) -> ModelExecution:
        # SDK 0.2.156 exposes these runtime fields, unlike the current reference
        # documentation's terminal_reason/total_cost_usd/input_tokens examples.
        model_usage = getattr(terminal, "model_usage", None)
        result_models = tuple(model_usage.keys()) if isinstance(model_usage, Mapping) else ()
        all_models = tuple(dict.fromkeys((*observed_models, *result_models)))
        mismatches = tuple(model for model in all_models if model != self._config.model)
        stop_reason = getattr(terminal, "stop_reason", None)
        terminal_reason = getattr(terminal, "terminal_reason", None)
        subtype = getattr(terminal, "subtype", None)
        total_cost_usd = getattr(terminal, "total_cost_usd", None)

        if self._operator_context_delivered and any(
            event.name == "handoff" for event in self._skill_events
        ):
            return self._evidence(
                ModelExecutionStatus.HANDOFF_REQUESTED,
                "Model invoked the handoff skill after an operator handoff request.",
                stop_reason,
                model_usage,
                all_models,
                terminal_reason=terminal_reason,
            )
        if self._handoff_context_delivered and any(
            event.name == "handoff" for event in self._skill_events
        ):
            return self._evidence(
                ModelExecutionStatus.HANDOFF_REQUESTED,
                "Model invoked the handoff skill after a cost soft-threshold instruction.",
                stop_reason,
                model_usage,
                all_models,
                terminal_reason=terminal_reason,
            )
        if mismatches:
            return self._evidence(
                ModelExecutionStatus.INFRASTRUCTURE_ERROR,
                f"Observed model mismatch: {', '.join(mismatches)}.",
                stop_reason,
                model_usage,
                all_models,
                terminal_reason=terminal_reason,
            )
        if (
            terminal_reason == "max_turns"
            or subtype == "error_max_turns"
            or stop_reason == "max_turns_exceeded"
        ):
            return self._evidence(
                ModelExecutionStatus.MODEL_LIMIT_REACHED,
                "Model execution reached max_turns.",
                stop_reason,
                model_usage,
                all_models,
                terminal_reason=terminal_reason,
            )
        if (
            stop_reason == "timeout"
            or (isinstance(stop_reason, str) and stop_reason.startswith("aborted_"))
            or terminal_reason in ("aborted_streaming", "aborted_tools")
        ):
            reason_name = terminal_reason or stop_reason
            return self._evidence(
                ModelExecutionStatus.INFRASTRUCTURE_ERROR,
                f"Model execution stopped with {reason_name}.",
                stop_reason,
                model_usage,
                all_models,
                terminal_reason=terminal_reason,
            )
        if (
            terminal_reason == "budget_exhausted"
            or subtype == "error_max_budget_usd"
            or stop_reason == "max_budget_usd_exceeded"
            or (
                getattr(terminal, "is_error", False)
                and total_cost_usd is not None
                and total_cost_usd >= self._config.max_budget_usd
            )
        ):
            return self._evidence(
                ModelExecutionStatus.MODEL_LIMIT_REACHED,
                "Model execution reached the hard cost ceiling "
                f"(total_cost_usd={total_cost_usd}, "
                f"num_turns={getattr(terminal, 'num_turns', None)}); no further implementation "
                "work was permitted.",
                stop_reason,
                model_usage,
                all_models,
                terminal_reason=terminal_reason,
            )
        if getattr(terminal, "is_error", True):
            return self._evidence(
                ModelExecutionStatus.MODEL_LIMIT_REACHED,
                "Claude SDK returned an error result "
                f"(stop_reason={stop_reason!r}, "
                f"terminal_reason={terminal_reason!r}, "
                f"subtype={subtype!r}, "
                f"api_error_status={getattr(terminal, 'api_error_status', None)!r}, "
                f"errors={getattr(terminal, 'errors', None)!r}).",
                stop_reason,
                model_usage,
                all_models,
                terminal_reason=terminal_reason,
            )
        return self._evidence(
            ModelExecutionStatus.SUCCEEDED,
            "Claude SDK completed the implementation workflow.",
            stop_reason,
            model_usage,
            all_models,
            terminal_reason=terminal_reason,
        )

    def _evidence(
        self,
        status: ModelExecutionStatus,
        explanation: str,
        stop_reason: str | None,
        model_usage: Mapping[str, Any] | None,
        observed_models: tuple[str, ...],
        terminal_reason: str | None = None,
    ) -> ModelExecution:
        self._log("model_execution_finished", f"status={status}; {explanation}")
        return ModelExecution(
            status=status,
            explanation=explanation,
            stop_reason=stop_reason,
            model_usage=model_usage,
            observed_models=observed_models,
            skill_events=self.skill_events,
            terminal_reason=terminal_reason,
        )

    async def _guard_and_record_pre_tool_use(
        self, hook_input: Any, tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        self._log_tool_use("tool_call", hook_input)
        await self._record_skill_event("PreToolUse", hook_input, tool_use_id, context)
        if _skill_name(hook_input) == "code-review":
            self._review_count += 1
            if self._review_count > 3:
                return _deny("At most two code-review repair cycles are allowed.")
        if self._soft_threshold_crossed and not any(
            event.name == "handoff" for event in self._skill_events
        ):
            tool_name = _hook_field(hook_input, "tool_name")
            if tool_name == "Skill" and _skill_name(hook_input) == "handoff":
                pass
            elif tool_name in ("Edit", "Write"):
                file_path = str(_hook_field(hook_input, "tool_input", {}).get("file_path", ""))
                if ".agent/handoff" not in file_path:
                    return _deny(
                        "Cost soft threshold reached. Further implementation work is disabled. "
                        "You must invoke the `handoff` skill now to preserve your progress."
                    )
            elif tool_name == "Bash":
                command = str(_hook_field(hook_input, "tool_input", {}).get("command", ""))
                if not _is_handoff_command(command):
                    return _deny(
                        "Cost soft threshold reached. Further implementation work is disabled. "
                        "You must invoke the `handoff` skill now to preserve your progress."
                    )
        if self._operator_context_delivered:
            tool_name = _hook_field(hook_input, "tool_name")
            if tool_name == "Skill":
                if _skill_name(hook_input) != "handoff":
                    return _deny(
                        "Operator handoff in progress. Further implementation work is"
                        " disabled. Invoke the `handoff` skill now to preserve your"
                        " progress."
                    )
            elif tool_name in ("Edit", "Write"):
                file_path = str(_hook_field(hook_input, "tool_input", {}).get("file_path", ""))
                if ".agent/handoff" not in file_path:
                    return _deny(
                        "Operator handoff in progress. Further implementation work is"
                        " disabled. Invoke the `handoff` skill now to preserve your"
                        " progress."
                    )
            elif tool_name == "Bash":
                command = str(_hook_field(hook_input, "tool_input", {}).get("command", ""))
                if not _is_handoff_command(command):
                    return _deny(
                        "Operator handoff in progress. Further implementation work is"
                        " disabled. Invoke the `handoff` skill now to preserve your"
                        " progress."
                    )
        return await publication_guard(hook_input, tool_use_id, context)

    async def _record_post_tool_use(
        self, hook_input: Any, tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        self._log_tool_use(
            "tool_result", hook_input, response=_hook_field(hook_input, "tool_response")
        )
        await self._record_skill_event("PostToolUse", hook_input, tool_use_id, context)
        if not self._cost_estimator.has_positive_usage:
            per_turn_cost = self._config.max_budget_usd / max(self._config.max_turns, 1)
            self._cost_estimator.observe_turn(per_turn_cost)
        self._check_limits()
        self._poll_operator_handoff()
        contexts: list[str] = []
        if self._soft_threshold_crossed and not self._handoff_context_delivered:
            self._handoff_context_delivered = True
            self._log("cost_soft_threshold_handoff_context_injected")
            contexts.append(_COST_HANDOFF_INSTRUCTION)
        if self._operator_handoff is not None and not self._operator_context_delivered:
            self._operator_context_delivered = True
            self._report_operator("delivered")
            self._log(
                "operator_handoff_context_delivered",
                f"request={self._operator_handoff.request_id}",
            )
            contexts.append(_OPERATOR_HANDOFF_INSTRUCTION)
        if contexts:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": "\n\n".join(contexts),
                }
            }
        return {}

    def _log_tool_use(self, event: str, hook_input: Any, *, response: Any = None) -> None:
        tool_name = _hook_field(hook_input, "tool_name")
        if not isinstance(tool_name, str):
            return
        skill_name = _skill_name(hook_input) if tool_name == "Skill" else None
        label = f"skill:{skill_name}" if skill_name else tool_name
        if event == "tool_result":
            event_name = "skill_result" if skill_name else event
            # Skill calls are already small; only offload plain tool payloads to disk.
            detail = (
                f"{label}: {response!r}"
                if skill_name
                else self._tool_result_detail(label, response)
            )
        else:
            tool_input = _hook_field(hook_input, "tool_input", {})
            event_name = "skill_call" if skill_name else event
            detail = (
                f"{label}: {tool_input!r}"
                if skill_name
                else self._tool_call_detail(label, tool_input)
            )
        self._log(event_name, _truncate(detail))

    def _tool_call_detail(self, label: str, tool_input: Any) -> str:
        path = self._archive_evidence("tool_call", _json_or_repr(tool_input), extension="json")
        if path is None:
            return f"{label}: {tool_input!r}"
        return f"{label} -> {path}"

    def _tool_result_detail(self, label: str, response: Any) -> str:
        path = self._archive_evidence("tool_result", _json_or_repr(response), extension="json")
        if path is None:
            return f"{label}: {response!r}"
        outcome = _outcome_hint(response)
        return f"{label} -> {path}" + (f" ({outcome})" if outcome else "")

    def _model_response_detail(self, text: str) -> str:
        path = self._archive_evidence("model_response", text, extension="txt")
        if path is None:
            return _truncate(text)
        return f"-> {path} (chars={len(text)})"

    async def _record_skill_event(
        self, phase: str, hook_input: Any, tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        if _hook_field(hook_input, "tool_name") != "Skill":
            return {}
        name = _skill_name(hook_input)
        if not isinstance(name, str):
            return {}
        self._skill_events.append(
            SkillEvent(
                phase=phase,
                name=name,
                agent_id=_hook_field(hook_input, "agent_id"),
                timestamp=_timestamp(self._clock()),
            )
        )
        if (
            name == "handoff"
            and self._operator_handoff is not None
            and not self._operator_begun
        ):
            self._operator_begun = True
            self._report_operator("begun")
            self._log(
                "operator_handoff_begun",
                f"request={self._operator_handoff.request_id}",
            )
        return {}


async def publication_guard(hook_input: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
    """Deny model-side publication while allowing the driving process to publish."""

    if _hook_field(hook_input, "tool_name") != "Bash":
        return {}
    tool_input = _hook_field(hook_input, "tool_input", {})
    command = tool_input.get("command") if isinstance(tool_input, Mapping) else None
    if not isinstance(command, str) or not _is_publication_command(command):
        return {}
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "Publication is owned by the driving process.",
        }
    }


def _deny(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _meta_environment(api_key: str, model: str) -> dict[str, str]:
    return {
        "ANTHROPIC_BASE_URL": _META_BASE_URL,
        "ANTHROPIC_AUTH_TOKEN": api_key,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "CLAUDE_STREAM_IDLE_TIMEOUT_MS": "60000",
    }


def _looks_like_result(message: object) -> bool:
    """Permit fake SDK streams without weakening production ResultMessage handling."""

    return all(hasattr(message, attribute) for attribute in ("is_error", "stop_reason", "model_usage"))


def _skill_name(hook_input: object) -> str | None:
    tool_input = _hook_field(hook_input, "tool_input", {})
    name = tool_input.get("skill") if isinstance(tool_input, Mapping) else None
    return name if isinstance(name, str) else None


def _hook_field(hook_input: object, name: str, default: Any = None) -> Any:
    """Read both SDK TypedDict hooks and test doubles without losing provenance."""

    if isinstance(hook_input, Mapping):
        return hook_input.get(name, default)
    return getattr(hook_input, name, default)


def _is_publication_command(command: str) -> bool:
    for segment in _shell_segments(command):
        if _is_prohibited_invocation(segment):
            return True
    return False


def _shell_segments(command: str) -> tuple[list[str], ...]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    tokens = list(lexer)
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token and set(token) <= {";", "&", "|"}:
            if segments[-1]:
                segments.append([])
        else:
            segments[-1].append(token)
    return tuple(segment for segment in segments if segment)


def _is_prohibited_invocation(tokens: list[str]) -> bool:
    index = 0
    while index < len(tokens) and "=" in tokens[index] and not tokens[index].startswith("-"):
        index += 1
    if index < len(tokens) and tokens[index] in {"command", "env"}:
        index += 1
    if index >= len(tokens):
        return False
    executable = Path(tokens[index]).name
    arguments = tokens[index + 1 :]
    if executable == "git":
        command_name = _git_subcommand(arguments)
        return command_name == "push"
    if executable == "gh":
        command_name = _gh_subcommand(arguments)
        return command_name in {("pr", "merge"), ("issue", "close")}
    if executable in {"sh", "bash", "zsh"} and "-c" in arguments:
        script_index = arguments.index("-c") + 1
        return script_index < len(arguments) and _is_publication_command(arguments[script_index])
    return False


def _is_handoff_command(command: str) -> bool:
    for segment in _shell_segments(command):
        if not segment:
            continue
        index = 0
        while index < len(segment) and "=" in segment[index] and not segment[index].startswith("-"):
            index += 1
        if index < len(segment) and segment[index] in {"command", "env"}:
            index += 1
        if index >= len(segment):
            continue
        executable = Path(segment[index]).name
        arguments = segment[index + 1 :]
        if executable == "git":
            subcommand = _git_subcommand(arguments)
            if subcommand in ("status", "diff", "add", "commit", "log", "rev-parse", "rev-list"):
                continue
            return False
        if executable in ("date", "mkdir", "echo", "cat", "pwd", "ls", "test", "true"):
            continue
        return False
    return True


def _git_subcommand(arguments: list[str]) -> str | None:
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "-C" and index + 1 < len(arguments):
            index += 2
        elif argument.startswith("-"):
            index += 1
        else:
            return argument
    return None


def _gh_subcommand(arguments: list[str]) -> tuple[str, str] | None:
    words = [argument for argument in arguments if not argument.startswith("-")]
    if len(words) < 2:
        return None
    return words[0], words[1]


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _truncate(text: str) -> str:
    if len(text) <= _MAX_LOG_DETAIL:
        return text
    return text[:_MAX_LOG_DETAIL] + "...[truncated]"


def _json_or_repr(value: Any) -> str:
    try:
        return json.dumps(value, indent=2, default=str, sort_keys=True)
    except TypeError:
        return repr(value)


def _outcome_hint(response: Any) -> str | None:
    """Best-effort success/error hint; most tool responses carry no such field."""

    if not isinstance(response, Mapping):
        return None
    if response.get("interrupted") is True:
        return "interrupted"
    is_error = response.get("is_error")
    if isinstance(is_error, bool):
        return "error" if is_error else "ok"
    return None
