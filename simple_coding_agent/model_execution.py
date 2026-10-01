"""Claude SDK boundary for one bounded implementation attempt."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
import json
from pathlib import Path
import re
import shlex
from typing import Any, Protocol

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    SystemMessage,
    StreamEvent,
    TextBlock,
    UserMessage,
)

from simple_coding_agent.config import RuntimeConfig
from simple_coding_agent.token_ledger import (
    USAGE_CATEGORIES,
    TokenLedger,
    hit_rate_of,
    usage_by_category,
)


_SKILLS = ["implement", "tdd", "code-review", "codebase-design", "handoff"]
_META_BASE_URL = "https://api.meta.ai"

# The per-attempt token ledger (``token_ledger.TokenLedger``) is the live
# budget reading: every streamed ``AssistantMessage`` counts its response as
# an estimate on first sight, ``StreamEvent`` ``message_delta`` usage and
# non-zero ``AssistantMessage.usage`` replace the estimate with the measured
# categories, and tool results / user messages grow the thread context. The
# soft threshold is evaluated against ``ledger.budget_tokens`` on every ledger
# update; only ``limits_checked`` logging waits for a response to settle.
_HANDOFF_FOLLOWUP_TIMEOUT = 300
# One-time ``prompt_cache_ineffective`` diagnostic: evaluated when the main
# thread's K-th measured response settles. Observational only.
PROMPT_CACHE_MIN_MEASURED_RESPONSES = 20
PROMPT_CACHE_HIT_RATE_THRESHOLD = 0.5

_OPERATOR_HANDOFF_DELIVERY_WINDOW_SECONDS = 60
_OPERATOR_HANDOFF_MODEL_DEADLINE_SECONDS = 240

_COST_HANDOFF_INSTRUCTION = (
    "This attempt is approaching its token budget. Invoke the `handoff` skill now "
    "to preserve your progress cooperatively instead of continuing further work."
)
_COST_HANDOFF_FOLLOWUP_PROMPT = (
    "This attempt crossed its token soft threshold and must now end by handing off. "
    "Invoke the `handoff` skill now to preserve your progress: commit any outstanding "
    "work, then write and commit the handoff note. Do not attempt further implementation work."
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
    token_hard_ceiling_reached: bool = False
    token_budget: Mapping[str, Any] | None = None


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
    """Run the approved upstream skills and retain terminal stream evidence.

    Token soft-threshold policy: once the ledger budget crosses the soft
    threshold, every running subagent (including background ones) is
    stopped: subsequent subagent tool calls are denied and the cancelled
    ids are recorded in a ``token_soft_threshold_subagents_stopped`` event.
    Only the main thread is steered toward the ``handoff`` skill after the
    crossing; new subagent launches from the main thread are denied. The
    handoff instruction is delivered on the next tool boundary once no
    subagent is left running, without waiting for a main-thread boundary.
    A ``handoff`` skill invocation counts only when observed in the main
    thread (``agent_id is None``). Partial subagent output already written
    to the working tree is left as is for the normal dirty-work
    preservation.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        client_factory: ClientFactory = ClaudeSDKClient,
        clock: Clock = lambda: datetime.now(UTC),
        event_log: EventLog = lambda event, detail="", level="INFO", issue_number=None: None,
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
        self._started_at = self._clock()
        self._soft_threshold_crossed = False
        self._soft_threshold_crossed_at_tokens: float | None = None
        self._total_cost_usd: float | None = None
        self._handoff_context_delivered = False
        self._running_subagents: set[str] = set()
        self._stopped_subagent_ids: list[str] = []
        self._usage_shape_logged = False
        self._operator_handoff_provider: OperatorHandoffProvider | None = None
        self._operator_handoff_reporter: OperatorHandoffReporter | None = None
        self._operator_handoff: OperatorHandoff | None = None
        self._operator_context_delivered = False
        self._operator_fallback_used = False
        self._operator_begun = False
        self._operator_deadline_expired = False
        self._cli_session_id: str | None = None
        self._token_hard_ceiling_reached = False
        self._reset_ledger_state()

    def _reset_ledger_state(self) -> None:
        """Start a fresh per-attempt ledger (hooks may run without ``execute``)."""

        self._ledger = TokenLedger()
        self._ledger_threads: set[str] = set()
        self._stream_response_ids: dict[str, str] = {}
        self._pending_usage: dict[tuple[str, str], dict[str, int]] = {}
        self._counted_response_ids: set[tuple[str, str]] = set()
        self._limits_logged: set[tuple[str, str]] = set()
        self._degraded_logged = False
        self._cache_warning_evaluated = False
        self._auto_response_seq = 0

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

    @property
    def cli_session_id(self) -> str | None:
        """The CLI session id latched during this execution, if any."""

        return self._cli_session_id

    def _log_context_compacted(self, message: SystemMessage) -> None:
        """Log a CLI context compaction.

        The budget is left alone; the ledger only learns of the compaction so
        the next response is excluded from cache-miss accounting. SDK 0.2.156
        has no typed message for it and ``compact_metadata`` is unverified,
        so ``trigger`` and ``pre_tokens`` are read best-effort.
        """

        data = getattr(message, "data", None)
        data = data if isinstance(data, Mapping) else {}
        metadata = data.get("compact_metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        thread = _thread_label_from(getattr(message, "parent_tool_use_id", None))
        self._ledger.notify_compaction(thread)
        self._log(
            "context_compacted",
            f"thread={thread}; trigger={metadata.get('trigger')}; "
            f"pre_tokens={metadata.get('pre_tokens')}",
        )

    def _record_cli_session_id(self, message: object) -> None:
        """Latch the CLI session id from the init message or any sessioned message.

        The pinned SDK exposes ``session_id`` directly on ``ResultMessage``
        (and on ``AssistantMessage``); the init ``SystemMessage`` carries it
        inside ``data`` instead, so both shapes are read. Every message in one
        execution belongs to the same CLI session, so later messages may
        overwrite earlier ones.
        """

        session_id = getattr(message, "session_id", None)
        if not isinstance(session_id, str) or not session_id:
            session_id = None
            if isinstance(message, SystemMessage) and message.subtype == "init":
                data = getattr(message, "data", None)
                if isinstance(data, Mapping):
                    candidate = data.get("session_id")
                    if isinstance(candidate, str) and candidate:
                        session_id = candidate
        if session_id is not None:
            self._cli_session_id = session_id

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

    def _hard_token_limit_reached(self) -> bool:
        return (
            self._token_hard_ceiling_reached
            or self._ledger.budget_tokens >= self._config.max_budget_tokens
        )

    def _operator_fallback_blocked(self) -> bool:
        """Whether the single fallback query would bypass the hard token limit."""

        return self._hard_token_limit_reached()

    async def _run_operator_fallback(
        self, client: SDKClient, observed_models: list[str]
    ) -> tuple[ResultMessage | None, tuple[str, ...]] | None:
        """Interrupt, drain, and issue the single follow-up handoff query.

        Returns the follow-up terminal result (and merged models) to
        propagate, or ``None`` to keep reading the original stream. A blocked
        fallback (hard token limit) consumes the single attempt without
        querying: the stream's own terminal result then decides the outcome.
        """

        latched = self._operator_handoff
        self._operator_fallback_used = True
        if latched is None:
            return None
        if self._operator_fallback_blocked():
            self._log(
                "operator_handoff_fallback_blocked",
                f"request={latched.request_id}; token budget reached the hard"
                " token limit, so no follow-up query is issued and the stream's"
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

    def _log(self, event: str, detail: str = "", level: str = "INFO") -> None:
        self._event_log(event, detail, level=level, issue_number=self._issue_number)

    def _start_launch_thread(self, hook_input: Any, tool_use_id: str | None) -> None:
        """Start a subagent ledger thread at launch, seeded with its prompt."""

        if tool_use_id is None:
            return
        tool_name = _hook_field(hook_input, "tool_name")
        if tool_name != "Agent" and not (
            tool_name == "Task" and _is_subagent_launch(hook_input)
        ):
            return
        tool_input = _hook_field(hook_input, "tool_input", {})
        prompt = ""
        subagent_type = None
        if isinstance(tool_input, Mapping):
            raw_prompt = tool_input.get("prompt", tool_input.get("description", ""))
            prompt = raw_prompt if isinstance(raw_prompt, str) else ""
            raw_type = tool_input.get("subagent_type")
            subagent_type = raw_type if isinstance(raw_type, str) else None
        self._ensure_ledger_thread(
            tool_use_id, subagent_type=subagent_type, prompt_chars=len(prompt)
        )

    def _track_subagent_seen(self, hook_input: Any) -> None:
        """Remember a subagent id seen before the soft-threshold crossing.

        Only pre-crossing ids count as running: after the crossing every
        subagent is stopped, so later ids are recorded as stopped instead
        of re-populating the running set.
        """

        agent_id = _hook_agent_id(hook_input)
        if agent_id is None:
            return
        if agent_id in self._stopped_subagent_ids:
            return
        self._running_subagents.add(agent_id)

    def _stop_running_subagents(self) -> None:
        """Cancel every running subagent and record the cancelled ids."""

        stopped = sorted(self._running_subagents)
        for agent_id in stopped:
            if agent_id not in self._stopped_subagent_ids:
                self._stopped_subagent_ids.append(agent_id)
        self._running_subagents.clear()
        ids_detail = ",".join(stopped)
        self._log(
            "token_soft_threshold_subagents_stopped",
            f"stopped_agent_ids={ids_detail}; count={len(stopped)}; "
            "in-flight subagent tools are denied after the token soft-threshold crossing "
            "and only handoff-related main-thread commands stay allowed.",
        )

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
        self._soft_threshold_crossed = False
        self._soft_threshold_crossed_at_tokens = None
        self._total_cost_usd = None
        self._token_hard_ceiling_reached = False
        self._handoff_context_delivered = False
        self._running_subagents = set()
        self._stopped_subagent_ids = []
        self._usage_shape_logged = False
        self._reset_ledger_state()
        self._ensure_ledger_thread("main", prompt_chars=len(issue_body))
        self._operator_handoff = None
        self._operator_context_delivered = False
        self._operator_fallback_used = False
        self._operator_begun = False
        self._operator_deadline_expired = False
        self._cli_session_id = None
        self._log(
            "model_execution_started",
            f"issue_body_length={len(issue_body)}; limits: "
            f"max_budget_usd={self._config.max_budget_usd:.4f}; "
            f"max_budget_tokens={self._config.max_budget_tokens}; "
            f"soft_threshold_tokens={self._config.soft_threshold_tokens}; "
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
                    if self._token_hard_ceiling_reached:
                        return self._hard_ceiling_evidence(None, ())
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
                if self._token_hard_ceiling_reached:
                    return self._classify(terminal, observed_models)
                if terminal is not None and self._is_model_limit(terminal, observed_models):
                    followup = await self._attempt_handoff_followup(client, observed_models)
                    if followup is not None:
                        return followup
                elif terminal is not None and self._needs_cost_handoff_followup(
                    terminal, observed_models
                ):
                    self._log(
                        "token_soft_threshold_handoff_followup",
                        "Terminal result arrived after the token soft-threshold"
                        " crossing without a main-thread handoff; issuing the"
                        " single token follow-up handoff prompt.",
                    )
                    followup = await self._attempt_handoff_followup(
                        client,
                        observed_models,
                        prompt=_COST_HANDOFF_FOLLOWUP_PROMPT,
                        success_explanation=(
                            "Model invoked the handoff skill after a token"
                            " soft-threshold instruction."
                        ),
                    )
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

    def _main_thread_handoff_invoked(self) -> bool:
        """Whether the ``handoff`` skill was invoked in the main thread.

        Subagent skill invocations cannot perform the handoff (they share
        neither the main transcript nor the terminal classification), so only
        main-thread invocations (``agent_id is None``) count.
        """

        return any(
            event.name == "handoff" and event.agent_id is None
            for event in self._skill_events
        )

    def _cost_followup_blocked(self) -> bool:
        """Whether the cost follow-up would bypass the hard token limit."""

        return self._hard_token_limit_reached()

    def _needs_cost_handoff_followup(
        self, terminal: ResultMessage, observed_models: tuple[str, ...]
    ) -> bool:
        """Whether a clean terminal result after the crossing deserves one handoff chance.

        Only ordinary successful-looking results qualify: limits already got
        their follow-up above (``elif``), errors/timeouts/aborts and model
        mismatches are not handoff-able successes, and a follow-up past the
        hard token limit would itself bypass the budget.
        """

        if not self._soft_threshold_crossed or self._main_thread_handoff_invoked():
            return False
        if self._cost_followup_blocked():
            self._log(
                "token_soft_threshold_handoff_followup_blocked",
                "token budget reached the hard token limit, so no token"
                " follow-up query is issued and the terminal result decides"
                " the outcome",
            )
            return False
        if getattr(terminal, "is_error", False):
            return False
        stop_reason = getattr(terminal, "stop_reason", None)
        if stop_reason == "timeout" or (
            isinstance(stop_reason, str) and stop_reason.startswith("aborted_")
        ):
            return False
        if getattr(terminal, "terminal_reason", None) in (
            "aborted_streaming",
            "aborted_tools",
        ):
            return False
        model_usage = getattr(terminal, "model_usage", None)
        result_models = (
            tuple(model_usage.keys()) if isinstance(model_usage, Mapping) else ()
        )
        all_models = tuple(dict.fromkeys((*observed_models, *result_models)))
        if any(model != self._config.model for model in all_models):
            return False
        return True

    async def _attempt_handoff_followup(
        self,
        client: SDKClient,
        observed_models: tuple[str, ...],
        *,
        prompt: str = _LIMIT_HANDOFF_FOLLOWUP_PROMPT,
        success_explanation: str = (
            "Model invoked the handoff skill after reaching a turns/timeout limit."
        ),
    ) -> ModelExecution | None:
        """Best-effort same-client follow-up after a turns/timeout/token limit.

        Returns a HANDOFF_REQUESTED evidence only when the model actually
        invoked the handoff skill in the main thread during the follow-up;
        otherwise returns None so the caller falls back to its ordinary limit
        classification. This never retries: a follow-up that fails or times
        out is itself evidence that handoff is not achievable right now.
        """

        try:
            await client.query(prompt)
            async with asyncio.timeout(_HANDOFF_FOLLOWUP_TIMEOUT):
                followup_terminal, followup_models = await self._receive_terminal(client)
        except Exception:
            return None
        if followup_terminal is None:
            return None
        all_models = tuple(dict.fromkeys((*observed_models, *followup_models)))
        if not self._main_thread_handoff_invoked():
            return None
        self._note_terminal(followup_terminal)
        return self._evidence(
            ModelExecutionStatus.HANDOFF_REQUESTED,
            success_explanation,
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
            include_partial_messages=True,
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
        # One pending ``__anext__()`` shared across loop iterations: waiting
        # for it with ``asyncio.wait`` (instead of ``asyncio.timeout``) lets
        # operator boundaries pass without cancelling the generator, which
        # would close it and end the stream with ``StopAsyncIteration``.
        pending: asyncio.Task | None = None
        try:
            while True:
                action = self._operator_due_action()
                if action == "fallback":
                    await _cancel_stream_wait(pending)
                    pending = None
                    outcome = await self._run_operator_fallback(client, observed_models)
                    if outcome is not None:
                        return outcome
                    continue
                if action == "expire":
                    await _cancel_stream_wait(pending)
                    pending = None
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
                if pending is None:
                    pending = asyncio.ensure_future(stream.__anext__())
                if wait is None:
                    try:
                        message = await pending
                    except StopAsyncIteration:
                        return None, tuple(observed_models)
                    pending = None
                else:
                    done, _ = await asyncio.wait({pending}, timeout=wait)
                    if not done:
                        # Operator boundary reached; the generator stays alive.
                        continue
                    task, pending = pending, None
                    try:
                        message = task.result()
                    except StopAsyncIteration:
                        return None, tuple(observed_models)
                self._record_cli_session_id(message)
                if isinstance(message, SystemMessage) and message.subtype == "compact_boundary":
                    self._log_context_compacted(message)
                if isinstance(message, ResultMessage) or _looks_like_result(message):
                    return message, tuple(observed_models)
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock) and block.text.strip():
                            self._log("model_response", self._model_response_detail(block.text))
                    self._observe_assistant_message(message)
                elif isinstance(message, StreamEvent):
                    self._observe_stream_event(message)
                elif isinstance(message, UserMessage):
                    self._observe_user_message(message)
                if self._token_hard_ceiling_reached:
                    await client.interrupt()
                    drained = await self._drain(client)
                    return drained, tuple(observed_models)
                model = getattr(message, "model", None)
                if isinstance(model, str):
                    observed_models.append(model)
        finally:
            # The outer model-timeout/follow-up timeout may cancel this method
            # while a ``__anext__()`` is still pending; never leave it dangling.
            await _cancel_stream_wait(pending)

    def _log_usage_shape(self, usage: Any) -> None:
        if not self._usage_shape_logged:
            self._usage_shape_logged = True
            self._log("model_usage_shape", f"usage={usage!r}")

    def _ensure_ledger_thread(
        self, thread_id: str, *, subagent_type: str | None = None, prompt_chars: int = 0
    ) -> None:
        """Start a ledger thread on first sight (``main`` or ``parent_tool_use_id``)."""

        if thread_id in self._ledger_threads:
            return
        self._ledger.start_thread(
            thread_id,
            kind="main" if thread_id == "main" else "subagent",
            prompt_chars=prompt_chars,
            subagent_type=subagent_type,
        )
        self._ledger_threads.add(thread_id)

    def _assistant_response_id(self, message: AssistantMessage, thread_id: str) -> str:
        """Stable response id: the API message id, else one id per message."""

        message_id = getattr(message, "message_id", None)
        if isinstance(message_id, str) and message_id:
            return message_id
        uuid = getattr(message, "uuid", None)
        if isinstance(uuid, str) and uuid:
            return uuid
        self._auto_response_seq += 1
        return f"auto-{thread_id}-{self._auto_response_seq}"

    def _observe_assistant_message(self, message: AssistantMessage) -> None:
        """Feed one streamed assistant message into the token ledger.

        The response is counted as an estimate on first sight; reported
        usage (here or stashed from an earlier ``message_delta``) replaces
        the estimate per category by max, keyed by response id.
        """

        thread_id = getattr(message, "parent_tool_use_id", None) or "main"
        self._ensure_ledger_thread(thread_id)
        response_id = self._assistant_response_id(message, thread_id)
        content = getattr(message, "content", [])
        usage = getattr(message, "usage", None)
        self._ledger.observe_response(thread_id, response_id, _content_chars(content))
        self._counted_response_ids.add((thread_id, response_id))
        if isinstance(usage, Mapping):
            self._ledger.observe_usage(thread_id, response_id, usage)
        pending = self._pending_usage.pop((thread_id, response_id), None)
        if pending:
            self._ledger.observe_usage(thread_id, response_id, pending)
        self._log_usage_shape(usage)
        self._after_ledger_update(thread_id)

    def _observe_stream_event(self, message: StreamEvent) -> None:
        """Merge streamed ``message_delta`` usage into the ledger.

        The response id comes from that stream's ``message_start``. A delta
        for a response not yet counted (it arrives just after PreToolUse,
        before the response's ``AssistantMessage``) is stashed until the
        response is counted; a delta with no ``message_start`` is ignored.
        """

        event = getattr(message, "event", None)
        if not isinstance(event, Mapping):
            return
        event_type = event.get("type")
        thread_id = getattr(message, "parent_tool_use_id", None) or "main"
        self._ensure_ledger_thread(thread_id)
        if event_type == "message_start":
            inner = event.get("message")
            if isinstance(inner, Mapping):
                response_id = inner.get("id")
                if isinstance(response_id, str) and response_id:
                    self._stream_response_ids[thread_id] = response_id
        elif event_type == "message_delta":
            usage = event.get("usage")
            if not isinstance(usage, Mapping):
                return
            response_id = self._stream_response_ids.get(thread_id)
            if response_id is None:
                return
            if (thread_id, response_id) in self._counted_response_ids:
                self._ledger.observe_usage(thread_id, response_id, usage)
            else:
                pending = self._pending_usage.setdefault(
                    (thread_id, response_id),
                    {category: 0 for category in USAGE_CATEGORIES},
                )
                for category in USAGE_CATEGORIES:
                    reported = usage.get(category, 0)
                    pending[category] = max(
                        pending[category], int(reported or 0)
                    )
            self._after_ledger_update(thread_id)

    def _observe_user_message(self, message: UserMessage) -> None:
        """Grow the thread context with a streamed tool result / user message."""

        thread_id = getattr(message, "parent_tool_use_id", None) or "main"
        self._ensure_ledger_thread(thread_id)
        self._ledger.observe_input(thread_id, _content_chars(getattr(message, "content", "")))
        self._after_ledger_update(thread_id)

    def _after_ledger_update(self, thread_id: str) -> None:
        """Check the soft threshold and log settled responses after any ledger mutation."""

        self._check_limits(thread_id)

        notice = self._ledger.first_settled_without_usage
        if notice is not None and not self._degraded_logged:
            self._degraded_logged = True
            self._log(
                "token_estimate_degraded",
                f"thread={notice.thread}; subagent_type={notice.subagent_type}; "
                f"estimated_tokens={self._ledger.estimated_tokens:.0f}",
                level="WARNING",
            )
        for response_id in self._ledger.settled_response_ids(thread_id):
            if (thread_id, response_id) not in self._limits_logged:
                self._limits_logged.add((thread_id, response_id))
                self._log("limits_checked", self._token_limits_detail(thread_id, response_id))
        self._check_prompt_cache()

    def _token_limits_detail(self, thread_id: str, response_id: str) -> str:
        if self._ledger.has_thread("main"):
            main_context = self._ledger.context_tokens("main")
            turns = self._ledger.response_count("main")
        else:
            main_context = 0.0
            turns = 0
        detail = (
            f"thread={thread_id}; "
            f"budget_tokens={self._ledger.budget_tokens:.0f}; "
            f"measured_tokens={self._ledger.measured_tokens:.0f}; "
            f"estimated_tokens={self._ledger.estimated_tokens:.0f}; "
            f"soft_threshold_tokens={self._config.soft_threshold_tokens}; "
            f"max_budget_tokens={self._config.max_budget_tokens}; "
            f"main_context_tokens={main_context:.0f}; "
            f"turns={turns}; "
            f"max_turns={self._config.max_turns}; "
            f"elapsed_seconds={self._elapsed_seconds()}; "
            f"timeout_seconds={self._config.model_timeout}"
        )
        metrics = self._ledger.response_cache_metrics(thread_id, response_id)
        if metrics is None:
            return detail
        miss = "excluded" if metrics.miss_tokens is None else metrics.miss_tokens
        return (
            f"{detail}; input_tokens={metrics.input_tokens}; "
            f"cache_read_tokens={metrics.cache_read_tokens}; "
            f"cache_creation_tokens={metrics.cache_creation_tokens}; "
            f"cache_hit_rate={_round_ratio(metrics.hit_rate)}; "
            f"cache_miss_tokens={miss}"
        )

    def _check_prompt_cache(self) -> None:
        """Warn once if the cache is ineffective over the first K measured main responses."""

        k = PROMPT_CACHE_MIN_MEASURED_RESPONSES
        if self._cache_warning_evaluated or self._ledger.main_measured_settled_count < k:
            return
        self._cache_warning_evaluated = True
        hit_rate = self._ledger.main_hit_rate(k)
        if hit_rate is None or hit_rate >= PROMPT_CACHE_HIT_RATE_THRESHOLD:
            return
        totals = self._ledger.main_measured_by_category(k)
        self._log(
            "prompt_cache_ineffective",
            f"measured_responses={k}; main_hit_rate={_round_ratio(hit_rate)}; "
            f"hit_rate_threshold={PROMPT_CACHE_HIT_RATE_THRESHOLD}; "
            f"input_tokens={totals['input_tokens']}; "
            f"cache_read_tokens={totals['cache_read_input_tokens']}; "
            f"cache_creation_tokens={totals['cache_creation_input_tokens']}",
            level="WARNING",
        )

    def _elapsed_seconds(self) -> int:
        delta = (self._clock() - self._started_at).total_seconds()
        return max(0, int(delta))

    def _check_limits(self, thread_id: str = "main") -> None:
        """Latch one-time soft-threshold and hard-ceiling crossings from the ledger."""

        budget = self._ledger.budget_tokens
        threshold = self._config.soft_threshold_tokens
        if not self._soft_threshold_crossed and budget >= threshold:
            self._soft_threshold_crossed = True
            self._soft_threshold_crossed_at_tokens = budget
            self._log(
                "token_soft_threshold_crossed",
                f"budget_tokens={budget:.0f}; soft_threshold_tokens={threshold}; "
                f"thread={thread_id}",
            )
            self._stop_running_subagents()
        ceiling = self._config.max_budget_tokens
        if not self._token_hard_ceiling_reached and budget >= ceiling:
            self._token_hard_ceiling_reached = True
            self._log(
                "token_hard_ceiling_reached",
                f"budget_tokens={budget:.0f}; max_budget_tokens={ceiling}; "
                f"thread={thread_id}; "
                f"in_flight_estimated_tokens={self._ledger.estimated_tokens:.0f}",
                level="WARNING",
            )

    async def _drain(self, client: SDKClient) -> ResultMessage | None:
        """Consume buffered messages after interrupting before client teardown.

        A terminal ``ResultMessage`` that arrives while draining is returned
        so its ``model_usage`` stays available for reconciliation; every other
        message is discarded.
        """

        terminal: ResultMessage | None = None
        try:
            async with asyncio.timeout(10):
                async for message in client.receive_response():
                    if terminal is None and (
                        isinstance(message, ResultMessage)
                        or _looks_like_result(message)
                    ):
                        terminal = message
        except Exception:
            # Client teardown still runs; drain evidence is secondary to avoiding an orphan.
            pass
        return terminal

    def _classify(
        self, terminal: ResultMessage | None, observed_models: tuple[str, ...]
    ) -> ModelExecution:
        self._note_terminal(terminal)
        if self._token_hard_ceiling_reached:
            return self._hard_ceiling_evidence(
                getattr(terminal, "model_usage", None),
                observed_models,
                stop_reason=getattr(terminal, "stop_reason", None),
                terminal_reason=getattr(terminal, "terminal_reason", None),
            )
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
        if self._soft_threshold_crossed and self._main_thread_handoff_invoked():
            return self._evidence(
                ModelExecutionStatus.HANDOFF_REQUESTED,
                "Model invoked the handoff skill after a token soft-threshold instruction.",
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
        if self._soft_threshold_crossed and not self._main_thread_handoff_invoked():
            # The crossing was never converted into a handoff (the instruction
            # may have gone unseen or the follow-up failed), so this must not
            # read as an ordinary success: the attempt ends incomplete.
            return self._evidence(
                ModelExecutionStatus.MODEL_LIMIT_REACHED,
                "Token soft threshold was crossed without a main-thread handoff, "
                "so the attempt ends incomplete instead of succeeding.",
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

    def _hard_ceiling_evidence(
        self,
        model_usage: Mapping[str, Any] | None,
        observed_models: tuple[str, ...],
        *,
        stop_reason: str | None = None,
        terminal_reason: str | None = None,
    ) -> ModelExecution:
        """Classify a latched token hard-ceiling stop as ``MODEL_LIMIT_REACHED``.

        The flag alone decides: a missing terminal ``ResultMessage`` after the
        interrupt is not an infrastructure error, and no follow-up of any
        kind is issued. ``model_usage`` from a terminal that arrived while
        draining is kept for reconciliation.
        """

        return self._evidence(
            ModelExecutionStatus.MODEL_LIMIT_REACHED,
            "Token budget reached the hard ceiling "
            f"(budget_tokens={self._ledger.budget_tokens:.0f}; "
            f"max_budget_tokens={self._config.max_budget_tokens}); "
            "the attempt was stopped unconditionally and no further "
            "implementation work was permitted.",
            stop_reason,
            model_usage,
            observed_models,
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
        token_budget = self._reconcile_token_budget(model_usage)
        self._log("model_execution_finished", f"status={status}; {explanation}")
        return ModelExecution(
            status=status,
            explanation=explanation,
            stop_reason=stop_reason,
            model_usage=model_usage,
            observed_models=observed_models,
            skill_events=self.skill_events,
            terminal_reason=terminal_reason,
            token_hard_ceiling_reached=self._token_hard_ceiling_reached,
            token_budget=token_budget,
        )

    def _note_terminal(self, terminal: object) -> None:
        cost = getattr(terminal, "total_cost_usd", None)
        if cost is not None:
            self._total_cost_usd = cost

    def _reconcile_token_budget(
        self, model_usage: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """Reconcile the ledger with ``model_usage`` and log ``token_budget_reconciled``."""

        usage = model_usage if isinstance(model_usage, Mapping) else None
        reconciliation = self._ledger.reconcile(usage)
        actual_totals = _sum_usage_by_category(usage) if usage is not None else None
        main_stats = self._ledger.main_cache_stats()
        all_threads_hit_rate = None if actual_totals is None else hit_rate_of(actual_totals)
        summary: dict[str, Any] = {
            "mode": reconciliation.mode,
            "max_budget_tokens": self._config.max_budget_tokens,
            "soft_threshold_tokens": self._config.soft_threshold_tokens,
            "estimated_tokens": round(reconciliation.estimated_tokens),
            "actual_tokens": reconciliation.actual_tokens,
            "error_ratio": _round_ratio(reconciliation.error_ratio),
            "unreported": {
                "estimated_tokens": round(reconciliation.unreported_estimated_tokens),
                "actual_tokens": reconciliation.unreported_actual_tokens,
                "error_ratio": _round_ratio(reconciliation.unreported_error_ratio),
            },
            "measured_by_category": _short_categories(self._ledger.measured_by_category),
            "actual_by_category": (
                None if actual_totals is None else _short_categories(actual_totals)
            ),
            "prompt_cache": {
                "main_hit_rate": _round_ratio(main_stats.hit_rate),
                "all_threads_hit_rate": _round_ratio(all_threads_hit_rate),
                "main_cache_miss_tokens": main_stats.miss_tokens,
                "main_miss_responses_counted": main_stats.counted,
                "main_miss_responses_excluded": main_stats.excluded,
            },
            "soft_threshold_crossed": self._soft_threshold_crossed,
            "soft_threshold_crossed_at_tokens": (
                None
                if self._soft_threshold_crossed_at_tokens is None
                else round(self._soft_threshold_crossed_at_tokens)
            ),
            "hard_ceiling_reached": self._token_hard_ceiling_reached,
            "peak_main_context_tokens": round(self._ledger.peak_main_context_tokens),
            "total_cost_usd": self._total_cost_usd,
        }
        self._log("token_budget_reconciled", json.dumps(summary, sort_keys=True))
        return summary

    async def _guard_and_record_pre_tool_use(
        self, hook_input: Any, tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        if self._token_hard_ceiling_reached:
            return _deny(
                "Token budget reached the hard ceiling "
                f"(max_budget_tokens={self._config.max_budget_tokens}); "
                "the attempt was stopped unconditionally and no further tool"
                " use is allowed, including the handoff skill and git commands."
            )
        self._log_tool_use("tool_call", hook_input)
        await self._record_skill_event("PreToolUse", hook_input, tool_use_id, context)
        self._start_launch_thread(hook_input, tool_use_id)
        if _skill_name(hook_input) == "code-review":
            self._review_count += 1
            if self._review_count > 3:
                return _deny("At most two code-review repair cycles are allowed.")
        if not self._soft_threshold_crossed:
            self._track_subagent_seen(hook_input)
        if self._soft_threshold_crossed and not self._main_thread_handoff_invoked():
            if not _is_main_thread(hook_input):
                agent_id = _hook_agent_id(hook_input)
                if agent_id is not None and agent_id not in self._stopped_subagent_ids:
                    self._stopped_subagent_ids.append(agent_id)
                return _cost_deny("Subagent work is stopped after the token soft threshold.")
            else:
                tool_name = _hook_field(hook_input, "tool_name")
                if tool_name == "Skill" and _skill_name(hook_input) == "handoff":
                    pass
                elif tool_name == "Agent" or (
                    tool_name == "Task" and _is_subagent_launch(hook_input)
                ):
                    return _cost_deny("New subagent launches are disabled.")
                elif tool_name in ("Edit", "Write"):
                    file_path = str(_hook_field(hook_input, "tool_input", {}).get("file_path", ""))
                    if ".agent/handoff" not in file_path:
                        return _cost_deny("Further implementation work is disabled.")
                elif tool_name == "Bash":
                    command = str(_hook_field(hook_input, "tool_input", {}).get("command", ""))
                    if not _is_handoff_command(command):
                        return _cost_deny("Further implementation work is disabled.")
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
        if not self._soft_threshold_crossed:
            self._track_subagent_seen(hook_input)
        self._poll_operator_handoff()
        contexts: list[str] = []
        if self._soft_threshold_crossed and not self._handoff_context_delivered:
            if _is_main_thread(hook_input) or not self._running_subagents:
                # Once no subagent is left running the handoff instruction is
                # delivered on the very next tool boundary (whatever thread
                # it arrives on) instead of waiting for a main-thread
                # boundary; stopping the subagents at the crossing is what
                # frees the remaining budget for the handoff.
                self._handoff_context_delivered = True
                self._log("token_soft_threshold_handoff_context_injected")
                contexts.append(_COST_HANDOFF_INSTRUCTION)
            else:
                # Subagents are still recorded as running (for example the
                # crossing flag was set without going through the stop
                # path): a subagent cannot perform the handoff, so the
                # instruction stays withheld until they are stopped.
                pass
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
        thread = _thread_label(hook_input)
        agent_id = _hook_agent_id(hook_input)
        if event == "tool_result":
            event_name = "skill_result" if skill_name else event
            # Skill calls are already small; only offload plain tool payloads to disk.
            detail = (
                f"{label}{thread}: {response!r}"
                if skill_name
                else self._tool_result_detail(label, response, thread=thread, agent_id=agent_id)
            )
        else:
            tool_input = _hook_field(hook_input, "tool_input", {})
            event_name = "skill_call" if skill_name else event
            detail = (
                f"{label}{thread}: {tool_input!r}"
                if skill_name
                else self._tool_call_detail(label, tool_input, thread=thread, agent_id=agent_id)
            )
        self._log(event_name, _truncate(detail))

    def _tool_call_detail(
        self, label: str, tool_input: Any, *, thread: str = "", agent_id: str | None = None
    ) -> str:
        path = self._archive_evidence(
            "tool_call", _json_or_repr(_with_agent_id(tool_input, agent_id)), extension="json"
        )
        if path is None:
            return f"{label}{thread}: {tool_input!r}"
        return f"{label}{thread} -> {path}"

    def _tool_result_detail(
        self, label: str, response: Any, *, thread: str = "", agent_id: str | None = None
    ) -> str:
        path = self._archive_evidence(
            "tool_result", _json_or_repr(_with_agent_id(response, agent_id)), extension="json"
        )
        if path is None:
            return f"{label}{thread}: {response!r}"
        outcome = _outcome_hint(response)
        return f"{label}{thread} -> {path}" + (f" ({outcome})" if outcome else "")

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


def _round_ratio(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _short_categories(totals: Mapping[str, int]) -> dict[str, int]:
    """Map usage category names to the short ``token_budget`` keys."""

    return {
        "input": totals["input_tokens"],
        "cache_read": totals["cache_read_input_tokens"],
        "cache_creation": totals["cache_creation_input_tokens"],
        "output": totals["output_tokens"],
    }


def _sum_usage_by_category(model_usage: Mapping[str, Any]) -> dict[str, int]:
    """Sum ``model_usage`` across models, per category."""

    totals = {category: 0 for category in USAGE_CATEGORIES}
    for usage in model_usage.values():
        if isinstance(usage, Mapping):
            for category, value in usage_by_category(usage).items():
                totals[category] += value
    return totals


def _thread_label_from(parent_tool_use_id: object) -> str:
    return parent_tool_use_id if isinstance(parent_tool_use_id, str) and parent_tool_use_id else "main"


def _deny(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _cost_deny(restriction: str) -> dict[str, Any]:
    """Deny with the shared cost-guard wording for the given restriction."""

    return _deny(
        "Token soft threshold reached. "
        f"{restriction} "
        "You must invoke the `handoff` skill now to preserve your progress."
    )


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


async def _cancel_stream_wait(pending: asyncio.Task | None) -> None:
    """Cancel a pending stream ``__anext__()`` left over by an abandoned wait.

    Used only when the stream itself is abandoned (operator fallback/expire
    switches to a fresh stream, or an outer timeout cancels collection), so
    closing the generator here is intended. Never used for an operator wait
    boundary on a live stream: that path keeps ``pending`` alive across loop
    iterations instead.
    """

    if pending is None:
        return
    pending.cancel()
    try:
        await pending
    except asyncio.CancelledError:
        pass
    except StopAsyncIteration:
        pass
    except Exception:
        pass


def _looks_like_result(message: object) -> bool:
    """Permit fake SDK streams without weakening production ResultMessage handling."""

    return all(hasattr(message, attribute) for attribute in ("is_error", "stop_reason", "model_usage"))


def _skill_name(hook_input: object) -> str | None:
    tool_input = _hook_field(hook_input, "tool_input", {})
    name = tool_input.get("skill") if isinstance(tool_input, Mapping) else None
    return name if isinstance(name, str) else None


def _content_chars(value: object) -> int:
    """Approximate character count of streamed content or a tool payload.

    Visible assistant output (text blocks, tool-use input) sizes the
    response estimate; tool results and user messages grow the thread
    context. Anything unrecognised falls back to its ``repr`` length.
    """

    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, Mapping):
        return sum(_content_chars(key) + _content_chars(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return sum(_content_chars(item) for item in value)
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return len(text)
    total = 0
    for attribute in ("content", "input"):
        inner = getattr(value, attribute, None)
        if inner is not None:
            total += _content_chars(inner)
    if total:
        return total
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return len(name)
    return len(repr(value))


def _hook_field(hook_input: object, name: str, default: Any = None) -> Any:
    """Read both SDK TypedDict hooks and test doubles without losing provenance."""

    if isinstance(hook_input, Mapping):
        return hook_input.get(name, default)
    return getattr(hook_input, name, default)


def _hook_agent_id(hook_input: object) -> str | None:
    """The subagent id of a hook invocation, or None on the main thread."""

    agent_id = _hook_field(hook_input, "agent_id")
    return agent_id if isinstance(agent_id, str) and agent_id else None


def _is_main_thread(hook_input: object) -> bool:
    """Whether a hook invocation came from the main thread (no subagent id)."""

    return _hook_agent_id(hook_input) is None


def _thread_label(hook_input: object) -> str:
    """Short calling-thread suffix so tool logs show main vs subagent work."""

    agent_id = _hook_agent_id(hook_input)
    if agent_id is not None:
        return f" [subagent:{agent_id}]"
    return " [main]"


def _is_subagent_launch(hook_input: object) -> bool:
    """Whether a Task-shaped tool call spawns a subagent rather than polling one."""

    tool_input = _hook_field(hook_input, "tool_input", {})
    if not isinstance(tool_input, Mapping):
        return True
    return any(key in tool_input for key in ("prompt", "description", "subagent_type"))


def _with_agent_id(payload: Any, agent_id: str | None) -> Any:
    """Envelope an archived tool payload with its calling thread.

    ``agent_id`` is None on the main thread, so the evidence files show
    whether each tool call/result came from the main thread or a subagent.
    """

    if isinstance(payload, Mapping):
        return {"agent_id": agent_id, **payload}
    return {"agent_id": agent_id, "payload": payload}


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


# Read-only stdin-to-stdout filters: safe to append to handoff git commands
# (e.g. `git log | tail -3`) because they cannot write files, run commands,
# or reach the network. Anything else in a pipeline (test runners, curl,
# file removal, …) still fails the handoff allowlist.
_READONLY_OUTPUT_FILTERS = frozenset(
    {"tail", "head", "wc", "grep", "sort", "uniq", "cut", "tr"}
)
# Shell stderr duplication (`2>&1`) is stripped before segmentation: shlex
# would otherwise split the bare `&` into its own pipeline segment and the
# leftover `1` would read as a non-allowlisted executable.
_COST_OUTPUT_REDIRECT = re.compile(r"\d*>&\d+")


def _is_handoff_command(command: str) -> bool:
    command = _COST_OUTPUT_REDIRECT.sub("", command)
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
        if executable in _READONLY_OUTPUT_FILTERS:
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
