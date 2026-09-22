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

_HANDOFF_FOLLOWUP_TIMEOUT = 300

_COST_HANDOFF_INSTRUCTION = (
    "This attempt is approaching its cost budget. Invoke the `handoff` skill now "
    "to preserve your progress cooperatively instead of continuing further work."
)
_LIMIT_HANDOFF_FOLLOWUP_PROMPT = (
    "Execution reached its turn or time limit. Invoke the `handoff` skill now to "
    "preserve your progress: commit any outstanding work, then write and commit "
    "the handoff note. Do not attempt further implementation work."
)


class _CostEstimator:
    """Accumulate an approximate USD spend from streamed token usage."""

    def __init__(self, rate_per_million_tokens: float = _ESTIMATED_USD_PER_MILLION_TOKENS) -> None:
        self._rate = rate_per_million_tokens
        self._tokens = 0

    def observe(self, usage: Any) -> float:
        if isinstance(usage, Mapping):
            for field in _USAGE_TOKEN_FIELDS:
                value = usage.get(field)
                if isinstance(value, int):
                    self._tokens += value
        return self.estimated_cost_usd

    @property
    def estimated_cost_usd(self) -> float:
        return (self._tokens / 1_000_000) * self._rate


class ModelExecutionStatus(StrEnum):
    """Status of the SDK stream; the evaluator maps it to an attempt outcome."""

    SUCCEEDED = "succeeded"
    MODEL_LIMIT_REACHED = "model_limit_reached"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    HANDOFF_REQUESTED = "handoff_requested"


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
        self._cost_estimator = _CostEstimator()
        self._soft_threshold_crossed = False
        self._handoff_context_delivered = False

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
        self._cost_estimator = _CostEstimator()
        self._soft_threshold_crossed = False
        self._handoff_context_delivered = False
        self._log("model_execution_started", f"issue_body_length={len(issue_body)}")
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
                if terminal is not None and terminal.stop_reason == "max_turns_exceeded":
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
            return self._evidence(
                ModelExecutionStatus.INFRASTRUCTURE_ERROR,
                "Claude SDK stream ended without a terminal ResultMessage.",
                None,
                None,
                observed_models,
            )
        return self._classify(terminal, observed_models)

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
        async for message in client.receive_response():
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
        return None, tuple(observed_models)

    def _observe_cost(self, usage: Any) -> None:
        """Latch a one-time soft-threshold crossing from estimated cumulative cost."""

        if self._soft_threshold_crossed:
            return
        estimated_cost = self._cost_estimator.observe(usage)
        soft_threshold = self._config.max_budget_usd * (1 - self._config.soft_threshold_percentage)
        if estimated_cost >= soft_threshold:
            self._soft_threshold_crossed = True
            self._log(
                "cost_soft_threshold_crossed",
                f"estimated_cost_usd={estimated_cost:.4f}; soft_threshold_usd={soft_threshold:.4f}",
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
        if self._handoff_context_delivered and any(
            event.name == "handoff" for event in self._skill_events
        ):
            return self._evidence(
                ModelExecutionStatus.HANDOFF_REQUESTED,
                "Model invoked the handoff skill after a cost soft-threshold instruction.",
                stop_reason,
                model_usage,
                all_models,
            )
        if mismatches:
            return self._evidence(
                ModelExecutionStatus.INFRASTRUCTURE_ERROR,
                f"Observed model mismatch: {', '.join(mismatches)}.",
                stop_reason,
                model_usage,
                all_models,
            )
        if stop_reason == "max_turns_exceeded":
            return self._evidence(
                ModelExecutionStatus.MODEL_LIMIT_REACHED,
                "Model execution reached max_turns.",
                stop_reason,
                model_usage,
                all_models,
            )
        if stop_reason == "timeout" or (isinstance(stop_reason, str) and stop_reason.startswith("aborted_")):
            return self._evidence(
                ModelExecutionStatus.INFRASTRUCTURE_ERROR,
                f"Model execution stopped with {stop_reason}.",
                stop_reason,
                model_usage,
                all_models,
            )
        if stop_reason == "max_budget_usd_exceeded":
            return self._evidence(
                ModelExecutionStatus.MODEL_LIMIT_REACHED,
                "Model execution reached the hard cost ceiling "
                f"(total_cost_usd={getattr(terminal, 'total_cost_usd', None)}, "
                f"num_turns={getattr(terminal, 'num_turns', None)}); no further model calls were made.",
                stop_reason,
                model_usage,
                all_models,
            )
        if getattr(terminal, "is_error", True):
            return self._evidence(
                ModelExecutionStatus.MODEL_LIMIT_REACHED,
                "Claude SDK returned an error result.",
                stop_reason,
                model_usage,
                all_models,
            )
        return self._evidence(
            ModelExecutionStatus.SUCCEEDED,
            "Claude SDK completed the implementation workflow.",
            stop_reason,
            model_usage,
            all_models,
        )

    def _evidence(
        self,
        status: ModelExecutionStatus,
        explanation: str,
        stop_reason: str | None,
        model_usage: Mapping[str, Any] | None,
        observed_models: tuple[str, ...],
    ) -> ModelExecution:
        self._log("model_execution_finished", f"status={status}; {explanation}")
        return ModelExecution(
            status=status,
            explanation=explanation,
            stop_reason=stop_reason,
            model_usage=model_usage,
            observed_models=observed_models,
            skill_events=self.skill_events,
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
        return await publication_guard(hook_input, tool_use_id, context)

    async def _record_post_tool_use(
        self, hook_input: Any, tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        self._log_tool_use(
            "tool_result", hook_input, response=_hook_field(hook_input, "tool_response")
        )
        await self._record_skill_event("PostToolUse", hook_input, tool_use_id, context)
        if self._soft_threshold_crossed and not self._handoff_context_delivered:
            self._handoff_context_delivered = True
            self._log("cost_soft_threshold_handoff_context_injected")
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": _COST_HANDOFF_INSTRUCTION,
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
