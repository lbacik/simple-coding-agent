"""Claude SDK boundary for one bounded implementation attempt."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
import shlex
from typing import Any, Protocol

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher, ResultMessage

from simple_coding_agent.config import RuntimeConfig


_SKILLS = ["implement", "tdd", "code-review", "codebase-design"]
_META_BASE_URL = "https://api.meta.ai"


class ExecutionOutcome(StrEnum):
    """Classification supplied to the completion evaluator."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


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

    outcome: ExecutionOutcome
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

    def receive_response(self) -> Any: ...


ClientFactory = Callable[[ClaudeAgentOptions], SDKClient]
Clock = Callable[[], datetime]


class ModelExecutor:
    """Run the approved upstream skills and retain terminal stream evidence."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        client_factory: ClientFactory = ClaudeSDKClient,
        clock: Clock = lambda: datetime.now(UTC),
    ) -> None:
        self._config = config
        self._client_factory = client_factory
        self._clock = clock
        self._skill_events: list[SkillEvent] = []
        self._review_count = 0

    @property
    def skill_events(self) -> tuple[SkillEvent, ...]:
        """Return the skill provenance recorded during this execution."""

        return tuple(self._skill_events)

    async def execute(self, *, issue_body: str, working_directory: Path) -> ModelExecution:
        """Dispatch ``/implement`` and classify its one terminal SDK result."""

        self._skill_events.clear()
        self._review_count = 0
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
                    return self._evidence(
                        ExecutionOutcome.INFRASTRUCTURE_ERROR,
                        "Model execution exceeded MODEL_TIMEOUT and was interrupted.",
                        None,
                        None,
                        (),
                    )
        except Exception as error:
            return self._evidence(
                ExecutionOutcome.INFRASTRUCTURE_ERROR,
                f"Claude SDK execution failed: {type(error).__name__}.",
                None,
                None,
                (),
            )

        if terminal is None:
            return self._evidence(
                ExecutionOutcome.INFRASTRUCTURE_ERROR,
                "Claude SDK stream ended without a terminal ResultMessage.",
                None,
                None,
                observed_models,
            )
        return self._classify(terminal, observed_models)

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
                "PostToolUse": [HookMatcher(matcher="Skill", hooks=[self._record_post_tool_use])],
            },
        )

    async def _receive_terminal(
        self, client: SDKClient
    ) -> tuple[ResultMessage | None, tuple[str, ...]]:
        observed_models: list[str] = []
        async for message in client.receive_response():
            if isinstance(message, ResultMessage) or _looks_like_result(message):
                return message, tuple(observed_models)
            model = getattr(message, "model", None)
            if isinstance(model, str):
                observed_models.append(model)
        return None, tuple(observed_models)

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
        if mismatches:
            return self._evidence(
                ExecutionOutcome.INFRASTRUCTURE_ERROR,
                f"Observed model mismatch: {', '.join(mismatches)}.",
                stop_reason,
                model_usage,
                all_models,
            )
        if stop_reason == "max_turns_exceeded":
            return self._evidence(
                ExecutionOutcome.INCOMPLETE,
                "Model execution reached max_turns.",
                stop_reason,
                model_usage,
                all_models,
            )
        if stop_reason == "timeout" or (isinstance(stop_reason, str) and stop_reason.startswith("aborted_")):
            return self._evidence(
                ExecutionOutcome.INFRASTRUCTURE_ERROR,
                f"Model execution stopped with {stop_reason}.",
                stop_reason,
                model_usage,
                all_models,
            )
        if getattr(terminal, "is_error", True):
            return self._evidence(
                ExecutionOutcome.INCOMPLETE,
                "Claude SDK returned an error result.",
                stop_reason,
                model_usage,
                all_models,
            )
        return self._evidence(
            ExecutionOutcome.COMPLETE,
            "Claude SDK completed the implementation workflow.",
            stop_reason,
            model_usage,
            all_models,
        )

    def _evidence(
        self,
        outcome: ExecutionOutcome,
        explanation: str,
        stop_reason: str | None,
        model_usage: Mapping[str, Any] | None,
        observed_models: tuple[str, ...],
    ) -> ModelExecution:
        return ModelExecution(
            outcome=outcome,
            explanation=explanation,
            stop_reason=stop_reason,
            model_usage=model_usage,
            observed_models=observed_models,
            skill_events=self.skill_events,
        )

    async def _guard_and_record_pre_tool_use(
        self, hook_input: Any, tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        await self._record_skill_event("PreToolUse", hook_input, tool_use_id, context)
        if _skill_name(hook_input) == "code-review":
            self._review_count += 1
            if self._review_count > 3:
                return _deny("At most two code-review repair cycles are allowed.")
        return await publication_guard(hook_input, tool_use_id, context)

    async def _record_post_tool_use(
        self, hook_input: Any, tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        return await self._record_skill_event("PostToolUse", hook_input, tool_use_id, context)

    async def _record_skill_event(
        self, phase: str, hook_input: Any, tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        if getattr(hook_input, "tool_name", None) != "Skill":
            return {}
        name = _skill_name(hook_input)
        if not isinstance(name, str):
            return {}
        self._skill_events.append(
            SkillEvent(
                phase=phase,
                name=name,
                agent_id=getattr(hook_input, "agent_id", None),
                timestamp=_timestamp(self._clock()),
            )
        )
        return {}


async def publication_guard(hook_input: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
    """Deny model-side publication while allowing the driving process to publish."""

    if getattr(hook_input, "tool_name", None) != "Bash":
        return {}
    tool_input = getattr(hook_input, "tool_input", {})
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
    tool_input = getattr(hook_input, "tool_input", {})
    name = tool_input.get("skill") if isinstance(tool_input, Mapping) else None
    return name if isinstance(name, str) else None


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
