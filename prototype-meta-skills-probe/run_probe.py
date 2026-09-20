"""PROTOTYPE, disposable. Runs the reduced-core acceptance probe for ticket #8:
https://github.com/lbacik/simple-coding-agent/issues/8

Sends one `/implement` prompt through claude-agent-sdk against the
muse-spark-1.3 configuration, against the fixture at /workspace/fixture, with
only the pinned mattpocock/skills bundle installed by agent-installer.
Records every observed model id, every Skill tool_use, and the terminal
ResultMessage, then independently reruns the test outside the agent loop.
Does not trust the agent's own success claim.
"""

import asyncio
import dataclasses
import json
import os
import subprocess
import sys

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    ToolUseBlock,
    query,
)

FIXTURE = "/workspace/fixture"
REPORT_PATH = "/workspace/probe-report.json"

TASK_PROMPT = """/implement

Task: fix calc.py so add(a, b) returns the sum of a and b, not the
difference. tests/test_calc.py::test_add_returns_sum currently fails.
This is a disposable fixture: skip seam-confirmation questions, use one
obvious seam, and proceed with the smallest possible fix. Run the review
step before committing.
"""


def redacted_env_summary() -> dict:
    keys = [
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
    ]
    summary = {k: os.environ.get(k) for k in keys}
    summary["ANTHROPIC_AUTH_TOKEN_present"] = bool(os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    return summary


async def main() -> int:
    options = ClaudeAgentOptions(
        model=os.environ.get("ANTHROPIC_MODEL", "muse-spark-1.3"),
        cwd=FIXTURE,
        setting_sources=["user", "project"],
        skills=["implement", "tdd", "code-review", "codebase-design"],
        tools=None,  # keep the default toolset, which retains Skill
        permission_mode="bypassPermissions",
        include_partial_messages=False,
        max_turns=40,
    )

    observed_models: set[str] = set()
    skill_invocations: list[dict] = []
    system_events: list[dict] = []
    result: ResultMessage | None = None
    transcript: list[str] = []

    async for message in query(prompt=TASK_PROMPT, options=options):
        transcript.append(f"{type(message).__name__}: {message!r}"[:2000])

        if isinstance(message, SystemMessage):
            system_events.append({"subtype": message.subtype, "data": message.data})

        elif isinstance(message, AssistantMessage):
            observed_models.add(message.model)
            for block in message.content:
                if isinstance(block, ToolUseBlock) and block.name == "Skill":
                    skill_invocations.append(
                        {"model": message.model, "input": block.input}
                    )

        elif isinstance(message, ResultMessage):
            result = message

    with open("/workspace/probe-transcript.log", "w") as fh:
        fh.write("\n".join(transcript))

    independent_test = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=FIXTURE,
        capture_output=True,
        text=True,
    )

    fixture_diff = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=FIXTURE,
        capture_output=True,
        text=True,
    ).stdout

    fixture_log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=FIXTURE,
        capture_output=True,
        text=True,
    ).stdout

    report = {
        "env_summary": redacted_env_summary(),
        "requested_model": options.model,
        "observed_models": sorted(observed_models),
        "routing_matches_requested": observed_models == {options.model} if observed_models else None,
        "skill_invocations": skill_invocations,
        "system_events": system_events,
        "result": dataclasses.asdict(result) if result is not None else None,
        "independent_test": {
            "returncode": independent_test.returncode,
            "stdout": independent_test.stdout,
            "stderr": independent_test.stderr,
        },
        "fixture_git_log": fixture_log,
        "fixture_diff": fixture_diff,
    }

    with open(REPORT_PATH, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))

    passed = (
        result is not None
        and not result.is_error
        and independent_test.returncode == 0
        and bool(fixture_diff.strip())
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
