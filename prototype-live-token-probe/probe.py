"""PROTOTYPE, disposable. Live check for ticket #129 (map #123, origin #119).

Runs one short, capped query against the Meta backend
(`muse-spark-1.3-contributor`) with claude-agent-sdk 0.2.156 and records the
facts that docs/research/live-token-sources.md left open:

1. raw StreamEvent `message_start` / `message_delta` usage, and whether
   `input_tokens` includes cached tokens;
2. `get_context_usage()["apiUsage"]` mid-run;
3. whether the main-transcript entry for a tool call is on disk, with merged
   usage, when its PostToolUse hook fires;
4. whether summed `message_delta` usage reconciles with `ResultMessage.usage`
   and, with subagent `task_notification` totals, with `model_usage`.

Credentials come from ../prototype-meta-skills-probe/.env (or PROBE_ENV_FILE).
The token is never printed or written to the output.

Usage: uv run python prototype-live-token-probe/probe.py
Output: prototype-live-token-probe/$PROBE_OUT (default out)/{events.jsonl,summary.json}
"""

import asyncio
import dataclasses
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    ToolUseBlock,
    UserMessage,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / os.environ.get("PROBE_OUT", "out")
ENV_FILE = Path(
    os.environ.get("PROBE_ENV_FILE", HERE.parent / "prototype-meta-skills-probe" / ".env")
)
MODEL = "muse-spark-1.3-contributor"

PROMPT = """This is a short measurement run. Do exactly these steps, nothing else:
1. Use the Read tool to read notes.txt.
2. Use the Bash tool to run: wc -l data.txt
3. Use the Agent tool to delegate to the `line-counter` agent: "Read data.txt and
   tell me how many lines start with the letter a."
4. Reply with one short sentence combining the three results, then stop.
"""

T0 = time.monotonic()
events: list[dict[str, Any]] = []


def log(kind: str, **data: Any) -> None:
    events.append({"t": round(time.monotonic() - T0, 3), "kind": kind, **data})


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    if not env.get("ANTHROPIC_AUTH_TOKEN") or "REPLACE_WITH" in env["ANTHROPIC_AUTH_TOKEN"]:
        raise SystemExit(f"No real ANTHROPIC_AUTH_TOKEN in {path}")
    return env


def make_fixture(root: Path) -> Path:
    work = root / "work"
    work.mkdir()
    (work / "notes.txt").write_text("The probe fixture has two files.\n")
    (work / "data.txt").write_text("apple\nbanana\navocado\ncherry\napricot\n")
    return work


def transcript_entry_for(path: str, tool_use_id: str) -> dict[str, Any]:
    """Find the main-transcript assistant entry that holds this tool_use."""

    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return {"found": False, "reason": "transcript missing"}
    assistant_entries = 0
    for raw in lines:
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "assistant":
            continue
        assistant_entries += 1
        message = entry.get("message", {})
        for block in message.get("content") or []:
            if block.get("type") == "tool_use" and block.get("id") == tool_use_id:
                return {
                    "found": True,
                    "message_id": message.get("id"),
                    "stop_reason": message.get("stop_reason"),
                    "usage": message.get("usage"),
                    "assistant_entries_on_disk": assistant_entries,
                }
    return {"found": False, "assistant_entries_on_disk": assistant_entries}


async def main() -> int:
    env = load_env(ENV_FILE)
    OUT.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="live-token-probe-"))
    home = root / "home"
    home.mkdir()
    work = make_fixture(root)

    client_ref: dict[str, ClaudeSDKClient] = {}

    async def pre_tool_use(input_data: dict, tool_use_id: str | None, context: Any) -> dict:
        log("pre_tool_use", tool_name=input_data.get("tool_name"), tool_use_id=tool_use_id,
            agent_id=input_data.get("agent_id"))
        return {}

    async def post_tool_use(input_data: dict, tool_use_id: str | None, context: Any) -> dict:
        record: dict[str, Any] = {
            "tool_name": input_data.get("tool_name"),
            "tool_use_id": tool_use_id,
            "agent_id": input_data.get("agent_id"),
        }
        if input_data.get("agent_id") is None and tool_use_id:
            record["transcript"] = transcript_entry_for(
                input_data["transcript_path"], tool_use_id
            )
        try:
            usage = await asyncio.wait_for(client_ref["c"].get_context_usage(), 10)
            record["context_usage"] = {
                "apiUsage": usage.get("apiUsage"),
                "totalTokens": usage.get("totalTokens"),
            }
        except Exception as exc:  # noqa: BLE001 - probe records every failure
            record["context_usage_error"] = repr(exc)
        log("post_tool_use", **record)
        return {}

    options = ClaudeAgentOptions(
        model=MODEL,
        cwd=str(work),
        env={
            **env,
            "HOME": str(home),
            "CLAUDE_STREAM_IDLE_TIMEOUT_MS": "60000",
        },
        setting_sources=[],
        permission_mode="bypassPermissions",
        include_partial_messages=True,
        max_turns=10,
        max_budget_usd=1.0,
        agents={
            "line-counter": AgentDefinition(
                description="Counts lines in a file that match a simple rule.",
                prompt="Read the requested file with the Read tool and answer in one sentence.",
                tools=["Read"],
                model="inherit",
            )
        },
        hooks={
            "PreToolUse": [HookMatcher(hooks=[pre_tool_use])],
            "PostToolUse": [HookMatcher(hooks=[post_tool_use])],
        },
    )

    result: ResultMessage | None = None
    async with ClaudeSDKClient(options=options) as client:
        client_ref["c"] = client
        await client.query(PROMPT)
        async for message in client.receive_response():
            if isinstance(message, StreamEvent):
                event = message.event
                etype = event.get("type")
                if etype == "message_start":
                    msg = event.get("message", {})
                    log("message_start", id=msg.get("id"), usage=msg.get("usage"),
                        parent=message.parent_tool_use_id)
                elif etype == "message_delta":
                    log("message_delta", usage=event.get("usage"),
                        stop_reason=event.get("delta", {}).get("stop_reason"),
                        parent=message.parent_tool_use_id)
            elif isinstance(message, AssistantMessage):
                log("assistant", id=message.message_id, stop_reason=message.stop_reason,
                    usage=message.usage, parent=message.parent_tool_use_id,
                    tool_uses=[b.name for b in message.content if isinstance(b, ToolUseBlock)])
            elif isinstance(message, UserMessage):
                log("user", parent=message.parent_tool_use_id)
            elif isinstance(message, ResultMessage):
                result = message
                log("result", **dataclasses.asdict(message))
            elif isinstance(message, SystemMessage):
                if message.subtype in ("task_progress", "task_notification", "task_started"):
                    log(message.subtype, usage=message.data.get("usage"),
                        task_id=message.data.get("task_id"))
                else:
                    log("system", subtype=message.subtype)

    with (OUT / "events.jsonl").open("w") as fh:
        for event in events:
            fh.write(json.dumps(event, default=str) + "\n")

    summary = summarize(result)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))

    # Keep the main transcript for inspection; it holds no credentials.
    for transcript in home.glob(".claude/projects/**/*.jsonl"):
        dest = OUT / "transcripts" / transcript.relative_to(home / ".claude" / "projects")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(transcript, dest)
    return 0 if result is not None and not result.is_error else 1


def summarize(result: ResultMessage | None) -> dict[str, Any]:
    keys = ["input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"]
    starts = [e for e in events if e["kind"] == "message_start"]
    deltas = [e for e in events if e["kind"] == "message_delta"]
    final_deltas = [e for e in deltas if e.get("stop_reason")]
    delta_sum = {k: sum((e["usage"] or {}).get(k) or 0 for e in final_deltas) for k in keys}
    notifications = [e for e in events if e["kind"] == "task_notification"]
    subagent_total = sum((e.get("usage") or {}).get("total_tokens") or 0 for e in notifications)
    model_usage = (result.model_usage or {}) if result else {}
    mu = next(iter(model_usage.values()), {}) if model_usage else {}
    return {
        "responses": {"message_start": len(starts), "message_delta": len(deltas),
                      "message_delta_with_stop_reason": len(final_deltas)},
        "message_start_usage": [e["usage"] for e in starts],
        "message_delta_usage": [e["usage"] for e in deltas],
        "stream_events_with_parent": sum(1 for e in starts + deltas if e.get("parent")),
        "delta_sum_main_loop": delta_sum,
        "result_usage": {k: (result.usage or {}).get(k) for k in keys} if result else None,
        "result_model_usage": mu,
        "result_total_cost_usd": result.total_cost_usd if result else None,
        "subagent_task_notification_total_tokens": subagent_total,
        "post_tool_use": [e for e in events if e["kind"] == "post_tool_use"],
        "task_progress_usage": [e.get("usage") for e in events if e["kind"] == "task_progress"],
    }


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
