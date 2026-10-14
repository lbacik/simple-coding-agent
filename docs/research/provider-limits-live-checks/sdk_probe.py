"""Live checks for "Run the live checks from the provider-limits research".
Run inside a throwaway container built from the agent image:

    docker run --rm -i --env-file .env --entrypoint python <agent-image> - [--steady] [--window | --window-only] < sdk_probe.py

Prints JSON lines; never prints credentials or
request headers. The model credential is read from the environment only to
build the CLI environment and the direct calls below."""

import asyncio
import json
import os
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, ResultMessage, SystemMessage

MODEL = os.environ.get("MODEL", "muse-spark-1.3-contributor")
UPSTREAM = (os.environ.get("MODEL_BASE_URL") or "https://api.meta.ai").rstrip("/")
KEY = os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY") or ""
PHASE = {"name": "init"}
REQUESTS = []
SKIP_HEADERS = {"set-cookie"}


def emit(kind, **data):
    print(json.dumps({"kind": kind, **data}, default=str), flush=True)


class Proxy(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _forward(self):
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        upstream = urlsplit(UPSTREAM)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")}
        req = urllib.request.Request(UPSTREAM + self.path, data=body or None, method=self.command, headers=headers)
        summary = {"phase": PHASE["name"], "method": self.command, "path": self.path}
        try:
            parsed = json.loads(body) if body else {}
            summary.update(
                model=parsed.get("model"),
                stream=parsed.get("stream"),
                max_tokens=parsed.get("max_tokens"),
                messages=len(parsed.get("messages") or []),
                tools=len(parsed.get("tools") or []),
            )
        except Exception:
            pass
        started = time.time()
        try:
            resp = urllib.request.urlopen(req, timeout=600)
            status, resp_headers, data = resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as error:
            status, resp_headers, data = error.code, error.headers, error.read()
        summary.update(
            status=status,
            ms=int((time.time() - started) * 1000),
            response_headers={k: v for k, v in resp_headers.items() if k.lower() not in SKIP_HEADERS},
        )
        text = data.decode("utf-8", "replace")
        usages = re.findall(r'"usage"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', text)
        if usages:
            summary["usage_last"] = usages[-1]
        if "count_tokens" in self.path or status >= 400:
            summary["body"] = text[:500]
        REQUESTS.append(summary)
        emit("http", **summary)
        self.send_response(status)
        for k, v in resp_headers.items():
            if k.lower() in ("transfer-encoding", "connection", "content-length"):
                continue
            self.send_header(k, v)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = do_PUT = do_DELETE = _forward


def start_proxy():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


def count(phase):
    return [r for r in REQUESTS if r["phase"] == phase]


def ctx_summary(usage):
    keep = ("totalTokens", "maxTokens", "rawMaxTokens", "percentage", "model", "isAutoCompactEnabled", "autoCompactThreshold")
    out = {k: usage.get(k) for k in keep}
    out["other_keys"] = sorted(set(usage) - set(keep) - {"categories", "memoryFiles"})
    out["categories"] = [{c.get("name"): c.get("tokens")} for c in usage.get("categories", [])]
    return out


async def drain(client, label):
    async for message in client.receive_response():
        if isinstance(message, SystemMessage):
            data = message.data if message.subtype != "init" else {k: message.data.get(k) for k in ("model", "claude_code_version", "apiKeySource")}
            emit("system", phase=label, subtype=message.subtype, data=data)
        elif isinstance(message, ResultMessage):
            emit(
                "result",
                phase=label,
                subtype=message.subtype,
                is_error=message.is_error,
                num_turns=message.num_turns,
                total_cost_usd=message.total_cost_usd,
                usage=message.usage,
                model_usage=message.model_usage,
                api_error_status=getattr(message, "api_error_status", None),
            )


async def scenario(name, proxy_url, extra_env, do_compact):
    env = {
        "ANTHROPIC_BASE_URL": proxy_url,
        "ANTHROPIC_AUTH_TOKEN": KEY,
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_MODEL": MODEL,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": MODEL,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": MODEL,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": MODEL,
        "CLAUDE_CODE_SUBAGENT_MODEL": MODEL,
        **extra_env,
    }
    options = ClaudeAgentOptions(
        model=MODEL,
        env=env,
        cwd=tempfile.mkdtemp(),
        permission_mode="bypassPermissions",
        include_partial_messages=False,
        setting_sources=[],
        allowed_tools=[],
        max_turns=1,
        system_prompt="You are a terse probe. Answer in one word.",
    )
    emit("scenario", name=name, extra_env=extra_env)
    async with ClaudeSDKClient(options=options) as client:
        PHASE["name"] = f"{name}:query"
        await client.query("Reply with the single word OK.")
        await drain(client, PHASE["name"])
        for i in (1, 2):
            PHASE["name"] = f"{name}:ctx{i}"
            t0 = time.time()
            usage = await client.get_context_usage()
            emit("context_usage", phase=PHASE["name"], ms=int((time.time() - t0) * 1000), requests=len(count(PHASE["name"])), **ctx_summary(usage))
        if do_compact:
            PHASE["name"] = f"{name}:compact"
            await client.query("/compact")
            await drain(client, PHASE["name"])
            PHASE["name"] = f"{name}:ctx_after_compact"
            usage = await client.get_context_usage()
            emit("context_usage", phase=PHASE["name"], requests=len(count(PHASE["name"])), **ctx_summary(usage))
        PHASE["name"] = f"{name}:idle"


async def steady(proxy_url):
    """Per-poll request cost of get_context_usage() once the session is warm."""
    env = {
        "ANTHROPIC_BASE_URL": proxy_url,
        "ANTHROPIC_AUTH_TOKEN": KEY,
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_MODEL": MODEL,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": MODEL,
        "CLAUDE_CODE_SUBAGENT_MODEL": MODEL,
    }
    options = ClaudeAgentOptions(
        model=MODEL, env=env, cwd=tempfile.mkdtemp(), permission_mode="bypassPermissions",
        setting_sources=[], max_turns=1, system_prompt="You are a terse probe. Answer in one word.",
    )
    async with ClaudeSDKClient(options=options) as client:
        for turn in range(1, 4):
            PHASE["name"] = f"steady:turn{turn}"
            await client.query(f"Turn {turn}: reply with the single word OK.")
            await drain(client, PHASE["name"])
            for poll in (1, 2):
                PHASE["name"] = f"steady:turn{turn}:poll{poll}"
                t0 = time.time()
                usage = await client.get_context_usage()
                emit("context_usage", phase=PHASE["name"], ms=int((time.time() - t0) * 1000),
                     requests=len(count(PHASE["name"])), totalTokens=usage.get("totalTokens"))


def direct(path, payload=None, method="POST"):
    headers = {"authorization": f"Bearer {KEY}", "anthropic-version": "2023-06-01", "content-type": "application/json"}
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(UPSTREAM + path, data=data, method=method, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=600)
        return resp.status, dict(resp.headers), resp.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read().decode()


def window_checks():
    PHASE["name"] = "window"
    status, headers, body = direct(f"/v1/models/{MODEL}", method="GET")
    emit("models_get", status=status, body=body[:1000])
    unit = "lorem ipsum dolor sit amet "
    status, _, body = direct("/v1/messages/count_tokens", {"model": MODEL, "messages": [{"role": "user", "content": unit * 10000}]})
    emit("calibrate", status=status, body=body[:300])
    per_unit = json.loads(body)["input_tokens"] / 10000
    for target, label in (
        (250_000, "above_cli_200k"),
        (1_040_000, "below_meta_1048576"),
        (1_055_000, "just_above_meta_1048576"),
        (1_100_000, "above_meta_1048576"),
    ):
        reps = int(target / per_unit)
        content = unit * reps
        status, _, body = direct("/v1/messages/count_tokens", {"model": MODEL, "messages": [{"role": "user", "content": content}]})
        emit("count_tokens", label=label, status=status, body=body[:300])
        t0 = time.time()
        status, headers, body = direct(
            "/v1/messages",
            {"model": MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": content + "\nReply OK."}]},
        )
        emit(
            "window_probe",
            label=label,
            status=status,
            ms=int((time.time() - t0) * 1000),
            ratelimit={k: v for k, v in headers.items() if "ratelimit" in k.lower() or "retry" in k.lower()},
            body=body[:600],
        )


async def main():
    emit("env", upstream_host=urlsplit(UPSTREAM).netloc, model=MODEL, key_present=bool(KEY))
    if "--window-only" in sys.argv:
        window_checks()
        return
    proxy_url = start_proxy()
    if "--steady" in sys.argv:
        await steady(proxy_url)
    else:
        await scenario("A_default", proxy_url, {}, do_compact=True)
        await scenario("B_max_ctx", proxy_url, {"CLAUDE_CODE_MAX_CONTEXT_TOKENS": "1048576"}, do_compact=False)
    if "--window" in sys.argv:
        window_checks()
    emit("done", total_requests=len(REQUESTS))


asyncio.run(main())
