"""Check 3 of "Run the live checks from the provider-limits research": raw
response headers from the LiteLLM gateway. Run with the env file of an instance
that points MODEL_BASE_URL at the gateway:

    docker run --rm -i --network <gateway-network> --env-file .env --entrypoint python <agent-image> - < gateway_probe.py

Prints response headers and selected non-secret fields only."""

import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit

MODEL = os.environ.get("MODEL", "muse-spark-1.3-contributor")
BASE = (os.environ.get("MODEL_BASE_URL") or "").rstrip("/")
KEY = os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY") or ""
INTERESTING = ("max", "token", "rpm", "tpm", "window", "context", "limit", "budget", "spend")


def call(path, payload=None, method="POST"):
    headers = {"authorization": f"Bearer {KEY}", "anthropic-version": "2023-06-01", "content-type": "application/json"}
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        return resp.status, dict(resp.headers), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read().decode("utf-8", "replace")


def pick(obj, prefix=""):
    """Non-secret numeric/boolean fields whose names hint at limits."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}{k}"
            if isinstance(v, (dict, list)):
                out.update(pick(v, name + "."))
            elif not isinstance(v, str) and any(t in k.lower() for t in INTERESTING) and "key" not in k.lower():
                out[name] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(pick(v, f"{prefix}{i}."))
    return out


def emit(kind, **data):
    print(json.dumps({"kind": kind, **data}), flush=True)


emit("env", base_host=urlsplit(BASE).netloc, model=MODEL, key_present=bool(KEY))
msg = {"model": MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": "Reply OK."}]}
for label, path, payload in (
    ("messages_nonstream", "/v1/messages", msg),
    ("messages_stream", "/v1/messages?beta=true", {**msg, "stream": True}),
    ("count_tokens", "/v1/messages/count_tokens?beta=true", {"model": MODEL, "messages": msg["messages"]}),
):
    status, headers, body = call(path, payload)
    headers = {k: v for k, v in headers.items() if k.lower() != "set-cookie"}
    emit("headers", label=label, status=status, headers=headers, body=body[:300] if status >= 400 or label == "count_tokens" else None)
for path in ("/model/info", f"/v1/model/info?litellm_model_id={MODEL}", "/key/info"):
    status, _, body = call(path, method="GET")
    try:
        parsed = json.loads(body)
        fields = pick(parsed)
        emit("endpoint", path=path, status=status, top_keys=sorted(parsed) if isinstance(parsed, dict) else None, limit_fields=dict(list(fields.items())[:60]))
    except Exception:
        emit("endpoint", path=path, status=status, body=body[:200])
