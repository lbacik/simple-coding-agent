"""Unit and smoke tests for the conformance harness (operator contract §15.1).

The validation tests cover the harness envelope (one JSON reply, argv and
document verbs agreeing, strict parameters, ``unsupported`` naming what the
engine lacks) without starting an engine. The smoke test drives the real
lifecycle wired to the shared fakes through every harness verb and asserts
each protocol state is reached: starting, recovering, idle, active,
draining, paused, halted (``last_halt``) and stopped (Offline Read).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from conformance_harness import harness as harness_module
from conformance_harness.harness import HARNESS_VERSION, run_harness


def invoke(*argv: str, state_dir: Path) -> tuple[int, dict]:
    """Run one harness invocation against an isolated state directory."""

    old = os.environ.get("SCA_HARNESS_STATE_DIR")
    os.environ["SCA_HARNESS_STATE_DIR"] = str(state_dir)
    try:
        return run_harness(list(argv))
    finally:
        if old is None:
            os.environ.pop("SCA_HARNESS_STATE_DIR", None)
        else:
            os.environ["SCA_HARNESS_STATE_DIR"] = old


def request(verb: str, params: dict) -> str:
    return json.dumps({"verb": verb, "params": params})


# ---------------------------------------------------------------------------
# Envelope and request validation (no engine needed)
# ---------------------------------------------------------------------------


def test_no_verb_is_usage(tmp_path: Path) -> None:
    code, reply = invoke(state_dir=tmp_path)
    assert code == 2
    assert reply["ok"] is False
    assert reply["error"]["code"] == "usage"
    assert "verb" not in reply


def test_unknown_verb_is_usage(tmp_path: Path) -> None:
    code, reply = invoke("frobnicate", state_dir=tmp_path)
    assert code == 2
    assert reply["error"]["code"] == "usage"


def test_missing_document_is_usage(tmp_path: Path) -> None:
    code, reply = invoke("setup", state_dir=tmp_path)
    assert code == 2
    assert reply["verb"] == "setup"
    assert reply["error"]["code"] == "usage"


def test_malformed_json_is_usage(tmp_path: Path) -> None:
    code, reply = invoke("setup", "{not json", state_dir=tmp_path)
    assert code == 2
    assert reply["error"]["code"] == "usage"


def test_verb_mismatch_is_usage(tmp_path: Path) -> None:
    code, reply = invoke(
        "setup", request("teardown", {}), state_dir=tmp_path
    )
    assert code == 2
    assert reply["verb"] == "setup"
    assert reply["error"]["code"] == "usage"


def test_extra_top_level_key_is_usage(tmp_path: Path) -> None:
    document = json.dumps({"verb": "teardown", "params": {}, "extra": 1})
    code, reply = invoke("teardown", document, state_dir=tmp_path)
    assert code == 2
    assert reply["error"]["code"] == "usage"


def test_unknown_params_are_usage(tmp_path: Path) -> None:
    code, reply = invoke(
        "teardown", request("teardown", {"surprise": True}), state_dir=tmp_path
    )
    assert code == 2
    assert reply["error"]["code"] == "usage"


def test_set_issue_rejects_bad_issue(tmp_path: Path) -> None:
    code, reply = invoke(
        "set-issue", request("set-issue", {"issue": 0, "eligible": True}),
        state_dir=tmp_path,
    )
    assert code == 2
    assert reply["error"]["code"] == "usage"


def test_hold_lock_rejects_bad_duration(tmp_path: Path) -> None:
    for duration in (0, 60001, "long"):
        code, reply = invoke(
            "hold-lock", request("hold-lock", {"duration_ms": duration}),
            state_dir=tmp_path,
        )
        assert code == 2
        assert reply["error"]["code"] == "usage"


def test_hold_requires_phase_exactly_at_native_phase(tmp_path: Path) -> None:
    code, reply = invoke(
        "hold", request("hold", {"point": "native-phase"}), state_dir=tmp_path
    )
    assert code == 2
    code, reply = invoke(
        "hold",
        request("hold", {"point": "after-claim", "phase": "model_running"}),
        state_dir=tmp_path,
    )
    assert code == 2


def test_setup_rejects_bad_repository(tmp_path: Path) -> None:
    code, reply = invoke(
        "setup", request("setup", {"repository": "not-a-repo"}), state_dir=tmp_path
    )
    assert code == 2
    assert reply["error"]["code"] == "usage"


def test_setup_rejects_non_native_config(tmp_path: Path) -> None:
    code, reply = invoke(
        "setup",
        request("setup", {"repository": "octo/example", "config": {"poll": 1}}),
        state_dir=tmp_path,
    )
    assert code == 2
    assert reply["error"]["code"] == "usage"


def test_setup_unsupported_capability_names_what_and_name(tmp_path: Path) -> None:
    code, reply = invoke(
        "setup",
        request("setup", {"repository": "octo/example", "capabilities": ["nope"]}),
        state_dir=tmp_path,
    )
    assert code == 3
    assert reply["verb"] == "setup"
    assert reply["error"]["code"] == "unsupported"
    assert reply["error"]["details"] == {"what": "capability", "name": "nope"}


def test_finish_attempt_unsupported_outcome(tmp_path: Path) -> None:
    harness_module._save_state(
        tmp_path, {**harness_module._default_state(), "repository": "octo/example"}
    )
    code, reply = invoke(
        "finish-attempt",
        request("finish-attempt", {"native_outcome": "teleported"}),
        state_dir=tmp_path,
    )
    assert code == 3
    assert reply["verb"] == "finish-attempt"
    assert reply["error"]["code"] == "unsupported"
    assert reply["error"]["details"] == {
        "what": "native-outcome",
        "name": "teleported",
    }


def test_verbs_without_instance_are_failed(tmp_path: Path) -> None:
    for verb, params in [
        ("set-issue", {"issue": 1, "eligible": True}),
        ("hold", {"point": "before-claim"}),
        ("release", {}),
        ("crash", {"point": "now"}),
        ("halt", {"reason": "x"}),
        ("grow-archive", {"append_bytes": 8}),
    ]:
        code, reply = invoke(verb, request(verb, params), state_dir=tmp_path)
        assert code == 1, verb
        assert reply["error"]["code"] == "failed"


def test_teardown_without_instance_removes_nothing(tmp_path: Path) -> None:
    code, reply = invoke("teardown", request("teardown", {}), state_dir=tmp_path)
    assert code == 0
    assert reply == {
        "harness": HARNESS_VERSION,
        "verb": "teardown",
        "ok": True,
        "result": {"removed": False},
    }


# ---------------------------------------------------------------------------
# Smoke test: every harness verb, every protocol state
# ---------------------------------------------------------------------------


def _contract_status(socket: Path) -> dict:
    from simple_coding_agent.control_server import send_request

    reply = send_request(socket, {"op": "contract_status"}, timeout=10.0)
    assert reply.get("ok") is True
    return reply["contract_status"]


def _wait_for_state(socket: Path, wanted: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        last = _contract_status(socket)
        if last["instance"]["state"] == wanted:
            return last
        time.sleep(0.2)
    raise AssertionError(f"status never reached {wanted!r}; last={last!r}")


def _agentctl(
    socket: Path, *tokens: str, data_dir: Path | None = None
) -> tuple[int, str]:
    """Run the real contract CLI against the harness engine.

    Returns the exit code and the captured stdout (the contract envelope),
    so offline reads can assert on their content.
    """

    import io
    from contextlib import redirect_stdout

    from simple_coding_agent import agentctl

    old_socket = os.environ.get("AGENTCTL_SOCKET")
    old_data = os.environ.get("DATA_DIR")
    os.environ["AGENTCTL_SOCKET"] = str(socket)
    if data_dir is not None:
        os.environ["DATA_DIR"] = str(data_dir)
    try:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = agentctl.main(["--contract", "1", *tokens])
        return code, buffer.getvalue()
    finally:
        if old_socket is None:
            os.environ.pop("AGENTCTL_SOCKET", None)
        else:
            os.environ["AGENTCTL_SOCKET"] = old_socket
        if old_data is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = old_data


def _agentctl_json(
    socket: Path, *tokens: str, data_dir: Path | None = None
) -> tuple[int, dict]:
    """Run the contract CLI and parse its stdout envelope as JSON."""

    code, output = _agentctl(socket, *tokens, data_dir=data_dir)
    return code, json.loads(output)


def test_harness_smoke_reaches_every_state(tmp_path: Path) -> None:
    root = tmp_path / "harness"
    socket = root / "daemon.sock"

    def call(verb: str, params: dict) -> dict:
        code, reply = invoke(verb, request(verb, params), state_dir=root)
        assert code == 0, reply
        assert reply["verb"] == verb
        assert reply["ok"] is True
        return reply["result"]

    # -- setup reports the six discovery facts ---------------------------
    result = call(
        "setup",
        {
            "repository": "octo/example",
            "config": {"native.startup_hold_ms": 3000},
        },
    )
    assert result["repository"] == "octo/example"
    assert result["container"]
    assert result["image"]
    assert result["volumes"] and result["volumes"][0]["path"].startswith("/")
    assert result["exec_user"] and result["exec_user"] != "root"
    assert "stop-after" in result["capabilities"]

    # -- starting: startup checks still running ---------------------------
    snapshot = _contract_status(socket)
    assert snapshot["instance"]["state"] == "starting"
    assert snapshot["instance"]["startup_check"] == "provenance"

    # -- idle: intake open, no active attempt ------------------------------
    snapshot = _wait_for_state(socket, "idle")
    assert snapshot["intake"]["state"] == "open"
    assert snapshot["current_attempt"] is None

    # -- offline while live: a held volume rejects the read ----------------
    code, offline = _agentctl_json(
        socket, "--offline", "status", data_dir=root / "instance"
    )
    assert code != 0
    assert offline["ok"] is False
    assert offline["error"]["code"] == "rejected"
    assert offline["error"]["details"]["reason"] == "instance_running"

    # -- active: hold after the claim, then make an issue eligible ---------
    assert call("hold", {"point": "after-claim"})["attempt_id"] is None
    assert call("set-issue", {"issue": 7, "eligible": True}) == {
        "issue": 7,
        "eligible": True,
    }
    snapshot = _wait_for_state(socket, "active")
    attempt_id = snapshot["current_attempt"]["attempt_id"]
    assert snapshot["current_attempt"]["issue"]["number"] == 7

    # A hold scoped to the active attempt answers the held attempt id.
    held = call("hold", {"point": "after-outcome", "scope": "active"})
    assert held["attempt_id"] == attempt_id

    # -- draining: close intake while the attempt is held ------------------
    code, _ = _agentctl(
        socket, "stop", "--request-id", "req-smoke-stop-1",
        "--expect-repo", "octo/example",
    )
    assert code == 0
    snapshot = _wait_for_state(socket, "draining")
    assert snapshot["intake"]["state"] == "closed"

    # -- finish: the held attempt cannot finish on its own, so finish-attempt
    #    (which releases the hold itself) is race-free ---------------------
    finished = call("finish-attempt", {"native_outcome": "complete"})
    assert finished["attempt_id"] == attempt_id
    assert finished["native_outcome"] == "complete"

    # The release verb on a before-claim hold (no timing involved).
    assert call("hold", {"point": "before-claim"})["attempt_id"] is None
    assert call("release", {}) == {"released": True}

    # Withdraw the issue so the later idle is stable (no re-claim churn).
    assert call("set-issue", {"issue": 7, "eligible": False}) == {
        "issue": 7,
        "eligible": False,
    }

    # -- paused: intake closed, no active attempt ---------------------------
    snapshot = _wait_for_state(socket, "paused")
    assert snapshot["intake"]["state"] == "closed"
    assert snapshot["current_attempt"] is None

    # -- offline stopped: the stopped volume reads without the engine -------
    assert call("stop-container", {}) == {"running": False}
    code, offline = _agentctl_json(
        socket, "--offline", "status", data_dir=root / "instance"
    )
    assert code == 0, offline
    assert offline["ok"] is True
    stopped = offline["result"]
    assert stopped["instance"]["state"] == "paused"
    assert stopped["instance"]["last_halt"] is None
    assert stopped["intake"]["state"] == "closed"
    assert stopped["current_attempt"] is None

    # -- recovering: an unreadable checkpoint holds intake for recovery ----
    assert call("start-container", {}) == {"running": True}
    snapshot = _wait_for_state(socket, "paused")
    assert call("stop-container", {}) == {"running": False}
    (root / "instance" / "state" / "attempt.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    assert call("start-container", {}) == {"running": True}
    snapshot = _wait_for_state(socket, "recovering", timeout=60.0)

    # Repair the checkpoint: the hold clears and intake goes quiet.
    (root / "instance" / "state" / "attempt.json").unlink()
    snapshot = _wait_for_state(socket, "paused")

    # -- idle again after reopening intake ----------------------------------
    code, _ = _agentctl(
        socket, "resume", "--request-id", "req-smoke-resume-1",
        "--expect-repo", "octo/example",
    )
    assert code == 0
    _wait_for_state(socket, "idle")

    # -- halted: the engine records its halt and exits by decision ----------
    assert call("halt", {"reason": "harness-smoke"}) == {"reason": "harness-smoke"}
    code, offline = _agentctl_json(
        socket, "--offline", "status", data_dir=root / "instance"
    )
    assert code == 0, offline
    assert offline["ok"] is True
    halted = offline["result"]
    assert halted["instance"]["state"] == "paused"
    assert halted["instance"]["last_halt"]["reason"] == "harness-smoke"
    assert halted["current_attempt"] is None

    # -- teardown removes the instance --------------------------------------
    code, reply = invoke("teardown", request("teardown", {}), state_dir=root)
    assert code == 0
    assert reply["result"] == {"removed": True}
