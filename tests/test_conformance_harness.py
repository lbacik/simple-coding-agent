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
import threading
import time
from pathlib import Path

import pytest

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


def test_the_harness_speaks_the_pinned_suite_protocol() -> None:
    from agent_operator_contract.documents import HARNESS_VERBS
    from agent_operator_contract.documents import HARNESS_VERSION as SUITE_VERSION

    assert HARNESS_VERSION == SUITE_VERSION
    assert set(harness_module.VERBS) == set(HARNESS_VERBS)


def test_fail_finalization_takes_no_parameters(tmp_path: Path) -> None:
    code, reply = invoke(
        "fail-finalization",
        request("fail-finalization", {"boundary": "publication"}),
        state_dir=tmp_path,
    )
    assert code == 2
    assert reply["error"]["code"] == "usage"


def test_fail_finalization_without_instance_is_failed(tmp_path: Path) -> None:
    code, reply = invoke(
        "fail-finalization", request("fail-finalization", {}), state_dir=tmp_path
    )
    assert code == 1
    assert reply["verb"] == "fail-finalization"
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
# Engine liveness and the docker container (issue #375)
# ---------------------------------------------------------------------------


def _fake_proc_stat(proc_root: Path, pid: int, state: str) -> None:
    stat = proc_root / str(pid) / "stat"
    stat.parent.mkdir(parents=True)
    # A command name may itself hold spaces and parentheses.
    stat.write_text(f"{pid} (py (x) y) {state} 1 1 1 0 -1 4194560\n", encoding="utf-8")


def test_pid_gone_treats_a_zombie_that_is_not_our_child_as_gone(
    tmp_path: Path, monkeypatch
) -> None:
    # The parent process is alive and signalable but is not our child, like
    # an engine spawned by an earlier harness CLI under ``docker exec``.
    pid = os.getppid()
    _fake_proc_stat(tmp_path, pid, "Z")
    monkeypatch.setattr(harness_module, "_PROC_ROOT", tmp_path)

    assert harness_module._pid_gone(pid) is True


def test_pid_gone_keeps_a_running_process_that_is_not_our_child_alive(
    tmp_path: Path, monkeypatch
) -> None:
    pid = os.getppid()
    _fake_proc_stat(tmp_path, pid, "S")
    monkeypatch.setattr(harness_module, "_PROC_ROOT", tmp_path)

    assert harness_module._pid_gone(pid) is False


def test_pid_gone_without_proc_falls_back_to_the_signal_probe(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(harness_module, "_PROC_ROOT", tmp_path / "missing")

    assert harness_module._pid_gone(os.getppid()) is False


def test_docker_setup_runs_the_container_with_an_init(
    tmp_path: Path, monkeypatch
) -> None:
    """A real init reaps the orphaned engine once its ``docker exec`` exits."""

    import subprocess

    calls: list[list[str]] = []

    def fake_run_docker(args: list[str], timeout: float = 60.0):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setenv("SCA_HARNESS_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(harness_module, "_run_docker", fake_run_docker)
    monkeypatch.setattr(
        harness_module, "_docker_inner", lambda verb, params, container: (1, None)
    )

    harness_module._docker_setup({})

    run = next(args for args in calls if args[0] == "run")
    assert "--init" in run
    assert run.index("--init") < run.index("--entrypoint")


def test_start_container_after_a_crash_waits_for_the_new_engine(
    tmp_path: Path, monkeypatch
) -> None:
    """A crashed engine leaves its markers; the restart must not trust them."""

    harness_module._save_state(
        tmp_path,
        {"repository": "octo/example", "engine_pid": None, "container_running": True},
    )
    # The markers the crashed engine wrote before ``os._exit``.
    (tmp_path / "daemon.ready").write_text("12345", encoding="utf-8")
    (tmp_path / "daemon.serving").write_text("12345", encoding="utf-8")
    # The new engine never serves.
    monkeypatch.setattr(harness_module, "_spawn_daemon", lambda root, repository: 0)
    monkeypatch.setattr(harness_module, "_DAEMON_READY_TIMEOUT_SECONDS", 0.2)

    code, reply = invoke(
        "start-container", request("start-container", {}), state_dir=tmp_path
    )

    assert code == 1
    assert "did not serve the control endpoint" in reply["error"]["message"]


def test_spawned_engine_has_a_commit_identity(tmp_path: Path, monkeypatch) -> None:
    """The harness container has no git identity; sealing attempt work commits."""

    seen: dict[str, str] = {}

    class _Process:
        pid = 4242

    def fake_popen(argv, **kwargs):
        seen.update(kwargs["env"])
        return _Process()

    for name in (
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Operator")
    monkeypatch.setattr(harness_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(harness_module, "_SPAWNED", {})

    assert harness_module._spawn_daemon(tmp_path, "octo/example") == 4242

    # An identity already in the environment wins; the rest is filled in.
    assert seen["GIT_AUTHOR_NAME"] == "Operator"
    assert seen["GIT_AUTHOR_EMAIL"]
    assert seen["GIT_COMMITTER_NAME"]
    assert seen["GIT_COMMITTER_EMAIL"]


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

    # -- active: make an issue eligible, then let the claim through ---------
    #    (no claim opens on its own; the attempt then waits for
    #    finish-attempt, so it stays active without any hold) --------------
    assert call("set-issue", {"issue": 7, "eligible": True}) == {
        "issue": 7,
        "eligible": True,
    }
    assert call("release", {}) == {"released": False}
    snapshot = _contract_status(socket)
    assert snapshot["instance"]["state"] == "active"
    attempt_id = snapshot["current_attempt"]["attempt_id"]
    assert snapshot["current_attempt"]["issue"]["number"] == 7

    # A hold scoped to the active attempt answers the held attempt id.
    held = call("hold", {"point": "after-claim", "scope": "active"})
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
    #    (which releases a hold that blocks the executor) is race-free -----
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
    assert stopped["instance"]["state"] == "stopped"
    assert stopped["instance"]["last_halt"] is None
    assert stopped["intake"]["state"] == "closed"
    assert stopped["current_attempt"] is None

    # -- a restarted instance stays starting until release ----------------
    assert call("start-container", {}) == {"running": True}
    snapshot = _contract_status(socket)
    assert snapshot["instance"]["state"] == "starting"
    assert call("release", {}) == {"released": True}
    snapshot = _wait_for_state(socket, "paused")

    # -- recovering: an unreadable checkpoint holds intake for recovery ----
    assert call("stop-container", {}) == {"running": False}
    (root / "instance" / "state" / "attempt.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    assert call("start-container", {}) == {"running": True}
    assert call("release", {}) == {"released": True}
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

    # -- halt from paused: close intake so the stopped volume reads stopped --
    code, _ = _agentctl(
        socket, "stop", "--request-id", "req-smoke-stop-2",
        "--expect-repo", "octo/example",
    )
    assert code == 0
    _wait_for_state(socket, "paused")

    # -- halted: the engine records its halt and exits by decision ----------
    assert call("halt", {"reason": "harness-smoke"}) == {"reason": "harness-smoke"}
    code, offline = _agentctl_json(
        socket, "--offline", "status", data_dir=root / "instance"
    )
    assert code == 0, offline
    assert offline["ok"] is True
    halted = offline["result"]
    assert halted["instance"]["state"] == "stopped"
    assert halted["instance"]["last_halt"]["reason"] == "harness-smoke"
    assert halted["current_attempt"] is None

    # -- teardown removes the instance --------------------------------------
    code, reply = invoke("teardown", request("teardown", {}), state_dir=root)
    assert code == 0
    assert reply["result"] == {"removed": True}


# ---------------------------------------------------------------------------
# Harness timing semantics the producer suite relies on (issue #376)
# ---------------------------------------------------------------------------

#: Many daemon poll intervals (0.2 s each), and longer than one whole claim
#: including the tracker's assignment settle window, so a spontaneous claim
#: would have produced an attempt checkpoint by the end of it.
_NO_SPONTANEOUS_CLAIM_WINDOW_SECONDS = 3.5


class _Engine:
    """One running fakes-wired engine behind the process-local harness."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.socket = root / "daemon.sock"

    def call(self, verb: str, params: dict) -> dict:
        code, reply = invoke(verb, request(verb, params), state_dir=self.root)
        assert code == 0, reply
        assert reply["ok"] is True
        return reply["result"]

    def status(self) -> dict:
        return _contract_status(self.socket)

    def agentctl(self, *tokens: str) -> tuple[int, dict]:
        return _agentctl_json(self.socket, *tokens)


@pytest.fixture
def engine(tmp_path: Path):
    harness = _Engine(tmp_path / "harness")
    harness.call(
        "setup",
        {"repository": "octo/example", "config": {"native.startup_hold_ms": 100}},
    )
    try:
        _wait_for_state(harness.socket, "idle")
        yield harness
    finally:
        invoke("teardown", request("teardown", {}), state_dir=harness.root)


def test_an_eligible_issue_is_claimed_only_through_release(engine: _Engine) -> None:
    engine.call("set-issue", {"issue": 11, "eligible": True})
    time.sleep(_NO_SPONTANEOUS_CLAIM_WINDOW_SECONDS)
    snapshot = engine.status()
    assert snapshot["current_attempt"] is None
    assert snapshot["last_finished"] is None  # no attempt ran meanwhile either
    assert not (engine.root / "instance" / "state" / "attempt.json").exists()

    assert engine.call("release", {}) == {"released": False}

    # No polling: the claim is visible as soon as release answers.
    current = engine.status()["current_attempt"]
    assert current is not None
    assert current["issue"]["number"] == 11


def test_stop_behind_a_before_claim_hold_completes_at_once(engine: _Engine) -> None:
    engine.call("set-issue", {"issue": 12, "eligible": True})
    engine.call("hold", {"point": "before-claim", "scope": "next"})

    code, reply = engine.agentctl(
        "stop", "--request-id", "req-376-stop", "--expect-repo", "octo/example"
    )

    assert code == 0, reply
    assert reply["result"]["command"]["acknowledgement"] == "completed"


def test_a_claimed_attempt_waits_for_finish_attempt(engine: _Engine) -> None:
    engine.call("set-issue", {"issue": 13, "eligible": True})
    engine.call("release", {})
    attempt_id = engine.status()["current_attempt"]["attempt_id"]

    time.sleep(_NO_SPONTANEOUS_CLAIM_WINDOW_SECONDS)
    snapshot = engine.status()
    assert snapshot["current_attempt"] is not None
    assert snapshot["current_attempt"]["attempt_id"] == attempt_id
    assert not (engine.root / "flags" / "scripted_outcome.json").exists()

    finished = engine.call(
        "finish-attempt", {"native_outcome": "infrastructure_error", "reason": "boom"}
    )

    assert finished == {"attempt_id": attempt_id, "native_outcome": "infrastructure_error"}
    snapshot = engine.status()
    assert snapshot["current_attempt"] is None
    assert snapshot["last_finished"]["attempt_id"] == attempt_id
    assert snapshot["last_finished"]["outcome"]["native"] == "infrastructure_error"
    # The outcome was consumed by that attempt, never left for the next one.
    assert not (engine.root / "flags" / "scripted_outcome.json").exists()


def test_status_during_hold_lock_serves_the_prepared_snapshot(engine: _Engine) -> None:
    before = engine.status()
    held: list[dict] = []
    holder = threading.Thread(
        target=lambda: held.append(engine.call("hold-lock", {"duration_ms": 4000}))
    )
    holder.start()
    try:
        time.sleep(0.5)
        started = time.monotonic()
        during = engine.status()
        elapsed = time.monotonic() - started
        still_held = holder.is_alive()
    finally:
        holder.join(timeout=30.0)

    assert still_held
    assert elapsed < 3.0
    assert held and held[0]["held_ms"] >= 4000
    for section in ("instance", "intake"):
        assert during[section]["observed_at"] <= before[section]["observed_at"]


def test_release_returns_promptly_with_nothing_eligible(engine: _Engine) -> None:
    started = time.monotonic()
    assert engine.call("release", {}) == {"released": False}
    assert time.monotonic() - started < 3.0
    assert engine.status()["current_attempt"] is None


def test_release_returns_promptly_with_intake_closed(engine: _Engine) -> None:
    code, _ = engine.agentctl(
        "stop", "--request-id", "req-376-close", "--expect-repo", "octo/example"
    )
    assert code == 0
    engine.call("set-issue", {"issue": 14, "eligible": True})

    started = time.monotonic()
    assert engine.call("release", {}) == {"released": False}
    assert time.monotonic() - started < 3.0
    assert engine.status()["current_attempt"] is None


def test_a_restart_holds_starting_and_release_honours_the_priority(
    engine: _Engine,
) -> None:
    engine.call("hold", {"point": "before-claim", "scope": "next"})
    engine.call("set-issue", {"issue": 21, "eligible": True})
    engine.call("set-issue", {"issue": 22, "eligible": True})
    code, reply = engine.agentctl(
        "next", "issue", "22", "--request-id", "req-376-next",
        "--expect-repo", "octo/example",
    )
    assert code == 0, reply
    assert reply["result"]["command"]["acknowledgement"] == "accepted"

    engine.call("stop-container", {})
    engine.call("start-container", {})
    time.sleep(1.0)  # well past native.startup_hold_ms: only release ends it
    assert engine.status()["instance"]["state"] == "starting"

    assert engine.call("release", {}) == {"released": True}

    snapshot = engine.status()
    assert snapshot["current_attempt"]["issue"]["number"] == 22
    assert snapshot["intake"]["priority"] is None


def test_finish_attempt_answers_when_deferred_publication_ends_the_engine(
    engine: _Engine,
) -> None:
    engine.call("set-issue", {"issue": 23, "eligible": True})
    engine.call("release", {})
    attempt_id = engine.status()["current_attempt"]["attempt_id"]
    engine.call("defer-publication", {})

    started = time.monotonic()
    finished = engine.call("finish-attempt", {"native_outcome": "complete"})

    assert finished == {"attempt_id": attempt_id, "native_outcome": "complete"}
    assert time.monotonic() - started < 30.0
    # The publication is left pending for the restart to recover.
    assert (engine.root / "instance" / "state" / "attempt.json").exists()


# ---------------------------------------------------------------------------
# Operator Handoff through the harness executor (#383)
# ---------------------------------------------------------------------------


def _run_executor_bounded(executor: object, working: Path, flags: Path) -> object:
    """Run one ``execute`` on a thread, unblocking it if it never returns.

    The harness executor blocks until something ends the attempt; when the
    behaviour under test is missing, a scripted outcome (matching the
    checkpoint-less attempt id ``None``) frees the thread so the test fails
    instead of hanging.
    """

    import asyncio

    result: list[object] = []
    worker = threading.Thread(
        target=lambda: result.append(
            asyncio.run(
                executor.execute(
                    issue_body="body", working_directory=working, issue_number=31
                )
            )
        ),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=10.0)
    if worker.is_alive():
        (flags / "scripted_outcome.json").write_text(
            json.dumps({"attempt_id": None, "native_outcome": "infrastructure_error"}),
            encoding="utf-8",
        )
        worker.join(timeout=10.0)
        raise AssertionError("the executor never acted on the pending handoff")
    return result[0]


def test_the_harness_executor_carries_out_a_pending_operator_handoff(
    tmp_path: Path,
) -> None:
    from datetime import datetime

    from conformance_harness.daemon import BlockingExecutor
    from simple_coding_agent.control import ControlStore, HandoffRejectedError
    from simple_coding_agent.model_execution import (
        ModelExecutionStatus,
        OperatorHandoff,
    )
    from tests.fakes import git, repository_with_main

    _, working = repository_with_main(tmp_path / "repo")
    data_dir = tmp_path / "data"
    (data_dir / "state").mkdir(parents=True)
    flags = tmp_path / "flags"
    flags.mkdir()
    store = ControlStore(data_dir)
    store.submit_handoff("req-383-1", has_active_attempt=True)

    def provider() -> OperatorHandoff | None:
        snapshot = store.handoff_snapshot()
        if snapshot is None or snapshot["acknowledgement"] != "accepted":
            return None
        accepted_at = datetime.fromisoformat(
            snapshot["accepted_at"].replace("Z", "+00:00")
        )
        return OperatorHandoff(snapshot["request_id"], accepted_at)

    events: list[tuple[str, str, str]] = []

    def reporter(event: str, request_id: str) -> None:
        if event == "delivered":
            store.mark_handoff_delivering(request_id)
        elif event == "begun":
            store.mark_handoff_begun(request_id)
        # The note is committed only after the handoff has begun.
        note = working / ".agent" / "handoff" / "31.md"
        events.append((event, request_id, "note" if note.exists() else "no note"))

    executor = BlockingExecutor(flags, data_dir)
    executor.set_operator_handoff_provider(provider)
    executor.set_operator_handoff_reporter(reporter)

    execution = _run_executor_bounded(executor, working, flags)

    assert execution.status is ModelExecutionStatus.HANDOFF_REQUESTED
    assert events == [
        ("delivered", "req-383-1", "no note"),
        ("begun", "req-383-1", "no note"),
    ]
    assert executor.operator_request_id == "req-383-1"
    note = (working / ".agent" / "handoff" / "31.md").read_text()
    assert "- reason: operator_request" in note
    assert git(working, "log", "-1", "--format=%s") == "Handoff note: issue #31"
    snapshot = store.handoff_snapshot()
    assert snapshot["begun"] is True
    assert snapshot["acknowledgement"] == "accepted"  # the engine finalizes it
    # Once begun, a second handoff now cannot undo it.
    with pytest.raises(HandoffRejectedError, match="already begun"):
        store.submit_handoff("req-383-2", has_active_attempt=True)


def test_the_harness_executor_latches_no_handoff_without_a_pending_one(
    tmp_path: Path,
) -> None:
    from conformance_harness.daemon import BlockingExecutor
    from simple_coding_agent.model_execution import ModelExecutionStatus
    from tests.fakes import repository_with_main

    _, working = repository_with_main(tmp_path / "repo")
    data_dir = tmp_path / "data"
    (data_dir / "state").mkdir(parents=True)
    flags = tmp_path / "flags"
    flags.mkdir()
    reported: list[tuple[str, str]] = []
    executor = BlockingExecutor(flags, data_dir)
    executor.set_operator_handoff_provider(lambda: None)
    executor.set_operator_handoff_reporter(lambda *event: reported.append(event))
    (flags / "scripted_outcome.json").write_text(
        json.dumps({"attempt_id": None, "native_outcome": "infrastructure_error"}),
        encoding="utf-8",
    )

    execution = _run_executor_bounded(executor, working, flags)

    assert execution.status is ModelExecutionStatus.INFRASTRUCTURE_ERROR
    assert executor.operator_request_id is None
    assert reported == []


def _await_final_acknowledgement(
    engine: _Engine, request_id: str, timeout: float = 60.0
) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        code, reply = engine.agentctl("command", request_id)
        assert code == 0, reply
        record = reply["result"]["command"]
        if record["acknowledgement"] != "accepted" or time.monotonic() >= deadline:
            return record
        time.sleep(0.2)


def test_handoff_now_completes_the_held_attempt_as_handoff(engine: _Engine) -> None:
    engine.call("set-issue", {"issue": 31, "eligible": True})
    engine.call("release", {})
    attempt_id = engine.status()["current_attempt"]["attempt_id"]
    engine.call("set-issue", {"issue": 31, "eligible": False})

    code, reply = engine.agentctl(
        "handoff", "now", "--request-id", "req-383-handoff",
        "--expect-repo", "octo/example",
    )

    assert code == 0, reply
    assert reply["result"]["command"]["acknowledgement"] in ("accepted", "completed")
    record = _await_final_acknowledgement(engine, "req-383-handoff")
    assert record["acknowledgement"] == "completed", record
    snapshot = engine.status()
    assert snapshot["current_attempt"] is None
    finished = snapshot["last_finished"]
    assert finished["attempt_id"] == attempt_id
    assert finished["outcome"]["native"] == "handoff"
    assert finished["sealed"] is True


def test_an_interrupted_handoff_ends_not_fulfilled_after_recovery(
    engine: _Engine,
) -> None:
    engine.call("set-issue", {"issue": 32, "eligible": True})
    engine.call("release", {})
    attempt_id = engine.status()["current_attempt"]["attempt_id"]
    engine.call("crash", {"point": "during-finalization"})

    engine.agentctl(
        "handoff", "now", "--request-id", "req-383-crash",
        "--expect-repo", "octo/example",
    )
    deadline = time.monotonic() + 60.0
    while (engine.root / "flags" / "armed_crash.json").exists():
        assert time.monotonic() < deadline, "the armed crash never fired"
        time.sleep(0.2)

    engine.call("stop-container", {})
    engine.call("start-container", {})
    assert engine.status()["instance"]["state"] == "starting"
    code, reply = engine.agentctl("command", "req-383-crash")
    assert code == 0, reply
    assert reply["result"]["command"]["acknowledgement"] in ("accepted", "not_fulfilled")

    engine.call("release", {})
    snapshot = engine.status()
    assert snapshot["instance"]["state"] == "recovering"
    assert snapshot["current_attempt"]["attempt_id"] == attempt_id
    engine.call("set-issue", {"issue": 32, "eligible": False})
    engine.call("release", {})

    snapshot = engine.status()
    assert snapshot["last_finished"]["attempt_id"] == attempt_id
    code, reply = engine.agentctl("command", "req-383-crash")
    assert code == 0, reply
    assert reply["result"]["command"]["acknowledgement"] == "not_fulfilled"


# ---------------------------------------------------------------------------
# After-outcome hold, archive growth and the halt seen after a restart (#384)
# ---------------------------------------------------------------------------


def _events_named(snapshot: dict, name: str) -> list[dict]:
    return [event for event in snapshot["recent_events"] if event.get("event") == name]


def test_finish_attempt_under_an_after_outcome_hold_leaves_the_attempt_unsealed(
    engine: _Engine,
) -> None:
    engine.call("set-issue", {"issue": 31, "eligible": True})
    engine.call("release", {})
    attempt_id = engine.status()["current_attempt"]["attempt_id"]
    engine.call("hold", {"point": "after-outcome", "scope": "active"})

    finished = engine.call("finish-attempt", {"native_outcome": "complete"})

    assert finished == {"attempt_id": attempt_id, "native_outcome": "complete"}
    # The hold is still armed: the outcome is in, finalization is paused.
    assert (engine.root / "flags" / "hold.json").exists()
    code, shown = engine.agentctl("attempts", "show", attempt_id)
    assert code == 0, shown
    attempt = shown["result"]["attempt"]
    assert attempt["sealed"] is False
    assert attempt["outcome"]["native"] == "complete"
    assert _events_named(engine.status(), "sealed") == []

    assert engine.call("release", {}) == {"released": True}

    # No polling: the seal is visible as soon as release answers.
    sealed = _events_named(engine.status(), "sealed")
    assert [event["attempt_id"] for event in sealed] == [attempt_id]
    code, shown = engine.agentctl("attempts", "show", attempt_id)
    assert code == 0, shown
    assert shown["result"]["attempt"]["sealed"] is True


def test_finish_attempt_still_clears_a_hold_that_blocks_the_executor(
    engine: _Engine,
) -> None:
    engine.call("set-issue", {"issue": 32, "eligible": True})
    engine.call("release", {})
    attempt_id = engine.status()["current_attempt"]["attempt_id"]
    engine.call("hold", {"point": "after-claim", "scope": "active"})

    finished = engine.call("finish-attempt", {"native_outcome": "no_changes"})

    assert finished == {"attempt_id": attempt_id, "native_outcome": "no_changes"}
    assert not (engine.root / "flags" / "hold.json").exists()
    assert engine.status()["current_attempt"] is None


def test_grow_archive_finds_the_archive_of_a_colon_bearing_attempt_id(
    engine: _Engine,
) -> None:
    engine.call("set-issue", {"issue": 33, "eligible": True})
    engine.call("release", {})
    attempt_id = engine.status()["current_attempt"]["attempt_id"]
    assert ":" in attempt_id

    for requested in (attempt_id, None):  # ``None`` names the active attempt
        grown = engine.call(
            "grow-archive", {"attempt_id": requested, "append_bytes": 16}
        )

        target = Path(grown["path"])
        assert grown["appended_bytes"] == 16
        assert target.read_bytes().endswith(b"x" * 16)
        assert target.parent.parent.name == "33"

    # The archive record still names the attempt after the growth.
    code, shown = engine.agentctl("attempts", "show", attempt_id)
    assert code == 0, shown
    assert shown["result"]["attempt"]["attempt_id"] == attempt_id
    engine.call("finish-attempt", {"native_outcome": "no_changes"})


def test_grow_archive_fails_without_an_unsealed_archive(engine: _Engine) -> None:
    code, reply = invoke(
        "grow-archive",
        request("grow-archive", {"attempt_id": None, "append_bytes": 8}),
        state_dir=engine.root,
    )

    assert code == 1
    assert reply["error"]["message"] == "no unsealed archive to grow"


def test_a_restart_after_a_halt_reports_one_halted_event(engine: _Engine) -> None:
    assert _events_named(engine.status(), "halted") == []
    engine.call("halt", {"reason": "harness-halt-test"})
    engine.call("start-container", {})
    engine.call("release", {})

    snapshot = engine.status()

    assert snapshot["instance"]["last_halt"]["reason"] == "harness-halt-test"
    halted = _events_named(snapshot, "halted")
    assert [event["reason"] for event in halted] == ["harness-halt-test"]
    assert halted[0]["at"] == snapshot["instance"]["last_halt"]["at"]
    # Oldest first: the halt predates everything this process emitted.
    assert snapshot["recent_events"][0] == halted[0]


def test_a_restart_without_a_halt_reports_no_halted_event(engine: _Engine) -> None:
    engine.call("stop-container", {})
    engine.call("start-container", {})
    engine.call("release", {})

    snapshot = engine.status()

    assert snapshot["instance"]["last_halt"] is None
    assert _events_named(snapshot, "halted") == []


# ---------------------------------------------------------------------------
# A failed finalization retains the attempt (#388, harness protocol 1.1)
# ---------------------------------------------------------------------------


def _retain_one_attempt(engine: _Engine, issue: int) -> str:
    """Claim ``issue``, fail its finalization and return the attempt ID."""

    engine.call("set-issue", {"issue": issue, "eligible": True})
    engine.call("release", {})
    attempt_id = engine.status()["current_attempt"]["attempt_id"]
    engine.call("set-issue", {"issue": issue, "eligible": False})
    assert engine.call("fail-finalization", {}) == {"armed": True}

    started = time.monotonic()
    finished = engine.call("finish-attempt", {"native_outcome": "complete"})

    assert finished == {"attempt_id": attempt_id, "native_outcome": "complete"}
    assert time.monotonic() - started < 30.0
    return attempt_id


def test_a_failed_finalization_leaves_a_retained_attempt(engine: _Engine) -> None:
    attempt_id = _retain_one_attempt(engine, 41)

    snapshot = engine.status()
    assert snapshot["instance"]["state"] == "recovering"
    assert snapshot["current_attempt"]["attempt_id"] == attempt_id
    last_finished = snapshot["last_finished"]
    assert last_finished is None or last_finished["attempt_id"] != attempt_id
    code, reply = engine.agentctl("attempts", "show", attempt_id)
    assert code == 0, reply
    summary = reply["result"]["attempt"]
    assert summary["attention_required"] is True
    assert summary["sealed"] is False

    # No Claim opens while the Retained Attempt exists.
    engine.call("set-issue", {"issue": 42, "eligible": True})
    assert engine.call("release", {}) == {"released": False}
    assert engine.status()["current_attempt"]["attempt_id"] == attempt_id


def test_finish_attempt_on_a_retained_attempt_is_failed(engine: _Engine) -> None:
    _retain_one_attempt(engine, 45)

    code, reply = invoke(
        "finish-attempt",
        request("finish-attempt", {"native_outcome": "complete"}),
        state_dir=engine.root,
    )

    assert code == 1, reply
    assert reply["error"]["code"] == "failed"
    assert engine.status()["instance"]["state"] == "recovering"


def test_recovery_retry_finalizes_a_retained_attempt_as_complete(
    engine: _Engine,
) -> None:
    attempt_id = _retain_one_attempt(engine, 43)

    code, reply = engine.agentctl(
        "recovery", "retry", attempt_id, "--request-id", "req-388-retry",
        "--expect-repo", "octo/example",
    )

    assert code == 0, reply
    assert reply["result"]["command"]["acknowledgement"] == "completed", reply
    snapshot = engine.status()
    assert snapshot["current_attempt"] is None
    assert snapshot["last_finished"]["attempt_id"] == attempt_id
    assert snapshot["last_finished"]["outcome"]["native"] == "complete"


@pytest.mark.xfail(
    reason="#390: recovery release recomputes the outcome instead of keeping"
    " the recorded verdict (contract section 14)",
    strict=True,
)
def test_recovery_release_finalizes_a_retained_attempt(engine: _Engine) -> None:
    attempt_id = _retain_one_attempt(engine, 44)

    code, reply = engine.agentctl(
        "recovery", "release", attempt_id, "--saved-at", "https://example.test/saved",
        "--request-id", "req-388-release", "--expect-repo", "octo/example",
    )

    assert code == 0, reply
    assert reply["result"]["command"]["acknowledgement"] == "completed", reply
    snapshot = engine.status()
    assert snapshot["current_attempt"] is None
    finished = snapshot["last_finished"]
    assert finished["attempt_id"] == attempt_id
    assert finished["lifecycle_state"] == "released"
    # No command may change the recorded verdict (contract section 14).
    assert finished["outcome"]["native"] == "complete"
