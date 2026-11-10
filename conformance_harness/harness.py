"""Host-side conformance harness executable (operator contract §15.1).

Invocation: ``python -m conformance_harness <verb> '<request-json>'``.
``<request-json>`` is ``{"verb": "<verb>", "params": {...}}``; the argv verb
and the document verb must agree. Every invocation writes exactly one JSON
document to stdout. Exit codes mirror the contract classes: ``0`` for
``ok: true``, ``1`` for ``failed``, ``2`` for ``usage``, ``3`` for
``unsupported``.

The harness manages one current instance at a time under
``$SCA_HARNESS_STATE_DIR`` (default ``~/.cache/sca-conformance-harness``):
``setup`` creates a fresh state volume and starts the fakes-wired engine
(the real lifecycle against the shared test fakes from ``tests/fakes.py``);
``teardown`` stops it and removes the volume.

Supported verbs: all fourteen. ``unsupported`` answers name what the engine
cannot do: ``capability`` (a requested Operator Capability the fixed engine
set does not advertise) and ``native-outcome`` (a ``finish-attempt``
outcome the engine has no notion of). Every hold point, crash point, verb
and the ``defer-publication`` one-shot are supported.

Docker mode (``SCA_HARNESS_DOCKER=1``) runs the same verbs inside a
container for the external producer suite, which speaks to the producer
only through real ``docker exec``/``docker run`` (spec section 3): ``setup``
creates a named volume and a container from ``SCA_HARNESS_IMAGE`` (default
``simple-coding-agent:harness``) whose entrypoint is overridden to
``sleep infinity`` behind ``--init`` (so orphaned engines are reaped), then
proxies every verb through ``docker exec`` into
the in-container harness CLI, which manages the engine on the volume
exactly as in process-local mode. The suite's ``docker exec`` and offline
``docker run`` reach the same volume because the harness image bakes the
volume paths in as its ``DATA_DIR``/``AGENTCTL_SOCKET`` defaults.
``teardown`` removes the container and the volume.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


HARNESS_VERSION = "1.0"

VERBS = (
    "setup",
    "teardown",
    "set-issue",
    "hold",
    "release",
    "finish-attempt",
    "hold-lock",
    "crash",
    "terminate",
    "stop-container",
    "start-container",
    "halt",
    "grow-archive",
    "defer-publication",
)

_HOLD_POINTS = ("before-claim", "after-claim", "native-phase", "after-outcome")
_CRASH_POINTS = (
    "between-commit-and-reply",
    "before-claim",
    "after-claim",
    "during-finalization",
    "now",
)
_SCOPES = ("next", "active")

#: Native outcomes the engine can finish an attempt with. Anything else is
#: answered ``unsupported`` with ``details.what: "native-outcome"``.
_NATIVE_OUTCOMES = (
    "complete",
    "incomplete",
    "infrastructure_error",
    "no_changes",
    "handoff",
)

_REPOSITORY_RE = re.compile(r"^[^/\s]+/[^/\s]+$")

#: How long ``setup``/``start-container`` wait for the engine to serve.
_DAEMON_READY_TIMEOUT_SECONDS = 30.0
#: How long ``finish-attempt``/``halt`` wait for the engine to react.
_COMPLETION_TIMEOUT_SECONDS = 120.0


# ---------------------------------------------------------------------------
# Paths and state
# ---------------------------------------------------------------------------


def state_dir() -> Path:
    """Resolve the harness state directory (one current instance)."""

    raw = os.environ.get("SCA_HARNESS_STATE_DIR", "~/.cache/sca-conformance-harness")
    return Path(raw).expanduser()


def _paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "state": root / "state.json",
        "issues": root / "issues.json",
        "instance": root / "instance",
        "remote": root / "remote.git",
        "clone": root / "clone",
        "socket": root / "daemon.sock",
        "pid": root / "daemon.pid",
        "ready": root / "daemon.ready",
        "log": root / "daemon.log",
        "flags": root / "flags",
        "hold": root / "flags" / "hold.json",
        "armed_crash": root / "flags" / "armed_crash.json",
        "scripted_outcome": root / "flags" / "scripted_outcome.json",
        "halt_request": root / "flags" / "halt_request.json",
        "deferred_publication": root / "flags" / "deferred_publication",
        "pending_publication": root / "flags" / "pending_publication.json",
        "engine_lock": root / "flags" / "engine.lock",
    }


def _default_state() -> dict:
    return {
        "version": 1,
        "repository": "",
        "capabilities": [],
        "container": "",
        "image": "",
        "exec_user": "",
        "container_running": False,
        "engine_pid": None,
        "startup_hold_ms": 500,
        "recovery_hold_ms": 1500,
    }


def _load_state(root: Path) -> dict | None:
    path = _paths(root)["state"]
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _save_state(root: Path, payload: dict) -> None:
    path = _paths(root)["state"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _load_issues(root: Path) -> dict[str, dict]:
    path = _paths(root)["issues"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_issues(root: Path, issues: dict[str, dict]) -> None:
    path = _paths(root)["issues"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(issues, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _engine_alive(root: Path) -> bool:
    state = _load_state(root)
    if not state:
        return False
    pid = state.get("engine_pid")
    if not isinstance(pid, int):
        return False
    try:
        return not _pid_gone(pid)
    except OverflowError:
        return False


def _read_json_file(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Replies
# ---------------------------------------------------------------------------


def _ok(verb: str, result: dict) -> tuple[int, dict]:
    return 0, {"harness": HARNESS_VERSION, "verb": verb, "ok": True, "result": result}


def _failed(verb: str | None, message: str, details: dict | None = None) -> tuple[int, dict]:
    error: dict = {"code": "failed", "message": message}
    if details is not None:
        error["details"] = details
    reply: dict = {"harness": HARNESS_VERSION, "ok": False, "error": error}
    if verb is not None:
        reply["verb"] = verb
    return 1, reply


def _usage(verb: str | None, message: str) -> tuple[int, dict]:
    reply: dict = {
        "harness": HARNESS_VERSION,
        "ok": False,
        "error": {"code": "usage", "message": message},
    }
    if verb is not None:
        reply["verb"] = verb
    return 2, reply


def _unsupported(verb: str, what: str, name: str) -> tuple[int, dict]:
    return 3, {
        "harness": HARNESS_VERSION,
        "verb": verb,
        "ok": False,
        "error": {
            "code": "unsupported",
            "message": f"this engine does not support {what} {name!r}",
            "details": {"what": what, "name": name},
        },
    }


# ---------------------------------------------------------------------------
# Request validation (strict: unknown parameters are usage errors)
# ---------------------------------------------------------------------------


def _check_no_unknown(verb: str, params: dict, allowed: set[str]) -> str | None:
    unknown = sorted(key for key in params if key not in allowed)
    if unknown:
        return f"unknown parameter(s) for {verb}: {', '.join(unknown)}"
    return None


def _validate(verb: str, params: object) -> str | None:
    """Validate one verb's parameters; return an error message or None."""

    if not isinstance(params, dict):
        return f"params for {verb} must be an object"
    if verb == "setup":
        error = _check_no_unknown(verb, params, {"repository", "capabilities", "config"})
        if error:
            return error
        repository = params.get("repository")
        if not isinstance(repository, str) or not _REPOSITORY_RE.match(repository):
            return "setup.repository must be owner/name"
        if "capabilities" in params and (
            not isinstance(params["capabilities"], list)
            or not all(isinstance(item, str) and item for item in params["capabilities"])
        ):
            return "setup.capabilities must be a list of non-empty strings"
        if "config" in params:
            config = params["config"]
            if not isinstance(config, dict):
                return "setup.config must be an object"
            bad = sorted(key for key in config if not key.startswith("native."))
            if bad:
                return f"setup.config allows only native.* keys: {', '.join(bad)}"
        return None
    if verb in ("teardown", "release", "terminate", "stop-container", "start-container",
                "defer-publication"):
        error = _check_no_unknown(verb, params, set())
        if error:
            return error
        return None
    if verb == "set-issue":
        error = _check_no_unknown(verb, params, {"issue", "eligible", "title"})
        if error:
            return error
        issue = params.get("issue")
        if not isinstance(issue, int) or isinstance(issue, bool) or issue < 1:
            return "set-issue.issue must be an integer >= 1"
        if not isinstance(params.get("eligible"), bool):
            return "set-issue.eligible must be a boolean"
        if "title" in params and (
            not isinstance(params["title"], str) or not params["title"]
        ):
            return "set-issue.title must be a non-empty string"
        return None
    if verb == "hold":
        error = _check_no_unknown(verb, params, {"point", "phase", "scope"})
        if error:
            return error
        if params.get("point") not in _HOLD_POINTS:
            return f"hold.point must be one of {', '.join(_HOLD_POINTS)}"
        if params.get("point") == "native-phase":
            if not isinstance(params.get("phase"), str) or not params["phase"]:
                return "hold.phase is required exactly when point is native-phase"
        elif "phase" in params:
            return "hold.phase is allowed only when point is native-phase"
        if "scope" in params and params["scope"] not in _SCOPES:
            return f"hold.scope must be one of {', '.join(_SCOPES)}"
        return None
    if verb == "finish-attempt":
        error = _check_no_unknown(verb, params, {"native_outcome", "reason"})
        if error:
            return error
        if not isinstance(params.get("native_outcome"), str) or not params["native_outcome"]:
            return "finish-attempt.native_outcome must be a non-empty string"
        if "reason" in params and not isinstance(params["reason"], (str, type(None))):
            return "finish-attempt.reason must be a string or null"
        return None
    if verb == "hold-lock":
        error = _check_no_unknown(verb, params, {"duration_ms"})
        if error:
            return error
        duration = params.get("duration_ms")
        if (
            not isinstance(duration, int)
            or isinstance(duration, bool)
            or duration < 1
            or duration > 60000
        ):
            return "hold-lock.duration_ms must be an integer between 1 and 60000"
        return None
    if verb == "crash":
        error = _check_no_unknown(verb, params, {"point"})
        if error:
            return error
        if params.get("point") not in _CRASH_POINTS:
            return f"crash.point must be one of {', '.join(_CRASH_POINTS)}"
        return None
    if verb == "halt":
        error = _check_no_unknown(verb, params, {"reason", "detail"})
        if error:
            return error
        if not isinstance(params.get("reason"), str) or not params["reason"]:
            return "halt.reason must be a non-empty string"
        if "detail" in params and not isinstance(params["detail"], str):
            return "halt.detail must be a string"
        return None
    if verb == "grow-archive":
        error = _check_no_unknown(verb, params, {"attempt_id", "append_bytes"})
        if error:
            return error
        if "attempt_id" in params and not isinstance(params["attempt_id"], (str, type(None))):
            return "grow-archive.attempt_id must be a string or null"
        append_bytes = params.get("append_bytes")
        if (
            not isinstance(append_bytes, int)
            or isinstance(append_bytes, bool)
            or append_bytes < 1
        ):
            return "grow-archive.append_bytes must be an integer >= 1"
        return None
    return f"unknown verb {verb!r}"  # pragma: no cover - dispatch guards this


# ---------------------------------------------------------------------------
# Engine management
# ---------------------------------------------------------------------------


def _daemon_module_available() -> bool:
    try:
        import conformance_harness.daemon  # noqa: F401
    except ImportError:
        return False
    return True


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


#: Engine processes spawned by this process, by pid. Keeping the handle
#: lets us reap an exited engine (``Popen.poll``) instead of mistaking its
#: zombie for a live engine. Separate CLI invocations are different
#: processes, so their engines are never in this table (init reaps those).
_SPAWNED: dict[int, subprocess.Popen] = {}


#: The commit identity the fakes-wired engine uses when none is configured.
_HARNESS_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "Conformance Harness",
    "GIT_AUTHOR_EMAIL": "harness@example.invalid",
    "GIT_COMMITTER_NAME": "Conformance Harness",
    "GIT_COMMITTER_EMAIL": "harness@example.invalid",
}


def _spawn_daemon(root: Path, repository: str) -> int:
    """Start the fakes-wired engine; return its pid."""

    paths = _paths(root)
    environ = dict(os.environ, SCA_HARNESS_STATE_DIR=str(root))
    # The daemon reuses the shared test fakes, so the repository checkout
    # must stay importable even when the harness runs from the image.
    repo = str(_repo_root())
    python_path = environ.get("PYTHONPATH", "")
    environ["PYTHONPATH"] = repo + (os.pathsep + python_path if python_path else "")
    # Sealing attempt work commits in the fake clone, and the harness image
    # has no git identity, so an unconfigured commit would fail and retain
    # every attempt for recovery. ``GIT_*`` variables already in the
    # environment win; otherwise these also take precedence over a host's
    # git config, which only affects commits in the throwaway fake clone.
    for name, value in _HARNESS_GIT_IDENTITY.items():
        environ.setdefault(name, value)
    log_file = paths["log"].open("ab")
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "conformance_harness.daemon", str(root)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=environ,
        )
    finally:
        log_file.close()
    _SPAWNED[process.pid] = process
    return process.pid


def _wait_for(predicate, timeout: float, interval: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _stop_daemon(root: Path, *, kill: bool = False, timeout: float = 15.0) -> bool:
    """Stop the engine process; return True when it is no longer alive."""

    state = _load_state(root)
    pid = state.get("engine_pid") if state else None
    if not isinstance(pid, int):
        return True
    try:
        os.kill(pid, signal.SIGKILL if kill else signal.SIGTERM)
    except OSError:
        return True
    gone = _wait_for(
        lambda: _pid_gone(pid), timeout=timeout,
    )
    if not gone and not kill:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            return True
        gone = _wait_for(lambda: _pid_gone(pid), timeout=5.0)
    return gone


#: Where the process table is read to tell a zombie from a live process.
#: Module-level so tests can point it at a fake table.
_PROC_ROOT = Path("/proc")


def _is_zombie(pid: int) -> bool:
    """Whether ``pid`` has exited but was never reaped (Linux ``/proc`` only).

    Without a readable ``/proc/<pid>/stat`` (another platform, or the process
    is already gone) the answer is ``False`` and callers fall back to the
    signal probe.
    """

    try:
        stat = (_PROC_ROOT / str(pid) / "stat").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return False
    # The command name is parenthesised and may itself hold spaces or
    # parentheses, so the state field is the first one after the last ")".
    fields = stat[stat.rfind(")") + 1 :].split()
    return bool(fields) and fields[0] == "Z"


def _pid_gone(pid: int) -> bool:
    # Reap our own exited engines first: a zombie is still signalable, so
    # ``kill(pid, 0)`` alone reports it alive and every stop/halt wait burns
    # its full timeout. Engines spawned by a different CLI process are not
    # our children (``waitpid`` raises ``ChildProcessError``). Those are
    # reaped by init once orphaned, but only when PID 1 is a real init; a
    # zombie nobody reaps still counts as gone, which ``/proc`` reveals.
    handle = _SPAWNED.get(pid)
    if handle is not None:
        if handle.poll() is None:
            return False
        _SPAWNED.pop(pid, None)
        return True
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    except OSError:
        return True
    else:
        if done == pid:
            return True
    if _is_zombie(pid):
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return True
    return False


def _active_attempt_id(root: Path) -> str | None:
    """The active attempt id (the checkpoint's ``started_at``), if any."""

    path = _paths(root)["instance"] / "state" / "attempt.json"
    payload = _read_json_file(path)
    if not payload:
        return None
    started_at = payload.get("started_at")
    return started_at if isinstance(started_at, str) and started_at else None


def _engine_status(root: Path) -> dict | None:
    """Read the live contract status snapshot through the control socket."""

    from simple_coding_agent.control_server import send_request

    socket_path = _paths(root)["socket"]
    if not socket_path.exists():
        return None
    try:
        reply = send_request(socket_path, {"op": "contract_status"}, timeout=10.0)
    except Exception:
        return None
    if not isinstance(reply, dict):
        return None
    snapshot = reply.get("contract_status")
    return snapshot if isinstance(snapshot, dict) else None


def _advertised_capabilities() -> list[str]:
    from simple_coding_agent.contract_adapter import advertised_capabilities

    try:
        return sorted(advertised_capabilities(""))
    except Exception:
        return sorted(
            [
                "errors-reset",
                "next-issue",
                "operator-handoff",
                "recovery-release",
                "recovery-retry",
                "stop-after",
            ]
        )


# ---------------------------------------------------------------------------
# Verbs
# ---------------------------------------------------------------------------


def _verb_setup(root: Path, params: dict) -> tuple[int, dict]:
    state = _load_state(root)
    if state is not None and _engine_alive(root):
        return _failed("setup", "an instance already exists; teardown first")
    if not _daemon_module_available():
        return _failed(
            "setup", "the harness engine is not installed (no conformance_harness.daemon)"
        )
    requested = params.get("capabilities") or []
    advertised = _advertised_capabilities()
    for name in requested:
        if name not in advertised:
            return _unsupported("setup", "capability", name)
    config = params.get("config") or {}
    try:
        startup_hold_ms = int(config.get("native.startup_hold_ms", 500))
        recovery_hold_ms = int(config.get("native.recovery_hold_ms", 1500))
    except (TypeError, ValueError):
        return _usage("setup", "native.* hold knobs must be integers")
    repository = params["repository"]
    # Fresh state volume: remove any previous instance data first.
    for name in ("instance", "remote.git", "clone"):
        target = root / name
        if target.is_dir() and not target.is_symlink():
            import shutil

            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink(missing_ok=True)
    for name in ("daemon.ready", "daemon.pid"):
        (root / name).unlink(missing_ok=True)
    flags = _paths(root)["flags"]
    if flags.is_dir():
        import shutil

        shutil.rmtree(flags, ignore_errors=True)
    flags.mkdir(parents=True, exist_ok=True)
    exec_user = getpass.getuser() or "agent"
    if exec_user == "root":
        exec_user = "agent"
    payload = {
        **_default_state(),
        "repository": repository,
        "capabilities": sorted(set(advertised) | set(requested)),
        "container": "sca-harness",
        "image": "simple-coding-agent:harness",
        "exec_user": exec_user,
        "container_running": True,
        "startup_hold_ms": startup_hold_ms,
        "recovery_hold_ms": recovery_hold_ms,
    }
    _save_state(root, payload)
    _save_issues(root, {})
    pid = _spawn_daemon(root, repository)
    payload["engine_pid"] = pid
    _save_state(root, payload)
    # Wait only for the control endpoint: the suite must be able to observe
    # ``starting`` itself, so setup returns before startup checks finish.
    serving = _wait_for(
        lambda: _paths(root)["socket"].exists(),
        timeout=_DAEMON_READY_TIMEOUT_SECONDS,
    )
    if not serving:
        return _failed(
            "setup", "the engine did not serve the control endpoint; see daemon.log"
        )
    return _ok(
        "setup",
        {
            "container": payload["container"],
            "image": payload["image"],
            "volumes": [{"name": "harness-data", "path": "/data"}],
            "exec_user": payload["exec_user"],
            "repository": repository,
            "capabilities": payload["capabilities"],
        },
    )


def _verb_teardown(root: Path) -> tuple[int, dict]:
    state = _load_state(root)
    if state is None:
        return _ok("teardown", {"removed": False})
    _stop_daemon(root, kill=True)
    import shutil

    for name in ("instance", "remote.git", "clone", "flags", "daemon.sock",
                 "daemon.pid", "daemon.ready", "daemon.log", "state.json",
                 "issues.json"):
        target = root / name
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink(missing_ok=True)
    # An armed one-shot crash that never fired is discarded with the instance.
    return _ok("teardown", {"removed": True})


def _require_instance(root: Path, verb: str) -> tuple[dict | None, tuple[int, dict] | None]:
    state = _load_state(root)
    if state is None:
        return None, _failed(verb, "no instance; setup first")
    return state, None


def _verb_set_issue(root: Path, params: dict) -> tuple[int, dict]:
    state, error = _require_instance(root, "set-issue")
    if error is not None:
        return error
    issues = _load_issues(root)
    number = params["issue"]
    entry = {"eligible": params["eligible"]}
    if params.get("title") is not None:
        entry["title"] = params["title"]
    elif number in issues and "title" in issues[number]:
        entry["title"] = issues[number]["title"]
    issues[str(number)] = entry
    _save_issues(root, issues)
    return _ok("set-issue", {"issue": number, "eligible": params["eligible"]})


def _verb_hold(root: Path, params: dict) -> tuple[int, dict]:
    state, error = _require_instance(root, "hold")
    if error is not None:
        return error
    if params.get("scope") == "active" and _active_attempt_id(root) is None:
        return _failed("hold", "no active attempt for scope active")
    hold = {"point": params["point"], "phase": params.get("phase"), "scope": params.get("scope")}
    _write_json_file(_paths(root)["hold"], hold)
    return _ok(
        "hold", {"point": params["point"], "attempt_id": _active_attempt_id(root)}
    )


def _verb_release(root: Path) -> tuple[int, dict]:
    state, error = _require_instance(root, "release")
    if error is not None:
        return error
    hold_path = _paths(root)["hold"]
    released = hold_path.exists()
    hold_path.unlink(missing_ok=True)
    return _ok("release", {"released": released})


def _verb_finish_attempt(root: Path, params: dict) -> tuple[int, dict]:
    state, error = _require_instance(root, "finish-attempt")
    if error is not None:
        return error
    outcome = params["native_outcome"]
    if outcome not in _NATIVE_OUTCOMES:
        return _unsupported("finish-attempt", "native-outcome", outcome)
    if not _engine_alive(root):
        return _failed("finish-attempt", "the engine is not running")
    attempt_id = _active_attempt_id(root)
    if attempt_id is None:
        return _failed("finish-attempt", "no active attempt")
    scripted = {"native_outcome": outcome}
    if params.get("reason") is not None:
        scripted["reason"] = params["reason"]
    _write_json_file(_paths(root)["scripted_outcome"], scripted)
    # Release any hold so the blocked attempt can run to finalization.
    _paths(root)["hold"].unlink(missing_ok=True)
    finished = _wait_for(
        lambda: _active_attempt_id(root) is None,
        timeout=_COMPLETION_TIMEOUT_SECONDS,
    )
    if not finished:
        return _failed("finish-attempt", "the attempt did not finish in time")
    return _ok("finish-attempt", {"attempt_id": attempt_id, "native_outcome": outcome})


def _verb_hold_lock(root: Path, params: dict) -> tuple[int, dict]:
    state, error = _require_instance(root, "hold-lock")
    if error is not None:
        return error
    import fcntl

    duration_ms = params["duration_ms"]
    lock_path = _paths(root)["engine_lock"]
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with lock_path.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            time.sleep(duration_ms / 1000.0)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    held_ms = max(1, int((time.monotonic() - started) * 1000))
    return _ok("hold-lock", {"held_ms": held_ms})


def _verb_crash(root: Path, params: dict) -> tuple[int, dict]:
    state, error = _require_instance(root, "crash")
    if error is not None:
        return error
    point = params["point"]
    if point == "now":
        if not _engine_alive(root):
            return _failed("crash", "the engine is not running")
        _stop_daemon(root, kill=True)
        state = _load_state(root) or {}
        state["engine_pid"] = None
        _save_state(root, state)
        return _ok("crash", {"point": "now"})
    if not _engine_alive(root):
        return _failed("crash", "the engine is not running")
    _write_json_file(_paths(root)["armed_crash"], {"point": point})
    return _ok("crash", {"point": point})


def _verb_terminate(root: Path) -> tuple[int, dict]:
    state, error = _require_instance(root, "terminate")
    if error is not None:
        return error
    if not _engine_alive(root):
        return _failed("terminate", "the engine is not running")
    stopped = _stop_daemon(root, kill=False)
    if not stopped:
        return _failed("terminate", "the engine did not stop")
    state = _load_state(root) or {}
    state["engine_pid"] = None
    _save_state(root, state)
    _paths(root)["ready"].unlink(missing_ok=True)
    return _ok("terminate", {"signal": "SIGTERM"})


def _verb_stop_container(root: Path) -> tuple[int, dict]:
    state, error = _require_instance(root, "stop-container")
    if error is not None:
        return error
    if not state.get("container_running"):
        return _failed("stop-container", "the container is already stopped")
    if _engine_alive(root):
        _stop_daemon(root, kill=False)
    state = _load_state(root) or {}
    state["engine_pid"] = None
    state["container_running"] = False
    _save_state(root, state)
    _paths(root)["ready"].unlink(missing_ok=True)
    return _ok("stop-container", {"running": False})


def _verb_start_container(root: Path) -> tuple[int, dict]:
    state, error = _require_instance(root, "start-container")
    if error is not None:
        return error
    if state.get("container_running") and _engine_alive(root):
        return _failed("start-container", "the container is already running")
    if not _daemon_module_available():
        return _failed("start-container", "the harness engine is not installed")
    # An engine that crashed (``os._exit``) never removes its ready marker;
    # left in place it would let this restart report ready before the new
    # engine serves its control socket.
    _paths(root)["ready"].unlink(missing_ok=True)
    pid = _spawn_daemon(root, state.get("repository", ""))
    state = _load_state(root) or {}
    state["engine_pid"] = pid
    state["container_running"] = True
    _save_state(root, state)
    ready = _wait_for(
        lambda: _paths(root)["ready"].exists(),
        timeout=_DAEMON_READY_TIMEOUT_SECONDS,
    )
    if not ready:
        return _failed("start-container", "the engine did not become ready; see daemon.log")
    return _ok("start-container", {"running": True})


def _verb_halt(root: Path, params: dict) -> tuple[int, dict]:
    state, error = _require_instance(root, "halt")
    if error is not None:
        return error
    if not _engine_alive(root):
        return _failed("halt", "the engine is not running")
    _write_json_file(
        _paths(root)["halt_request"],
        {"reason": params["reason"], "detail": params.get("detail", "")},
    )
    stopped = _wait_for(lambda: not _engine_alive(root), timeout=30.0)
    # The halt exits the process by decision; reap the record either way.
    pid = (_load_state(root) or {}).get("engine_pid")
    if isinstance(pid, int) and not _pid_gone(pid):
        _stop_daemon(root, kill=True)
    state = _load_state(root) or {}
    state["engine_pid"] = None
    state["container_running"] = False
    _save_state(root, state)
    _paths(root)["ready"].unlink(missing_ok=True)
    if not stopped:
        return _failed("halt", "the engine did not halt in time")
    return _ok("halt", {"reason": params["reason"]})


def _archive_candidates(root: Path, attempt_id: str | None) -> list[Path]:
    """Candidate unsealed archive files, newest first."""

    logs = _paths(root)["instance"] / "logs"
    if not logs.is_dir():
        return []
    files: list[Path] = []
    for issue_dir in sorted(logs.iterdir()):
        if not issue_dir.is_dir():
            continue
        for run_dir in sorted(issue_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if attempt_id is not None and attempt_id not in run_dir.name:
                continue
            for member in sorted(run_dir.rglob("*")):
                if member.is_file() and member.suffix != ".tmp":
                    files.append(member)
    # Newest first so growth lands on the live run.
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return files


def _verb_grow_archive(root: Path, params: dict) -> tuple[int, dict]:
    state, error = _require_instance(root, "grow-archive")
    if error is not None:
        return error
    attempt_id = params.get("attempt_id")
    if attempt_id is None:
        attempt_id = _active_attempt_id(root)
    candidates = _archive_candidates(root, attempt_id)
    if not candidates:
        return _failed("grow-archive", "no unsealed archive to grow")
    target = candidates[0]
    append_bytes = params["append_bytes"]
    with target.open("ab") as handle:
        handle.write(b"x" * append_bytes)
    return _ok(
        "grow-archive", {"path": str(target), "appended_bytes": append_bytes}
    )


def _verb_defer_publication(root: Path) -> tuple[int, dict]:
    state, error = _require_instance(root, "defer-publication")
    if error is not None:
        return error
    _paths(root)["flags"].mkdir(parents=True, exist_ok=True)
    _paths(root)["deferred_publication"].write_text("armed", encoding="utf-8")
    return _ok("defer-publication", {"armed": True})


_VERB_HANDLERS = {
    "teardown": lambda root, params: _verb_teardown(root),
    "set-issue": _verb_set_issue,
    "hold": _verb_hold,
    "release": lambda root, params: _verb_release(root),
    "finish-attempt": _verb_finish_attempt,
    "hold-lock": _verb_hold_lock,
    "crash": _verb_crash,
    "terminate": lambda root, params: _verb_terminate(root),
    "stop-container": lambda root, params: _verb_stop_container(root),
    "start-container": lambda root, params: _verb_start_container(root),
    "halt": _verb_halt,
    "grow-archive": _verb_grow_archive,
    "defer-publication": lambda root, params: _verb_defer_publication(root),
}


# ---------------------------------------------------------------------------
# Docker mode (operator contract §3 + §15.1)
# ---------------------------------------------------------------------------


def _docker_mode() -> bool:
    """Whether this invocation proxies verbs into a harness container."""

    return os.environ.get("SCA_HARNESS_DOCKER", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _docker() -> str:
    return os.environ.get("DOCKER", "docker")


def _docker_image() -> str:
    return os.environ.get("SCA_HARNESS_IMAGE", "simple-coding-agent:harness")


#: How long a proxied verb may take before the proxy reports failure. The
#: suite enforces its own per-call timeout by killing the proxy, so this is
#: only a backstop against hanging forever.
_DOCKER_PROXY_TIMEOUT = 300.0


def _run_docker(args: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    """Run one docker command; raise on spawn or timeout failures."""

    return subprocess.run(
        [_docker(), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _docker_host_state() -> dict | None:
    """The host-side docker-mode record (container and volume names), if any."""

    state = _load_state(state_dir())
    if isinstance(state, dict) and state.get("docker_mode") is True:
        return state
    return None


def _docker_inner(verb: str, params: dict, container: str) -> tuple[int, dict] | tuple[int, None]:
    """Proxy one verb into the container's harness CLI; ``(exit, reply)``.

    The reply is the parsed inner JSON document, or ``None`` when the proxy
    itself broke (docker unavailable, container gone, non-JSON output).
    """

    request = json.dumps({"verb": verb, "params": params}, sort_keys=True)
    try:
        completed = _run_docker(
            [
                "exec",
                "-w",
                "/app",
                container,
                "python",
                "-m",
                "conformance_harness",
                verb,
                request,
            ],
            timeout=_DOCKER_PROXY_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 1, None
    try:
        reply = json.loads(completed.stdout)
    except ValueError:
        return 1, None
    if not isinstance(reply, dict):
        return 1, None
    return completed.returncode, reply


def _docker_setup(params: dict) -> tuple[int, dict]:
    """Create the volume and container, then run setup inside it."""

    if _docker_host_state() is not None:
        return _failed("setup", "an instance already exists; teardown first")
    name = f"sca-{uuid.uuid4().hex[:12]}"
    volume = f"{name}-data"
    image = _docker_image()
    try:
        created = _run_docker(["volume", "create", volume])
        if created.returncode != 0:
            return _failed(
                "setup", f"docker volume create failed: {created.stderr.strip()}"
            )
        run = _run_docker(
            [
                "run",
                "-d",
                "--name",
                name,
                "--volume",
                f"{volume}:/data",
                # Every verb runs in its own ``docker exec``, so the engine
                # it spawns is orphaned to PID 1; ``sleep`` never reaps it,
                # and a crashed engine would linger as a zombie.
                "--init",
                "--entrypoint",
                "sleep",
                image,
                "infinity",
            ],
            timeout=120.0,
        )
        if run.returncode != 0:
            _run_docker(["volume", "rm", volume])
            return _failed("setup", f"docker run failed: {run.stderr.strip()}")
    except (OSError, subprocess.TimeoutExpired) as error:
        return _failed("setup", f"docker is unavailable: {error}")
    exit_code, inner = _docker_inner("setup", params, name)
    if inner is None or exit_code not in (0, 1, 3):
        _run_docker(["rm", "--force", name])
        _run_docker(["volume", "rm", volume])
        detail = "the in-container setup broke" if inner is None else json.dumps(inner, sort_keys=True)
        return _failed("setup", f"harness container setup failed: {detail}")
    if exit_code != 0:
        _run_docker(["rm", "--force", name])
        _run_docker(["volume", "rm", volume])
        return exit_code, inner
    result = dict(inner.get("result") or {})
    try:
        inspected = _run_docker(["inspect", "--format", "{{.Id}}", image])
        image_id = inspected.stdout.strip() if inspected.returncode == 0 else image
    except (OSError, subprocess.TimeoutExpired):
        image_id = image
    result.update(
        {
            "container": name,
            "image": image_id,
            "volumes": [{"name": volume, "path": "/data"}],
            "exec_user": "agent",
        }
    )
    _save_state(
        state_dir(),
        {"version": 1, "docker_mode": True, "container": name, "volume": volume},
    )
    return 0, {
        "harness": HARNESS_VERSION,
        "verb": "setup",
        "ok": True,
        "result": result,
    }


def _docker_teardown() -> tuple[int, dict]:
    """Run teardown inside the container, then remove it and the volume."""

    state = _docker_host_state()
    if state is None:
        return _ok("teardown", {"removed": False})
    container = str(state.get("container", ""))
    volume = str(state.get("volume", ""))
    if container:
        _docker_inner("teardown", {}, container)
        _run_docker(["rm", "--force", container])
    if volume:
        _run_docker(["volume", "rm", volume])
    root = state_dir()
    for name in ("state.json",):
        try:
            (root / name).unlink(missing_ok=True)
        except OSError:
            pass
    return _ok("teardown", {"removed": True})


def _docker_dispatch(verb: str, params: dict) -> tuple[int, dict]:
    """Proxy one verb into the harness container (docker mode)."""

    if verb == "setup":
        return _docker_setup(params)
    if verb == "teardown":
        return _docker_teardown()
    state = _docker_host_state()
    if state is None:
        return _failed(verb, "no instance; setup first")
    container = str(state.get("container", ""))
    if not container:
        return _failed(verb, "no instance; setup first")
    exit_code, inner = _docker_inner(verb, params, container)
    if inner is None:
        return _failed(verb, "the harness container is unreachable")
    return exit_code, inner


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_harness(argv: list[str]) -> tuple[int, dict]:
    """Execute one harness invocation; return ``(exit_code, reply)``."""

    if not argv:
        return _usage(None, "no verb; usage: <harness> <verb> '<request-json>'")
    verb = argv[0]
    if verb not in VERBS:
        return _usage(None, f"unknown verb {verb!r}")
    if len(argv) < 2:
        return _usage(verb, f"missing request document for {verb!r}")
    try:
        document = json.loads(argv[1])
    except ValueError:
        return _usage(verb, f"request document for {verb!r} is not valid JSON")
    if not isinstance(document, dict):
        return _usage(verb, f"request document for {verb!r} must be an object")
    if set(document) != {"verb", "params"}:
        return _usage(verb, f"request document for {verb!r} must hold exactly verb and params")
    if document.get("verb") != verb:
        return _usage(verb, "the argv verb and the document verb must agree")
    error = _validate(verb, document.get("params"))
    if error is not None:
        return _usage(verb, error)
    params = document["params"]
    if _docker_mode():
        return _docker_dispatch(verb, params)
    root = state_dir()
    if verb == "setup":
        return _verb_setup(root, params)
    handler = _VERB_HANDLERS[verb]
    return handler(root, params)


def main(argv: list[str] | None = None) -> int:
    """Write one JSON reply to stdout; exit with the harness class code."""

    exit_code, reply = run_harness(list(sys.argv[1:] if argv is None else argv))
    sys.stdout.write(json.dumps(reply, sort_keys=True) + "\n")
    sys.stdout.flush()
    return exit_code


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
