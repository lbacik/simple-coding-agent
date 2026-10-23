"""Per-attempt process cleanup: marker environment and post-session reaping.

Every model session carries a unique ``SCA_ATTEMPT_ID`` in the CLI
environment. After the SDK client disconnects -- on every exit path -- the
executor SIGTERMs every process still carrying that session's marker value
(SIGKILL after a grace period for survivors) and logs
``attempt_processes_reaped``. Processes without the marker, or with another
session's value, are never signalled.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_coding_agent.attempt_processes import (
    ATTEMPT_PROCESS_MARKER,
    find_marked_pids,
    reap_marked_processes,
)
from simple_coding_agent.config import RuntimeConfig
from simple_coding_agent.model_execution import ModelExecutionStatus, ModelExecutor


def runtime_config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        github_token="github-secret",
        meta_api_key="meta-secret",
        target_repo="octo/example",
        data_dir=tmp_path,
        clone_dir=tmp_path / "repo",
        poll_interval=60,
        log_level="INFO",
        model_timeout=60,
        publish_timeout=120,
        max_retries=3,
        max_consecutive_errors=3,
        agent_trust_project_settings=False,
        review_blocking_severities=frozenset({"must-fix"}),
    )


class FakeClient:
    def __init__(self, options: object, messages: list[object]) -> None:
        self.options = options
        self._messages = messages
        self.prompt: str | None = None
        self.interrupted = False
        self.queried_prompts: list[str] = []

    async def disconnect(self) -> None:
        return None

    async def connect(self, prompt: str) -> None:
        self.prompt = prompt

    async def interrupt(self) -> None:
        self.interrupted = True

    async def stop_task(self, task_id: str) -> None:
        return None

    async def query(self, prompt: str) -> None:
        self.queried_prompts.append(prompt)

    async def receive_response(self):
        for message in self._messages:
            yield message


class HangingClient(FakeClient):
    async def receive_response(self):  # type: ignore[override]
        while not self.interrupted:
            await asyncio.sleep(0)
        return
        yield  # pragma: no cover - makes this an async generator


class RaisingClient(FakeClient):
    async def connect(self, prompt: str) -> None:  # type: ignore[override]
        raise RuntimeError("boom")


def success_result() -> object:
    return SimpleNamespace(
        is_error=False,
        stop_reason="end_turn",
        model_usage={"muse-spark-1.3-contributor": {"input_tokens": 12}},
    )


def _with_options(client: FakeClient, options: object) -> FakeClient:
    client.options = options
    return client


def run_executor(
    tmp_path: Path,
    client: FakeClient,
    monkeypatch: pytest.MonkeyPatch,
    **overrides: object,
) -> tuple[object, list[tuple[str, str]], FakeClient]:
    """Run one execution with a fast reap grace; return (execution, events)."""

    monkeypatch.setattr(
        "simple_coding_agent.model_execution.ATTEMPT_PROCESS_GRACE_SECONDS", 0.2
    )
    events: list[tuple[str, str]] = []
    executor = ModelExecutor(
        runtime_config(tmp_path),
        client_factory=lambda options: _with_options(client, options),
        event_log=lambda event, detail="", level="INFO", issue_number=None: (
            events.append((event, detail))
        ),
    )
    execution = asyncio.run(
        executor.execute(
            issue_body="Fix it.",
            working_directory=tmp_path,
            **overrides,  # type: ignore[arg-type]
        )
    )
    return execution, events, client


def spawn_term_ignorer(marker_value: str) -> subprocess.Popen[bytes]:
    """Start a detached process ignoring SIGTERM; return once its guard is up."""

    env = dict(os.environ)
    env[ATTEMPT_PROCESS_MARKER] = marker_value
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, sys, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "sys.stdout.write('R'); sys.stdout.flush(); "
            "time.sleep(60)",
        ],
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    assert proc.stdout is not None
    assert proc.stdout.read(1) == b"R"
    return proc


def spawn_detached(
    argv: list[str], marker_value: str | None
) -> subprocess.Popen[bytes]:
    """Start a detached process, optionally carrying the session marker."""

    env = dict(os.environ)
    if marker_value is None:
        env.pop(ATTEMPT_PROCESS_MARKER, None)
    else:
        env[ATTEMPT_PROCESS_MARKER] = marker_value
    return subprocess.Popen(
        argv,
        env=env,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )


def terminate(proc: subprocess.Popen[bytes]) -> None:
    try:
        proc.kill()
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10)
    except (subprocess.TimeoutExpired, ChildProcessError):
        pass
    if proc.stdout is not None:
        try:
            proc.stdout.close()
        except (BrokenPipeError, OSError):
            pass


# --- Marker environment ----------------------------------------------------


def test_marker_env_carries_the_attempt_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeClient(None, [success_result()])

    execution, _, _ = run_executor(
        tmp_path, client, monkeypatch, attempt_id="attempt-123"
    )

    assert execution.status is ModelExecutionStatus.SUCCEEDED
    assert client.options.env[ATTEMPT_PROCESS_MARKER] == "attempt-123"  # type: ignore[union-attr]


def test_marker_is_unique_per_session_without_attempt_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = FakeClient(None, [success_result()])
    run_executor(tmp_path, first, monkeypatch)
    second = FakeClient(None, [success_result()])
    run_executor(tmp_path, second, monkeypatch)

    first_marker = first.options.env[ATTEMPT_PROCESS_MARKER]  # type: ignore[union-attr]
    second_marker = second.options.env[ATTEMPT_PROCESS_MARKER]  # type: ignore[union-attr]
    assert first_marker
    assert second_marker
    assert first_marker != second_marker


# --- /proc scanning --------------------------------------------------------


def test_find_marked_pids_scans_environ(tmp_path: Path) -> None:
    root = tmp_path / "proc"
    (root / "100").mkdir(parents=True)
    (root / "100" / "environ").write_bytes(b"PATH=/x\x00SCA_ATTEMPT_ID=abc\x00")
    (root / "101").mkdir()
    (root / "101" / "environ").write_bytes(b"SCA_ATTEMPT_ID=other\x00")
    (root / "102").mkdir()
    (root / "102" / "environ").write_bytes(b"SCA_ATTEMPT_ID=abc2\x00")
    (root / "103").mkdir()  # no environ file: exited mid-scan
    (root / "notapid").mkdir()

    assert find_marked_pids("SCA_ATTEMPT_ID", "abc", root) == [100]


# --- Reaper against real processes -----------------------------------------


def test_reap_kills_a_real_detached_process() -> None:
    marker = "test-reap-detached"
    proc = spawn_detached(["sleep", "60"], marker)
    try:
        assert proc.poll() is None
        reaped = reap_marked_processes(
            ATTEMPT_PROCESS_MARKER, marker, grace_seconds=0.2
        )
        assert [entry.pid for entry in reaped] == [proc.pid]
        assert proc.wait(timeout=10) is not None
    finally:
        terminate(proc)


def test_reap_leaves_unmarked_and_other_marker_processes_alive() -> None:
    plain = spawn_detached(["sleep", "60"], None)
    other = spawn_detached(["sleep", "60"], "someone-elses-session")
    try:
        reaped = reap_marked_processes(
            ATTEMPT_PROCESS_MARKER, "this-session", grace_seconds=0.2
        )
        assert reaped == []
        assert plain.poll() is None
        assert other.poll() is None
    finally:
        terminate(plain)
        terminate(other)


def test_reap_sigkills_a_process_ignoring_sigterm() -> None:
    marker = "test-reap-sigkill"
    proc = spawn_term_ignorer(marker)
    try:
        reaped = reap_marked_processes(
            ATTEMPT_PROCESS_MARKER, marker, grace_seconds=0.2
        )
        assert len(reaped) == 1
        assert reaped[0].pid == proc.pid
        assert reaped[0].sigkill is True
        assert proc.wait(timeout=10) is not None
    finally:
        terminate(proc)


# --- Executor integration: the reap runs on every exit path ----------------


def test_reap_runs_on_the_normal_path_and_names_what_it_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = "attempt-normal-path"
    proc = spawn_detached(["sleep", "60"], marker)
    try:
        execution, events, _ = run_executor(
            tmp_path,
            FakeClient(None, [success_result()]),
            monkeypatch,
            attempt_id=marker,
        )
        assert execution.status is ModelExecutionStatus.SUCCEEDED
        assert proc.wait(timeout=10) is not None
    finally:
        terminate(proc)
    reaped = [detail for name, detail in events if name == "attempt_processes_reaped"]
    assert len(reaped) == 1
    assert f"pids={proc.pid}" in reaped[0]
    assert "sleep" in reaped[0]
    assert "sigkill_pids=" in reaped[0]


def test_reap_sigkill_is_reported_in_the_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = "attempt-sigkill-path"
    proc = spawn_term_ignorer(marker)
    try:
        execution, events, _ = run_executor(
            tmp_path,
            FakeClient(None, [success_result()]),
            monkeypatch,
            attempt_id=marker,
        )
        assert execution.status is ModelExecutionStatus.SUCCEEDED
        assert proc.wait(timeout=10) is not None
    finally:
        terminate(proc)
    reaped = [detail for name, detail in events if name == "attempt_processes_reaped"]
    assert len(reaped) == 1
    assert f"pids={proc.pid}" in reaped[0]
    assert f"sigkill_pids={proc.pid}" in reaped[0]


def test_reap_runs_on_the_model_timeout_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = "attempt-timeout-path"
    proc = spawn_detached(["sleep", "60"], marker)
    try:
        import dataclasses

        monkeypatch.setattr(
            "simple_coding_agent.model_execution.ATTEMPT_PROCESS_GRACE_SECONDS", 0.2
        )
        events: list[tuple[str, str]] = []
        config = dataclasses.replace(runtime_config(tmp_path), model_timeout=1)
        client = HangingClient(None, [])
        executor = ModelExecutor(
            config,
            client_factory=lambda options: client,
            event_log=lambda event, detail="", level="INFO", issue_number=None: (
                events.append((event, detail))
            ),
        )
        execution = asyncio.run(
            executor.execute(
                issue_body="Fix it.",
                working_directory=tmp_path,
                attempt_id=marker,
            )
        )
        assert execution.status is ModelExecutionStatus.INFRASTRUCTURE_ERROR
        assert "MODEL_TIMEOUT" in execution.explanation
        assert client.interrupted is True
        assert proc.wait(timeout=10) is not None
    finally:
        terminate(proc)
    assert any(
        name == "attempt_processes_reaped" and str(proc.pid) in detail
        for name, detail in events
    )


def test_reap_runs_when_the_sdk_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = "attempt-exception-path"
    proc = spawn_detached(["sleep", "60"], marker)
    try:
        execution, events, _ = run_executor(
            tmp_path, RaisingClient(None, []), monkeypatch, attempt_id=marker
        )
        assert execution.status is ModelExecutionStatus.INFRASTRUCTURE_ERROR
        assert "Claude SDK execution failed" in execution.explanation
        assert proc.wait(timeout=10) is not None
    finally:
        terminate(proc)
    assert any(
        name == "attempt_processes_reaped" and str(proc.pid) in detail
        for name, detail in events
    )


def test_reaper_failure_is_logged_and_leaves_the_outcome_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("proc unavailable")

    monkeypatch.setattr(
        "simple_coding_agent.model_execution.reap_marked_processes", boom
    )
    execution, events, _ = run_executor(
        tmp_path, FakeClient(None, [success_result()]), monkeypatch
    )

    assert execution.status is ModelExecutionStatus.SUCCEEDED
    failures = [
        (name, detail)
        for name, detail in events
        if name == "attempt_process_reap_failed"
    ]
    assert len(failures) == 1
    assert "proc unavailable" in failures[0][1]


def test_no_event_when_nothing_was_left_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution, events, _ = run_executor(
        tmp_path, FakeClient(None, [success_result()]), monkeypatch
    )

    assert execution.status is ModelExecutionStatus.SUCCEEDED
    assert not [name for name, _ in events if name.startswith("attempt_process")]
