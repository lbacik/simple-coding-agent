"""Default sleepers resolve at call time and stay instant suite-wide.

Regression tests for the wall-clock trap where ``AgentLifecycle`` (and the
same-shaped ``Publisher`` / ``GitWorkspace`` defaults) bound ``time.sleep``
at definition time: any new sleep on a tested path then turned every test
that builds those classes without ``sleeper=`` into real waiting without
failing anything. The defaults are now ``None`` with a module-level
``_default_sleeper`` helper consulted at construction, the suite replaces
that binding with a recording instant stub (see ``tests/conftest.py``), and
these tests pin both sides: the idle paths stay instant with a
``poll_interval`` that would hang the suite if it ever waited for real, and
the production helper still reaches the real ``time.sleep`` binding.
"""

from __future__ import annotations

import inspect
import time
import types
from pathlib import Path

import simple_coding_agent.lifecycle as lifecycle_module
from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.control import ControlStore
from simple_coding_agent.git_workspace import GitWorkspace
from simple_coding_agent.lifecycle import AgentLifecycle, LifecycleStatus
from simple_coding_agent.publication import Publisher
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace

POLL_INTERVAL = 3600
INSTANT_ENOUGH = 5.0


def _idle_lifecycle(
    tmp_path: Path, *, control: bool = False, dirty: bool = False
) -> AgentLifecycle:
    """Build a lifecycle with no injected sleeper parked on an idle path."""

    workspace = InMemoryWorkspace(tmp_path / "repo", issue_number=24)
    if dirty:
        workspace.make_dirty("uncommitted.txt", "wip")
    lifecycle = AgentLifecycle(
        tracker=FakeTracker(),
        attempt_state=AttemptStateStore(tmp_path),
        workspace=workspace,
        profile_loader=lambda _: None,
        publisher=FakePublisher(),
        poll_interval=POLL_INTERVAL,
        control_store=ControlStore(tmp_path) if control else None,
    )
    return lifecycle


def _assert_first_cycle_sleeps(seconds: list[float], *, spread: bool) -> None:
    """Assert one first-cycle idle run's sleeps through the default sleeper.

    With ``spread`` (intake may claim) the cycle first takes the one-off
    startup spread delay, uniform within ``[0, POLL_INTERVAL / 2]``; every
    idle cycle then sleeps one jittered poll, uniform within ±20% of
    ``POLL_INTERVAL``.
    """

    assert len(seconds) == (2 if spread else 1)
    if spread:
        assert 0 <= seconds[0] <= POLL_INTERVAL / 2
    assert 0.8 * POLL_INTERVAL <= seconds[-1] <= 1.2 * POLL_INTERVAL


def _lifecycle_sleeps(calls: list[tuple[str, float]]) -> list[float]:
    assert {source for source, _ in calls} <= {"lifecycle"}
    return [seconds for _, seconds in calls]


def test_sleeper_defaults_are_none_so_lookup_happens_at_call_time() -> None:
    """A definition-time ``time.sleep`` default would bypass every test stub."""

    assert (
        inspect.signature(AgentLifecycle.__init__).parameters["sleeper"].default is None
    )
    assert inspect.signature(Publisher.__init__).parameters["sleeper"].default is None
    assert inspect.signature(GitWorkspace.__init__).parameters["sleeper"].default is None


def test_empty_queue_without_sleeper_is_instant(
    tmp_path: Path, default_sleeper_calls: list[tuple[str, float]]
) -> None:
    started = time.monotonic()
    result = _idle_lifecycle(tmp_path).run_once()
    elapsed = time.monotonic() - started

    assert result.status is LifecycleStatus.IDLE
    assert elapsed < INSTANT_ENOUGH
    _assert_first_cycle_sleeps(_lifecycle_sleeps(default_sleeper_calls), spread=True)
    default_sleeper_calls.clear()


def test_stopped_intake_without_sleeper_is_instant(
    tmp_path: Path, default_sleeper_calls: list[tuple[str, float]]
) -> None:
    lifecycle = _idle_lifecycle(tmp_path, control=True)
    lifecycle.submit_stop("req-stop")

    started = time.monotonic()
    result = lifecycle.run_once()
    elapsed = time.monotonic() - started

    assert result.status is LifecycleStatus.IDLE
    assert elapsed < INSTANT_ENOUGH
    # Intake is stopped, so the startup spread is kept for the first cycle
    # that can claim; this cycle sleeps only the jittered poll.
    _assert_first_cycle_sleeps(_lifecycle_sleeps(default_sleeper_calls), spread=False)
    default_sleeper_calls.clear()


def test_dirty_tree_hold_without_sleeper_is_instant(
    tmp_path: Path, default_sleeper_calls: list[tuple[str, float]]
) -> None:
    started = time.monotonic()
    result = _idle_lifecycle(tmp_path, dirty=True).run_once()
    elapsed = time.monotonic() - started

    assert result.status is LifecycleStatus.IDLE
    assert elapsed < INSTANT_ENOUGH
    _assert_first_cycle_sleeps(_lifecycle_sleeps(default_sleeper_calls), spread=True)
    default_sleeper_calls.clear()


def test_production_default_sleeper_calls_the_real_time_sleep_binding(
    monkeypatch, tmp_path: Path, default_sleeper_calls: list[tuple[str, float]]
) -> None:
    """Without an injected sleeper the agent still really sleeps in production.

    Restores the production ``_default_sleeper`` helper (undoing the suite
    stub for this test only, mirroring how the claim-hardening tests
    override the settle stub) and records what the module ``time`` binding
    receives, so the assertion proves the call-time lookup reaches the real
    sleep with the startup spread and the jittered idle duration.
    """

    monkeypatch.setattr(
        lifecycle_module, "_default_sleeper", lifecycle_module._default_sleeper.__wrapped__
    )
    calls: list[float] = []
    monkeypatch.setattr(
        lifecycle_module, "time", types.SimpleNamespace(sleep=calls.append)
    )

    result = _idle_lifecycle(tmp_path, dirty=True).run_once()

    assert result.status is LifecycleStatus.IDLE
    _assert_first_cycle_sleeps(calls, spread=True)
    assert default_sleeper_calls == []
