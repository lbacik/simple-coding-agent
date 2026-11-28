"""Test-wide fixtures.

macOS resolves ``TMPDIR`` to a long, per-test-name path
(``/private/var/folders/.../pytest-of-<user>/pytest-N/<test-id>/``), which
easily exceeds the 104-byte ``sun_path`` limit once a control socket file is
appended (see ``simple_coding_agent.control_server``). Control-socket tests
need a short, fixed-length temp directory regardless of test name length, so
this overrides pytest's built-in ``tmp_path`` fixture for the whole suite
with one rooted directly under ``/tmp``.
"""

from __future__ import annotations

import functools
import shutil
import sys
import tempfile
import time as real_time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest

from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace

# Repository automation helpers (e.g. scripts/release_labels.py) are plain
# modules, not part of the product package. Put scripts/ on the import path
# so unit tests (and the release backfill script later) import the same rule.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(dir="/tmp", prefix="sca-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def workspace(tmp_path: Path) -> InMemoryWorkspace:
    """A fresh shared workspace double rooted under the test's temp dir."""

    return InMemoryWorkspace(tmp_path / "repo")


@pytest.fixture
def tracker() -> FakeTracker:
    """A fresh shared tracker double with no seeded issues."""

    return FakeTracker()


@pytest.fixture
def publisher() -> FakePublisher:
    """A fresh shared publisher double with default publication behaviour."""

    return FakePublisher()


@pytest.fixture(autouse=True)
def _instant_claim_settle(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep the claim settle-window verification instant suite-wide.

    Production always waits :data:`CLAIM_SETTLE_DELAY_SECONDS
    <simple_coding_agent.github_tracker.CLAIM_SETTLE_DELAY_SECONDS>` between
    assigning and re-reading; the dedicated hardening tests record the call
    and assert the delay. Every other test replaces only the tracker's own
    ``time`` binding with an instant sleep so the suite never waits out the
    window (and unrelated ``time.sleep`` users, e.g. telemetry timing, are
    untouched).
    """

    import types

    import simple_coding_agent.github_tracker as tracker_module

    stub = types.SimpleNamespace(sleep=lambda seconds: None)
    monkeypatch.setattr(tracker_module, "time", stub)
    yield


@pytest.fixture
def default_sleeper_calls() -> Iterator[list[tuple[str, float]]]:
    """Per-test record of default-sleeper invocations.

    The suite-wide instant stubs (:func:`_instant_default_sleep`) append
    ``(source, seconds)`` here instead of waiting, where ``source`` is one of
    ``"lifecycle"``, ``"publication"`` or ``"git_workspace"``. Tests that
    deliberately drive an unstubbed default sleep assert on this record and
    then clear it; anything left over at teardown fails the test, so a new
    production sleep on a tested path shows up as a failure instead of a
    silent slowdown. Tests that inject their own ``sleeper=`` never touch
    this record.
    """

    yield []


def _instant_stub_for(
    source: str,
    record: list[tuple[str, float]],
    production: Callable[[float], None],
) -> Callable[[float], None]:
    """Build the recording instant stub standing in for one default sleeper."""

    @functools.wraps(production)
    def _instant_sleep(seconds: float) -> None:
        record.append((source, seconds))

    return _instant_sleep


@pytest.fixture(autouse=True)
def _instant_default_sleep(
    monkeypatch: pytest.MonkeyPatch, default_sleeper_calls: list[tuple[str, float]]
) -> Iterator[None]:
    """Keep the product default sleepers instant suite-wide.

    ``AgentLifecycle``, ``Publisher`` and ``GitWorkspace`` resolve their
    default sleeper at construction through their module-level
    ``_default_sleeper`` binding (production still really sleeps). Replacing
    the binding keeps every test that builds those classes without
    ``sleeper=`` instant, while tests that inject their own sleeper keep
    recording exact durations exactly as before. The stub wraps production
    (``functools.wraps``), so ``_default_sleeper.__wrapped__`` still reaches
    the real helper for the production-wiring test.
    """

    import simple_coding_agent.git_workspace as git_workspace_module
    import simple_coding_agent.lifecycle as lifecycle_module
    import simple_coding_agent.publication as publication_module

    for module, source in (
        (lifecycle_module, "lifecycle"),
        (publication_module, "publication"),
        (git_workspace_module, "git_workspace"),
    ):
        monkeypatch.setattr(
            module,
            "_default_sleeper",
            _instant_stub_for(source, default_sleeper_calls, module._default_sleeper),
        )
    yield
    if default_sleeper_calls:
        pytest.fail(
            f"unexpected default-sleeper call(s) {default_sleeper_calls!r}: "
            "inject a sleeper=... or assert-and-clear default_sleeper_calls; "
            "the suite must never wait in real time"
        )


class _RealSleepGuard:
    """Per-module ``time`` stand-in that fails on ``sleep``.

    Delegates every other attribute (``monotonic``, ``time``, ...) to the
    real ``time`` module, so clock reads keep working; only waiting fails.
    ``pytest.fail`` raises ``BaseException``, so it propagates even through
    broad product ``except`` handlers instead of blocking the suite. Hits are
    also recorded so a swallowed failure still fails the test at teardown.
    """

    def __init__(
        self, module_name: str, hits: list[tuple[str, float]], clock: ModuleType
    ) -> None:
        self._module_name = module_name
        self._hits = hits
        self._clock = clock

    def __getattr__(self, name: str) -> object:
        return getattr(self._clock, name)

    def sleep(self, seconds: float) -> None:
        self._hits.append((self._module_name, seconds))
        pytest.fail(
            f"real time.sleep({seconds!r}) reached from {self._module_name}: "
            "stub the seam or inject a sleeper so the suite never waits in real time"
        )


@pytest.fixture(autouse=True)
def _guard_unstubbed_product_sleep(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail instead of waiting when product code reaches a real sleep.

    Every already-imported ``simple_coding_agent`` module whose ``time``
    binding is still the standard library gets a stand-in whose ``sleep``
    fails the test while every clock read delegates to the real module, so
    telemetry timing and other non-sleep users keep working and
    ``time.sleep`` itself is never stubbed globally. Two modules opt out
    explicitly: ``github_tracker`` (its settle and retry sleeps are already
    instant through :func:`_instant_claim_settle`) and
    ``attempt_processes`` (process-termination grace periods intentionally
    sleep for real and tests use short durations). Only modules using
    ``import time`` are covered, which is the only import style in the
    package today; a new direct ``time.sleep`` anywhere covered fails
    loudly instead of slowing the suite.
    """

    skipped = {
        "simple_coding_agent.github_tracker",
        "simple_coding_agent.attempt_processes",
    }
    hits: list[tuple[str, float]] = []
    for name, module in list(sys.modules.items()):
        if name != "simple_coding_agent" and not name.startswith("simple_coding_agent."):
            continue
        if name in skipped:
            continue
        if vars(module).get("time") is not real_time:
            continue
        monkeypatch.setattr(module, "time", _RealSleepGuard(name, hits, real_time))
    yield
    if hits:
        pytest.fail(
            f"real time.sleep call(s) reached from product code {hits!r}: "
            "stub the seam or inject a sleeper so the suite never waits in real time"
        )
