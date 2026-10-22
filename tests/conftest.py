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

import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

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
