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
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace


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
