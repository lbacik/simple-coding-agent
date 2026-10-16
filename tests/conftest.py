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
