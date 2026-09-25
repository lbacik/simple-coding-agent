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


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(dir="/tmp", prefix="sca-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
