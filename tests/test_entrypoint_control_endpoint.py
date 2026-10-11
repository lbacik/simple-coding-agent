"""Startup failure path: control endpoint unavailable must log and exit 1."""

from __future__ import annotations

import io
import json

import pytest

from simple_coding_agent import __main__ as entrypoint
from simple_coding_agent.observability import JsonEventLogger


class _FailingControlServer:
    def __init__(self, error: Exception) -> None:
        self._error = error
        self.start_called = False

    def start(self) -> None:
        self.start_called = True
        raise self._error


class _SucceedingControlServer:
    def __init__(self) -> None:
        self.start_called = False

    def start(self) -> None:
        self.start_called = True


def _read_single_record(stream: io.StringIO) -> dict:
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1, (
        f"expected exactly one JSON line, got {len(lines)}: {lines!r}"
    )
    return json.loads(lines[0])


def test_control_endpoint_failure_logs_event_and_exits_1() -> None:
    stream = io.StringIO()
    logger = JsonEventLogger(stream)
    underlying = OSError(30, "Read-only file system")
    server = _FailingControlServer(underlying)

    with pytest.raises(SystemExit) as exc_info:
        entrypoint._start_control_endpoint(logger, server)

    assert exc_info.value.code == 1
    assert server.start_called
    record = _read_single_record(stream)
    assert record["event"] == "control_endpoint_unavailable"
    assert record["level"] == "ERROR"
    assert record["phase"] == "startup"
    assert "Read-only file system" in record["detail"]


def test_control_endpoint_start_success_emits_nothing() -> None:
    stream = io.StringIO()
    logger = JsonEventLogger(stream)
    server = _SucceedingControlServer()

    entrypoint._start_control_endpoint(logger, server)

    assert server.start_called
    assert stream.getvalue() == ""
