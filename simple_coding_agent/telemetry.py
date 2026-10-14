"""Versioned, read-only telemetry snapshots served over the control socket.

A managed instance reports itself with a schema-v1 telemetry snapshot: its
control state, the current (or last finished) implementation attempt's
figures, and its most recent events. The dashboard only reads the snapshot;
this module owns its assembly. The contract (shape, versioning rules, size
bound, and ``agentctl telemetry --json`` mechanics) is published in
``docs/agents/telemetry-snapshot.md``; the transport decision (control
socket, not HTTP) is ADR 0006.

Tracer scope: identity and control are real. ``current_attempt`` and
``last_finished`` are ``None`` and ``recent_events`` is empty until later
tickets fill them with the writer side (``begin_attempt``,
``publish_attempt``, ``finish_attempt``).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
import json
import threading
from typing import Protocol


#: Telemetry snapshot schema major version served by this process.
SCHEMA_VERSION = 1

#: Serialised snapshot cap: a quarter of the control socket's 1 MiB reply
#: limit. Over the cap the snapshot is trimmed (oldest events first, then
#: subagents) and served with ``truncated: true``.
SNAPSHOT_SIZE_LIMIT_BYTES = 256 * 1024

#: How long snapshot assembly waits for the control lock before reusing the
#: last built control section. Telemetry never holds up other control
#: commands: it only ever waits, never holds, the lock.
CONTROL_LOCK_TIMEOUT_SECONDS = 1.0

#: Connect+read timeout ``agentctl telemetry`` uses so it always answers
#: inside the dashboard's per-instance budget. Other commands keep theirs.
TELEMETRY_TIMEOUT_SECONDS = 3.0

#: Minimum gap between two ``telemetry_failed`` events: a persistently
#: failing snapshotter must not flood the log.
TELEMETRY_FAILURE_COOLDOWN_SECONDS = 60.0


class SnapshotFailedError(RuntimeError):
    """Raised when a telemetry snapshot cannot be built right now."""


class TelemetryControl(Protocol):
    """The reader interface snapshot assembly needs from the live agent."""

    def telemetry_control_snapshot(self) -> dict:
        """Return the network-free control section for one snapshot.

        The same payload ``status`` builds, but without workspace or GitHub
        I/O: publication fields that would need a lookup stay ``None`` ("not
        known yet"). Carries ``control_observed_at`` saying when the section
        was built. Raises :class:`SnapshotFailedError` when no section has
        been built yet.
        """
        ...


class TelemetryReporter:
    """Assemble schema-v1 snapshots behind the reader interface.

    Owns the process-local identity (recorded at startup), the process-local
    sequence (reset with the process), and the size bound. The ``control``
    argument of :meth:`snapshot` supplies the live control section; later
    tickets add the writer side that fills ``current_attempt``,
    ``last_finished``, and ``recent_events``.
    """

    def __init__(
        self,
        *,
        repository: str,
        agent_version: str,
        sdk_version: str,
        cli_version: str,
        process_started_at: str,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._identity = {
            "repository": repository,
            "agent_version": agent_version,
            "sdk_version": sdk_version,
            "cli_version": cli_version,
            "process_started_at": process_started_at,
        }
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sequence = 0
        self._sequence_lock = threading.Lock()

    def snapshot(self, control: TelemetryControl) -> dict:
        """Build one schema-v1 snapshot; raise when the control section is missing.

        ``SnapshotFailedError`` from the control reader propagates unchanged
        so the socket handler can answer ``snapshot_failed``.
        """

        with self._sequence_lock:
            self._sequence += 1
            sequence = self._sequence
        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _format_timestamp(self._clock()),
            "sequence": sequence,
            "identity": dict(self._identity),
            "control": control.telemetry_control_snapshot(),
            "current_attempt": None,
            "last_finished": None,
            "recent_events": [],
            "truncated": False,
        }
        return apply_size_cap(snapshot)


def apply_size_cap(snapshot: dict) -> dict:
    """Trim an assembled snapshot to the serialised size bound, in place.

    Over the cap the snapshot drops ``recent_events`` oldest first, then
    ``current_attempt.context.subagents`` oldest first, and reports
    ``truncated: true``. Under the cap the snapshot is returned untouched.
    """

    if _serialised_size(snapshot) <= SNAPSHOT_SIZE_LIMIT_BYTES:
        return snapshot
    snapshot["truncated"] = True
    events = snapshot.get("recent_events")
    while (
        isinstance(events, list)
        and events
        and _serialised_size(snapshot) > SNAPSHOT_SIZE_LIMIT_BYTES
    ):
        events.pop(0)
    attempt = snapshot.get("current_attempt")
    while _serialised_size(snapshot) > SNAPSHOT_SIZE_LIMIT_BYTES:
        if not isinstance(attempt, dict):
            break
        context = attempt.get("context")
        if not isinstance(context, dict):
            break
        subagents = context.get("subagents")
        if not isinstance(subagents, list) or not subagents:
            break
        subagents.pop(0)
    return snapshot


def format_timestamp(value: datetime) -> str:
    """Render a timestamp the way the snapshot records them (``Z``-suffixed)."""

    return _format_timestamp(value)


def _format_timestamp(value: datetime) -> str:
    parsed = value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return parsed.isoformat().replace("+00:00", "Z")


def _serialised_size(snapshot: dict) -> int:
    return len(json.dumps(snapshot, sort_keys=True).encode("utf-8"))
