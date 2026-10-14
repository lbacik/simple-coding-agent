"""Versioned, read-only telemetry snapshots served over the control socket.

A managed instance reports itself with a schema-v1 telemetry snapshot: its
control state, the current (or last finished) implementation attempt's
figures, and its most recent events. The dashboard only reads the snapshot;
this module owns its assembly. The contract (shape, versioning rules, size
bound, and ``agentctl telemetry --json`` mechanics) is published in
``docs/agents/telemetry-snapshot.md``; the transport decision (control
socket, not HTTP) is ADR 0006.

``current_attempt`` is present from claim (``begin_attempt``) until the
attempt is finalised (``finish_attempt`` or, equivalently, a snapshot whose
control section shows no active attempt). Its configured limits come from
the seed (claim-time values) falling back to the reporter construction
defaults; live figures stay ``unavailable`` until later tickets publish
them. Building the section never raises into the attempt: a failure logs
once and keeps the previous view.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
import copy
import json
import logging
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


logger = logging.getLogger(__name__)


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


def _figure_configured(value: object) -> dict:
    """A configured figure, or unavailable when no value is known."""

    if value is None:
        return {"value": None, "basis": "unavailable", "reason": "not configured"}
    return {"value": value, "basis": "configured"}


def _figure_unavailable(reason: str = "not yet published") -> dict:
    """An explicitly unavailable figure (never zero)."""

    return {"value": None, "basis": "unavailable", "reason": reason}


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _seed_int(candidate: object) -> int | None:
    """Extract an int from a bare value or a figure object."""

    if isinstance(candidate, dict):
        return _as_int(candidate.get("value"))
    return _as_int(candidate)


def _seed_str(candidate: object) -> str | None:
    if isinstance(candidate, dict):
        inner = candidate.get("value")
        return inner if isinstance(inner, str) else None
    return candidate if isinstance(candidate, str) else None


def _normalize_seed(seed: object) -> dict:
    """Normalize a ``begin_attempt`` seed to the stored claim-time view.

    Accepts flat keys (``attempt_id``, ``issue_number``, ``issue_title``,
    ``continuation_index``, ``continuation_max``, ``hard_ceiling``,
    ``turns_max``, ``timeout_seconds``) with common aliases
    (``max_budget_tokens`` for the hard ceiling, ``max_turns`` for the turn
    limit, ``model_timeout`` for the timeout, ``max_auto_continuations`` for
    the continuation max), plus nested shapes (``continuation: {index, max}``,
    ``budget: {hard_ceiling}``, ``turns: {max}``) where each leaf may be a
    bare value or a figure object. Raises ``ValueError`` on a missing or
    malformed attempt identity.
    """

    if not isinstance(seed, dict):
        raise ValueError("telemetry seed must be a mapping")
    source: dict = dict(seed)

    continuation = source.get("continuation")
    budget = source.get("budget")
    turns = source.get("turns")
    if not isinstance(continuation, dict):
        continuation = None
    if not isinstance(budget, dict):
        budget = None
    if not isinstance(turns, dict):
        turns = None

    def _first(*candidates: object) -> object:
        for candidate in candidates:
            if candidate is not None:
                return candidate
        return None

    attempt_id = _seed_str(
        _first(source.get("attempt_id"), source.get("attemptId"), source.get("started_at"))
    )
    issue_number_raw = _first(
        source.get("issue_number"), source.get("issueNumber"), source.get("issue")
    )
    if isinstance(issue_number_raw, dict):
        issue_number_raw = issue_number_raw.get("value")
    issue_title = _seed_str(
        _first(
            source.get("issue_title"),
            source.get("issueTitle"),
            source.get("title"),
            (continuation or {}).get("issue_title") if continuation else None,
        )
    )
    continuation_index_raw = _first(
        source.get("continuation_index"),
        source.get("continuationIndex"),
        source.get("index"),
        (continuation or {}).get("index") if continuation else None,
    )
    continuation_max_raw = _first(
        source.get("continuation_max"),
        source.get("continuationMax"),
        source.get("max_auto_continuations"),
        source.get("max"),
        (continuation or {}).get("max") if continuation else None,
    )
    hard_ceiling_raw = _first(
        source.get("hard_ceiling"),
        source.get("hardCeiling"),
        source.get("max_budget_tokens"),
        source.get("maxBudgetTokens"),
        (budget or {}).get("hard_ceiling") if budget else None,
    )
    turns_max_raw = _first(
        source.get("turns_max"),
        source.get("turnsMax"),
        source.get("max_turns"),
        source.get("maxTurns"),
        (turns or {}).get("max") if turns else None,
    )
    timeout_raw = _first(
        source.get("timeout_seconds"),
        source.get("timeoutSeconds"),
        source.get("model_timeout"),
        source.get("timeout"),
    )

    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("telemetry seed must carry a non-empty attempt_id")
    if isinstance(issue_number_raw, bool) or not isinstance(issue_number_raw, int):
        # Accept numeric strings for convenience, but never booleans.
        try:
            issue_number = int(str(issue_number_raw).strip(), 10)
        except (TypeError, ValueError) as error:
            raise ValueError("telemetry seed must carry an issue_number") from error
    else:
        issue_number = issue_number_raw
    if issue_number <= 0:
        raise ValueError("telemetry seed must carry a positive issue_number")

    continuation_index = _seed_int(continuation_index_raw)
    if continuation_index_raw is not None and continuation_index is None:
        raise ValueError("telemetry continuation index must be an integer or None")
    if continuation_index is not None and continuation_index < 0:
        raise ValueError("telemetry continuation index must be non-negative")
    continuation_max = _seed_int(continuation_max_raw)
    if continuation_max_raw is not None and continuation_max is None:
        raise ValueError("telemetry continuation max must be an integer or None")
    if continuation_max is not None and continuation_max < 0:
        raise ValueError("telemetry continuation max must be non-negative")
    hard_ceiling = _seed_int(hard_ceiling_raw)
    if hard_ceiling_raw is not None and hard_ceiling is None:
        raise ValueError("telemetry hard ceiling must be an integer or None")
    turns_max = _seed_int(turns_max_raw)
    if turns_max_raw is not None and turns_max is None:
        raise ValueError("telemetry turns max must be an integer or None")
    timeout_seconds = _seed_int(timeout_raw)
    if timeout_raw is not None and timeout_seconds is None:
        raise ValueError("telemetry timeout must be an integer or None")

    return {
        "attempt_id": attempt_id,
        "issue_number": issue_number,
        "issue_title": issue_title,
        "continuation_index": continuation_index,
        "continuation_max": continuation_max,
        "hard_ceiling": hard_ceiling,
        "turns_max": turns_max,
        "timeout_seconds": timeout_seconds,
    }


class TelemetryReporter:
    """Assemble schema-v1 snapshots behind the reader interface.

    Owns the process-local identity (recorded at startup), the process-local
    sequence (reset with the process), the size bound, and the claim-time
    ``current_attempt`` view (set by :meth:`begin_attempt`, cleared by
    :meth:`finish_attempt`). Configured limits passed at construction are
    defaults; seed values from the claim override them per attempt.
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
        hard_ceiling: int | None = None,
        turns_max: int | None = None,
        timeout_seconds: int | None = None,
        continuation_max: int | None = None,
        max_budget_tokens: int | None = None,
        max_turns: int | None = None,
        model_timeout: int | None = None,
        max_auto_continuations: int | None = None,
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
        self._attempt_lock = threading.Lock()
        self._configured = {
            "hard_ceiling": hard_ceiling if hard_ceiling is not None else max_budget_tokens,
            "turns_max": turns_max if turns_max is not None else max_turns,
            "timeout_seconds": timeout_seconds if timeout_seconds is not None else model_timeout,
            "continuation_max": (
                continuation_max if continuation_max is not None else max_auto_continuations
            ),
        }
        self._current_seed: dict | None = None
        self._last_good_current: dict | None = None
        self._build_failed_logged = False

    def begin_attempt(self, seed: dict) -> None:
        """Seed the ``current_attempt`` view at claim; never raises.

        A malformed seed logs once and keeps the previous view so a
        telemetry failure can never break the claim it observes.
        """

        try:
            normalized = _normalize_seed(seed)
        except Exception as error:
            with self._attempt_lock:
                if not self._build_failed_logged:
                    self._build_failed_logged = True
                    logger.warning("telemetry begin_attempt failed: %s; keeping previous view", error)
            return
        with self._attempt_lock:
            self._current_seed = normalized
            self._build_failed_logged = False

    def finish_attempt(self) -> None:
        """Clear the ``current_attempt`` view at finalisation; never raises."""

        try:
            with self._attempt_lock:
                self._current_seed = None
                self._last_good_current = None
                self._build_failed_logged = False
        except Exception as error:  # pragma: no cover - lock failure
            logger.warning("telemetry finish_attempt failed: %s", error)

    def snapshot(self, control: TelemetryControl) -> dict:
        """Build one schema-v1 snapshot; raise when the control section is missing.

        ``SnapshotFailedError`` from the control reader propagates unchanged
        so the socket handler can answer ``snapshot_failed``. Building the
        ``current_attempt`` section never raises: a failure logs once and
        keeps the previous view.
        """

        # The control section first: only a snapshot that is actually served
        # consumes a sequence number.
        section = control.telemetry_control_snapshot()
        with self._sequence_lock:
            self._sequence += 1
            sequence = self._sequence
        current: dict | None
        try:
            current = self._build_current_attempt()
        except Exception as error:
            with self._attempt_lock:
                if not self._build_failed_logged:
                    self._build_failed_logged = True
                    logger.warning(
                        "telemetry current_attempt build failed: %s; keeping previous view",
                        error,
                    )
                current = copy.deepcopy(self._last_good_current) if self._last_good_current is not None else None
        else:
            with self._attempt_lock:
                self._build_failed_logged = False
                self._last_good_current = copy.deepcopy(current) if current is not None else None
        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _format_timestamp(self._clock()),
            "sequence": sequence,
            "identity": dict(self._identity),
            "control": section,
            "current_attempt": current,
            "last_finished": None,
            "recent_events": [],
            "truncated": False,
        }
        return apply_size_cap(snapshot)

    def _build_current_attempt(self) -> dict | None:
        with self._attempt_lock:
            seed = copy.deepcopy(self._current_seed) if self._current_seed is not None else None
            configured = dict(self._configured)
        if seed is None:
            return None
        continuation_index = seed.get("continuation_index")
        continuation_max = seed.get("continuation_max")
        if continuation_max is None:
            continuation_max = configured.get("continuation_max")
        hard_ceiling = seed.get("hard_ceiling")
        if hard_ceiling is None:
            hard_ceiling = configured.get("hard_ceiling")
        turns_max = seed.get("turns_max")
        if turns_max is None:
            turns_max = configured.get("turns_max")
        timeout_seconds = seed.get("timeout_seconds")
        if timeout_seconds is None:
            timeout_seconds = configured.get("timeout_seconds")

        if continuation_index is not None and (
            isinstance(continuation_index, bool)
            or not isinstance(continuation_index, int)
            or continuation_index < 0
        ):
            raise ValueError("stored continuation index is invalid")
        for name, value in (
            ("continuation_max", continuation_max),
            ("hard_ceiling", hard_ceiling),
            ("turns_max", turns_max),
            ("timeout_seconds", timeout_seconds),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"stored {name} is invalid")

        attempt: dict = {
            "attempt_id": seed["attempt_id"],
            "issue_number": seed["issue_number"],
            "observed_at": _format_timestamp(self._clock()),
            "continuation": {
                "index": (
                    _figure_configured(continuation_index)
                    if continuation_index is not None
                    else _figure_unavailable()
                ),
                "max": (
                    _figure_configured(continuation_max)
                    if continuation_max is not None
                    else _figure_unavailable()
                ),
            },
            "budget": {
                "used": _figure_unavailable(),
                "hard_ceiling": (
                    _figure_configured(hard_ceiling)
                    if hard_ceiling is not None
                    else _figure_unavailable()
                ),
            },
            "turns": {
                "value": None,
                "basis": "unavailable",
                "reason": "not yet published",
                "max": (
                    _figure_configured(turns_max)
                    if turns_max is not None
                    else _figure_unavailable()
                ),
            },
            "timeout_seconds": (
                _figure_configured(timeout_seconds)
                if timeout_seconds is not None
                else _figure_unavailable()
            ),
            "elapsed_seconds": _figure_unavailable(),
            "provider_rate_limit": _figure_unavailable("not exposed through the gateway"),
        }
        issue_title = seed.get("issue_title")
        if issue_title is not None:
            attempt["issue_title"] = issue_title
        return attempt


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
