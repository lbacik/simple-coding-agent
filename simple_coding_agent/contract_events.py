"""Operator-contract major-1 Contract Events projection (``recent_events``).

Every backend's history reads the same. The structured event logger keeps
emitting native engine events to stdout unchanged; this module maps each
redacted native record onto one Contract Event (spec section 10):

- native events that correspond to a Contract Event (``issue_claimed``,
  ``attempt_phase_transitioned``, and the engine-lifecycle natives emitted
  alongside them) produce that Contract Event;
- every other engine event is exposed only as ``native.<event>``.

A Contract Event is ``{at, level, event, attempt_id?, phase?, text}`` with
``level`` in ``debug|info|warning|error`` and ``text`` at most 300
characters, redacted. The process-local :class:`ContractEventsBuffer` ring
holds the newest 50 Contract Events of the current process and feeds
``status.recent_events``; each Attempt's Contract Events are additionally
appended to ``events.jsonl`` in its archive directory (manifest kind
``events``). The main process's stdout is not part of the contract.
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
from typing import Any, Mapping
import threading


#: How many Contract Events the process-local ring keeps, newest last.
#: In-memory only, so it starts empty after a restart.
CONTRACT_EVENTS_LIMIT = 50

#: How many characters of an event's text survive into a Contract Event.
CONTRACT_EVENT_TEXT_LIMIT = 300

#: Contract Event names (spec section 10). ``sealed`` arrives with the
#: sealing ticket; this backend never emits it yet, but the mapper already
#: passes a native ``sealed`` record through unchanged.
CONTRACT_EVENT_NAMES = frozenset(
    {
        "process_started",
        "startup_check",
        "ready",
        "halted",
        "claim_opened",
        "phase_changed",
        "outcome_recorded",
        "attempt_finished",
        "attempt_released",
        "sealed",
        "command_accepted",
        "command_resolved",
    }
)

#: Native claim events: each opens the same Claim, so each maps to one
#: ``claim_opened``. A prioritized claim emits both ``next_issue_claimed``
#: (at the priority boundary) and ``issue_claimed`` (at the claim
#: boundary) for a single Attempt; the buffer keeps the first
#: ``claim_opened`` per Attempt ID so one Claim never opens twice.
_CLAIM_NATIVE_EVENTS = frozenset({"issue_claimed", "next_issue_claimed"})

#: Native event for the phase-transition Contract Event.
_PHASE_NATIVE_EVENT = "attempt_phase_transitioned"

#: Native levels onto the contract vocabulary (validated, never invented).
_LEVELS = {
    "DEBUG": "debug",
    "INFO": "info",
    "WARNING": "warning",
    "ERROR": "error",
}

#: Extra fields each Contract Event may carry (spec section 10). Unknown
#: detail keys are dropped, never promoted.
_EXTRA_KEYS = {
    "process_started": ("agent_version",),
    "startup_check": ("check", "result"),
    "ready": (),
    "halted": ("reason",),
    "claim_opened": ("attempt_id", "issue"),
    "phase_changed": ("attempt_id", "phase"),
    "outcome_recorded": ("attempt_id", "outcome", "reason"),
    "attempt_finished": ("attempt_id",),
    "attempt_released": ("attempt_id",),
    "sealed": ("attempt_id",),
    "command_accepted": ("request_id", "kind", "acknowledgement"),
    "command_resolved": ("request_id", "acknowledgement"),
}


def parse_detail_fields(detail: object) -> dict[str, str]:
    """Parse a ``key=value; key=value`` native detail into its fields.

    Only the first ``=`` splits a pair, so values keep their own ``=``;
    pairs without ``=`` are dropped. Never raises.
    """

    fields: dict[str, str] = {}
    if not isinstance(detail, str) or not detail:
        return fields
    try:
        for part in detail.split(";"):
            key, separator, value = part.partition("=")
            if not separator:
                continue
            key = key.strip()
            if not key or key in fields:
                continue
            fields[key] = value.strip()
    except Exception:
        return {}
    return fields


def _contract_level(level: object) -> str:
    """Map a native log level onto the contract vocabulary, defaulting to info."""

    if isinstance(level, str):
        lowered = _LEVELS.get(level.upper(), level.lower())
        if lowered in ("debug", "info", "warning", "error"):
            return lowered
    return "info"


def _event_at(record: Mapping[str, Any]) -> str:
    """The Contract Event timestamp: the redacted native record's own time."""

    timestamp = record.get("timestamp")
    if isinstance(timestamp, str) and timestamp:
        return timestamp
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _event_text(detail: object) -> str:
    """The Contract Event text: the redacted detail, capped at 300 characters."""

    text = detail if isinstance(detail, str) else str(detail)
    return text[:CONTRACT_EVENT_TEXT_LIMIT]


def to_contract_event(
    record: Mapping[str, Any], *, attempt_id: str | None = None
) -> dict[str, Any]:
    """Map one redacted native log record onto one Contract Event.

    ``attempt_id`` is the stable attempt identity resolved by the caller
    (the archive's ``started_at`` for the record's issue, if any); it is
    included whenever it names a real attempt. ``phase`` is included only
    for ``phase_changed`` (parsed from the native detail), never the
    logger's own phase label. Extra fields follow the spec table; unknown
    detail keys are dropped. Never raises.
    """

    try:
        name = record.get("event")
        if not isinstance(name, str) or not name:
            name = "unknown"
        clean_attempt: str | None = (
            attempt_id if isinstance(attempt_id, str) and attempt_id else None
        )
        fields = parse_detail_fields(record.get("detail"))
        event: dict[str, Any] = {
            "at": _event_at(record),
            "level": _contract_level(record.get("level")),
            "event": name,
            "text": _event_text(record.get("detail", "")),
        }
        if name in _CLAIM_NATIVE_EVENTS:
            event["event"] = "claim_opened"
            if clean_attempt is not None:
                event["attempt_id"] = clean_attempt
            issue = record.get("issue_number")
            if isinstance(issue, int) and not isinstance(issue, bool) and issue > 0:
                event["issue"] = issue
            return event
        if name == _PHASE_NATIVE_EVENT:
            event["event"] = "phase_changed"
            if clean_attempt is not None:
                event["attempt_id"] = clean_attempt
            phase = fields.get("phase")
            if isinstance(phase, str) and phase:
                event["phase"] = phase
            return event
        if name in CONTRACT_EVENT_NAMES:
            event["event"] = name
            if clean_attempt is not None:
                event["attempt_id"] = clean_attempt
            for key in _EXTRA_KEYS.get(name, ()):
                if key == "attempt_id":
                    continue  # resolved above, never parsed from free text
                value = fields.get(key)
                if isinstance(value, str) and value:
                    event[key] = value
            return event
        event["event"] = f"native.{name}"
        if clean_attempt is not None:
            event["attempt_id"] = clean_attempt
        return event
    except Exception:
        return {
            "at": _event_at({} if not isinstance(record, Mapping) else record),
            "level": "info",
            "event": "native.unknown",
            "text": "",
        }


class ContractEventsBuffer:
    """Process-local ring buffer of the newest Contract Events.

    Fed by the JSON event logger after redaction, so every buffered entry
    has passed through the redactor. Keeps the last ``max_events`` entries,
    newest last. The first ``claim_opened`` per Attempt ID wins: a
    prioritized claim emits two native claim events for one Claim, and one
    Claim must never open twice. Thread-safe: the logger emits from
    several threads while the status reader copies concurrently. Never
    raises: contract observability must not break logging.
    """

    def __init__(self, max_events: int = CONTRACT_EVENTS_LIMIT) -> None:
        self._entries: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._claimed_attempts: set[str] = set()
        self._lock = threading.Lock()

    def record(self, contract_event: dict[str, Any]) -> bool:
        """Append one Contract Event, newest last; ``False`` when deduped."""

        try:
            name = contract_event.get("event")
            attempt = contract_event.get("attempt_id")
            with self._lock:
                if (
                    name == "claim_opened"
                    and isinstance(attempt, str)
                    and attempt
                ):
                    if attempt in self._claimed_attempts:
                        return False
                    self._claimed_attempts.add(attempt)
                self._entries.append(dict(contract_event))
                return True
        except Exception:
            return False

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a copy of the buffered events, newest last."""

        try:
            with self._lock:
                return [dict(entry) for entry in self._entries]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Native detail builders: the single encoding the mapper parses back.
# ---------------------------------------------------------------------------


def process_started_detail(agent_version: str) -> str:
    """Detail for the ``process_started`` native event."""

    return f"agent_version={agent_version}"


def startup_check_detail(check: str, result: str) -> str:
    """Detail for a ``startup_check`` native event (``result`` started/ok/skipped/failed)."""

    return f"check={check}; result={result}"


def halted_detail(reason: str) -> str:
    """Detail for the ``halted`` native event."""

    return f"reason={reason}"


def outcome_recorded_detail(outcome: str) -> str:
    """Detail for the ``outcome_recorded`` native event."""

    return f"outcome={outcome}"


def command_accepted_detail(request_id: str, kind: str, acknowledgement: str) -> str:
    """Detail for a ``command_accepted`` native event."""

    return f"request_id={request_id}; kind={kind}; acknowledgement={acknowledgement}"


def command_resolved_detail(request_id: str, acknowledgement: str) -> str:
    """Detail for a ``command_resolved`` native event."""

    return f"request_id={request_id}; acknowledgement={acknowledgement}"
