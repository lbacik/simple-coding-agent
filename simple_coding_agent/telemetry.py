"""Versioned, read-only telemetry snapshots served over the control socket.

A managed instance reports itself with a schema-v1 telemetry snapshot: its
control state, the current (or last finished) implementation attempt's
figures, and its most recent events. The dashboard only reads the snapshot;
this module owns its assembly. The contract (shape, versioning rules, size
bound, and ``agentctl telemetry --json`` mechanics) is published in
``docs/agents/telemetry-snapshot.md``; the transport decision (control
socket, not HTTP) is ADR 0006.

``current_attempt`` is present from claim (``begin_attempt``) until the
attempt is finalised (``finish_attempt``). The lifecycle calls ``begin`` at
every claim boundary (including resuming a retained attempt after a
restart) and ``finish`` at every finalization boundary (including the
startup-leftover path), so the stored seed alone decides the section: a
set seed means an active attempt. Its configured limits come from the seed
(claim-time values) falling back to the reporter construction defaults;
its live budget, token, turn and cache figures come from the immutable
attempt views the executor publishes via ``publish_attempt_view`` after
every limit check. Until the first view arrives they stay ``unavailable``.
Building the section never raises into the attempt: a failure logs once
per episode and keeps the previous view.

``last_finished`` reports the last finished attempt (issue, title, outcome,
tokens, turns, finish time and pull request) between attempts and alongside
a running one. The lifecycle passes its summary to ``finish_attempt`` at
finalisation, and seeds it from disk at startup (the latest completed
attempt archive record) so it survives restarts. Archives written before
the pull-request and title fields existed seed ``pull_request: None`` and
an ``"unknown"`` title.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
import copy
import json
import logging
from pathlib import Path
import threading
from typing import Protocol

from simple_coding_agent.observability import read_latest_completed_attempt


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

#: Minimum measured main-thread responses before the prompt-cache hit rate
#: is reported as ``measured``; below it the figure is ``unavailable``
#: (never zero) while ``counted`` still reports the responses seen so far.
#: Mirrors ``model_execution.PROMPT_CACHE_MIN_MEASURED_RESPONSES`` without
#: importing the SDK-bound executor module on the listener thread.
CACHE_MIN_MEASURED_RESPONSES = 20

#: Reason carried by a hit-rate figure with too few measured responses.
CACHE_NOT_ENOUGH_RESPONSES_REASON = "not enough measured responses"

#: Reason carried by a hit-rate figure with no measured input to score.
CACHE_NO_MEASURED_INPUT_REASON = "no measured input to score"

#: Token categories published under ``tokens_by_category`` (measured only).
TOKEN_CATEGORIES = (
    "input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "output_tokens",
)

#: Bases a successful context-usage window reading may carry. Anything else
#: on the view is treated as ``unavailable`` so a malformed publisher can
#: never smuggle an unknown basis into the snapshot.
CONTEXT_WINDOW_BASES = ("sdk_assumed", "configured")

#: Maximum telemetry subagent entries in one snapshot section (running
#: first). The executor caps its views at the same size; the reporter
#: re-enforces it so an oversized view never reaches the socket.
MAX_CONTEXT_SUBAGENTS = 20


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


def _plain_dict(view: object) -> dict:
    """Copy a published attempt view into plain JSON-safe containers.

    The executor publishes an immutable mapping; the copy detaches the
    stored view from whatever the publisher holds so later mutation on
    either side cannot leak into the served snapshot. Raises ``ValueError``
    when the view is not a mapping.
    """

    if not isinstance(view, dict):
        from collections.abc import Mapping as _Mapping

        if not isinstance(view, _Mapping):
            raise ValueError("published attempt view must be a mapping")
        view = dict(view)
    plain = _plain_value(view)
    if not isinstance(plain, dict):
        raise ValueError("published attempt view must be a mapping")
    return plain


def _plain_value(value: object) -> object:
    from collections.abc import Mapping as _Mapping

    if isinstance(value, _Mapping):
        return {key: _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


def _view_tokens(view: dict, key: str) -> int:
    """A rounded token total from a published view (never negative)."""

    value = view.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"published attempt view {key} must be a number")
    rounded = int(round(value))
    if rounded < 0:
        raise ValueError(f"published attempt view {key} must be non-negative")
    return rounded


def _view_int(view: dict, key: str) -> int:
    """A non-negative integer from a published view."""

    value = view.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"published attempt view {key} must be an integer")
    if value < 0:
        raise ValueError(f"published attempt view {key} must be non-negative")
    return value


def _view_category(categories: dict, category: str) -> int:
    """One measured token category from a published view."""

    value = categories.get(category)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"published attempt view tokens_by_category[{category}] must be a non-negative integer"
        )
    return value


def _view_main_context(view: dict) -> int:
    """Main-thread context from a published view (ledger reading, rounded).

    Missing keys (views published before context landed) read as zero so
    older publishers still build; a malformed value raises ``ValueError``.
    """

    if "main_context_tokens" not in view:
        return 0
    return _view_tokens(view, "main_context_tokens")


def _view_main_peak_context(view: dict) -> int:
    """Peak main-thread context from a published view (rounded)."""

    if "main_peak_context_tokens" not in view:
        # Views published before the peak landed carry no peak; fall back to
        # the current main reading so the section stays consistent.
        return _view_main_context(view)
    return _view_tokens(view, "main_peak_context_tokens")


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


def _seed_or_configured(seed: dict, configured: dict, key: str) -> int | None:
    """A claim-time seed value falling back to the construction default."""

    value = seed.get(key)
    if value is None:
        value = configured.get(key)
    return value  # type: ignore[return-value]


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


def _measured_figure(candidate: object) -> dict:
    """A measured figure, or unavailable when no trustworthy value is known."""

    value = candidate.get("value") if isinstance(candidate, dict) else candidate
    if isinstance(value, bool):
        value = None
    elif isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int) or value < 0:
        return {"value": None, "basis": "unavailable", "reason": "not reported"}
    return {"value": value, "basis": "measured"}


def _context_window_figure(view: dict, prefix: str) -> dict:
    """One window/threshold figure from a published view.

    A present non-negative integer with an ``sdk_assumed``/``configured``
    basis passes through; anything else (missing value, unknown basis, or
    an explicit ``unavailable`` basis) is ``unavailable`` with the view's
    reason (or a default). Raises ``ValueError`` on a malformed basis or
    reason so the caller keeps the last good section.
    """

    value = view.get(prefix)
    basis = view.get(f"{prefix}_basis", "unavailable")
    reason = view.get(f"{prefix}_reason")
    if basis in CONTEXT_WINDOW_BASES:
        value = _as_int(value)
        if value is None or value < 0:
            raise ValueError(f"published attempt view {prefix} must be a non-negative integer")
        return {"value": value, "basis": basis}
    if basis != "unavailable":
        raise ValueError(f"published attempt view {prefix}_basis must be sdk_assumed, configured or unavailable")
    if reason is not None and not isinstance(reason, str):
        raise ValueError(f"published attempt view {prefix}_reason must be a string or None")
    return {
        "value": None,
        "basis": "unavailable",
        "reason": reason if isinstance(reason, str) and reason else "not yet published",
    }


def _compaction_token_figure(value: object) -> dict:
    """One compaction pre/post figure (measured when present)."""

    value = _as_int(value)
    if value is not None and value >= 0:
        return {"value": value, "basis": "measured"}
    return {"value": None, "basis": "unavailable", "reason": "not reported"}


def _build_compactions(view: dict) -> dict:
    """The ``context.compactions`` section from a published view.

    Missing keys (views published before context landed) read as zero
    compactions; a malformed count or entry raises ``ValueError``.
    """

    count = view.get("compaction_count", 0)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("published attempt view compaction_count must be a non-negative integer")
    raw_last = view.get("compaction_last")
    if raw_last is None:
        return {"count": count, "last": None}
    if not isinstance(raw_last, dict):
        raise ValueError("published attempt view compaction_last must be a mapping or None")
    at = raw_last.get("at")
    trigger = raw_last.get("trigger")
    if not isinstance(at, str) or not at:
        raise ValueError("published attempt view compaction_last.at must be a non-empty string")
    if not isinstance(trigger, str) or not trigger:
        raise ValueError("published attempt view compaction_last.trigger must be a non-empty string")
    return {
        "count": count,
        "last": {
            "at": at,
            "trigger": trigger,
            "pre_tokens": _compaction_token_figure(raw_last.get("pre_tokens")),
            "post_tokens": _compaction_token_figure(raw_last.get("post_tokens")),
        },
    }


def _view_peak_basis(view: dict, main_basis: str) -> str:
    """Basis of the peak main-thread context from a published view.

    Views published before the peak basis landed fall back to the current
    main basis; a present-but-unknown basis raises ``ValueError`` so the
    caller keeps the last good section.
    """

    peak_basis = view.get("main_peak_context_basis", main_basis)
    if peak_basis not in ("measured", "estimated"):
        raise ValueError("published attempt view main_peak_context_basis must be measured or estimated")
    return peak_basis


def _build_subagents(view: dict) -> list[dict]:
    """The ``context.subagents`` list from a published view, running first.

    Each entry is ``{id, type, context (estimated), running}``; the list is
    capped at 20 with running entries first (stable within each group).
    Missing views read as empty; malformed entries raise ``ValueError``.
    """

    raw = view.get("subagents", [])
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("published attempt view subagents must be a list")
    entries: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("published attempt view subagents entries must be mappings")
        agent_id = item.get("id")
        agent_type = item.get("type", "unknown")
        running = item.get("running")
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("published attempt view subagents id must be a non-empty string")
        if not isinstance(agent_type, str) or not agent_type:
            agent_type = "unknown"
        if not isinstance(running, bool):
            raise ValueError("published attempt view subagents running must be a boolean")
        context_value = item.get("context", 0)
        if isinstance(context_value, bool):
            raise ValueError("published attempt view subagents context must be a number")
        if isinstance(context_value, float):
            context_value = int(round(context_value))
        if not isinstance(context_value, int) or context_value < 0:
            raise ValueError("published attempt view subagents context must be non-negative")
        entries.append(
            {
                "id": agent_id,
                "type": agent_type,
                "context": {"value": context_value, "basis": "estimated"},
                "running": running,
            }
        )
    running_first = [entry for entry in entries if entry["running"]]
    finished = [entry for entry in entries if not entry["running"]]
    return (running_first + finished)[:MAX_CONTEXT_SUBAGENTS]


def _normalize_pull_request(candidate: object) -> dict | None:
    """A ``{number, url}`` pull request, or ``None`` when there is none."""

    if candidate is None:
        return None
    if isinstance(candidate, dict):
        number = candidate.get("number")
        url = candidate.get("url")
    else:
        number = getattr(candidate, "number", None)
        url = getattr(candidate, "url", None)
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        return None
    if not isinstance(url, str) or not url:
        return None
    return {"number": number, "url": url}


def _normalize_last_finished(summary: object) -> dict:
    """Normalize a ``finish_attempt`` summary to the stored ``last_finished`` view.

    Accepts bare token/turn counts (or measured figure objects) plus an
    optional ``{number, url}`` pull request. Archives written before the
    title and pull-request fields existed normalize to an ``"unknown"``
    title and a ``None`` pull request. Raises ``ValueError`` on a missing
    or malformed attempt identity, outcome, or finish time.
    """

    if not isinstance(summary, dict):
        raise ValueError("last_finished summary must be a mapping")
    attempt_id = summary.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("last_finished must carry a non-empty attempt_id")
    issue_number_raw = summary.get("issue_number")
    if isinstance(issue_number_raw, bool) or not isinstance(issue_number_raw, int):
        try:
            issue_number = int(str(issue_number_raw).strip(), 10)
        except (TypeError, ValueError) as error:
            raise ValueError("last_finished must carry an issue_number") from error
    else:
        issue_number = issue_number_raw
    if issue_number <= 0:
        raise ValueError("last_finished must carry a positive issue_number")
    issue_title = summary.get("issue_title")
    if not isinstance(issue_title, str) or not issue_title:
        issue_title = "unknown"
    outcome = summary.get("outcome")
    if hasattr(outcome, "value"):
        outcome = outcome.value  # type: ignore[union-attr]
    if not isinstance(outcome, str) or not outcome:
        raise ValueError("last_finished must carry an outcome")
    finished_at = summary.get("finished_at", summary.get("completed_at"))
    if not isinstance(finished_at, str) or not finished_at:
        raise ValueError("last_finished must carry a finished_at timestamp")
    return {
        "attempt_id": attempt_id,
        "issue_number": issue_number,
        "issue_title": issue_title,
        "outcome": outcome,
        "finished_at": finished_at,
        "budget_used": _measured_figure(summary.get("budget_used")),
        "turns": _measured_figure(summary.get("turns")),
        "pull_request": _normalize_pull_request(summary.get("pull_request")),
    }


def load_last_finished(data_dir: Path) -> dict | None:
    """Seed ``last_finished`` from disk: the latest completed attempt archive.

    Returns the normalized ``last_finished`` view built from that attempt's
    archive record (token budget and pull request included), or ``None``
    when no completed attempt is on disk. Skips unreadable records.
    """

    found = read_latest_completed_attempt(data_dir)
    if found is None:
        return None
    issue_number, record = found
    archived_issue = record.get("issue_number")
    if (
        isinstance(archived_issue, int)
        and not isinstance(archived_issue, bool)
        and archived_issue > 0
    ):
        issue_number = archived_issue
    token_budget = record.get("token_budget")
    budget_used = None
    turns = None
    if isinstance(token_budget, dict):
        budget_used = token_budget.get("actual_tokens")
        turns = token_budget.get("turns")
    return _normalize_last_finished(
        {
            "attempt_id": record.get("started_at"),
            "issue_number": issue_number,
            "issue_title": record.get("issue_title"),
            "outcome": record.get("outcome"),
            "finished_at": record.get("completed_at"),
            "budget_used": budget_used,
            "turns": turns,
            "pull_request": record.get("pull_request"),
        }
    )


class TelemetryReporter:
    """Assemble schema-v1 snapshots behind the reader interface.

    Owns the process-local identity (recorded at startup), the process-local
    sequence (reset with the process), the size bound, the claim-time
    ``current_attempt`` view (set by :meth:`begin_attempt`, cleared by
    :meth:`finish_attempt`), and the ``last_finished`` view (set by
    :meth:`finish_attempt` with a summary, seeded from disk at startup).
    Configured limits passed at construction are defaults; seed values from
    the claim override them per attempt.
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
        cache_min_measured_responses: int = CACHE_MIN_MEASURED_RESPONSES,
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
        self._last_finished: dict | None = None
        self._build_failed_logged = False
        self._cache_min_measured_responses = cache_min_measured_responses
        # The latest immutable live attempt view published by the executor.
        # Swapped by reference under ``_attempt_view_lock`` (held only for
        # the swap); the listener thread serves this copy and never reads
        # the token ledger.
        self._attempt_view: dict | None = None
        self._attempt_view_lock = threading.Lock()

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
        # A new attempt starts with no live figures until the executor
        # publishes its first view after a limit check.
        try:
            with self._attempt_view_lock:
                self._attempt_view = None
        except Exception as error:  # pragma: no cover - lock failure
            logger.warning("telemetry begin_attempt failed: %s", error)

    def finish_attempt(self, summary: dict | None = None) -> None:
        """Record the last finished attempt at finalisation; never raises.

        Always clears the ``current_attempt`` view. When ``summary`` carries
        the finished attempt (attempt id, issue, title, outcome, finish time,
        token budget, turns, pull request) it is normalized and stored as
        ``last_finished``; without one the previous ``last_finished`` is
        kept (so the startup-leftover path never wipes the disk seed). A
        malformed summary logs once and keeps the previous view.
        """

        try:
            normalized = (
                _normalize_last_finished(summary) if summary is not None else None
            )
        except Exception as error:
            with self._attempt_lock:
                if not self._build_failed_logged:
                    self._build_failed_logged = True
                    logger.warning(
                        "telemetry finish_attempt failed: %s; keeping previous view",
                        error,
                    )
            return
        try:
            with self._attempt_lock:
                self._current_seed = None
                self._last_good_current = None
                if normalized is not None:
                    self._last_finished = normalized
                self._build_failed_logged = False
        except Exception as error:  # pragma: no cover - lock failure
            logger.warning("telemetry finish_attempt failed: %s", error)
        try:
            with self._attempt_view_lock:
                self._attempt_view = None
        except Exception as error:  # pragma: no cover - lock failure
            logger.warning("telemetry finish_attempt failed: %s", error)

    def publish_attempt_view(self, view: object) -> None:
        """Store one immutable live attempt view from the executor; never raises.

        The view carries the executor's raw live readings (budget totals,
        threshold state, measured categories, turns, cache and elapsed
        figures); this method stamps ``observed_at`` and swaps the stored
        reference. Only the swap holds ``_attempt_view_lock``: the copy and
        the timestamp are prepared outside it, and the listener thread later
        serves the stored copy without ever reading the token ledger.
        After model execution ends no further view arrives, so the served
        figures — including ``observed_at`` — stay frozen until
        :meth:`finish_attempt`.
        """

        try:
            staged = _plain_dict(view)
            staged["observed_at"] = _format_timestamp(self._clock())
            with self._attempt_view_lock:
                self._attempt_view = staged
        except Exception as error:
            logger.warning(
                "telemetry publish_attempt_view failed: %s; keeping previous view", error
            )

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
        with self._attempt_lock:
            last_finished = (
                copy.deepcopy(self._last_finished)
                if self._last_finished is not None
                else None
            )
        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _format_timestamp(self._clock()),
            "sequence": sequence,
            "identity": dict(self._identity),
            "control": section,
            "current_attempt": current,
            "last_finished": last_finished,
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
        with self._attempt_view_lock:
            view = copy.deepcopy(self._attempt_view) if self._attempt_view is not None else None
        if view is None:
            return self._tracer_attempt(seed, configured)
        return self._live_attempt(seed, configured, view)

    def _tracer_attempt(self, seed: dict, configured: dict) -> dict:
        continuation_index = seed.get("continuation_index")
        continuation_max = _seed_or_configured(seed, configured, "continuation_max")
        hard_ceiling = _seed_or_configured(seed, configured, "hard_ceiling")
        turns_max = _seed_or_configured(seed, configured, "turns_max")
        timeout_seconds = _seed_or_configured(seed, configured, "timeout_seconds")

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
            "context": {
                "main": _figure_unavailable(),
                "main_peak": _figure_unavailable(),
                "window": _figure_unavailable(),
                "auto_compact_threshold": _figure_unavailable(),
                "compactions": {"count": 0, "last": None},
                "subagents": [],
            },
            "provider_rate_limit": _figure_unavailable("not exposed through the gateway"),
        }
        issue_title = seed.get("issue_title")
        if issue_title is not None:
            attempt["issue_title"] = issue_title
        return attempt

    def _live_attempt(self, seed: dict, configured: dict, view: dict) -> dict:
        """Build the ``current_attempt`` section from a published live view.

        The seed still owns identity and configured limits; the view owns
        every live figure and ``observed_at`` (stamped at publication, so a
        frozen view stays frozen). Raises ``ValueError`` on a malformed
        view so the caller keeps the last good section.
        """

        continuation_index = seed.get("continuation_index")
        continuation_max = _seed_or_configured(seed, configured, "continuation_max")
        hard_ceiling = _seed_or_configured(seed, configured, "hard_ceiling")
        turns_max = _seed_or_configured(seed, configured, "turns_max")
        timeout_seconds = _seed_or_configured(seed, configured, "timeout_seconds")
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

        observed_at = view.get("observed_at")
        if not isinstance(observed_at, str) or not observed_at:
            raise ValueError("published attempt view must carry observed_at")
        used = _view_tokens(view, "budget_tokens")
        measured = _view_tokens(view, "measured_tokens")
        estimated = _view_tokens(view, "estimated_tokens")
        hard_ceiling_reached = view.get("hard_ceiling_reached")
        if not isinstance(hard_ceiling_reached, bool):
            raise ValueError("published attempt view hard_ceiling_reached must be a boolean")

        soft_value = _view_int(view, "soft_threshold_value")
        rule = view.get("soft_threshold_rule")
        if rule not in ("percentage", "reserve"):
            raise ValueError("published attempt view soft_threshold_rule must be percentage or reserve")
        percentage = view.get("soft_threshold_percentage")
        if isinstance(percentage, bool) or not isinstance(percentage, (int, float)):
            raise ValueError("published attempt view soft_threshold_percentage must be a number")
        reserve_turns = _view_int(view, "soft_threshold_reserve_turns")
        crossed = view.get("soft_threshold_crossed")
        if not isinstance(crossed, bool):
            raise ValueError("published attempt view soft_threshold_crossed must be a boolean")
        crossed_at_tokens = view.get("soft_threshold_crossed_at_tokens")
        if crossed_at_tokens is not None and (
            isinstance(crossed_at_tokens, bool) or not isinstance(crossed_at_tokens, int)
        ):
            raise ValueError("published attempt view soft_threshold_crossed_at_tokens must be an integer or None")
        crossed_at = view.get("soft_threshold_crossed_at")
        if crossed_at is not None and not isinstance(crossed_at, str):
            raise ValueError("published attempt view soft_threshold_crossed_at must be a string or None")
        main_basis = view.get("main_context_basis")
        if main_basis not in ("measured", "estimated"):
            main_basis = "estimated"
        soft_basis = "configured" if rule == "percentage" else main_basis

        categories = view.get("tokens_by_category")
        if not isinstance(categories, dict):
            raise ValueError("published attempt view tokens_by_category must be a mapping")
        tokens_by_category = {}
        for category in TOKEN_CATEGORIES:
            tokens_by_category[category] = {
                "value": _view_category(categories, category),
                "basis": "measured",
            }

        turns = _view_int(view, "turns")
        hit_rate = view.get("cache_hit_rate")
        if hit_rate is not None and (
            isinstance(hit_rate, bool) or not isinstance(hit_rate, (int, float))
        ):
            raise ValueError("published attempt view cache_hit_rate must be a number or None")
        counted = _view_int(view, "cache_counted")
        elapsed_seconds = _view_int(view, "elapsed_seconds")
        if counted < 0 or elapsed_seconds < 0 or turns < 0:
            raise ValueError("published attempt view counts must be non-negative")

        if counted >= self._cache_min_measured_responses and hit_rate is not None:
            main_hit_rate: dict = {
                "value": float(hit_rate),
                "basis": "measured",
                "counted": counted,
            }
        elif hit_rate is None and counted >= self._cache_min_measured_responses:
            main_hit_rate = {
                "value": None,
                "basis": "unavailable",
                "reason": CACHE_NO_MEASURED_INPUT_REASON,
                "counted": counted,
            }
        else:
            main_hit_rate = {
                "value": None,
                "basis": "unavailable",
                "reason": CACHE_NOT_ENOUGH_RESPONSES_REASON,
                "counted": counted,
            }

        attempt: dict = {
            "attempt_id": seed["attempt_id"],
            "issue_number": seed["issue_number"],
            "observed_at": observed_at,
            "budget": {
                "used": {
                    "value": used,
                    "basis": "measured" if estimated == 0 else "estimated",
                },
                "measured": {"value": measured, "basis": "measured"},
                "estimated": {"value": estimated, "basis": "estimated"},
                "hard_ceiling": (
                    _figure_configured(hard_ceiling)
                    if hard_ceiling is not None
                    else _figure_unavailable()
                ),
                "soft_threshold": {
                    "value": soft_value,
                    "basis": soft_basis,
                    "rule": rule,
                    "percentage": percentage,
                    "reserve_turns": reserve_turns,
                    "crossed": crossed,
                    "crossed_at_tokens": crossed_at_tokens,
                    "crossed_at": crossed_at,
                },
                "hard_ceiling_reached": hard_ceiling_reached,
            },
            "tokens_by_category": tokens_by_category,
            "turns": {
                "value": turns,
                "basis": "measured",
                "max": (
                    _figure_configured(turns_max)
                    if turns_max is not None
                    else _figure_unavailable()
                ),
            },
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
            "cache": {"main_hit_rate": main_hit_rate},
            "context": {
                "main": {
                    "value": _view_main_context(view),
                    "basis": main_basis,
                },
                "main_peak": {
                    "value": _view_main_peak_context(view),
                    "basis": _view_peak_basis(view, main_basis),
                },
                "window": _context_window_figure(view, "context_window"),
                "auto_compact_threshold": _context_window_figure(
                    view, "auto_compact_threshold"
                ),
                "compactions": _build_compactions(view),
                "subagents": _build_subagents(view),
            },
            "provider_rate_limit": _figure_unavailable("not exposed through the gateway"),
            "elapsed_seconds": {"value": elapsed_seconds, "basis": "measured"},
            "timeout_seconds": (
                _figure_configured(timeout_seconds)
                if timeout_seconds is not None
                else _figure_unavailable()
            ),
        }
        issue_title = seed.get("issue_title")
        if issue_title is not None:
            attempt["issue_title"] = issue_title
        return attempt


def apply_size_cap(snapshot: dict) -> dict:
    """Trim an assembled snapshot to the serialised size bound, in place.

    Over the cap the snapshot drops ``recent_events`` oldest first, then
    ``current_attempt.context.subagents`` from the tail (the list is
    running-first, so the tail is the oldest finished entry), and reports
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
        subagents.pop()
    return snapshot


def format_timestamp(value: datetime) -> str:
    """Render a timestamp the way the snapshot records them (``Z``-suffixed)."""

    return _format_timestamp(value)


def _format_timestamp(value: datetime) -> str:
    parsed = value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return parsed.isoformat().replace("+00:00", "Z")


def _serialised_size(snapshot: dict) -> int:
    return len(json.dumps(snapshot, sort_keys=True).encode("utf-8"))
