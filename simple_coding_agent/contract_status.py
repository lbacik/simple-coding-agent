"""Operator contract major-1 ``status`` projection (``StatusSnapshot``).

A dashboard renders a running instance from one ``agentctl --contract 1
status`` call. This module projects the engine's prepared state — the
control store, the telemetry reporter and the attempt checkpoint — onto
the contract's Status Snapshot without network, git or provider I/O:

- ``generated_at`` plus the engine ``identity`` (from
  :meth:`lifecycle.control_identity <...>`, lock-free),
- ``instance`` (``idle``/``active``/``draining``/``paused``, derived from
  intake and the active attempt; ``starting`` while construction is
  unfinished and ``recovering`` while reconciliation or a recovery hold
  is active, with ``startup_check`` and ``last_halt``),
- ``intake`` (open/closed with ``pending_stop``, ``stop_plan`` and
  ``priority``),
- the current attempt (native phase, issue, start time, last activity,
  budget and figures, each with ``observed_at``),
- the last finished attempt, recent commands and recent events.

Recent commands reuse the contract ``CommandRecord`` shape the mutation
tickets serve (acknowledgement ``not fulfilled`` renamed to
``not_fulfilled``, ``arguments`` in the contract shapes). Recent events
are the process-local Contract Events ring (newest 50, redacted, text
capped at 300 characters), served oldest-first like the telemetry ring.

Every figure carries a ``unit`` and a contract ``basis``
(``measured``/``estimated``/``configured``/``assumed``/``unavailable``/``not_applicable``;
telemetry's ``sdk_assumed`` is reported as ``assumed``). A value the
engine does not know is ``unavailable`` — never a measured zero — and a
concept this backend lacks is ``not_applicable``. Required budget figures
and the enforced-ceiling limits are always present. The snapshot is
capped at 256 KiB (oldest events, then optional figures, then oldest
commands) with ``truncated: true``.
"""

from __future__ import annotations

import json

from simple_coding_agent import telemetry as telemetry_module
from simple_coding_agent.control import CommandAcknowledgement, IntakeState


#: Operator-contract acknowledgement names (native ``not fulfilled`` renamed).
_ACKNOWLEDGEMENTS = {
    CommandAcknowledgement.ACCEPTED.value: "accepted",
    CommandAcknowledgement.COMPLETED.value: "completed",
    CommandAcknowledgement.SUPERSEDED.value: "superseded",
    CommandAcknowledgement.NOT_FULFILLED.value: "not_fulfilled",
}

#: Pending commands that close intake immediately (shown as ``pending_stop``).
_PENDING_STOP_KINDS = ("stop", "handoff")

#: Display labels for the native terminal outcomes.
_OUTCOME_LABELS = {
    "complete": "Complete",
    "incomplete": "Incomplete",
    "infrastructure_error": "Infrastructure error",
    "no_changes": "No changes",
    "handoff": "Handoff",
}

#: Optional figure names served in ``current_attempt.figures`` (fixed order).
_OPTIONAL_FIGURE_ORDER = (
    "cache_hit_rate",
    "context_used",
    "context_peak",
    "context_window",
    "turns",
    "tool_calls",
    "continuations",
    "review_seats_running",
    "elapsed_seconds",
    "native.input_tokens",
    "native.cache_read_input_tokens",
    "native.cache_creation_input_tokens",
    "native.output_tokens",
    "native.soft_threshold_tokens",
    "native.auto_compact_threshold",
    "native.compactions",
    "native.continuation_max",
    "native.timeout_seconds",
)

#: Basis vocabulary of the contract (telemetry ``sdk_assumed`` maps to ``assumed``).
_FIGURE_BASES = (
    "measured",
    "estimated",
    "configured",
    "assumed",
    "unavailable",
    "not_applicable",
)


def _telemetry_basis(basis: object) -> str:
    """Map a telemetry figure basis onto the contract vocabulary."""

    if basis == "sdk_assumed":
        return "assumed"
    if isinstance(basis, str) and basis in _FIGURE_BASES:
        return basis
    return "unavailable"


def _figure(
    value: object, unit: str, basis: str, reason: str | None = None
) -> dict:
    """Build one contract figure; unknown values carry a ``null`` value."""

    if basis in ("unavailable", "not_applicable"):
        figure: dict = {"value": None, "unit": unit, "basis": basis}
        if reason:
            figure["reason"] = reason
        return figure
    return {"value": value, "unit": unit, "basis": basis}


def _configured_figure(value: object, unit: str) -> dict:
    """A configured figure, or unavailable when no value is known."""

    if value is None:
        return _figure(None, unit, "unavailable", "not configured")
    return _figure(value, unit, "configured")


def _unavailable_figure(unit: str, reason: str = "not yet published") -> dict:
    """An explicitly unavailable figure (never a measured zero)."""

    return _figure(None, unit, "unavailable", reason)


def _telemetry_figure(section: object, unit: str) -> dict:
    """Project one telemetry ``{value, basis, reason?}`` figure, adding its unit."""

    if not isinstance(section, dict):
        return _unavailable_figure(unit)
    basis = _telemetry_basis(section.get("basis"))
    if basis in ("unavailable", "not_applicable"):
        reason = section.get("reason")
        return _figure(
            None, unit, basis, reason if isinstance(reason, str) else None
        )
    return _figure(section.get("value"), unit, basis)


def _turns_used(turns: object) -> dict:
    """Project the telemetry turns reading (a bare value with a basis)."""

    if not isinstance(turns, dict):
        return _unavailable_figure("count")
    value = turns.get("value")
    if isinstance(value, dict):
        return _telemetry_figure(value, "count")
    if isinstance(value, int) and not isinstance(value, bool):
        return _figure(value, "count", _telemetry_basis(turns.get("basis")))
    reason = turns.get("reason")
    return _figure(
        None,
        "count",
        "unavailable",
        reason if isinstance(reason, str) else "not yet published",
    )


def _one_line(detail: object) -> str:
    """Render a command detail on a single line (the contract requires it)."""

    return " ".join(str(detail).splitlines())


# ---------------------------------------------------------------------------
# Instance and intake
# ---------------------------------------------------------------------------


def project_instance_state(
    intake: IntakeState,
    *,
    recovering: bool,
    attempt_active: bool,
    starting: bool = False,
) -> str:
    """Derive the contract instance state from intake and the active attempt.

    ``starting`` (the engine has opened its control endpoint but has not
    finished construction) wins over every other state. ``recovering``
    covers startup reconciliation while recorded effects of earlier
    attempts finish, and intake held for recovery (an unusable checkpoint
    or completion ledger, or retained work awaiting a recovery command).
    Otherwise native ``running`` is open (``idle`` without an attempt,
    ``active`` with one); ``stopping`` while an attempt is active is
    closed (``draining``); ``stopped`` is closed (``paused``).
    """

    if starting:
        return "starting"
    if recovering:
        return "recovering"
    if intake is IntakeState.RUNNING:
        return "active" if attempt_active else "idle"
    return "draining" if attempt_active else "paused"


def project_intake(
    *,
    intake: IntakeState,
    pending_record: object,
    stop_plan: dict | None,
    next_issue: dict | None,
    observed_at: str,
) -> dict:
    """Project the native intake state, pending stop, stop plan and priority."""

    pending_stop: dict | None = None
    if pending_record is not None:
        kind = getattr(pending_record, "kind", None)
        if kind in _PENDING_STOP_KINDS:
            pending_stop = {
                "request_id": pending_record.request_id,
                "sequence": pending_record.sequence,
            }
    plan: dict | None = None
    if stop_plan is not None:
        requested = stop_plan.get("requested")
        remaining = stop_plan.get("remaining")
        if isinstance(requested, int) and isinstance(remaining, int):
            # The contract's ``remaining`` is the plan's N as requested (the
            # reference producer and the producer suite read it that way);
            # only ``counted`` advances. The store's own ``remaining`` is the
            # count still to go, so ``counted`` is the difference.
            plan = {
                "request_id": stop_plan.get("request_id"),
                "remaining": requested,
                "counted": max(0, requested - remaining),
            }
    priority: dict | None = None
    if next_issue is not None:
        priority = {
            "request_id": next_issue.get("request_id"),
            "issue": next_issue.get("issue_number"),
        }
    return {
        "state": "open" if intake is IntakeState.RUNNING else "closed",
        "pending_stop": pending_stop,
        "stop_plan": plan,
        "priority": priority,
        "observed_at": observed_at,
    }


# ---------------------------------------------------------------------------
# Current attempt: budget and figures
# ---------------------------------------------------------------------------


def _limit_entry(name: str, used: dict, limit: dict) -> dict:
    return {"name": name, "used": used, "limit": limit}


def _budget_limits(
    current: dict | None, *, usd_ceiling: int | None
) -> list[dict]:
    """The enforced-ceiling limits: token budget, USD backstop, turn limit."""

    budget = current.get("budget") if isinstance(current, dict) else None
    budget = budget if isinstance(budget, dict) else {}
    turns = current.get("turns") if isinstance(current, dict) else None
    turns = turns if isinstance(turns, dict) else {}

    raw_used = _telemetry_figure(budget.get("used"), "tokens")
    raw_limit = _telemetry_figure(budget.get("hard_ceiling"), "tokens")
    cost_used = _unavailable_figure("usd", "no live cost figure is published")
    cost_limit = _configured_figure(usd_ceiling, "usd")
    turns_used = _turns_used(turns)
    turns_limit = _telemetry_figure(turns.get("max"), "count")
    return [
        _limit_entry("raw_tokens", raw_used, raw_limit),
        _limit_entry("cost_usd", cost_used, cost_limit),
        _limit_entry("turns", turns_used, turns_limit),
    ]


def _project_figures(current: dict | None) -> dict:
    """Project telemetry figures onto the contract's optional figure names.

    Backend-only figures go under ``native.<name>``. Every figure carries
    a unit; unknown values are ``unavailable`` and absent concepts are
    ``not_applicable``, never a measured zero.
    """

    section = current if isinstance(current, dict) else {}
    budget = section.get("budget")
    budget = budget if isinstance(budget, dict) else {}
    cache = section.get("cache")
    cache = cache if isinstance(cache, dict) else {}
    context = section.get("context")
    context = context if isinstance(context, dict) else {}
    continuation = section.get("continuation")
    continuation = continuation if isinstance(continuation, dict) else {}
    categories = section.get("tokens_by_category")
    categories = categories if isinstance(categories, dict) else {}

    figures: dict = {}
    figures["cache_hit_rate"] = _telemetry_figure(
        cache.get("main_hit_rate"), "ratio"
    )
    figures["context_used"] = _telemetry_figure(context.get("main"), "tokens")
    figures["context_peak"] = _telemetry_figure(context.get("main_peak"), "tokens")
    figures["context_window"] = _telemetry_figure(context.get("window"), "tokens")
    figures["turns"] = _turns_used(section.get("turns"))
    figures["tool_calls"] = _unavailable_figure("count", "tool calls are not counted")
    figures["continuations"] = _telemetry_figure(
        continuation.get("index"), "count"
    )
    figures["review_seats_running"] = _figure(
        None, "count", "not_applicable", "subagents are generic, not review seats"
    )
    figures["elapsed_seconds"] = _telemetry_figure(
        section.get("elapsed_seconds"), "seconds"
    )
    for category in (
        "input_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "output_tokens",
    ):
        figures[f"native.{category}"] = _telemetry_figure(
            categories.get(category), "tokens"
        )
    soft = budget.get("soft_threshold")
    if isinstance(soft, dict) and isinstance(soft.get("value"), int):
        figures["native.soft_threshold_tokens"] = _telemetry_figure(
            {"value": soft.get("value"), "basis": soft.get("basis")}, "tokens"
        )
    else:
        figures["native.soft_threshold_tokens"] = _unavailable_figure("tokens")
    figures["native.auto_compact_threshold"] = _telemetry_figure(
        context.get("auto_compact_threshold"), "tokens"
    )
    compactions = context.get("compactions")
    if isinstance(compactions, dict) and isinstance(compactions.get("count"), int):
        figures["native.compactions"] = _figure(
            compactions.get("count"), "count", "measured"
        )
    else:
        figures["native.compactions"] = _unavailable_figure("count")
    figures["native.continuation_max"] = _telemetry_figure(
        continuation.get("max"), "count"
    )
    figures["native.timeout_seconds"] = _telemetry_figure(
        section.get("timeout_seconds"), "seconds"
    )
    return figures


def project_current_attempt(
    *,
    attempt: dict | None,
    current: dict | None,
    usd_ceiling: int | None,
    observed_at: str,
) -> dict | None:
    """Project the active attempt, or ``None`` when no attempt is active.

    ``attempt`` carries the checkpoint identity (attempt id, issue, native
    phase, start and last-activity times); ``current`` is the telemetry
    current-attempt section (or ``None`` before the first claim seeding).
    """

    if attempt is None:
        return None
    budget_section = current.get("budget") if isinstance(current, dict) else None
    if isinstance(budget_section, dict):
        raw_used = _telemetry_figure(budget_section.get("used"), "tokens")
    else:
        raw_used = _unavailable_figure("tokens")
    budget = {
        "raw_tokens": raw_used,
        "effective_tokens": dict(raw_used),
        "enforced_tokens": "raw",
        "limits": _budget_limits(current, usd_ceiling=usd_ceiling),
    }
    figures = _project_figures(current)
    issue_title = attempt.get("issue_title")
    return {
        "attempt_id": attempt.get("attempt_id"),
        "issue": {
            "number": attempt.get("issue_number"),
            "title": issue_title if isinstance(issue_title, str) else None,
        },
        "native_phase": attempt.get("native_phase"),
        "started_at": attempt.get("started_at"),
        "last_activity_at": attempt.get("last_activity_at"),
        "observed_at": observed_at,
        "budget": budget,
        "figures": figures,
    }


# ---------------------------------------------------------------------------
# Last finished attempt
# ---------------------------------------------------------------------------


def project_last_finished(summary: dict | None) -> dict | None:
    """Project the telemetry last-finished summary onto an AttemptSummary."""

    if summary is None:
        return None
    attempt_id = summary.get("attempt_id")
    issue_number = summary.get("issue_number")
    title = summary.get("issue_title")
    if title == "unknown":
        title = None
    outcome = summary.get("outcome")
    outcome = outcome if isinstance(outcome, str) else "unknown"
    pull_request = summary.get("pull_request")
    if (
        isinstance(pull_request, dict)
        and isinstance(pull_request.get("number"), int)
        and isinstance(pull_request.get("url"), str)
    ):
        delivery: dict | None = {
            "pull_request": {
                "number": pull_request["number"],
                "url": pull_request["url"],
            }
        }
    else:
        delivery = None
    finished_at = summary.get("finished_at")
    lifecycle_state = summary.get("lifecycle_state", "finished")
    if lifecycle_state not in ("finished", "released"):
        lifecycle_state = "finished"
    return {
        "attempt_id": attempt_id,
        "issue": {
            "number": issue_number,
            "title": title if isinstance(title, str) else None,
        },
        "started_at": attempt_id,
        "finished_at": finished_at if isinstance(finished_at, str) else None,
        "lifecycle_state": lifecycle_state,
        "outcome": {
            "native": outcome,
            "reason": None,
            "label": _OUTCOME_LABELS.get(outcome, outcome),
        },
        "delivery": delivery,
        "attention_required": outcome == "handoff",
        "sealed": True,
        "sealed_at": finished_at if isinstance(finished_at, str) else None,
    }


# ---------------------------------------------------------------------------
# Recent commands
# ---------------------------------------------------------------------------


_ARGUMENT_KEYS = {
    "stop": (),
    "stop_after": ("after",),
    "resume": (),
    "next_issue": ("issue",),
    "handoff": (),
    "recovery_retry": ("attempt_id",),
    "recovery_release": ("attempt_id", "saved_at"),
    "errors_reset": (),
}


def project_command(record: object, payload: dict, *, repository: str) -> dict:
    """Project one durable command onto the contract CommandRecord shape."""

    kind = getattr(record, "kind", "")
    acknowledgement = getattr(getattr(record, "acknowledgement", ""), "value", "")
    keys = _ARGUMENT_KEYS.get(kind, ())
    arguments = {
        key: payload[key] for key in keys if key in payload
    }
    return {
        "request_id": record.request_id,
        "sequence": record.sequence,
        "kind": kind,
        "arguments": arguments,
        "repository": repository,
        "acknowledgement": _ACKNOWLEDGEMENTS.get(acknowledgement, acknowledgement),
        "detail": _one_line(getattr(record, "detail", "")),
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


# ---------------------------------------------------------------------------
# Assembly and size cap
# ---------------------------------------------------------------------------


def _serialised_size(snapshot: dict) -> int:
    return len(json.dumps(snapshot, sort_keys=True).encode("utf-8"))


def _figure_drop_order(figures: dict) -> list[str]:
    """Order optional figures for trimming: backend-only first, then by name."""

    native = sorted(key for key in figures if key.startswith("native."))
    rest = sorted(key for key in figures if not key.startswith("native."))
    return [*native, *rest]


def apply_contract_size_cap(snapshot: dict) -> dict:
    """Trim an assembled snapshot to the 256 KiB bound, in place.

    Over the cap the snapshot drops the oldest ``recent_events`` first,
    then optional figures, then the oldest ``recent_commands``, and
    reports ``truncated: true``. Required budget figures and limits are
    never dropped.
    """

    limit = telemetry_module.SNAPSHOT_SIZE_LIMIT_BYTES
    if _serialised_size(snapshot) <= limit:
        return snapshot
    snapshot["truncated"] = True
    events = snapshot.get("recent_events")
    while (
        isinstance(events, list)
        and events
        and _serialised_size(snapshot) > limit
    ):
        events.pop(0)
    current = snapshot.get("current_attempt")
    figures = current.get("figures") if isinstance(current, dict) else None
    if isinstance(figures, dict):
        for key in _figure_drop_order(figures):
            if _serialised_size(snapshot) <= limit:
                break
            figures.pop(key, None)
    commands = snapshot.get("recent_commands")
    while (
        isinstance(commands, list)
        and commands
        and _serialised_size(snapshot) > limit
    ):
        commands.pop(0)
    return snapshot


def build_status_snapshot(
    *,
    repository: str,
    identity: dict,
    intake: IntakeState,
    recovering: bool,
    attempt: dict | None,
    pending_record: object,
    stop_plan: dict | None,
    next_issue: dict | None,
    command_details: list,
    current: dict | None,
    last_finished: dict | None,
    usd_ceiling: int | None,
    generated_at: str,
    observed_at: str,
    starting: bool = False,
    startup_check: str | None = None,
    last_halt: dict | None = None,
    recent_events: list | None = None,
) -> dict:
    """Assemble one contract Status Snapshot from prepared state.

    ``starting`` serves the early control endpoint before construction
    finishes (with ``startup_check`` naming provenance verification while
    held). ``recovering`` covers startup reconciliation and intake held
    for recovery. ``last_halt`` is the durable halt from the previous
    engine-decided exit, if any; a successful start never erases it.
    """

    attempt_active = attempt is not None
    snapshot = {
        "generated_at": generated_at,
        "identity": dict(identity),
        "instance": {
            "state": project_instance_state(
                intake,
                recovering=recovering,
                attempt_active=attempt_active,
                starting=starting,
            ),
            "startup_check": startup_check if starting else None,
            "last_halt": dict(last_halt) if last_halt is not None else None,
            "observed_at": observed_at,
        },
        "intake": project_intake(
            intake=intake,
            pending_record=pending_record,
            stop_plan=stop_plan,
            next_issue=next_issue,
            observed_at=observed_at,
        ),
        "current_attempt": project_current_attempt(
            attempt=attempt,
            current=current,
            usd_ceiling=usd_ceiling,
            observed_at=observed_at,
        ),
        "last_finished": project_last_finished(last_finished),
        "recent_commands": [
            project_command(record, payload, repository=repository)
            for record, payload in command_details
        ],
        "recent_events": list(recent_events) if recent_events is not None else [],
        "truncated": False,
    }
    return apply_contract_size_cap(snapshot)
