"""The Operator handoff deep module (issue #112).

Interprets the handoff request lifecycle the ControlStore persists: the
request lifecycle (accepted → delivering → begun → complete / not
fulfilled), the deadlines computed *and* parsed in one place, and the
verdict handed to the executor (``fallback`` / ``expire``) and to the
lifecycle (complete / fail / hold).

The ControlStore stays persistence only; the clock is an injected port so
tests run on a fake clock against a real sqlite ControlStore in tmp.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol

# Operator-handoff timing from the durable acceptance commit: the model must
# finish its local handoff work within 240 seconds, and the driving process
# gets at most a further 120 seconds to confirm publication (360 total).
# Restart never resets these bounds; they are recomputed from the original
# ``accepted_at`` on every read.
HANDOFF_MODEL_DEADLINE_SECONDS = 240
HANDOFF_PUBLICATION_ALLOWANCE_SECONDS = 120
HANDOFF_ABSOLUTE_DEADLINE_SECONDS = 360

#: How long the executor waits before nudging a latched operator request
#: with the single follow-up handoff prompt.
OPERATOR_HANDOFF_DELIVERY_WINDOW_SECONDS = 60

#: Narrow verdict handed to the lifecycle after finalization.
HandoffVerdict = Literal["owned", "failed", "hold"]

#: Wall-clock port. Production passes ``datetime.now(UTC)``; tests pass a
#: fake clock.
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class OperatorHandoff:
    """An operator ``handoff now`` request the model must observe.

    ``accepted_at`` is the wall-clock acceptance time of the durable command
    commit; delivery and model-deadline windows are measured from it.
    """

    request_id: str
    accepted_at: datetime


OperatorHandoffProvider = Callable[[], "OperatorHandoff | None"]
OperatorHandoffReporter = Callable[[str, str], None]


class OperatorHandoffControl(Protocol):
    """The control boundary a model attempt uses to serve one handoff request."""

    def snapshot(self) -> dict | None: ...

    def mark_delivering(self, request_id: str) -> bool: ...

    def mark_begun(self, request_id: str) -> bool: ...

    def complete(self, request_id: str, detail: str) -> None: ...

    def fail(self, request_id: str, reason: str) -> None: ...


def parse_operator_accepted_at(value: object) -> datetime | None:
    """Parse a control ``accepted_at`` timestamp for executor windows.

    This is the single parser for handoff acceptance times: deadlines are
    computed *and* re-parsed here so ``lifecycle.py`` no longer parses them
    again beside ``control.py``.
    """

    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def format_control_timestamp(value: datetime) -> str:
    """Render a timestamp the way the control store records them."""

    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def handoff_deadlines(accepted_at: object) -> dict[str, str | None]:
    """Compute the handoff deadlines from the acceptance timestamp.

    Returns ``model_deadline_at`` (acceptance + 240s) and
    ``publication_deadline_at`` (acceptance + 360s) as ``Z``-suffixed ISO
    strings, or ``None`` values when the acceptance time cannot be parsed.
    The bounds always derive from the original durable commit, so a process
    restart can neither extend nor renew them.
    """

    parsed = parse_operator_accepted_at(accepted_at)
    if parsed is None:
        return {"model_deadline_at": None, "publication_deadline_at": None}
    return {
        "model_deadline_at": format_control_timestamp(
            parsed + timedelta(seconds=HANDOFF_MODEL_DEADLINE_SECONDS)
        ),
        "publication_deadline_at": format_control_timestamp(
            parsed + timedelta(seconds=HANDOFF_ABSOLUTE_DEADLINE_SECONDS)
        ),
    }


def publish_budget_for_snapshot(snapshot: dict | None, now: datetime) -> float | None:
    """Bound one publication call by the pending handoff's remaining budget.

    Returns ``min(120, 360 - elapsed-since-acceptance)`` seconds, clamped at
    zero, while an operator handoff is accepted; otherwise None, keeping the
    publisher's configured timeout. The budget is anchored at the original
    acceptance stored in control, so a restart never renews it.
    """

    if snapshot is None or snapshot.get("acknowledgement") != "accepted":
        return None
    accepted_at = parse_operator_accepted_at(snapshot.get("accepted_at"))
    if accepted_at is None:
        return None
    remaining = HANDOFF_ABSOLUTE_DEADLINE_SECONDS - (now - accepted_at).total_seconds()
    return max(0.0, min(float(HANDOFF_PUBLICATION_ALLOWANCE_SECONDS), remaining))


def handoff_completion_cause(
    snapshot: dict,
    *,
    outcome_value: str,
    handoff_outcome_value: str,
    note_valid: bool,
    note_why: str,
    begun: bool,
    publication_started_at: datetime | None,
    now: datetime,
    check_allowance: bool = True,
) -> str | None:
    """Decide whether an accepted handoff can complete; None means it can.

    Pure core of the lifecycle's verdict: the handoff completes only when the
    attempt outcome is a published handoff, the model began its skill, the
    committed note is a valid ``operator_request`` note, publication
    confirmed within 120 seconds after the valid local handoff, and the whole
    command finished within 360 seconds of its original acceptance.
    Otherwise returns the ``not fulfilled`` cause. ``check_allowance`` is
    False only for the pre-commit recheck after a confirmed publication.
    """

    if outcome_value != handoff_outcome_value:
        return f"the attempt finalized with the ordinary outcome {outcome_value}"
    if not begun:
        return "model execution never began the handoff skill"
    if not note_valid:
        return f"the committed handoff note is not a valid operator note ({note_why})"
    accepted_at = parse_operator_accepted_at(snapshot.get("accepted_at"))
    if accepted_at is None:
        return (
            "the original acceptance time is unavailable, so the"
            " 360-second budget cannot be verified"
        )
    if (now - accepted_at).total_seconds() > HANDOFF_ABSOLUTE_DEADLINE_SECONDS:
        return (
            "the 360-second budget from acceptance elapsed before"
            " publication, release, and cleanup finished"
            f" (accepted {snapshot.get('accepted_at')})"
        )
    if (
        check_allowance
        and publication_started_at is not None
        and (now - publication_started_at).total_seconds()
        > HANDOFF_PUBLICATION_ALLOWANCE_SECONDS
    ):
        return (
            "publication was not confirmed within 120 seconds after"
            " the valid local handoff"
        )
    return None


def handoff_owned_by_attempt(detail: object, attempt_id: str) -> bool:
    """Whether the tracked handoff failure was recorded for this attempt.

    The hold records its retaining attempt ID in the ``not fulfilled``
    reason, so a crash replay can tell a hold it must preserve from a stale
    failure owned by an earlier attempt (which finalizes normally).
    """

    return (
        bool(attempt_id)
        and isinstance(detail, str)
        and f"of attempt {attempt_id}." in detail
    )


def classify_verdict(
    acknowledgement: object,
    *,
    completion_cause: str | None,
    owned_by_attempt: bool,
) -> HandoffVerdict | None:
    """Map a snapshot plus the completion cause to the lifecycle verdict.

    Returns ``"owned"`` when the handoff completes for this attempt,
    ``"hold"`` when the attempt must retain everything for operator
    recovery, ``"failed"`` when the failure belongs to another attempt (or
    is already terminal), and ``None`` when no handoff owns this attempt.
    """

    if acknowledgement == "completed":
        return "owned"
    if acknowledgement != "accepted" and acknowledgement != "not fulfilled":
        return None
    if acknowledgement == "not fulfilled":
        return "hold" if owned_by_attempt else "failed"
    # accepted
    if completion_cause is None:
        return "owned"
    return "hold"


def operator_due_action(
    *,
    accepted_at: datetime,
    now: datetime,
    context_delivered: bool,
    begun: bool,
    fallback_used: bool,
    deadline_expired: bool,
) -> Literal["fallback", "expire"] | None:
    """Return ``fallback``/``expire`` when an operator boundary is due now.

    Narrow port the executor consults: ``expire`` once the 240-second model
    deadline elapsed, ``fallback`` once the 60-second delivery window elapsed
    while the handoff context was never delivered and the skill never began.
    """

    if deadline_expired:
        return None
    elapsed = (now - accepted_at).total_seconds()
    if elapsed >= HANDOFF_MODEL_DEADLINE_SECONDS:
        return "expire"
    if (
        not fallback_used
        and not context_delivered
        and not begun
        and elapsed >= OPERATOR_HANDOFF_DELIVERY_WINDOW_SECONDS
    ):
        return "fallback"
    return None


def operator_stream_wait_seconds(
    *,
    accepted_at: datetime | None,
    now: datetime,
    provider_present: bool,
    context_delivered: bool,
    begun: bool,
    fallback_used: bool,
) -> float | None:
    """How long the stream may wait before the next operator boundary.

    ``None`` waits indefinitely (no provider is watching). Otherwise the wait
    ends at the next due action — the single fallback or the model deadline —
    or at the delivery-window cadence that re-polls the provider.
    """

    if accepted_at is None:
        if provider_present:
            return float(OPERATOR_HANDOFF_DELIVERY_WINDOW_SECONDS)
        return None
    elapsed = (now - accepted_at).total_seconds()
    targets = [HANDOFF_MODEL_DEADLINE_SECONDS - elapsed]
    if not fallback_used and not context_delivered and not begun:
        targets.append(OPERATOR_HANDOFF_DELIVERY_WINDOW_SECONDS - elapsed)
    return max(min(targets), 0.0)


def operator_handoff_provider(
    control: OperatorHandoffControl,
) -> OperatorHandoffProvider:
    """Build the executor provider reading the live control snapshot."""

    def provider() -> OperatorHandoff | None:
        try:
            snapshot = control.snapshot()
        except Exception:
            return None
        if snapshot is None or snapshot.get("acknowledgement") != "accepted":
            return None
        request_id = snapshot.get("request_id")
        accepted_at = parse_operator_accepted_at(snapshot.get("accepted_at"))
        if not isinstance(request_id, str) or not request_id or accepted_at is None:
            return None
        return OperatorHandoff(request_id=request_id, accepted_at=accepted_at)

    return provider


def operator_handoff_reporter(
    control: OperatorHandoffControl,
    event_log: Callable[..., None] | None = None,
) -> OperatorHandoffReporter:
    """Build the executor reporter for ``delivered`` / ``begun`` events."""

    def reporter(event: str, request_id: str) -> None:
        try:
            if event == "delivered":
                control.mark_delivering(request_id)
            elif event == "begun":
                control.mark_begun(request_id)
        except Exception as error:
            if event_log is not None:
                try:
                    event_log(
                        "operator_handoff_report_failed",
                        f"event={event}; request={request_id}; "
                        f"{type(error).__name__}: {error}",
                        level="WARNING",
                    )
                except Exception:
                    pass

    return reporter


def wire_operator_handoff(
    executor: object,
    control: OperatorHandoffControl,
    event_log: Callable[..., None] | None = None,
) -> None:
    """Serve the pending operator handoff through one model execution.

    The provider reads the live control snapshot on every poll, so a request
    accepted during setup is picked up once execution starts; executors
    without the operator seams (older fakes) are left alone.
    """

    set_provider = getattr(executor, "set_operator_handoff_provider", None)
    if callable(set_provider):
        set_provider(operator_handoff_provider(control))
    set_reporter = getattr(executor, "set_operator_handoff_reporter", None)
    if callable(set_reporter):
        set_reporter(operator_handoff_reporter(control, event_log))


class ControlOperatorHandoff:
    """Adapt the durable control store to the runner's operator boundary."""

    def __init__(
        self,
        store: object,
        event_log: Callable[..., None],
        on_resolved: Callable[[object], None] | None = None,
    ) -> None:
        self._store = store
        self._event_log = event_log
        self._on_resolved = on_resolved

    def snapshot(self) -> dict | None:
        return self._store.handoff_snapshot()  # type: ignore[union-attr]

    def mark_delivering(self, request_id: str) -> bool:
        return self._store.mark_handoff_delivering(request_id)  # type: ignore[union-attr]

    def mark_begun(self, request_id: str) -> bool:
        return self._store.mark_handoff_begun(request_id)  # type: ignore[union-attr]

    def complete(self, request_id: str, detail: str) -> None:
        record = self._store.complete_handoff(request_id, detail)  # type: ignore[union-attr]
        if record is None:
            self._event_log(
                "operator_handoff_resolution_unknown",
                f"request={request_id}; the handoff command is no longer recorded",
                level="WARNING",
            )
        elif self._on_resolved is not None:
            try:
                self._on_resolved(record)
            except Exception:
                pass

    def fail(self, request_id: str, reason: str) -> None:
        record = self._store.fail_handoff(request_id, reason)  # type: ignore[union-attr]
        if record is None:
            self._event_log(
                "operator_handoff_resolution_unknown",
                f"request={request_id}; the handoff command is no longer recorded",
                level="WARNING",
            )
        elif self._on_resolved is not None:
            try:
                self._on_resolved(record)
            except Exception:
                pass


__all__ = [
    "HANDOFF_ABSOLUTE_DEADLINE_SECONDS",
    "HANDOFF_MODEL_DEADLINE_SECONDS",
    "HANDOFF_PUBLICATION_ALLOWANCE_SECONDS",
    "OPERATOR_HANDOFF_DELIVERY_WINDOW_SECONDS",
    "Clock",
    "ControlOperatorHandoff",
    "HandoffVerdict",
    "OperatorHandoff",
    "OperatorHandoffControl",
    "OperatorHandoffProvider",
    "OperatorHandoffReporter",
    "classify_verdict",
    "format_control_timestamp",
    "handoff_completion_cause",
    "handoff_deadlines",
    "handoff_owned_by_attempt",
    "operator_due_action",
    "operator_handoff_provider",
    "operator_handoff_reporter",
    "operator_stream_wait_seconds",
    "parse_operator_accepted_at",
    "publish_budget_for_snapshot",
    "wire_operator_handoff",
]
