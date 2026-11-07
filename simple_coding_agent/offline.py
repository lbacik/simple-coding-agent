"""Volume-direct offline projections for a stopped instance.

An operator can observe a stopped instance without starting its engine:
every ``agentctl --contract 1 --offline <read>`` is served from the volume
alone, without the volume lock and without writing. The control snapshot
comes from the reader proven in :mod:`simple_coding_agent.control_offline`
(copy the main database plus its WAL, checkpoint and verify the copy, never
stale); identity comes from ``state/instance.json``; the halt comes from
``state/halt.json``; the last finished attempt and the attempt archives come
from ``logs/``; nothing is created or modified next to the volume.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from simple_coding_agent.attempt_history import AttemptHistoryError
from simple_coding_agent.control_offline import (
    OfflineControlSnapshot,
    OfflineReadError,
    read_offline_control,
)
from simple_coding_agent.contract_adapter import (
    BACKEND,
    CONTRACT_MAJOR,
    CONTRACT_MINOR,
    advertised_capabilities,
)
from simple_coding_agent.halt import HaltStore


def _data_dir(data_dir: Path | None = None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    import os as _os

    return Path(
        _os.environ.get("DATA_DIR", "~/.simple-coding-agent/")
    ).expanduser()


def offline_identity(data_dir: Path | None = None) -> dict:
    """Build the offline ``identity`` result from the volume alone."""

    from simple_coding_agent.instance import InstanceReadError, read_instance

    directory = _data_dir(data_dir)
    try:
        recorded = read_instance(directory)
    except InstanceReadError as error:
        raise OfflineReadError(str(error)) from error
    repository = recorded["repository"]
    worker_id = recorded.get("worker_id", "")
    version = recorded.get("agent_version")
    if not isinstance(version, str) or not version:
        raise OfflineReadError("Instance identity is malformed.")
    return {
        "backend": BACKEND,
        "agent_version": version,
        "capabilities": advertised_capabilities(worker_id),
        "contract": {"major": CONTRACT_MAJOR, "minor": CONTRACT_MINOR},
        "mode": "offline",
        "process_started_at": None,
        "repository": repository,
    }


def _offline_command_details(
    snapshot: OfflineControlSnapshot, *, repository: str, limit: int = 20
) -> list:
    """Recent commands with payloads, newest last, like the online projection."""

    tail = snapshot.commands[-limit:] if limit else ()
    return [
        (
            record,
            dict(snapshot.payload_for(record.request_id)),
        )
        for record in tail
    ]


def _offline_stop_plan(snapshot: OfflineControlSnapshot) -> dict | None:
    if snapshot.stop_after_requested is None or snapshot.pending_command_id is None:
        return None
    requested = snapshot.stop_after_requested
    remaining = snapshot.stop_after_remaining
    if not isinstance(requested, int) or not isinstance(remaining, int):
        raise OfflineReadError("The control state is malformed.")
    return {
        "request_id": snapshot.pending_command_id,
        "kind": "stop_after",
        "requested": requested,
        "remaining": remaining,
        "includes_active_attempt": snapshot.stop_after_active_attempt_id is not None,
        "active_attempt_id": snapshot.stop_after_active_attempt_id,
        "latest_counted_attempt_id": snapshot.latest_counted_attempt_id,
        "latest_counted_outcome": snapshot.latest_counted_outcome,
    }


def _offline_next_issue(snapshot: OfflineControlSnapshot) -> dict | None:
    if snapshot.next_issue_request_id is None or snapshot.next_issue_number is None:
        return None
    return {
        "request_id": snapshot.next_issue_request_id,
        "issue_number": int(snapshot.next_issue_number),
    }


def _offline_handoff(snapshot: OfflineControlSnapshot) -> dict | None:
    from simple_coding_agent.control import handoff_deadlines

    request_id = snapshot.handoff_request_id
    if request_id is None:
        return None
    record = next(
        (entry for entry in snapshot.commands if entry.request_id == request_id),
        None,
    )
    deadlines = handoff_deadlines(snapshot.handoff_accepted_at)
    return {
        "request_id": request_id,
        "accepted_at": snapshot.handoff_accepted_at,
        "begun": bool(snapshot.handoff_begun),
        "phase": snapshot.handoff_phase or "accepted",
        "acknowledgement": record.acknowledgement.value if record is not None else None,
        "detail": record.detail if record is not None else None,
        "model_deadline_at": deadlines.get("model_deadline_at"),
        "publication_deadline_at": deadlines.get("publication_deadline_at"),
    }


def _offline_pending_record(snapshot: OfflineControlSnapshot):  # type: ignore[no-untyped-def]
    if snapshot.pending_command_id is None:
        return None
    candidate = next(
        (
            entry
            for entry in snapshot.commands
            if entry.request_id == snapshot.pending_command_id
        ),
        None,
    )
    if candidate is not None and candidate.kind in ("stop", "handoff"):
        return candidate
    return None


def offline_status(data_dir: Path | None = None) -> dict:
    """Build the offline contract ``status`` snapshot from the volume alone."""

    from simple_coding_agent import contract_status as contract_status_module
    from simple_coding_agent.telemetry import load_last_finished

    directory = _data_dir(data_dir)
    identity = offline_identity(directory)
    snapshot = read_offline_control(directory)
    try:
        halt = HaltStore(directory).read()
    except Exception as error:
        raise OfflineReadError(f"Halt record could not be read: {error}.") from error
    last_halt = (
        {"reason": halt.reason, "detail": halt.detail, "at": halt.at}
        if halt is not None
        else None
    )
    try:
        last_finished = load_last_finished(directory)
    except Exception as error:
        raise OfflineReadError(
            f"Last finished attempt could not be read: {error}."
        ) from error
    if last_finished is not None and not isinstance(last_finished, dict):
        raise OfflineReadError("Last finished attempt is malformed.")
    stamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    try:
        assembled = contract_status_module.build_status_snapshot(
            repository=identity["repository"],
            identity=identity,
            intake=snapshot.intake,
            recovering=False,
            attempt=None,
            pending_record=_offline_pending_record(snapshot),
            stop_plan=_offline_stop_plan(snapshot),
            next_issue=_offline_next_issue(snapshot),
            command_details=_offline_command_details(
                snapshot, repository=identity["repository"]
            ),
            current=None,
            last_finished=last_finished,
            usd_ceiling=None,
            generated_at=stamp,
            observed_at=stamp,
            starting=False,
            startup_check=None,
            last_halt=last_halt,
            recent_events=[],
        )
    except OfflineReadError:
        raise
    except Exception as error:
        raise OfflineReadError(
            f"Offline status could not be built: {error}."
        ) from error
    # A stopped volume never reports a live attempt: the checkpoint, when
    # one survived an unclean stop, belongs to no running process.
    if assembled.get("current_attempt") is not None:
        raise OfflineReadError("Offline status is inconsistent.")
    return assembled


def offline_command(data_dir: Path | None, request_id: str) -> dict:
    """Build one offline ``command`` result from the volume alone."""

    from simple_coding_agent import contract_status as contract_status_module

    directory = _data_dir(data_dir)
    identity = offline_identity(directory)
    snapshot = read_offline_control(directory)
    for record in snapshot.commands:
        if record.request_id == request_id:
            try:
                return contract_status_module.project_command(
                    record,
                    snapshot.payload_for(request_id),
                    repository=identity["repository"],
                )
            except Exception as error:
                raise OfflineReadError(
                    f"Offline command could not be built: {error}."
                ) from error
    raise AttemptHistoryError("not_found", f"no command is recorded for {request_id}")


def _require_readable_volume(data_dir: Path | None) -> Path:
    """Resolve the volume and fail closed unless it reads as a stopped instance.

    A missing or malformed identity is a failed read, never an empty
    listing; an unreadable control store fails closed the same way other
    offline reads do, so a torn WAL never yields a partial listing.
    """

    directory = _data_dir(data_dir)
    offline_identity(directory)
    read_offline_control(directory)
    return directory


def offline_attempts_list(
    data_dir: Path | None, limit: int | None, before: str | None
) -> dict:
    """Build one offline ``attempts list`` page from the archives alone."""

    from simple_coding_agent import attempt_history as attempt_history_module

    directory = _require_readable_volume(data_dir)
    return attempt_history_module.list_attempts(
        directory, active=None, held=frozenset(), limit=limit, before=before
    )


def offline_attempts_show(data_dir: Path | None, attempt_id: str) -> dict:
    """Build one offline ``attempts show`` result from the archives alone."""

    from simple_coding_agent import attempt_history as attempt_history_module

    directory = _require_readable_volume(data_dir)
    return attempt_history_module.show_attempt(
        directory, attempt_id, active=None, held=frozenset()
    )


def offline_attempts_export(data_dir: Path | None, attempt_id: str) -> dict:
    """Build one offline ``attempts export`` projection from the archives alone."""

    from simple_coding_agent import attempt_history as attempt_history_module

    directory = _require_readable_volume(data_dir)
    return attempt_history_module.export_projection(
        directory, attempt_id, active=None, held=frozenset()
    )
