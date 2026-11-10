"""Operator contract major-1 adapter (``agentctl --contract 1 ...``).

A contract consumer runs ``agentctl --contract 1 <operation>`` through
``docker exec`` without a TTY and always gets exactly one JSON envelope
with a trailing newline on stdout plus the contract's distinct exit
code. This module owns the contract check order, envelope rendering and
exit codes; it reaches the engine only through the existing control
socket, plus the read-only ``identity`` socket operation. Nothing here is
shared with the legacy path: without ``--contract`` :func:`agentctl.main`
keeps today's grammar, replies, exit codes and stderr byte-compatible.

Check order (spec section 4.1): ``--contract`` major, argv grammar
(``usage``), capability (``unsupported``), argument values
(``invalid_argument``); expected repository, idempotency, state and
commit run on the engine side in later tickets. Operations those later
tickets implement are parsed and gated here but answer ``unsupported``
until their projection exists; an unreachable engine answers
``agent_unreachable`` and an engine-side failure answers ``internal``.

The normative source is agent-operator-contract ``v1.0.0``. The
contract-side rendering (DTOs, envelope, exit codes, CLI grammar and
capability checks) uses the shared part of that library
(:mod:`agent_operator_contract.shared`), pinned to the same tag; the
grammar, value bounds, envelope shapes and exit codes here are that
library's, not local equivalents. Engine-side behaviour (control socket
wire format, projections, redaction, sealing, export) stays private.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path, PurePosixPath

from agent_operator_contract import CONTRACT_MAJOR, shared

from simple_coding_agent.control_server import (
    ControlUnavailableError,
    UnreadableReplyError,
    resolve_socket_path,
    send_request,
)
from simple_coding_agent.attempt_history import AttemptHistoryError

#: The one contract major this producer serves (spec section 5).
assert CONTRACT_MAJOR == 1
#: The contract minor reported in every envelope.
CONTRACT_MINOR = shared.CONTRACT_MINOR
#: The ``contract`` member of every envelope.
CONTRACT_VERSION = shared.CONTRACT_VERSION

#: Backend name reported by ``identity``.
BACKEND = "simple-coding-agent"

#: Optional capabilities this backend always advertises, sorted.
BASE_CAPABILITIES = sorted(
    capability
    for capability in shared.OPERATOR_CAPABILITIES
    if capability != "pool"
)

#: Declaration-only capability advertised exactly when a Worker id is set.
POOL_CAPABILITY = "pool"

#: Socket timeout for contract reads (consumer default; the engine's own
#: budget for ``identity`` is 3 s, served lock-free without I/O).
SOCKET_TIMEOUT_SECONDS = 10.0

EXIT_OK = 0
EXIT_REJECTED = shared.exit_code_for("rejected")
EXIT_USAGE = shared.exit_code_for("usage")
EXIT_UNSUPPORTED = shared.exit_code_for("unsupported")
EXIT_CONTRACT_MISMATCH = shared.exit_code_for("contract_mismatch")
EXIT_NOT_FOUND = shared.exit_code_for("not_found")
EXIT_AGENT_UNREACHABLE = shared.exit_code_for("agent_unreachable")
EXIT_INTERNAL = shared.exit_code_for("internal")

_ONLINE_MODES = ("online", "offline")


def advertised_capabilities(worker_id: str) -> list[str]:
    """Sorted capabilities for one instance: the base set plus ``pool``.

    ``pool`` is advertised exactly when a Worker id is configured, so a
    deployment adapter refuses a second same-repository instance for an
    unpooled deployment.
    """

    capabilities = list(BASE_CAPABILITIES)
    if worker_id.strip():
        capabilities.append(POOL_CAPABILITY)
    return sorted(capabilities)


# ---------------------------------------------------------------------------
# Envelope rendering (stdout only; stderr stays empty on this path)
# ---------------------------------------------------------------------------
#
# Rendering is the shared library's: :func:`_emit_ok` and
# :func:`_emit_error` only write what ``shared`` built, so the envelope
# shape, the ``contract`` member, key order and the exit codes cannot drift
# from the pinned tag.


def _emit_ok(result: dict) -> int:
    """Write one success envelope; always exit 0."""

    sys.stdout.write(shared.render_success(result))
    return EXIT_OK


def _emit_error(
    *,
    code: str,
    message: str,
    request_id: str | None = None,
    details: dict | None = None,
) -> int:
    """Write one error envelope with the contract's exit code."""

    text, exit_code = shared.render_error(
        code, message, details=details, request_id=request_id
    )
    sys.stdout.write(text)
    return exit_code


def _emit_failure(failure: shared.ArgvFailure) -> int:
    """Write one grammar/gate failure from the shared parser; return its exit."""

    return _emit_error(
        code=failure.code,
        message=failure.message,
        request_id=failure.request_id,
        details=failure.details or None,
    )


def _unsupported(
    operation: str,
    *,
    capability: str | None,
    request_id: str | None,
    message: str,
    details: dict | None = None,
) -> int:
    """Answer ``unsupported`` (exit 3) without consuming the request ID."""

    resolved = dict(details) if details is not None else {}
    if capability is not None:
        resolved.setdefault("capability", capability)
    return _emit_error(
        code="unsupported",
        message=message,
        request_id=request_id,
        details=resolved,
    )


# ---------------------------------------------------------------------------
# Grammar and argument values (the shared library owns both)
# ---------------------------------------------------------------------------
#
# ``shared.parse_argv`` parses the contract argv and returns the first
# failure in the section 4.1 check order (``contract_mismatch`` ->
# ``usage`` -> ``unsupported`` -> ``invalid_argument``). This adapter never
# parses argv itself: :func:`main` runs the shared parser once before
# touching the engine (so grammar failures need no engine) and once more
# with the instance's advertised capabilities (so the capability gate runs
# before value checks). The shared operation names (``"attempts-list"``,
# ``"next-issue"``, ``"handoff"``, ...) are translated onto the engine's
# socket operations by :func:`_mutation_socket_op` and the read runners.
#
# ``ParsedCall`` is ``shared.ParsedCall``: ``operation`` is the canonical
# name, ``offline`` marks an Offline Read, ``request_id``/``expect_repo``
# are present exactly for mutations, ``arguments`` carries the validated
# raw values, and ``capability`` is the required optional capability.








# ---------------------------------------------------------------------------
# Engine contact (control socket only)
# ---------------------------------------------------------------------------


class _Unreachable(Exception):
    """The engine process could not be reached (ambiguous for mutations)."""


class _EngineFailed(Exception):
    """The engine answered but could not serve the request."""


class _EngineNotFound(Exception):
    """The engine has no record of the requested Attempt, command or path."""


class _EngineInvalidArgument(Exception):
    """The engine rejected an argument value the CLI grammar accepted."""


def _fetch_identity() -> dict:
    """Read the engine's identity; raise :class:`_Unreachable`/`_EngineFailed`."""

    try:
        reply = send_request(
            resolve_socket_path(), {"op": "identity"}, timeout=SOCKET_TIMEOUT_SECONDS
        )
    except UnreadableReplyError as error:
        raise _EngineFailed(
            f"the agent process answered but the reply could not be used: {error}"
        ) from error
    except ControlUnavailableError as error:
        raise _Unreachable(str(error)) from error
    if not reply.get("ok"):
        raise _EngineFailed(str(reply.get("error", "identity is unavailable")))
    identity = reply.get("identity")
    problem = _identity_problem(identity)
    if problem is not None:
        raise _EngineFailed(f"the agent process answered malformed identity: {problem}")
    assert isinstance(identity, dict)
    return identity


def _identity_problem(identity: object) -> str | None:
    """Name the first malformation of an engine identity, if any."""

    if not isinstance(identity, dict):
        return "identity is not an object"
    for key in ("backend", "agent_version", "repository"):
        if not isinstance(identity.get(key), str) or not identity.get(key):
            return f"identity.{key} is not a non-empty string"
    contract = identity.get("contract")
    if (
        not isinstance(contract, dict)
        or contract.get("major") != CONTRACT_MAJOR
        or not isinstance(contract.get("minor"), int)
    ):
        return "identity.contract is not {major: 1, minor: int}"
    if identity.get("mode") not in _ONLINE_MODES:
        return "identity.mode is not online or offline"
    started = identity.get("process_started_at")
    if started is not None and not isinstance(started, str):
        return "identity.process_started_at is not a string or null"
    capabilities = identity.get("capabilities")
    if not isinstance(capabilities, list) or not all(
        isinstance(entry, str) for entry in capabilities
    ):
        return "identity.capabilities is not a list of strings"
    return None


# ---------------------------------------------------------------------------
# Contract status
# ---------------------------------------------------------------------------


_STATUS_RESULT_KEYS = frozenset(
    {
        "generated_at",
        "identity",
        "instance",
        "intake",
        "current_attempt",
        "last_finished",
        "recent_commands",
        "recent_events",
        "truncated",
    }
)


def _fetch_status() -> dict:
    """Read the engine's contract status; raise :class:`_Unreachable`/`_EngineFailed`."""

    try:
        reply = send_request(
            resolve_socket_path(), {"op": "contract_status"}, timeout=SOCKET_TIMEOUT_SECONDS
        )
    except UnreadableReplyError as error:
        raise _EngineFailed(
            f"the agent process answered but the reply could not be used: {error}"
        ) from error
    except ControlUnavailableError as error:
        raise _Unreachable(str(error)) from error
    if not reply.get("ok"):
        raise _EngineFailed(str(reply.get("error", "contract status is unavailable")))
    status = reply.get("contract_status")
    problem = _status_problem(status)
    if problem is not None:
        raise _EngineFailed(
            f"the agent process answered malformed contract status: {problem}"
        )
    assert isinstance(status, dict)
    return status


def _status_problem(status: object) -> str | None:
    """Name the first malformation of an engine contract status, if any."""

    if not isinstance(status, dict):
        return "status is not an object"
    missing = _STATUS_RESULT_KEYS - set(status)
    if missing:
        return f"status is missing keys: {sorted(missing)}"
    if not isinstance(status.get("generated_at"), str):
        return "status.generated_at is not a string"
    for key in ("identity", "instance", "intake"):
        if not isinstance(status.get(key), dict):
            return f"status.{key} is not an object"
    if status.get("current_attempt") is not None and not isinstance(
        status.get("current_attempt"), dict
    ):
        return "status.current_attempt is not an object or null"
    if status.get("last_finished") is not None and not isinstance(
        status.get("last_finished"), dict
    ):
        return "status.last_finished is not an object or null"
    if not isinstance(status.get("recent_commands"), list):
        return "status.recent_commands is not a list"
    if not isinstance(status.get("recent_events"), list):
        return "status.recent_events is not a list"
    if not isinstance(status.get("truncated"), bool):
        return "status.truncated is not a boolean"
    return None


# ---------------------------------------------------------------------------
# Contract mutations (stop, resume) and command lookup
# ---------------------------------------------------------------------------


_COMMAND_RESULT_KEYS = frozenset(
    {
        "acknowledgement",
        "arguments",
        "created_at",
        "detail",
        "kind",
        "repository",
        "request_id",
        "sequence",
        "updated_at",
    }
)


def _command_problem(command: object) -> str | None:
    """Name the first malformation of an engine CommandRecord, if any."""

    if not isinstance(command, dict):
        return "command is not an object"
    missing = _COMMAND_RESULT_KEYS - set(command)
    if missing:
        return f"command is missing keys: {sorted(missing)}"
    if not isinstance(command.get("request_id"), str):
        return "command.request_id is not a string"
    if not isinstance(command.get("sequence"), int):
        return "command.sequence is not an integer"
    if not isinstance(command.get("kind"), str):
        return "command.kind is not a string"
    if not isinstance(command.get("arguments"), dict):
        return "command.arguments is not an object"
    if not isinstance(command.get("repository"), str):
        return "command.repository is not a string"
    if command.get("acknowledgement") not in (
        "accepted",
        "completed",
        "superseded",
        "not_fulfilled",
    ):
        return "command.acknowledgement is not a known acknowledgement"
    if not isinstance(command.get("detail"), str):
        return "command.detail is not a string"
    if "\n" in command.get("detail", ""):
        return "command.detail carries a line break"
    if not isinstance(command.get("created_at"), str):
        return "command.created_at is not a string"
    if not isinstance(command.get("updated_at"), str):
        return "command.updated_at is not a string"
    return None


def _send_contract_mutation(
    op: str, request_id: str, expect_repo: str, extra: dict | None = None
) -> dict:
    """Send one contract mutation; raise :class:`_Unreachable`/`_EngineFailed`."""

    payload = {
        "op": op,
        "request_id": request_id,
        "expect_repo": expect_repo,
        "contract": True,
    }
    if extra:
        payload.update(extra)
    try:
        return send_request(
            resolve_socket_path(),
            payload,
            timeout=SOCKET_TIMEOUT_SECONDS,
        )
    except UnreadableReplyError as error:
        raise _EngineFailed(
            f"the agent process answered but the reply could not be used: {error}"
        ) from error
    except ControlUnavailableError as error:
        raise _Unreachable(str(error)) from error


def _send_contract_command(request_id: str) -> dict:
    """Send one contract ``command`` lookup; raise on transport failure."""

    try:
        return send_request(
            resolve_socket_path(),
            {"op": "command", "request_id": request_id, "contract": True},
            timeout=SOCKET_TIMEOUT_SECONDS,
        )
    except UnreadableReplyError as error:
        raise _EngineFailed(
            f"the agent process answered but the reply could not be used: {error}"
        ) from error
    except ControlUnavailableError as error:
        raise _Unreachable(str(error)) from error


# ---------------------------------------------------------------------------
# Contract attempts
# ---------------------------------------------------------------------------


def _engine_error_for(reply: dict, operation: str) -> Exception:
    """Map a failed engine reply onto the contract error it carries."""

    message = str(reply.get("error", f"{operation} is unavailable"))
    code = reply.get("code")
    if code == "not_found":
        return _EngineNotFound(message)
    if code == "invalid_argument":
        return _EngineInvalidArgument(message)
    return _EngineFailed(message)


def _fetch_attempts_list(limit: int | None, before: str | None) -> dict:
    """Read one ``attempts list`` page; raise :class:`_Unreachable`/`_EngineFailed`."""

    try:
        reply = send_request(
            resolve_socket_path(),
            {"op": "attempts_list", "limit": limit, "before": before},
            timeout=SOCKET_TIMEOUT_SECONDS,
        )
    except UnreadableReplyError as error:
        raise _EngineFailed(
            f"the agent process answered but the reply could not be used: {error}"
        ) from error
    except ControlUnavailableError as error:
        raise _Unreachable(str(error)) from error
    if not reply.get("ok"):
        raise _engine_error_for(reply, "attempts list")
    page = reply.get("attempts_list")
    problem = _attempts_list_problem(page)
    if problem is not None:
        raise _EngineFailed(
            f"the agent process answered malformed attempts list: {problem}"
        )
    assert isinstance(page, dict)
    return page


def _attempts_list_problem(page: object) -> str | None:
    """Name the first malformation of an engine ``attempts list`` page, if any."""

    if not isinstance(page, dict):
        return "page is not an object"
    attempts = page.get("attempts")
    if not isinstance(attempts, list) or not all(
        isinstance(entry, dict) for entry in attempts
    ):
        return "page.attempts is not a list of objects"
    if page.get("next_cursor") is not None and not isinstance(
        page.get("next_cursor"), str
    ):
        return "page.next_cursor is not a string or null"
    return None


def _fetch_attempts_show(attempt_id: str) -> dict:
    """Read one ``attempts show`` result; raise :class:`_Unreachable`/`_EngineFailed`."""

    try:
        reply = send_request(
            resolve_socket_path(),
            {"op": "attempts_show", "attempt_id": attempt_id},
            timeout=SOCKET_TIMEOUT_SECONDS,
        )
    except UnreadableReplyError as error:
        raise _EngineFailed(
            f"the agent process answered but the reply could not be used: {error}"
        ) from error
    except ControlUnavailableError as error:
        raise _Unreachable(str(error)) from error
    if not reply.get("ok"):
        raise _engine_error_for(reply, "attempts show")
    result = reply.get("attempts_show")
    problem = _attempts_show_problem(result)
    if problem is not None:
        raise _EngineFailed(
            f"the agent process answered malformed attempts show: {problem}"
        )
    assert isinstance(result, dict)
    return result


def _attempts_show_problem(result: object) -> str | None:
    """Name the first malformation of an engine ``attempts show`` result, if any."""

    if not isinstance(result, dict):
        return "result is not an object"
    if not isinstance(result.get("attempt"), dict):
        return "result.attempt is not an object"
    manifest = result.get("manifest")
    if not isinstance(manifest, list) or not all(
        isinstance(entry, dict) for entry in manifest
    ):
        return "result.manifest is not a list of objects"
    return None


def _run_attempts_list(parsed: shared.ParsedCall) -> int:
    """Serve ``attempts list`` from the engine's history projection."""

    raw_limit = parsed.arguments.get("limit")
    before = parsed.arguments.get("before")
    try:
        limit = int(raw_limit, 10) if raw_limit is not None else None
        page = _fetch_attempts_list(limit, before)
    except _Unreachable:
        return _emit_error(
            code="agent_unreachable",
            message="cannot reach the agent process",


        )
    except (ValueError, _EngineInvalidArgument) as error:
        return _emit_error(
            code="invalid_argument",
            message=str(error),


        )
    except _EngineFailed as error:
        return _emit_error(
            code="internal",
            message=f"the agent process could not serve 'attempts list': {error}",


            details={"stored": False},
        )
    return _emit_ok(page)


def _fetch_attempts_export(attempt_id: str) -> dict:
    """Read one ``attempts export`` projection; raise :class:`_Unreachable`/`_EngineFailed`."""

    try:
        reply = send_request(
            resolve_socket_path(),
            {"op": "attempts_export", "attempt_id": attempt_id},
            timeout=SOCKET_TIMEOUT_SECONDS,
        )
    except UnreadableReplyError as error:
        raise _EngineFailed(
            f"the agent process answered but the reply could not be used: {error}"
        ) from error
    except ControlUnavailableError as error:
        raise _Unreachable(str(error)) from error
    if not reply.get("ok"):
        raise _engine_error_for(reply, "attempts export")
    result = reply.get("attempts_export")
    problem = _attempts_show_problem(result)
    if problem is not None:
        raise _EngineFailed(
            f"the agent process answered malformed attempts export: {problem}"
        )
    assert isinstance(result, dict)
    return result


def _export_data_dir() -> Path:
    """The archive volume the export streams bytes from (never the socket)."""

    return Path(os.environ.get("DATA_DIR", "~/.simple-coding-agent/")).expanduser()


def _run_attempts_export(parsed: shared.ParsedCall) -> int:
    """Stream ``attempts export`` as an uncompressed pax tar to stdout.

    The summary and export manifest come from the engine's history
    projection; the file bytes are read volume-direct so the tar never
    crosses the control socket. ``manifest.json`` is the first member.
    An unknown Attempt or ``--path`` answers ``not_found`` with no tar
    bytes on stdout; a file that grew after the snapshot keeps the
    snapshot's manifest entry while streaming the bytes actually read,
    so the consumer detects the torn read as a ``sha256`` mismatch.
    """

    from simple_coding_agent import attempt_history as attempt_history_module

    attempt_id = parsed.arguments.get("attempt_id")
    assert isinstance(attempt_id, str)
    try:
        projection = _fetch_attempts_export(attempt_id)
    except _Unreachable:
        return _emit_error(
            code="agent_unreachable",
            message="cannot reach the agent process",


        )
    except _EngineNotFound as error:
        return _emit_error(
            code="not_found",
            message=str(error),


        )
    except _EngineInvalidArgument as error:
        return _emit_error(
            code="invalid_argument",
            message=str(error),


        )
    except _EngineFailed as error:
        return _emit_error(
            code="internal",
            message=f"the agent process could not serve 'attempts export': {error}",


            details={"stored": False},
        )
    manifest = projection.get("manifest")
    assert isinstance(manifest, list)
    known = {
        str(entry.get("path"))
        for entry in manifest
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    requested = parsed.arguments.get("paths", [])
    for path in requested:
        if not _is_export_path(path) or path not in known:
            return _emit_error(
                code="not_found",
                message=f"unknown archive path {path!r} for attempt {attempt_id!r}",
            )
    entries = [
        entry
        for entry in manifest
        if isinstance(entry, dict) and (not requested or entry.get("path") in requested)
    ]
    data_dir = _export_data_dir()
    run_dir = attempt_history_module.find_run_directory(data_dir, attempt_id)
    if run_dir is None:
        return _emit_error(
            code="not_found",
            message=f"unknown attempt {attempt_id!r}",
        )
    live: dict[str, bytes] = {}
    for entry in entries:
        assert isinstance(entry, dict)
        relative = str(entry.get("path"))
        full = run_dir / relative
        try:
            if full.is_symlink() or not full.is_file():
                raise OSError(f"archive path {relative!r} is not a regular file")
            live[relative] = full.read_bytes()
        except OSError as error:
            return _emit_error(
                code="internal",
                message=f"the agent process could not serve 'attempts export': {error}",
                details={"stored": False},
            )
    attempt = projection.get("attempt")
    assert isinstance(attempt, dict)
    try:
        attempt_history_module.write_export_tar(
            sys.stdout.buffer,
            attempt=attempt,
            entries=entries,
            read_bytes=live.__getitem__,
        )
        sys.stdout.buffer.flush()
    except (OSError, KeyError) as error:
        return _emit_error(
            code="internal",
            message=f"the agent process could not serve 'attempts export': {error}",


            details={"stored": False},
        )
    return EXIT_OK


def _is_export_path(path: object) -> bool:
    """Whether one ``--path`` value is shaped like an archive member."""

    if not isinstance(path, str) or not path:
        return False
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return False
    return True


def _run_attempts_show(parsed: shared.ParsedCall) -> int:
    """Serve ``attempts show`` from the engine's history projection."""

    attempt_id = parsed.arguments.get("attempt_id")
    assert isinstance(attempt_id, str)
    try:
        result = _fetch_attempts_show(attempt_id)
    except _Unreachable:
        return _emit_error(
            code="agent_unreachable",
            message="cannot reach the agent process",


        )
    except _EngineNotFound as error:
        return _emit_error(
            code="not_found",
            message=str(error),


        )
    except _EngineInvalidArgument as error:
        return _emit_error(
            code="invalid_argument",
            message=str(error),


        )
    except _EngineFailed as error:
        return _emit_error(
            code="internal",
            message=f"the agent process could not serve 'attempts show': {error}",


            details={"stored": False},
        )
    return _emit_ok(result)


# ---------------------------------------------------------------------------
# Contract mutation runners (stop, resume) and command lookup
# ---------------------------------------------------------------------------


def _mutation_error(
    parsed: shared.ParsedCall, reply: dict
) -> int:
    """Map one answered contract rejection to its error code and exit."""

    kind = reply.get("error_kind")
    message = str(reply.get("error", "the agent process refused the command"))
    repository = reply.get("repository")
    details: dict | None = None
    raw_details = reply.get("details")
    if isinstance(raw_details, dict):
        details = dict(raw_details)
    if kind == "repository_mismatch":
        resolved = dict(details) if details is not None else {}
        resolved.setdefault("repository", repository)
        return _emit_error(
            code="repository_mismatch",
            message=message,


            request_id=parsed.request_id,
            details=resolved,
        )
    if kind == "payload_mismatch":
        return _emit_error(
            code="payload_mismatch",
            message=message,


            request_id=parsed.request_id,
            details=details,
        )
    if kind == "invalid_argument":
        return _emit_error(
            code="invalid_argument",
            message=message,


            request_id=parsed.request_id,
            details=details,
        )
    if kind == "rejected":
        return _emit_error(
            code="rejected",
            message=message,


            request_id=parsed.request_id,
            details=details,
        )
    if kind == "not_found":
        return _emit_error(
            code="not_found",
            message=message,
            request_id=parsed.request_id,
            details=details,
        )
    if kind == "internal":
        stored: dict | None = None
        if reply.get("stored") is False:
            stored = {"stored": False}
        elif isinstance(details, dict) and details.get("stored") is False:
            stored = {"stored": False}
        return _emit_error(
            code="internal",
            message=message,


            request_id=parsed.request_id,
            details=stored,
        )
    # No machine-readable kind: an engine exception of unknown shape.
    return _emit_error(
        code="internal",
        message=message,
        request_id=parsed.request_id,
        details={"stored": False} if reply.get("stored") is False else None,
    )


def _mutation_socket_op(parsed: shared.ParsedCall) -> tuple[str, dict]:
    """Map one parsed mutation onto its socket operation and payload."""

    if parsed.operation == "stop":
        if "after" in parsed.arguments:
            return "stop_after", {"after": int(parsed.arguments["after"], 10)}
        return "stop", {}
    if parsed.operation == "resume":
        return "resume", {}
    if parsed.operation == "next-issue":
        return "next_issue", {"issue": int(parsed.arguments["issue"], 10)}
    if parsed.operation == "handoff":
        return "handoff", {}
    if parsed.operation == "recovery-retry":
        return "recovery_retry", {"attempt_id": parsed.arguments["attempt_id"]}
    if parsed.operation == "recovery-release":
        payload: dict = {"attempt_id": parsed.arguments["attempt_id"]}
        if "saved_at" in parsed.arguments:
            payload["saved_at"] = parsed.arguments["saved_at"]
        return "recovery_release", payload
    if parsed.operation == "errors-reset":
        return "errors_reset", {}
    raise AssertionError(f"not a contract mutation: {parsed.operation}")


def _run_contract_mutation(parsed: shared.ParsedCall, op: str | None = None) -> int:
    """Serve one contract mutation through the control socket."""

    assert parsed.request_id is not None and parsed.expect_repo is not None
    if op is None:
        op, extra = _mutation_socket_op(parsed)
    else:
        # Explicit-op callers ("stop", "resume") carry no extra payload;
        # every other mutation maps through _mutation_socket_op above.
        extra = {}
    try:
        reply = _send_contract_mutation(op, parsed.request_id, parsed.expect_repo, extra)
    except _Unreachable:
        return _emit_error(
            code="agent_unreachable",
            message=(
                "cannot reach the agent process; the outcome is unknown,"
                " retry with the same request ID"
            ),


            request_id=parsed.request_id,
        )
    except _EngineFailed as error:
        return _emit_error(
            code="internal",
            message=f"the agent process could not serve '{parsed.operation}': {error}",
            request_id=parsed.request_id,
            details={"stored": False},
        )
    if not reply.get("ok"):
        return _mutation_error(parsed, reply)
    command = reply.get("command")
    problem = _command_problem(command)
    if problem is not None:
        return _emit_error(
            code="internal",
            message=(
                f"the agent process answered malformed contract command: {problem}"
            ),


            request_id=parsed.request_id,
        )
    assert isinstance(command, dict)
    replayed = reply.get("replayed")
    if not isinstance(replayed, bool):
        return _emit_error(
            code="internal",
            message="the agent process answered malformed contract command:"
            " replayed is not a boolean",


            request_id=parsed.request_id,
        )
    return _emit_ok({"command": command, "replayed": replayed})


def _run_contract_command(request_id: str) -> int:
    """Serve one contract ``command`` lookup through the control socket."""

    try:
        reply = _send_contract_command(request_id)
    except _Unreachable:
        return _emit_error(
            code="agent_unreachable",
            message="cannot reach the agent process",


        )
    except _EngineFailed as error:
        return _emit_error(
            code="internal",
            message=f"the agent process could not serve 'command': {error}",


            details={"stored": False},
        )
    if not reply.get("ok"):
        kind = reply.get("error_kind")
        message = str(reply.get("error", "command lookup is unavailable"))
        if kind == "not_found":
            return _emit_error(
                code="not_found",
                message=message,
            )
        if kind == "invalid_argument":
            return _emit_error(
                code="invalid_argument",
                message=message,
            )
        stored: dict | None = None
        if reply.get("stored") is False:
            stored = {"stored": False}
        return _emit_error(
            code="internal",
            message=message,


            details=stored,
        )
    command = reply.get("command")
    problem = _command_problem(command)
    if problem is not None:
        return _emit_error(
            code="internal",
            message=(
                f"the agent process answered malformed contract command: {problem}"
            ),


        )
    assert isinstance(command, dict)
    return _emit_ok({"command": command})


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _request_id_for_error(parsed: shared.ParsedCall | None) -> str | None:
    """Echo the request ID for error envelopes (mutations carry one)."""

    if parsed is not None and parsed.request_id is not None:
        return parsed.request_id
    return None


def main(argv: list[str]) -> int:
    """Contract CLI entry point; ``argv`` starts with ``--contract``."""

    if argv[2:3] == ["--offline"]:
        result = shared.parse_argv(argv, capabilities=shared.OPERATOR_CAPABILITIES)
        if isinstance(result, shared.ArgvFailure):
            return _emit_failure(result)
        return _run_offline(result)
    early = shared.parse_argv(argv, capabilities=shared.OPERATOR_CAPABILITIES)
    if isinstance(early, shared.ArgvFailure) and early.code in (
        "usage",
        "contract_mismatch",
    ):
        return _emit_failure(early)
    if isinstance(early, shared.ParsedCall) and early.operation == "identity":
        return _run_identity()
    try:
        identity = _fetch_identity()
    except _Unreachable:
        if isinstance(early, shared.ParsedCall):
            is_mutation = early.operation in shared.MUTATIONS
            request_id = early.request_id
        else:
            # A value failure on a mutation still leaves the outcome unknown.
            is_mutation = True
            request_id = early.request_id
        return _emit_error(
            code="agent_unreachable",
            message=(
                "cannot reach the agent process; the outcome is unknown,"
                " retry with the same request ID"
                if is_mutation
                else "cannot reach the agent process"
            ),
            request_id=request_id,
        )
    except _EngineFailed as error:
        operation = (
            early.operation if isinstance(early, shared.ParsedCall) else "contract"
        )
        return _emit_error(
            code="internal",
            message=f"the agent process could not serve '{operation}': {error}",
            request_id=early.request_id,
            details={"stored": False},
        )
    parsed = shared.parse_argv(argv, capabilities=identity.get("capabilities", []))
    if isinstance(parsed, shared.ArgvFailure):
        return _emit_failure(parsed)
    if parsed.operation == "identity":
        return _run_identity()
    if parsed.operation == "status":
        return _run_status()
    if parsed.operation == "attempts-list":
        return _run_attempts_list(parsed)
    if parsed.operation == "attempts-show":
        return _run_attempts_show(parsed)
    if parsed.operation == "attempts-export":
        return _run_attempts_export(parsed)
    if parsed.operation in (
        "stop",
        "resume",
        "next-issue",
        "handoff",
        "recovery-retry",
        "recovery-release",
        "errors-reset",
    ):
        return _run_contract_mutation(parsed)
    if parsed.operation == "command":
        return _run_contract_command(parsed.arguments["request_id"])
    # Forward-compatible fallback: the shared grammar only produces operations
    # listed above, so this is unreachable today. A future operation answers
    # ``unsupported`` without storing anything.
    return _unsupported(
        parsed.operation,
        capability=parsed.capability,
        request_id=_request_id_for_error(parsed),
        message=f"operation '{parsed.operation}' is not yet served by this producer",
        details={"operation": parsed.operation},
    )


# ---------------------------------------------------------------------------
# Offline reads (``agentctl --contract 1 --offline <read>``)
# ---------------------------------------------------------------------------


def _run_offline(parsed: shared.ParsedCall) -> int:
    """Serve one offline read from the stopped volume without the engine.

    The call never takes the volume lock and never writes: the control
    snapshot comes from the volume-direct reader, identity from
    ``state/instance.json``, the halt from ``state/halt.json`` and
    attempts from the ``logs/`` archives. A volume a live engine still
    holds answers ``rejected`` with ``details.reason: "instance_running"``;
    an unreadable volume answers ``internal``; mutations never reach here
    (the entry point answers ``unsupported`` first).
    """

    from simple_coding_agent import offline as offline_module
    from simple_coding_agent.control_offline import OfflineReadError
    from simple_coding_agent.volume_lock import VolumeLockError, is_volume_locked

    # Grammar and values are already validated by ``shared.parse_argv`` in
    # :func:`main`; mutations never reach here (the shared parser answers
    # ``unsupported`` with ``details.mode: "offline"`` first).
    data_dir = _export_data_dir()
    try:
        locked = is_volume_locked(data_dir)
    except VolumeLockError as error:
        return _emit_error(
            code="internal",
            message=f"the volume lock could not be probed: {error}",


            request_id=_request_id_for_error(parsed),
            details={"stored": False},
        )
    if locked:
        return _emit_error(
            code="rejected",
            message="the volume is still held by a live engine",


            details={"reason": "instance_running"},
        )
    if parsed.operation == "attempts-export":
        return _run_offline_attempts_export(parsed, data_dir)
    try:
        if parsed.operation == "identity":
            return _emit_ok(offline_module.offline_identity(data_dir))
        if parsed.operation == "status":
            return _emit_ok(offline_module.offline_status(data_dir))
        if parsed.operation == "command":
            request_id = parsed.arguments.get("request_id")
            assert isinstance(request_id, str)
            return _emit_ok(
                {"command": offline_module.offline_command(data_dir, request_id)}
            )
        if parsed.operation == "attempts-list":
            raw_limit = parsed.arguments.get("limit")
            before = parsed.arguments.get("before")
            limit = int(raw_limit, 10) if raw_limit is not None else None
            return _emit_ok(
                offline_module.offline_attempts_list(data_dir, limit, before)
            )
        if parsed.operation == "attempts-show":
            attempt_id = parsed.arguments.get("attempt_id")
            assert isinstance(attempt_id, str)
            return _emit_ok(
                offline_module.offline_attempts_show(data_dir, attempt_id)
            )
    except OfflineReadError as error:
        return _emit_error(
            code="internal",
            message=f"the stopped volume could not be read: {error}",
            details={"stored": False},
        )
    except AttemptHistoryError as error:
        return _offline_history_error(error)
    return _unsupported(
        parsed.operation,
        capability=parsed.capability,
        request_id=_request_id_for_error(parsed),
        message=f"operation '{parsed.operation}' is not available offline",
        details={"mode": "offline"},
    )


def _offline_history_error(error: AttemptHistoryError) -> int:
    """Map one offline history rejection onto its contract error and exit."""

    if error.code == "not_found":
        return _emit_error(
            code="not_found",
            message=str(error),


        )
    if error.code == "invalid_argument":
        return _emit_error(
            code="invalid_argument",
            message=str(error),


        )
    return _emit_error(
        code="internal",
        message=f"the stopped volume could not be read: {error}",
        details={"stored": False},
    )


def _run_offline_attempts_export(
    parsed: shared.ParsedCall, data_dir: Path
) -> int:
    """Stream one offline ``attempts export`` tar to stdout, volume-direct.

    The summary and export manifest come from the volume-direct
    projection; the file bytes are read from the archive run directory,
    so the tar never crosses the control socket. ``manifest.json`` is the
    first member, like the online export.
    """

    from simple_coding_agent import attempt_history as attempt_history_module
    from simple_coding_agent import offline as offline_module
    from simple_coding_agent.control_offline import OfflineReadError

    attempt_id = parsed.arguments.get("attempt_id")
    assert isinstance(attempt_id, str)
    try:
        projection = offline_module.offline_attempts_export(data_dir, attempt_id)
    except AttemptHistoryError as error:
        return _offline_history_error(error)
    except OfflineReadError as error:
        return _emit_error(
            code="internal",
            message=f"the stopped volume could not be read: {error}",


            details={"stored": False},
        )
    manifest = projection.get("manifest")
    assert isinstance(manifest, list)
    known = {
        str(entry.get("path"))
        for entry in manifest
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    requested = parsed.arguments.get("paths", [])
    for path in requested:
        if not _is_export_path(path) or path not in known:
            return _emit_error(
                code="not_found",
                message=f"unknown archive path {path!r} for attempt {attempt_id!r}",
            )
    entries = [
        entry
        for entry in manifest
        if isinstance(entry, dict) and (not requested or entry.get("path") in requested)
    ]
    run_dir = attempt_history_module.find_run_directory(data_dir, attempt_id)
    if run_dir is None:
        return _emit_error(
            code="not_found",
            message=f"unknown attempt {attempt_id!r}",


        )
    live: dict[str, bytes] = {}
    for entry in entries:
        assert isinstance(entry, dict)
        relative = str(entry.get("path"))
        full = run_dir / relative
        try:
            if full.is_symlink() or not full.is_file():
                raise OSError(f"archive path {relative!r} is not a regular file")
            live[relative] = full.read_bytes()
        except OSError as error:
            return _emit_error(
                code="internal",
                message=f"the stopped volume could not be read: {error}",
                details={"stored": False},
            )
    attempt = projection.get("attempt")
    assert isinstance(attempt, dict)
    try:
        attempt_history_module.write_export_tar(
            sys.stdout.buffer,
            attempt=attempt,
            entries=entries,
            read_bytes=live.__getitem__,
        )
        sys.stdout.buffer.flush()
    except (OSError, KeyError) as error:
        return _emit_error(
            code="internal",
            message=f"the stopped volume could not be read: {error}",
            details={"stored": False},
        )
    return EXIT_OK


def _run_identity() -> int:
    """Serve ``identity`` from the engine's read-only identity operation."""

    try:
        identity = _fetch_identity()
    except _Unreachable:
        return _emit_error(
            code="agent_unreachable",
            message="cannot reach the agent process",


        )
    except _EngineFailed as error:
        return _emit_error(
            code="internal",
            message=f"the agent process could not serve 'identity': {error}",


            details={"stored": False},
        )
    return _emit_ok(
        {
            "backend": identity["backend"],
            "agent_version": identity["agent_version"],
            "capabilities": sorted(identity["capabilities"]),
            "contract": {
                "major": identity["contract"]["major"],
                "minor": identity["contract"]["minor"],
            },
            "mode": identity["mode"],
            "process_started_at": identity["process_started_at"],
            "repository": identity["repository"],
        }
    )


def _run_status() -> int:
    """Serve ``status`` from the engine's prepared-state projection."""

    try:
        status = _fetch_status()
    except _Unreachable:
        return _emit_error(
            code="agent_unreachable",
            message="cannot reach the agent process",


        )
    except _EngineFailed as error:
        return _emit_error(
            code="internal",
            message=f"the agent process could not serve 'status': {error}",


            details={"stored": False},
        )
    return _emit_ok(status)


if __name__ == "__main__":  # pragma: no cover - exercised via agentctl.main
    raise SystemExit(main(sys.argv[1:]))
