"""Operator contract major-1 adapter (``agentctl --contract 1 ...``).

A contract consumer runs ``agentctl --contract 1 <operation>`` through
``docker exec`` without a TTY and always gets exactly one JSON envelope
with a trailing newline on stdout plus the contract's distinct exit
code. This module owns the contract grammar, the check order, envelope
rendering and exit codes; it reaches the engine only through the
existing control socket, plus the read-only ``identity`` socket
operation. Nothing here is shared with the legacy path: without
``--contract`` :func:`agentctl.main` keeps today's grammar, replies,
exit codes and stderr byte-compatible.

Check order (spec section 4.1): ``--contract`` major, argv grammar
(``usage``), capability (``unsupported``), argument values
(``invalid_argument``); expected repository, idempotency, state and
commit run on the engine side in later tickets. Operations those later
tickets implement are parsed and gated here but answer ``unsupported``
until their projection exists; an unreachable engine answers
``agent_unreachable`` and an engine-side failure answers ``internal``.

Operand shapes and reply envelopes follow the prose and the
``examples/agentctl/`` fixtures of agent-operator-contract ``06ee46c``.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from simple_coding_agent.control_server import (
    ControlUnavailableError,
    UnreadableReplyError,
    resolve_socket_path,
    send_request,
)

#: The one contract major this producer serves.
CONTRACT_MAJOR = 1
#: The contract minor reported in every envelope.
CONTRACT_MINOR = 0
#: The ``contract`` member of every envelope.
CONTRACT_VERSION = f"{CONTRACT_MAJOR}.{CONTRACT_MINOR}"

#: Backend name reported by ``identity``.
BACKEND = "simple-coding-agent"

#: Optional capabilities this backend always advertises, sorted.
BASE_CAPABILITIES = sorted(
    [
        "errors-reset",
        "next-issue",
        "operator-handoff",
        "recovery-release",
        "recovery-retry",
        "stop-after",
    ]
)

#: Declaration-only capability advertised exactly when a Worker id is set.
POOL_CAPABILITY = "pool"

#: Consumer request IDs must match this (spec section 6.2).
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

#: ``stop --after`` countdown bounds (spec section 6.2).
STOP_AFTER_MIN = 1
STOP_AFTER_MAX = 999

#: ``attempts list`` page-size bounds (spec section 6.1).
ATTEMPTS_LIMIT_MIN = 1
ATTEMPTS_LIMIT_MAX = 100

#: Socket timeout for contract reads (consumer default; the engine's own
#: budget for ``identity`` is 3 s, served lock-free without I/O).
SOCKET_TIMEOUT_SECONDS = 10.0

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_USAGE = 2
EXIT_UNSUPPORTED = 3
EXIT_CONTRACT_MISMATCH = 4
# Exit 5 (`not_found`) belongs to the `command` ticket; it is defined
# there when the lookup lands.
EXIT_AGENT_UNREACHABLE = 6
EXIT_INTERNAL = 7

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


def _emit_ok(result: dict) -> int:
    """Write one success envelope; always exit 0."""

    print(
        json.dumps(
            {"contract": CONTRACT_VERSION, "ok": True, "result": result},
            separators=(",", ":"),
        )
    )
    return EXIT_OK


def _emit_error(
    *,
    code: str,
    message: str,
    exit_code: int,
    retryable: bool,
    request_id: str | None = None,
    details: dict | None = None,
) -> int:
    """Write one error envelope with the contract's exit code."""

    error: dict = {"code": code}
    if details is not None:
        error["details"] = details
    error["message"] = message
    if request_id is not None:
        error["request_id"] = request_id
    error["retryable"] = retryable
    print(
        json.dumps(
            {"contract": CONTRACT_VERSION, "error": error, "ok": False},
            separators=(",", ":"),
        )
    )
    return exit_code


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
        exit_code=EXIT_UNSUPPORTED,
        retryable=False,
        request_id=request_id,
        details=resolved,
    )


# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------


class _Usage(Exception):
    """Malformed argv (contract ``usage``, exit 2)."""


@dataclass(frozen=True)
class ParsedOperation:
    """One well-formed contract operation with its raw option values."""

    #: Canonical name, e.g. ``"stop --after"`` or ``"attempts list"``.
    name: str
    #: Whether the call carries a request ID (a mutation).
    mutation: bool
    #: Optional capability gating this operation, if any.
    capability: str | None
    #: Raw ``--request-id`` value, if the flag was present.
    request_id: str | None = None
    #: Raw ``--expect-repo`` value, if the flag was present.
    expect_repo: str | None = None
    #: Raw operation-specific values (``after``, ``issue``, ...).
    args: dict = field(default_factory=dict)


def _split_flag(token: str) -> tuple[str, str | None]:
    """Split ``--flag=value``; a bare ``--flag`` yields ``None``."""

    if "=" in token:
        flag, _, value = token.partition("=")
        return flag, value
    return token, None


def _take_value(
    operation: str, flag: str, tokens: list[str], index: int
) -> tuple[str, int]:
    """Take the value of one ``--flag value`` (or ``--flag=value``)."""

    _, inline = _split_flag(tokens[index])
    if inline is not None:
        return inline, index + 1
    if index + 1 >= len(tokens):
        raise _Usage(f"argument {flag}: expected one argument")
    # A dash-leading next token is taken as the value and validated
    # later, so `--after -3` is `invalid_argument`, not a missing value.
    return tokens[index + 1], index + 2


def _parse_mutation_options(
    operation: str,
    tokens: list[str],
    extra: dict[str, str | None],
) -> tuple[str | None, str | None, dict]:
    """Parse the option tail of a mutation.

    ``extra`` maps each additional accepted flag to its inline-or-taken
    raw value holder (``None`` means flag absent). Returns
    ``(request_id, expect_repo, values)``; missing ``--request-id`` or
    ``--expect-repo`` is malformed argv (``usage``), since every mutation
    requires both.
    """

    request_id: str | None = None
    expect_repo: str | None = None
    values: dict[str, str | None] = dict(extra)
    seen: set[str] = set()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise _Usage(f"unexpected argument '{token}' for '{operation}'")
        flag, _ = _split_flag(token)
        if flag in ("--request-id", "--expect-repo") or flag in values:
            if flag in seen:
                raise _Usage(f"duplicate argument '{flag}' for '{operation}'")
            seen.add(flag)
            value, index = _take_value(operation, flag, tokens, index)
            if flag == "--request-id":
                request_id = value
            elif flag == "--expect-repo":
                expect_repo = value
            else:
                values[flag] = value
        else:
            raise _Usage(f"unexpected argument '{token}' for '{operation}'")
    if request_id is None:
        raise _Usage(f"missing --request-id for '{operation}'")
    if expect_repo is None:
        raise _Usage(f"missing --expect-repo for '{operation}'")
    return request_id, expect_repo, values


def _parse_operation(operation: str, tokens: list[str]) -> ParsedOperation:
    """Parse one operation and its tokens; raise :class:`_Usage` if malformed."""

    if operation == "identity":
        if tokens:
            raise _Usage(f"unexpected argument '{tokens[0]}' for 'identity'")
        return ParsedOperation(name="identity", mutation=False, capability=None)
    if operation == "status":
        if tokens:
            raise _Usage(f"unexpected argument '{tokens[0]}' for 'status'")
        return ParsedOperation(name="status", mutation=False, capability=None)
    if operation == "command":
        if not tokens:
            raise _Usage("missing request ID for 'command'")
        if len(tokens) > 1:
            raise _Usage(f"unexpected argument '{tokens[1]}' for 'command'")
        request_id = tokens[0]
        if request_id.startswith("--"):
            raise _Usage(f"unexpected argument '{request_id}' for 'command'")
        return ParsedOperation(
            name="command",
            mutation=False,
            capability=None,
            args={"request_id": request_id},
        )
    if operation == "attempts":
        return _parse_attempts(tokens)
    if operation == "stop":
        return _parse_stop(tokens)
    if operation == "resume":
        request_id, expect_repo, _ = _parse_mutation_options("resume", tokens, {})
        return ParsedOperation(
            name="resume",
            mutation=True,
            capability=None,
            request_id=request_id,
            expect_repo=expect_repo,
        )
    if operation == "next":
        return _parse_next(tokens)
    if operation == "handoff":
        return _parse_handoff(tokens)
    if operation == "recovery":
        return _parse_recovery(tokens)
    if operation == "errors":
        return _parse_errors(tokens)
    raise _Usage(f"unknown operation '{operation}'")


def _parse_attempts(tokens: list[str]) -> ParsedOperation:
    if not tokens:
        raise _Usage("missing subcommand for 'attempts'")
    sub, rest = tokens[0], tokens[1:]
    if sub == "list":
        limit: str | None = None
        before: str | None = None
        seen: set[str] = set()
        index = 0
        while index < len(rest):
            token = rest[index]
            if not token.startswith("--"):
                raise _Usage(
                    f"unexpected argument '{token}' for 'attempts list'"
                )
            flag, _ = _split_flag(token)
            if flag not in ("--limit", "--before"):
                raise _Usage(
                    f"unexpected argument '{token}' for 'attempts list'"
                )
            if flag in seen:
                raise _Usage(
                    f"duplicate argument '{flag}' for 'attempts list'"
                )
            seen.add(flag)
            value, index = _take_value("attempts list", flag, rest, index)
            if flag == "--limit":
                limit = value
            else:
                before = value
        return ParsedOperation(
            name="attempts list",
            mutation=False,
            capability=None,
            args={"limit": limit, "before": before},
        )
    if sub == "show":
        return _parse_attempts_with_id("attempts show", rest, allow_path=False)
    if sub == "export":
        return _parse_attempts_with_id("attempts export", rest, allow_path=True)
    raise _Usage(f"unknown operation 'attempts {sub}'")


def _parse_attempts_with_id(
    name: str, tokens: list[str], *, allow_path: bool
) -> ParsedOperation:
    accepted = {"--path"} if allow_path else set()
    attempt_id: str | None = None
    paths: list[str] = []
    options_seen = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--"):
            options_seen = True
            flag, _ = _split_flag(token)
            if flag not in accepted:
                raise _Usage(f"unexpected argument '{token}' for '{name}'")
            value, index = _take_value(name, flag, tokens, index)
            paths.append(value)
        else:
            if attempt_id is not None or options_seen:
                raise _Usage(f"unexpected argument '{token}' for '{name}'")
            attempt_id = token
            index += 1
    if attempt_id is None:
        raise _Usage(f"missing attempt ID for '{name}'")
    args: dict = {"attempt_id": attempt_id}
    if allow_path:
        args["paths"] = paths
    return ParsedOperation(name=name, mutation=False, capability=None, args=args)


def _parse_stop(tokens: list[str]) -> ParsedOperation:
    after: str | None = None
    rest: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        flag, _ = _split_flag(token)
        if flag == "--after":
            if after is not None:
                raise _Usage("duplicate argument '--after' for 'stop'")
            after, index = _take_value("stop", "--after", tokens, index)
        else:
            rest.append(token)
            index += 1
    request_id, expect_repo, _ = _parse_mutation_options("stop", rest, {})
    if after is None:
        return ParsedOperation(
            name="stop",
            mutation=True,
            capability=None,
            request_id=request_id,
            expect_repo=expect_repo,
        )
    return ParsedOperation(
        name="stop --after",
        mutation=True,
        capability="stop-after",
        request_id=request_id,
        expect_repo=expect_repo,
        args={"after": after},
    )


def _parse_next(tokens: list[str]) -> ParsedOperation:
    if not tokens:
        raise _Usage("missing subcommand for 'next'")
    if tokens[0] != "issue":
        raise _Usage(f"unknown operation 'next {tokens[0]}'")
    rest = tokens[1:]
    issue: str | None = None
    tail: list[str] = []
    for token in rest:
        if token.startswith("--") or issue is not None:
            tail.append(token)
        else:
            issue = token
    if issue is None:
        raise _Usage("missing issue number for 'next issue'")
    request_id, expect_repo, _ = _parse_mutation_options("next issue", tail, {})
    return ParsedOperation(
        name="next issue",
        mutation=True,
        capability="next-issue",
        request_id=request_id,
        expect_repo=expect_repo,
        args={"issue": issue},
    )


def _parse_handoff(tokens: list[str]) -> ParsedOperation:
    if not tokens:
        raise _Usage("missing subcommand for 'handoff'")
    if tokens[0] != "now":
        raise _Usage(f"unknown operation 'handoff {tokens[0]}'")
    request_id, expect_repo, _ = _parse_mutation_options(
        "handoff now", tokens[1:], {}
    )
    return ParsedOperation(
        name="handoff now",
        mutation=True,
        capability="operator-handoff",
        request_id=request_id,
        expect_repo=expect_repo,
    )


def _split_attempt_id(name: str, tokens: list[str]) -> tuple[str, list[str]]:
    """Split one leading positional attempt ID from the option tail."""

    attempt_id: str | None = None
    tail: list[str] = []
    for token in tokens:
        if token.startswith("--") or attempt_id is not None:
            tail.append(token)
        else:
            attempt_id = token
    if attempt_id is None:
        raise _Usage(f"missing attempt ID for '{name}'")
    return attempt_id, tail


def _parse_recovery(tokens: list[str]) -> ParsedOperation:
    if not tokens:
        raise _Usage("missing subcommand for 'recovery'")
    sub, rest = tokens[0], tokens[1:]
    if sub == "retry":
        attempt_id, tail = _split_attempt_id("recovery retry", rest)
        request_id, expect_repo, _ = _parse_mutation_options(
            "recovery retry", tail, {}
        )
        return ParsedOperation(
            name="recovery retry",
            mutation=True,
            capability="recovery-retry",
            request_id=request_id,
            expect_repo=expect_repo,
            args={"attempt_id": attempt_id},
        )
    if sub == "release":
        attempt_id, tail = _split_attempt_id("recovery release", rest)
        request_id, expect_repo, values = _parse_mutation_options(
            "recovery release", tail, {"--saved-at": None}
        )
        saved_at = values["--saved-at"]
        if saved_at is None:
            raise _Usage("missing --saved-at for 'recovery release'")
        return ParsedOperation(
            name="recovery release",
            mutation=True,
            capability="recovery-release",
            request_id=request_id,
            expect_repo=expect_repo,
            args={"attempt_id": attempt_id, "saved_at": saved_at},
        )
    raise _Usage(f"unknown operation 'recovery {sub}'")


def _parse_errors(tokens: list[str]) -> ParsedOperation:
    if not tokens:
        raise _Usage("missing subcommand for 'errors'")
    if tokens[0] != "reset":
        raise _Usage(f"unknown operation 'errors {tokens[0]}'")
    request_id, expect_repo, _ = _parse_mutation_options(
        "errors reset", tokens[1:], {}
    )
    return ParsedOperation(
        name="errors reset",
        mutation=True,
        capability="errors-reset",
        request_id=request_id,
        expect_repo=expect_repo,
    )


# ---------------------------------------------------------------------------
# Argument values (``invalid_argument``, exit 1)
# ---------------------------------------------------------------------------


class _InvalidArgument(Exception):
    """Well-formed argv with an invalid value."""


def _check_request_id(value: str | None) -> None:
    if not isinstance(value, str) or REQUEST_ID_RE.fullmatch(value) is None:
        raise _InvalidArgument(
            f"invalid request ID {value!r}: must match"
            " ^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
        )


def _check_positive_int(raw: str, what: str) -> int:
    if not re.fullmatch(r"[0-9]+", raw):
        raise _InvalidArgument(
            f"invalid {what} {raw!r}: must be a positive integer"
        )
    return int(raw, 10)


def _check_int_range(raw: str, what: str, low: int, high: int) -> int:
    """Validate one bounded integer option; the message names the range."""

    try:
        count = _check_positive_int(raw, what)
    except _InvalidArgument:
        raise _InvalidArgument(
            f"invalid {what} {raw!r}: must be an integer {low}..{high}"
        ) from None
    if not low <= count <= high:
        raise _InvalidArgument(
            f"invalid {what} {raw!r}: must be an integer {low}..{high}"
        )
    return count


def _check_values(parsed: ParsedOperation) -> None:
    """Validate raw option values; engine-side checks stay on the engine."""

    if parsed.mutation:
        _check_request_id(parsed.request_id)
        if not isinstance(parsed.expect_repo, str) or not parsed.expect_repo:
            raise _InvalidArgument("invalid --expect-repo: must not be empty")
    if parsed.name == "stop --after":
        raw = parsed.args["after"]
        try:
            count = _check_positive_int(raw, "--after value")
        except _InvalidArgument:
            raise _InvalidArgument(
                f"invalid --after value {raw!r}: must be an integer"
                f" {STOP_AFTER_MIN}..{STOP_AFTER_MAX}"
            ) from None
        if not STOP_AFTER_MIN <= count <= STOP_AFTER_MAX:
            raise _InvalidArgument(
                f"invalid --after value {raw!r}: must be an integer"
                f" {STOP_AFTER_MIN}..{STOP_AFTER_MAX}"
            )
    elif parsed.name == "next issue":
        raw = parsed.args["issue"]
        number = _check_positive_int(raw, "issue number")
        if number <= 0:  # pragma: no cover - unreachable via the regex
            raise _InvalidArgument(
                f"invalid issue number {raw!r}: must be a positive integer"
            )
    elif parsed.name == "attempts list":
        limit = parsed.args.get("limit")
        if limit is not None:
            try:
                count = _check_positive_int(limit, "--limit value")
            except _InvalidArgument:
                raise _InvalidArgument(
                    f"invalid --limit value {limit!r}: must be an integer"
                    f" {ATTEMPTS_LIMIT_MIN}..{ATTEMPTS_LIMIT_MAX}"
                ) from None
            if not ATTEMPTS_LIMIT_MIN <= count <= ATTEMPTS_LIMIT_MAX:
                raise _InvalidArgument(
                    f"invalid --limit value {limit!r}: must be an integer"
                    f" {ATTEMPTS_LIMIT_MIN}..{ATTEMPTS_LIMIT_MAX}"
                )
        before = parsed.args.get("before")
        if before is not None and not before:
            raise _InvalidArgument("invalid --before value: must not be empty")
    elif parsed.name in ("attempts show", "attempts export"):
        if not parsed.args.get("attempt_id"):
            raise _InvalidArgument(
                f"invalid attempt ID for '{parsed.name}': must not be empty"
            )
        for path in parsed.args.get("paths", []):
            if not path:
                raise _InvalidArgument(
                    "invalid --path value: must not be empty"
                )
    elif parsed.name == "command":
        if not parsed.args.get("request_id"):
            raise _InvalidArgument(
                "invalid request ID for 'command': must not be empty"
            )
    elif parsed.name in ("recovery retry", "recovery release"):
        if not parsed.args.get("attempt_id"):
            raise _InvalidArgument(
                f"invalid attempt ID for '{parsed.name}': must not be empty"
            )
        if parsed.name == "recovery release" and not parsed.args.get("saved_at"):
            raise _InvalidArgument(
                "invalid --saved-at value: must not be empty"
            )


# ---------------------------------------------------------------------------
# Engine contact (control socket only)
# ---------------------------------------------------------------------------


class _Unreachable(Exception):
    """The engine process could not be reached (ambiguous for mutations)."""


class _EngineFailed(Exception):
    """The engine answered but could not serve the request."""


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
# Entry point
# ---------------------------------------------------------------------------


def _request_id_for_error(parsed: ParsedOperation | None) -> str | None:
    if parsed is not None and parsed.mutation:
        return parsed.request_id
    return None


def main(argv: list[str]) -> int:
    """Contract CLI entry point; ``argv`` starts with ``--contract``."""

    if len(argv) < 2:
        return _emit_error(
            code="usage",
            message="usage: missing contract major after '--contract'",
            exit_code=EXIT_USAGE,
            retryable=False,
        )
    if argv[1] != str(CONTRACT_MAJOR):
        return _emit_error(
            code="contract_mismatch",
            message=f"this instance speaks contract major {CONTRACT_MAJOR}",
            exit_code=EXIT_CONTRACT_MISMATCH,
            retryable=False,
            details={"supported": [CONTRACT_MAJOR]},
        )
    tokens = argv[2:]
    offline = False
    if tokens[:1] == ["--offline"]:
        offline = True
        tokens = tokens[1:]
    if not tokens:
        return _emit_error(
            code="usage",
            message="usage: missing operation after '--contract 1'",
            exit_code=EXIT_USAGE,
            retryable=False,
        )
    operation, rest = tokens[0], tokens[1:]
    try:
        parsed = _parse_operation(operation, rest)
    except _Usage as error:
        request_id = _scan_request_id(rest)
        return _emit_error(
            code="usage",
            message=f"usage: {error}",
            exit_code=EXIT_USAGE,
            retryable=False,
            request_id=request_id,
        )
    if offline:
        return _unsupported(
            parsed.name,
            capability=parsed.capability,
            request_id=_request_id_for_error(parsed),
            message=f"operation '{parsed.name}' is not available offline",
            details={"offline": True},
        )
    if parsed.name == "identity":
        return _run_identity()
    try:
        identity = _fetch_identity()
    except _Unreachable:
        return _emit_error(
            code="agent_unreachable",
            message=(
                "cannot reach the agent process; the outcome is unknown,"
                " retry with the same request ID"
                if parsed.mutation
                else "cannot reach the agent process"
            ),
            exit_code=EXIT_AGENT_UNREACHABLE,
            retryable=True,
            request_id=_request_id_for_error(parsed),
        )
    except _EngineFailed as error:
        return _emit_error(
            code="internal",
            message=f"the agent process could not serve '{parsed.name}': {error}",
            exit_code=EXIT_INTERNAL,
            retryable=True,
            request_id=_request_id_for_error(parsed),
            details={"stored": False},
        )
    if (
        parsed.capability is not None
        and parsed.capability not in identity.get("capabilities", [])
    ):
        return _unsupported(
            parsed.name,
            capability=parsed.capability,
            request_id=_request_id_for_error(parsed),
            message=f"this instance does not advertise {parsed.capability}",
            details={"capability": parsed.capability},
        )
    try:
        _check_values(parsed)
    except _InvalidArgument as error:
        return _emit_error(
            code="invalid_argument",
            message=str(error),
            exit_code=EXIT_REJECTED,
            retryable=False,
            request_id=_request_id_for_error(parsed),
        )
    # Later tickets project each operation onto the contract's shared
    # types (status snapshot, command records, attempt archives). Until
    # then the grammar and checks above are in place and the operation
    # answers ``unsupported`` without storing anything.
    return _unsupported(
        parsed.name,
        capability=parsed.capability,
        request_id=_request_id_for_error(parsed),
        message=f"operation '{parsed.name}' is not yet served by this producer",
        details={"operation": parsed.name},
    )


def _scan_request_id(tokens: list[str]) -> str | None:
    """Best-effort ``--request-id`` extraction for malformed argv errors."""

    for index, token in enumerate(tokens):
        flag, inline = _split_flag(token)
        if flag == "--request-id":
            if inline is not None:
                return inline
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
                return tokens[index + 1]
            return None
    return None


def _run_identity() -> int:
    """Serve ``identity`` from the engine's read-only identity operation."""

    try:
        identity = _fetch_identity()
    except _Unreachable:
        return _emit_error(
            code="agent_unreachable",
            message="cannot reach the agent process",
            exit_code=EXIT_AGENT_UNREACHABLE,
            retryable=True,
        )
    except _EngineFailed as error:
        return _emit_error(
            code="internal",
            message=f"the agent process could not serve 'identity': {error}",
            exit_code=EXIT_INTERNAL,
            retryable=True,
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


if __name__ == "__main__":  # pragma: no cover - exercised via agentctl.main
    raise SystemExit(main(sys.argv[1:]))
