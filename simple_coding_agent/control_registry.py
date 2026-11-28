"""Data-driven registry for operator control commands.

One entry per mutating operator command replaces the five hand-written
copies that used to live in ``agentctl`` (CLI function + parser entry),
``control_server`` (protocol method + reply listing rejection errors),
``lifecycle`` (live-state handler) and ``control`` (store submission).

Each :class:`CommandSpec` carries the command's socket ``op`` name, its
durable store ``kind``, how to parse its canonical payload, which
rejections the socket server answers without a control change, whether it
needs an active attempt, whether it must run through the live lifecycle
(pure store commands bypass it), and how to add its CLI parser.

Queries (``status``, ``command``, ``telemetry``, ``identity``,
``contract_status``, ``attempts_*``) intentionally stay out of the
registry: they are read-only snapshots, not commands.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CommandSpec:
    """One operator command's data-driven definition."""

    #: Socket ``op`` and the CLI dispatch key (``stop_after`` serves
    #: ``stop --after N``).
    op: str
    #: Durable store kind recorded in ``commands.kind``.
    kind: str
    #: Short human summary (used for CLI help).
    summary: str
    #: Exceptions answered as rejections (no control change).
    rejected: tuple[type[Exception], ...]
    #: Parse a raw socket request mapping into canonical submit kwargs
    #: (excluding ``request_id``). Raises one of ``rejected`` on bad input.
    parse_request: Callable[[dict], dict]
    #: Invoke the command against a lifecycle/control surface. ``request``
    #: is the raw socket request; ``contract_mode`` selects the
    #: operator-contract rule for ``resume``. Returns the committed record.
    invoke: Callable[[Any, str, dict, bool], Any]
    #: CLI route as argparse dest path, e.g. ``("stop",)`` or
    #: ``("handoff", "now")``.
    cli_route: tuple[str, ...]
    #: Whether the CLI for this command offers ``--expect-repo``.
    supports_expect_repo: bool
    #: Whether the command is rejected without an active attempt
    #: (today only ``handoff``).
    requires_active_attempt: bool
    #: Whether the command must run through the live lifecycle. Pure store
    #: commands set this to ``False`` and bypass the lifecycle entirely.
    needs_lifecycle: bool
    #: Add this command's argparse parser(s) under ``subparsers``.
    configure_parser: Callable[[Any], None] | None = None
    #: Extract the canonical submit kwargs from parsed CLI args.
    parse_cli_args: Callable[[Any], dict] | None = None
    #: Human label for ``submit_mutating`` (e.g. ``"stop --after 2"``).
    cli_kind_label: Callable[[dict], str] | None = None
    #: Build the socket request mapping for this command.
    build_request: Callable[..., dict] | None = None


def _stop_parse_request(request: dict) -> dict:
    return {}


def _resume_parse_request(request: dict) -> dict:
    return {}


def _handoff_parse_request(request: dict) -> dict:
    return {}


def _errors_reset_parse_request(request: dict) -> dict:
    return {}


def _stop_after_parse_request(request: dict) -> dict:
    from simple_coding_agent.control import parse_stop_after

    return {"after": parse_stop_after(request.get("after"))}


def _next_issue_parse_request(request: dict) -> dict:
    from simple_coding_agent.control import parse_next_issue

    return {"issue": parse_next_issue(request.get("issue"))}


def _recovery_retry_parse_request(request: dict) -> dict:
    from simple_coding_agent.recovery import parse_attempt_id

    return {"attempt_id": parse_attempt_id(request.get("attempt_id"))}


def _recovery_release_parse_request(request: dict) -> dict:
    from simple_coding_agent.recovery import parse_attempt_id, parse_saved_at

    return {
        "attempt_id": parse_attempt_id(request.get("attempt_id")),
        "saved_at": parse_saved_at(request.get("saved_at")),
    }


def _invoke_stop(control: Any, request_id: str, request: dict, contract_mode: bool) -> Any:
    return control.submit_stop(request_id)


def _invoke_stop_after(control: Any, request_id: str, request: dict, contract_mode: bool) -> Any:
    return control.submit_stop_after(request_id, request.get("after"))


def _invoke_next_issue(control: Any, request_id: str, request: dict, contract_mode: bool) -> Any:
    return control.submit_next_issue(request_id, request.get("issue"))


def _invoke_resume(control: Any, request_id: str, request: dict, contract_mode: bool) -> Any:
    if contract_mode:
        return control.submit_resume(request_id, reject_open_intake=True)
    return control.submit_resume(request_id)


def _invoke_handoff(control: Any, request_id: str, request: dict, contract_mode: bool) -> Any:
    return control.submit_handoff(request_id)


def _invoke_recovery_retry(control: Any, request_id: str, request: dict, contract_mode: bool) -> Any:
    return control.submit_recovery_retry(request_id, request.get("attempt_id"))


def _invoke_recovery_release(control: Any, request_id: str, request: dict, contract_mode: bool) -> Any:
    return control.submit_recovery_release(
        request_id, request.get("attempt_id"), request.get("saved_at")
    )


def _invoke_errors_reset(control: Any, request_id: str, request: dict, contract_mode: bool) -> Any:
    return control.submit_errors_reset(request_id)


def _stop_parse_cli(args: Any) -> dict:
    if args.after is not None:
        from simple_coding_agent.control import parse_stop_after

        return {"after": parse_stop_after(args.after), "expect_repo": args.expect_repo}
    return {"expect_repo": args.expect_repo}


def _stop_after_cli_label(payload: dict) -> str:
    if "after" in payload:
        return f"stop --after {payload['after']}"
    return "stop"


def _resume_parse_cli(args: Any) -> dict:
    return {"expect_repo": args.expect_repo}


def _handoff_parse_cli(args: Any) -> dict:
    return {"expect_repo": args.expect_repo}


def _next_issue_parse_cli(args: Any) -> dict:
    from simple_coding_agent.control import parse_next_issue

    return {"issue": parse_next_issue(args.issue_number), "expect_repo": args.expect_repo}


def _recovery_retry_parse_cli(args: Any) -> dict:
    from simple_coding_agent.recovery import parse_attempt_id

    return {"attempt_id": parse_attempt_id(args.attempt_id)}


def _recovery_release_parse_cli(args: Any) -> dict:
    from simple_coding_agent.recovery import parse_attempt_id, parse_saved_at

    return {
        "attempt_id": parse_attempt_id(args.attempt_id),
        "saved_at": parse_saved_at(args.saved_at),
    }


def _errors_reset_parse_cli(args: Any) -> dict:
    return {}


_REQUEST_ID_HELP = "Retry a previously generated request ID with the identical payload."
_EXPECT_REPO_HELP = (
    "Refuse the command before anything is committed unless this instance targets "
    "the given owner/name repository (case-insensitive)."
)


def _add_request_id_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--request-id",
        dest="request_id",
        default=None,
        help=_REQUEST_ID_HELP,
    )


def _add_expect_repo_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--expect-repo",
        dest="expect_repo",
        default=None,
        help=_EXPECT_REPO_HELP,
    )


def _configure_stop(subparsers: Any) -> None:
    stop_parser = subparsers.add_parser(
        "stop", help="Finish the active attempt, then stop intake."
    )
    _add_request_id_arg(stop_parser)
    stop_parser.add_argument(
        "--after",
        dest="after",
        default=None,
        help="Stop intake after N fully finalized attempts, counting the active one first.",
    )
    _add_expect_repo_arg(stop_parser)


def _configure_resume(subparsers: Any) -> None:
    resume_parser = subparsers.add_parser(
        "resume", help="Permit issue intake again; replace any pending stop plan."
    )
    _add_request_id_arg(resume_parser)
    _add_expect_repo_arg(resume_parser)


def _configure_handoff(subparsers: Any, parents: dict) -> None:
    handoff_parser = parents.get("handoff")
    if handoff_parser is None:
        handoff_parser = subparsers.add_parser(
            "handoff",
            help="Hand the active attempt to a human with a published handoff note.",
        )
        parents["handoff"] = handoff_parser
    handoff_subparsers = getattr(handoff_parser, "_subparsers", None)
    if handoff_subparsers is None:
        handoff_subparsers = handoff_parser.add_subparsers(
            dest="handoff_command", required=True
        )
        handoff_parser._subparsers = handoff_subparsers  # type: ignore[attr-defined]
    handoff_now_parser = handoff_subparsers.add_parser(
        "now",
        help="Deliver a handoff request to the active attempt at the next safe model boundary.",
    )
    _add_request_id_arg(handoff_now_parser)
    _add_expect_repo_arg(handoff_now_parser)


def _configure_next_issue(subparsers: Any, parents: dict) -> None:
    next_parser = parents.get("next")
    if next_parser is None:
        next_parser = subparsers.add_parser(
            "next", help="Prioritize one eligible issue for the next permitted claim."
        )
        parents["next"] = next_parser
    next_subparsers = getattr(next_parser, "_subparsers", None)
    if next_subparsers is None:
        next_subparsers = next_parser.add_subparsers(dest="next_command", required=True)
        next_parser._subparsers = next_subparsers  # type: ignore[attr-defined]
    next_issue_parser = next_subparsers.add_parser(
        "issue", help="Prioritize one eligible issue for the next permitted claim."
    )
    next_issue_parser.add_argument(
        "issue_number", help="The implementation issue number to prioritize."
    )
    _add_request_id_arg(next_issue_parser)
    _add_expect_repo_arg(next_issue_parser)


def _configure_recovery_retry(subparsers: Any, parents: dict) -> None:
    recovery_parser = parents.get("recovery")
    if recovery_parser is None:
        recovery_parser = subparsers.add_parser(
            "recovery", help="Inspect or resolve a retained implementation attempt."
        )
        parents["recovery"] = recovery_parser
    recovery_subparsers = getattr(recovery_parser, "_subparsers", None)
    if recovery_subparsers is None:
        recovery_subparsers = recovery_parser.add_subparsers(
            dest="recovery_command", required=True
        )
        recovery_parser._subparsers = recovery_subparsers  # type: ignore[attr-defined]
    retry_parser = recovery_subparsers.add_parser(
        "retry",
        help="Recheck evidence and retry unconfirmed publication, release, cleanup, and accounting.",
    )
    retry_parser.add_argument("attempt_id", help="The stable attempt identity to retry.")
    _add_request_id_arg(retry_parser)


def _configure_recovery_release(subparsers: Any, parents: dict) -> None:
    recovery_parser = parents.get("recovery")
    if recovery_parser is None:
        recovery_parser = subparsers.add_parser(
            "recovery", help="Inspect or resolve a retained implementation attempt."
        )
        parents["recovery"] = recovery_parser
    recovery_subparsers = getattr(recovery_parser, "_subparsers", None)
    if recovery_subparsers is None:
        recovery_subparsers = recovery_parser.add_subparsers(
            dest="recovery_command", required=True
        )
        recovery_parser._subparsers = recovery_subparsers  # type: ignore[attr-defined]
    release_parser = recovery_subparsers.add_parser(
        "release",
        help="Abandon a retained attempt after securing its work elsewhere.",
    )
    release_parser.add_argument("attempt_id", help="The stable attempt identity to release.")
    release_parser.add_argument(
        "--saved-at",
        dest="saved_at",
        required=True,
        help="Path or URL where the retained work was secured.",
    )
    _add_request_id_arg(release_parser)


def _configure_errors_reset(subparsers: Any, parents: dict) -> None:
    errors_parser = parents.get("errors")
    if errors_parser is None:
        errors_parser = subparsers.add_parser(
            "errors",
            aliases=["consecutive-errors"],
            help="Inspect or reset the consecutive-error guard.",
        )
        parents["errors"] = errors_parser
    errors_subparsers = getattr(errors_parser, "_subparsers", None)
    if errors_subparsers is None:
        errors_subparsers = errors_parser.add_subparsers(
            dest="errors_command", required=True
        )
        errors_parser._subparsers = errors_subparsers  # type: ignore[attr-defined]
    errors_reset_parser = errors_subparsers.add_parser(
        "reset",
        help="Reset the consecutive-error counter to 0 through the live process.",
    )
    _add_request_id_arg(errors_reset_parser)


def _build_request_stop(request_id: str, payload: dict) -> dict:
    request: dict = {"op": "stop", "request_id": request_id}
    if payload.get("expect_repo") is not None:
        request["expect_repo"] = payload["expect_repo"]
    return request


def _build_request_stop_after(request_id: str, payload: dict) -> dict:
    request = {"op": "stop_after", "request_id": request_id, "after": payload["after"]}
    if payload.get("expect_repo") is not None:
        request["expect_repo"] = payload["expect_repo"]
    return request


def _build_request_next_issue(request_id: str, payload: dict) -> dict:
    request = {"op": "next_issue", "request_id": request_id, "issue": payload["issue"]}
    if payload.get("expect_repo") is not None:
        request["expect_repo"] = payload["expect_repo"]
    return request


def _build_request_resume(request_id: str, payload: dict) -> dict:
    request = {"op": "resume", "request_id": request_id}
    if payload.get("expect_repo") is not None:
        request["expect_repo"] = payload["expect_repo"]
    return request


def _build_request_handoff(request_id: str, payload: dict) -> dict:
    request = {"op": "handoff", "request_id": request_id}
    if payload.get("expect_repo") is not None:
        request["expect_repo"] = payload["expect_repo"]
    return request


def _build_request_recovery_retry(request_id: str, payload: dict) -> dict:
    return {
        "op": "recovery_retry",
        "request_id": request_id,
        "attempt_id": payload["attempt_id"],
    }


def _build_request_recovery_release(request_id: str, payload: dict) -> dict:
    return {
        "op": "recovery_release",
        "request_id": request_id,
        "attempt_id": payload["attempt_id"],
        "saved_at": payload["saved_at"],
    }


def _build_request_errors_reset(request_id: str, payload: dict) -> dict:
    return {"op": "errors_reset", "request_id": request_id}


def _rejected_stop() -> tuple[type[Exception], ...]:
    from simple_coding_agent.control import PayloadMismatchError, RequestIdError

    return (RequestIdError, PayloadMismatchError)


def _rejected_stop_after() -> tuple[type[Exception], ...]:
    from simple_coding_agent.control import (
        PayloadMismatchError,
        RequestIdError,
        StopAfterRejectedError,
    )

    return (RequestIdError, PayloadMismatchError, StopAfterRejectedError)


def _rejected_next_issue() -> tuple[type[Exception], ...]:
    from simple_coding_agent.control import (
        NextIssueRejectedError,
        PayloadMismatchError,
        RequestIdError,
    )

    return (RequestIdError, PayloadMismatchError, NextIssueRejectedError)


def _rejected_resume() -> tuple[type[Exception], ...]:
    from simple_coding_agent.control import (
        PayloadMismatchError,
        RequestIdError,
        ResumeBlockedError,
        ResumeIntakeOpenError,
    )

    return (
        RequestIdError,
        PayloadMismatchError,
        ResumeBlockedError,
        ResumeIntakeOpenError,
    )


def _rejected_handoff() -> tuple[type[Exception], ...]:
    from simple_coding_agent.control import (
        HandoffRejectedError,
        PayloadMismatchError,
        RequestIdError,
    )

    return (RequestIdError, PayloadMismatchError, HandoffRejectedError)


def _rejected_recovery() -> tuple[type[Exception], ...]:
    from simple_coding_agent.control import PayloadMismatchError, RequestIdError
    from simple_coding_agent.recovery import RecoveryRejectedError

    return (RequestIdError, PayloadMismatchError, RecoveryRejectedError)


def _rejected_errors_reset() -> tuple[type[Exception], ...]:
    from simple_coding_agent.control import (
        ErrorsResetRejectedError,
        PayloadMismatchError,
        RequestIdError,
    )

    return (RequestIdError, PayloadMismatchError, ErrorsResetRejectedError)


def build_registry() -> dict[str, CommandSpec]:
    """Build the one registry entry per operator command."""

    return {
        "stop": CommandSpec(
            op="stop",
            kind="stop",
            summary="Finish the active attempt, then stop intake.",
            rejected=_rejected_stop(),
            parse_request=_stop_parse_request,
            invoke=_invoke_stop,
            cli_route=("stop",),
            supports_expect_repo=True,
            requires_active_attempt=False,
            needs_lifecycle=True,
            configure_parser=_configure_stop,
            parse_cli_args=_stop_parse_cli,
            cli_kind_label=_stop_after_cli_label,
            build_request=_build_request_stop,
        ),
        "stop_after": CommandSpec(
            op="stop_after",
            kind="stop_after",
            summary="Stop intake after N fully finalized attempts.",
            rejected=_rejected_stop_after(),
            parse_request=_stop_after_parse_request,
            invoke=_invoke_stop_after,
            cli_route=("stop",),
            supports_expect_repo=True,
            requires_active_attempt=False,
            needs_lifecycle=True,
            configure_parser=None,
            parse_cli_args=_stop_parse_cli,
            cli_kind_label=_stop_after_cli_label,
            build_request=_build_request_stop_after,
        ),
        "resume": CommandSpec(
            op="resume",
            kind="resume",
            summary="Permit issue intake again; replace any pending stop plan.",
            rejected=_rejected_resume(),
            parse_request=_resume_parse_request,
            invoke=_invoke_resume,
            cli_route=("resume",),
            supports_expect_repo=True,
            requires_active_attempt=False,
            needs_lifecycle=True,
            configure_parser=_configure_resume,
            parse_cli_args=_resume_parse_cli,
            cli_kind_label=lambda payload: "resume",
            build_request=_build_request_resume,
        ),
        "handoff": CommandSpec(
            op="handoff",
            kind="handoff",
            summary="Hand the active attempt to a human with a published handoff note.",
            rejected=_rejected_handoff(),
            parse_request=_handoff_parse_request,
            invoke=_invoke_handoff,
            cli_route=("handoff", "now"),
            supports_expect_repo=True,
            requires_active_attempt=True,
            needs_lifecycle=True,
            configure_parser=_configure_handoff,
            parse_cli_args=_handoff_parse_cli,
            cli_kind_label=lambda payload: "handoff now",
            build_request=_build_request_handoff,
        ),
        "next_issue": CommandSpec(
            op="next_issue",
            kind="next_issue",
            summary="Prioritize one eligible issue for the next permitted claim.",
            rejected=_rejected_next_issue(),
            parse_request=_next_issue_parse_request,
            invoke=_invoke_next_issue,
            cli_route=("next", "issue"),
            supports_expect_repo=True,
            requires_active_attempt=False,
            needs_lifecycle=True,
            configure_parser=_configure_next_issue,
            parse_cli_args=_next_issue_parse_cli,
            cli_kind_label=lambda payload: f"next issue {payload['issue']}",
            build_request=_build_request_next_issue,
        ),
        "recovery_retry": CommandSpec(
            op="recovery_retry",
            kind="recovery_retry",
            summary="Recheck evidence and retry unconfirmed finalization steps.",
            rejected=_rejected_recovery(),
            parse_request=_recovery_retry_parse_request,
            invoke=_invoke_recovery_retry,
            cli_route=("recovery", "retry"),
            supports_expect_repo=False,
            requires_active_attempt=False,
            needs_lifecycle=True,
            configure_parser=_configure_recovery_retry,
            parse_cli_args=_recovery_retry_parse_cli,
            cli_kind_label=lambda payload: f"recovery retry {payload['attempt_id']}",
            build_request=_build_request_recovery_retry,
        ),
        "recovery_release": CommandSpec(
            op="recovery_release",
            kind="recovery_release",
            summary="Abandon a retained attempt after securing its work elsewhere.",
            rejected=_rejected_recovery(),
            parse_request=_recovery_release_parse_request,
            invoke=_invoke_recovery_release,
            cli_route=("recovery", "release"),
            supports_expect_repo=False,
            requires_active_attempt=False,
            needs_lifecycle=True,
            configure_parser=_configure_recovery_release,
            parse_cli_args=_recovery_release_parse_cli,
            cli_kind_label=lambda payload: (
                f"recovery release {payload['attempt_id']} "
                f"--saved-at {payload['saved_at']}"
            ),
            build_request=_build_request_recovery_release,
        ),
        "errors_reset": CommandSpec(
            op="errors_reset",
            kind="errors_reset",
            summary="Reset the consecutive-error counter to 0.",
            rejected=_rejected_errors_reset(),
            parse_request=_errors_reset_parse_request,
            invoke=_invoke_errors_reset,
            cli_route=("errors", "reset"),
            supports_expect_repo=False,
            requires_active_attempt=False,
            needs_lifecycle=True,
            configure_parser=_configure_errors_reset,
            parse_cli_args=_errors_reset_parse_cli,
            cli_kind_label=lambda payload: "errors reset",
            build_request=_build_request_errors_reset,
        ),
    }


COMMAND_REGISTRY: dict[str, CommandSpec] = build_registry()

#: Ops served generically through the registry (the eight mutating commands).
REGISTERED_OPS: frozenset[str] = frozenset(COMMAND_REGISTRY.keys())


def get_spec(op: str) -> CommandSpec | None:
    """Return the registry entry for one socket ``op``, if registered."""

    return COMMAND_REGISTRY.get(op)


def rejection_errors_for(op: str) -> tuple[type[Exception], ...]:
    """Return the rejection errors for one registered ``op``."""

    spec = COMMAND_REGISTRY.get(op)
    if spec is None:
        raise KeyError(f"Unknown operation: {op!r}.")
    return spec.rejected


def parse_payload(op: str, request: dict) -> dict:
    """Parse one raw socket request into canonical submit kwargs."""

    spec = COMMAND_REGISTRY.get(op)
    if spec is None:
        raise KeyError(f"Unknown operation: {op!r}.")
    return spec.parse_request(request)


def invoke_registered(
    control: Any, op: str, request_id: str, request: dict, contract_mode: bool = False
) -> Any:
    """Invoke one registered command against a control surface."""

    spec = COMMAND_REGISTRY.get(op)
    if spec is None:
        raise KeyError(f"Unknown operation: {op!r}.")
    return spec.invoke(control, request_id, request, contract_mode)


def add_mutating_parsers(subparsers: Any) -> dict:
    """Generate every mutating command's CLI parser from the registry.

    Returns the shared ``parents`` mapping for nested subcommands so
    callers can introspect it in tests. Query parsers (``status``,
    ``telemetry``, ``command``) stay hand-written in ``agentctl``: they
    are read-only snapshots, not commands.
    """

    parents: dict = {}
    # Order mirrors the historical CLI: stop, resume, handoff, next,
    # recovery retry/release, errors reset.
    for op in (
        "stop",
        "resume",
        "handoff",
        "next_issue",
        "recovery_retry",
        "recovery_release",
        "errors_reset",
    ):
        spec = COMMAND_REGISTRY[op]
        configure = spec.configure_parser
        if configure is None:
            continue
        try:
            configure(subparsers, parents)  # type: ignore[arg-type]
        except TypeError:
            configure(subparsers)  # type: ignore[operator]
    return parents


def cli_route_to_op(args: Any) -> str | None:
    """Map parsed CLI args to the registered ``op``, if mutating."""

    command = getattr(args, "command", None)
    if command == "stop":
        if getattr(args, "after", None) is not None:
            return "stop_after"
        return "stop"
    if command == "resume":
        return "resume"
    if command == "handoff" and getattr(args, "handoff_command", None) == "now":
        return "handoff"
    if command == "next" and getattr(args, "next_command", None) == "issue":
        return "next_issue"
    if command == "recovery" and getattr(args, "recovery_command", None) == "retry":
        return "recovery_retry"
    if command == "recovery" and getattr(args, "recovery_command", None) == "release":
        return "recovery_release"
    if command in ("errors", "consecutive-errors") and getattr(
        args, "errors_command", None
    ) == "reset":
        return "errors_reset"
    return None


def build_socket_request(op: str, request_id: str, payload: dict) -> dict:
    """Build the socket request mapping for one registered command."""

    spec = COMMAND_REGISTRY.get(op)
    if spec is None or spec.build_request is None:
        raise KeyError(f"Unknown operation: {op!r}.")
    return spec.build_request(request_id, payload)


def parse_cli_payload(op: str, args: Any) -> dict:
    """Extract canonical submit kwargs from parsed CLI args."""

    spec = COMMAND_REGISTRY.get(op)
    if spec is None or spec.parse_cli_args is None:
        raise KeyError(f"Unknown operation: {op!r}.")
    return spec.parse_cli_args(args)


def cli_label(op: str, payload: dict) -> str:
    """Human label for ``submit_mutating`` for one registered command."""

    spec = COMMAND_REGISTRY.get(op)
    if spec is None or spec.cli_kind_label is None:
        return op
    return spec.cli_kind_label(payload)
