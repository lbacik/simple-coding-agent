"""Operator control CLI for one running agent instance (``agentctl``).

Run inside the selected agent container; container selection determines
the controlled instance, so the CLI takes no instance selector. Mutating
commands generate and print a unique request ID before submission and
accept ``--request-id`` to retry the same payload after an ambiguous
connection loss. ``status`` and ``command`` are read-only snapshots from
the live process: when the process cannot be reached they report a
connection error instead of a saved snapshot.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Callable
from pathlib import Path

from simple_coding_agent.control_server import (
    ControlUnavailableError,
    resolve_socket_path,
    send_request,
)
from simple_coding_agent.control import StopAfterRejectedError, parse_stop_after


def default_socket_path() -> Path:
    """Resolve the instance endpoint: the container-local runtime socket.

    Container selection determines the controlled instance, so the default is
    the fixed ``/run/simple-coding-agent/control.sock`` path. ``AGENTCTL_SOCKET``
    overrides it for tests and local development only.
    """

    return resolve_socket_path()


def generate_request_id() -> str:
    """Create the unique request ID printed before a mutating submission."""

    return f"req-{uuid.uuid4().hex[:12]}"


def run_stop(socket_path: Path, request_id: str) -> dict:
    """Submit ``stop`` once; never present a failure as accepted."""

    return send_request(socket_path, {"op": "stop", "request_id": request_id})


def run_stop_after(socket_path: Path, request_id: str, after: int) -> dict:
    """Submit ``stop --after N`` once; never present a failure as accepted."""

    return send_request(
        socket_path, {"op": "stop_after", "request_id": request_id, "after": after}
    )


def run_next_issue(socket_path: Path, request_id: str, issue: int) -> dict:
    """Submit ``next issue`` once; never present a failure as accepted."""

    return send_request(
        socket_path, {"op": "next_issue", "request_id": request_id, "issue": issue}
    )


def run_resume(socket_path: Path, request_id: str) -> dict:
    """Submit ``resume`` once; never present a failure as accepted."""

    return send_request(socket_path, {"op": "resume", "request_id": request_id})


def run_handoff(socket_path: Path, request_id: str) -> dict:
    """Submit ``handoff now`` once; never present a failure as accepted."""

    return send_request(socket_path, {"op": "handoff", "request_id": request_id})


def run_recovery_retry(socket_path: Path, request_id: str, attempt_id: str) -> dict:
    """Submit ``recovery retry`` once; never present a failure as accepted."""

    return send_request(
        socket_path,
        {"op": "recovery_retry", "request_id": request_id, "attempt_id": attempt_id},
    )


def run_recovery_release(
    socket_path: Path, request_id: str, attempt_id: str, saved_at: str
) -> dict:
    """Submit ``recovery release`` once; never present a failure as accepted."""

    return send_request(
        socket_path,
        {
            "op": "recovery_release",
            "request_id": request_id,
            "attempt_id": attempt_id,
            "saved_at": saved_at,
        },
    )


def run_status(socket_path: Path) -> dict:
    """Read one consistent live snapshot; raise when the process is down."""

    reply = send_request(socket_path, {"op": "status"})
    if not reply.get("ok"):
        raise ControlUnavailableError(reply.get("error", "Status is unavailable."))
    return reply["status"]


def run_command_lookup(socket_path: Path, request_id: str) -> dict | None:
    """Look up one command's current durable acknowledgement.

    The returned record carries the ``repository`` of the instance that
    answered, so the operator can verify the selected container.
    """

    reply = send_request(socket_path, {"op": "command", "request_id": request_id})
    if not reply.get("ok"):
        raise ControlUnavailableError(reply.get("error", "Lookup is unavailable."))
    record = reply.get("command")
    if record is not None and reply.get("repository") is not None:
        record = {**record, "repository": reply["repository"]}
    return record


def format_command(reply: dict) -> str:
    return (
        f"repository: {reply.get('repository', 'unknown')}\n"
        f"request_id: {reply.get('request_id')}\n"
        f"sequence: {reply.get('sequence')}\n"
        f"acknowledgement: {reply.get('acknowledgement')}\n"
        f"effect: {reply.get('detail')}"
    )


def submit_mutating(
    socket_path: Path,
    kind: str,
    request_id: str,
    submit: Callable[[Path, str], dict],
) -> int:
    """Submit one mutating command; return the process exit code.

    A lost connection keeps the request ID for an identical retry; a
    rejection reports the reason with no control change accepted.
    """

    try:
        reply = submit(socket_path, request_id)
    except ControlUnavailableError as error:
        # The ambiguous-loss case: the command may or may not have
        # committed, so the ID must survive for an identical retry.
        print(
            f"{request_id} not submitted — cannot connect to the agent process:"
            f" {error} Retry with: agentctl {kind} --request-id {request_id}",
            file=sys.stderr,
        )
        return 2
    if not reply.get("ok"):
        print(
            f"{request_id} rejected — {reply.get('error', 'unknown error')}. "
            "No control change was accepted.",
            file=sys.stderr,
        )
        return 1
    print(format_command(reply))
    return 0


def format_status(status: dict) -> str:
    lines = [
        f"repository: {status.get('repository')}",
        f"intake: {status.get('intake')}",
    ]
    active = status.get("active_attempt")
    if active is None:
        lines.append("active_attempt: none")
    else:
        lines.append(
            "active_attempt:"
            f" issue #{active.get('issue_number')}"
            f" phase {active.get('phase')}"
            f" branch {active.get('branch')}"
        )
    pending = status.get("pending_command")
    if pending is None:
        lines.append("pending_command: none")
    else:
        lines.append(
            f"pending_command: {pending.get('request_id')}"
            f" ({pending.get('kind')}, {pending.get('acknowledgement')}"
            f" — {pending.get('detail')})"
        )
    plan = status.get("stop_plan")
    if plan is None:
        lines.append("stop_plan: none")
    else:
        if plan.get("includes_active_attempt"):
            inclusion = f"includes active attempt {plan.get('active_attempt_id')}"
        else:
            inclusion = "starts with next attempt"
        latest_id = plan.get("latest_counted_attempt_id")
        if latest_id is None:
            last_counted = "none"
        else:
            last_counted = f"{latest_id} ({plan.get('latest_counted_outcome')})"
        lines.append(
            f"stop_plan: {plan.get('request_id')}"
            f" ({plan.get('kind')}, requested {plan.get('requested')},"
            f" remaining {plan.get('remaining')}, {inclusion},"
            f" last counted: {last_counted})"
        )
    pending_next = status.get("pending_next_issue")
    if pending_next is None:
        lines.append("pending_next_issue: none")
    else:
        lines.append(
            f"pending_next_issue: #{pending_next.get('issue_number')}"
            f" ({pending_next.get('request_id')})"
        )
    recovery = status.get("recovery")
    if recovery is None:
        lines.append("recovery: none")
    else:
        attempt_id = recovery.get("attempt_id") or "unknown"
        issue_number = recovery.get("issue_number")
        phase = recovery.get("phase") or "unknown"
        outcome = recovery.get("outcome") or "unknown"
        branch = recovery.get("branch") or "unknown"
        workspace = recovery.get("workspace") or "unknown"
        publication = recovery.get("publication") or {}
        comment_confirmed = publication.get("comment_confirmed")
        if comment_confirmed is True:
            comment_state = "comment confirmed"
        elif comment_confirmed is False:
            comment_state = "comment unconfirmed"
        else:
            comment_state = "comment unknown"
        pull_request = publication.get("pull_request")
        if isinstance(pull_request, dict) and pull_request.get("number") is not None:
            pr_state = f"PR #{pull_request.get('number')}"
        else:
            pr_state = "PR none/unknown"
        lines.append(
            f"recovery: attempt {attempt_id}"
            f" issue #{issue_number}"
            f" phase {phase}"
            f" outcome {outcome}"
            f" branch {branch}"
            f" workspace {workspace}"
            f" checkpoint {'present' if recovery.get('checkpoint') else 'absent'}"
            f" ({comment_state}; {pr_state})"
        )
        lines.append(f"hold_reason: {recovery.get('hold_reason')}")
        lines.append(f"next_action: {recovery.get('next_action')}")
    handoff = status.get("handoff")
    if handoff is None:
        lines.append("handoff: none")
    else:
        lines.append(
            f"handoff: {handoff.get('request_id')}"
            f" ({handoff.get('acknowledgement')},"
            f" begun {handoff.get('begun')},"
            f" phase {handoff.get('phase')}"
            f" — {handoff.get('detail')})"
        )
        lines.append(
            f"handoff_deadlines: accepted {handoff.get('accepted_at')}"
            f" model {handoff.get('model_deadline_at')}"
            f" publication {handoff.get('publication_deadline_at')}"
        )
    commands = status.get("commands") or {}
    if commands:
        lines.append("recent_commands:")
        for request_id in sorted(commands, key=lambda rid: commands[rid].get("sequence", 0)):
            entry = commands[request_id]
            lines.append(
                f"  {request_id}: #{entry.get('sequence')}"
                f" {entry.get('acknowledgement')} — {entry.get('detail')}"
            )
    else:
        lines.append("recent_commands: none")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentctl", description="Control one running agent instance."
    )
    parser.add_argument(
        "--socket",
        dest="socket",
        default=None,
        help="Path to the agent control socket (defaults to /run/simple-coding-agent/control.sock).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stop_parser = subparsers.add_parser("stop", help="Finish the active attempt, then stop intake.")
    stop_parser.add_argument(
        "--request-id",
        dest="request_id",
        default=None,
        help="Retry a previously generated request ID with the identical payload.",
    )
    stop_parser.add_argument(
        "--after",
        dest="after",
        default=None,
        help="Stop intake after N fully finalized attempts, counting the active one first.",
    )

    resume_parser = subparsers.add_parser(
        "resume", help="Permit issue intake again; replace any pending stop plan."
    )
    resume_parser.add_argument(
        "--request-id",
        dest="request_id",
        default=None,
        help="Retry a previously generated request ID with the identical payload.",
    )

    handoff_parser = subparsers.add_parser(
        "handoff",
        help="Hand the active attempt to a human with a published handoff note.",
    )
    handoff_subparsers = handoff_parser.add_subparsers(
        dest="handoff_command", required=True
    )
    handoff_now_parser = handoff_subparsers.add_parser(
        "now",
        help="Deliver a handoff request to the active attempt at the next safe model boundary.",
    )
    handoff_now_parser.add_argument(
        "--request-id",
        dest="request_id",
        default=None,
        help="Retry a previously generated request ID with the identical payload.",
    )

    next_parser = subparsers.add_parser(
        "next", help="Prioritize one eligible issue for the next permitted claim."
    )
    next_subparsers = next_parser.add_subparsers(dest="next_command", required=True)
    next_issue_parser = next_subparsers.add_parser(
        "issue", help="Prioritize one eligible issue for the next permitted claim."
    )
    next_issue_parser.add_argument(
        "issue_number", help="The implementation issue number to prioritize."
    )
    next_issue_parser.add_argument(
        "--request-id",
        dest="request_id",
        default=None,
        help="Retry a previously generated request ID with the identical payload.",
    )

    subparsers.add_parser("status", help="Show one consistent live status snapshot.")
    command_parser = subparsers.add_parser(
        "command", help="Show one command's current durable acknowledgement."
    )
    command_parser.add_argument("request_id", help="The request ID printed at submission.")

    recovery_parser = subparsers.add_parser(
        "recovery", help="Inspect or resolve a retained implementation attempt."
    )
    recovery_subparsers = recovery_parser.add_subparsers(
        dest="recovery_command", required=True
    )
    retry_parser = recovery_subparsers.add_parser(
        "retry",
        help="Recheck evidence and retry unconfirmed publication, release, cleanup, and accounting.",
    )
    retry_parser.add_argument("attempt_id", help="The stable attempt identity to retry.")
    retry_parser.add_argument(
        "--request-id",
        dest="request_id",
        default=None,
        help="Retry a previously generated request ID with the identical payload.",
    )
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
    release_parser.add_argument(
        "--request-id",
        dest="request_id",
        default=None,
        help="Retry a previously generated request ID with the identical payload.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    socket_path = Path(args.socket) if args.socket else default_socket_path()

    try:
        if args.command == "stop":
            request_id = args.request_id or generate_request_id()
            if args.after is not None:
                try:
                    after = parse_stop_after(args.after)
                except StopAfterRejectedError as error:
                    print(f"rejected — {error}", file=sys.stderr)
                    return 1
                return submit_mutating(
                    socket_path,
                    f"stop --after {after}",
                    request_id,
                    lambda path, rid: run_stop_after(path, rid, after),
                )
            return submit_mutating(socket_path, "stop", request_id, run_stop)
        if args.command == "resume":
            request_id = args.request_id or generate_request_id()
            return submit_mutating(socket_path, "resume", request_id, run_resume)
        if args.command == "handoff" and args.handoff_command == "now":
            request_id = args.request_id or generate_request_id()
            return submit_mutating(socket_path, "handoff now", request_id, run_handoff)
        if args.command == "next" and args.next_command == "issue":
            from simple_coding_agent.control import (
                NextIssueRejectedError,
                parse_next_issue,
            )

            request_id = args.request_id or generate_request_id()
            try:
                issue = parse_next_issue(args.issue_number)
            except NextIssueRejectedError as error:
                print(f"{request_id} rejected — {error}", file=sys.stderr)
                return 1
            return submit_mutating(
                socket_path,
                f"next issue {issue}",
                request_id,
                lambda path, rid: run_next_issue(path, rid, issue),
            )
        if args.command == "status":
            print(format_status(run_status(socket_path)))
            return 0
        if args.command == "command":
            record = run_command_lookup(socket_path, args.request_id)
            if record is None:
                print(f"{args.request_id}: no such command is recorded.", file=sys.stderr)
                return 1
            print(format_command(record))
            return 0
        if args.command == "recovery" and args.recovery_command == "retry":
            from simple_coding_agent.recovery import RecoveryRejectedError, parse_attempt_id

            request_id = args.request_id or generate_request_id()
            try:
                attempt_id = parse_attempt_id(args.attempt_id)
            except RecoveryRejectedError as error:
                print(f"{request_id} rejected — {error}", file=sys.stderr)
                return 1
            return submit_mutating(
                socket_path,
                f"recovery retry {attempt_id}",
                request_id,
                lambda path, rid: run_recovery_retry(path, rid, attempt_id),
            )
        if args.command == "recovery" and args.recovery_command == "release":
            from simple_coding_agent.recovery import (
                RecoveryRejectedError,
                parse_attempt_id,
                parse_saved_at,
            )

            request_id = args.request_id or generate_request_id()
            try:
                attempt_id = parse_attempt_id(args.attempt_id)
                saved_at = parse_saved_at(args.saved_at)
            except RecoveryRejectedError as error:
                print(f"{request_id} rejected — {error}", file=sys.stderr)
                return 1
            return submit_mutating(
                socket_path,
                f"recovery release {attempt_id}",
                request_id,
                lambda path, rid: run_recovery_release(path, rid, attempt_id, saved_at),
            )
    except ControlUnavailableError as error:
        print(f"Cannot connect to the agent process: {error}", file=sys.stderr)
        return 2
    raise AssertionError("unreachable")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
