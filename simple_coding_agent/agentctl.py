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
import os
import sys
import uuid
from pathlib import Path

from simple_coding_agent.control_server import (
    ControlUnavailableError,
    send_request,
    socket_path_for,
)


def default_socket_path() -> Path:
    """Resolve the instance endpoint from ``DATA_DIR``."""

    return socket_path_for(Path(os.environ.get("DATA_DIR", "~/.simple-coding-agent/")).expanduser())


def generate_request_id() -> str:
    """Create the unique request ID printed before a mutating submission."""

    return f"req-{uuid.uuid4().hex[:12]}"


def run_stop(socket_path: Path, request_id: str) -> dict:
    """Submit ``stop`` once; never present a failure as accepted."""

    return send_request(socket_path, {"op": "stop", "request_id": request_id})


def run_status(socket_path: Path) -> dict:
    """Read one consistent live snapshot; raise when the process is down."""

    reply = send_request(socket_path, {"op": "status"})
    if not reply.get("ok"):
        raise ControlUnavailableError(reply.get("error", "Status is unavailable."))
    return reply["status"]


def run_command_lookup(socket_path: Path, request_id: str) -> dict | None:
    """Look up one command's current durable acknowledgement."""

    reply = send_request(socket_path, {"op": "command", "request_id": request_id})
    if not reply.get("ok"):
        raise ControlUnavailableError(reply.get("error", "Lookup is unavailable."))
    return reply.get("command")


def format_command(reply: dict) -> str:
    return (
        f"request_id: {reply.get('request_id')}\n"
        f"sequence: {reply.get('sequence')}\n"
        f"acknowledgement: {reply.get('acknowledgement')}\n"
        f"effect: {reply.get('detail')}"
    )


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
        help="Path to the agent control socket (defaults to $DATA_DIR/state/agentctl.sock).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stop_parser = subparsers.add_parser("stop", help="Finish the active attempt, then stop intake.")
    stop_parser.add_argument(
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    socket_path = Path(args.socket) if args.socket else default_socket_path()

    try:
        if args.command == "stop":
            request_id = args.request_id or generate_request_id()
            try:
                reply = run_stop(socket_path, request_id)
            except ControlUnavailableError as error:
                # The ambiguous-loss case: the command may or may not have
                # committed, so the ID must survive for an identical retry.
                print(
                    f"{request_id} not submitted — cannot connect to the agent process:"
                    f" {error} Retry with: agentctl stop --request-id {request_id}",
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
    except ControlUnavailableError as error:
        print(f"Cannot connect to the agent process: {error}", file=sys.stderr)
        return 2
    raise AssertionError("unreachable")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
