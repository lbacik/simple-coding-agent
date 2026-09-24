"""Private local socket delivering operator commands to the live agent.

The socket is a runtime endpoint, not the durable record: the agent
recreates it on startup, handles a stale endpoint safely, and restricts it
to the agent's OS user (and container root). The CLI never opens the
SQLite store directly; every request is serialized through the control
surface so command acceptance shares one order with the claim and
attempt-completion boundaries.
"""

from __future__ import annotations

import json
import os
import socket
import stat
import threading
from pathlib import Path
from typing import Protocol


class ControlSurface(Protocol):
    """The lifecycle-facing control operations served over the socket."""

    def submit_stop(self, request_id: str, *, has_active_attempt: bool = ...) -> object: ...

    def get_command(self, request_id: str) -> object | None: ...

    def control_status(self, repository: str | None = ...) -> dict: ...


class ControlUnavailableError(ConnectionError):
    """Raised when the agent process cannot be reached."""


_SOCKET_BACKLOG = 16
_BUFFER_LIMIT = 1 << 20


class ControlServer:
    """Serve one agent instance's control surface on a Unix domain socket."""

    def __init__(self, socket_path: Path, control: ControlSurface) -> None:
        self._socket_path = socket_path
        self._control = control
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    def start(self) -> None:
        """Bind the endpoint and serve requests on a background thread.

        The listener stays responsive while an implementation attempt runs:
        it only performs short serialized control operations, never the
        attempt work itself.
        """

        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self._socket_path.parent, 0o700)
        except OSError:
            pass
        try:
            if self._socket_path.exists() or self._socket_path.is_socket():
                self._socket_path.unlink()
        except OSError as error:
            raise ControlUnavailableError(
                f"Control endpoint {self._socket_path} could not be recreated: {error}"
            ) from error
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self._socket_path))
            try:
                os.chmod(self._socket_path, 0o600)
            except OSError:
                pass
            listener.listen(_SOCKET_BACKLOG)
        except OSError as error:
            listener.close()
            raise ControlUnavailableError(
                f"Control endpoint {self._socket_path} could not be bound: {error}"
            ) from error
        self._listener = listener
        self._stopped.clear()
        self._thread = threading.Thread(
            target=self._serve, name="agentctl-listener", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop serving and remove the runtime endpoint."""

        self._stopped.set()
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10)
        try:
            self._socket_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _serve(self) -> None:
        listener = self._listener
        if listener is None:
            return
        listener.settimeout(0.2)
        while not self._stopped.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                self._handle(connection)
            finally:
                try:
                    connection.close()
                except OSError:
                    pass

    def _handle(self, connection: socket.socket) -> None:
        try:
            connection.settimeout(10)
            raw = _read_line(connection)
            request = json.loads(raw)
        except (OSError, ValueError):
            _send(connection, {"ok": False, "error": "Request must be one JSON object per line."})
            return
        if not isinstance(request, dict):
            _send(connection, {"ok": False, "error": "Request must be a JSON object."})
            return
        operation = request.get("op")
        if operation == "stop":
            self._reply_stop(connection, request)
        elif operation == "status":
            self._reply_status(connection)
        elif operation == "command":
            self._reply_command(connection, request)
        else:
            _send(connection, {"ok": False, "error": f"Unknown operation: {operation!r}."})

    def _reply_stop(self, connection: socket.socket, request: dict) -> None:
        from simple_coding_agent.control import (
            ControlStoreError,
            PayloadMismatchError,
            RequestIdError,
            command_to_json,
        )

        request_id = request.get("request_id")
        try:
            record = self._control.submit_stop(request_id)
        except (RequestIdError, PayloadMismatchError) as error:
            _send(connection, {"ok": False, "error": str(error), "request_id": request_id})
            return
        except (ControlStoreError, RuntimeError, OSError) as error:
            _send(
                connection,
                {
                    "ok": False,
                    "error": f"Command could not be committed: {error}",
                    "request_id": request_id,
                },
            )
            return
        _send(connection, {"ok": True, **command_to_json(record)})

    def _reply_status(self, connection: socket.socket) -> None:
        from simple_coding_agent.control import ControlStoreError

        try:
            snapshot = self._control.control_status()
        except (ControlStoreError, RuntimeError, OSError) as error:
            _send(connection, {"ok": False, "error": f"Status is unavailable: {error}"})
            return
        _send(connection, {"ok": True, "status": snapshot})

    def _reply_command(self, connection: socket.socket, request: dict) -> None:
        from simple_coding_agent.control import ControlStoreError, command_to_json

        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            _send(connection, {"ok": False, "error": "A request ID is required."})
            return
        try:
            record = self._control.get_command(request_id)
        except (ControlStoreError, RuntimeError, OSError) as error:
            _send(connection, {"ok": False, "error": f"Lookup is unavailable: {error}"})
            return
        _send(
            connection,
            {
                "ok": True,
                "request_id": request_id,
                "command": command_to_json(record) if record is not None else None,
            },
        )


def socket_path_for(data_dir: Path) -> Path:
    """Return the runtime endpoint for one agent instance's data directory."""

    return data_dir / "state" / "agentctl.sock"


def send_request(socket_path: Path, request: dict) -> dict:
    """Send one request to the live agent; raise when it cannot be reached."""

    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    except OSError as error:
        raise ControlUnavailableError(
            f"Cannot connect to the agent process at {socket_path}: {error}"
        ) from error
    try:
        client.settimeout(10)
        try:
            client.connect(str(socket_path))
        except OSError as error:
            raise ControlUnavailableError(
                f"Cannot connect to the agent process at {socket_path}: {error}. "
                "The agent may not be running in this instance."
            ) from error
        try:
            client.sendall((json.dumps(request) + "\n").encode())
            raw = _read_line(client)
        except OSError as error:
            raise ControlUnavailableError(
                f"Lost connection to the agent process at {socket_path}: {error}"
            ) from error
    finally:
        try:
            client.close()
        except OSError:
            pass
    try:
        reply = json.loads(raw)
    except ValueError as error:
        raise ControlUnavailableError(
            f"Agent process at {socket_path} returned an unreadable reply."
        ) from error
    if not isinstance(reply, dict):
        raise ControlUnavailableError(
            f"Agent process at {socket_path} returned an unreadable reply."
        )
    return reply


def check_socket_private(socket_path: Path) -> bool:
    """Whether the endpoint is inaccessible to group and other users."""

    try:
        mode = stat.S_IMODE(os.stat(socket_path).st_mode)
    except OSError:
        return False
    return mode & 0o077 == 0


def _read_line(connection: socket.socket) -> str:
    data = bytearray()
    while True:
        chunk = connection.recv(65536)
        if not chunk:
            break
        data += chunk
        if len(data) > _BUFFER_LIMIT:
            raise ValueError("Request is too large")
        if data.endswith(b"\n"):
            break
    return bytes(data).decode().strip()


def _send(connection: socket.socket, payload: dict) -> None:
    try:
        connection.sendall((json.dumps(payload) + "\n").encode())
    except OSError:
        pass
