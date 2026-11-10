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
import time
from pathlib import Path
from typing import Callable, Protocol


class ControlSurface(Protocol):
    """The lifecycle-facing control operations served over the socket."""

    @property
    def repository(self) -> str: ...

    def submit_stop(self, request_id: str, *, has_active_attempt: bool = ...) -> object: ...

    def submit_stop_after(self, request_id: str, after: object) -> object: ...

    def submit_next_issue(self, request_id: str, issue: object) -> object: ...

    def submit_resume(
        self, request_id: str, *, reject_open_intake: bool = ...
    ) -> object: ...

    def submit_handoff(self, request_id: str) -> object: ...

    def submit_recovery_retry(self, request_id: str, attempt_id: object) -> object: ...

    def submit_recovery_release(
        self, request_id: str, attempt_id: object, saved_at: object
    ) -> object: ...

    def submit_errors_reset(self, request_id: str) -> object: ...

    def get_command(self, request_id: str) -> object | None: ...

    def control_identity(self) -> dict:
        """The contract identity of this instance (backend, version, caps)."""
        ...

    def control_status(self, repository: str | None = ...) -> dict: ...

    def telemetry_snapshot(self) -> dict:
        """Build one telemetry snapshot for the ``telemetry`` operation."""
        ...

    def contract_status_snapshot(self) -> dict:
        """Build one contract status snapshot for the ``contract_status`` op."""
        ...

    def control_attempts_list(
        self, limit: int | None = ..., before: str | None = ...
    ) -> dict:
        """Build one ``attempts list`` page over archives plus the checkpoint."""
        ...

    def control_attempts_show(self, attempt_id: str) -> dict:
        """Build one ``attempts show`` summary plus evidence manifest."""
        ...

    def control_attempts_export(self, attempt_id: str) -> dict:
        """Build one ``attempts export`` summary plus export manifest."""
        ...


class ControlUnavailableError(ConnectionError):
    """Raised when the agent process cannot be reached."""


class UnreadableReplyError(ControlUnavailableError):
    """Raised when the agent answered but the reply cannot be used.

    A subclass of :class:`ControlUnavailableError` so existing callers keep
    reporting it as a connection problem; ``agentctl telemetry`` distinguishes
    it to report ``snapshot_failed`` instead of ``agent_unreachable``.
    """


_SOCKET_BACKLOG = 16
_BUFFER_LIMIT = 1 << 20

#: Container-local runtime directory holding the private control socket. It is
#: never bind-mounted or shared: each container has its own filesystem, so the
#: fixed path below addresses exactly one agent instance.
RUNTIME_DIR = Path("/run/simple-coding-agent")

#: The one control endpoint inside the selected container.
DEFAULT_SOCKET_PATH = RUNTIME_DIR / "control.sock"

#: Test/dev override selecting a different endpoint (e.g. per-instance
#: temporary sockets when several instances share one host).
SOCKET_ENV_VAR = "AGENTCTL_SOCKET"


class ControlServer:
    """Serve one agent instance's control surface on a Unix domain socket."""

    def __init__(
        self,
        socket_path: Path,
        control: ControlSurface,
        *,
        event_log: Callable[..., None] | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._control = control
        self._event_log = event_log
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self._telemetry_failed_at: float | None = None

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
        """Stop serving and remove the runtime endpoint.

        Only the listener thread is joined (bounded at 10 s). In-flight
        connection threads are daemons owned by this server: they finish
        their single reply on their own and never hold the listener open
        or keep the socket path in use after ``stop()`` unlinks it.
        """

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

    def _log_control_error(self, detail: str) -> None:
        """Record one failed control request as a structured event, if wired."""

        event_log = self._event_log
        if event_log is None:
            return
        try:
            event_log("control_request_failed", detail=detail, level="ERROR")
        except Exception:
            pass

    def _safe_repository(self) -> str:
        try:
            return self._repository()
        except Exception:
            return "unknown"

    def _send_failure(self, connection: socket.socket, error: object) -> None:
        """Answer one failed request without ever raising out of the listener."""

        try:
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "error": f"Control request failed: {error}",
                },
            )
        except Exception:
            pass

    def _serve(self) -> None:
        """Accept connections until :meth:`stop` closes the listener.

        Each connection is served on its own daemon thread so one
        connection waiting on the control lock (a mutating submit, or the
        intake claim across GitHub) never holds up ``telemetry`` or
        ``command <id>`` behind it. A failing connection is contained to
        its own thread and never takes down the listener.
        """

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
            worker = threading.Thread(
                target=self._serve_connection,
                args=(connection,),
                name="agentctl-connection",
                daemon=True,
            )
            worker.start()

    def _serve_connection(self, connection: socket.socket) -> None:
        """Serve one connection on its own thread, then close it."""

        try:
            try:
                self._handle(connection)
            except Exception as error:
                # One failing request must never stop the listener: log
                # the failure and answer on the same connection when it
                # is still writable, then close this connection.
                self._log_control_error(f"{type(error).__name__}: {error}")
                self._send_failure(connection, error)
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
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "error": "Request must be one JSON object per line.",
                },
            )
            return
        if not isinstance(request, dict):
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "error": "Request must be a JSON object.",
                },
            )
            return
        operation = request.get("op")
        try:
            if operation == "stop":
                self._reply_stop(connection, request)
            elif operation == "stop_after":
                self._reply_stop_after(connection, request)
            elif operation == "next_issue":
                self._reply_next_issue(connection, request)
            elif operation == "resume":
                self._reply_resume(connection, request)
            elif operation == "handoff":
                self._reply_handoff(connection, request)
            elif operation == "recovery_retry":
                self._reply_recovery_retry(connection, request)
            elif operation == "recovery_release":
                self._reply_recovery_release(connection, request)
            elif operation == "errors_reset":
                self._reply_errors_reset(connection, request)
            elif operation == "identity":
                self._reply_identity(connection)
            elif operation == "status":
                self._reply_status(connection)
            elif operation == "command":
                self._reply_command(connection, request)
            elif operation == "telemetry":
                self._reply_telemetry(connection)
            elif operation == "contract_status":
                self._reply_contract_status(connection)
            elif operation == "attempts_list":
                self._reply_attempts_list(connection, request)
            elif operation == "attempts_show":
                self._reply_attempts_show(connection, request)
            elif operation == "attempts_export":
                self._reply_attempts_export(connection, request)
            else:
                _send(
                    connection,
                    {
                        "ok": False,
                        "repository": self._safe_repository(),
                        "error": f"Unknown operation: {operation!r}.",
                    },
                )
        except Exception as error:
            self._log_control_error(f"{operation}: {type(error).__name__}: {error}")
            self._send_failure(connection, error)

    def _reply_stop(self, connection: socket.socket, request: dict) -> None:
        from simple_coding_agent.control import PayloadMismatchError, RequestIdError

        _reply_mutating(
            connection,
            request,
            self._control.submit_stop,
            rejected=(RequestIdError, PayloadMismatchError),
            repository=self._repository(),
            check_expect_repo=True,
            control=self._control,
        )

    def _reply_stop_after(self, connection: socket.socket, request: dict) -> None:
        from simple_coding_agent.control import (
            PayloadMismatchError,
            RequestIdError,
            StopAfterRejectedError,
        )

        after = request.get("after")
        _reply_mutating(
            connection,
            request,
            lambda request_id: self._control.submit_stop_after(request_id, after),
            rejected=(RequestIdError, PayloadMismatchError, StopAfterRejectedError),
            repository=self._repository(),
            check_expect_repo=True,
            control=self._control,
        )

    def _reply_next_issue(self, connection: socket.socket, request: dict) -> None:
        from simple_coding_agent.control import (
            NextIssueRejectedError,
            PayloadMismatchError,
            RequestIdError,
        )

        issue = request.get("issue")
        _reply_mutating(
            connection,
            request,
            lambda request_id: self._control.submit_next_issue(request_id, issue),
            rejected=(RequestIdError, PayloadMismatchError, NextIssueRejectedError),
            repository=self._repository(),
            check_expect_repo=True,
            control=self._control,
        )

    def _reply_resume(self, connection: socket.socket, request: dict) -> None:
        from simple_coding_agent.control import (
            PayloadMismatchError,
            RequestIdError,
            ResumeBlockedError,
            ResumeIntakeOpenError,
        )

        # The operator contract only lets ``resume`` reopen closed intake; the
        # legacy surface keeps answering an idempotent ``completed`` no-op.
        contract_mode = _is_contract_request(request)

        def submit(request_id: object) -> object:
            if contract_mode:
                return self._control.submit_resume(request_id, reject_open_intake=True)
            return self._control.submit_resume(request_id)

        _reply_mutating(
            connection,
            request,
            submit,
            rejected=(
                RequestIdError,
                PayloadMismatchError,
                ResumeBlockedError,
                ResumeIntakeOpenError,
            ),
            repository=self._repository(),
            check_expect_repo=True,
            control=self._control,
        )

    def _reply_handoff(self, connection: socket.socket, request: dict) -> None:
        from simple_coding_agent.control import (
            HandoffRejectedError,
            PayloadMismatchError,
            RequestIdError,
        )

        _reply_mutating(
            connection,
            request,
            self._control.submit_handoff,
            rejected=(RequestIdError, PayloadMismatchError, HandoffRejectedError),
            repository=self._repository(),
            check_expect_repo=True,
            control=self._control,
        )

    def _reply_recovery_retry(self, connection: socket.socket, request: dict) -> None:
        from simple_coding_agent.control import PayloadMismatchError, RequestIdError
        from simple_coding_agent.recovery import RecoveryRejectedError

        attempt_id = request.get("attempt_id")
        _reply_mutating(
            connection,
            request,
            lambda request_id: self._control.submit_recovery_retry(
                request_id, attempt_id
            ),
            rejected=(RequestIdError, PayloadMismatchError, RecoveryRejectedError),
            repository=self._repository(),
            check_expect_repo=True,
            control=self._control,
        )

    def _reply_recovery_release(self, connection: socket.socket, request: dict) -> None:
        from simple_coding_agent.control import PayloadMismatchError, RequestIdError
        from simple_coding_agent.recovery import RecoveryRejectedError

        attempt_id = request.get("attempt_id")
        saved_at = request.get("saved_at")
        _reply_mutating(
            connection,
            request,
            lambda request_id: self._control.submit_recovery_release(
                request_id, attempt_id, saved_at
            ),
            rejected=(RequestIdError, PayloadMismatchError, RecoveryRejectedError),
            repository=self._repository(),
            check_expect_repo=True,
            control=self._control,
        )

    def _reply_errors_reset(self, connection: socket.socket, request: dict) -> None:
        from simple_coding_agent.control import (
            ErrorsResetRejectedError,
            PayloadMismatchError,
            RequestIdError,
        )

        _reply_mutating(
            connection,
            request,
            self._control.submit_errors_reset,
            rejected=(RequestIdError, PayloadMismatchError, ErrorsResetRejectedError),
            repository=self._repository(),
            check_expect_repo=True,
            control=self._control,
        )

    def _reply_identity(self, connection: socket.socket) -> None:
        """Answer one read-only contract ``identity`` request.

        Lock-free and without network, git, or provider I/O, so
        ``identity`` stays within the contract's read budget even while a
        mutation holds the control lock.
        """

        def _fail_identity(reason: str) -> None:
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "error": f"Identity is unavailable: {reason}",
                },
            )

        snapshotter = getattr(self._control, "control_identity", None)
        if not callable(snapshotter):
            _fail_identity("this agent version does not serve identity.")
            return
        try:
            snapshot = snapshotter()
        except Exception as error:
            _fail_identity(str(error))
            return
        if not isinstance(snapshot, dict):
            _fail_identity("the identity snapshot was not a JSON object.")
            return
        _send(connection, {"ok": True, "identity": snapshot})

    def _reply_status(self, connection: socket.socket) -> None:
        from simple_coding_agent.control import ControlStoreError

        try:
            snapshot = self._control.control_status()
        except (ControlStoreError, RuntimeError, OSError) as error:
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._repository(),
                    "error": f"Status is unavailable: {error}",
                },
            )
            return
        _send(connection, {"ok": True, "status": snapshot})

    def _reply_telemetry(self, connection: socket.socket) -> None:
        """Answer one telemetry request without ever raising out of the listener.

        Controls without a telemetry reader (older agent versions) answer
        ``Unknown operation`` so ``agentctl`` can report
        ``unsupported_operation``. Any other failure answers
        ``snapshot_failed`` with a rate-limited ``telemetry_failed`` event.
        """

        from simple_coding_agent.telemetry import TELEMETRY_FAILURE_COOLDOWN_SECONDS

        snapshotter = getattr(self._control, "telemetry_snapshot", None)
        if not callable(snapshotter):
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "error": "Unknown operation: 'telemetry'.",
                },
            )
            return
        try:
            snapshot = snapshotter()
        except Exception as error:
            self._log_telemetry_failed(
                f"{type(error).__name__}: {error}",
                cooldown=TELEMETRY_FAILURE_COOLDOWN_SECONDS,
            )
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "error": f"Telemetry snapshot failed: {error}",
                },
            )
            return
        if not isinstance(snapshot, dict):
            self._log_telemetry_failed(
                "telemetry reader returned a non-object snapshot",
                cooldown=TELEMETRY_FAILURE_COOLDOWN_SECONDS,
            )
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "error": "Telemetry snapshot failed: the snapshot was not a JSON object.",
                },
            )
            return
        try:
            json.dumps(snapshot)
        except (TypeError, ValueError) as error:
            self._log_telemetry_failed(
                f"telemetry snapshot is not JSON-serialisable: {error}",
                cooldown=TELEMETRY_FAILURE_COOLDOWN_SECONDS,
            )
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "error": f"Telemetry snapshot failed: {error}",
                },
            )
            return
        _send(connection, {"ok": True, "telemetry": snapshot})

    def _reply_contract_status(self, connection: socket.socket) -> None:
        """Answer one contract ``status`` request without raising out of the listener.

        Controls without a contract projection (older agent versions)
        answer ``Unknown operation`` so ``agentctl`` can report
        ``unsupported_operation``. Any other failure answers ``ok: False``
        so the adapter reports ``internal``.
        """

        snapshotter = getattr(self._control, "contract_status_snapshot", None)
        if not callable(snapshotter):
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "error": "Unknown operation: 'contract_status'.",
                },
            )
            return
        try:
            snapshot = snapshotter()
        except Exception as error:
            self._log_control_error(f"contract_status: {type(error).__name__}: {error}")
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "error": f"Contract status failed: {error}",
                },
            )
            return
        if not isinstance(snapshot, dict):
            self._log_control_error("contract_status returned a non-object snapshot")
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "error": "Contract status failed: the snapshot was not a JSON object.",
                },
            )
            return
        try:
            json.dumps(snapshot)
        except (TypeError, ValueError) as error:
            self._log_control_error(
                f"contract_status snapshot is not JSON-serialisable: {error}"
            )
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "error": f"Contract status failed: {error}",
                },
            )
            return
        _send(connection, {"ok": True, "contract_status": snapshot})

    def _reply_attempts_list(
        self, connection: socket.socket, request: dict
    ) -> None:
        """Answer one contract ``attempts list`` request.

        Controls without the history projection (older agent versions)
        answer ``Unknown operation`` so the adapter reports ``internal``.
        A rejected page answers ``ok: False`` with its contract code
        (``invalid_argument``); any other failure answers ``internal``.
        """

        from simple_coding_agent.attempt_history import AttemptHistoryError

        snapshotter = getattr(self._control, "control_attempts_list", None)
        if not callable(snapshotter):
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "error": "Unknown operation: 'attempts_list'.",
                },
            )
            return
        try:
            page = snapshotter(request.get("limit"), request.get("before"))
        except AttemptHistoryError as error:
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "code": error.code,
                    "error": str(error),
                },
            )
            return
        except Exception as error:
            self._log_control_error(f"attempts_list: {type(error).__name__}: {error}")
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "code": "internal",
                    "error": f"Attempts list failed: {error}",
                },
            )
            return
        if not isinstance(page, dict):
            self._log_control_error("attempts_list returned a non-object page")
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "code": "internal",
                    "error": "Attempts list failed: the page was not a JSON object.",
                },
            )
            return
        try:
            json.dumps(page)
        except (TypeError, ValueError) as error:
            self._log_control_error(f"attempts_list page is not JSON-serialisable: {error}")
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "code": "internal",
                    "error": f"Attempts list failed: {error}",
                },
            )
            return
        _send(connection, {"ok": True, "attempts_list": page})

    def _reply_attempts_show(
        self, connection: socket.socket, request: dict
    ) -> None:
        """Answer one contract ``attempts show`` request.

        A missing Attempt answers ``ok: False`` with ``not_found`` and a
        bad Attempt ID with ``invalid_argument``; any other failure answers
        ``internal``.
        """

        from simple_coding_agent.attempt_history import AttemptHistoryError

        snapshotter = getattr(self._control, "control_attempts_show", None)
        if not callable(snapshotter):
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "error": "Unknown operation: 'attempts_show'.",
                },
            )
            return
        try:
            result = snapshotter(request.get("attempt_id"))
        except AttemptHistoryError as error:
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "code": error.code,
                    "error": str(error),
                },
            )
            return
        except Exception as error:
            self._log_control_error(f"attempts_show: {type(error).__name__}: {error}")
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "code": "internal",
                    "error": f"Attempts show failed: {error}",
                },
            )
            return
        if not isinstance(result, dict):
            self._log_control_error("attempts_show returned a non-object result")
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "code": "internal",
                    "error": "Attempts show failed: the result was not a JSON object.",
                },
            )
            return
        try:
            json.dumps(result)
        except (TypeError, ValueError) as error:
            self._log_control_error(f"attempts_show result is not JSON-serialisable: {error}")
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "code": "internal",
                    "error": f"Attempts show failed: {error}",
                },
            )
            return
        _send(connection, {"ok": True, "attempts_show": result})

    def _reply_attempts_export(
        self, connection: socket.socket, request: dict
    ) -> None:
        """Answer one contract ``attempts export`` read.

        Serves the same shape as ``attempts show`` (summary plus export
        manifest over the snapshot) so the consumer can stream the tar
        from the volume itself; the tar never crosses this socket. A
        missing Attempt answers ``ok: False`` with ``not_found`` and a
        bad Attempt ID with ``invalid_argument``; any other failure
        answers ``internal``.
        """

        from simple_coding_agent.attempt_history import AttemptHistoryError

        snapshotter = getattr(self._control, "control_attempts_export", None)
        if not callable(snapshotter):
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "error": "Unknown operation: 'attempts_export'.",
                },
            )
            return
        try:
            result = snapshotter(request.get("attempt_id"))
        except AttemptHistoryError as error:
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "code": error.code,
                    "error": str(error),
                },
            )
            return
        except Exception as error:
            self._log_control_error(f"attempts_export: {type(error).__name__}: {error}")
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "code": "internal",
                    "error": f"Attempts export failed: {error}",
                },
            )
            return
        if not isinstance(result, dict):
            self._log_control_error("attempts_export returned a non-object result")
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "code": "internal",
                    "error": "Attempts export failed: the result was not a JSON object.",
                },
            )
            return
        try:
            json.dumps(result)
        except (TypeError, ValueError) as error:
            self._log_control_error(f"attempts_export result is not JSON-serialisable: {error}")
            _send(
                connection,
                {
                    "ok": False,
                    "repository": self._safe_repository(),
                    "code": "internal",
                    "error": f"Attempts export failed: {error}",
                },
            )
            return
        _send(connection, {"ok": True, "attempts_export": result})

    def _log_telemetry_failed(self, detail: str, *, cooldown: float) -> None:
        """Record a failed telemetry build, at most once per cooldown window."""

        now = time.monotonic()
        previous = self._telemetry_failed_at
        if previous is not None and now - previous < cooldown:
            return
        self._telemetry_failed_at = now
        event_log = self._event_log
        if event_log is None:
            return
        try:
            event_log("telemetry_failed", detail=detail, level="ERROR")
        except Exception:
            pass

    def _reply_command(self, connection: socket.socket, request: dict) -> None:
        from simple_coding_agent.control import ControlStoreError, command_to_json

        # The repository label here must never take the control lock:
        # ``control_status()`` serializes on it, which would make the
        # lookup wait behind the very submit it is meant to observe.
        repository = getattr(self._control, "repository", None)
        if not isinstance(repository, str) or not repository:
            repository = "unknown"
        request_id = request.get("request_id")
        contract_mode = _is_contract_request(request)
        if not isinstance(request_id, str) or not request_id:
            payload: dict = {
                "ok": False,
                "repository": repository,
                "error": "A request ID is required.",
            }
            if contract_mode:
                payload["error_kind"] = "invalid_argument"
                payload["request_id"] = request_id
            _send(connection, payload)
            return
        if contract_mode and _CONTRACT_REQUEST_ID_RE.fullmatch(request_id) is None:
            _send(
                connection,
                {
                    "ok": False,
                    "repository": repository,
                    "error": (
                        f"invalid request ID {request_id!r}: must match"
                        " ^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
                    ),
                    "request_id": request_id,
                    "error_kind": "invalid_argument",
                },
            )
            return
        try:
            if contract_mode:
                getter = getattr(
                    self._control, "get_command_with_payload", None
                )
                if callable(getter):
                    with_payload = getter(request_id)
                else:
                    record_only = self._control.get_command(request_id)
                    with_payload = (
                        (record_only, {}) if record_only is not None else None
                    )
            else:
                record_only = self._control.get_command(request_id)
                with_payload = None
                record = record_only
        except (ControlStoreError, RuntimeError, OSError) as error:
            payload = {
                "ok": False,
                "repository": repository,
                "error": f"Lookup is unavailable: {error}",
            }
            if contract_mode:
                payload["error_kind"] = "internal"
                payload["request_id"] = request_id
            _send(connection, payload)
            return
        if not contract_mode:
            _send(
                connection,
                {
                    "ok": True,
                    "request_id": request_id,
                    "repository": repository,
                    "command": command_to_json(record) if record is not None else None,
                },
            )
            return
        if with_payload is None:
            _send(
                connection,
                {
                    "ok": False,
                    "repository": repository,
                    "error": f"no command is recorded for {request_id}",
                    "request_id": request_id,
                    "error_kind": "not_found",
                },
            )
            return
        try:
            from simple_coding_agent import contract_status as contract_status_module

            found_record, found_payload = with_payload
            found_payload = (
                found_payload if isinstance(found_payload, dict) else {}
            )
            command = contract_status_module.project_command(
                found_record, found_payload, repository=repository
            )
        except Exception as error:
            _send(
                connection,
                {
                    "ok": False,
                    "repository": repository,
                    "error": f"Lookup is unavailable: {error}",
                    "request_id": request_id,
                    "error_kind": "internal",
                },
            )
            return
        _send(
            connection,
            {
                "ok": True,
                "request_id": request_id,
                "repository": repository,
                "command": command,
            },
        )

    def _repository(self) -> str:
        """The configured ``TARGET_REPO`` identifying the controlled instance."""

        repository = getattr(self._control, "repository", None)
        if isinstance(repository, str) and repository:
            return repository
        try:
            snapshot = self._control.control_status()
        except Exception:
            return "unknown"
        if isinstance(snapshot, dict) and snapshot.get("repository"):
            return str(snapshot["repository"])
        return "unknown"


def resolve_socket_path(explicit: str | Path | None = None) -> Path:
    """Resolve the control endpoint shared by the agent and its CLI.

    Production containers always use the container-local runtime socket at
    ``/run/simple-coding-agent/control.sock``: each container owns its
    filesystem, so that fixed path addresses exactly the selected instance.
    An explicit path (the CLI's ``--socket``) wins; otherwise
    ``AGENTCTL_SOCKET`` overrides the endpoint for tests and local
    development where several instances share one host.
    """

    if explicit is not None:
        return Path(explicit).expanduser()
    override = os.environ.get(SOCKET_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return DEFAULT_SOCKET_PATH


def socket_path_for(data_dir: Path | None = None) -> Path:
    """Return the control endpoint for one agent instance.

    ``data_dir`` is accepted for backward compatibility and otherwise unused:
    the endpoint is container-local runtime state, not derived from the
    persistent data directory. Prefer :func:`resolve_socket_path`.
    """

    return resolve_socket_path()


def _repository_matches(expected: object, repository: str) -> bool:
    """Whether an ``expect_repo`` value names this instance's repository.

    The comparison is case-insensitive with surrounding whitespace ignored;
    anything else (including a malformed value) simply does not match.
    """

    if not isinstance(expected, str):
        return False
    return expected.strip().lower() == repository.strip().lower()


#: Contract request IDs match the operator-contract pattern (spec 6.2).
_CONTRACT_REQUEST_ID_RE = __import__("re").compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
)


def _is_contract_request(request: dict) -> bool:
    """Whether one socket request runs on the operator-contract path."""

    return bool(request.get("contract"))


def _rejection_kind(error: Exception) -> str:
    """Map one answered rejection to its machine-readable contract kind."""

    from simple_coding_agent.control import (
        NextIssueRejectedError,
        PayloadMismatchError,
        RequestIdError,
        StopAfterRejectedError,
    )
    from simple_coding_agent.recovery import (
        RecoveryAttemptNotFoundError,
        RecoveryRejectedError,
    )

    if isinstance(error, PayloadMismatchError):
        return "payload_mismatch"
    if isinstance(error, RequestIdError):
        return "invalid_argument"
    if isinstance(error, RecoveryAttemptNotFoundError):
        return "not_found"
    if isinstance(
        error, (StopAfterRejectedError, NextIssueRejectedError, RecoveryRejectedError)
    ) and str(error).startswith("Invalid"):
        return "invalid_argument"
    return "rejected"


def _reply_mutating(
    connection: socket.socket,
    request: dict,
    submit: Callable[[object], object],
    *,
    rejected: tuple[type[Exception], ...],
    repository: str,
    check_expect_repo: bool = False,
    control: object | None = None,
) -> None:
    """Reply to one mutating command with its durable acknowledgement.

    Rejections (bad IDs, reused IDs, recovery holds) report the reason with
    no control change; storage or liveness failures never appear accepted.

    When ``check_expect_repo`` is set, an ``expect_repo`` request field is
    compared against this instance's repository before the command reaches
    ``submit`` (so the check also runs before the request-ID idempotency
    lookup): a mismatch is an ordinary rejection with nothing stored and
    no control change. The flag is never part of the stored canonical
    payload. Operations with ``check_expect_repo`` unset ignore the field.

    On the contract path (``request["contract"]`` truthy) ``expect_repo``
    is mandatory for every kind, the check still runs before the
    idempotency lookup, rejections carry a machine-readable ``error_kind``
    next to the human message, successes carry the projected
    ``command`` plus ``replayed``, and engine exceptions carry
    ``error_kind`` ``internal`` with ``stored: False`` (the failure
    happened before the commit). The legacy path is byte-compatible:
    it never sends the flag and ignores the added fields.
    """

    from simple_coding_agent.control import ControlStoreError, command_to_json

    request_id = request.get("request_id")
    contract_mode = _is_contract_request(request)
    needs_expect_repo = check_expect_repo or contract_mode
    if needs_expect_repo and (
        contract_mode or request.get("expect_repo") is not None
    ):
        expected = request.get("expect_repo")
        if not _repository_matches(expected, repository):
            shown = expected.strip() if isinstance(expected, str) else expected
            payload: dict = {
                "ok": False,
                "repository": repository,
                "error": (
                    f"expected repository {shown},"
                    f" this instance targets {repository}"
                ),
                "request_id": request_id,
            }
            if contract_mode:
                payload["error_kind"] = "repository_mismatch"
                payload["details"] = {"repository": repository}
            _send(connection, payload)
            return
    if contract_mode and (
        not isinstance(request_id, str)
        or _CONTRACT_REQUEST_ID_RE.fullmatch(request_id) is None
    ):
        _send(
            connection,
            {
                "ok": False,
                "repository": repository,
                "error": (
                    f"invalid request ID {request_id!r}: must match"
                    " ^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
                ),
                "request_id": request_id,
                "error_kind": "invalid_argument",
            },
        )
        return
    if contract_mode:
        try:
            lookup = control if control is not None else getattr(submit, "__self__", None)
            get_command = getattr(lookup, "get_command", None)
            existing = get_command(request_id) if callable(get_command) else None
        except Exception:
            existing = None
        replayed_before = existing is not None
    else:
        replayed_before = False
    try:
        record = submit(request_id)
    except rejected as error:
        payload = {
            "ok": False,
            "repository": repository,
            "error": str(error),
            "request_id": request_id,
        }
        if contract_mode:
            kind = _rejection_kind(error)
            payload["error_kind"] = kind
            if kind == "payload_mismatch":
                pass
            elif kind == "repository_mismatch":
                payload["details"] = {"repository": repository}
        _send(connection, payload)
        return
    except (ControlStoreError, RuntimeError, OSError) as error:
        payload = {
            "ok": False,
            "repository": repository,
            "error": f"Command could not be committed: {error}",
            "request_id": request_id,
        }
        if contract_mode:
            payload["error_kind"] = "internal"
            payload["stored"] = False
        _send(connection, payload)
        return
    if not contract_mode:
        _send(
            connection,
            {"ok": True, "repository": repository, **command_to_json(record)},
        )
        return
    try:
        from simple_coding_agent import contract_status as contract_status_module

        payload_dict: dict = {}
        try:
            lookup = control if control is not None else getattr(submit, "__self__", None)
            get_with_payload = getattr(lookup, "get_command_with_payload", None)
            with_payload = (
                get_with_payload(request_id) if callable(get_with_payload) else None
            )
        except Exception:
            with_payload = None
        if with_payload is not None:
            payload_dict = with_payload[1] if isinstance(with_payload[1], dict) else {}
        command = contract_status_module.project_command(
            record, payload_dict, repository=repository
        )
    except Exception as error:
        _send(
            connection,
            {
                "ok": False,
                "repository": repository,
                "error": f"Command could not be committed: {error}",
                "request_id": request_id,
                "error_kind": "internal",
            },
        )
        return
    _send(
        connection,
        {
            "ok": True,
            "repository": repository,
            "command": command,
            "replayed": bool(replayed_before),
        },
    )


def send_request(socket_path: Path, request: dict, *, timeout: float = 10.0) -> dict:
    """Send one request to the live agent; raise when it cannot be reached.

    ``timeout`` covers the connect and the read. Unreadable or oversized
    replies raise :class:`UnreadableReplyError`; every other failure raises
    :class:`ControlUnavailableError`.
    """

    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    except OSError as error:
        raise ControlUnavailableError(
            f"Cannot connect to the agent process at {socket_path}: {error}"
        ) from error
    try:
        client.settimeout(timeout)
        try:
            client.connect(str(socket_path))
        except OSError as error:
            raise ControlUnavailableError(
                f"Cannot connect to the agent process at {socket_path}: {error}. "
                "The agent may not be running in this instance."
            ) from error
        encoded = (json.dumps(request) + "\n").encode()
        try:
            client.sendall(encoded)
            raw = _read_line(client)
        except OSError as error:
            raise ControlUnavailableError(
                f"Lost connection to the agent process at {socket_path}: {error}"
            ) from error
        except ValueError as error:
            raise UnreadableReplyError(
                f"Agent process at {socket_path} returned an oversized reply: {error}."
            ) from error
    finally:
        try:
            client.close()
        except OSError:
            pass
    try:
        reply = json.loads(raw)
    except ValueError as error:
        raise UnreadableReplyError(
            f"Agent process at {socket_path} returned an unreadable reply."
        ) from error
    if not isinstance(reply, dict):
        raise UnreadableReplyError(
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
