"""Reap processes a model session started once that session ends.

Every model session runs with a unique per-session marker in its
environment (``SCA_ATTEMPT_ID``). Child processes inherit the marker, so it
survives reparenting to PID 1 and ``setsid``. After the SDK client
disconnects, the executor scans ``/proc/*/environ`` for this session's
marker value and terminates every match: SIGTERM first, a short grace
period, then SIGKILL for survivors.

Processes without the marker -- or carrying a different session's value --
are never signalled. That includes operator ``docker exec`` shells,
``agentctl`` commands, PID 1, and the agent itself. Unreadable ``/proc``
entries and processes that exit mid-scan are skipped silently.
"""

from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path

#: Environment variable carrying the per-session process marker. The CLI
#: environment built for every model session sets it; children inherit it.
ATTEMPT_PROCESS_MARKER = "SCA_ATTEMPT_ID"

#: Grace period between SIGTERM and SIGKILL when reaping a session's
#: leftover processes.
ATTEMPT_PROCESS_GRACE_SECONDS = 2.0


@dataclass(frozen=True)
class ReapedProcess:
    """One terminated process: its PID, command line, and escalation."""

    pid: int
    cmdline: str
    sigkill: bool


def _environ_has_marker(environ: bytes, marker: bytes) -> bool:
    """Whether a raw ``/proc/<pid>/environ`` blob carries the marker entry."""

    return marker in environ.split(b"\x00")


def _read_cmdline(proc_root: Path, pid: int) -> str:
    """Best-effort command line for ``pid``; never raises."""

    try:
        raw = (proc_root / str(pid) / "cmdline").read_bytes()
    except OSError:
        raw = b""
    parts = [part.decode(errors="replace") for part in raw.split(b"\x00") if part]
    if parts:
        return " ".join(parts)
    try:
        comm = (proc_root / str(pid) / "comm").read_text().strip()
    except OSError:
        comm = ""
    return comm or f"<pid {pid}>"


def _is_zombie(proc_root: Path, pid: int) -> bool:
    """Whether ``pid`` is already a zombie (dead, awaiting reaping)."""

    try:
        stat = (proc_root / str(pid) / "stat").read_bytes()
    except OSError:
        return True
    try:
        state = stat.rsplit(b")", 1)[1].split()[0]
    except IndexError:
        return False
    return state == b"Z"


def find_marked_pids(
    marker_name: str, marker_value: str, proc_root: Path = Path("/proc")
) -> list[int]:
    """Return the PIDs whose environment carries this session's marker."""

    marker = f"{marker_name}={marker_value}".encode()
    own_pid = os.getpid()
    found: list[int] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in (1, own_pid):
            continue
        try:
            environ = (proc_root / entry.name / "environ").read_bytes()
        except OSError:
            continue
        if _environ_has_marker(environ, marker):
            found.append(pid)
    return sorted(found)


def reap_marked_processes(
    marker_name: str,
    marker_value: str,
    *,
    grace_seconds: float = ATTEMPT_PROCESS_GRACE_SECONDS,
    proc_root: Path = Path("/proc"),
) -> list[ReapedProcess]:
    """Terminate every process carrying this session's marker value.

    Sends SIGTERM to all matches, waits ``grace_seconds``, then sends
    SIGKILL to survivors. Zombies count as terminated (they hold no ports
    or memory; init reaps them). Returns one entry per terminated process.
    """

    targets = [pid for pid in find_marked_pids(marker_name, marker_value, proc_root)]
    if not targets:
        return []
    cmdlines = {pid: _read_cmdline(proc_root, pid) for pid in targets}
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
    if grace_seconds > 0:
        time.sleep(grace_seconds)
    reaped: list[ReapedProcess] = []
    for pid in targets:
        try:
            os.kill(pid, 0)
        except OSError:
            reaped.append(ReapedProcess(pid=pid, cmdline=cmdlines[pid], sigkill=False))
            continue
        if _is_zombie(proc_root, pid):
            reaped.append(ReapedProcess(pid=pid, cmdline=cmdlines[pid], sigkill=False))
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            # Exited between the liveness check and the kill: gone either way.
            reaped.append(ReapedProcess(pid=pid, cmdline=cmdlines[pid], sigkill=False))
            continue
        reaped.append(ReapedProcess(pid=pid, cmdline=cmdlines[pid], sigkill=True))
    # Give SIGKILL a moment to land so a caller that re-scans immediately
    # (or asserts the port is free) sees the effect.
    time.sleep(0.2)
    return reaped
