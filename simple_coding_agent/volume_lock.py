"""Exclusive single-instance guard for one agent state directory.

Only one engine may own a state directory at a time: the lock is an
exclusive advisory lock (``fcntl.flock`` ``LOCK_EX|LOCK_NB``) on a lock
file inside the state directory, held for the process lifetime. Failing
to acquire it exits non-zero before serving or claiming; it is the only
new single-instance guard (claim semantics are unchanged).

The offline reader never takes this lock: it copies the volume files
read-only so a stopped instance stays readable while a live engine holds
the lock.
"""

from __future__ import annotations

import errno
import fcntl
from pathlib import Path


#: The lock file inside the state directory (``<data_dir>/state/...``).
LOCK_FILENAME = "volume.lock"


class VolumeLockError(RuntimeError):
    """Raised when the state directory is already owned by a live engine."""


class VolumeLock:
    """Hold an exclusive advisory lock on the state directory for one process."""

    def __init__(self, data_dir: Path) -> None:
        self._path = Path(data_dir) / "state" / LOCK_FILENAME
        self._file: object | None = None

    @property
    def path(self) -> Path:
        """The lock file this guard locks (created on acquire)."""

        return self._path

    @property
    def held(self) -> bool:
        """Whether this instance currently holds the lock."""

        return self._file is not None

    def acquire(self) -> None:
        """Acquire the exclusive lock without blocking; raise when held elsewhere.

        The lock file is created (with its parent directory) when missing.
        The file descriptor stays open for the process lifetime: closing it
        would release the advisory lock, so :meth:`release` is only for
        tests. A second engine on the same state directory fails here and
        must exit before serving or claiming.
        """

        if self._file is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self._path, "a+b")  # noqa: PTH123 - lock file by design
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            if error.errno in (errno.EACCES, errno.EAGAIN):
                raise VolumeLockError(
                    f"State directory {self._path.parent} is already owned by a live engine;"
                    f" lock file {self._path} is held by another process."
                ) from error
            raise VolumeLockError(
                f"Volume lock {self._path} could not be acquired: {error}."
            ) from error
        self._file = handle

    def release(self) -> None:
        """Release the lock (tests only; production holds it for the process lifetime)."""

        handle, self._file = self._file, None
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[union-attr]
        finally:
            handle.close()  # type: ignore[union-attr]

    def __enter__(self) -> VolumeLock:
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
