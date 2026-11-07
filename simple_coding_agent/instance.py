"""Durable instance identity for offline reads (operator contract).

The engine records its configured repository, Worker id and agent version
atomically in the state directory at startup. An offline reader on a
read-only mount reports the same repository and capabilities without the
engine, the lock, or any write.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


#: The instance record inside the state directory (``<data_dir>/state/...``).
INSTANCE_FILENAME = "instance.json"


class InstanceReadError(RuntimeError):
    """Raised when the stopped instance's identity cannot be read reliably."""


def record_instance(
    data_dir: Path, *, repository: str, worker_id: str = "", agent_version: str = ""
) -> None:
    """Atomically record this instance's identity (replacing any previous one)."""

    if not isinstance(repository, str) or not repository.strip():
        raise InstanceReadError("Instance repository must be a non-empty string")
    if not isinstance(worker_id, str):
        raise InstanceReadError("Instance worker id must be a string")
    path = Path(data_dir) / "state" / INSTANCE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repository": repository,
        "worker_id": worker_id,
        "agent_version": agent_version,
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".instance-",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(payload, file, sort_keys=True)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except OSError as error:
        raise InstanceReadError(
            "Instance identity could not be written"
        ) from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def read_instance(data_dir: Path) -> dict:
    """Return the recorded ``{"repository", "worker_id", "agent_version"}``.

    Raises :class:`InstanceReadError` when the record is missing or
    malformed instead of returning a guess.
    """

    path = Path(data_dir) / "state" / INSTANCE_FILENAME
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise InstanceReadError(
            f"Instance identity {path} is missing; the stopped instance"
            " cannot be identified reliably."
        ) from error
    except OSError as error:
        raise InstanceReadError(
            f"Instance identity {path} could not be read: {error}."
        ) from error
    try:
        payload = json.loads(raw)
    except (ValueError, json.JSONDecodeError) as error:
        raise InstanceReadError("Instance identity is malformed") from error
    if not isinstance(payload, dict):
        raise InstanceReadError("Instance identity is malformed")
    repository = payload.get("repository")
    worker_id = payload.get("worker_id", "")
    agent_version = payload.get("agent_version", "")
    if not isinstance(repository, str) or not repository.strip():
        raise InstanceReadError("Instance identity is malformed")
    if not isinstance(worker_id, str):
        raise InstanceReadError("Instance identity is malformed")
    if not isinstance(agent_version, str):
        raise InstanceReadError("Instance identity is malformed")
    return {
        "repository": repository,
        "worker_id": worker_id,
        "agent_version": agent_version,
    }
