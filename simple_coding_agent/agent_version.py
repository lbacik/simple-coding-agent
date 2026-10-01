"""The running agent's own version, read from installed package metadata."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

PACKAGE_NAME = "simple-coding-agent"

#: Reported when the installed metadata cannot be read.
UNKNOWN_VERSION = "unknown"


def agent_version() -> str:
    """Return the installed agent version, or ``unknown`` if it cannot be read."""

    try:
        return version(PACKAGE_NAME)
    except (PackageNotFoundError, ValueError, OSError):
        return UNKNOWN_VERSION
