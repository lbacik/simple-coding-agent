"""Fail unless a release tag name is exactly ``v`` + the pyproject version.

Usage: check_tag_version.py <tag> [pyproject.toml]
"""

from __future__ import annotations

from pathlib import Path
import sys
import tomllib


def check_tag(tag: str, pyproject: Path) -> str | None:
    """Return an error message when the tag does not match, else ``None``."""

    version = tomllib.loads(pyproject.read_text())["project"]["version"]
    if tag == f"v{version}":
        return None
    return f"Tag {tag!r} does not match pyproject.toml version {version!r} (expected 'v{version}')."


def main(argv: list[str]) -> int:
    if not 2 <= len(argv) <= 3:
        print(__doc__, file=sys.stderr)
        return 2
    pyproject = Path(argv[2]) if len(argv) == 3 else Path("pyproject.toml")
    error = check_tag(argv[1], pyproject)
    if error is not None:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
