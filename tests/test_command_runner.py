from __future__ import annotations

import os
from pathlib import Path
import sys

from simple_coding_agent.command_runner import CommandRunner, CommandStatus


def test_runs_a_string_through_sh_and_captures_its_output(tmp_path: Path) -> None:
    result = CommandRunner().run(("printf 'hello'; printf 'problem' >&2",), timeout=5, cwd=tmp_path)

    assert result.status is CommandStatus.SUCCEEDED
    assert result.commands[0].stdout == "hello"
    assert result.commands[0].stderr == "problem"


def test_runs_list_in_order_and_stops_after_the_first_failure(tmp_path: Path) -> None:
    runner = CommandRunner()
    result = runner.run(("printf first", "printf failed >&2; exit 7", "printf never"), timeout=5, cwd=tmp_path)

    assert result.status is CommandStatus.FAILED
    assert [command.command for command in result.commands] == [
        "printf first",
        "printf failed >&2; exit 7",
    ]
    assert result.exit_code == 7


def test_timeout_covers_the_whole_command_sequence(tmp_path: Path) -> None:
    result = CommandRunner().run(
        (f"{sys.executable} -c 'import time; time.sleep(.12)'",) * 2,
        timeout=0.18,
        cwd=tmp_path,
    )

    assert result.status is CommandStatus.TIMED_OUT
    assert 1 <= len(result.commands) <= 2
    assert result.duration < 0.35


def test_subprocess_environment_is_limited_to_profile_and_path(tmp_path: Path) -> None:
    inherited = os.environ.copy()
    os.environ["GITHUB_TOKEN"] = "github-secret"
    os.environ["META_API_KEY"] = "meta-secret"
    try:
        result = CommandRunner().run(
            ("printf '%s|%s|%s|%s' \"$PROFILE_VALUE\" \"${GITHUB_TOKEN-unset}\" \"${META_API_KEY-unset}\" \"$PATH\"",),
            timeout=5,
            cwd=tmp_path,
            environment={"PROFILE_VALUE": "present"},
        )
    finally:
        os.environ.clear()
        os.environ.update(inherited)

    assert result.commands[0].stdout.startswith("present|unset|unset|")
    assert result.commands[0].stdout.split("|")[-1] == os.defpath


def test_subprocess_environment_strips_all_backend_credential_variables(
    tmp_path: Path,
) -> None:
    result = CommandRunner().run(
        (
            "printf '%s|%s|%s|%s|%s' "
            "\"${GITHUB_TOKEN-unset}\" \"${META_API_KEY-unset}\" "
            "\"${MODEL_API_KEY-unset}\" \"${ANTHROPIC_API_KEY-unset}\" "
            "\"${ANTHROPIC_AUTH_TOKEN-unset}\"",
        ),
        timeout=5,
        cwd=tmp_path,
        environment={
            "GITHUB_TOKEN": "github-secret",
            "META_API_KEY": "meta-secret",
            "MODEL_API_KEY": "model-secret",
            "ANTHROPIC_API_KEY": "api-key-secret",
            "ANTHROPIC_AUTH_TOKEN": "auth-token-secret",
        },
    )

    assert result.commands[0].stdout == "unset|unset|unset|unset|unset"


def test_scrubs_all_backend_credentials_from_captured_output(tmp_path: Path) -> None:
    result = CommandRunner(
        redactions=("meta-secret", "model-secret", "api-key-secret", "auth-token-secret")
    ).run(
        (
            "printf 'meta-secret model-secret api-key-secret auth-token-secret'; "
            "printf 'meta-secret model-secret api-key-secret auth-token-secret' >&2",
        ),
        timeout=5,
        cwd=tmp_path,
    )

    assert result.commands[0].stdout == "[REDACTED] [REDACTED] [REDACTED] [REDACTED]"
    assert result.commands[0].stderr == "[REDACTED] [REDACTED] [REDACTED] [REDACTED]"


def test_extra_path_is_prepended_to_the_subprocess_path(tmp_path: Path) -> None:
    result = CommandRunner(extra_path="/opt/homebrew/bin").run(
        ("printf '%s' \"$PATH\"",), timeout=5, cwd=tmp_path
    )

    assert result.commands[0].stdout == f"/opt/homebrew/bin:{os.defpath}"


def test_scrubs_explicit_redactions_from_captured_output(tmp_path: Path) -> None:
    result = CommandRunner(redactions=("top-secret",)).run(
        ("printf top-secret; printf top-secret >&2",), timeout=5, cwd=tmp_path
    )

    assert result.commands[0].stdout == "[REDACTED]"
    assert result.commands[0].stderr == "[REDACTED]"
