"""Validated configuration supplied by the operator."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Mapping

import yaml


class ConfigurationError(ValueError):
    """Raised when operator configuration cannot be used safely."""


class ProfileError(ValueError):
    """Raised when a repository profile cannot support an implementation attempt."""

    outcome = "infrastructure_error"


_TARGET_REPOSITORY = re.compile(r"^[^/\s]+/[^/\s]+$")
_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})
_DEFAULT_MODEL_NAME = "muse-spark-1.3-contributor"
_DEFAULT_CLAUDE_AGENT_SDK_VERSION = "0.2.156"
_DEFAULT_CLAUDE_CODE_VERSION = "2.1.278"


@dataclass(frozen=True)
class RuntimeConfig:
    """Operator-controlled settings for a single agent process."""

    github_token: str
    meta_api_key: str
    target_repo: str
    data_dir: Path
    clone_dir: Path
    poll_interval: int
    log_level: str
    model_timeout: int
    publish_timeout: int
    max_retries: int
    max_consecutive_errors: int
    agent_trust_project_settings: bool
    review_blocking_severities: frozenset[str]
    profile_extra_path: str = ""
    profile_path: str = ""
    model: str = _DEFAULT_MODEL_NAME
    claude_agent_sdk_version: str = _DEFAULT_CLAUDE_AGENT_SDK_VERSION
    claude_code_version: str = _DEFAULT_CLAUDE_CODE_VERSION
    max_turns: int = 60
    max_budget_usd: int = 5
    soft_threshold_percentage: float = 0.2


@dataclass(frozen=True)
class RepositoryProfile:
    """Repository-controlled commands and their bounded execution settings."""

    setup: tuple[str, ...]
    check: tuple[str, ...]
    base_branch: str
    timeout: int
    setup_timeout: int
    env: dict[str, str]


def load_runtime_config(environ: Mapping[str, str] | None = None) -> RuntimeConfig:
    """Load operator settings without reporting the values of credentials."""

    values = os.environ if environ is None else environ
    github_token = _required(values, "GITHUB_TOKEN")
    meta_api_key = _required(values, "META_API_KEY")
    target_repo = _required(values, "TARGET_REPO")
    if not _TARGET_REPOSITORY.fullmatch(target_repo):
        raise ConfigurationError("TARGET_REPO must use the owner/repo format")

    data_dir = _path(values.get("DATA_DIR", "~/.simple-coding-agent/"), "DATA_DIR")
    clone_dir = _path(values.get("CLONE_DIR", str(data_dir / "repo")), "CLONE_DIR")
    log_level = values.get("LOG_LEVEL", "INFO").upper()
    if log_level not in _LOG_LEVELS:
        raise ConfigurationError("LOG_LEVEL must be a standard Python logging level")

    return RuntimeConfig(
        github_token=github_token,
        meta_api_key=meta_api_key,
        target_repo=target_repo,
        data_dir=data_dir,
        clone_dir=clone_dir,
        poll_interval=_positive_integer(values, "POLL_INTERVAL", 60),
        log_level=log_level,
        model_timeout=_positive_integer(values, "MODEL_TIMEOUT", 3600),
        publish_timeout=_positive_integer(values, "PUBLISH_TIMEOUT", 120),
        max_retries=_positive_integer(values, "MAX_RETRIES", 3),
        max_consecutive_errors=_positive_integer(values, "MAX_CONSECUTIVE_ERRORS", 3),
        agent_trust_project_settings=_boolean(values, "AGENT_TRUST_PROJECT_SETTINGS", False),
        review_blocking_severities=_review_severities(values),
        profile_extra_path=values.get("PROFILE_EXTRA_PATH", ""),
        profile_path=values.get("PROFILE_PATH", ""),
        model=_non_empty(values, "MODEL_NAME", _DEFAULT_MODEL_NAME),
        claude_agent_sdk_version=_non_empty(
            values, "CLAUDE_AGENT_SDK_VERSION", _DEFAULT_CLAUDE_AGENT_SDK_VERSION
        ),
        claude_code_version=_non_empty(
            values, "CLAUDE_CODE_VERSION", _DEFAULT_CLAUDE_CODE_VERSION
        ),
        max_turns=_positive_integer(values, "MAX_TURNS", 60),
        max_budget_usd=_positive_integer(values, "MAX_BUDGET_USD", 5),
        soft_threshold_percentage=_fraction(values, "SOFT_THRESHOLD_PERCENTAGE", 0.2),
    )


def load_repository_profile(repository_dir: Path, profile_path: str = "") -> RepositoryProfile:
    """Load the required repository profile for lifecycle configuration.

    When ``profile_path`` is set, the profile is read from that location instead of
    the default in-repo path, allowing it to live alongside the agent's own config.
    """

    path = (
        Path(profile_path)
        if profile_path
        else repository_dir / "docs" / "agents" / "simple-coding-agent-profile.yml"
    )

    try:
        raw_profile = yaml.safe_load(path.read_text())
    except OSError as error:
        raise ProfileError("Repository profile could not be read") from error
    except yaml.YAMLError as error:
        raise ProfileError("Repository profile is not valid YAML") from error

    if not isinstance(raw_profile, dict):
        raise ProfileError("Repository profile must be a mapping")

    allowed_keys = {"setup", "check", "base_branch", "timeout", "setup_timeout", "env"}
    unknown_keys = raw_profile.keys() - allowed_keys
    if unknown_keys:
        raise ProfileError("Repository profile contains unsupported settings")

    return RepositoryProfile(
        setup=_commands(raw_profile, "setup"),
        check=_commands(raw_profile, "check"),
        base_branch=_profile_string(raw_profile, "base_branch", "main"),
        timeout=_profile_duration(raw_profile, "timeout", 300),
        setup_timeout=_profile_duration(raw_profile, "setup_timeout", 120),
        env=_profile_environment(raw_profile),
    )


def _required(values: Mapping[str, str], name: str) -> str:
    value = values.get(name, "")
    if not value.strip():
        raise ConfigurationError(f"{name} must be set")
    return value


def _non_empty(values: Mapping[str, str], name: str, default: str) -> str:
    value = values.get(name, default)
    if not value.strip():
        raise ConfigurationError(f"{name} must not be empty")
    return value


def _path(value: str, name: str) -> Path:
    if not value.strip():
        raise ConfigurationError(f"{name} must not be empty")
    return Path(value).expanduser()


def _positive_integer(values: Mapping[str, str], name: str, default: int) -> int:
    value = values.get(name, str(default))
    try:
        number = int(value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a positive integer") from error
    if number <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return number


def _fraction(values: Mapping[str, str], name: str, default: float) -> float:
    value = values.get(name, str(default))
    try:
        number = float(value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number between 0 and 1") from error
    if not 0.0 <= number < 1.0:
        raise ConfigurationError(f"{name} must be a number between 0 and 1")
    return number


def _boolean(values: Mapping[str, str], name: str, default: bool) -> bool:
    value = values.get(name, str(default).lower()).lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ConfigurationError(f"{name} must be true or false")


def _review_severities(values: Mapping[str, str]) -> frozenset[str]:
    value = values.get("REVIEW_BLOCKING_SEVERITIES", "must-fix")
    if not value:
        return frozenset()
    severity_names = [part.strip() for part in value.split(",")]
    if not all(severity_names):
        raise ConfigurationError(
            "REVIEW_BLOCKING_SEVERITIES must be empty or a comma-separated severity list"
        )
    severities = frozenset(severity_names)
    invalid = severities - {"must-fix", "suggestion"}
    if invalid:
        raise ConfigurationError("REVIEW_BLOCKING_SEVERITIES contains an unknown severity")
    return severities


def _commands(profile: dict[object, object], name: str) -> tuple[str, ...]:
    value = profile.get(name)
    if isinstance(value, str) and value.strip():
        return (value,)
    if isinstance(value, list) and value and all(
        isinstance(command, str) and command.strip() for command in value
    ):
        return tuple(value)
    raise ProfileError(f"Repository profile {name} must be a command or non-empty command list")


def _profile_string(profile: dict[object, object], name: str, default: str) -> str:
    value = profile.get(name, default)
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"Repository profile {name} must be a non-empty string")
    return value


def _profile_duration(profile: dict[object, object], name: str, default: int) -> int:
    value = profile.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProfileError(f"Repository profile {name} must be a positive integer")
    return value


def _profile_environment(profile: dict[object, object]) -> dict[str, str]:
    value = profile.get("env", {})
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(item, str) for name, item in value.items()
    ):
        raise ProfileError("Repository profile env must map strings to strings")
    return dict(value)
