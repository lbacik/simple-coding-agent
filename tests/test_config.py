from pathlib import Path

import pytest

from simple_coding_agent.config import (
    ConfigurationError,
    ProfileError,
    load_repository_profile,
    load_runtime_config,
)


def test_loads_required_operator_settings_and_defaults(tmp_path: Path) -> None:
    config = load_runtime_config(
        {
            "GITHUB_TOKEN": "github-secret",
            "META_API_KEY": "meta-secret",
            "TARGET_REPO": "octo/example",
            "DATA_DIR": str(tmp_path),
        }
    )

    assert config.target_repo == "octo/example"
    assert config.data_dir == tmp_path
    assert config.clone_dir == tmp_path / "repo"
    assert config.poll_interval == 60
    assert config.log_level == "INFO"
    assert config.model_timeout == 3600
    assert config.publish_timeout == 120
    assert config.max_retries == 3
    assert config.max_consecutive_errors == 3
    assert config.agent_trust_project_settings is False
    assert config.model == "muse-spark-1.3-contributor"
    assert config.max_turns == 60
    assert config.max_budget_usd == 5
    assert config.soft_threshold_percentage == 0.2
    assert config.claude_agent_sdk_version == "0.2.156"
    assert config.claude_code_version == "2.1.278"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("GITHUB_TOKEN", ""),
        ("META_API_KEY", ""),
        ("TARGET_REPO", "owner"),
        ("TARGET_REPO", "owner/repo/extra"),
        ("POLL_INTERVAL", "zero"),
        ("MODEL_TIMEOUT", "0"),
        ("AGENT_TRUST_PROJECT_SETTINGS", "sometimes"),
        ("REVIEW_BLOCKING_SEVERITIES", " "),
        ("REVIEW_BLOCKING_SEVERITIES", ","),
        ("MAX_TURNS", "0"),
        ("MAX_TURNS", "many"),
        ("MAX_BUDGET_USD", "-1"),
        ("SOFT_THRESHOLD_PERCENTAGE", "1"),
        ("SOFT_THRESHOLD_PERCENTAGE", "-0.1"),
        ("SOFT_THRESHOLD_PERCENTAGE", "not-a-number"),
    ],
)
def test_rejects_invalid_operator_settings_without_leaking_secrets(
    tmp_path: Path, name: str, value: str
) -> None:
    environment = {
        "GITHUB_TOKEN": "github-secret",
        "META_API_KEY": "meta-secret",
        "TARGET_REPO": "octo/example",
        "DATA_DIR": str(tmp_path),
    }
    environment[name] = value

    with pytest.raises(ConfigurationError) as error:
        load_runtime_config(environment)

    assert "github-secret" not in str(error.value)
    assert "meta-secret" not in str(error.value)


def test_applies_operator_overrides_including_the_test_only_review_gate(
    tmp_path: Path,
) -> None:
    config = load_runtime_config(
        {
            "GITHUB_TOKEN": "github-secret",
            "META_API_KEY": "meta-secret",
            "TARGET_REPO": "octo/example",
            "DATA_DIR": str(tmp_path / "data"),
            "CLONE_DIR": str(tmp_path / "clone"),
            "POLL_INTERVAL": "15",
            "LOG_LEVEL": "debug",
            "MODEL_TIMEOUT": "120",
            "PUBLISH_TIMEOUT": "45",
            "MAX_RETRIES": "4",
            "MAX_CONSECUTIVE_ERRORS": "2",
            "AGENT_TRUST_PROJECT_SETTINGS": "true",
            "REVIEW_BLOCKING_SEVERITIES": "",
            "MODEL_NAME": "muse-spark-2.0",
            "CLAUDE_AGENT_SDK_VERSION": "0.3.0",
            "CLAUDE_CODE_VERSION": "3.0.0",
            "MAX_TURNS": "90",
            "MAX_BUDGET_USD": "10",
            "SOFT_THRESHOLD_PERCENTAGE": "0.3",
        }
    )

    assert config.clone_dir == tmp_path / "clone"
    assert config.max_turns == 90
    assert config.max_budget_usd == 10
    assert config.soft_threshold_percentage == 0.3
    assert config.poll_interval == 15
    assert config.log_level == "DEBUG"
    assert config.model_timeout == 120
    assert config.publish_timeout == 45
    assert config.max_retries == 4
    assert config.max_consecutive_errors == 2
    assert config.agent_trust_project_settings is True
    assert config.review_blocking_severities == frozenset()
    assert config.model == "muse-spark-2.0"
    assert config.claude_agent_sdk_version == "0.3.0"
    assert config.claude_code_version == "3.0.0"


def test_profile_path_defaults_to_empty_string(tmp_path: Path) -> None:
    config = load_runtime_config(
        {
            "GITHUB_TOKEN": "github-secret",
            "META_API_KEY": "meta-secret",
            "TARGET_REPO": "octo/example",
            "DATA_DIR": str(tmp_path),
        }
    )

    assert config.profile_path == ""


def test_profile_path_is_read_from_the_environment(tmp_path: Path) -> None:
    config = load_runtime_config(
        {
            "GITHUB_TOKEN": "github-secret",
            "META_API_KEY": "meta-secret",
            "TARGET_REPO": "octo/example",
            "DATA_DIR": str(tmp_path),
            "PROFILE_PATH": str(tmp_path / "profile.yml"),
        }
    )

    assert config.profile_path == str(tmp_path / "profile.yml")


def test_loads_repository_profile_with_defaults_and_environment(tmp_path: Path) -> None:
    profile_path = tmp_path / "docs" / "agents" / "simple-coding-agent-profile.yml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(
        "setup:\n  - pip install -e '.[dev]'\ncheck: pytest\nenv:\n  APP_ENV: test\n"
    )

    profile = load_repository_profile(tmp_path)

    assert profile.setup == ("pip install -e '.[dev]'",)
    assert profile.check == ("pytest",)
    assert profile.base_branch == "main"
    assert profile.timeout == 300
    assert profile.setup_timeout == 120
    assert profile.env == {"APP_ENV": "test"}


@pytest.mark.parametrize(
    ("setting", "value", "attribute", "expected"),
    [
        ("base_branch", "release", "base_branch", "release"),
        ("timeout", "90", "timeout", 90),
        ("setup_timeout", "30", "setup_timeout", 30),
    ],
)
def test_applies_repository_profile_overrides(
    tmp_path: Path, setting: str, value: str, attribute: str, expected: object
) -> None:
    profile_path = tmp_path / "docs" / "agents" / "simple-coding-agent-profile.yml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(f"setup: pytest\ncheck: pytest\n{setting}: {value}\n")

    profile = load_repository_profile(tmp_path)

    assert getattr(profile, attribute) == expected


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "setup: []\ncheck: pytest\n",
        "setup: [pytest, 5]\ncheck: pytest\n",
        "setup: pytest\ncheck: {}\n",
        "setup: pytest\ncheck: pytest\ntimeout: 0\n",
        "setup: pytest\ncheck: pytest\nsetup_timeout: slow\n",
        "setup: pytest\ncheck: pytest\nenv: [APP_ENV]\n",
    ],
)
def test_rejects_invalid_repository_profile_as_infrastructure_error(
    tmp_path: Path, contents: str
) -> None:
    profile_path = tmp_path / "docs" / "agents" / "simple-coding-agent-profile.yml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(contents)

    with pytest.raises(ProfileError) as error:
        load_repository_profile(tmp_path)

    assert error.value.outcome == "infrastructure_error"


def test_missing_required_repository_profile_is_an_infrastructure_error(tmp_path: Path) -> None:
    with pytest.raises(ProfileError) as error:
        load_repository_profile(tmp_path)

    assert error.value.outcome == "infrastructure_error"


def test_loads_repository_profile_from_an_override_path(tmp_path: Path) -> None:
    override_path = tmp_path / "colocated" / "profile.yml"
    override_path.parent.mkdir(parents=True)
    override_path.write_text("setup: pytest\ncheck: pytest\n")

    profile = load_repository_profile(tmp_path, str(override_path))

    assert profile.setup == ("pytest",)
    assert profile.check == ("pytest",)


def test_override_path_is_used_even_without_a_default_in_repo_profile(tmp_path: Path) -> None:
    repository_dir = tmp_path / "repo"
    repository_dir.mkdir()
    override_path = tmp_path / "profile.yml"
    override_path.write_text("setup: pytest\ncheck: pytest\n")

    profile = load_repository_profile(repository_dir, str(override_path))

    assert profile.setup == ("pytest",)


def test_missing_override_path_is_an_infrastructure_error(tmp_path: Path) -> None:
    with pytest.raises(ProfileError) as error:
        load_repository_profile(tmp_path, str(tmp_path / "does-not-exist.yml"))

    assert error.value.outcome == "infrastructure_error"


def test_empty_override_path_falls_back_to_the_default_in_repo_location(tmp_path: Path) -> None:
    profile_path = tmp_path / "docs" / "agents" / "simple-coding-agent-profile.yml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("setup: pytest\ncheck: pytest\n")

    profile = load_repository_profile(tmp_path, "")

    assert profile.setup == ("pytest",)
