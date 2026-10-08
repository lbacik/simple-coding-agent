import re
from pathlib import Path

import pytest

from simple_coding_agent.config import (
    ConfigurationError,
    ProfileError,
    effective_soft_threshold_tokens,
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
    assert config.max_budget_usd == 20
    assert config.max_budget_tokens == 4_000_000
    assert config.soft_threshold_percentage == 0.2
    assert config.soft_threshold_tokens == 3_200_000
    assert config.handoff_reserve_turns == 6


def test_default_runtime_versions_match_dockerfile_pins(tmp_path: Path) -> None:
    # The provenance check compares these defaults with the installed pair, so
    # a container built without overrides only starts if they match the image.
    dockerfile = (Path(__file__).parent.parent / "Dockerfile").read_text()
    sdk_pin = re.search(r'"claude-agent-sdk==([^"]+)"', dockerfile)
    cli_pin = re.search(r"npm install --global @anthropic-ai/claude-code@(\S+)", dockerfile)
    assert sdk_pin and cli_pin, "Dockerfile must pin both the SDK and the CLI"

    config = load_runtime_config(_operator_env(tmp_path))

    assert config.claude_agent_sdk_version == sdk_pin.group(1)
    assert config.claude_code_version == cli_pin.group(1)


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
        ("MAX_BUDGET_TOKENS", "0"),
        ("MAX_BUDGET_TOKENS", "-1"),
        ("MAX_BUDGET_TOKENS", "many"),
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
            "MAX_BUDGET_TOKENS": "1000000",
            "SOFT_THRESHOLD_PERCENTAGE": "0.3",
        }
    )

    assert config.clone_dir == tmp_path / "clone"
    assert config.max_turns == 90
    assert config.max_budget_usd == 10
    assert config.max_budget_tokens == 1_000_000
    assert config.soft_threshold_percentage == 0.3
    assert config.soft_threshold_tokens == 700_000
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


def test_model_backend_defaults_to_meta(tmp_path: Path) -> None:
    config = load_runtime_config(
        {
            "GITHUB_TOKEN": "github-secret",
            "META_API_KEY": "meta-secret",
            "TARGET_REPO": "octo/example",
            "DATA_DIR": str(tmp_path),
        }
    )

    assert config.model_base_url == "https://api.meta.ai"
    assert config.model_api_key == "meta-secret"
    assert config.model_auth_mode == "auth_token"
    assert config.model_stream_idle_timeout_ms == 60000


def test_meta_api_key_only_keeps_working_without_new_variables(tmp_path: Path) -> None:
    config = load_runtime_config(_operator_env(tmp_path))

    assert config.meta_api_key == "meta-secret"
    assert config.model_api_key == "meta-secret"


def test_model_api_key_takes_precedence_over_meta_api_key(tmp_path: Path) -> None:
    config = load_runtime_config(
        _operator_env(tmp_path, MODEL_API_KEY="model-secret", META_API_KEY="meta-secret")
    )

    assert config.model_api_key == "model-secret"
    assert config.meta_api_key == "meta-secret"


def test_model_api_key_alone_satisfies_the_credential_requirement(tmp_path: Path) -> None:
    environment = _operator_env(tmp_path, MODEL_API_KEY="model-secret")
    del environment["META_API_KEY"]

    config = load_runtime_config(environment)

    assert config.model_api_key == "model-secret"


def test_empty_model_base_url_means_direct_anthropic_api(tmp_path: Path) -> None:
    config = load_runtime_config(_operator_env(tmp_path, MODEL_BASE_URL=""))

    assert config.model_base_url == ""


def test_rejects_unknown_model_auth_mode_without_leaking_secrets(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError) as error:
        load_runtime_config(_operator_env(tmp_path, MODEL_AUTH_MODE="bearer"))

    assert "github-secret" not in str(error.value)
    assert "meta-secret" not in str(error.value)


def test_rejects_missing_credential_without_leaking_secrets(tmp_path: Path) -> None:
    environment = _operator_env(tmp_path, MODEL_API_KEY="")
    del environment["META_API_KEY"]

    with pytest.raises(ConfigurationError) as error:
        load_runtime_config(environment)

    assert "github-secret" not in str(error.value)


def test_rejects_blank_credentials_without_leaking_secrets(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError) as error:
        load_runtime_config(
            _operator_env(tmp_path, MODEL_API_KEY="  ", META_API_KEY="  ")
        )

    assert "github-secret" not in str(error.value)


def test_credential_redactions_cover_both_api_key_settings(tmp_path: Path) -> None:
    config = load_runtime_config(
        _operator_env(tmp_path, MODEL_API_KEY="model-secret", META_API_KEY="meta-secret")
    )

    assert config.credential_redactions == (
        "github-secret",
        "meta-secret",
        "model-secret",
    )


def test_applies_model_backend_overrides(tmp_path: Path) -> None:
    config = load_runtime_config(
        _operator_env(
            tmp_path,
            MODEL_BASE_URL="",
            MODEL_AUTH_MODE="api_key",
            MODEL_API_KEY="model-secret",
            MODEL_NAME="claude-sonnet-5-5",
            MODEL_STREAM_IDLE_TIMEOUT_MS="120000",
        )
    )

    assert config.model_base_url == ""
    assert config.model_auth_mode == "api_key"
    assert config.model_api_key == "model-secret"
    assert config.model == "claude-sonnet-5-5"
    assert config.model_stream_idle_timeout_ms == 120000


def _operator_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    return {
        "GITHUB_TOKEN": "github-secret",
        "META_API_KEY": "meta-secret",
        "TARGET_REPO": "octo/example",
        "DATA_DIR": str(tmp_path),
        **overrides,
    }


def test_warns_when_usd_backstop_is_tighter_than_token_budget(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING", logger="simple_coding_agent.config"):
        load_runtime_config(_operator_env(tmp_path, MAX_BUDGET_USD="19"))

    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert any("MAX_BUDGET_USD=19" in record.message for record in warnings)


def test_no_warning_when_usd_backstop_covers_token_budget(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING", logger="simple_coding_agent.config"):
        load_runtime_config(_operator_env(tmp_path))

    assert [record for record in caplog.records if record.levelname == "WARNING"] == []


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


def test_handoff_reserve_turns_accepts_the_full_range(tmp_path: Path) -> None:
    for value in ("1", "6", "20"):
        config = load_runtime_config(_operator_env(tmp_path, HANDOFF_RESERVE_TURNS=value))

        assert config.handoff_reserve_turns == int(value)


@pytest.mark.parametrize("value", ["0", "21", "-1", "many", "6.5", ""])
def test_handoff_reserve_turns_rejects_out_of_range_or_non_integer(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(ConfigurationError) as error:
        load_runtime_config(_operator_env(tmp_path, HANDOFF_RESERVE_TURNS=value))

    assert "HANDOFF_RESERVE_TURNS" in str(error.value)
    assert "github-secret" not in str(error.value)
    assert "meta-secret" not in str(error.value)


def test_effective_soft_threshold_uses_the_reserve_rule_for_a_large_context() -> None:
    threshold, rule, reserve_tokens = effective_soft_threshold_tokens(
        4_000_000, 0.2, 6, 150_000
    )

    assert threshold == 2_950_000
    assert rule == "reserve"
    assert reserve_tokens == 7 * 150_000


def test_effective_soft_threshold_uses_the_percentage_rule_for_a_small_context() -> None:
    threshold, rule, reserve_tokens = effective_soft_threshold_tokens(
        4_000_000, 0.2, 6, 50_000
    )

    assert threshold == 3_200_000
    assert rule == "percentage"
    assert reserve_tokens == 7 * 50_000


def test_effective_soft_threshold_fires_immediately_when_the_reserve_exceeds_the_budget() -> (
    None
):
    threshold, rule, _ = effective_soft_threshold_tokens(4_000_000, 0.2, 6, 600_000)

    assert rule == "reserve"
    assert threshold <= 0


def test_runtime_config_effective_soft_threshold_matches_the_helper(tmp_path: Path) -> None:
    config = load_runtime_config(_operator_env(tmp_path))

    assert config.effective_soft_threshold(150_000) == (2_950_000, "reserve", 1_050_000)
    assert config.effective_soft_threshold(50_000) == (3_200_000, "percentage", 350_000)
