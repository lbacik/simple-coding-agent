"""Offline acceptance tests for packaging and deployment artefacts.

These tests are fully deterministic and make no network calls.  They
validate that the Dockerfile, docker-compose.yml, scripts/install-skills.sh,
and documentation files satisfy the acceptance criteria for issue #27.

Opt-in smoke tests against a live Docker daemon are gated behind
DOCKER_SMOKE=1 and are kept in a separate section at the end of this module.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------


def _dockerfile() -> str:
    return (ROOT / "Dockerfile").read_text()


def test_dockerfile_exists() -> None:
    assert (ROOT / "Dockerfile").is_file(), "Dockerfile must exist at the repository root"


def test_dockerfile_uses_dedicated_agent_user_uid_1000() -> None:
    content = _dockerfile()
    assert re.search(r"useradd\s+--uid\s+1000", content), (
        "Dockerfile must create the agent user with UID 1000"
    )


def test_dockerfile_sets_agent_home() -> None:
    content = _dockerfile()
    assert "HOME=/home/agent" in content, "Dockerfile must set HOME=/home/agent"


def test_dockerfile_sets_data_dir_to_slash_data() -> None:
    content = _dockerfile()
    assert "DATA_DIR=/data" in content, "Dockerfile must set DATA_DIR=/data"


def test_dockerfile_declares_volume_slash_data() -> None:
    content = _dockerfile()
    assert "VOLUME /data" in content, "Dockerfile must declare a /data volume"


def test_dockerfile_does_not_mount_docker_socket() -> None:
    content = _dockerfile()
    assert "/var/run/docker.sock" not in content, (
        "Dockerfile must not reference the Docker socket"
    )


def test_dockerfile_installs_pinned_python_sdk() -> None:
    content = _dockerfile()
    assert "claude-agent-sdk==0.2.156" in content, (
        "Dockerfile must install claude-agent-sdk==0.2.156"
    )


def test_dockerfile_installs_pinned_claude_code_cli() -> None:
    content = _dockerfile()
    assert "@anthropic-ai/claude-code@2.1.276" in content, (
        "Dockerfile must install @anthropic-ai/claude-code@2.1.276"
    )


def test_dockerfile_includes_node_20_or_higher() -> None:
    content = _dockerfile()
    # Accepts nodesource setup_20.x or higher
    assert re.search(r"nodesource\.com/setup_(2[0-9]|[3-9][0-9])\.x", content), (
        "Dockerfile must install Node.js >=20 via NodeSource"
    )


def test_dockerfile_includes_git() -> None:
    assert "git" in _dockerfile(), "Dockerfile must install git"


def test_dockerfile_includes_gh_cli() -> None:
    content = _dockerfile()
    assert "cli.github.com" in content or " gh " in content or "gh\n" in content or "gh \\\n" in content, (
        "Dockerfile must install the GitHub CLI (gh)"
    )


def test_dockerfile_includes_php() -> None:
    assert "php" in _dockerfile(), "Dockerfile must install PHP"


def test_dockerfile_runs_install_skills() -> None:
    content = _dockerfile()
    assert "install-skills.sh" in content, (
        "Dockerfile must invoke install-skills.sh to install the skill bundle"
    )


def test_dockerfile_entrypoint_runs_the_agent() -> None:
    content = _dockerfile()
    assert "python" in content and "simple_coding_agent" in content, (
        "Dockerfile ENTRYPOINT must invoke the agent package"
    )


def test_dockerfile_copies_the_project_owned_skills_directory() -> None:
    content = _dockerfile()
    assert "COPY skills/" in content, (
        "Dockerfile must copy the repository's project-owned skills/ directory"
    )
    assert "PROJECT_SKILLS_DIR" in content, (
        "Dockerfile must point install-skills.sh at the copied project skills"
    )


# ---------------------------------------------------------------------------
# docker-compose.yml
# ---------------------------------------------------------------------------


def _compose() -> str:
    return (ROOT / "docker-compose.yml").read_text()


def test_docker_compose_file_exists() -> None:
    assert (ROOT / "docker-compose.yml").is_file(), "docker-compose.yml must exist"


def test_compose_loads_env_file() -> None:
    content = _compose()
    assert "env_file" in content and ".env" in content, (
        "docker-compose.yml must reference .env via env_file"
    )


def test_compose_defines_persistent_named_volume() -> None:
    content = _compose()
    # Must declare at least one named volume
    assert re.search(r"^volumes:", content, re.MULTILINE), (
        "docker-compose.yml must declare a top-level volumes section"
    )


def test_compose_mounts_persistent_volume_to_data() -> None:
    content = _compose()
    assert "/data" in content, (
        "docker-compose.yml must mount a volume at /data"
    )


def test_compose_does_not_mount_docker_socket() -> None:
    content = _compose()
    assert "/var/run/docker.sock" not in content, (
        "docker-compose.yml must not mount the Docker socket"
    )


def test_compose_service_notes_single_instance_scope() -> None:
    content = _compose()
    # Either explicit scale:1 or a comment noting sequential-worker scope
    assert "scale" not in content or "1" in content, (
        "Compose configuration must preserve the single-instance scope"
    )


# ---------------------------------------------------------------------------
# scripts/install-skills.sh
# ---------------------------------------------------------------------------


def _script() -> str:
    return (ROOT / "scripts" / "install-skills.sh").read_text()


def test_install_skills_script_exists() -> None:
    assert (ROOT / "scripts" / "install-skills.sh").is_file(), (
        "scripts/install-skills.sh must exist"
    )


def test_install_skills_is_executable() -> None:
    path = ROOT / "scripts" / "install-skills.sh"
    assert os.access(path, os.X_OK), "scripts/install-skills.sh must be executable"


def test_install_skills_pins_agent_installer_version() -> None:
    content = _script()
    assert "agent-installer@0.6.0" in content or 'INSTALLER_VERSION="0.6.0"' in content, (
        "install-skills.sh must pin agent-installer to 0.6.0"
    )


def test_install_skills_pins_source_commit() -> None:
    content = _script()
    assert "c55ee46073ed923f86ce59a5eb3b6d895095d1b7" in content, (
        "install-skills.sh must reference the canonical skills source commit"
    )


def test_install_skills_installs_all_four_required_skills() -> None:
    content = _script()
    for skill in ("implement", "tdd", "code-review", "codebase-design"):
        assert skill in content, f"install-skills.sh must select skill '{skill}'"


def test_install_skills_verifies_manifest_and_symlinks() -> None:
    content = _script()
    assert "agent-installer list" in content, (
        "install-skills.sh must verify installation via 'agent-installer list'"
    )
    assert ".claude/skills" in content or "link_path" in content or "symlink" in content.lower(), (
        "install-skills.sh must verify HOME symlinks"
    )


def test_install_skills_uses_strict_error_handling() -> None:
    content = _script()
    assert "set -euo pipefail" in content or "set -e" in content, (
        "install-skills.sh must enable strict error handling"
    )


def test_install_skills_installs_the_project_owned_handoff_skill() -> None:
    content = _script()
    assert "handoff" in content, "install-skills.sh must install the project-owned handoff skill"
    assert "PROJECT_SKILLS_DIR" in content, (
        "install-skills.sh must resolve project skills from a configurable directory"
    )


def test_project_owned_handoff_skill_exists_in_the_repository() -> None:
    skill_file = ROOT / "skills" / "handoff" / "SKILL.md"
    assert skill_file.is_file(), "skills/handoff/SKILL.md must exist"
    content = skill_file.read_text()
    assert "name: handoff" in content, "skills/handoff/SKILL.md must declare the handoff skill name"


# ---------------------------------------------------------------------------
# Operator onboarding documentation
# ---------------------------------------------------------------------------


def test_onboarding_doc_exists() -> None:
    assert (ROOT / "docs" / "agents" / "operator-onboarding.md").is_file(), (
        "docs/agents/operator-onboarding.md must exist"
    )


def _onboarding() -> str:
    return (ROOT / "docs" / "agents" / "operator-onboarding.md").read_text()


def test_onboarding_documents_pat_permissions() -> None:
    content = _onboarding()
    assert "contents" in content and "pull-requests" in content and "issues" in content, (
        "Onboarding docs must list all required PAT permissions"
    )


def test_onboarding_documents_ready_for_agent_trust_policy() -> None:
    content = _onboarding()
    assert "ready-for-agent" in content and "write access" in content, (
        "Onboarding docs must explain the ready-for-agent trust policy"
    )


def test_onboarding_documents_project_settings_risks() -> None:
    content = _onboarding()
    assert "AGENT_TRUST_PROJECT_SETTINGS" in content and "residual" in content, (
        "Onboarding docs must explain project settings risks"
    )


def test_onboarding_documents_log_retention() -> None:
    content = _onboarding()
    assert "DATA_DIR/logs" in content or "$DATA_DIR/logs" in content, (
        "Onboarding docs must document the log retention path"
    )


def test_onboarding_documents_spending_caps() -> None:
    content = _onboarding()
    assert "max_budget_usd" in content or "spending" in content, (
        "Onboarding docs must document provider spending caps"
    )


def test_onboarding_documents_error_guard_recovery() -> None:
    content = _onboarding()
    assert "consecutive_errors.json" in content and ("reset" in content or "recovery" in content), (
        "Onboarding docs must document persisted error-guard recovery"
    )


def test_onboarding_documents_required_tracker_context() -> None:
    content = _onboarding()
    assert "CONTEXT.md" in content and "issue-tracker.md" in content, (
        "Onboarding docs must list required tracker context files"
    )


# ---------------------------------------------------------------------------
# .gitignore: credentials must not be committed
# ---------------------------------------------------------------------------


def test_gitignore_excludes_env_file() -> None:
    gitignore_path = ROOT / ".gitignore"
    assert gitignore_path.is_file(), ".gitignore must exist"
    content = gitignore_path.read_text()
    assert ".env" in content, ".gitignore must exclude the .env file to prevent credential leaks"


# ---------------------------------------------------------------------------
# .env.example: template exists with required variables
# ---------------------------------------------------------------------------


def test_env_example_exists() -> None:
    assert (ROOT / ".env.example").is_file(), ".env.example must exist"


def test_env_example_documents_required_variables() -> None:
    content = (ROOT / ".env.example").read_text()
    for var in ("GITHUB_TOKEN", "META_API_KEY", "TARGET_REPO"):
        assert var in content, f".env.example must document {var}"


# ---------------------------------------------------------------------------
# README: local installation and Compose launch paths
# ---------------------------------------------------------------------------


def test_readme_documents_local_python_launch() -> None:
    content = (ROOT / "README.md").read_text()
    assert "python -m simple_coding_agent" in content, (
        "README must document the local python -m launch path"
    )


def test_readme_documents_compose_up_launch() -> None:
    content = (ROOT / "README.md").read_text()
    assert "docker compose up" in content or "docker-compose up" in content, (
        "README must document the docker compose up launch path"
    )


def test_readme_documents_compose_run_launch() -> None:
    content = (ROOT / "README.md").read_text()
    assert "docker compose run" in content or "docker-compose run" in content, (
        "README must document the docker compose run --rm launch path"
    )


# ---------------------------------------------------------------------------
# Opt-in Docker smoke tests (DOCKER_SMOKE=1)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("DOCKER_SMOKE") != "1",
    reason="Set DOCKER_SMOKE=1 to run live Docker build and smoke tests",
)
class TestDockerSmoke:
    """Live Docker build and runtime smoke tests.

    These tests require a running Docker daemon and will perform a full image
    build.  They are opt-in to avoid unattended provider calls.

    Report any evidence from these tests separately; they must NOT be counted
    as offline acceptance evidence.
    """

    def test_image_builds_successfully(self) -> None:
        result = subprocess.run(
            ["docker", "build", "-t", "simple-coding-agent:smoke-test", "."],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Docker build failed:\n{result.stderr}"

    def test_agent_user_is_uid_1000(self) -> None:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "simple-coding-agent:smoke-test",
                "id", "-u",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "1000", (
            f"Agent user must be UID 1000; got: {result.stdout.strip()}"
        )

    def test_python_sdk_version_matches_pin(self) -> None:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "simple-coding-agent:smoke-test",
                "python", "-c",
                "from importlib.metadata import version; print(version('claude-agent-sdk'))",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "0.2.156", (
            f"SDK version mismatch: {result.stdout.strip()}"
        )

    def test_claude_code_cli_version_matches_pin(self) -> None:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "simple-coding-agent:smoke-test",
                "claude", "--version",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        version = result.stdout.strip().split()[-1].lstrip("v")
        assert version == "2.1.276", f"CLI version mismatch: {version}"

    def test_skills_are_visible_in_agent_home(self) -> None:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "simple-coding-agent:smoke-test",
                "ls", "/home/agent/.agents/skills/",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        installed = result.stdout.split()
        for skill in ("implement", "tdd", "code-review", "codebase-design", "handoff"):
            assert skill in installed, f"Skill '{skill}' missing from /home/agent/.agents/skills/"

    def test_data_volume_is_writable_by_agent_user(self) -> None:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "simple-coding-agent:smoke-test",
                "sh", "-c", "touch /data/smoke-test && echo ok",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "ok" in result.stdout

    def test_volume_preserves_data_through_container_replacement(self, tmp_path: Path) -> None:
        import uuid
        volume_name = f"smoke-test-{uuid.uuid4().hex[:8]}"
        marker = "smoke-persistence-marker"
        try:
            # Write to volume
            r1 = subprocess.run(
                [
                    "docker", "run", "--rm", "-v", f"{volume_name}:/data",
                    "simple-coding-agent:smoke-test",
                    "sh", "-c", f"echo {marker} > /data/marker.txt",
                ],
                capture_output=True, text=True,
            )
            assert r1.returncode == 0, r1.stderr

            # Read back from a fresh container
            r2 = subprocess.run(
                [
                    "docker", "run", "--rm", "-v", f"{volume_name}:/data",
                    "simple-coding-agent:smoke-test",
                    "cat", "/data/marker.txt",
                ],
                capture_output=True, text=True,
            )
            assert r2.returncode == 0
            assert marker in r2.stdout
        finally:
            subprocess.run(["docker", "volume", "rm", volume_name], capture_output=True)

    def test_agentctl_is_installed_at_its_absolute_path(self) -> None:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "simple-coding-agent:smoke-test",
                "test", "-x", "/usr/local/bin/agentctl",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            "agentctl must be executable at /usr/local/bin/agentctl in every instance image"
        )

    def test_private_runtime_directory_exists_in_the_image(self) -> None:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "simple-coding-agent:smoke-test",
                "sh", "-c",
                "stat -c '%U %a' /run/simple-coding-agent",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "agent 700", (
            f"Runtime directory must be owned by agent with mode 0700; got: {result.stdout.strip()}"
        )
