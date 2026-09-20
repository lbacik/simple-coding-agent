#!/usr/bin/env bash
# Install the pinned upstream skill bundle for the simple-coding-agent.
#
# Suitable for local developer use and for image builds.  Run as the user
# whose HOME directory should receive the skills (agent user in Docker, your
# own account locally).
#
# Pinned versions:
#   agent-installer : 0.6.0
#   source commit   : c55ee46073ed923f86ce59a5eb3b6d895095d1b7
#   skills          : implement, tdd, code-review, codebase-design

set -euo pipefail

INSTALLER_VERSION="0.6.0"
SKILLS_COMMIT="c55ee46073ed923f86ce59a5eb3b6d895095d1b7"
SKILLS=(implement tdd code-review codebase-design)

# ---------------------------------------------------------------------------
# Locate or install agent-installer
# ---------------------------------------------------------------------------
if ! command -v agent-installer &>/dev/null; then
    echo "[install-skills] Installing agent-installer@${INSTALLER_VERSION} ..."
    npm install --global "agent-installer@${INSTALLER_VERSION}"
fi

# Verify the installed version matches the pin.
INSTALLED_VERSION="$(agent-installer --version 2>/dev/null | awk '{print $NF}' | sed 's/^v//')"
if [[ "${INSTALLED_VERSION}" != "${INSTALLER_VERSION}" ]]; then
    echo "[install-skills] ERROR: agent-installer ${INSTALLED_VERSION} is installed but ${INSTALLER_VERSION} is required." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Install each required skill from the pinned commit
# ---------------------------------------------------------------------------
ONLY_FLAGS=()
for skill in "${SKILLS[@]}"; do
    ONLY_FLAGS+=(--only "${skill}")
done

echo "[install-skills] Installing skills from commit ${SKILLS_COMMIT} ..."
agent-installer install \
    --source "${SKILLS_COMMIT}" \
    "${ONLY_FLAGS[@]}"

# ---------------------------------------------------------------------------
# Verify installation manifest and symlinks
# ---------------------------------------------------------------------------
echo "[install-skills] Verifying installed skills ..."
LISTING="$(agent-installer list --json)"
for skill in "${SKILLS[@]}"; do
    resolved_commit="$(echo "${LISTING}" | jq -r --arg id "skill:${skill}" \
        '.artifacts[] | select(.id == $id) | .resolvedCommit')"
    if [[ "${resolved_commit}" != "${SKILLS_COMMIT}" ]]; then
        echo "[install-skills] ERROR: skill '${skill}' commit mismatch: ${resolved_commit}" >&2
        exit 1
    fi

    skill_dir="${HOME}/.agents/skills/${skill}"
    link_path="${HOME}/.claude/skills/${skill}"
    if [[ ! -d "${skill_dir}" ]]; then
        echo "[install-skills] ERROR: skill directory missing: ${skill_dir}" >&2
        exit 1
    fi
    if [[ ! -L "${link_path}" ]]; then
        echo "[install-skills] ERROR: HOME symlink missing: ${link_path}" >&2
        exit 1
    fi
done

echo "[install-skills] All skills installed and verified."
