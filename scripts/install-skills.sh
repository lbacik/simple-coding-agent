#!/usr/bin/env bash
# Install the pinned upstream skill bundle for the simple-coding-agent.
#
# Suitable for local developer use and for image builds.  Run as the user
# whose HOME directory should receive the skills (agent user in Docker, your
# own account locally).
#
# Also installs this repository's own project-owned skill (handoff), which
# has no upstream commit to pin: it is vendored under skills/ and installed
# by copying it into the same HOME layout the upstream skills use.
#
# Pinned versions:
#   agent-installer : 0.6.0
#   source repo     : https://github.com/mattpocock/skills
#   source commit   : c55ee46073ed923f86ce59a5eb3b6d895095d1b7
#   skills          : implement, tdd, code-review, codebase-design
#   project skills  : handoff

set -euo pipefail

INSTALLER_VERSION="0.6.0"
SOURCE_REPO="https://github.com/mattpocock/skills"
SKILLS_COMMIT="c55ee46073ed923f86ce59a5eb3b6d895095d1b7"
SKILLS=(implement tdd code-review codebase-design)
PROJECT_SKILLS=(handoff)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Local dev runs this script from within the repo checkout, where the
# project skills live at <repo>/skills. A container build instead copies
# just skills/ to a fixed path and sets PROJECT_SKILLS_DIR, since the
# script itself is copied alone, without the rest of the repository.
PROJECT_SKILLS_DIR="${PROJECT_SKILLS_DIR:-${REPO_ROOT}/skills}"

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
    ONLY_FLAGS+=(--only "skill:${skill}")
done

echo "[install-skills] Installing skills from ${SOURCE_REPO}@${SKILLS_COMMIT} ..."
agent-installer install \
    "${SOURCE_REPO}" \
    --ref "${SKILLS_COMMIT}" \
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

# ---------------------------------------------------------------------------
# Install and verify the project-owned skill bundle (no agent-installer,
# no upstream commit: copied from this repository's own skills/ directory)
# ---------------------------------------------------------------------------
echo "[install-skills] Installing project-owned skills from ${PROJECT_SKILLS_DIR} ..."
mkdir -p "${HOME}/.agents/skills" "${HOME}/.claude/skills"
for skill in "${PROJECT_SKILLS[@]}"; do
    source_dir="${PROJECT_SKILLS_DIR}/${skill}"
    if [[ ! -d "${source_dir}" ]]; then
        echo "[install-skills] ERROR: project skill source missing: ${source_dir}" >&2
        exit 1
    fi

    skill_dir="${HOME}/.agents/skills/${skill}"
    link_path="${HOME}/.claude/skills/${skill}"
    rm -rf "${skill_dir}"
    cp -R "${source_dir}" "${skill_dir}"
    rm -f "${link_path}"
    ln -s "${skill_dir}" "${link_path}"

    if [[ ! -d "${skill_dir}" ]]; then
        echo "[install-skills] ERROR: project skill directory missing: ${skill_dir}" >&2
        exit 1
    fi
    if [[ ! -L "${link_path}" ]]; then
        echo "[install-skills] ERROR: HOME symlink missing: ${link_path}" >&2
        exit 1
    fi
done

echo "[install-skills] All project-owned skills installed and verified."
