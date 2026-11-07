# syntax=docker/dockerfile:1
# Unattended single-repository coding agent.
#
# Pinned runtime: claude-agent-sdk (pin in pyproject.toml) with its bundled Claude Code CLI
# Dedicated agent user: UID 1000, HOME=/home/agent
# Persistent data root: /data  (mount as a named volume)

FROM python:3.12-slim AS base

# Contract discovery lives on the managed service (Compose `labels:`), not in
# this image: the same image is shared with one-off containers, and only the
# managed service is a Managed Instance. Contract images MUST NOT carry the
# legacy `io.github.lbacik.simple-coding-agent.role` label, so a consumer
# without contract support never mistakes a contract instance for a legacy one
# (operator contract, deployment and discovery). A dashboard lists managed
# instances by the contract labels, e.g.:
#   docker ps --filter "label=io.github.lbacik.agent-operator.contract=1"
# Visible with `docker inspect` on the managed container.

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    gnupg \
    php-cli \
    procps \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Node.js >=20 (installer-compatible)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Dedicated agent user (UID 1000)
# ---------------------------------------------------------------------------
RUN useradd --uid 1000 --create-home --shell /bin/bash agent

# ---------------------------------------------------------------------------
# Application code (installed as the agent user)
# ---------------------------------------------------------------------------
WORKDIR /app
COPY pyproject.toml ./
COPY simple_coding_agent/ ./simple_coding_agent/

# Python runtime: dependencies, including the pinned claude-agent-sdk, come
# from pyproject.toml.
RUN pip install --no-cache-dir -e .

# The control entry point is installed by `pip install -e .` (via the
# `agentctl` project script) as root, so it lands at its absolute path.
# Operator instructions use that path: the image prepends the target
# checkout's virtual environment to PATH, where a bare `agentctl` could
# resolve to the target package instead of this agent's installed package.
RUN test -x /usr/local/bin/agentctl

# Pinned skill installer (installed globally as root: the agent user cannot
# write the global npm prefix). install-skills.sh rejects any other version,
# so keep this in step with its INSTALLER_VERSION.
RUN npm install --global agent-installer@0.7.2

# Managed settings (root-owned): Meta model pricing for the CLI cost ledger.
# The pinned CLI reads managed settings from /etc/claude-code/ on Linux and
# prices muse-spark-1.3-contributor with its default-model rates ($5 input /
# $25 output / $0.50 cache read per MTok, 50-250x Meta's rates) unless overridden
# here. Row keys (input, output, cacheRead, cacheWrite) are USD-per-million-
# token rates per the pinned CLI's modelPricing schema; see README.md.
# Anthropic models are intentionally absent, so they keep the CLI price table.
# The file is installed as root with a non-writable mode: the agent user can
# read it but cannot change the rates its own budget is measured against.
COPY --chmod=644 docker/managed-settings.json /etc/claude-code/managed-settings.json

# ---------------------------------------------------------------------------
# Skill bundle (run as agent user so paths land in /home/agent)
# ---------------------------------------------------------------------------
COPY scripts/install-skills.sh /usr/local/bin/install-skills.sh
RUN chmod +x /usr/local/bin/install-skills.sh
COPY skills/ /opt/simple-coding-agent-skills/
ENV PROJECT_SKILLS_DIR=/opt/simple-coding-agent-skills

USER agent
RUN /usr/local/bin/install-skills.sh

# Persistent data volume: /data is owned by the agent user
USER root
RUN mkdir -p /data && chown agent:agent /data
VOLUME /data

# Private control-socket runtime directory: container-local state for
# /run/simple-coding-agent/control.sock, owned by the agent user with mode
# 0700. It is never bind-mounted or shared between instances. The live agent
# recreates it on startup (covering tmpfs-mounted /run) and binds the socket
# with mode 0600.
RUN mkdir -p /run/simple-coding-agent \
    && chown agent:agent /run/simple-coding-agent \
    && chmod 0700 /run/simple-coding-agent

USER agent
ENV HOME=/home/agent
ENV DATA_DIR=/data

ENTRYPOINT ["python", "-m", "simple_coding_agent"]

# ---------------------------------------------------------------------------
# Conformance harness image (operator contract §15.1; never production)
# ---------------------------------------------------------------------------
# The external producer suite drives this target through the Conformance
# Harness API: `docker run <harness-image> <verb> '<request-json>'` writes
# one JSON reply to stdout. It wires the real lifecycle to the shared test
# fakes (`tests/fakes.py`, reused — never copies) via `conformance_harness/`,
# whose daemon the host-side CLI supervises itself, so no daemon
# ENTRYPOINT is needed: the harness CLI is the container entry point and
# `SCA_HARNESS_STATE_DIR` (default `~/.cache/sca-conformance-harness`)
# selects the state volume. Build with
# `docker build --target harness .`.
FROM base AS harness
USER root
COPY conformance_harness/ ./conformance_harness/
COPY tests/fakes.py ./tests/fakes.py
USER agent
ENTRYPOINT ["python", "-m", "conformance_harness"]

# ---------------------------------------------------------------------------
# Production image (default build target; carries no test fakes)
# ---------------------------------------------------------------------------
# `docker build .` with no `--target` builds this last stage, so the default
# output is exactly the historical single-stage image: only
# `simple_coding_agent/` is present and neither `conformance_harness/` nor
# `tests/` exists (the harness stage above is not an ancestor of this one).
FROM base AS production
ENTRYPOINT ["python", "-m", "simple_coding_agent"]
