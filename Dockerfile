# syntax=docker/dockerfile:1
# Unattended single-repository coding agent.
#
# Runtime pair: claude-agent-sdk==0.2.163 + @anthropic-ai/claude-code@2.1.286
# Dedicated agent user: UID 1000, HOME=/home/agent
# Persistent data root: /data  (mount as a named volume)

FROM python:3.12-slim

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    gnupg \
    php-cli \
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

# Pinned Python runtime
RUN pip install --no-cache-dir \
    "claude-agent-sdk==0.2.163" \
    "PyYAML>=6.0,<7" \
    && pip install --no-cache-dir -e .

# The control entry point is installed by `pip install -e .` (via the
# `agentctl` project script) as root, so it lands at its absolute path.
# Operator instructions use that path: the image prepends the target
# checkout's virtual environment to PATH, where a bare `agentctl` could
# resolve to the target package instead of this agent's installed package.
RUN test -x /usr/local/bin/agentctl

# Pinned CLI runtime (installed globally so agent user can use it)
COPY package.json ./
RUN npm install --global @anthropic-ai/claude-code@2.1.286

# Pinned skill installer (installed globally as root: the agent user cannot
# write the global npm prefix). install-skills.sh rejects any other version,
# so keep this in step with its INSTALLER_VERSION.
RUN npm install --global agent-installer@0.7.2

# Managed settings (root-owned): Meta model pricing for the CLI cost ledger.
# The pinned CLI reads managed settings from /etc/claude-code/ on Linux and
# prices muse-spark-1.3-contributor with its default-model rates ($5 input /
# $25 output / $0.50 cache read per MTok, ~4x Meta's rates) unless overridden
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
