# syntax=docker/dockerfile:1
# Unattended single-repository coding agent.
#
# Runtime pair: claude-agent-sdk==0.2.156 + @anthropic-ai/claude-code@2.1.276
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
    "claude-agent-sdk==0.2.156" \
    "PyYAML>=6.0,<7" \
    && pip install --no-cache-dir -e .

# Pinned CLI runtime (installed globally so agent user can use it)
COPY package.json ./
RUN npm install --global @anthropic-ai/claude-code@2.1.276

# ---------------------------------------------------------------------------
# Skill bundle (run as agent user so paths land in /home/agent)
# ---------------------------------------------------------------------------
COPY scripts/install-skills.sh /usr/local/bin/install-skills.sh
RUN chmod +x /usr/local/bin/install-skills.sh

USER agent
RUN /usr/local/bin/install-skills.sh

# Persistent data volume: /data is owned by the agent user
USER root
RUN mkdir -p /data && chown agent:agent /data
VOLUME /data

USER agent
ENV HOME=/home/agent
ENV DATA_DIR=/data

ENTRYPOINT ["python", "-m", "simple_coding_agent"]
