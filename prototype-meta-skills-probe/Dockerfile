# PROTOTYPE, disposable image. Answers: does claude-agent-sdk==0.2.156 against
# Meta muse-spark-1.3, running the pinned mattpocock/skills bundle installed by
# agent-installer 0.6.0, actually discover /implement, edit a fixture, pass a
# test, resolve the code-review collision correctly, and produce a structured
# result? See ticket #8 on https://github.com/lbacik/simple-coding-agent/issues/1
#
# Not a production Dockerfile: no multi-stage slimming, no non-root polish
# beyond what the installer requires.

FROM node:22-bookworm-slim AS installer

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG AGENT_INSTALLER_VERSION=0.6.0
ARG SKILLS_REF=c55ee46073ed923f86ce59a5eb3b6d895095d1b7

RUN npm install -g agent-installer@${AGENT_INSTALLER_VERSION}

# Runtime user's HOME must match the SDK's setting_sources=["user"] discovery path.
ENV HOME=/home/agent
RUN useradd -m -d /home/agent agent
USER agent
WORKDIR /home/agent

RUN agent-installer install https://github.com/mattpocock/skills.git \
    --ref ${SKILLS_REF} \
    --only skill:implement --only skill:tdd \
    --only skill:code-review --only skill:codebase-design \
    --json | tee /home/agent/install-result.json

FROM python:3.13-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG CLAUDE_AGENT_SDK_VERSION=0.2.156
RUN pip install --no-cache-dir claude-agent-sdk==${CLAUDE_AGENT_SDK_VERSION} pytest==8.3.5

# claude-agent-sdk 0.2.156 does not vendor the CLI binary; it drives whatever
# `claude` it finds on PATH and records only which version it expects
# (_cli_version.py == 2.1.276). Install that exact CLI so the runtime
# actually matches the pinned candidate instead of silently drifting.
COPY --from=installer /usr/local/bin/node /usr/local/bin/node
COPY --from=installer /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && npm install -g @anthropic-ai/claude-code@2.1.276

ENV HOME=/home/agent
RUN useradd -m -d /home/agent agent
COPY --from=installer --chown=agent:agent /home/agent/.agents /home/agent/.agents
COPY --from=installer --chown=agent:agent /home/agent/.claude /home/agent/.claude
COPY --from=installer --chown=agent:agent /home/agent/install-result.json /home/agent/install-result.json

USER agent
WORKDIR /workspace
COPY --chown=agent:agent fixture /workspace/fixture
COPY --chown=agent:agent run_probe.py /home/agent/run_probe.py

# code-review needs a real base branch and a non-empty git history to diff against.
RUN cd /workspace/fixture \
    && git init -q -b main \
    && git config user.email "prototype@example.invalid" \
    && git config user.name "Probe Fixture" \
    && git add -A \
    && git commit -q -m "Initial fixture: add() has a subtraction bug"

ENTRYPOINT ["python3", "/home/agent/run_probe.py"]
