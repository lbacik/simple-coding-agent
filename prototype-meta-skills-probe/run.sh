#!/usr/bin/env bash
# PROTOTYPE, disposable. Builds and runs the ticket #8 probe container, then
# copies the report and transcript back next to this script.
set -euo pipefail
cd "$(dirname "$0")"

if grep -q REPLACE_WITH_META_MODEL_API_KEY .env; then
  echo "Edit .env and put a real ANTHROPIC_AUTH_TOKEN in before running." >&2
  exit 1
fi

docker build -t meta-skills-probe .

container_id=$(docker create --env-file .env meta-skills-probe)
trap 'docker rm -f "$container_id" >/dev/null 2>&1 || true' EXIT

docker start -a "$container_id"
status=$?

docker cp "$container_id:/workspace/probe-report.json" ./probe-report.json || true
docker cp "$container_id:/workspace/probe-transcript.log" ./probe-transcript.log || true

exit "$status"
