# Conformance harness (operator contract §15.1)

The external producer suite drives this repository through the Conformance
Harness API with `python -m conformance_harness <verb> '<request-json>'`.
Every invocation writes exactly one JSON reply to stdout; the exit code
mirrors the contract class (`0` ok, `1` failed, `2` usage, `3`
unsupported). The argv verb and the document verb must agree, and unknown
parameters are usage errors.

## Verbs

Fourteen verbs, all implemented against the real `AgentLifecycle` wired to
the shared test fakes (`tests/fakes.py` — reused, never copied):

- `setup` — create a fresh state volume and start the fakes-wired engine.
  Reports the six discovery facts (container, image, volumes, exec user,
  repository, capabilities). Returns before startup checks finish so the
  suite can observe `starting` itself.
- `teardown` — stop the engine and remove the volume.
- `set-issue` — seed one issue and steer its eligibility label.
- `hold` / `release` — arm and clear a hold (`before-claim`, `after-claim`,
  `native-phase`, `after-outcome`, optionally scoped to the active attempt).
- `finish-attempt` — script one native outcome for the active attempt and
  release any hold so it runs to finalization.
- `hold-lock` — hold the cross-process engine lock for `duration_ms`.
- `crash` — kill the engine now, or arm a one-shot crash at
  `between-commit-and-reply`, `before-claim`, `after-claim`, or
  `during-finalization`.
- `terminate` — graceful `SIGTERM` stop.
- `stop-container` / `start-container` — stop and restart the engine while
  keeping the volume, for Offline Read and recovery coverage.
- `halt` — record a halt and exit the engine by decision.
- `grow-archive` — append bytes to the newest unsealed archive file.
- `defer-publication` — arm the one-shot deferred-publication crash.

`setup` answers `unsupported` naming an unknown Operator Capability, and
`finish-attempt` answers `unsupported` naming an unknown native outcome.

## State directory

One current instance lives under `$SCA_HARNESS_STATE_DIR` (default
`~/.cache/sca-conformance-harness`): `state.json`, `issues.json`, the
`instance/` data volume, a local `remote.git`, the `clone/`, the control
`daemon.sock`, and the `flags/` the suite uses to steer the engine.

## Image targets

- `docker build --target harness .` — the suite image: base plus
  `conformance_harness/` and `tests/fakes.py`, with
  `python -m conformance_harness` as the entry point.
- `docker build .` — the production image (the last `production` stage, so
  the default build output is unchanged): `simple_coding_agent/` only, with
  neither the harness nor the fakes.

## Local smoke run

`tests/test_conformance_harness.py` drives every verb and asserts each
protocol state is reached — `starting`, `recovering`, `idle`, `active`,
`draining`, `paused`, halted (`last_halt`) and stopped (Offline Read):

```shell
UV_PROJECT_ENVIRONMENT=/home/agent/venv-py311 \
  uv run --python 3.11 --extra dev pytest tests/test_conformance_harness.py
```

## Docker mode (external producer suite)

The pinned producer suite speaks to the producer only through real
`docker exec`/`docker run`, so CI runs the harness in Docker mode: every
verb is proxied into a container built from the `harness` image target,
which bakes the volume paths in as its `DATA_DIR`/`AGENTCTL_SOCKET`
defaults so the suite's offline `docker run` (no environment) reads the
same volume the daemon writes.

```shell
docker build --target harness -t simple-coding-agent:harness .
SCA_HARNESS_DOCKER=1 SCA_HARNESS_IMAGE=simple-coding-agent:harness \
  python -m agent_operator_contract.producer_suite --harness scripts/harness.sh \
    --capabilities stop-after,next-issue,operator-handoff,recovery-retry,recovery-release,errors-reset \
    --require-all --timeout 180
```

`SCA_HARNESS_DOCKER=1` makes `setup` create a named volume and a
`sleep infinity` container from `SCA_HARNESS_IMAGE` (overriding the image
entrypoint), then run `setup` through `docker exec` so the engine starts
inside the container on the volume; every later verb is the same
`docker exec` proxy, and `teardown` removes the container and the volume.
`DOCKER` selects the docker executable. Without the flag the harness stays
process-local (unit tests, local smoke runs).
