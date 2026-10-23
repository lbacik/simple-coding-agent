# Mutating `agentctl` command-reply contract

The dashboard issues Control actions through the mutating `agentctl`
commands and parses their replies; CLI operators script against the same
text. This document publishes that reply format as a contract, the same
way [telemetry-snapshot.md](telemetry-snapshot.md) publishes the
telemetry snapshot. The socket wire format is out of scope and does not
change.

Covered commands: `stop`, `stop --after N`, `resume`, `handoff now`,
`next issue <n>`, `recovery retry <attempt>`,
`recovery release <attempt> --saved-at <path-or-url>`, `errors reset`
(alias `consecutive-errors reset`), and `command <id>` (read-only lookup
of one command's durable acknowledgement).

## stdout on exit 0: one `key: value` block

A committed command prints five `key: value` lines to stdout:

```text
repository: example-owner/example-repo
request_id: req-example-0001
sequence: 1
acknowledgement: accepted
effect: finish active attempt, then stop intake
```

- One line per key; the value runs to the end of the line.
- Values contain no newline (since v0.7.3): `effect:` carries the
  stored free-text `detail`, and the CLI collapses any embedded line
  break to a space. The same guarantee covers every other value.
- Key order is not part of the contract.
- New keys are additive; readers ignore unknown keys, the same rule as
  the telemetry snapshot.

`acknowledgement` is one of `accepted` (the effect is pending; watch
`status` until it lands), `completed` (the effect is durably reached),
`superseded`, or `not fulfilled`.

## stderr on exit 1/2: optional `repository:` line, then a reason line

A command the agent answered prints the answering instance first:

```text
repository: example-owner/example-repo
req-example-0009 rejected — stop --after N is rejected while intake is stopped; resume first. No control change was accepted.
```

- The first stderr line is `repository: <TARGET_REPO>` whenever the
  socket reply carries it (since v0.7.3): every rejection the agent
  answered, and `command <id>` "no such command". "Not submitted"
  (exit 2) has no reply, so it prints no `repository:` line; neither do
  unreachable errors and local validation failures.
- Then comes the reason line. The reason text is for display only and is
  not stable: it always names the request ID (since v0.7.3, including
  local `stop --after` validation), and a rejection prints "No control
  change was accepted." exactly once (since v0.7.3), whether or not the
  underlying message already contains it.
- `command <id>` for an unknown ID prints
  `<id>: no such command is recorded.` with the same leading
  `repository:` line.

## Exit codes

- `0`: committed. The reply above is on stdout.
- `1`: rejected with no change, or an unknown request ID for
  `command <id>`.
- `2`: outcome unknown — retry with the same request ID
  (`agentctl <command> --request-id <id>`), then `command <id>` to
  confirm — or a usage error, recognised by the `usage:` prefix on
  stderr. Exit 2 stays shared so existing operator scripts don't break.

There is deliberately no `--json` mode for these commands and no
distinct usage-error exit code.

## Versioning

The output carries no version number. New flags and behaviours are
marked "since vX.Y" in this document. The telemetry `schema_version`
does not change.

## Synthetic examples

`examples/agentctl/` holds synthetic command replies a dashboard can
build and test against: invented repositories (`example-owner/...`) and
fixed request IDs, so each file is byte-identical to real CLI output.
No data was captured from live instances. The dashboard repository is
public and copies them.

- `accepted.json` — `stop` accepted (active attempt finishes first).
- `completed.json` — `stop` completed (no active attempt).
- `rejected.json` — `stop --after` rejected by a stopped instance, with
  the `repository:` line.
- `not-submitted.json` — connection loss: exit 2, no reply, no
  `repository:` line, retry hint naming the request ID.
- `usage-error.json` — argparse usage error: exit 2, `usage:` prefix on
  stderr, empty stdout.
- `command-found.json` — `command <id>` for a recorded command.
- `command-not-found.json` — `command <id>` for an unknown ID, with the
  `repository:` line.

Each file is `{"argv", "exit_code", "stdout", "stderr"}`; `{{SOCKET}}`
is the control socket the test substitutes. Contract:
this document. Conformance test:
`tests/test_agentctl_command_replies_examples.py`, which reproduces
every example from the real CLI against a test control socket and fails
when the CLI output drifts from the pinned files.
