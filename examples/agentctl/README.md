# Synthetic `agentctl` command-reply examples

These files are **synthetic**: invented repositories (`example-owner/...`)
and fixed request IDs (`req-example-...`), so each file is byte-identical
to real CLI output. No data was captured from live instances. The
dashboard repository is public and copies them.

- `accepted.json` — `stop` accepted (an active attempt finishes first).
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

Each file is `{"argv", "exit_code", "stdout", "stderr"}`. `{{SOCKET}}` in
`argv` is the control socket the test substitutes (a fresh test server
per file, so every committed command has sequence 1); `not-submitted`
pins the missing socket path it was recorded with. The trailing newline
`print()` adds is part of `stdout`/`stderr`.

Contract: `docs/agents/agentctl-command-replies.md`. Conformance test:
`tests/test_agentctl_command_replies_examples.py`, which reproduces every
example from the real CLI and fails when the output drifts.
