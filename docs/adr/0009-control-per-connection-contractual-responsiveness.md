# Serve each control connection on its own thread as a contractual responsiveness requirement

ADR 0006 (telemetry through `agentctl`) says "the control server answers connections one at a time, so the `telemetry` handler must only serialise data that is already prepared". That rationale is out of date on both counts: the control server already serves each connection on its own daemon thread (`ControlServer._serve` in `simple_coding_agent/control_server.py`), so one connection waiting on the control lock never blocks another; and the operator contract (major 1) now requires that behaviour. We decided per-connection serving stays and is recorded here as a contractual obligation: `identity`, `status`, and `command` answer within 3 s even while a mutation waits on the control lock, doing no network, git, or provider I/O on the serving path.

## Considered Options

- **Keep serial serving and accept head-of-line blocking.** Rejected: a dashboard polling a busy instance would stall behind a mutating submit holding the control lock, which the contract's responsiveness requirement forbids.
- **Serve per connection but let handlers do live I/O.** Rejected: the same stall returns through a slower door. Handlers keep serving prepared state only, reusing the bounded control-lock path that `telemetry` uses today.

## Consequences

- The listener accepts connections in a loop and hands each one to its own daemon thread; a failing connection is contained to its own thread and never takes down the listener.
- Telemetry, status, and command handlers serialise data the lifecycle already prepared. They wait at most briefly on the control lock and never perform network, git, or provider I/O.
- This ADR supersedes the serving rationale of ADR 0006 (telemetry through `agentctl`): the transport decision there (serve through `agentctl` over the control socket, no HTTP endpoint) stands; only the "one at a time" reasoning is replaced.
- This ADR respects ADR 0001: workspace Sealing (committing workspace changes after every model execution) is unchanged, and archive sealing — marking a complete, immutable Attempt Archive — is a separate, later concept owned by the contract work, not by this decision.
