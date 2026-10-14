# Serve fleet telemetry through agentctl over the control socket, not an HTTP endpoint

A dashboard needs each managed instance's telemetry snapshot. We decided the agent serves it only as a read-only `telemetry` operation on the existing container-local control socket, exposed as `agentctl telemetry --json`, and the dashboard reads it by running `docker exec --user agent <container-id> /usr/local/bin/agentctl telemetry --json`. The agent publishes no port and opens no listener beyond the control socket, so "container = instance, no published control port" stays true, and the telemetry needs neither a token nor a setting to turn it on.

## Considered Options

- **An opt-in HTTP endpoint inside the agent, guarded by a token.** Rejected: it adds a network listener, a secret to provision and rotate per instance, and a second protocol to keep in step with `agentctl`, all to save the dashboard from needing Docker `exec`. The operator commands planned for the dashboard's second version would need the same endpoint to become writable, which would turn a read-only convenience into a remote control surface.
- **Status written to a shared volume or file.** Rejected: it needs a mount per instance and gives no way to tell a stalled agent from a quiet one, which a live socket reply does.

## Consequences

- The dashboard holds more Docker power than an HTTP pull would need (`exec` into the agent containers). The dashboard project contains this by running as the operator's own host process, bound to `127.0.0.1`, with one gateway module that can only list instances and run `agentctl telemetry`.
- The control server answers connections one at a time, so the `telemetry` handler must only serialise data that is already prepared. It does no network or git I/O, waits at most briefly for the control lock, and never lets an error escape into the listener.
- Every poll costs one `docker exec` and one Python start-up per instance, so dashboards poll on the order of seconds, not sub-second.
