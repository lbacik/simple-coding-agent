# Seal attempt work unconditionally, and guard cleanup against a dirty workspace

Uncommitted model work used to be preserved only for an enumerated set of model execution statuses, through an optional workspace method. Each real preservation bug (the success path, then the infrastructure-error path in #103) was fixed by widening that list, and fakes without the method silently skipped preservation. We decided that the model attempt runner seals the attempt workspace after every model execution, whatever its status and also when execution raises. The one exception is a workspace with an unresolved rebase or merge, which is aborted instead of sealed. Independently, workspace cleanup refuses to switch to the base branch while the tree is dirty and pauses the agent loop instead. We keep both safeguards on purpose: sealing gives each commit its attribution (issue, attempt ID, status), and the cleanup guard makes any path that forgets to seal fail loudly instead of carrying work onto the base.

## Considered Options

- **Sealing only in the runner.** Rejected: a future exit path that skips the seal call would bring back the silent leak.
- **Sealing only inside cleanup.** Rejected: cleanup does not know which attempt or status produced the work, so the commit could not be attributed.
