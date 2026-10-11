# Keep PR auto-merge an operator-only setting, never a repository profile key

The orchestrator can merge the pull request it opens for a complete implementation without a human merging it (`PR_AUTO_MERGE=true`, with `PR_MERGE_METHOD` selecting `merge`, `squash`, or `rebase`). The orchestrator enables GitHub's native auto-merge right after creating the PR — or finding it already existing — so GitHub merges once the base branch's protection rules are met, and merges directly when GitHub refuses because the PR is already mergeable. We decided this is an operator setting read from `.env` only, alongside `AGENT_TRUST_PROJECT_SETTINGS`, and that the repository profile keeps rejecting unknown keys, so an auto-merge key in the profile is refused.

The profile lives in the target repository, the same repository the agent opens PRs against. If auto-merge could be enabled there, an auto-merged PR could edit the profile and grant itself more trust — for example, widening the check command or the environment its own solution is judged by. The operator's `.env` is outside the agent's reach, so a PR can never change the conditions of its own merging.

## Considered Options

- **Allow an auto-merge key in the repository profile.** Rejected: the profile is repository-controlled content that the agent's own PRs can modify, so a self-modifying PR could escalate its own trust.
- **Let the model merge with `gh pr merge`.** Rejected: the model runs with `bypassPermissions` and its merge decision would be model judgment; the SDK guard already denies model-side merges. Only the orchestrator merges, after deterministic completion evidence, and the safety decision stays with branch protection.
- **Merge directly without trying native auto-merge first.** Rejected: a direct merge at creation time would race required checks. Native auto-merge first means GitHub waits for protection; the direct merge is only the fallback for a PR that is already mergeable.

## Consequences

- Operators opt in with `PR_AUTO_MERGE=true` and pick a method with `PR_MERGE_METHOD` (default `merge`); invalid values fail at startup with a configuration error.
- The repository must turn on "Allow auto-merge" and protect the base branch with required checks (and required reviews where a human must still approve); otherwise auto-merged PRs land with no human review.
- The existing PAT scopes (`contents: write`, `pull-requests: write`) are enough; no new scope is required.
- A merge failure never redefines completion: the attempt stays `complete` and the result comment reports the merge step (`auto-merge enabled (<method>)`, `merged`, or `auto-merge failed: <reason>`).
