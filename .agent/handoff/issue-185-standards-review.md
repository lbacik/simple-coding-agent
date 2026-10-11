# Handoff: Review diff main...HEAD for STANDARDS only

## Task
Review the diff `git diff main...HEAD` in /data/repo (branch agent/issue-185, one commit on top of main) for STANDARDS only.

Standards sources:
- AGENTS.md: Issues and specs live in GitHub Issues via gh CLI (docs/agents/issue-tracker.md); triage labels doc; domain docs use root CONTEXT.md and docs/adr/ (docs/agents/domain.md).
- Repo conventions visible in codebase: frozen dataclasses, Protocol seams for transports with deterministic fakes in tests/fakes.py, ConfigurationError/ProfileError taxonomy, credential redaction, GraphQL transport in simple_coding_agent/github_tracker.py.

Smell baseline (Fowler, judgement calls only; documented standard overrides; skip ruff-enforced):
Mysterious Name, Duplicated Code, Feature Envy, Data Clumps, Primitive Obsession, Repeated Switches, Shotgun Surgery, Divergent Change, Speculative Generality, Message Chains, Middle Man, Refused Bequest.

Report per file/hunk: (a) every documented-standard violation with cite (file + rule); (b) baseline smells with name + quoted hunk. Distinguish hard violations from judgement calls. Under 400 words. Read-only, no edits.

## Progress
- No reads completed. Blocked immediately: Bash `git diff` calls and `Skill(handoff)` invocation both rejected by PreToolUse token-soft-threshold hook.
- Branch state at start: agent/issue-185, clean, one commit on top of main (per caller).

## Next steps for continuation
1. Run `git diff main...HEAD` (allowed command once threshold clears).
2. Read AGENTS.md, docs/agents/issue-tracker.md, docs/agents/domain.md, CONTEXT.md, docs/adr/.
3. Check diff against: frozen dataclasses, Protocol/fakes, error taxonomy, credential redaction, GraphQL transport location.
4. Return <400-word report, absolute file paths, no file writes.
