# Upstream skills and installer contract

Research for [Establish the upstream skills and installer contract](https://github.com/lbacik/simple-coding-agent/issues/3), inspected 2026-09-19. This is source research, not a successful runtime integration test or an approved implementation design. No upstream installer or skill scripts were executed.

## Finding

The requested upstream skills can be installed for Claude Agent SDK, but their current instructions are not an unattended workflow contract. Human-confirmed test seams, a committed review diff, an explicit review base, repository tracker instructions, and functioning review subagents must be supplied or deliberately adapted. Installing only `implement` and `tdd` is insufficient.

## Immutable inputs

- Skills: [`mattpocock/skills@c55ee46073ed923f86ce59a5eb3b6d895095d1b7`](https://github.com/mattpocock/skills/tree/c55ee46073ed923f86ce59a5eb3b6d895095d1b7).
- Installer: published `agent-installer@0.6.0`; [npm version metadata](https://registry.npmjs.org/agent-installer/0.6.0) identifies source commit `7878768d226769214b49c59c4e4d98e4d17a2f93`, Node `>=20`, and the [release tarball](https://registry.npmjs.org/agent-installer/-/agent-installer-0.6.0.tgz).
- Downloaded tarball integrity was independently calculated and matched npm: `sha512-k6FXrRrlLFUaI+eX0ipYmRGxCA+qBrZP8ydogiEDNVKq7vFIin4X9MIq0FGMzWgqWGdbGH/rFUPHZ6yTSV+Mng==`. Its bundled CLI was read without execution and contains the expected HOME path resolver and exact-selector implementation.

Pinning a package version alone does not freeze transitive npm dependencies; the package metadata includes dependency ranges. A reproducible image should additionally record a dependency lock, image digest and installed manifest. This is a recommendation, not an existing installer feature.

## Skill closure and behavior

| Artifact | Required behavior or dependency |
|---|---|
| [`implement`](https://github.com/mattpocock/skills/blob/c55ee46073ed923f86ce59a5eb3b6d895095d1b7/skills/engineering/implement/SKILL.md) | Explicit invocation only (`disable-model-invocation: true`). Implement the supplied spec/tickets, use TDD where possible at pre-agreed seams, run typechecking and focused tests regularly, full suite once at the end, invoke code-review, then commit to the current branch. |
| [`tdd`](https://github.com/mattpocock/skills/blob/c55ee46073ed923f86ce59a5eb3b6d895095d1b7/skills/engineering/tdd/SKILL.md) | Before any test, document seams and obtain user confirmation. Read available CONTEXT/ADRs. Calls `codebase-design` when interface shape is uncertain. References local `tests.md` and `mocking.md`, and assigns refactoring to code-review. |
| [`code-review`](https://github.com/mattpocock/skills/blob/c55ee46073ed923f86ce59a5eb3b6d895095d1b7/skills/engineering/code-review/SKILL.md) | Requires tracker instructions; asks for missing base/spec; requires a valid nonempty `git diff <base>...HEAD`; runs Standards and Spec reviewers as two parallel subagents. It reports findings, but does not explicitly prescribe an automatic fix-and-repeat loop. |
| [`codebase-design`](https://github.com/mattpocock/skills/blob/c55ee46073ed923f86ce59a5eb3b6d895095d1b7/skills/engineering/codebase-design/SKILL.md) | Conditional reference dependency from TDD. Includes `DEEPENING.md` and `DESIGN-IT-TWICE.md`. The latter is an optional design exploration requiring three or more parallel subagents, not a mandatory implementation step. |

Thus the runtime closure is `implement`, `tdd`, `code-review`, and conditional `codebase-design`, with their complete directories. The setup skill is a remediation pointer: code-review tells a user to run `/setup-matt-pocock-skills` if `docs/agents/issue-tracker.md` is absent. It need not run for each task if repository onboarding supplies the file. No direct dependency in this closure requires `grilling`, `domain-modeling`, or `diagnosing-bugs`; repositories can impose additional requirements through their own instructions. [Setup source](https://github.com/mattpocock/skills/blob/c55ee46073ed923f86ce59a5eb3b6d895095d1b7/skills/engineering/setup-matt-pocock-skills/SKILL.md)

There is an ordering conflict on a fresh branch: implement requests review before its final commit, while code-review compares committed HEAD against the base. Uncommitted work alone produces an empty review diff. The plan must choose checkpoint commits before review or an explicit review adaptation covering the working tree. This conclusion follows directly from the two skill instructions, not from a runtime experiment.

The inspected implementation closure mandates a local commit, but does not mandate push, issue comments, PR creation, merge, or issue closure. Tracker access is delegated to repository instructions; this project's instructions use `gh issue view --comments` to fetch issues. Publication ownership therefore remains an application decision, and preserving the upstream commit instruction must be considered separately from allowing GitHub writes. [Repository tracker instructions](https://github.com/lbacik/simple-coding-agent/blob/main/docs/agents/issue-tracker.md)

## Published installer contract

Version 0.6.0 scans nested `skills/` directories (default depth 3), identifies a skill by its directory basename, and copies the complete directory. The upstream `skills/engineering/<name>` layout fits. `--only` selects exact artifact IDs; it does not read prose references or resolve dependency graphs. Consequently, the closure above must be explicitly enumerated. [Scanner source](https://github.com/lbacik/agent-installer/blob/7878768d226769214b49c59c4e4d98e4d17a2f93/src/source.ts), [selection and copy source](https://github.com/lbacik/agent-installer/blob/7878768d226769214b49c59c4e4d98e4d17a2f93/src/install.ts)

Illustrative build invocation after installing the pinned npm release, not executed during this research:

```sh
agent-installer install https://github.com/mattpocock/skills.git \
  --ref c55ee46073ed923f86ce59a5eb3b6d895095d1b7 \
  --only skill:implement --only skill:tdd \
  --only skill:code-review --only skill:codebase-design --json
```

Install under the final runtime user's HOME, for example `/home/agent`. Version 0.6.0 stores skills at `$HOME/.agents/skills/<name>`, links `$HOME/.claude/skills/<name>` to their absolute store paths, and writes state under `$HOME/.agents/agent-installer/state.json`. Moving only these trees to a different HOME breaks links. The runtime user must be able to read them; avoid mounting a volume over the installed HOME and hiding the bundle. [Path resolver](https://github.com/lbacik/agent-installer/blob/7878768d226769214b49c59c4e4d98e4d17a2f93/src/paths.ts), [build guidance](https://github.com/lbacik/agent-installer/blob/7878768d226769214b49c59c4e4d98e4d17a2f93/README.md#build-time-installation)

Remote installation requires Node >=20, git, and network access to GitHub. npm is needed to obtain the package. Node/installer need not remain solely for skills installation in the runtime stage. Worker tasks still need git, shell and project-specific test/typecheck tools; gh is required if their tracker instructions are executed unchanged. These are environment requirements, not skill dependencies installed automatically.

`--json` returns schemaVersion 1 and artifact provenance including requestedRef, resolvedCommit and hashes. Unmatched selectors and unmanaged conflicts fail; do not use `--allow-conflicts` to conceal a missing required skill. Preserve the install result and verify `list --json`, files, and link resolution during image validation. [Release README](https://github.com/lbacik/agent-installer/blob/7878768d226769214b49c59c4e4d98e4d17a2f93/README.md)

Do not substitute current installer main for 0.6.0: at [`3e92054f527b0fb92c2f8d7cf1e841218db51a7e`](https://github.com/lbacik/agent-installer/blob/3e92054f527b0fb92c2f8d7cf1e841218db51a7e/src/config.ts), absence of `config.yaml` means base-store-only installation with no configured exposures. This differs materially from 0.6.0's automatic Claude links.

## SDK discovery and validation

Explicit `setting_sources` must include `user` for the installer's HOME-based skills. `project` additionally enables repository settings/skills and should be a deliberate trust choice. Dispatch `/implement` in the prompt; enable its helper names for model invocation. Current documentation supports a `skills` allowlist; if setting a tools list, include `Skill`. Check initialization metadata and actual invocation: matching names alone cannot prove that the expected source won a collision, especially since the runtime bundles a `code-review` skill. SDK defaults and fields are version-sensitive; pin and test the chosen SDK/runtime. [Official SDK skills documentation](https://code.claude.com/docs/en/agent-sdk/skills)

This research establishes installation and instruction contracts, not Muse Spark compatibility. The integration test must demonstrate discovery of the intended upstream bytes, `/implement` dispatch, helper invocation, parallel reviewers, file edits, test execution, and structured completion/failure with the selected provider.

## Decisions still requiring the owner

1. Require approved seams in ready issues, or authorize an explicit AFK seam-selection policy? Missing approval must not silently count as success.
2. Keep upstream skills unchanged with supplied prerequisites and checkpoint commits, or add a separately versioned instruction overlay? The user requires skills to originate from upstream; a fork is not assumed authorized.
3. Allow worker commits and tracker reads while reserving push/comments/PRs for the orchestrator, or choose another publication model?
4. Select the four-skill closure or a larger audited bundle; decide whether project settings may introduce additional skills or override names.
5. Define review finding severity, repair limits and terminal blocked behavior. Upstream review itself does not supply an unattended completion gate.
