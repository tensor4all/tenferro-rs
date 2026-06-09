# Repository Remediation Workflow

## Summary

This session added a repository-local workflow for batched, agent-assisted
remediation of repository-rule violations. The workflow is linked from
`AGENTS.md` and listed in `ai/contribution-workflows/README.md`.

## Context Read

- `AGENTS.md`
- `CONTRIBUTING.md`
- `REPOSITORY_RULES.md`
- `ai/README.md`
- `ai/contribution-workflows/README.md`
- `ai/contribution-workflows/bugfix-pr.md`

## Decisions

- Put the reusable workflow in
  `ai/contribution-workflows/repository-remediation.md` rather than in
  `REPOSITORY_RULES.md`. The new file describes remediation mechanics; durable
  implementation contracts remain in `REPOSITORY_RULES.md`.
- Link the workflow from `AGENTS.md` so agents see it before starting
  multi-issue remediation work.
- Make batched repository-rule remediation a deliberate exception to the normal
  one-bug-fix PR path: one branch, one final PR, coherent commits, local-first
  verification, and non-squash merge.
- Add a neighborhood-scan rule so agents look for same-root-cause problems
  near the fix without expanding into unrelated subsystem work.
- Add an interactive-session rule-improvement policy: agents may fix typos
  directly, but must propose behavior-changing rule improvements to the user
  before applying them.

## Rejected Alternatives

- Adding the remediation workflow directly to `REPOSITORY_RULES.md` was
  rejected because it would mix implementation contracts with contribution
  mechanics.
- Opening PRs early to use GitHub CI as a discovery mechanism was rejected
  because CI includes costly GPU checks. The workflow requires local completion
  and verification before PR creation.
- Squash-merging remediation batches was rejected because the batch may contain
  several reviewable units whose commit boundaries should remain visible.

## Verification

- Reviewed the new workflow against `REPOSITORY_RULES.md`.
- Ran `git diff --check`.

## Residual Risks

- The workflow is not yet exposed through a thin tool adapter in
  `.agents/skills/`, `.claude/skills/`, or `.opencode/commands/`. Agents can
  still reach it through `AGENTS.md`; adding adapters can be a follow-up if
  maintainers want command-level entry points.
