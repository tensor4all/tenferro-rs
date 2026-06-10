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
- After beginning the first issue-986 remediation pass, add a classification
  ledger requirement, stale-finding handling, historical-docs scope guard, and
  subagent coordination rules. These were added because the first pass needed a
  durable way to integrate explorer/worker outputs and distinguish current
  failures from already-fixed historical comments.
- Allow autonomous rule growth only when the user explicitly authorizes it for
  a remediation session. The rule update must stay within the approved
  objective and remain reviewable as its own coherent unit.
- After integrating the first runtime/performance fixes, add a rule that
  replaced hot-path helpers should be removed when obsolete instead of kept
  alive through artificial references. This was added after reviewing an
  automated patch that otherwise preserved a dead helper only to avoid an
  unused-code warning.
- After replacing graph compile-cache fingerprints, add a rule that cache-key
  remediations must preserve exact equality and cannot use hash-only identity
  unless collisions are still resolved by structural equality or an equivalent
  payload-equality contract.

## Rejected Alternatives

- Adding the remediation workflow directly to `REPOSITORY_RULES.md` was
  rejected because it would mix implementation contracts with contribution
  mechanics.
- Opening PRs early to use GitHub CI as a discovery mechanism was rejected
  because CI includes costly GPU checks. The workflow requires local completion
  and verification before PR creation.
- Squash-merging remediation batches was rejected because the batch may contain
  several reviewable units whose commit boundaries should remain visible.
- Treating subagent output as final proof was rejected. The coordinating agent
  remains responsible for integration, verification, ledger updates, and final
  reporting.

## Verification

- Reviewed the new workflow against `REPOSITORY_RULES.md`.
- Ran `git diff --check`.
- During the first remediation pass, verified that docs/rustdoc fixes surfaced
  the need for a classification ledger and subagent coordination rule before
  applying those rule improvements.
- During the runtime/performance pass, verified that the obsolete-helper rule
  matched the actual integration review by removing the replaced einsum cost
  helper and rerunning focused tests.
- During the compile-cache pass, verified the exact-equality rule with an
  extension payload-hash collision test that still misses the cache when
  `payload_eq` is false.

## Residual Risks

- The workflow is not yet exposed through a thin tool adapter in
  `.agents/skills/`, `.claude/skills/`, or `.opencode/commands/`. Agents can
  still reach it through `AGENTS.md`; adding adapters can be a follow-up if
  maintainers want command-level entry points.
