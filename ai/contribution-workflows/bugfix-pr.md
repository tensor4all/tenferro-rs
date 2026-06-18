# Bug-Fix PR Workflow

Use this workflow when the user wants to prepare a pull request that fixes
incorrect behavior for an existing intended path in tenferro-rs.

Do not use this workflow for new features. If the work requires a new public
API, operation family, backend, dependency, feature flag, architectural layer,
or AD semantics change, stop and use `ai/contribution-workflows/issue-intake.md`
to create or refine a feature request or design discussion issue.

## Operating Rules

- Read `CONTRIBUTING.md`, `AGENTS.md`, `REPOSITORY_RULES.md`, and
  `ai/contribution-workflows/repository-remediation.md` before implementation
  work. Always read the remediation workflow so the agent can recognize when a
  one-bug PR should become a related-finding batch.
- If the request covers multiple related bug reports, a repository-rule audit,
  or a bug batch that should stay in one PR, follow the remediation workflow.
  That workflow adds the classification ledger, coherent commit, and
  non-squash merge rules for batched work.
- Keep the PR focused on the bug. Do not include opportunistic refactors,
  unrelated cleanup, feature work, or dependency changes.
- Preserve user changes in the working tree. Do not reset, stash, or overwrite
  unrelated local work unless the user explicitly asks.
- Ask only for missing facts. Prefer reproducing, searching, and reading code
  over asking the user for facts available in the repository.
- Ask at most three questions at a time.
- Do not use squash merge for bug-fix batches. Preserve coherent commits with a
  merge commit unless maintainers explicitly choose another non-squash method.
- Maintainers retain review, merge, and policy authority.

## Step 1: Scope Gate

Before editing code, verify that the change is a bug fix.

It may proceed as a bug-fix PR only when all of these are true:

- The target behavior is already intended by current docs, tests, issue
  discussion, or existing API contracts.
- The fix can be implemented without new public API.
- The fix can be implemented without a new operation family, backend,
  dependency, feature flag, or architectural layer.
- The fix does not change roadmap direction or require design acceptance.

If any condition fails, stop the PR path and switch to issue intake.

## Step 2: Reproducer And Expected Result

Collect or derive:

- Related issue number, if any
- Expected behavior
- Actual behavior
- Minimal reproducer, failing test, command, panic, error, or log
- Affected crate, API, backend, device, dtype, and feature flags when known
- Verification command or test that should pass after the fix

If no reproducer exists, create the smallest one that demonstrates the current
incorrect behavior before changing implementation code whenever practical.

## Step 3: Branch And Inspect

Work from the latest `origin/main` when starting a new bug-fix branch. If the
current checkout is dirty, do not disturb unrelated changes; either work with
the user's branch or ask before creating a separate worktree.

Inspect the implementation boundary before editing:

- Public API and rustdoc
- Relevant tests and examples
- Crate ownership and layering
- Dependency and feature declarations
- Backend, device, layout, cache, and AD contracts when touched

Before committing, search for same-root-cause and same-pattern bugs in the
neighborhood:

- the touched file and module;
- the same operation family, trait implementation family, generated wrapper, or
  cache/runtime/backend boundary;
- equivalent eager/traced/CPU/GPU/FFI paths when the bug pattern is shared.

Fix same-root-cause instances in the same PR when they share the same contract
and verification path. Stop expanding when the search reaches another
subsystem, requires a different policy decision, or would dominate review.

## Step 4: Implement The Smallest Correct Fix

Patch the root cause at the lowest appropriate layer.

Prefer:

- One focused behavior change
- A regression test or minimal verification case
- Existing local helper APIs and crate patterns
- Documentation updates only when the bug fix changes user-visible behavior or
  clarifies an existing contract
- A small repository-rule or workflow-rule proposal when the root cause can be
  detected by a general audit rule and the rule would prevent similar bugs.
  Before adding a new rule, check existing audit and repository rules for
  overlap and prefer merging, tightening, or relocating an existing rule over
  adding another near-duplicate bullet.

Avoid:

- New feature work
- Broad rewrites
- New dependencies
- Public API expansion
- Backend/provider policy changes
- Hidden materialization or performance regressions in tensor paths

If implementation reveals that the fix needs design work, stop and create or
update an issue instead of stretching the PR.

If the reported bug is a false positive because the code intentionally relies
on a non-obvious invariant, do not silently skip it. Record the evidence in the
issue, PR body, or work log, and add a nearby source comment, rustdoc note, or
source-contract test when that would prevent future humans or agents from
misidentifying the same code as a bug.

## Step 5: Verify

Run the narrowest meaningful checks first, then broader checks as risk
requires.

Typical checks:

```bash
cargo fmt --all --check
cargo test -p <crate> <test-name>
cargo test -p <crate>
```

For PR-ready changes, follow the repository checklist in `AGENTS.md` when
practical. If a full check is too expensive or environment-dependent, state
exactly what was run and what remains unverified.

## Step 6: Draft The PR

Before creating the PR, prepare a body that includes:

- Summary of the bug and root cause
- What changed
- Why the change stays within bug-fix scope
- Same-root-cause search performed, related instances fixed, and related
  findings intentionally deferred
- Any audit rule, repository-rule update, or workflow-rule proposal created to
  prevent the same class of bug
- The rule-inventory decision when adding or changing an audit rule: reused an
  existing rule, merged overlapping rules, or added a new rule because no
  existing rule fit
- Regression test or verification commands
- Related issue
- Any skipped checks or residual risk

Use `.github/pull_request_template.md`. If the PR is AI-assisted, state that
the bug-fix scope gate was applied.

Create the PR only after the user approves when working interactively, unless
the user explicitly asked for direct PR creation.
