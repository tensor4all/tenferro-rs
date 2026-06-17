---
name: tenferro-bugfix-pr
description: Prepare a tenferro-rs pull request that fixes incorrect behavior for an existing intended path. Use for bug-fix PRs, regression fixes, panic fixes, wrong-result fixes, and compatibility fixes. Do not use for new features, new public APIs, new backends, new dependencies, feature flags, or architectural changes; redirect those to issue intake.
---

# Tenferro Bug-Fix PR

Follow `ai/contribution-workflows/bugfix-pr.md` as the canonical workflow.
When the user asks for a batch of related bug reports, repository-rule audit
findings, or same-root-cause remediation in one PR, also follow
`ai/contribution-workflows/repository-remediation.md`.

Read the applicable workflow files before editing code. Apply the scope gate
before implementation.

Proceed only when the change fixes behavior already intended by current docs,
tests, issue discussion, or API contracts. If the fix needs a new public API,
operation family, backend, dependency, feature flag, architectural layer, or AD
semantics change, stop and use the `tenferro-issue-intake` workflow instead.

Keep the interaction incremental:

1. Confirm bug-fix scope.
2. Collect or derive the reproducer and expected result.
3. Inspect the affected crate boundary and current tests.
4. Search for same-root-cause and same-pattern bugs in nearby modules and
   equivalent eager/traced/CPU/GPU/FFI paths.
5. Implement the smallest correct fix, or a coherent batch when related
   instances share the same contract and verification path.
6. Add or update a regression test when practical.
7. Propose or update a general audit/repository rule when it would prevent the
   same class of bug.
8. For false positives, record the evidence and add a nearby source comment,
   rustdoc note, or source-contract test when the invariant is not obvious.
9. Verify with targeted checks and document any skipped checks.
10. Draft the PR body from `.github/pull_request_template.md`; for batches,
    preserve coherent commits and do not use squash merge.
