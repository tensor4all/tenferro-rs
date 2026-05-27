---
name: tenferro-bugfix-pr
description: Prepare a tenferro-rs pull request that fixes incorrect behavior for an existing intended path. Use for bug-fix PRs, regression fixes, panic fixes, wrong-result fixes, and compatibility fixes. Do not use for new features, new public APIs, new backends, new dependencies, feature flags, or architectural changes; redirect those to issue intake.
---

# Tenferro Bug-Fix PR

Follow `ai/contribution-workflows/bugfix-pr.md` as the canonical workflow.

Read that file before editing code. Apply the scope gate before implementation.

Proceed only when the change fixes behavior already intended by current docs,
tests, issue discussion, or API contracts. If the fix needs a new public API,
operation family, backend, dependency, feature flag, architectural layer, or AD
semantics change, stop and use the `tenferro-issue-intake` workflow instead.

Keep the interaction incremental:

1. Confirm bug-fix scope.
2. Collect or derive the reproducer and expected result.
3. Inspect the affected crate boundary and current tests.
4. Implement the smallest correct fix.
5. Add or update a regression test when practical.
6. Verify with targeted checks and document any skipped checks.
7. Draft the PR body from `.github/pull_request_template.md`.
