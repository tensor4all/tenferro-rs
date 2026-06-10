# Repository Remediation Workflow

Use this workflow when an agent is asked to automatically resolve multiple
repository-rule violations, audit findings, or related issues in tenferro-rs.
This workflow is for batched remediation work, not for ordinary one-bug PRs.

The goal is to let agents make steady progress without turning repository-rule
cleanup into unbounded refactoring. A remediation PR should fix verified,
related problems; it should not silently change project policy, public API, or
AD semantics.

## Operating Rules

- Read `CONTRIBUTING.md`, `AGENTS.md`, `REPOSITORY_RULES.md`, and this file
  before implementation work.
- Start from the latest `origin/main` in an isolated worktree unless the user
  explicitly requests a different base.
- Preserve unrelated user changes. Do not reset, stash, overwrite, or rebase
  unrelated local work unless the user explicitly asks.
- Keep one remediation branch and one final PR for the approved batch.
- Use coherent commits for reviewable units of work. Do not collapse the whole
  batch into one commit unless the batch is genuinely one atomic change.
- Do not open a PR until all approved local fixes and local verification are
  complete. GitHub CI is expensive, including GPU checks, so avoid speculative
  PRs that exist only to discover basic failures.
- Do not use squash merge for remediation batches. Preserve the coherent commit
  structure with a merge commit unless maintainers explicitly choose another
  non-squash method.
- Maintainers retain review, merge, policy, and roadmap authority.

## Classification Ledger

Before implementation, maintain a local classification ledger for the findings
in scope. The ledger may live in a work log, PR body draft, or uncommitted
scratch file outside the repository, but the final PR must include the same
information in reviewable form.

For each finding, record:

- the source issue/comment or historical finding number;
- `Auto Fix`, `Verify First`, or `Design Gate`;
- current evidence from the working hash, not only historical comments;
- the smallest useful verification target;
- whether the item was fixed, narrowed, marked stale, or deferred;
- residual risk or follow-up issue when not fixed.

Historical comments are evidence, not authority. Reclassify an item when the
current worktree or an earlier commit in the same batch already fixed it.

If new issue comments, maintainer notes, or agent verification reports arrive
during the remediation session, pause before starting the next fix and
reconcile the ledger against the latest remote issue state. Narrow over-broad
items, mark stale items out of scope, and record any newly confirmed residual
paths before continuing implementation.

## Scope Classification

Classify each candidate finding before editing.

### Auto Fix

Proceed in this workflow when the fix stays within existing intended behavior:

- stale docs, README, rustdoc, or examples;
- missing required rustdoc examples for existing public items;
- incorrect behavior covered by current API contracts, docs, or tests;
- missing regression tests for an already intended behavior;
- internal performance, layout, cache, or allocation problems;
- hidden materialization or avoidable clones where the correct boundary is
  already defined by `REPOSITORY_RULES.md`;
- device-placement validation that enforces an existing backend contract.

### Verify First

Inspect and narrow before editing when a finding is broad, historical,
partially fixed, or phrased as coverage debt without a concrete failing path.
The agent must either:

- narrow it to a current source path and expected behavior,
- remove it from the active batch with evidence that it is stale, or
- move it to a design-gated issue if the intended behavior is unclear.

Do not keep broad historical wording as an active task after a narrower current
state is known. Replace it with the concrete failing path or mark it stale.

### Design Gate

Do not implement directly when a finding requires any of the following:

- new public API or removal of a public API with compatibility impact;
- crate-boundary or publishability changes;
- new operation family, backend, dependency, or feature flag;
- AD semantics policy that current docs/tests do not define;
- coverage-policy changes rather than ordinary test additions;
- changing backend support claims beyond matching docs to current behavior;
- roadmap, governance, or maintainer-priority decisions.

For design-gated findings, draft or update a focused issue or design document
instead of stretching the remediation PR.

## Batch And Commit Rules

- Prefer batches grouped by subsystem and verification path, such as
  docs/rustdoc, AD correctness, runtime/cache/performance, or GPU placement.
- Keep all approved batch work on one branch and in one PR.
- Split commits by reviewable unit, for example:
  - docs and rustdoc wording;
  - one AD rule family plus tests;
  - one runtime/cache optimization plus tests;
  - one GPU placement validation fix plus tests.
- Do not mix unrelated design-gated work into an auto-fix commit.
- Do not add `Closes #...` for an umbrella issue unless the PR completes every
  item tracked by that issue.

## Neighborhood Scan Rule

When fixing a problem, lightly search for the same problem nearby before
committing.

Search at least:

- the touched file;
- the same module or operation family;
- the same trait or generated wrapper family;
- adjacent docs section or guide;
- equivalent CPU/GPU/eager/traced wrappers when the pattern is shared.

Fix nearby same-root-cause instances in the same batch when they have the same
contract and verification path. Stop expanding when the search reaches another
subsystem, requires a different policy decision, or would dominate the review.
Record deferred related findings as residual risk or follow-up issues.

Historical `docs/plans/` material is normally out of remediation scope unless
an active document links it as current guidance. Fix active links or active
summaries instead of rewriting archived plans.

## Subagent Coordination

Use subagents for independent inspection or implementation domains when the
user authorizes subagent work.

- Prefer read-only explorer subagents for classification across independent
  domains such as docs, AD, runtime/performance, and GPU placement.
- Use worker subagents only for disjoint write scopes.
- Tell workers they are not alone in the codebase and must not revert unrelated
  edits.
- The coordinating agent owns integration: review subagent diffs, run the
  relevant verification, update the classification ledger, and decide the next
  batch.
- Subagent results are evidence to inspect, not a substitute for final
  verification.
- Close completed subagents once their result has been integrated.

## Interactive Session Rule Improvements

When working in an interactive user session, the agent should propose useful
improvements to these remediation rules instead of silently changing agent
authority.

If the agent notices a rule improvement while preparing or applying fixes, it
must tell the user:

- the concrete rule text or summary;
- why it would help future automated fixes;
- whether it changes agent authority or only clarifies existing behavior;
- any tradeoff or risk.

The agent may directly fix typos, broken links, or wording that does not change
behavior. Any change that expands scope, changes PR or merge behavior, changes
verification requirements, or changes what agents may automatically modify
requires user approval first.

In non-interactive or headless runs, record rule-improvement suggestions in the
PR body or work log instead of applying them silently.

If the user explicitly authorizes autonomous rule growth for a remediation
session, the agent may apply rule improvements that make future automatic
resolution safer or more precise. Such updates must stay within the user's
approved objective, must not silently expand project policy, and must be
committed as their own coherent review unit or clearly separated in the PR.

## Category-Specific Fix Rules

### Public Surface

- Prefer private or `pub(crate)` items for implementation details.
- Treat `#[doc(hidden)] pub` as public API unless it is a deliberate
  sibling-crate bridge with a documented contract.
- Do not remove or reshape public APIs with compatibility impact without a
  design-gated decision.
- If an item remains public, document its contract and add examples when
  required by `AGENTS.md`.

### Rustdoc And Examples

- Every public type, trait, function, and method needs a minimal `# Examples`
  block unless it is no longer public.
- Examples must compile and run as doctests. Do not use `ignore` or `no_run` to
  bypass missing setup.
- Examples must use real public symbols. Do not leave placeholder pointers,
  fake imports, or nonexistent facade crates.

### AD Semantics

- Start from current primal semantics and the machine-readable AD support
  manifests before changing rules.
- `linearize` and `transpose_rule` must preserve arity: return one tangent or
  cotangent slot per primal input, using `None` for nondifferentiable inputs.
- Add targeted regression or oracle coverage for the reported edge case before
  or alongside the fix.
- Cover relevant zero, tie, boundary, rectangular, repeated-label, complex,
  mixed real/complex, and dtype-conversion behavior.
- Project complex cotangents back to real tangent space for real inputs when
  required by the mathematical adjoint.
- Do not synthesize hidden host tensors for AD seeds, missing tangents, or
  constants on device paths.
- If the correct AD convention is not established, use the design gate.

### Runtime, Performance, Layout, And Cache

- Preserve column-major semantics and metadata-only view behavior.
- Avoid hidden materialization, repeated full-program scans, avoidable large
  clones, and string/debug fingerprints in hot paths.
- Prefer stable structural keys, one-pass summaries, explicit ownership
  transfer, workspace reuse, and backend-aware allocation.
- Cache keys must preserve exact equality. Do not replace string/debug keys
  with hash-only identifiers unless collisions are still resolved by structural
  equality or an equivalent payload-equality contract.
- When replacing a hot-path helper, remove obsolete private helpers instead of
  keeping artificial references only to suppress dead-code warnings.
- Add focused tests, assertions, benchmarks, or complexity checks when
  practical.

### GPU And Device Placement

- Validate input residency, runtime/device ordinal, dtype support, and backend
  support before zero-sized-output or early-return shortcuts.
- Eager helper-generated constants, indices, shape scalars, AD seeds, and
  missing tangents must be backend-aware or explicitly uploaded through the
  runtime contract.
- Unsupported CUDA paths must fail with clear placement/backend diagnostics
  rather than silently falling back to host behavior.
- HIP/ROCm remains stubbed unless a separate accepted backend feature issue
  says otherwise.

### Active Docs And Specs

- Active docs must match current crate boundaries, public imports, feature
  gates, backend support, and CI coverage.
- Historical design docs may remain, but they must be clearly marked
  historical and must not be linked as the current implementation direction.
- Source code and machine-readable registries are authoritative over stale
  prose.

### Coverage Policy

- Add tests for touched code paths.
- Do not lower thresholds just to pass CI.
- If changing the repository coverage policy or ratchet, use the design gate.

## Verification

Run focused checks first, then broader checks as the batch stabilizes.

Typical focused checks:

```bash
cargo fmt --all --check
cargo test -p <crate> <test-name>
cargo test -p <crate>
cargo doc -p <crate> --no-deps
```

Before opening the PR, run the repository-required checks from `AGENTS.md` when
the environment supports them:

```bash
cargo fmt --all --check
cargo test --workspace --release
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

For CUDA/GPU work, run the documented CUDA ignored tests when hardware and
libraries are available. If they are unavailable, add CPU-side or ignored CUDA
regression coverage where practical and state the exact gap in the PR body.

## Repository-Rules Side Review

Before creating the PR:

1. Re-read `REPOSITORY_RULES.md`.
2. Review the local diff against public surface, AD, performance/layout/cache,
   GPU placement, docs, tests, and work-log requirements.
3. Fix any findings discovered by the side review.
4. Document residual risks only when they are outside the approved batch or
   require a design-gated decision.

## PR Timing And Body

Create the PR only after the approved batch is locally complete and verified.
Use `.github/pull_request_template.md` and include:

- base hash and branch name;
- summary of the batch;
- issue or finding list with status;
- commit grouping rationale when useful for review;
- neighborhood scans performed;
- tests and commands run;
- skipped checks and why;
- `REPOSITORY_RULES.md` side-review outcome;
- residual risks and design-gated follow-ups.

Use `gh pr create` for PRs to `main`. After creating the PR, enable auto-merge
with a non-squash method:

```bash
gh pr merge --auto --merge --delete-branch
```

If branch protection or repository settings prevent non-squash auto-merge,
state the blocker and ask maintainers how to proceed. Do not switch to squash
merge silently.
