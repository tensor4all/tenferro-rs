# Terasaki Bug Batch 2

## Summary

Fixed a second bounded batch of recent `terasakisatoshi` reports after the
previous batch merged. The implemented fixes focus on low-risk shared patterns:
AD transpose metadata for `BroadcastInDim`, rank-changing reshape preservation
in compiler layout simplification, checked fused-dimension products in einsum
planning, saturating cache retained-byte accounting, and CubeCL raw device
pointer runtime-residency checks.

The PR also updates the repository remediation rule for false positives:
intentional invariants must be recorded in the issue or PR ledger, and unclear
invariants should get a nearby comment, rustdoc note, or source-contract test
so later humans and AI agents do not rediscover the same non-bug.
As a follow-up, the `tenferro-bugfix-pr` workflow and skill adapters now always
read the remediation workflow, route related bug batches to it, and require
same-root-cause scans, non-squash batch PRs, audit-rule proposals, and
false-positive source comments or source-contract tests when appropriate.
Codex, Claude Code, OpenCode, and Kimi CLI entry points are kept in sync, and
new audit-rule proposals must inventory and merge overlapping existing rules
where possible.

## Context Read

- `AGENTS.md`, `CONTRIBUTING.md`, `REPOSITORY_RULES.md`
- `ai/contribution-workflows/repository-remediation.md`
- Previous work log `docs/worklogs/2026-06-17-terasaki-bug-batch.md`
- Mini-agent audit slices for tensor shape/arithmetic, AD/eager metadata,
  runtime/compiler/cache, and GPU/FFI contracts
- Affected modules in `tenferro-internal-ops`, `tenferro-runtime`,
  `tenferro-einsum`, and `tenferro-gpu`

## Decisions

- Fixed #1092 by making the `BroadcastInDim` transpose rule restore the
  cotangent axis order after reducing broadcast-only axes. Non-monotonic
  `dims` now produces the required transpose before reshape.
- Fixed #1107 in the algebraic layout simplifier by requiring known input rank
  metadata before removing reshapes. Rank-reducing reshapes are no longer
  treated as identity layout operations.
- Treated the `dot_conj_folding` part of #1107 as a false positive: the pass
  moves `Conj` through transparent layout ops but keeps the `DotGeneral`
  operand wired to the reshape output. Added a source-contract test and a
  nearby comment to prevent future misclassification.
- Fixed #1090 by replacing unchecked products for fused einsum matrix
  dimensions with checked products in both pairwise planning and strict binary
  lowering.
- Fixed #1108 by making runtime graph cache retained-byte accounting saturate
  instead of wrapping or panicking. The follow-up audit found the same
  retained-byte pattern in `tenferro-einsum` extension caches, so those
  estimates now share saturating helpers too.
- Fixed the low-risk raw-pointer part of #1088 by validating CubeCL tensor
  residency before exposing raw device pointers to interop or cuTENSOR GEMM
  FFI. Each raw pointer path carries a one-line invariant comment.
- Updated `ai/contribution-workflows/bugfix-pr.md` plus Codex, Claude Code,
  OpenCode, and Kimi `tenferro-bugfix-pr` adapters so future ordinary bug-fix
  agents read the remediation workflow and repeat the batch pattern when a user
  asks for related issues in one PR.
- Added the audit-rule inventory rule: before adding a new audit or repository
  rule, check nearby existing rules and merge, tighten, or relocate overlapping
  guidance when possible.
- Kept the batch scoped to single-PR minor fixes. Reports that require broad
  fallible API changes, poisoning policy changes, or unsafe/FFI lifecycle
  redesign are classified as design-gated instead of being partially rewritten.

## Classification Ledger

- Fixed in this PR: #1090, #1092, #1107 layout-simplifier case, #1108 retained
  byte accounting, and the raw CubeCL pointer residency portion of #1088.
- False positive with source-contract coverage: #1107 `dot_conj_folding`
  reshape bypass concern.
- Stale or already fixed on current `origin/main`: #1089, #1094, #1095, #1097,
  #1100, and the cache-key payload side of #1108.
- Design-gated follow-up: #1082 and #1098 need fallible `DimExpr` and shape
  evaluation APIs; #1093, #1102, #1103, and broad #1084 need a public
  panic/poison policy sweep; #1096 needs an explicit policy for legacy
  `extension::apply`; #1099 needs an unsafe-proof/design update for borrowed
  slot workspaces; #1091, #1105, #1106, and #1109 need a broader GPU FFI and
  lifecycle review.

## Audit Notes

- `rg` audit for retained-byte arithmetic found unchecked `sum::<usize>()`,
  `capacity() *`, and `+` chains in runtime graph cache and einsum extension
  cache estimates. Runtime and einsum cache estimates now saturate.
- `rg` audit for einsum planning products found the fused-dimension products
  in pairwise and strict binary planning. Both now return `InvalidArgument` on
  overflow.
- The remaining runtime graph/ad-support boolean zero/one tensor products are
  design-gated because making them fallible affects public tangent/default-input
  APIs and the broader #1082/#1084 shape-arithmetic policy.
- The raw CubeCL pointer audit found two exposed paths. Both now validate
  runtime/device residency before obtaining backend-native resources.

## Verification

- `cargo fmt --all`
- `git diff --check`
- `cargo test -p tenferro-runtime`
- `cargo test -p tenferro-einsum`
- `cargo test -p tenferro-einsum --features autodiff`
- `cargo test -p tenferro-internal-ops`
- `cargo test -p tenferro-gpu --test cubecl_launch_contract`
