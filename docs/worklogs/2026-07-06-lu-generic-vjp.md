# LU Generic VJP

## Summary

This session removed the handwritten primary-graph LU transpose rule after
confirming that traced VJP already builds the LU cotangent through the generic
`linearize -> linear_transpose -> optimize` path.

The replacement is not justified by numerical agreement alone. A new structural
regression test snapshots the optimized LU gradient graph for square, wide, and
tall matrices:

- square LU: 12 ops total, including 2 triangular solves and 3 matmuls
- wide/tall LU: 13 ops total, including 2 triangular solves and 4 matmuls
- no `Extension(Lu)` transpose carrier remains in the final VJP graph
- no multi-output op remains in the final VJP graph

These counts match the operation-family structure of the handwritten LU rule
introduced by `cb6aff6c` (`Add LU and QR transpose AD rules`) while letting LU
use the same generic transform pipeline as other linearizable decompositions.

## Reference Code Consulted

- `cb6aff6c Add LU and QR transpose AD rules`
- `crates/tenferro-linalg/src/ad/rules/decomposition_transpose.rs`
- `crates/tenferro-linalg/src/ad/rules/mod.rs::linearize_lu`
- `crates/tenferro-ad/src/traced.rs::compute_linear_vjp_transform`
- `crates/tenferro-ad/src/traced/optimizer.rs`
- `docs/worklogs/2026-07-04-issue-1256-ad-vjp-default.md`

## Decisions

- Delete `transpose_lu` and its LU-only primal-output helper.
- Keep QR's handwritten transpose rule. This change only proves the LU case.
- Mark LU direct transpose support as unsupported in the linalg AD manifest.
  LU remains differentiable through its partially supported linearize rule and
  generic VJP.
- Keep finite-difference tests for LU/QR gradients, but add a separate
  structural test because value equality is not enough to prove useful graph
  simplification.

## Rejected Alternatives

- Do not add LU-specific optimizer peepholes. The optimized generic LU VJP is
  already compact enough.
- If this regresses later, fix it as a generic AD graph optimization problem:
  extend operation legality hooks, value-use DCE, or algebraic canonicalization
  over the transform graph. Do not reintroduce an LU-only shortcut unless the
  generic pipeline cannot represent the rule.

## Verification

- `cargo test -p tenferro-linalg --features autodiff --test traced_ad_explicit lu_sum_grad_optimized_graph_is_structurally_compact`
- `cargo test -p tenferro-linalg --features autodiff --test traced_ad_explicit lu_qr_sum_grads_match_finite_diff`
- `cargo test -p tenferro-linalg --features autodiff --test ad_support_manifest`
- `cargo test -p tenferro-linalg --features autodiff --lib`
- `cargo test -p tenferro-linalg --features autodiff --test traced_ad_explicit`

## Remaining Risks

- The structural snapshot is intentionally scoped to the LU `grad(sum(l) +
  sum(u))` family for square, wide, and tall matrices. Other LU observables may
  need their own structural snapshots if they become public acceptance criteria.
- The test summarizes the optimized VJP transform graph, not backend-lowered
  execution IR. Runtime compiler regressions should be caught by separate
  compiler/lowering tests.
