# Generic Decomposition VJP Cleanup

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

The same cleanup was then applied to `EighVals`, the values-only Hermitian
eigenvalue path used by `eigvalsh`. Historical `eigen_rrule` examples seeded
`values: ones` and `vectors: None`, which is the `sum(eigenvalues)` case. The
current values-only traced path now relies on `linearize_eigh_values` followed
by generic transpose instead of the direct `transpose_eigh_values` helper.

The optimized `grad(sum(eigvalsh(a)))` graph is also structurally snapshotted:

- 15 ops total
- 2 matmuls around embedded diagonal cotangents
- no `Extension(EighVals)` transpose carrier remains
- no internal `Extension(Eigh)` carrier remains in the final VJP graph
- no multi-output op remains in the final VJP graph

## Reference Code Consulted

- `cb6aff6c Add LU and QR transpose AD rules`
- `4e570a28 Add fast eigh VJP transpose path`
- `d0c57626 optimize linalg prepared and values-only paths`
- `crates/tenferro-linalg/src/ad/rules/decomposition_transpose.rs`
- `crates/tenferro-linalg/src/ad/rules/eigh.rs`
- `crates/tenferro-linalg/src/ad/rules/mod.rs::linearize_lu`
- `crates/tenferro-linalg/src/ad/rules/mod.rs::linearize_eigh_values`
- `crates/tenferro-ad/src/traced.rs::compute_linear_vjp_transform`
- `crates/tenferro-ad/src/traced/optimizer.rs`
- `docs/worklogs/2026-07-04-issue-1256-ad-vjp-default.md`

## Decisions

- Delete `transpose_lu` and its LU-only primal-output helper.
- Delete `transpose_eigh_values`. Values-only Hermitian eigenvalue reverse
  support is compact through generic VJP.
- Keep QR's and full Eigh's handwritten transpose rules. This change only
  proves LU and values-only Eigh.
- Mark LU direct transpose support as unsupported in the linalg AD manifest.
  LU remains differentiable through its partially supported linearize rule and
  generic VJP.
- Mark EighVals direct transpose support as unsupported in the linalg AD
  manifest. EighVals remains differentiable through linearize plus generic VJP.
- Keep finite-difference tests for LU/QR gradients, but add a separate
  structural test because value equality is not enough to prove useful graph
  simplification. Apply the same standard to `eigvalsh`.

## Rejected Alternatives

- Do not add LU-specific optimizer peepholes. The optimized generic LU VJP is
  already compact enough.
- Do not add an `EighVals`-specific optimizer peephole or special-case
  `sum(eigenvalues) -> identity` rewrite in this PR. The generic VJP matches
  the removed handwritten path's structure and remains small.
- If this regresses later, fix it as a generic AD graph optimization problem:
  extend operation legality hooks, value-use DCE, or algebraic canonicalization
  over the transform graph. Do not reintroduce operation-specific shortcuts
  unless the generic pipeline cannot represent the rule.

## Verification

- `cargo test -p tenferro-linalg --features autodiff --test traced_ad_explicit lu_sum_grad_optimized_graph_is_structurally_compact`
- `cargo test -p tenferro-linalg --features autodiff --test traced_ad_explicit eigvalsh_sum_grad_optimized_graph_is_structurally_compact`
- `cargo test -p tenferro-linalg --features autodiff eigvalsh_jvp_matches_finite_diff_through_values_only_eigh`
- `cargo test -p tenferro-linalg --features autodiff --test traced_ad_explicit lu_qr_sum_grads_match_finite_diff`
- `cargo test -p tenferro-linalg --features autodiff --test ad_support_manifest`
- `cargo test -p tenferro-linalg --features autodiff --lib`
- `cargo test -p tenferro-linalg --features autodiff --test traced_ad_explicit`

## Remaining Risks

- The structural snapshot is intentionally scoped to the LU `grad(sum(l) +
  sum(u))` family for square, wide, and tall matrices, and to the
  `grad(sum(eigvalsh(a)))` values-only Hermitian eigenvalue family. Other LU,
  Eigh, or eigenvector observables may need their own structural snapshots if
  they become public acceptance criteria.
- The test summarizes the optimized VJP transform graph, not backend-lowered
  execution IR. Runtime compiler regressions should be caught by separate
  compiler/lowering tests.
