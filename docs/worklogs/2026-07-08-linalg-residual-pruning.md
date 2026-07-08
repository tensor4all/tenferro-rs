# Linalg Residual-Aware Decomposition Pruning

Date: 2026-07-08

## Summary

This work reduces traced AD overhead for values-only decomposition users such
as spectral norm. Values-only traced helpers now build full decomposition
graphs so AD can reuse residual outputs, while runtime compilation prunes
forward-only full decompositions back to lighter values-only extension ops.

## Context Read

- `crates/tenferro-linalg/src/traced.rs`
- `crates/tenferro-linalg/src/ad/rules/mod.rs`
- `crates/tenferro-linalg/src/extension.rs`
- `crates/tenferro-runtime/src/graph/compiler.rs`
- `crates/tenferro-runtime/src/compiler/mod.rs`
- `crates/tenferro-internal-ops/src/ext_op.rs`
- `tidu::linear_transpose` in the pinned tidu checkout

## Decisions

- Runtime graph compilation now performs an extension-output pruning pass over
  the compiled SSA program before lowering to `ExecProgram`.
- The pruning pass uses `ExtensionOp::prune_outputs` as an operation-replacement
  hook, not just as a dead-output filter.
- `LinalgOp::Svd`, `Eigh`, and `Eig` prune to `SvdVals`, `EighVals`, and
  `EigVals` when only the values output is live.
- Traced `svd_values`, `eigh_values`, and `eig_values` helpers now construct the
  full decomposition and return only the values output. This lets AD reuse U/VT
  or eigenvectors from the primal graph instead of emitting another
  decomposition inside the derivative graph.

## Rejected Or Deferred

- Eager values-only decomposition residual hooks are not added because the
  current public eager linalg surface exposes full `svd`, `eigh`, and `eig`, not
  values-only helpers.
- Solve-transpose residual reuse is deferred. The relevant transpose rules need
  the primal solution output, but the current pinned `tidu::linear_transpose`
  does not pass operation-output primal keys through `ShapeGuardContext`. The
  sibling `tidu-rs` checkout is dirty with conflicts, so this PR avoids mixing
  an upstream tidu API change into the linalg pruning work.

## Verification

- `cargo test -p tenferro-runtime --lib`
- `cargo test -p tenferro-linalg --lib --tests`
- `cargo test -p tenferro-linalg --features autodiff --lib --tests`

