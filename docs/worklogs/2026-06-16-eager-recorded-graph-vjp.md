# Eager Recorded Graph VJP

## Summary

Implemented the tenferro side of the eager recorded-graph VJP design for issues
#1060 and #1061. The dependency PR in `tidu-rs` adds graph-invocation eager
recording; tenferro now routes primitive eager recording through one-op
`RecordedGraph`s and routes tracked n-ary eager einsum through one composite
`StdTensorOp` graph node.

## Context Read

- `crates/tenferro-ad/src/eager.rs`
- `crates/tenferro-ad/src/eager_exec.rs`
- `crates/tenferro-ad/src/eager/backward.rs`
- `crates/tenferro-ad/src/eager_tensor.rs`
- `crates/tenferro-einsum/src/eager_tensor.rs`
- `crates/tenferro-einsum/src/builder.rs`
- `docs/superpowers/specs/2026-06-16-eager-recorded-graph-vjp-design.md`
- `REPOSITORY_RULES.md`

## Decisions

- Kept `tidu` as the owner of tape structure and graph linearization. Tenferro
  only builds `RecordedGraph<StdTensorOp>` values and supplies concrete forward
  data.
- Added a narrow `tenferro_ad::eager_tensor::apply_standard_graph` hook for
  operation-family crates that need to execute and record a standard-op graph
  as one eager result.
- Executed tracked eager einsum forward by walking the lowered standard graph in
  one `BackendSession`, instead of replaying the old per-op eager API.
- Retained conservative O(N) residual data by saving the graph input values and
  every concrete value produced by the forward graph execution.
- Left untracked eager einsum's existing cached expanded-program path intact
  unless the existing whole-program env gate is enabled.

## Deferred

- `apply_standard_graph` is intentionally a standard-op graph hook. Extension
  ops inside that graph return an explicit missing-runtime error rather than
  trying to dispatch through an extension executor mid-session.
- Generated host ops (`Constant`, `ShapeOf`) are still handled by the existing
  single-op eager paths before entering a backend session. The new graph-session
  executor returns an explicit error for those ops.
- Residual tightening, checkpoint/rematerialization, and graph cache reuse for
  tracked graph-recorded einsum are follow-up optimizations.

## Verification

- `cargo test -p tenferro-ad --release standard_graph_op`
- `cargo test -p tenferro-einsum --features autodiff --release tracked_whole_program_einsum_records_one_graph_residual`
- `cargo test -p tenferro-ad --release eager_builder`
- `cargo test --workspace --release`
- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
