# Strided-Einsum2 Removal

## Session Summary

Removed the `strided-einsum2` dependency from `tenferro-cpu` by moving the
dot-general-specific Faer preparation algorithm into tenferro. The replacement
keeps the optimized GEMM structure: canonical dot-general axis grouping,
metadata stride analysis, operand-local col-major copies only when an operand
cannot be represented as strided GEMM input, and batched Faer GEMM dispatch.

## Context Read

- `AGENTS.md`
- `REPOSITORY_RULES.md`
- Shared tensor4all common, Rust, performance, and numerical rules
- `docs/superpowers/specs/2026-06-23-strided-einsum2-removal-design.md`
- `docs/superpowers/plans/2026-06-23-strided-einsum2-removal-implementation.md`
- `crates/tenferro-cpu/src/gemm/mod.rs`
- `crates/tenferro-cpu/src/gemm/strided_dot.rs`
- tenferro-pinned `strided-einsum2` `dot_general.rs`, `plan.rs`,
  `contiguous.rs`, and `bgemm_faer.rs`

## Decisions Made

- Did not port public `strided-einsum2` APIs, trace-axis reduction, BLAS
  provider selection, or generic binary einsum. `tenferro-einsum` continues to
  own public einsum parsing and planning.
- Added `gemm/faer_prepared.rs` as an internal Faer dot-general preparation
  module. It computes the same canonical grouping used by the prior adapter:
  lhs `[free..., contract..., batch...]`, rhs `[contract..., free..., batch...]`,
  and output `[lhs_free..., rhs_free..., batch...]`.
- Preserved the important performance property from `strided-einsum2`: inputs
  that can be represented as fused strided GEMM operands are passed directly to
  Faer; only non-fusable operands are copied into pooled col-major temporaries.
- Kept BLAS behavior on the existing tenferro path. The dependency removal only
  changes Faer `dot_general` routing and manifest feature forwarding.
- Removed `strided-einsum2` from workspace dependencies, `tenferro-cpu`
  features, provider feature contract tests, and v0.1 publish prerequisites.

## Verification Performed

- `cargo fmt --all --check`
- `cargo test -p tenferro-cpu --features cpu-faer`
- `cargo check -p tenferro-cpu --no-default-features --features cpu-blas`
- `cargo package -p tenferro-cpu --allow-dirty`
  - This now fails at unpublished `tenferro-tensor`, not at `strided-einsum2`.
- `rg "strided-einsum2|strided_einsum2|strided_dot" Cargo.toml Cargo.lock crates/tenferro-cpu docs/worklogs/2026-06-22-v0.1-publish-readiness.md`

## Remaining Risks

- The new Faer path intentionally rebuilds lightweight preparation metadata per
  call, matching the previous `strided-einsum2` adapter behavior. A future
  optimization can add a tenferro-owned prepared-plan cache if benchmarks show
  metadata cost is visible.
- Full workspace tests, coverage, rustdoc, and docs-site checks were not run in
  this release-prep pass.
