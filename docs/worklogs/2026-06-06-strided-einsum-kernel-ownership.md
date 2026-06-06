# Strided Einsum Kernel Ownership

## Session Summary

Moved CPU binary einsum support toward `strided-rs` ownership and cleaned up
tenferro's adapter layer. The work keeps tenferro responsible for dtype
dispatch, backend selection, buffer-pool allocation, and runtime value/view
semantics, while `strided-kernel` and `strided-einsum2` own broadcast
multiplication, batched outer product, and non-conjugated dot-general execution.

## Context Read

- `AGENTS.md`
- `REPOSITORY_RULES.md`
- Shared tensor4all common repository, performance, docs/tests, and Rust
  performance rules
- `crates/tenferro-cpu/src/elementwise.rs`
- `crates/tenferro-cpu/src/gemm/mod.rs`
- `crates/tenferro-cpu/src/gemm/strided_dot.rs`
- `../strided-rs/strided-kernel/src/map_view.rs`
- `../strided-rs/strided-kernel/src/outer_product.rs`
- `../strided-rs/strided-einsum2/src/dot_general.rs`

## Decisions Made

- `strided-kernel::batched_outer_product_into` is a semantic wrapper over the
  general `broadcast_mul_into` path. This keeps explicit outer-product calls
  and equivalent broadcasted multiplication on one implementation path.
- tenferro's `broadcast_multiply_read_with_pool` now uses
  `typed_broadcast_mul_view_with_pool` for the fallback path instead of
  materializing broadcast views before calling `mul`.
- Removed the tenferro `cpu-blas` outer-product GEMM shortcut. BLAS remains the
  provider for `dot_general`; elementwise and broadcast multiply go through
  `strided-kernel` for all CPU provider selections.
- `strided_einsum2::DotGeneralConfig` borrows axis slices. The tenferro adapter
  passes existing `DotGeneralConfig` vectors as slices instead of cloning them
  into a second owned config.
- Kept current non-conjugated faer dot-general routing through
  `strided_einsum2`; conjugated dot-general still uses the existing tenferro
  materialization/canonicalization fallback until strided-einsum2 exposes a
  deliberate conjugating API.

## Rejected Or Deferred Alternatives

- Did not add more shape-specific einsum fast paths. Current benchmark analysis
  does not show a kernel/executor loss above the configured threshold; the
  remaining large gap is classified as a path/intermediate issue.
- Did not keep a BLAS outer-product special case in tenferro. It duplicated
  elementwise ownership and made behavior depend on CPU provider features.
- Did not add a prepared dot-general cache to `strided-einsum2` yet. The current
  adapter overhead should be benchmarked before adding another cache layer.
- Did not lower N-ary einsum graphs to expanded dot/reduce/transpose graphs in
  this pass. That remains a path-planning/compiler optimization, not a kernel
  patch.

## Verification Performed

- `cargo fmt --check` in `../strided-rs`
- `cargo test -p strided-einsum2 --features parallel,blas-accelerate --no-default-features`
- `cargo test -p strided-kernel --features parallel`
- `cargo check -p strided-einsum2 --features parallel,blas-inject --no-default-features`
- `cargo check -p strided-einsum2 --features parallel,blas-mkl --no-default-features`
- `cargo fmt --all --check`
- `cargo test -p tenferro-cpu --features cpu-faer -- --nocapture`
- `cargo test -p tenferro-cpu --no-default-features --features cpu-blas,src-accelerate -- --nocapture`
- `uv run python scripts/analyze_einsum_gaps.py --report result/cpu/einsum.md --instances data/instances --threshold 1.15` in `../tenferro-benchmark`

## Remaining Risks

- The latest saved benchmark still has `gm_queen5_5_3.wcsp` with
  `opt_flops` slower than PyTorch, but the analyzer classifies it as
  `path_intermediate`. A focused `TENFERRO_ANALYZE_PATH=1` run reported a
  rank-17 maximum intermediate, `129140163` maximum intermediate elements, and
  `592127821` total intermediate elements for `opt_flops`; the `opt_size` path
  caps the maximum intermediate at `43046721` elements. More kernel-level
  optimization is not justified by the current evidence.
- `strided_einsum2` does not yet expose a conjugating dot-general API, so
  tenferro's conjugated CPU faer path still uses the existing fallback.
- Full workspace release, coverage, rustdoc, and docs-site gates were not run
  in this session.
