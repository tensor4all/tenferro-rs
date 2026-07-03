# Issue 1285 Dot-General Accumulation

## Session Summary

Implemented the first pass of issue #1285: a dot-general accumulation backend
contract for `out = alpha * op(lhs) * op(rhs) + beta * out`, with CPU faer/BLAS
native wiring and fallback composition for backends that do not override the
new hook.

## Context Read

- Shared tensor4all rules: common repository, performance, docs/tests, Rust
  performance, and Rust numerical rules.
- Repository rules in `REPOSITORY_RULES.md`, especially public-surface
  discipline, hidden backend hook behavior, materialization boundaries, CPU
  GEMM ownership, cache ownership, and work-log requirements.
- Issue #1285 and the existing `TensorDot`, `SessionCachedDot`, and
  `BackendCachedDot` separation.
- Claude side review of the issue proposal. It agreed the feature was valid,
  but that putting `cache_slot` directly on the non-cached `TensorDot` API would
  be inconsistent and that a separate config-style accumulation contract was
  cleaner.
- CPU GEMM helpers in `crates/tenferro-cpu/src/gemm/`, including the faer and
  BLAS raw strided GEMM traits that already accepted `alpha`, `beta`, and
  conjugation flags.

The private `tensor4all-agent-knowledge` checkout was unavailable in this
workspace, so no private knowledge rules were loaded.

## Decisions Made

- Added `DotGeneralAccumulation` instead of extending `DotGeneralConfig`.
  `DotGeneralConfig` remains the dimension-role contract; accumulation carries
  conjugation flags and output update coefficients.
- Added `ContractionScalar` instead of a broad scalar enum. It intentionally
  covers only `F32`, `F64`, `C32`, and `C64`, matching GEMM-style contraction
  accumulation.
- Preserved the cached/non-cached trait split. `TensorDot` has the non-cached
  accumulation method; `SessionCachedDot` and `BackendCachedDot` own the cached
  variants and `cache_slot`/runtime-cache plumbing.
- Kept `dot_general_read_into` as overwrite shorthand by delegating to
  accumulation with `alpha = 1`, `beta = 0`, and no conjugation.
- Added `TensorWrite::as_read()` for explicit read-modify-write callers.
- Wired CPU faer and BLAS accumulation through existing `GemmAnalysisCache` and
  provider GEMM traits. Output views can use native paths when their output
  dimensions fuse to GEMM row, column, and batch strides; otherwise the explicit
  temp fallback remains available.
- Preserved BLAS/faer `beta = 0` semantics: the output element is not read when
  beta is zero.

## Rejected Or Deferred Alternatives

- Rejected placing `cache_slot` directly on `TensorDot`; that would mix cache
  ownership into the non-cached backend trait.
- Rejected placing `alpha`, `beta`, and conj flags directly into
  `DotGeneralConfig`; that would mix contraction shape semantics with output
  update semantics.
- Deferred CUDA/cuTENSOR native accumulation. The current CUDA path mainly
  handles owned compact tensors and needs separate pointer/offset work for full
  `TensorRead`/`TensorWrite` parity.
- Deferred WebGPU native accumulation. It can continue through the explicit
  fallback unless and until the backend grows native alpha/beta support.

## Verification Performed

- `cargo test -p tenferro-tensor`
- `cargo test -p tenferro-cpu`
- `cargo check --workspace`
- `cargo check -p tenferro-cpu --features cpu-blas`
- `cargo fmt --all --check`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD --output-json /tmp/repository-rules-review.json`

Focused tests cover `TensorWrite::as_read()`, accumulation dtype validation,
`F32`/`F64`/`C32`/`C64` coefficients, complex conjugation, cached session
dispatch, and strided output accumulation.

## Remaining Risks

- CUDA native accumulation is still a follow-up. GPU backends compile through
  the fallback trait path but do not yet wire cuTENSOR alpha/beta.
- BLAS execution with an actual linked provider was compile-checked through
  `cpu-blas`; provider-backed runtime execution remains covered by the normal
  provider-specific CI lanes.
