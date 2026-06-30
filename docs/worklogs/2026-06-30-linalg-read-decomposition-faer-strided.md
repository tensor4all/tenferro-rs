# 2026-06-30 Linalg read-view decomposition family + faer strided fast path

## Summary

Completes the borrowed-view `_read` execution-boundary API for the linalg
decomposition family and makes the faer CPU path consume strided 2-D views
directly instead of materializing a contiguous copy. Direct continuation of the
merged #1248 (`svd_read`/`qr_read`/`eigh_read`); maintainer-directed, AI-assisted
implementation.

Three coordinated changes:

1. Unify the dtype-error helper (`unsupported_view_dtype` in `gpu/mod.rs` was a
   byte-identical duplicate of `unsupported_dtype`); GPU `_read` methods adopt the
   CPU grouped match arm.
2. faer buffer-avoidance: on `CpuBackend` with the faer provider, a 2-D host view
   with non-negative strides is wrapped as a strided `faer::MatRef` and fed to the
   same decomposition core. faer copies its input into a packed workspace anyway,
   so the strided→packed gather is absorbed there — one copy instead of two
   (`to_contiguous` gather + faer workspace copy).
3. Add `cholesky_read`/`lu_read`/`full_piv_lu_read`/`eig_read` across trait
   default, `CpuBackend`, `CudaBackend`, and `EagerBackend`, matching the merged
   `svd_read`/`qr_read`/`eigh_read` conventions.

Out of scope by design: the solve family (`solve`/`triangular_solve`/
`full_piv_lu_solve`, multi-input signature) and internal `#[doc(hidden)]` ops.

## Context read

- The merged #1248 `svd_read`/`qr_read`/`eigh_read` (the pattern this mirrors):
  trait default impl, `CpuBackend`/`CudaBackend` overrides, `EagerBackend`
  dispatch, and their tests.
- `cpu/linalg/faer_linalg.rs`: the `*_2d` cores and the
  `impl_faer_linalg_for_real!` / `impl_faer_linalg_for_complex!` macros; the
  `MatRef::from_column_major_slice` + workspace `copy_from` pattern.
- `tenferro-tensor` `TypedTensorView` accessors (`strides`/`offset`/
  `host_storage`/`backend_buffer`) and `CpuBackend::to_contiguous` (host-placement
  rejection).
- `REPOSITORY_RULES.md`: Faer Integration (prefer zero-copy `MatRef`),
  Materialization And Copies (avoid dense copy-in around strided-capable ops),
  Unsafe Code Boundary (backend-leaf only, `// SAFETY:` notes), Public Surface
  Discipline (`_read` suffix for borrowed-view inputs), AD Rule Coverage (this
  change touches no AD rule, so the Oracle Gate does not apply).

## Design decisions

- **MatRef-in refactor over additive `*_view` duplication.** Faer output helpers
  are templated on `&Placement`, and each `*_2d` was split into a `*_core` taking
  a `faer::MatRef`. Both the compact path (`faer_mat_ref_compact`) and the strided
  path (`faer_mat_ref_strided`) feed the same core, so there is no duplicated
  decomposition logic. Bodies live once per real/complex macro.
- **Strided fast path is strictly additive.** Gated by
  `#[cfg(feature = "cpu-faer")]` + `matches!(self.kind(), CpuBackendKind::Faer)` +
  `faer_strided_ok(view)` (host buffer, rank 2, all strides ≥ 0). Any other case
  (BLAS, GPU, batched/rank≠2, negative strides, feature off) falls through to the
  unchanged `to_contiguous` → owned-`Tensor` path. The documented "no silent
  CPU↔GPU transfer" contract holds: GPU-buffer views fail the predicate and are
  rejected by `to_contiguous` as before.
- **eig has no fast path (fallback only).** `eig` is non-Hermitian and real input
  yields complex output (`Vec<TypedTensor<Complex>>` from a real `TypedTensor`),
  so a generic `eig_core` over `FaerLinalg<T>` is not type-expressible. `eig_read`
  materializes and calls `self.eig`, producing the same complex public tensors as
  the slow path. Documented in code.
- **cholesky is single-output** (`Result<Tensor>`); its fast path returns the
  `Tensor` directly.

## Alternatives rejected / deferred

- Additive `*_2d_view` siblings beside the existing `*_2d`: rejected — duplicates
  each decomposition body.
- Batched (rank > 2) strided fast path: deferred — the batch stride adds
  complexity for uncertain benefit; batched views still work correctly via the
  `to_contiguous` fallback. The 2-D matrix case is the documented contract.
- Lowering `coverage-thresholds.json` for the touched files: not done; prefer real
  fallback-path tests if CI coverage flags the additive arms.

## Verification

- `cargo test -p tenferro-linalg` (default faer build): lib 62, doctests 29,
  integration suites (backend_errors, gpu_linalg_source_contract incl. 7 `_read`
  contract tests, full_piv_lu, traced_correctness, …) all pass, 0 failures.
- `cargo clippy -p tenferro-linalg --all-targets -- -D warnings`: clean.
- `cargo fmt --all --check`: clean.
- The pre-existing `*_canonicalizes_transposed_*` reconstruction tests now route
  through the strided fast path (transposed view → positive strides →
  `faer_strided_ok`), so the optimization is covered by already-trusted numerical
  reconstruction, plus new `*_faer_strided_*` regressions for svd/qr/eigh and
  factor-reconstruction tests for lu/full_piv_lu.

## Residual risks

- **Coverage**: depending on the coverage build's CPU provider features, either
  the fast-path arms or the float `to_contiguous` fallback arms in
  `cpu/backend.rs` may be unexercised, possibly dropping the file below the 90%
  default. Fix by adding fallback-forcing tests (e.g. a rank-3 view) rather than
  padding — pending the CI coverage report.
- **faer generic-stride input**: a transposed view has row stride ≠ 1, which is
  not faer's preferred contiguous layout. Justified: faer copies the input into a
  packed workspace before factoring, so the non-unit stride only affects that
  one unavoidable pack read, not the decomposition itself; the alternative is a
  full `to_contiguous` allocation + gather.
- **eig** gets no buffer-avoidance win (fallback only); acceptable and documented.
- Full-workspace `--release` tests, `llvm-cov` coverage, `cargo doc`,
  `check-docs-site.py`, repository-rules review, and the GPU lane are verified in
  CI rather than locally for this crate-scoped change.
