# Issue 981 BLAS Provider Features

Date: 2026-06-11

## Summary

Implemented the accepted issue to expose explicit BLAS provider features through
the tenferro CPU-facing crates. The change keeps `cpu-blas` as the generic
CBLAS/LAPACK backend while adding provider-specific Cargo features that also
select the matching `strided-einsum2` provider.

## Context Read

- `AGENTS.md`
- shared tensor4all rules:
  - `rules/index.md`
  - `rules/common/repository.md`
  - `rules/common/docs-and-tests.md`
  - `rules/common/performance.md`
  - `rules/rust/index.md`
  - `rules/rust/performance.md`
- `REPOSITORY_RULES.md`
- `CONTRIBUTING.md`
- GitHub issue #981
- Current Cargo feature manifests for `tenferro-cpu`, `tenferro-runtime`,
  `tenferro-ad`, `tenferro-einsum`, `tenferro-linalg`, `tenferro-fft`, and
  `tenferro-gpu`
- Cargo metadata for `strided-einsum2`, `blas-src`, `cblas-src`, and
  `lapack-src`

## Decisions

- Added public provider features named `blas-openblas`, `blas-accelerate`, and
  `blas-mkl`.
- Kept `cpu-blas` as the generic CBLAS/LAPACK feature for users who link a
  provider through their system or application build environment.
- Made each explicit provider feature enable:
  - `provider-src`
  - `dep:strided-einsum2`
  - the matching `blas-src` provider feature
  - the matching `lapack-src` provider feature
  - the matching `strided-einsum2` provider feature
- Preserved the legacy `src-openblas`, `src-accelerate`, and
  `src-intel-mkl-dynamic-parallel` feature names as aliases.
- Added compile-time guards so multiple explicit BLAS providers cannot be
  enabled together through Cargo feature unification. `provider-inject` is also
  rejected when combined with the explicit source-provider features.
- Added provider passthrough features to the CPU-using public crates using the
  repository's existing feature-forwarding style.
- Updated user docs and the MPS benchmark helper to use the explicit provider
  feature names. The benchmark helper now targets `tenferro-einsum`, which owns
  the MPS benchmark targets in the current workspace.

## Rejected Or Deferred

- Did not change the default CPU backend behavior; default features still use
  `cpu-faer`.
- Did not remove legacy `src-*` features because existing scripts and downstream
  users may still reference them.
- Did not make `cpu-blas` itself select a concrete provider. The issue calls out
  that plain `blas` remains the generic CBLAS backend on the strided side, and
  the same distinction is preserved in tenferro.
- Did not build-test every explicit source provider locally because OpenBLAS,
  MKL, and Accelerate linking is platform and environment dependent. Feature
  resolution is covered by contract tests and `cargo metadata`.

## Verification

- `cargo test -p tenferro-cpu --test provider_feature_contract` failed before
  the implementation and passed after the feature changes.
- `cargo metadata --format-version 1 --no-default-features --features blas-openblas`
  showed `strided-einsum2` features `blas` and `blas-openblas`.
- `cargo check -p tenferro-runtime --no-default-features --features blas-openblas --lib`
- `cargo check -p tenferro-cpu --no-default-features --features cpu-blas --lib`
- `cargo check -p tenferro-einsum --no-default-features --features cpu-blas --benches`
- `bash -n scripts/bench-mps-inner-product.sh`
- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`

## Residual Risks

- Source-provider link behavior still depends on platform libraries and provider
  build-script environment variables such as `OPENBLAS_LIB_DIR`, `MKLROOT`, and
  `MKL_LIB_DIR`.
