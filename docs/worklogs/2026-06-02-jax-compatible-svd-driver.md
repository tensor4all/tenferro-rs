# JAX-Compatible CUDA SVD Driver Selection

Date: 2026-06-02

## Session summary

Changed the CUDA SVD path to use JAX-compatible default driver selection:
`gesvdj` for concrete matrices with both dimensions at most 1024, and the
existing `gesvd` path for larger matrices. This targets the benchmark gap where
tenferro used QR-based `gesvd` for 256x256 CUDA SVD while JAX and PyTorch use
Jacobi `gesvdj` by default.

## Context read

- `REPOSITORY_RULES.md`
- Shared tensor4all common, Rust performance, and numerical rules
- Existing CUDA linalg implementation and source-contract tests
- JAX SVD lowering in `../jax/jax/_src/lax/linalg.py`
- JAX cuSOLVER FFI implementation in `../jax/jaxlib/gpu/solver_kernels_ffi.cc`
- Local cuSOLVER header `/usr/include/cusolverDn.h`

## Reference code

JAX resolves CUDA SVD defaults to Jacobi for `m, n <= 1024`, otherwise
`gesvd`. Its `gesvdj` lowering returns V, then transposes and conjugates it to
produce V^H. JAX also allocates U and V buffers for Jacobi even when
`compute_uv=false`; cuSOLVER rejects null U/V pointers in that path on this
host.

PyTorch documentation and local benchmark A/B showed the same practical default
for the 256x256 benchmark: `driver=None` matches `gesvdj`, while explicit
`gesvd` matches tenferro's previous slower path.

## Decisions made

- Added cuSOLVER `gesvdj` FFI and `gesvdjInfo` lifetime management inside the
  existing linalg CUDA FFI module.
- Kept the public SVD API unchanged. The driver choice is an internal backend
  policy, not a user-visible driver argument.
- Added an internal `select_svd_driver(m, n)` helper with JAX's 1024 threshold.
- For full SVD, `gesvdj` writes U and V; tenferro then copies V to V^H with a
  small CUDA kernel so the existing `vt` contract remains unchanged.
- For values-only SVD, `gesvdj` still allocates scratch U/V buffers and passes
  them to cuSOLVER, matching JAX's cuSOLVER contract handling.

## Rejected or deferred alternatives

- Did not add a public driver-selection API. That would widen the public
  surface for a backend policy change.
- Did not use `gesvda` as the default. PyTorch exposes it, but neither JAX nor
  PyTorch uses it as the normal exact SVD default.
- Did not route V-to-VT through runtime typed transpose helpers because CubeCL
  rejects borrowed tensor views at that execution boundary.
- Did not add `gesvdjBatched`; the current PR only changes the default driver
  selection needed for the observed benchmark mismatch.

## Verification performed

- `cargo fmt --all --check`
- `cargo check -p tenferro-linalg --features cuda`
- `cargo test -p tenferro-linalg gpu_svd_uses_jax_compatible_default_driver_selection`
- `CUBECL_DEBUG_LOG=0 CUDA_PATH=/usr/local/cuda-12.6 LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH cargo test -p tenferro-linalg --features cuda test_cubecl_svd -- --ignored --nocapture`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`

## Remaining risks

- The implementation does not yet benchmark the new path inside this PR. The
  prior benchmark comparison indicates `gesvdj` is the required driver for the
  256x256 gap, but fresh benchmark result updates can happen in the benchmark
  repository.
- Small batched SVD could be further optimized with `gesvdjBatched`, but that is
  separate from the JAX-compatible default driver policy.
