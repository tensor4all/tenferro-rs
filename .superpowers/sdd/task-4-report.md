# Task 4 report: Apple managed rank-2 Cholesky

## Scope implemented

- Added a guarded managed-storage branch only for rank-2 CPU `cholesky`.
- Kept the existing host and batched Cholesky path unchanged.
- Preserved the selected `CpuBackendKind`, `CpuContext`, faer threading policy,
  BLAS/LAPACK provider path, buffer-pool ownership, and compact column-major
  output semantics.
- Refactored the faer and LAPACK rank-2 numerical cores to return provider
  scratch as a compact lower-triangular `Vec<T>`; the managed path writes that
  data directly into a new allocation from the input's configured domain.
- Routed borrowed extension execution for Cholesky through `cholesky_read`, so
  concrete, eager, and traced public paths all reach the audited hook without
  generic CPU `to_contiguous_read` materialization.
- Limited the claim to rank-2 Cholesky. Other CPU linalg operations still use
  their existing host-only behavior.

## Storage and error behavior

- Requires matching allocation-domain identity, `MemoryKind::Managed`, compact
  column-major layout at offset zero, and full logical allocation coverage.
- Holds the read mapping only while the provider copies/factors the input, then
  drops it before allocating and write-mapping the result.
- Uses one full write-only `copy_from_slice` for the output.
- Rejects foreign domains and device-local backend buffers without an implicit
  upload or download.
- Supports the existing CPU Cholesky dtype contract: F32, F64, C32, and C64.

## TDD evidence

The new macOS Apple integration test was first run against the unmodified CPU
linalg path and failed at the expected boundary:

```text
RuntimeState { op: "cholesky", message: "CPU linalg backend received a backend buffer; download the tensor to host before CPU execution" }
```

After the guarded implementation, the same test passes through concrete,
eager, and traced execution.

## Verification

- `cargo test -p tenferro-linalg`
- `cargo test -p tenferro-linalg --test integration --features autodiff,webgpu apple_shared -- --nocapture`
- `cargo test -p tenferro-linalg --no-default-features --features cpu-blas,blas-accelerate cpu::tests::linalg::test_complex_cholesky -- --nocapture`
- `cargo test -p tenferro-linalg --no-default-features --features cpu-blas,blas-accelerate,autodiff,webgpu --test integration apple_shared -- --nocapture`
- `cargo clippy -p tenferro-linalg --all-targets --features autodiff,webgpu -- -D warnings`
- `cargo clippy -p tenferro-linalg --all-targets --no-default-features --features cpu-blas,blas-accelerate,autodiff,webgpu -- -D warnings`
- `cargo fmt --all -- --check`
- `python3 scripts/check-public-error-docs.py` over every changed Rust source

## Residual scope

- Managed borrowed views that are not owned compact tensors remain rejected.
- No generic managed CPU linalg parity is claimed; prepared solve, LU, QR, SVD,
  eig/eigh, and triangular solve remain follow-up work.
- The Apple tests skip when a host-visible Metal runtime cannot be initialized,
  matching the existing hardware-gated Apple test convention.

## Review follow-up

The two Important findings in `task-4-review.md` were addressed in a follow-up:

- `CpuBackend::cholesky` now selects managed execution from the input's actual
  `Buffer::Backend` storage. A domain-bound backend preserves the ordinary host
  path for both direct owned tensors and `TensorRead::from_tensor`.
- Hardware-neutral fake guarded buffers and a fake allocation domain exercise
  F32/F64/C32/C64 factorization, direct and owned-read host parity, guarded
  reads and full writes, matching domain and distinct allocation identity, and
  foreign, device-local, and busy-buffer failures. These tests run under both
  the default faer configuration and the Accelerate-backed BLAS configuration;
  the macOS Apple integration remains the end-to-end complement.
