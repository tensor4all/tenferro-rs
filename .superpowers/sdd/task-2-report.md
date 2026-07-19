# Task 2 report: RustFFT over guarded Apple mappings

## Implemented

- Kept the existing host-only `CpuBackend` FFT path and RustFFT kernels intact.
- Added a shared-domain path that accepts only compact managed buffers from the
  `CpuBackend`'s configured allocation domain.
- Held each typed read guard through lane execution, computed into owned RustFFT
  scratch/output storage, allocated the result through
  `SharedTensorAllocationDomain`, and completed it with a write-only
  `copy_from_slice` mapping.
- Preserved F32/F64/C32/C64 dispatch, axes and batches, normalization,
  padding/truncation, and caller/runtime plan caches.
- Rejected foreign-domain and domainless/device-local buffers with typed host
  access errors and without explicit transfers.
- Added the `tenferro-fft/webgpu` feature and a macOS-gated Apple CPU integration
  test target.

## TDD evidence

- RED: the initial Apple integration run failed both tests because managed
  WebGPU inputs were rejected by the host-only validation path and foreign
  domains did not yet produce `HostAccessError::ForeignDomain`.
- GREEN: the same focused test now passes on local Apple Metal hardware.

## Tests added

- Managed F32 batched/non-last-axis RFFT with padding and orthonormal scaling.
- Managed F64 RFFT/IRFFT with forward normalization.
- Managed C32 FFT truncation and forward normalization.
- Managed C64 FFT and caller-owned plan-cache reuse.
- Output-domain identity and unchanged post-creation transfer counters.
- Foreign-domain and ordinary device-local WebGPU rejection.

## Verification

- `cargo test -p tenferro-fft --features webgpu --test apple_cpu`
- `cargo test -p tenferro-fft`
- `cargo test -p tenferro-fft --features autodiff`
- `cargo clippy -p tenferro-fft --all-targets --features webgpu,autodiff -- -D warnings`
- `python3 scripts/check-public-error-docs.py --changed-from 06fad2dd`
- `cargo fmt --all -- --check`

All listed checks passed locally; the Apple integration test ran on Metal rather
than taking its hardware-unavailable skip path.
