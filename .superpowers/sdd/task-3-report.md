# Task 3 report: CubeK FFT on Apple Metal

## Implemented

- Added a narrow `#[doc(hidden)]` WebGPU extension interop surface that is
  scoped to a specific `WebGpuBackend`: exact client access, resident compact
  F32/C32 input handles, unaliased raw allocation, and owned-handle finalization.
- Implemented `FftBackend for WebGpuBackend` through CubeK's explicit-client
  launch APIs at pinned CubeK revision
  `43e8521885f141cb8ccdf99a766bfde118412010`.
- Preserved tenferro column-major logical strides, including C32 logical strides
  over CubeK's interleaved `[re, im]` scalar ABI.
- Implemented C32 forward/inverse CFFT, F32 one-sided RFFT, and C32-to-F32
  IRFFT. RFFT padding/truncation uses CubeK's padded launch APIs.
- Mapped `Backward`, `Forward`, and `Ortho` normalization directionally.
- Kept output handles unique until after CubeK's overlap check: allocate raw,
  move into the CubeK output wrapper, launch, recover the raw handle, then build
  the tenferro backend buffer.
- Added checked preflight for dtype/operation, power-of-two length, minimum
  length, CFFT requested-length changes, full real FFT, device maximum,
  shared-memory/unit availability, lane counts, launch integer widths, shapes,
  strides, element counts, and byte lengths.
- Preserved CubeK `FftError` values as typed `BackendSource` causes. There is no
  CPU fallback, implicit transfer, or default-client launch.
- Intentionally left `FftExecutionCache` unused: CubeCL's exact compute client
  owns compiled-kernel caching and CubeK exposes no host plan object.

## Capability limits

- Supported: F32/C32 only, power-of-two lengths at least 2, one-sided real FFT,
  and same-length CFFT.
- Typed `Unsupported` errors: F64/C64, full real FFT, length 1,
  non-power-of-two transforms, CFFT padding/truncation, and lengths beyond the
  active device maximum.
- Foreign managed domains and ordinary device-local WebGPU buffers are rejected
  by the owner-scoped residency boundary without transfers.

## TDD evidence

- RED: `webgpu_metal_fft` initially failed to compile because
  `WebGpuBackend: FftBackend` was not implemented.
- GREEN: the dedicated serialized Metal matrix passes through the public FFT
  surface and exercises both the last shared-memory and first four-step lengths.

## Tests added

- C32 forward/inverse transforms against RustFFT for all normalization modes.
- Column-major batches on axis 0 and a nonzero middle axis.
- F32 RFFT padding and truncation plus C32 IRFFT round trips.
- Small/large CFFT and RFFT/IRFFT paths derived from active Metal properties.
- Output domain, distinct allocation identity, unchanged transfer counters, and
  successful launch through the unique-output lifecycle.
- F64, C64, full-RFFT, length-1, non-power-of-two, oversized, CFFT
  length-change, foreign-domain, and device-local error paths.
- A non-macOS compile-only public surface check and a portable source contract
  for explicit-client launch, raw output finalization, no transfer helpers,
  unused tenferro cache, and typed backend sources.

## Verification

- `cargo test -p tenferro-fft --features webgpu --test webgpu_metal_fft -- --nocapture`
- `cargo test -p tenferro-fft`
- `cargo test -p tenferro-fft --features autodiff`
- `cargo test -p tenferro-fft --features webgpu --doc`
- `cargo clippy -p tenferro-fft --all-targets --features webgpu,autodiff -- -D warnings`
- `python3 scripts/check-public-error-docs.py --changed-from 1c3a455e`
- `cargo fmt --all -- --check`

All listed checks passed locally; the hardware matrix executed on Apple Metal.
Architecture guides, runnable tutorials, and the consolidated Apple worklog
remain Task 5 so that documentation describes the completed Task 1-4 surface.
