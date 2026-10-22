# Real FFT correctness and nonfinite padding

## Decisions

- Preserve the final positive-frequency bin for odd real transforms: it has a
  Hermitian partner and its imaginary part contributes to the inverse. Only
  even lengths have a Nyquist bin with no separate partner.
- Validate nonfinite padding before cuFFT and compare implicit with explicit
  padding on CUDA. A minimal probe with realfft 3.5.0/RustFFT 6.4.1 returns NaN
  for the real F64 input `[Inf, 0]` on Linux x86_64, while native macOS returns
  Inf, matching the observed cuFFT result. Exact CPU nonfinite classification
  was an unsuitable padding oracle. The durable validation boundary is in
  [FFT backend execution](../design/fft-backend-execution.md).

## Verification conclusions and constraints

- Independent DFT regressions reproduce both odd-length CPU defects before
  the fix and pass afterward, for both precisions, three normalization modes,
  multiple axes, owned/borrowed inputs, and serial/parallel engines.
- The FFT/autodiff suite and doctests pass (119 tests), along with 64
  CUDA-feature host unit tests and the local formatting/clippy gates.
- CUDA padding tests compile but require NVIDIA hardware to execute. This
  macOS host cannot execute them; the hosted CUDA gate must establish device
  execution.
- Refused-allocation readback is also repaired in this batch; its separate
  lifecycle decisions and validation constraints are recorded in
  [the #2247 work log](cuda-refused-download-2247.md).
