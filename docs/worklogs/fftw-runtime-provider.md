# FFTW runtime provider

## Decisions

- Select FFTW per CPU engine and reuse the existing lane execution and plan cache. See the [FFT backend design](../design/fft-backend-execution.md).
- Reuse plans across lane buffers by using FFTW’s unaligned mode. This can limit SIMD selection. Serialize native planning and destruction while keeping execution concurrent across separate buffers.
- Threaded providers require FFTW 3.3.9 or later. They call `fftw_make_planner_thread_safe` and restore the previous planner thread count after each plan, so other FFTW users in the process keep their planner setting.

## Verification conclusions and constraints

- Runtime coverage checks both precisions, odd and even lengths, concrete and traced execution, eager in-place execution, and reuse of threaded plans.
- Local runtime testing uses macOS FFTW. The Linux CI job exercises the separate threading libraries.
