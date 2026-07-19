# Apple shared FFT and Cholesky work log

Date: 2026-07-19

## Session summary

This work completed the first explicit Apple shared CPU/Metal workflow. A
single `AppleContext` now owns a host-visible Metal allocation domain, guarded
CPU RustFFT execution, CubeK Metal FFT execution, and one truthful mapped CPU
linalg operation: rank-2 Cholesky. Runnable tutorials demonstrate backend
selection and assert that guarded mapping and Metal launch do not introduce
post-creation transfers.

Durable contracts are recorded in [GPU Backend Design](../design/gpu-backend-design.md)
and [FFT Backend Execution](../design/fft-backend-execution.md).

## Context read

| Source | Decision impact |
| --- | --- |
| Repository, Rust, performance, numerical, documentation, and test rules | Kept provider selection explicit, preserved typed errors, used focused hardware tests, and added runnable tutorials plus this work log. |
| CubeCL host-visible allocation change `11b52669f13e27bbe188f988fd696df6d989a562` | Retained the resolved managed resource lease and enforced CPU/GPU exclusion through guards. |
| CubeK FFT change `43e8521885f141cb8ccdf99a766bfde118412010` | Used configured-client CFFT/RFFT/IRFFT launch APIs without weakening CubeK's unique-output ownership checks. |
| Existing FFT capability/cache boundary and public linalg lowering | Kept `FftPlanSpec` backend-neutral and selected Cholesky because its public concrete/eager/traced route could genuinely use guarded mappings. |

CodeGraph was used before direct source inspection to locate allocation,
backend, FFT, and linalg boundaries.

## Decisions made

- CPU and Metal backends are always passed explicitly. Unsupported Metal
  requests return typed errors and never transfer or dispatch to RustFFT.
- RustFFT maps matching managed inputs for `F32`, `F64`, `C32`, and `C64`, then
  allocates and writes results through the same domain owner.
- CubeK Metal FFT is limited to F32/C32 power-of-two CFFT, one-sided RFFT, and
  IRFFT. CubeCL's configured client owns compiled kernels; CubeK exposes no
  vendor-plan object to store in `FftPlanCache`.
- Rank-2 Cholesky is the initial mapped CPU linalg operation for
  F32/F64/C32/C64. Documentation does not imply solve or general linalg parity.
- Physical allocation identity is an invariant for the same tensor as CPU and
  Metal consume it. Operation outputs are distinct allocations in the same
  domain.
- CubeCL and CubeK remain pinned to reviewed Git revisions. Publishing and
  release migration are deferred to a separate task.

## Rejected or deferred alternatives

- Implicit dtype/size routing between RustFFT and CubeK was rejected because it
  would hide provider choice and make transfer behavior surprising.
- Metal F64/C64 support was not emulated: current Metal shader support and the
  CubeK FFT API are F32/C32-only.
- A broad mapped CPU linalg claim was rejected. Only the implemented and tested
  rank-2 Cholesky path is documented.
- WASM and CUDA/cuFFT remain separate future work; the CPU provider already has
  a portable RustFFT implementation and this change targets Apple sharing.

## Verification performed

- Focused Apple context tests on Metal, including foreign-domain, overlap,
  allocation-ID, and transfer-counter behavior.
- CPU RustFFT tests for F32/F64/C32/C64 managed tensors and cache reuse.
- CubeK Metal FFT tests for small/large paths, axes, batches, normalization,
  padding/truncation, capability errors, placement, and transfer invariants.
- Managed Cholesky tests for all four CPU floating/complex dtypes, public
  concrete/eager/traced routing, residuals, and typed placement failures.
- The `apple_shared_fft` and `apple_shared_cholesky` tutorial binaries were
  compile-checked and executed through `tutorial_binaries` on macOS Metal.
- Formatting, relevant clippy targets, rustdoc, public error documentation, and
  documentation-site checks are final PR gates.

## Remaining risks and follow-up

- Metal FFT remains limited to CubeK's current F32/C32 power-of-two surface.
- Only rank-2 Cholesky has a mapped CPU linalg path.
- The exact Git dependency pins intentionally remain until a later release
  task; this PR does not publish crates or merge dependency PRs.
