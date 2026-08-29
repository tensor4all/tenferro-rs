# CUDA dispatch and cuBLAS BLAS-1

## Session summary

PR #1733 reduces fixed CUDA issue overhead and adds cuBLAS-backed `vdot`,
`norm_squared`, and `axpby`. Review then found that the first implementation
cached by unbounded logical stream IDs, allowed shared cuBLAS handle state to
race, and did not retain cross-stream allocations until vendor work completed.
The final implementation keys fixed tables by CubeCL's physical stream pool,
locks each cuBLAS handle through configuration and enqueue, gives each
cuTENSOR plan one lazy workspace per physical stream, and retires all relevant
streams before releasing resources.

The complex zero-fill workaround was removed. CubeCL PR #17 fixes complex cast
lowering at the emitter and this PR pins that fix. CubeK PR #12 advances its
CubeCL pins to the same revision so workspace feature unions use one compatible
type universe. The independently observed cuTENSOR/Volta wrong-result boundary
is handled separately by tenferro PR #1734.

## Context read

- `AGENTS.md`, `REPOSITORY_RULES.md`, `CONTRIBUTING.md`, and the shared common,
  Rust, performance, numerical, documentation, and test rules
- `docs/design/gpu-backend-design.md`
- CubeCL CUDA stream-pool, event, allocation, and C++ emitter implementations
- cuBLAS stream/pointer-mode and cuTENSOR workspace lifecycle call sites
- PR #1733 review comments and hosted-CI failures

## Measurement record

- baseline source: `8e8d879da79d5dfac573cf52924f84d4bb903c48`
- originally measured candidate: `30151a5a6dc689e952e32b5daee39b55a4c6a5ce`
- hardware: NVIDIA A800
- software: Rust 1.96, CUDA 12, cuTENSOR 2.3.1
- cases: C64 `dot_general` at `256x192x64`, allocating and preallocated enqueue,
  per-operation synchronization, empty-queue synchronization, a dimension-4096
  conjugated self-dot plus scalar readback, and two-site TDVP at bond dimensions
  64, 128, and 256 against a 16-thread CPU baseline

The original measurement session did not record warmup count, repetition
count, sample dispersion, exact CUDA minor version, or benchmark harness SHA.
The reported medians remain useful directional evidence, but are not treated as
a reproducible performance gate. The review correctness changes were not
remeasured locally because this machine has no CUDA GPU.

## Decisions

1. Match CubeCL's configured physical stream count and modulo mapping. This is
   bounded and avoids an LRU whose eviction would add synchronization to thread
   churn.
2. Keep the same-physical-stream path asynchronous. Synchronize only when raw
   vendor work borrows an allocation owned by another physical stream.
3. Hold the per-slot cuBLAS mutex across pointer-mode selection and enqueue.
4. Allocate cuTENSOR workspaces lazily per physical stream and count them in
   retained-byte limits. Drop workspaces first so their stream barriers precede
   plan and descriptor destruction.
5. Propagate scalar pointer-resolution failures. If a completion barrier fails,
   retain the involved allocation handle instead of permitting unproven reuse.
6. Keep compact BLAS-1 inputs allocation-free; box only the noncontiguous
   materialization slow path.

## Verification

- CUDA-feature library tests, including plan-cache accounting
- CUDA-feature clippy with warnings denied
- CUDA integration source-contract tests
- CPU BLAS-1 thread-count and pool-aliasing tests
- build-artifact dependency-contract tests
- formatting and diff checks
- expanded CUDA hardware tests compile locally; execution remains hosted-CI
  owned

## Remaining risks

- Same-stream performance after the correctness patch has not been remeasured.
- Cross-stream raw vendor calls intentionally pay a stream barrier.
- CUDA and V100 hardware validation depend on hosted or maintainer hardware.
- The temporary CubeCL Git pin should be replaced by a published dependency
  once the emitter fix is released.
