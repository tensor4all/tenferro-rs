# Deferred GPU, Safety, And Large Issues Dispatch Design

**Date:** 2026-05-02

**Status:** Proposed triage spec

## Issues

Large implementation or design issues:

- #212: perf(tropical-ad): optimized argmax forward path
- #602: Design: compute-device inference primary, runtime override secondary
- #629 and #797: add F32/C32 CPU linalg support
- #701: CPU elementwise fusion via closure composition
- #745: global metadata registry is an unbounded memory leak
- #755: shared execution handle for EagerTensor and plain Tensor ops
- #760: provider-inject runtime path after cblas/lapack dual ABI support
- #774: cpu-faer + cpu-blas both enabled causes eig compile error

GPU and CubeCL issues:

- #764: inconsistent I64 reduce_sum/reduce_prod CPU vs GPU
- #765: GPU i64 index conversion roundtrip
- #770: GPU LU decomposition roundtrip
- #771: GPU scatter kernel is single-threaded
- #793: GPU convert_float_to_complex_raw sets resident_device to None
- #796: GEMM zero-alloc waste before GPU upload
- #809: no I64 GPU tests
- #810: no complex GPU fusion tests
- #817: CudaDataType lacks explicit repr
- #819: CubeCL real-conj clone path does not verify device residency

Low-level safety and performance issues:

- #782: buffer_pool set_len leaves uninitialized memory tail
- #806: analytic ops use zero-initialized output instead of uninit
- #812: consolidated minor audit issues
- #813: missing/weak safety contracts around unsafe blocks
- #815: integer overflow risks in tril/triu and GEMM planning
- #816: unnecessary clone before reduction on contiguous inputs

Infrastructure:

- #586: use self-host runner

Explicitly excluded from this batch:

- #724 and #791: complex CubeCL math/fusion support
- #728: GPU delegation benchmark matrix

## Goal

Keep the first single PR bounded. These issues need either design review, GPU
hardware, external dependency readiness, broad API changes, or performance
measurement. They should not be silently mixed into the first implementation
passes.

## Dispatch Policy

This document is not an instruction to implement all listed issues in one pass.
It is the triage boundary for issues that should be deferred or split into
dedicated future PRs unless the user explicitly reprioritizes them.

## Classification

### Requires design approval before implementation

- #602 compute-device/runtime selection,
- #745 global metadata registry ownership,
- #755 shared execution handle,
- #701 CPU fusion,
- #212 tropical argmax primitive,
- #782 buffer-pool initialization contract if runtime checking or MaybeUninit
  migration is considered.

### Requires external dependency or upstream readiness

- #760 provider-inject ABI path,
- #629 / #797 if faer API support or trait coverage is incomplete.
- #774 if it is invalidated by the repository's exactly-one-CPU-backend
  compile-time contract rather than a supported feature combination.

### Waiting on CubeCL PR

- #724 / #791 are blocked on the in-flight CubeCL repository PR for complex
  math support. Do not attempt a tenferro-side implementation in this batch.
  Once that PR lands, the tenferro follow-up should be limited to a fork rev
  bump plus focused complex GPU fusion/runtime coverage.

### Requires GPU hardware

- #764 / #765 / #770 / #771 / #793 / #796 / #809 / #810 / #819.

### Excluded benchmark work

- #728 is out of scope for this batch. Do not create benchmark harnesses, run
  performance comparisons, or adjust dispatch strategy based on benchmark goals
  in the first single PR.

### Small but safety-sensitive follow-ups

- #806 uninitialized analytic outputs,
- #813 unsafe contracts,
- #815 integer overflow,
- #816 reduction clone,
- #817 C ABI representation.

These may become their own dispatches after the first five specs are complete.

## Acceptance Specification

For the first single PR:

- do not implement deferred issues unless explicitly pulled into scope,
- do not implement #724 / #791 until the CubeCL PR lands,
- do not implement #728 or any GPU benchmark harness in this batch,
- mention deferred issues in the PR body if related files were touched,
- avoid partial fixes that make future dedicated work harder,
- prefer opening or updating issue comments over speculative code changes.

For a future dispatch from this document:

1. select one subsection only,
2. write a dedicated design doc for that subsection,
3. define required hardware, benchmark, or oracle prerequisites,
4. then dispatch implementation.

## Testing Expectations

No tests are required for this triage document itself.

Any future dispatch must define its own verification. For GPU issues, the spec
must state whether local non-GPU tests are sufficient or whether CUDA hardware
is required before claiming completion.

## Dispatch Prompt

```text
Do not implement this document as one patch. Use
docs/plans/2026-05-02-deferred-gpu-safety-large-issues-design.md as the triage
boundary for issues intentionally excluded from the first single PR.

If asked to work on one listed subsection, first write a dedicated design doc
for that subsection, including hardware, benchmark, API, and safety
prerequisites.
```

## Review Checklist

- Deferred issues are not accidentally mixed into unrelated dispatches.
- Any touched deferred area has an explicit reason in the PR summary.
- GPU completion claims are not made without GPU verification.
- Design issues are not implemented from issue text alone.

## Stop Conditions

Stop and report if:

- a first-pass dispatch starts requiring GPU hardware,
- a first-pass fix requires redesigning public execution context APIs,
- external provider or CubeCL changes are needed before tenferro can proceed.
