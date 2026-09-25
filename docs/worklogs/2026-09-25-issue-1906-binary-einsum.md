# Issue #1906: non-AD binary einsum dispatch

## Decision and scope

Supported concrete two-input `_into` expressions dispatch a checked dot configuration without constructing a contraction tree. ASCII string labels are borrowed as bytes; integer subscripts share the same compact planner. Operand swapping matches the requested output order without a materialized transpose. Prepared plans retain shape/dtype-bound metadata, not strides or storage: execution still checks current layouts at the backend. The ordinary string path requires neither a caller-prepared plan nor a cache hit.

The CPU rank-two/single-contraction/no-batch case derives GEMM dimensions after the existing axis validation instead of constructing generic dimension groups. Other configurations keep the generic analyzer. No kernel, public API, dependency, cache, provider/device contract, or AD semantics changed. Tiny tensor-view enum conversions permit cross-crate inlining; their independent timing contribution has not been established.

The separate `EagerTensor`/`eager_ad.rs` route is not the target of the AD-disabled workload. The attempted optimization there was removed rather than retaining an unrelated change. Broadcast, repeated-label, ellipsis, unsupported-character and general-layout cases retain existing fallbacks. Input count, dtype, shape, and paired extents are validated before dispatch. In particular, swapped configurations must inspect shapes in swapped order, and a two-input fast path must never ignore additional operands.

## Verification

The concrete regression tests exercise operand swapping, unequal ranks, contracting-dimension broadcasting, parsed-input count errors without output mutation, complex values and strided destinations. Compact byte/integer configurations are checked against full plans over 1,053 label combinations. A source contract guards the actual non-AD typed-view string entry point before generic parsing, not an AD-only function.

The independent release suite passes seven test functions: 98 expressions across offset, noncompact and negative-stride layouts; string, prepared, notation and integer-subscript entry points; owned inputs; f32; complex32/64 alpha/beta and conjugation; and malformed/count/dtype/output errors with unchanged destinations. The same suite passes against the baseline. Default/autodiff einsum tests, tensor tests, and focused clippy pass. CPU tests passed for the unchanged rank-two analyzer slice. GPU/Accelerate execution and line coverage were not measured locally; no full workspace or hosted CI claim is made.

## Performance result: improvement, target not met

The predeclared paired experiment is **VALID**, but its primary objective is **FAIL**, not completion. With Faer, AD disabled, explicit one-thread backend/`Par::Seq`, CPU 8 affinity on EPYC 7713P, release rustc 1.97.1, five alternating baseline/candidate process rounds and three rotated repetitions per process:

| Case | Baseline string total | Candidate string total | Candidate overhead above same-process faer |
| --- | ---: | ---: | ---: |
| n=1, p=1 | 18.105 µs | 1.513 µs | 1.481 µs |
| n=96, p=1 | 20.315 µs | 2.745 µs | 1.484 µs |
| n=96, p=10 | 27.169 µs | 6.517 µs | 1.525 µs |
| n=96, p=120 | 120.966 µs | 48.720 µs | 2.638 µs |

Ordinary string calls now allocate twice (64 bytes), versus 39 allocations/2424 bytes after the initial dispatch fix and many more in the original implementation. Prepared calls allocate zero. The remaining two ordinary-call allocations are owned `DotGeneralConfig` axis vectors; replacing their public representation is outside this compatibility-preserving change. Direct backend overhead remains about 0.87–0.88 µs for the two small primary cases. Neither the ≤1 µs unprepared-string target nor the auxiliary zero-allocation criterion is met. Bounds, placement, aliasing and other safety checks were not removed to chase the target.

Evidence is retained locally under `/tmp/tenferro-1906-implementation/`: `protocol.md`, `paired-parent-9469d0b3/` (all samples, confidence intervals, host observations and allocation counts), and `parent-final-semantics.log`. The candidate is commit `223d14d6bb87a203dfdb70d1d1b00230e20ef609` plus `parent-borrowed.patch` (SHA-256 `9469d0b3540e5a8679989a544c43616b599cea4e9c66d7ef6f4ba73d96a04ef9`); subsequent edits only update this record and design prose. Baseline source is v0.7.1 `8a1839febeb3c868a502e26397ca10761bbc568d`, with relevant dependency sources identical to starting main `aebc3148d3aab9f0030d7ee353759581da12f826`. Benchmark source `/tmp/tenferro-1906-probe/src/main.rs` SHA-256 is `4a65f716a355c7fab35c55a28ea12f7dab03194e75d8c5c447f3529116268748`. These are shared-host CPU measurements, not Apple/Accelerate certification or GPU evidence.
