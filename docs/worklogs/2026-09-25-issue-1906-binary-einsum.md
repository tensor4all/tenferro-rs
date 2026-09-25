# Issue #1906: non-AD binary einsum dispatch

The initial compatibility-preserving stage is recorded first. The final stage
below supersedes its public-type restriction and performance results.

## Initial decision and scope

Supported concrete two-input `_into` expressions dispatch a checked dot configuration without constructing a contraction tree. ASCII string labels are borrowed as bytes; integer subscripts share the same compact planner. Operand swapping matches the requested output order without a materialized transpose. Prepared plans retain shape/dtype-bound metadata, not strides or storage: execution still checks current layouts at the backend. The ordinary string path requires neither a caller-prepared plan nor a cache hit.

The CPU rank-two/single-contraction/no-batch case derives GEMM dimensions after the existing axis validation instead of constructing generic dimension groups. Other configurations keep the generic analyzer. No kernel, public API, dependency, cache, provider/device contract, or AD semantics changed. Tiny tensor-view enum conversions permit cross-crate inlining; their independent timing contribution has not been established.

The separate `EagerTensor`/`eager_ad.rs` route is not the target of the AD-disabled workload. The attempted optimization there was removed rather than retaining an unrelated change. Broadcast, repeated-label, ellipsis, unsupported-character and general-layout cases retain existing fallbacks. Input count, dtype, shape, and paired extents are validated before dispatch. In particular, swapped configurations must inspect shapes in swapped order, and a two-input fast path must never ignore additional operands.

## Verification

The concrete regression tests exercise operand swapping, unequal ranks, contracting-dimension broadcasting, parsed-input count errors without output mutation, complex values and strided destinations. Compact byte/integer configurations are checked against full plans over 1,053 label combinations. A source contract guards the actual non-AD typed-view string entry point before generic parsing, not an AD-only function.

The independent release suite passes seven test functions: 98 expressions across offset, noncompact and negative-stride layouts; string, prepared, notation and integer-subscript entry points; owned inputs; f32; complex32/64 alpha/beta and conjugation; and malformed/count/dtype/output errors with unchanged destinations. The same suite passes against the baseline. Default/autodiff einsum tests, tensor tests, and focused clippy pass. CPU tests passed for the unchanged rank-two analyzer slice. GPU/Accelerate execution and line coverage were not measured locally; no full workspace or hosted CI claim is made.

## Initial performance result: improvement, target not met

The predeclared paired experiment is **VALID**, but its primary objective is **FAIL**, not completion. With Faer, AD disabled, explicit one-thread backend/`Par::Seq`, CPU 8 affinity on EPYC 7713P, release rustc 1.97.1, five alternating baseline/candidate process rounds and three rotated repetitions per process:

| Case | Baseline string total | Candidate string total | Candidate overhead above same-process faer |
| --- | ---: | ---: | ---: |
| n=1, p=1 | 18.105 µs | 1.513 µs | 1.481 µs |
| n=96, p=1 | 20.315 µs | 2.745 µs | 1.484 µs |
| n=96, p=10 | 27.169 µs | 6.517 µs | 1.525 µs |
| n=96, p=120 | 120.966 µs | 48.720 µs | 2.638 µs |

Ordinary string calls now allocate twice (64 bytes), versus 39 allocations/2424 bytes after the initial dispatch fix and many more in the original implementation. Prepared calls allocate zero. The remaining two ordinary-call allocations are owned `DotGeneralConfig` axis vectors; replacing their public representation is outside this compatibility-preserving change. Direct backend overhead remains about 0.87–0.88 µs for the two small primary cases. Neither the ≤1 µs unprepared-string target nor the auxiliary zero-allocation criterion is met. Bounds, placement, aliasing and other safety checks were not removed to chase the target.

Evidence is retained locally under `/tmp/tenferro-1906-implementation/`: `protocol.md`, `paired-parent-9469d0b3/` (all samples, confidence intervals, host observations and allocation counts), and `parent-final-semantics.log`. The candidate is commit `223d14d6bb87a203dfdb70d1d1b00230e20ef609` plus `parent-borrowed.patch` (SHA-256 `9469d0b3540e5a8679989a544c43616b599cea4e9c66d7ef6f4ba73d96a04ef9`); subsequent edits only update this record and design prose. Baseline source is v0.7.1 `8a1839febeb3c868a502e26397ca10761bbc568d`, with relevant dependency sources identical to starting main `aebc3148d3aab9f0030d7ee353759581da12f826`. Benchmark source `/tmp/tenferro-1906-probe/src/main.rs` SHA-256 is `4a65f716a355c7fab35c55a28ea12f7dab03194e75d8c5c447f3529116268748`. These are shared-host CPU measurements, not Apple/Accelerate certification or GPU evidence.

## Final stage: inline axes and checked rank-two analysis

The maintainer permitted public type changes and requested four inline axes per
list rather than two. All four `DotGeneralConfig` fields now use the existing
`SmallVec<[usize; 4]>` dependency. Five or more axes spill; rank and axis ordering
remain unrestricted. On this 64-bit build the config grows from 96 to 160 bytes
(capacity eight would require 288 bytes). Direct construction and collected
axes avoid temporary heap vectors in the einsum planners. Most caller changes
are mechanical literal/conversion migrations, including AD, tests and examples.
Equality, hashing and canonical identity still describe ordered axis values,
not inline/spilled storage. Cache accounting excludes inline bytes already
included in the enclosing object and retains the existing logical-length versus
capacity policy for spilled buffers.

The final bounded CPU adjustment validates the complete rank-two/single-axis/
no-batch case inside the existing metadata analyzer: both ranks and axis-list
lengths, empty batch lists, axis bounds and paired extents. It does not repeat
the generic axis-group validator on success. Rejected cases retain the general
validator and its original error. It does not bypass runtime placement, layout,
dtype, output, accumulation or provider validation. Passing validation tokens
through the other preparation paths was deliberately deferred instead of adding
new internal protocols to finish this optimization.

The final five-round paired experiment `paired-inline4-validated/` is **VALID**
and the primary objective remains **FAIL**:

| Case | Baseline string total | Final string total | Overhead above same-process faer |
| --- | ---: | ---: | ---: |
| n=1, p=1 | 17.593 µs | 1.176 µs | 1.149 µs |
| n=96, p=1 | 18.905 µs | 2.468 µs | 1.084 µs |
| n=96, p=10 | 26.116 µs | 6.057 µs | 1.180 µs |
| n=96, p=120 | 118.341 µs | 46.634 µs | 1.760 µs |

All measured ordinary string cases now have **zero heap allocations**. Backend
small-case overhead is 0.765–0.774 µs. The preceding inline-only experiment
`paired-inline4/` also remains available (primary overhead 1.158/1.219 µs);
separate experiment differences are not a controlled attribution of every
nanosecond. Neither experiment reaches the requested ≤1 µs in both primary
cases. No further tuning is claimed.

Final verification:

- CPU tests: 840 passed. Einsum default/autodiff: 319/357 passed, including
  doctests. Tensor tests and doctests passed with the cache wrapper disabled in
  a fresh target directory; initial UI failures were diagnostic-path differences
  caused by kache, not blessed or suppressed snapshots.
- Seven affected crate library suites passed; workspace/all-target compile
  check and CUDA+WebGPU/all-target compile check passed. GPU/Apple numerical
  execution was not performed.
- The independent release suite passes all seven test functions against both
  candidate and baseline. It now covers 103 expressions, adding four- and
  five-contracting-axis cases, reversed contracting order, and five batch axes
  across the existing three layouts and public API paths.
- Added inline/spill, validation, hash/equality, ordered canonical identity,
  retained-byte and planner tests. Twelve malformed rank-two configurations
  preserve the general validator's error. Focused llvm-cov coverage is 100% for
  `config.rs` and `binary_dot.rs`; broader files remain below 90% in this local
  subset (for example GEMM 88.16%, runtime preparation 65.72%). This is not a
  claim that the hosted full-coverage gate passed.
- The local fast gate (formatting, root and standalone-extension clippy,
  documentation sync, focused tests) passed against the original agreed base
  `aebc3148`. The default freshness check failed because `origin/main` advanced;
  no rebase or latest-main integration is claimed, and that remains required
  before any PR. The changed dynamic-shape SVD tutorial also ran successfully.
- Self-review found no new dependency, kernel, hidden cache, unchecked entry
  point or AD semantic change. Shipped compute skills contain no affected
  `DotGeneralConfig` literals and need no migration. No push or PR was made.

Evidence stays under `/tmp/tenferro-1906-implementation/`. The measured production
source is `a5e52128` plus `inline-final.patch` (SHA-256
`b2acee8fa47a45f7d1348fa99e33c43875ebd2b9d7060c0b6aaaaf535052b947`);
later changes only add tests/documentation. Candidate binary SHA-256 is
`343731fab2986d7e1b6b1298f911d8714fa64ca4b6dfb6d52f68a88fd2f94aea`.
The candidate harness source SHA-256 is
`6b677b73aabf0e03c10b2febda1b7f7a78b3c5efa04dd0178e5c30769e876ead`:
only its preconstructed dot config literal changed for type compatibility;
measurement loops, direct faer reference and frozen baseline binary are unchanged.
The same explicit 1T/provider assertions and original validity rules apply.
