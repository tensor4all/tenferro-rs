# Householder QR small-append performance investigation (#1750)

## Summary

Added a separate factor-5 plus nine append-3 benchmark for 64, 128, and 256
rows. It measures raw append, R extraction, newly appended Q columns, the
complete workflow in one reused CPU session, the same workflow with one fresh
session per operation, and the eager workflow.

The compact reflector kernel is not the reported regression. On the measured
faer path, raw compact append was 4.6--7.0x faster than the explicit-Q BCGS2
diagnostic. Across F64/C64, opening a session for each append/R/Q operation made
the complete compact workflow 1.9--3.4x slower than reusing one session; eager
dispatch made it 4.0--7.2x slower. This localizes the first tenferro-visible cost to operation
submission/session boundaries, before Matrix conversion or inverse-adjoint
maintenance is added.

## Context and protocol

- Base commit: `c27977aee2db33b33f18b2d6c91e5a8d36daa236`.
- Design: `docs/design/incremental-householder-qr-small-append-performance.md`.
- Pre-implementation review: `reviewer-flash`, high thinking, read-only.
  Initial verdict required fixed-cost attribution and a batched/noise protocol;
  closure verdict was **Correct-to-merge**.
- CPU: AMD EPYC 7713P, logical CPU 17, release profile, faer, F64 and C64.
- Threads: `RAYON_NUM_THREADS=OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`.
- Each sample batches 64 independently reset workflows after an untimed 50 ms
  pinned-core clock warmup. Seven fresh processes per case used alternating
  lane order; each process used three warmups and ten measurements. Every
  process-median CoV was below 7.4%. Every record reported affinity `17` and
  CPU frequency 3097--3100 MHz; the runner treats missing observations as
  `INCONCLUSIVE`.
- `TENFERRO_BENCH_GIT_COMMIT` is mandatory; benchmark records also retain
  affinity, CPU frequency, and the four thread-control variables.
- Accepted raw artifacts are local under `/tmp`: F64
  `tenferro-1750-small-appends-f64-final.jsonl` (SHA-256
  `00d8edb3b15810d049899e0413988427f521ae2ea79be779652ba7e4d391e875`) and
  C64 `tenferro-1750-small-appends-c64-final.jsonl` (SHA-256
  `fcd16a635a9f45f83b70a3787ef1840359a468595577226cbc1d8924c4998a42`), for
  benchmark source SHA-256
  `3a0662f81e4171f57c228f867563c2393747949b0d92b0b23527c31a1ce0f703`.
  The external sequential runner injected the source hash, cycle, and order
  fields into each raw record.

Two complete predecessor runs were `INCONCLUSIVE`: precomputing and retaining
all nine post-append states made the R/Q component lanes cache/lifetime-mismatched
with the complete workflow and produced CoV above 10%. No favorable cases were
selected from those runs. The benchmark was corrected to execute the identical
interleaved workflow in every component lane and accumulate only the selected
operation intervals; the full suite was then rerun from scratch. The final
report is `PASS` under the predeclared 10% CoV and >=1 ms gates.

## Results

Times below are median microseconds per complete nine-append sequence (the raw
records contain batched milliseconds).

### F64

| rows | append | R outputs | selected Q | reused-session complete | fresh-session complete | eager complete |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 56.9 | 19.3 | 37.7 | 113.7 | 391.9 | 814.9 |
| 128 | 74.6 | 19.2 | 47.5 | 141.2 | 422.0 | 856.3 |
| 256 | 111.3 | 19.5 | 66.3 | 197.3 | 467.7 | 1019.0 |

### C64

| rows | append | R outputs | selected Q | reused-session complete | fresh-session complete | eager complete |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 76.2 | 19.6 | 52.0 | 147.3 | 430.1 | 874.0 |
| 128 | 113.6 | 19.9 | 80.1 | 212.8 | 498.5 | 1034.0 |
| 256 | 180.1 | 19.9 | 133.5 | 334.2 | 621.2 | 1342.6 |

The reused complete lane was 99.6--100.2% of the separately attributed
component sum. Per-operation session entry made F64 2.4--3.4x and C64
1.9--2.9x slower than session reuse. Eager made F64 5.2--7.2x and C64 4.0--5.9x
slower. The fixed boundary cost remains clear, but its fraction shrinks as
complex arithmetic grows.

A diagnostic run of the existing #1735 benchmark on the same exact schedule
measured compact append versus explicit-Q BCGS2 at approximately 57/401,
77/496, and 131/601 microseconds for 64/128/256 rows. These sub-millisecond
single-process aggregates are diagnostic rather than acceptance evidence. The
256-row append estimate differs by about 17% from the new batched lane, which
reinforces that limitation; even the slower estimate leaves a 4.6x margin over
BCGS2 and cannot explain the downstream 41--52% regression as a slow reflector
kernel.

All benchmark lanes reconstructed the accumulated matrix with relative error
at or below `3.6e-16`; final Q orthogonality error was at or below `5.3e-16`.

## Why the previous delivery was not replacement-optimal

1. The #1735 gate measured append only inside one reused backend session, on
   2,048--32,768 rows. It did not measure the 64--256-row public workflow.
2. Its correspondence ledger explicitly omitted inverse-adjoint/error-estimate
   maintenance. That made the BCGS2 comparison conservative for raw append but
   could not establish downstream replacement competitiveness.
3. The rejected tensor4all adapter opened a default session independently for
   append, R, and selected-Q, then converted every output between `Matrix` and
   tenferro tensors. The concrete tenferro API already permits all three calls
   in one `with_backend_session` closure, as its executable guide demonstrates.
4. The design said append would consume state where Rust permits reuse, but the
   shipped API takes `&self`; CPU append copies the old packed matrix and
   coefficients into new buffers. This is a real design/implementation mismatch,
   though the current measurements show session boundaries dominate it.
5. Python keeps inverse-R information in its compact state and updates the new
   triangular block. Tenferro intentionally stores raw provider-neutral R and
   exposes only full R materialization, so the adapter must extract and
   reinvert R. A generic triangular solve/update seam still needs a benchmark
   including estimator maintenance before it is justified.

## Downstream adapter and SRC rerun

The rejected tensor4all adapter was reconstructed in a disposable worktree at
its recorded tenferro pin `548fd5a1`. The same 64-sequence, seven-process,
alternating-order protocol included Matrix conversion, Q/R retention, selected-Q
copy, and error estimation after every append.

Contrary to the historical 41--52% slowdown, compact Householder was faster
than explicit-Q BCGS2 in fresh reruns:

| dtype | rows 64 | rows 128 | rows 256 |
| --- | ---: | ---: | ---: |
| F64 compact / BCGS2 | 0.496 | 0.437 | 0.505 |
| C64 compact / BCGS2 | 0.579 | 0.551 | 0.613 |

The C64 adapter-complete variant that copied newly selected Q columns and ran
the estimator after every append remained 39--45% faster than BCGS2. Thus the
reported adapter regression is not reproducible on the current recorded source
and pinned environment.

Two measured candidates were rejected:

- Combining append/R/Q under one downstream default-session closure changed
  medians by -2.1%, -1.6%, and +1.1% at 64/128/256 rows: neutral. Maximum
  process-median CoV was 2.0%.
- Reusing the existing BCGS2 block inverse-adjoint updater in the compact path
  was 27--44% slower. Maximum process-median CoV was 2.9%. At rank <=32, its
  small solve, multiple Matrix matmuls, and allocations cost more than the
  existing full triangular solve.

Finally, seven paired end-to-end adaptive SRC processes (10 sites,
`rank_increment=3`, `max_rank=32`, C64 sketch path) measured compact/BCGS2
ratios of 0.982, 0.955, and 0.986 at input bonds 32, 64, and 128. Compact was
neutral to 4.5% faster, with process CoV below 6.3%; no correctness tolerance
was changed. This is insufficient evidence for a broad replacement speedup
claim, but it rules out the historical slowdown on the tested source.

## API and rule findings

- **Concrete API:** sufficient for session reuse. Although per-operation entry
  is visible in isolation, downstream session fusion was neutral, so no new
  public API is justified for this part.
- **Eager API:** every extension operation reaches
  `with_extension_execution_context`, locks runtime/cache state, and opens a
  backend session. It has no public sequence scope that lets append/R/Q share
  one session. This is the measured tenferro API gap.
- **Ownership contract:** `append_columns(&self, ...) -> Self` contradicts the
  accepted design's consume-and-reuse statement. Any future ownership-based
  optimization needs an owned backend hook and a source/performance contract;
  changing only the public receiver would not remove provider copies.
- **Triangular estimator contract:** adding raw packed access would leak provider
  state and is rejected. The existing block inverse updater was slower at the
  measured ranks, so a new solve/update API is not justified by current data.
- **Rule gap:** the shared performance protocol requires an end-to-end need gate
  for optimization candidates, but it did not require a proposed replacement
  API to benchmark the exact downstream public operation sequence, mandatory
  outputs, conversions, smallest representative shapes, and session policy.
- **Rule gap:** benchmark guidance separates setup from operation timing but does
  not require paired internal-session and intended-public-surface lanes when
  fixed per-call dispatch may dominate.
- **Rule enforcement gap:** the reviewed design's consume/reuse promise was not
  tied to a source-contract test, allowing the shipped `&self` API and full
  state copy to pass review.

The minimal rule correction is to tighten the existing performance-gated
experiment protocol, not add another standalone policy: replacement claims must
include the exact intended public workflow and its mandatory output/state
maintenance, plus an internal boundary breakdown. Performance claims based on
ownership reuse must identify and test the owned execution seam.

## Verification

- `cargo check -p tenferro-linalg --features autodiff --bench householder_qr_small_appends`
- `cargo test -p tenferro-linalg --test integration small_append_benchmark_pins_issue_1750_workflow`
- release smoke and seven-process faer measurements for every benchmark lane
- reconstruction and orthogonality checks embedded in every benchmark record

## Next steps

1. Land the benchmark and retain both F64 and C64 rows.
2. Do not land the neutral session-fusion candidate or the slower block-inverse
   candidate.
3. Treat #1750's optimization request as not currently reproducible; reopen the
   API/kernel decision only if the long-term suite shows a regression.
4. Track the operation sequence in
   [tenferro-benchmark#94](https://github.com/tensor4all/tenferro-benchmark/issues/94)
   so future reports retain both reused-session and public/eager boundary costs.
