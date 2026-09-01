# Incremental Householder QR Phase 5 performance gate

### Scope and immutable identities

Phase 5 adds the benchmark/gate harness and user documentation. A failed
frozen CUDA primary gate may admit only a separately reviewed provider
optimization followed by a complete rerun; the accepted candidate uses the
blocked-WY reflector application documented in
[`incremental-householder-qr.md`](incremental-householder-qr.md). The baseline algorithm is the explicit-Q two-pass BCGS2 implementation from
tensor4all-rs#694 commit `da0775a208006352f6e5eab18bc6bb09ca39a1f6`,
source `crates/tensor4all-tensorbackend/src/incremental_qr.rs`, reproduced in the
benchmark with tenferro backend primitives. Before timing, the Phase-5 design
review freezes a [line-by-line correspondence ledger](../performance/incremental-householder-qr-bcgs2-ledger.md) for its two
projection/reconstruction passes, correction accumulation, residual-only QR,
and block Q/R assembly. A missing/unreviewed ledger blocks the gate. Repeated one-shot QR of the
accumulated matrix is a diagnostic baseline, not the primary comparator.

The worklog records before any run: exact commits, dirty-state check, compiler,
release profile, benchmark/checker source hashes, CPU/GPU model, NUMA/affinity, provider
versions, thread variables, CUDA runtime/driver, case list, repetitions, and
host load/frequency observations.

### SRC-derived matrices

Use deterministic F64 Gaussian matrices with seed 7 and column scaling bounded
away from rank deficiency. For each tensor4all-rs#694 bond `b in {32,64,128}`:

- rows `m = 2 * b * b` (physical dimension 2 times MPO×MPS product bond);
- initial rank 2;
- appended block width 3;
- append until rank 32 (the accepted adaptive maximum rank);
- rank increment 3, matching #694.

Secondary shape cases use rows 4096 and initial rank 2. Width 1 runs 30
appends to rank 32; width 8 runs three full appends to rank 26 (the next full
block would exceed the maximum rank 32, and no partial block is synthesized). Correctness inputs are identical across compact, BCGS2, and one-shot
paths.

### Algorithms

1. **compact**: `HouseholderQr::from_factors` for the initial state followed by
   `append_columns`; no Q materialization during timed append.
2. **bcgs2**: explicit Q/R state; two `Q^H B` projection/reconstruction passes,
   QR only of the residual block, then explicit block Q/R assembly.
3. **full-qr diagnostic**: concatenate and refactor the complete accumulated
   matrix after every append.

Setup, deterministic input generation, initial factorization, final Q/R
materialization, and correctness checks are outside timed regions. CPU provider
objects and GPU contexts/handles are constructed once and reused.

### Measurement

- Release profile, warmed incremental build; all three algorithm arms are in
  the same binary/commit, so build state cannot favor one arm. Runtime timing
  uses Rust's monotonic `std::time::Instant`; `std::hint::black_box` wraps
  inputs/results.
- CPU: one pinned logical CPU on the local AMD EPYC 7713P; one-thread
  `RAYON_NUM_THREADS=OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`.
  Run CPU-faer and CPU-BLAS separately.
- CUDA: local NVIDIA A100 80GB, one reused CUDA backend, CUDA 12.6 local tier;
  synchronize immediately before and after each timed aggregate sample (the
  fixed batch of four complete append sequences).
- Three untimed warmups, then seven independent process cycles. Cycles 1/3/5/7
  run `compact -> bcgs2 -> full-qr`; cycles 2/4/6 run the exact reverse order.
  Each arm is one fresh process and performs five repetitions for bonds 32/64
  and three for bond 128. Each timing sample is a fixed batch of four
  independently reset append sequences; initial-state preparation remains
  outside timing. The aggregate batch time is recorded and all arms of every
  case use the same batch. The reported statistic is the median of the seven
  per-process medians. A fixed untimed 50 ms spin on the single pinned logical
  CPU immediately before each aggregate sample stabilizes its frequency.
- Record every raw aggregate timing and a fixed-seed 10,000-resample 95%
  bootstrap confidence interval for each reported median and paired ratio. No
  selective reruns or case omission. A batched process median below 1 ms is
  `INCONCLUSIVE` because it still lacks adequate resolution.

A case is `INCONCLUSIVE` if process-median coefficient of variation exceeds
10%, system load exceeds 1.5× `max(pre-run load, 0.1)` in any cycle, or
thermal/frequency observations differ by more than 10%. Before measured
subprocesses start, five 50 ms active samples on the single pinned CPU define an
independent median reference. Each benchmark process reads the same CPU after
each warm and before timing; its median must remain within 10% of the active
reference, and process affinity must exactly equal runner affinity. GPU clock
variation is compared with the median active process clock because the pre-run
clock is idle; every nonzero throttle reason remains invalid. CUDA errors, or a
correctness check fails solely for a demonstrated environmental reason. A
reproducible numeric or source-contract failure is `FAIL`, never
`INCONCLUSIVE`. Reconsideration requires a
complete paired rerun.

### Correctness and scaling gates

For every measured path/case:

- reconstruction relative Frobenius error <= `5e-11` on CPU and `2e-9` on CUDA;
- Q orthogonality relative Frobenius error <= `5e-11` on CPU and `2e-9` on CUDA;
- compact agrees with one-shot positive-diagonal R to relative error <= `1e-9`;
- source contracts continue to reject full-QR append and host tensor transfer.

Performance is `PASS` only if all conditions hold:

- At bonds 64 and 128, compact median append-sequence time is no slower than
  explicit-Q BCGS2 for both CPU providers; bond 32 may be at most 15% slower.
- At least one of bonds 64/128 shows >=5% compact improvement over BCGS2 for
  each CPU provider.
- At bond 128, compact is at least 2× faster than repeated full QR.
- CUDA compact is no slower than CUDA BCGS2 at bonds 64/128 and at least 2×
  faster than repeated full QR at bond 128. The bond-32 15% allowance and the
  >=5% improvement requirement apply only to the two CPU providers.
- The scaling case is fixed at `m=32768` (bond 128), width 3, prior ranks
  8/16/29, and runs on CPU-faer, CPU-BLAS, and CUDA. It must not exhibit
  one-shot-QR scaling: compact append time divided by the
  `(m * prior_rank * block_width)` work proxy may grow by at most 35%.
  The exact-candidate source-contract commands must pass and their logs must be
  included in the artifact; there is no unimplemented runtime-counter
  dependency.

The #694 downstream end-to-end reference medians are recorded as context:
54.656/108.898/433.856 ms at bonds 32/64/128 for optimized Rust SRC and
11.358/56.101/348.851 ms for the independently generated Python/LAPACK
Householder diagnostic. They are not substituted for the same-input
microbenchmark ratios, but absence of the exact commit/source/ledger above is a
validity failure.

Any failed performance threshold is `FAIL`, not reclassified post hoc. A failed
primary gate requires a kernel optimization under a newly reviewed candidate
and a complete paired rerun; thresholds remain unchanged.

### Deliverables

- `crates/tenferro-linalg/benches/incremental_householder_qr.rs` with compact,
  BCGS2, and full-QR paths and machine-readable JSON/CSV output;
- a deterministic checker script that classifies every case PASS/FAIL/
  INCONCLUSIVE against this frozen protocol;
- source-contract/unit tests for benchmark case inventory and thresholds;
- Phase-5 worklog with raw artifacts and exact commands;
- final user documentation of compact-state usage, explicit device transfer,
  provider selection, AD full-rank domain, CUDA requirements, and measured
  performance scope.