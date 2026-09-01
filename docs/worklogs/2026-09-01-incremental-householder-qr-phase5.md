# Incremental Householder QR Phase 5: performance gate (#1735)

## Session summary

Added the frozen benchmark/checker, a reviewed tensor4all-rs#694 BCGS2
correspondence ledger, and final user documentation. Measurements compare
compact Householder append with an explicit-Q two-pass BCGS2 baseline and a
repeated full-QR diagnostic on CPU-faer, CPU-BLAS, and CUDA.

## Context reviewed

- issue #1735 and tensor4all-rs#694 commit
  `da0775a208006352f6e5eab18bc6bb09ca39a1f6`
- `REPOSITORY_RULES.md` Performance-Gated Experiment Protocol
- `docs/design/incremental-householder-qr.md` and all Phase-2/3/4 worklogs
- #694 `IncrementalQr::append`: two projection/reconstruction passes,
  residual-only QR, and block Q/R assembly
- repository benchmark, provider, CUDA synchronization, documentation snippet,
  and source-contract conventions

## Frozen protocol and review gate

The protocol in
`docs/design/incremental-householder-qr-performance-gate.md` was reviewed before
any full-suite timing. `reviewer-flash` required and then approved:

- exact SRC-derived bonds/rows, block schedules, providers, and scaling cases;
- the pinned #694 source and correspondence ledger;
- seven alternating process cycles, fixed repetitions, monotonic timing,
  bootstrap confidence intervals, and noise gates;
- immutable correctness/performance/scaling thresholds;
- deterministic PASS/FAIL/INCONCLUSIVE classification and exact-candidate
  source/hash evidence.

Closure verdict: **Correct-to-merge** with no Critical or Important findings.
The width-8 secondary case is three full blocks ending at rank 26.

## Implementation

- `benches/incremental_householder_qr.rs` emits one machine-readable record per
  process. Setup, initial factorization, final canonicalization, and correctness
  materialization are outside timed append sequences. CUDA synchronizes before
  and after timing.
- `scripts/incremental-householder-qr-performance.py` owns the fixed case
  inventory, process ordering, environment capture, runner, bootstrap
  statistics, thresholds, exact-source hashes, and deterministic classifier.
- `docs/performance/incremental-householder-qr-bcgs2-ledger.md` records the
  one-to-one #694 baseline mapping. Omitting inverse-adjoint estimation favors
  BCGS2 and therefore cannot manufacture a compact-path speedup.
- `docs/guides/linear-algebra.md` documents compact state, append, selected-Q/R,
  device transfer, AD domain, and benchmark scope using an executable tutorial
  snippet.

## Post-implementation review

Bounded reviewer lanes returned **Correct-to-merge** after fixes:

- Rust benchmark math/timing lane: BCGS2 algebra, R assembly, timed boundaries,
  state reset, CUDA synchronization, and error calculations passed review.
- Python runner/checker lane: malformed/incomplete artifacts now emit a
  deterministic FAIL report; paired-ratio bootstrap preserves process pairing;
  environment-aware correctness, CUDA-specific gates, schemas, cycle order,
  source hashes, and current HEAD are enforced.

No Critical, Important, or unresolved Minor finding remains. Absolute paths in
artifact-error diagnostics are intentionally retained because they identify the
failed local boundary and cannot affect the verdict.

## Pre-measurement verification

- CPU and CUDA benchmark feature compilation passed.
- Small compact/BCGS2/full-QR smoke runs reconstructed correctly.
- Python self-test and Rust performance-contract test passed.
- Missing and malformed artifact smoke tests produced deterministic FAIL
  reports.

## First full-suite result and targeted optimization

The complete paired suite ran from clean candidate `6ed32040`. It produced a
real primary-gate FAIL for CUDA bond 128: compact/full-QR was 0.656, above the
frozen 0.5 limit. Compact/BCGS2 was 0.191, so the evidence localized the cost to
per-reflector fixed overhead rather than refactorization or asymptotic work.
The report and raw JSONL remain under the ignored local artifact directory
`target/iqr-performance-6ed32040/`.

A new pre-implementation `reviewer-flash` gate approved a narrow optimization
with verdict **Correct-to-merge**:

- build the packed state's explicit reflector vectors once into device-local
  function scratch V (`0` above, `1` on, packed tails below the diagonal);
- point GEMV/GEMM and GER/GERU at checked `V[j,j]` offsets;
- remove only the old scalar-one and tail device copies from each reflector;
- retain reflector order, coefficient conjugation, target updates, stream
  ordering, pointer-mode restoration, and all no-Q/no-full-QR contracts.

The same review approved a checker-fidelity correction. The frozen rule says
normalized work may *grow* by at most 35% with prior rank; the first checker used
`max/min` and incorrectly failed decreasing normalized cost. Rank-16/29 proxies
are now compared to the rank-8 proxy with the unchanged 1.35 threshold. A
source self-test pins decreasing proxies as acceptable and growth above 35% as
failure.

Focused CUDA correctness tests passed for F32/F64/C32/C64 reconstruction,
append, factor import, wide/rank-deficient/zero cases, and placement. A bounded
bond-128 diagnostic reduced compact time from about 12.73 ms to about 6.37 ms;
this is diagnostic evidence only, not the acceptance run. The post-implementation
`reviewer-flash` closure review traced the six-file diff and returned
**Correct-to-merge** with no Critical or Important findings.

## Measurement status

Thresholds, cases, repetitions, and source correspondence remain frozen. A new
clean exact candidate must rerun the entire paired suite; Phase-5 acceptance
remains pending until its deterministic report is PASS.
