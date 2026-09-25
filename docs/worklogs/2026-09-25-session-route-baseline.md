# Session-route unification baseline (#1926 / umbrella #1929)

## Session summary

Umbrella #1929 was reordered so benchmarks land before the route/API
unification (#1926). The unification deletes the one-shot operation spelling
and keeps the session surface, which changes the execution path that every
later performance claim is about; without a measured pre-refactor baseline the
regression question cannot be answered afterwards.

This session added the missing before-side instrument, collected the baseline
at the pinned pre-refactor revision, and committed it as a machine-readable
artifact.

## Artifacts

| Artifact | Purpose |
|---|---|
| `crates/tenferro-cpu/benches/route_matrix.rs` | the same logical operation through every coexisting CPU entry mechanism, per-op and marginal-in-entry |
| `crates/tenferro-runtime/benches/session_chain.rs` | 10-op chain through one session vs one-shot-per-op vs one execution scope (arms added in this session) |
| `scripts/run-session-route-performance-gate.sh` | campaign runner: pinned baseline, 1T environment, CPU affinity, recorded manifest |
| `scripts/capture-session-route-baseline.py` | run logs → versioned JSON, fail-closed on case-count drift |
| `scripts/compare-session-route-baseline.py` | baseline vs candidate paired comparison with predeclared thresholds |
| `docs/testing/session-route-baseline.json` | the baseline data (`tenferro.session-route-baseline.v1`) |

The campaign is deliberately separate from the execution-engine terminal gate
(`scripts/run-unification-performance-gate.sh`), which pins its own baseline
commit `c6418eecf` and its own harness identity. Reusing that campaign with a
different pin would corrupt both records.

## Baseline identity and environment

- harness commit: `1202238872a2fc7e68f40d5960aa37c3be3f83ae`
- library baseline commit: `25da8d431` (pre-unification `origin/main`)
- CPU affinity: `taskset -c 0`; `RAYON_NUM_THREADS=OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=VECLIB_MAXIMUM_THREADS=NUMEXPR_NUM_THREADS=TENFERRO_BENCH_THREADS=1`
- `nproc` as seen by the runner: 1 — this is the effective-thread confirmation, not host capacity (GNU `nproc` honours `OMP_NUM_THREADS`)
- host 1-minute load average at start: 4.19 on a 64-CPU host → normalised 0.065, under the 0.25 host-noise gate
- criterion: warm-up 2 s, measurement 5 s, sample size 100
- cases: 6 targets, 99 cases, of which 17 are before-only (deleted-route) references

## Findings

### 1. The premise of #1926 is confirmed at the chain level

10-op chain (`add → exp → mul` ×3, then `reduce_sum([0])`), one worker:

| Route | Time | Entries |
|---|---:|---:|
| `one_shot` (current per-op API) | 83.35 µs | 10 |
| `one_session` | 22.13 µs | 1 |
| `execution_scope` | 20.58 µs | 1 (permit reused) |

**3.77× for the route that #1926 unifies.** This is the number that must not
regress and that the post-refactor eager path should inherit.

### 2. Entry cost vs per-operation body cost

Elementwise `add`, 1 element, 16 operations in one entry:

| Route | 16 ops total | per op |
|---|---:|---:|
| `oneshot/marginal16` | 133.47 µs | 8.34 µs |
| `session/marginal16` | 28.09 µs | 1.76 µs |
| `scope/marginal16` | 28.22 µs | 1.76 µs |

So a fresh entry costs roughly **6.6 µs** on this host and the kernel plus
dispatch is **~1.76 µs**. The 6.6 µs figure is the whole prize: it is what a
caller can amortize, and what the unified route must keep amortizable.

### 3. Price of the unification on the single-operation route

The one-shot spelling is what users will be moved off. Per-operation
single-entry comparison, before side:

| Case | `oneshot/single` | `session/single` | delta |
|---|---:|---:|---:|
| add f64, len 1 | 8.81 µs | 9.12 µs | +3.5% |
| add f64, len 65536 | 29.08 µs | 29.59 µs | +1.7% |
| dot_general f64, 2×2 | 9.33 µs | 9.25 µs | −0.9% |
| dot_general f64, 256×256 | 638.81 µs | 638.06 µs | −0.1% |
| reduce_sum f64, 65536 | 15.91 µs | 16.33 µs | +2.6% |
| slice f64, 64 | 8.94 µs | 9.13 µs | +2.1% |
| cast f64→f32, 4096 | 9.18 µs | 9.32 µs | +1.6% |

The session entry is **~0.2–0.3 µs more expensive per single operation** than
`install_with_pool_context` on this host, and free on contraction-dominated
cases. That is the expected size of the change and sets the scale of the
predeclared threshold: a microsecond-scale case is expected to move by a few
percent, not by tens of percent.

### 4. Surviving system-level cases (must not regress)

| Case | Baseline |
|---|---:|
| eager lazy `neg_f64`/`add_f64`/`reduce_sum_f64`, len 1–64 | 11.45 / 13.37 / 12.09 µs (len 1) |
| eager materialized `add_f64`, len 1–64 | 15.39 µs (len 1) |
| eager lazy `slice_f64`, len 1–64 | 2.13–2.17 µs |
| eager lazy `dot_general_f64`, 1–2 | 13.75–13.99 µs |
| graph `add_mul` prepared / unprepared, 4096 | 51.31 / 116.07 µs |
| graph `add_mul` prepared, 1048576 | 18.92 ms |
| graph `broadcast_mul_add` prepared, 1024×1024 | 19.00 ms |
| einsum/linalg VJP `triangular_solve_vjp` 8 / 16 | 323.88 / 325.31 µs |
| linalg VJP `svd_values_vjp` 8 / 16 | 339.42 / 339.30 µs |
| eager backward shape churn, 16 bond-dimension pairs | 7.2688 ms |

## Comparison rules fixed before the refactor

Thresholds are predeclared and encoded in
`scripts/compare-session-route-baseline.py`:

- baseline median ≤ 10 µs → blocking regression at **+50%**, and a blocking
  finding requires reproduction in a second complete paired run;
- other eager small-op cases → **+5%**;
- shape-churn, prepare, and linalg VJP → **+10%**;
- overlapping baseline/candidate intervals → `NOISY` (unresolved, not a pass
  claim);
- a baseline case that is neither a deleted-route case nor present in the
  candidate → **failure** (fail-closed).

Deleted-route cases are `*/oneshot/*` and `*/one_shot`. They cannot be
reproduced after the refactor; their baseline values above are the only record.

## Known limitations of this baseline

- The `execution_scope` arms in `route_matrix.rs` and `session_chain.rs` are
  implemented with the one-shot spelling on a backend clone. After the
  refactor that spelling is gone, so those arms must be rewritten (scope +
  session) and become **non-paired**: the width of the scope mechanism is
  measured, but the two runs are not the same workload. They are recorded here
  for the mechanism comparison only.
- `route_matrix` scope arms are the same logical operation as the one-shot
  arms, but `with_execution_scope` requires a Tenferro-managed CPU domain, so
  the scope rows do not generalise to externally managed executors.
- GPU routes are not in this campaign. The CUDA/WebGPU one-shot
  `Tensor*` implementations are a separate measurement and need GPU-capable
  hardware.
- One complete baseline run was collected. The terminal claim requires three
  alternating complete pairs (baseline→candidate, candidate→baseline,
  baseline→candidate) with unchanged settings; that happens at the end of the
  refactor, not here.

## Next

Phase B of the reordered umbrella: the route/API unification itself, on a
branch off this baseline, followed by the paired candidate run and the
`compare-session-route-baseline.py` gate.
