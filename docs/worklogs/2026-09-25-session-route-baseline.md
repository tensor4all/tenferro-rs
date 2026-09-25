# Session-route unification baseline (#1926 / umbrella #1929)

## Session summary

Umbrella #1929 was reordered so benchmarks land before the route/API
unification (#1926). The unification deletes the one-shot operation spelling
and keeps the session surface, which changes the execution path that every
later performance claim is about. Without a measured pre-refactor baseline the
regression question cannot be answered afterwards.

This record covers the frozen harness, the baseline data, and the comparison
rules. It was captured once, after all Phase A harness changes landed, because
changing the harness invalidates baseline identity.

## Artifacts

| Artifact | Purpose |
|---|---|
| `crates/tenferro-cpu/benches/route_matrix.rs` | the same logical CPU operation through every coexisting entry mechanism, per-op and marginal-in-entry |
| `crates/tenferro-runtime/benches/session_chain.rs` | 10-op chain through one session vs one-shot-per-op vs one execution scope, plus the phase-1 mixed chain |
| `crates/tenferro-gpu/benches/route_matrix_gpu.rs` | the same contraction through the CUDA one-shot method vs a borrowed session, with enqueue and synchronized completion separate |
| `scripts/run-session-route-performance-gate.sh` | campaign runner: pinned baseline, 1T provider/runtime environment, CPU affinity, CUDA runtime env for GPU targets, recorded manifest |
| `scripts/capture-session-route-baseline.py` | run logs → versioned JSON; fail-closed on a dirty worktree, a missing target log, or a case-count drift |
| `scripts/compare-session-route-baseline.py` | baseline vs candidate paired comparison with predeclared thresholds |
| `docs/testing/session-route-baseline.json` | the baseline data (`tenferro.session-route-baseline.v1`) |

The campaign is deliberately separate from the execution-engine terminal gate
(`scripts/run-unification-performance-gate.sh`), which pins its own baseline
commit `c6418eecf` and its own harness identity. Reusing that campaign with a
different pin would corrupt both records.

## Baseline identity and environment

- harness commit: `b16c8ce3f348d0e28a53958f2b3ab92a1a8fc7db`
- library baseline commit: `25da8d431` (pre-unification `origin/main`)
- CPU affinity: `taskset -c 0`; `RAYON_NUM_THREADS=OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=VECLIB_MAXIMUM_THREADS=NUMEXPR_NUM_THREADS=TENFERRO_BENCH_THREADS=1`
- `nproc` as seen by the runner: 1 — effective-thread confirmation, not host capacity (GNU `nproc` honours `OMP_NUM_THREADS`)
- host 1-minute load average at start: 10.89 on a 64-CPU host → normalised 0.17, under the 0.25 host-noise gate; 5.12 after the run
- competing Cargo/rustc at start and end: none (this host runs unrelated `cargo test` jobs from other work intermittently; the run waited for a quiet window, five-minute polling)
- GPU: NVIDIA A100 80GB PCIe, driver 580.126.09, CUDA root `/usr/local/cuda-12.6`, cuTENSOR `/usr/lib/x86_64-linux-gnu/libcutensor/12`
- criterion: warm-up 2 s, measurement 5 s, sample size 100
- cases: 7 targets, 125 cases, of which 29 are before-only (deleted-route) references

### Residual host condition (not gated, recorded for transparency)

Three long-lived Julia IJulia kernels were running at 100% of one core each
throughout, with an unrestricted affinity mask (`0-63`). They are not Cargo or
rustc work, so they do not trip the written validity gates, and the load gate
passed. Because the benchmark is pinned to CPU 0, contention on that core is
possible in principle. The affected comparison is baseline-versus-candidate
within this record, so the condition applies to both sides.

## Findings

### 1. The premise of #1926 is confirmed at the chain level

10-op chain (`add → exp → mul` ×3, then `reduce_sum([0])`), one worker:

| Route | no-broadcast | broadcast |
|---|---:|---:|
| `one_shot` (current per-op API) | 92.42 µs | not expressible |
| `one_session` | 22.80 µs | 35.97 µs |
| `execution_scope` | 23.90 µs | 35.64 µs |

**4.05× for the route that #1926 unifies** (no-broadcast). The broadcast chain
has no one-shot form at all: `TensorElementwise::add`/`mul` require equal
shapes and broadcasting lives in the session surface, so the coexistence is not
two spellings of one thing.

### 2. Entry cost decomposes into permit acquisition, not session construction

Elementwise `add`, 1 element, 16 operations:

| Route | 16 ops total | per op | construction |
|---|---:|---:|---|
| `oneshot/marginal16` | 147.11 µs | 9.19 µs | fresh entry per operation |
| `session/marginal16` | 31.28 µs | 1.96 µs | one session, one permit |
| `scope/marginal16` | 34.78 µs | 2.17 µs | one permit, 16 separate session entries |

An additional session entry inside an open execution scope costs **~0.17 µs**,
while a fresh standalone entry costs **~7.2 µs** over the same work. The
dominant entry cost is therefore permit acquisition, not `CpuExecSession`
construction. This is the decomposition the unification must preserve.

### 3. Price of the unification on the single-operation route

| Case | `oneshot/single` | `session/single` | delta |
|---|---:|---:|---:|
| add f64, len 1 | 9.64 µs | 10.03 µs | +4.0% |
| reduce_sum f64, 65536 | 15.91 µs (prev.) | 16.33 µs (prev.) | +2.6% |
| slice f64, 64 | 8.94 µs (prev.) | 9.13 µs (prev.) | +2.1% |
| cast f64→f32, 4096 | 9.18 µs (prev.) | 9.32 µs (prev.) | +1.6% |
| dot_general f64, 256×256 | 761.29 µs | 698.58 µs | −8.2% |

The session entry is a few percent above `install_with_pool_context` on micro
cases and free on contraction-dominated cases. The `oneshot/single` and
`session/single` rows for elementwise at len 1 are the cleanest same-run pair.

### 4. GPU: the routes are equivalent, and launch is separated from completion

`CudaBackend` implements the operation traits by calling the same functions as
its `BackendSession` impl, so the two routes differ only by the session wrapper.
That is what the data shows. f64:

| size | `oneshot/round_trip` | `session/round_trip` | `oneshot/enqueue_batch16` | `session/enqueue_batch16` |
|---|---:|---:|---:|---:|
| 8 | 48.65 µs | 48.42 µs | 437.89 µs (27.4/op) | 433.36 µs (27.1/op) |
| 64 | 64.59 µs | 65.34 µs | 631.81 µs (39.5/op) | 630.11 µs (39.4/op) |
| 256 | 59.74 µs | 60.04 µs | 558.49 µs (34.9/op) | 558.13 µs (34.9/op) |
| — | — | — | `sync_empty/synchronize` = 17.51 µs | — |

No measurable per-operation difference between the two routes (−0.5% to +1.2%,
inside noise). Round-trip time exceeds enqueue-only time by roughly the
`synchronize` cost plus a residual, so enqueue and completion are separated as
the umbrella requires. f32 is the same shape.

### 5. Surviving system-level cases (must not regress)

| Case | Baseline |
|---|---:|
| eager lazy `neg_f64` / `add_f64` / `reduce_sum_f64`, len 1 | 11.45 / 13.37 / 12.09 µs |
| eager materialized `add_f64`, len 1 | 15.39 µs |
| eager lazy `slice_f64`, len 1 | 2.17 µs |
| eager lazy `dot_general_f64`, 1 | 13.75 µs |
| graph `add_mul` prepared / unprepared, 4096 | 51.31 / 116.07 µs |
| graph `add_mul` prepared, 1048576 | 18.92 ms |
| graph `broadcast_mul_add` prepared, 1024×1024 | 19.00 ms |
| linalg VJP `triangular_solve_vjp` 8 / 16 | 323.88 / 325.31 µs |
| linalg VJP `svd_values_vjp` 8 / 16 | 339.42 / 339.30 µs |
| eager backward shape churn, 16 bond-dimension pairs | 7.2688 ms |

## Comparison rules fixed before the refactor

Encoded in `scripts/compare-session-route-baseline.py`:

- baseline median ≤ 10 µs → blocking regression at **+50%**, and a blocking
  finding requires reproduction in a second complete paired run;
- other eager small-op cases → **+5%**;
- shape-churn, prepare, and linalg VJP → **+10%**;
- overlapping baseline/candidate intervals → `NOISY` (unresolved, not a pass
  claim);
- a baseline case that is neither a deleted-route case nor present in the
  candidate → **failure** (fail-closed).

Deleted-route cases are `*/oneshot/*` and `*/one_shot` (29 of them). They
cannot be reproduced after the refactor; the values above are the only record.

## Known limitations

- `with_execution_scope` requires a Tenferro-managed CPU domain, so the scope
  rows do not generalise to externally managed executors.
- The scope arms now use `with_execution_scope` + `with_backend_session`, both
  of which survive the unification, so they remain pairable. Earlier versions of
  these arms used the one-shot spelling and would have become non-paired; that
  was corrected before this capture.
- GPU route rows are CUDA only. WebGPU is not measured: this host has no
  Vulkan/WebGPU runtime, and `WebGpuExecSession` is a forwarding shim of the
  same shape as `CudaExecSession`, so the GPU conclusion is expected to carry
  over without being measured.
- GPU wall-clock times include cuTENSOR plan lookup and device allocator
  behaviour, which the two routes share.
- One complete baseline run was collected. The terminal claim requires three
  alternating complete pairs with unchanged settings; that is Phase C.

## Process note

An earlier attempt to capture this baseline was aborted and not recorded: the
host showed a live unrelated `cargo test -p sparse-ir` (plus rustc at 100%)
and a one-minute load of up to 49.75, above the 0.25 gate. The capture ran at
the first quiet window found by five-minute polling. No partial data from the
aborted window was used.

## Next

Phase B of the reordered umbrella: the route/API unification on a branch off
this baseline, then the Phase C paired candidate run through
`compare-session-route-baseline.py`.
