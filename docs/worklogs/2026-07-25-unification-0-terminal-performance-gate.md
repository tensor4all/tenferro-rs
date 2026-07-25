# Unification 0 terminal performance gate

## Session summary

This worklog defines the terminal performance gate for the execution-path
unification sequence tracked by #1433 and #1454. It fixes the baseline commit,
workload set, measurement protocol, statistic, threshold, repetition policy,
and runner script before terminal candidate data is collected.

The gate is intentionally terminal. Intermediate measurements during
Unification 1 through 7 are diagnostics only; they are not baselines and do not
close the non-inferiority requirement.

## Context read

- #1433 unification sequence and comments through issue comment `5076395906`.
- #1454 terminal performance-gate issue.
- `REPOSITORY_RULES.md`, especially Performance-Gated Experiment Protocol.
- `docs/design/execution-engine-provider-architecture.md`, especially Gate C.
- `docs/superpowers/specs/2026-07-20-execution-engine-provider-umbrella-design.md`.
- `docs/superpowers/specs/2026-07-24-execution-engine-phase9-restart-design.md`.
- `docs/worklogs/2026-07-20-eager-main-baseline.md`.
- `docs/worklogs/2026-07-25-phase-8-pr-benchmark-gate.md`.
- Existing benchmark sources:
  - `crates/tenferro-ad/benches/eager_dispatch_baseline.rs`
  - `crates/tenferro-runtime/benches/elementwise_fusion.rs`
  - `crates/tenferro-einsum/benches/einsum_cpu_bench.rs`
  - `crates/tenferro-einsum/benches/mps_inner_product.rs`
  - `crates/tenferro-ad/benches/eager_ad_transform_cache.rs`
  - `crates/tenferro-linalg/benches/lu_ad_breakdown.rs`

## Baseline identity

The pinned pre-migration baseline commit is:

```text
c6418eecfe2d38ca09d6e6386760fcb23982691e
```

This is `origin/main` at the start of Unification 0 in this worktree.

Several terminal-gate benchmark targets do not exist on that baseline commit.
The baseline measurement therefore uses the fixed benchmark harness from the
integration branch applied to the pinned baseline implementation, without
candidate implementation changes. The harness identity is the commit that adds
this worklog and `scripts/run-unification-performance-gate.sh`; if the harness
source changes after collecting baseline numbers, the baseline collection must
restart.

The earlier eager baseline at `85855e272b1495611deb601a9ee06f3546772c3c` remains
historical evidence for Phase 1. It is not the terminal baseline for this
campaign because the terminal gate pins `c6418eec...`.

## Workload matrix

All workloads run with one logical CPU thread unless a workload-specific child
issue explicitly extends the matrix. Thread-related environment variables are
fixed to one: `RAYON_NUM_THREADS`, `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and
`NUMEXPR_NUM_THREADS`.

| Workload | Package / bench target | Cargo features | Required cases | Gate purpose |
|---|---|---|---|---|
| Eager small-op dispatch | `tenferro-ad` / `eager_dispatch_baseline` | default | lazy and materialized `neg_f64`, `add_f64`, `reduce_sum_f64`, indexed `slice_f64` for lengths 1, 8, 64; lazy and materialized `dot_general_f64` for 1x1 and 2x2 | fixed eager orchestration cost |
| Graph steady-state execution | `tenferro-runtime` / `elementwise_fusion` | default | add-mul for 4,096, 65,536, 1,048,576 elements; broadcast-mul and broadcast-mul-add for 256x256 and 1024x1024 | compile-once execute-many graph replay |
| Changing-shape einsum prepare throughput | `tenferro-einsum` / `changing_shape_prepare` | `autodiff` | 129 distinct `abc,cde,ef->abdf` concrete shapes, exceeding the default prepared-plan cache entry limit of 128 | prepare cost under normal shape churn, not same-shape cache hits |
| Eager backward shape churn | `tenferro-ad` / `eager_backward_shape_churn` | default | 16 DMRG-like left/right bond-dimension pairs with captured same-shape weights and repeated scalar-loss backward calls | semantic transform/cache path versus tidu-era eager backward behavior |
| Linalg VJP | `tenferro-linalg` / `linalg_vjp_gate` | `autodiff` | triangular-solve reverse pass and SVD-singular-value reverse pass at sizes 8 and 16 | extension-bearing AD evaluation through the unified runtime |

The final three target names are reserved by this gate. They may be introduced
by later Unification issues, but their case list cannot be selected after
observing terminal candidate results.

## Measurement command

The reproducible runner is:

```console
bash scripts/run-unification-performance-gate.sh --mode run --label baseline
bash scripts/run-unification-performance-gate.sh --mode run --label candidate
```

For local protocol checks that should not build or execute benchmarks:

```console
bash scripts/run-unification-performance-gate.sh --mode dry-run
```

The script records a manifest under `target/unification-performance-gate/` and
runs each Criterion benchmark with:

- warm-up time: 2 seconds;
- measurement time: 5 seconds;
- sample size: 100;
- one-thread CPU/backend/provider environment;
- `taskset -c 0` when `taskset` is available.

## Statistic and threshold

The primary statistic is Criterion's per-case relative-change interval between
the pinned baseline and terminal candidate. A case passes when the upper bound
of the 95% relative-change interval is at or below its threshold.

Thresholds:

- Microsecond-scale dispatch/orchestration cases whose baseline median is at
  most 10 us use the restart protocol: a blocking regression requires at least
  +50% slowdown on a predeclared primary case and reproduction in a second
  complete paired A/B run.
- Other eager small-op cases use +5% upper 95% CI bound.
- Graph steady-state throughput cases use +5% upper 95% CI bound on time per
  iteration, with throughput recorded diagnostically.
- Changing-shape einsum prepare throughput uses +10% upper 95% CI bound.
- Eager backward shape churn uses +10% upper 95% CI bound.
- Linalg VJP uses +10% upper 95% CI bound.

The campaign passes only when every required case passes. It fails when any
case exceeds its blocking threshold. Valid runs that are neither pass nor fail
are classified `INCONCLUSIVE`.

Median point estimates are diagnostic. They do not override confidence bounds.

## Repetition and ordering

Run three complete paired comparisons in alternating order:

1. baseline then candidate;
2. candidate then baseline;
3. baseline then candidate.

Every pair uses the same harness source, Rust toolchain, Cargo profile,
features, thread settings, affinity, and case list. Do not retry individual
cases or selected favorable pairs. If a retry is needed, rerun the complete
three-pair campaign with unchanged settings.

## Validity and noise gates

One invalid observation makes the complete campaign `INCONCLUSIVE`.

Invalid observations:

- the process loses the requested CPU affinity;
- a benchmark process overlaps with unrelated local Cargo or rustc work;
- normalized one-minute host load exceeds 0.25 of the process-allowed CPU
  count at a pair boundary;
- any required case is missing;
- any benchmark target fails to build or exits nonzero;
- the harness source differs between baseline and candidate;
- the baseline commit differs from `c6418eecfe2d38ca09d6e6386760fcb23982691e`.

Intermediate measurements on Unification 1 through 7 must be labeled
`diagnostic` or `transitional`. They cannot be copied into the terminal
baseline table.

## Baseline collection status

Baseline collection for the full terminal matrix has not been performed in this
worktree. The runner now exposes all reserved targets in dry-run mode; #1454
still cannot close until the pinned-main baseline table is collected with this
fixed harness source.

The prior eager-only table in
`docs/worklogs/2026-07-20-eager-main-baseline.md` remains useful for comparison
sanity checks, but it is not the complete #1454 baseline table.

### Invalid baseline attempt: 2026-07-25

An attempted baseline run against pinned commit
`c6418eecfe2d38ca09d6e6386760fcb23982691e`, with the harness files from commit
`effa6ca1`, was stopped during `tenferro-runtime/elementwise_fusion`.

Classification: `INCONCLUSIVE`, not a baseline.

Reason: external CPU load was present during the run. `ps` showed a separate
`cargo check -p koushi-desktop` with multiple `rustc` processes at 100% CPU and
several long-running Julia kernels at roughly 100% CPU while the benchmark
process was running. This violates the host-noise validity gate, so all partial
Criterion output from the interrupted run is diagnostic only and must not be
copied into the baseline table.

Partial logs were written under:

```text
target/unification-performance-gate/baseline-c6418-effa6ca1/
```

## Remaining risks and follow-up

- Baseline numbers must be collected from the pinned baseline implementation
  with the fixed harness source applied.
- The final campaign needs an extraction/reporting step that turns Criterion
  estimates into the pass/fail/inconclusive table required by the repository
  protocol.
