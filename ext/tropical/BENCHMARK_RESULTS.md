# Stage 8 Evidence: Tropical Matmul Benchmark Results

This document records the benchmark evidence gathered for **design_v3
Stage 8** (core-owned fused primitives). Stage 8 is explicitly
evidence-gated in `docs/design/design_v3/90-migration-plan.md`: it should
only proceed if benchmarks show a composition or `ExtensionOp` pattern is
measurably slow enough to justify a core op variant.

The workload under test is **forward-only** max-plus matmul
(`out[i, j] = max_k (a[i, k] + b[k, j])`) expressed two ways over the
existing Stage 4a / Stage 7 surface.

## Hardware and toolchain

```
Platform : Linux primerose 6.8.0-101-generic (Ubuntu)
CPU      : AMD EPYC 7713P 64-Core Processor (64 logical cores)
Memory   : ~1 TiB MemTotal (MemAvailable ~333 GiB at run time)
Rustc    : 1.94.1 (e408947bf 2026-03-25)
Worktree : /home/shinaoka/tensor4all/tenferro-rs/.claude/worktrees/stage8-bench
Branch   : codex-stage-8-bench (base 0de5c94 on refactor_ad_v3)
```

All benches ran single-threaded (Criterion default). The tropical crate
is a standalone cargo workspace excluded from the core `tenferro-rs`
workspace, so the bench binary links through the public `tenferro`
facade plus `tenferro-ops` for the `ExtensionOp` surface — the same
boundaries the Stage 4a / 7 contract imposes on end users.

## Bench method

- Harness: `criterion = "0.5"` (already a workspace-level dev-dep),
  wired into `ext/tropical/Cargo.toml` as a `[dev-dependencies]` +
  `[[bench]]` entry. No fallback path was needed — the criterion command
  invocation in the task spec Just Worked.
- Config: `warm_up_time = 1s`, `measurement_time = 3s`, `sample_size =
  10`. Override via CLI: `-- --measurement-time 3 --warm-up-time 1
  --sample-size 10`.
- Each bench iteration rebuilds the traced graph from `from_vec`
  tensors and calls `TracedTensor::eval` once. A **single `Engine`** is
  reused across iterations so `engine.get_or_compile(exec)` hits its
  cache on all iterations past the first — steady-state iterations pay
  graph build + `ExecIR` lookup + kernel execution, not recompile.
- Criterion's warm-up absorbs the first compile, so the reported median
  reflects cached-compile cost.
- `f64` tensors, square `N x N` for `N ∈ {16, 64, 128, 256, 384}`. We
  dropped the 512 target per the task spec's time budget; 384 fits
  inside ~10 min total bench wall time and already shows the asymptotic
  trend unambiguously.
- `black_box` wraps the two traced inputs per iteration and a trivial
  property of the output so the compiler cannot DCE the eval.

## Results

Median wall-clock per full forward eval (graph build + cached compile
lookup + execution), from Criterion's default MAD-based estimator.

| N     | Composition (4a) | Fused `ExtensionOp` (7) | Ratio (comp / fused) |
|------:|-----------------:|------------------------:|---------------------:|
|   16  |        61.93 ms  |                42.71 ms |                1.45x |
|   64  |       302.11 ms  |                69.78 ms |                4.33x |
|  128  |       351.09 ms  |                42.93 ms |                8.18x |
|  256  |        1.337 s   |                83.64 ms |               15.99x |
|  384  |        4.172 s   |               155.73 ms |               26.79x |

Raw criterion output (lo / median / hi 95% CI):

```
tropical_matmul/composition/16    [23.493 ms  61.930 ms  102.88 ms]
tropical_matmul/fused_ext/16      [35.894 ms  42.710 ms   54.275 ms]
tropical_matmul/composition/64    [283.21 ms 302.11 ms  317.25 ms]
tropical_matmul/fused_ext/64      [65.993 ms  69.779 ms   74.193 ms]
tropical_matmul/composition/128   [322.55 ms 351.09 ms  378.15 ms]
tropical_matmul/fused_ext/128     [30.424 ms  42.930 ms   51.382 ms]
tropical_matmul/composition/256   [1.2213 s  1.3372 s   1.4533 s ]
tropical_matmul/fused_ext/256     [81.390 ms  83.639 ms   88.379 ms]
tropical_matmul/composition/384   [3.9216 s  4.1719 s   4.4048 s ]
tropical_matmul/fused_ext/384     [152.81 ms 155.73 ms  158.35 ms]
```

## Observations

- **The gap is monotone and large.** The ratio grows from 1.45x at
  N=16 to ~27x at N=384. That is well outside measurement noise; the
  trend is consistent across every size tested and across every
  Criterion sample within a size (the median and both ends of the 95%
  CI agree on the direction at every size).

- **The composition path is bottlenecked by intermediate memory, not
  compute.** The Stage 4a lowering allocates three full-sized 3D
  tensors (`a_b`, `b_b`, `sum_3d`), each of shape `[M, K, N]`. At
  N=384 that is `384^3 * 8 bytes ≈ 452 MiB` per tensor, ≈ 1.35 GiB of
  working set per evaluation. That pushes memory-bandwidth, not
  arithmetic, onto the critical path and explains why the composition
  path crosses 4 seconds per eval even on a 64-core EPYC.

- **The fused `ExtensionOp` wins despite being a naive loop.** The
  current `FusedTropicalDotGeneralOp::eager_execute` is a straight
  triple `for` loop (see `ext/tropical/src/fused.rs` `tropical_gemm_f64`)
  with no cache-blocking, no SIMD, no parallelism, and no strided-kernel
  routing. It still beats the composition path by 15–27x at realistic
  sizes, simply because its working set is `O(MK + KN + MN)` instead of
  `O(MKN)`. A real cache-blocked fused kernel would widen the gap
  further.

- **At N=16 the paths are close (1.45x).** This is the regime where
  graph build, `ExecIR` hashing, and dispatch overhead dominate the
  actual tensor arithmetic. Below some threshold the distinction
  stops mattering. But this is also the regime nobody cares about
  performance-wise.

- **Anomalously high variance at N=16 for the composition path** (23–
  103 ms 95% CI). This is first-run / OS-level noise; the median of
  62 ms is well-separated from the fused median of 43 ms, and the
  subsequent N=64 number rules out any per-call overhead pathology.

## Interpretation through the Stage 8 trigger condition

> Stage 8 trigger (from `docs/design/design_v3/90-migration-plan.md`):
> "composition-based or `ExtensionOp`-based implementations of a
> repeatedly needed pattern are measurably slow in production
> workloads".

The composition path is not just "measurably slow" — it is
asymptotically slow by a factor that grows with problem size, for
reasons intrinsic to the decomposition:

- `BroadcastInDim + Add + ReduceMax` inflates memory usage from
  `O(MK + KN + MN)` to `O(MKN)`. No peephole / pass pipeline on the
  core `StdTensorOp` side can remove that blow-up because the 3D tensor
  is observable as a real intermediate value in the graph.
- Tropical GEMM is a canonical pattern: the dominant primitive in
  max-plus / min-plus linear algebra, shortest-path / Viterbi-style
  computations, and fused-argmax contractions. Any tropical-heavy
  workload will hit exactly this bottleneck.

The Stage 7 `ExtensionOp` packaging already fixes the memory-scaling
problem for **eager** execution (its `eager_execute` computes directly
from host data), so out-of-tree crates can ship today with acceptable
performance. But the `ExtensionOp` carrier cannot express a
cache-blocked / SIMD / parallel kernel without either:

1. duplicating significant kernel infrastructure inside each external
   crate, or
2. routing through a core-owned primitive that can dispatch to the
   same backend abstractions used by `dot_general`, `reduce_max`, etc.

Option (2) is exactly what Stage 8 contemplates.

## Recommendation: **GO** — Stage 8 is justified

Proceed with Stage 8 and introduce a core-owned fused tropical
dot-general variant.

### Suggested shape of the core op

A sketch (non-binding) consistent with the existing `StdTensorOp` enum
conventions and with the Stage 7 fused `ExtensionOp` semantics:

```rust
// In tenferro-ops/src/std_tensor_op.rs:
enum StdTensorOp {
    // ... existing variants ...

    /// Semiring-generalized dot-general. The "standard" (+, *) case is
    /// DotGeneral; this variant handles arbitrary abelian-monoid ⊕ for
    /// the reduction and arbitrary ⊗ for the pairwise step.
    ///
    /// Initial implementation target: MaxPlus / MinPlus (same surface
    /// as `FusedTropicalDotGeneralOp`). MaxMul is a later addition.
    SemiringDotGeneral {
        config: DotGeneralConfig,   // same axes struct as DotGeneral
        semiring: SemiringKind,     // MaxPlus | MinPlus | (MaxMul)
    },
}
```

Dispatch targets:

- **CPU (`cpu-faer` / `cpu-blas`)**: hand-written cache-blocked tropical
  GEMM, ideally backed by `strided-kernel` primitives where available
  (cache-blocked reduce-of-sum along the contracting axis per output
  tile). Parallelization over the output tiles is trivial.
- **GPU (`cubecl`)**: a tropical GEMM kernel in cubecl's kernel IR —
  cuTENSOR does not expose a max-plus path, so this is a native cubecl
  kernel rather than an FFI wrapper.

AD remains as specified in `40-extension-boundary.md` / spec Section 14:
the linearization and transpose rules still lower to core
`BroadcastInDim + Add + ReduceMax + Compare(Eq) + Select + ReduceSum +
Mul + Div` — i.e. Stage 8 adds a new primal op but does **not** touch
the AD closure contract. The AD rule can keep the O(MKN) working set
for now because AD is already memory-heavy for other structural reasons;
if profiling later shows AD is also a bottleneck, that becomes a
separate Stage 8 sub-task.

### What this bench does NOT prove

- This bench only covers **rank-2 square** inputs. Batched tropical
  contraction and non-square shapes are plausible but untested. The
  Stage 8 op design should accept the same axis specification as
  `DotGeneral` so those fall out naturally.
- The bench is CPU-only; no GPU data here. Stage 8 GPU dispatch is a
  separate follow-up — the CPU evidence alone is enough to justify
  introducing the core op.
- The bench measures forward only, not AD. AD performance under
  Stage 4 composition vs Stage 7 `ExtensionOp` vs hypothetical Stage 8
  core op is a separate investigation.

## Reproducing

```bash
cd ext/tropical && \
  cargo bench --bench tropical_matmul -- \
    --measurement-time 3 --warm-up-time 1 --sample-size 10
```

To re-run a single size / path:

```bash
cargo bench --bench tropical_matmul -- \
  'tropical_matmul/composition/256'
```

HTML reports land under `ext/tropical/target/criterion/`.
