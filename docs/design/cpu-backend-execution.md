# CPU Engine Execution Contract

`CpuEngine` is a caller-held execution value. It owns one `threadteam::Pool` (or none
for `CpuEngine::serial()`), the buffer pool that recycles outputs, the N-ary
contraction scratch, the plan cache and the accounting of memory that unfinished
work retains. The caller builds the pool, hands it over, and passes `&mut` to
every operation; tenferro never starts a hidden or global pool, keeps no
process-wide engine registry, and does not arbitrate between callers. The full
ownership model is in
[execution-engine-ownership.md](./execution-engine-ownership.md).

Pinning is the caller's job. When a caller pins the pool's workers, they share
the declared mask instead of owning one CPU each, because provider-created
threads inherit the creating worker's mask and a one-CPU mask would confine a
provider's whole thread team.

## Provider Selection Is Compile-Time

Exactly one CPU backend is compiled: `native` (the default) or `blas`.

- `native` uses faer-backed tlinalg plus cpueinsum/tcontract.
- `blas` adds the `cpueinsum-blas` and `tlinalg-blas` adapters and keeps the
  native path for steps those adapters decline.

Enabling both, or neither, is a compile-time error. There is no runtime
provider kind, provider bundle, kernel slot, or per-handle provider selection.
`tenferro_cpu::cpu_provider_id()` names the compiled provider for diagnostics
only; it is not a dispatch key.

## Engine Ownership And Entry

- `CpuEngine::new(pool)` takes the pool by value, so one pool can never back two
  engines. `CpuEngine::serial()` has no pool and runs everything on the calling
  thread, including inside another pool.
- `&mut engine` is the exclusivity mechanism. There is no session, admission
  permit, execution scope, held session or resource arbiter; concurrent
  computations use separate engines and separate pools.
- Every operation starts on the calling thread with one placement token minted
  from the engine's pool (`threadteam::Exec`, budget = the pool's width). The
  token enters the pool only when an entry splits the work into more than one
  task.
- The token is correct from a plain thread, from a worker of the engine's own
  pool (the entry runs in place), and from a worker of a different pool. A
  provider strategy that needs every worker at once (a barrier team) asks the
  token with `Exec::can_broadcast` and takes its barrier-free path when it is
  refused, so there is no engine-level refusal for a nested call.
- tenferro does not narrow and restore CPU masks per operation. A serial engine
  never starts or enters a pool and is safe inside any pool.

## The Lower-Library Seam

The `tcontract_exec::Exec` bundle is the one engine-scoped interop seam: a
`threadteam::Exec` placement token over the engine's pool (or the serial
token), the engine's workspace and the resolved batch policy. cpueinsum,
tcontract, and tlinalg take it by `&mut` (`*_workspace` queries by `&`). The
lower library owns lane selection, vendor batching, scratch, and scheduling
within the token's budget, splitting it with `fork` and `for_ranges`; tenferro
owns tensor semantics, placement, output allocation, validation, and cache
lifetime.

Strided-rs kernels, FFT lanes and every other tenferro fan-out take the same
token by value, last, and split it with `fork`, `for_ranges` or `external`;
`threadteam` is the only crate that starts tasks. tenferro never installs the
caller's continuation.

A caller that wants concurrent numerical work holds several engines over several
pools, and each engine is used from one calling thread at a time. tenferro hands
out its provider bundle for one engine but never installs the caller's
continuation, and vendor threading remains outside tenferro's budget.

Vendor BLAS/LAPACK is called from the coordinator thread. tenferro guarantees
nothing about the vendor's own worker team, thread count, or placement;
`num_threads() == 1` on a tenferro engine does not imply a one-thread vendor
call.

The token is not uniform across the two linalg providers, by design: the
faer-backed `tlinalg` owns batch lanes and takes a `tcontract_exec::Exec` (token
plus workspace plus the resolved batch policy), while `tlinalg-blas` is
deliberately token-free because LAPACK/BLAS own their own threading, its batch
loop is serial, and a Rayon fan-out around a vendor call would fight the
vendor's own pool. The host places the vendor call instead; its scratch still
comes from tenferro's buffer pool through the workspace seam. Because the BLAS
linalg lane has no `Exec` and no tenferro lanes, tenferro refuses a forced batch
strategy there before any write with a typed unsupported error; see the
[CPU-execution guide's Batched Operation Strategy](../guides/cpu-execution.md#batched-operation-strategy).

## Data Path

CPU floating/complex `dot_general` and grouped GEMM go to cpueinsum prepared
binary/grouped plans; CPU N-ary concrete einsum delegates to cpueinsum with the
order tenferro's planner selected; CPU linalg families go to tlinalg /
tlinalg-blas. The integer/all-batch elementwise contraction stays in
tenferro-cpu. tenferro keeps axis, layout, dtype, placement, and output-form
validation, the `MaybeUninit` complete-coverage proof for fresh binary outputs,
and the typed unsupported mapping.

Prepared plans are immutable and cacheable: they hold no live operands, no
execution lease, and no mutable scratch. Execution scratch lives in an
engine-owned workspace with its own retention bound and accounting; retained
plan and buffer caches are bounded and clearable through the engine.

An engine may be bound to a shared allocation domain. Fresh outputs belong to
that domain and carry its id; the id is routing/locality metadata, not
allocation ownership or evidence of NUMA page placement, worker pinning, or
memory residency. Storage-sharing views and metadata-only reshapes preserve the
source metadata; caller-owned `_into` outputs preserve the caller's metadata.

## A Fused Elementwise Region Is Planned And Selected

The runtime plans a fused elementwise region for a prepared program and asks the
engine to execute it as one command. The engine replays it through the CPU fused
executor; the executor still declines (returns `None`, so the runtime dispatches
the region's instructions one by one) below its 16,384-element floor, above its
four-input limit, and for dtypes or ops without a fused kernel.

`strided-fused 0.5.2` made the selection viable. Its chunked register machine
runs the `elem-simd-math` slice kernels over chunk-sized buffers and now treats a
stride-`0` operand as a broadcast value, so a broadcast on the contiguity axis —
the case that fell back to `interpret_inner_loop_uninit` and lost about 16x on
0.5.1 — stays on the vectorized replay. `tenferro-cpu-fused` also compiles one
`strided-fused` plan carrying every live output and executes it with
`ErasedFusedPlan::execute_outputs_uninit`, so a region with several live outputs
replays its instruction list once instead of once per output.

The selection is scoped to what was measured. Only `F32`/`F64` regions are
selected, because 0.6.0's vectorized chunk replay covers only those dtypes;
C32/C64, integers and bool stay unfused rather than run the unmeasured
per-element interpreter. Ordered real ops (`Maximum`/`Minimum`/`Clamp`) are
selected: `strided-fused 0.6.0` propagates NaN through the same
`strided_basic` select the ordinary kernels use (strided-rs#315), so the fused
route matches the per-instruction route bit for bit. `Remainder` and `Erf`
remain declined, because they have no `strided_fused` instruction.

The measurement (2026-10-17, `AMD EPYC 7713P`, 1T and 4T) is recorded in
[`docs/worklogs/perf-cpu-fusion-declined-2140.md`](../worklogs/perf-cpu-fusion-declined-2140.md),
section "Re-enabled on `strided-fused 0.5.2` (#2161)". The decision rule actually
applied was "median over all recorded runs per case `<= 1.05`", and it passes
with the largest per-case median 1.02; the contiguity-axis broadcast and the
two-live-output region, the two cases that lost on 0.5.1, are among the wins.
The GPU backend reaches the same runtime machinery with its own `#[cube]` code
generator and is unaffected.

The runtime capability reports the executor's four-input limit; the engine
selects a region only inside it.

## Diagnostics

`CpuEngine::num_threads()`, `buffer_pool_stats()`,
`buffer_pool_limit_bytes()`, `plan_cache()` and `allocation_domain()` are the
observation accessors. They report the engine's own state; they do not grant
execution or storage authority.
