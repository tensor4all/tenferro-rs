# CPU Execution and Thread Pools

tenferro never builds threads. The caller builds a `Pool` (or decides to use
none), hands it to `CpuEngine::new`, and keeps the engine for as long as the
computation needs it. Thread count, affinity and nesting are properties of the
pool you build.

One mechanism decides where CPU work runs and how many tasks run at once: the
placement token `threadteam::Exec`, a reference to the engine's pool plus a
thread budget. The engine mints one token for each operation; every kernel that
can fan out takes it as its last parameter and splits it (`fork`,
`for_ranges`, `external`). A token cannot be copied and every parallel entry
consumes it, so the number of tasks running at once under one operation is at
most its budget, at every nesting depth. `threadteam` is the only crate of the
stack that starts tasks; `tenferro-cpu` re-exports `Pool` and `PoolBuilder`, and the
`threadteam` crate itself, so the token is named
`tenferro_cpu::threadteam::Exec` and you need no separate dependency. (It is
not re-exported at the root: `CpuEngine::exec` returns the different
`tcontract_exec::Exec` bundle.)

For the dispatch cost of an operation once the engine exists, see
[Dispatch and Engine Cost](dispatch-and-engine-cost.md).

## The Caller Owns The Pool

`CpuEngine` has exactly two construction modes:

- `CpuEngine::new(pool)` takes over a `tenferro_cpu::Pool` you built. Every
  operation may use all of its workers.
- `CpuEngine::serial()` has no pool at all: every operation runs on the calling
  thread, and no pool is started or entered.

Build the pool with `Pool::builder()`. You set the width, optionally name or pin
the workers (`thread_name`, `on_start`), and build. The builder reserves a 16 MiB
worker stack by default, which the CPU providers need:

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#cpu_execution_28 -->
```rust
use tenferro_cpu::{Pool, CpuEngine};

// The caller owns the pool: tenferro builds no threads of its own.
let pool = Pool::builder().threads(4).build()?;
let engine = CpuEngine::new(pool);
println!("workers={}", engine.num_threads());
```
<!-- end-snippet-source -->

The worker stack size matters: provider code recurses with large private
frames (a `NUM_THREADS=64` OpenBLAS build keeps a `job_t job[64]` on the stack
of every recursive `dgetrf_parallel` frame), and the 2 MiB `std::thread`
default aborts with a stack overflow at moderate matrix sizes. That is why
`PoolBuilder` defaults to 16 MiB; lower it with `stack_size` only for pools that
never run provider code.

## Affinity, NUMA, And Nesting

Because the pool is yours, affinity and NUMA locality are your decision. The
portable ways to express them are:

- build the pool on a thread whose CPU affinity mask is already the set you
  want (`taskset`, `sched_setaffinity`, a cgroup cpuset, a scheduler), so every
  worker it creates inherits that mask;
- pin the whole process, or run one process per locality domain;
- choose a smaller pool instead of oversubscribing an outer one.

Pinning workers restricts where computation may run. It does **not** make tensor
allocation NUMA-local, choose a first-touch policy, migrate existing pages, or
configure page interleaving. Applications that require memory locality must
arrange allocation/first-touch or OS memory policy separately and measure the
result on their deployment topology.

A serial engine is the safe way to run inside a pool you already own, because
it never enters a pool:

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#cpu_execution_29 -->
```rust
use tenferro_cpu::CpuEngine;

// A serial engine has no pool at all: every operation runs on the calling
// thread, so it is safe to use from inside a pool the caller already owns.
let engine = CpuEngine::serial();
assert_eq!(engine.num_threads(), 1);
```
<!-- end-snippet-source -->

## Direct faer calls on the engine's pool

Downstream code that needs a faer routine not exposed by tenferro calls faer on
the host slices it borrows from the tensor. `CpuEngine::exec` lends the engine's
provider bundle; its token hands the budget to faer as a plain width with
`Exec::external`, which enters the engine's pool once (inline for a serial
engine or a width of one):

```text
engine.exec()?.into_token().external(|width| {
    let par = if width.get() == 1 { Par::Seq } else { Par::rayon(width.get()) };
    faer_operation(..., par)
});
```

This keeps direct faer work on the same threads as tenferro's own operations,
and a small sequential call never enters a pool. tenferro hands out no
`faer::Par`; spell the types with `tenferro_cpu::faer` (the build tenferro
links) to avoid compiling a second faer into one binary. How faer divides the
width it receives is faer's concern; this guarantee does not extend to workers
created internally by OpenBLAS, MKL, Accelerate, or OpenMP.

## Vendor BLAS/LAPACK

A `blas` build adds cpueinsum-blas and tlinalg-blas as compiled adapters; it is
not a vendor-only execution world, and the native path stays available inside
it. tenferro calls the vendor from the coordinator thread and makes no placement
or thread-count promise for the threads the vendor creates. The calling
thread's mask is the one you control: a caller already pinned to one CPU also
confines a newly created vendor team.

Provider variables such as `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and
`OMP_NUM_THREADS` still control vendor counts where supported. They are outside
tenferro's resource contract, and a one-worker `CpuEngine` does not imply a
one-thread vendor call.

## Batched Operation Strategy

Lane selection, vendor batching, and the per-item route for batched work
(strided-batched GEMM, packed LU factor/solve, and the batched linalg families)
belong to the lower numerical libraries: cpueinsum, tlinalg, and tlinalg-blas.
tenferro passes one `tcontract_exec::Exec` bundle (the placement token, the
engine's workspace and a batch policy), and the lower library decides its own
lanes within the token's budget by forking it.

The policy vocabulary is re-exported from `tcontract-exec`: `CpuBatchPolicy` is
`tcontract_exec::BatchPolicy`, `CpuBatchStrategy` is `BatchStrategy`, and
`CpuBatchThresholds` is `WorkModel`. The old `OuterParallel`/`ProviderItems`
spellings are now `Outer`/`Inner`; `Sequential` and `Auto` are unchanged (the old
vendor-dimension cutoff is not restored).

Precedence is the innermost scope, then the engine default:

1. `tenferro_cpu::with_batch_policy(engine, policy, |engine| ...)` scopes an
   override to the callback and restores the previous policy on return, error
   and unwind. Wrapping a single operation expresses a per-operation choice; a
   non-CPU engine is rejected with a typed unsupported error.
2. `CpuEngine::with_batch_policy(policy)` sets the engine default when the
   engine is built. The default is `Auto`.

`Auto` is resolved inside the lower library and keeps that library's route rule
until the shared work model lands (tcontract-rs#123 step 3): `tlinalg`'s
64-extent RHS cutoff in the linalg lane and `cpueinsum`'s width rule in the
contraction lane. A forced strategy is honored or refused, never silently
downgraded to `Auto`. `tlinalg-blas` takes no `Exec` and has no tenferro lanes,
so the BLAS linalg lane refuses a forced `Outer`, `Inner`, or `Split` before any
write with a typed unsupported error; `Auto` and `Sequential` are accepted.

A contraction whose axes are all batch axes is an elementwise product and never
lowers to per-element GEMMs.

## Where Parallel Work Runs

An operation starts on the calling thread. The token enters the engine's pool
only when an entry splits the work into more than one task, so small
operations never pay a pool entry at all, and a serial engine never touches a
pool. A fork divides the budget among its children, so nested work inside a
partition stays within that partition's share.

For a BLAS backend the vendor owns its worker fan-out; the engine's pool is not
used as BLAS workers. An elementwise operation adjacent to a BLAS call still
runs on the engine's pool, never on an unconstrained global one.

Supported Host instructions, native instructions, and session-capable GEMM FFI
instructions share one engine. Extension runtimes that cannot execute through
`Engine` remain explicit boundaries.

## Diagnostics

Use `CpuEngine::num_threads()`,
`CpuEngine::batch_policy()`, `CpuEngine::buffer_pool_stats()`, and
`CpuEngine::plan_cache()` for logs:

<!-- snippet-source: docs/tutorial-code/src/bin/core_tensor_snippets.rs#cpu_execution_30 -->
```rust
let engine = tenferro_cpu::CpuEngine::new(
    tenferro_cpu::Pool::builder().threads(2).build()?,
);
println!(
    "workers={} batch_policy={:?}",
    engine.num_threads(),
    engine.batch_policy()
);
```
<!-- end-snippet-source -->

`tenferro_cpu::runtime_engine_id()` names the engine kind the runtime
registration binds to, and `runtime_engine_registration_with_id` lets one
process register two CPU engines under distinct IDs. The engine carries no
placement or domain identity: two engines differ by the pool they own and by
the ID the caller registered them under.
