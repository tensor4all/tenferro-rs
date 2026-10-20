# Parallelism and Caching

tenferro keeps CPU parallelism and execution caches explicit. Use this page to
control CPU thread counts, avoid provider oversubscription, and release cached
memory in long-running processes.

The caller owns the pool, so thread count and affinity are pool properties. For
how to build the pool, affinity/NUMA guidance, external BLAS safety, and the
location of elementwise Rayon work, see
[CPU Execution and Thread Pools](cpu-execution.md).

For tensor memory layout and column-major buffers, see
[Memory Order](memory-order.md). That is part of the tensor data model, not the
parallelism contract.

For measured per-call dispatch and Rayon worker handoff costs, batching and
one-thread alternatives, see [Dispatch and Engine Cost](dispatch-and-engine-cost.md).

## CPU Backend Provider

Provider choice, feature combinations, thread ownership, and the external
TBLIS example are maintained in [Choosing A Backend](choosing-a-backend.md).
Use that guide for the decision and capability matrix; this page keeps the
runtime mechanics below for users who have already selected a provider.

## CPU Thread Count

Build one pool with the width you want when one engine should carry a fixed CPU
parallelism policy:

<!-- snippet-source: docs/tutorial-code/src/bin/execution_snippets.rs#parallelism_and_caching_1 -->
```rust
use tenferro_cpu::{thread_pool_builder, CpuEngine};

let pool = thread_pool_builder().num_threads(4).build()?;
let engine = CpuEngine::new(pool);
assert_eq!(engine.num_threads(), 4);
```
<!-- end-snippet-source -->

A Rayon pool built without an explicit `num_threads` honors `RAYON_NUM_THREADS`
and falls back to the process-visible CPU count, so
`thread_pool_builder().build()` follows the same convention:

```bash
RAYON_NUM_THREADS=4 cargo run --release
```

For `native`, tenferro passes the engine's worker count to faer-backed kernels.
A serial engine uses sequential faer execution; a pooled engine uses faer's
Rayon parallelism with the pool's width. The external TBLIS provider example
clamps TBLIS calls to one thread while it owns the call, then restores the
previous TBLIS setting. The engine's worker count still applies to
tenferro-owned CPU kernels and fallback paths.

## CPU Operation Parallelism

The `CpuEngine` owns the workspace and the buffer pool; the Rayon pool it
computes on is the one the caller passed to `CpuEngine::new`. An operation
starts on the calling thread and enters that pool only after a provider has
chosen more than one worker for it. An ambient Rayon pool is never an implicit
fallback, and `CpuEngine::serial` never enters any pool. External BLAS/LAPACK
workers remain provider-owned and may fan out independently.

| Operation family | Threading behavior |
| --- | --- |
| Elementwise and analytic ops | Shared typed paths use `strided-basic`, ordinary dtype dispatch uses `strided-kernel`, and erased fused replay uses `strided-fused`; all use the engine's pool. Rayon-capable `Inner` work may use the pool; other modes stay sequential. |
| Reductions | sum/product erased replay and typed max/min reductions use the same engine policy and never ambient Rayon. Upstream strided plan coverage determines whether a specific multi-axis reduction shape can actually partition work. |
| View materialization, transpose/permute, broadcast, convert, and diagonal extraction | Strided copy/map kernels use the same engine policy; layout fallback and linalg input materialization are included. |
| `dot_general` through `native` | faer receives `Par::rayon(n)` only for `Inner` execution on a pooled engine; otherwise it receives `Par::Seq`. Downstream direct faer calls can obtain the same scoped value through `FaerParallelismExt::with_faer_parallelism`, without exposing the pool. |
| GEMM and linalg through `blas` | Threading is owned by the linked BLAS/LAPACK provider, not Rayon. Configure the provider variables below. |
| Supported `dot_general` contractions through an external TBLIS provider | The example provider clamps TBLIS to one thread per call; unsupported TBLIS shapes fall back to the compiled faer/BLAS provider in preferred mode. |
| Indexed gather/scatter and dynamic slice/update | These delegate to strided erased plans with an explicit `ExecContext` derived from the engine; upstream strided plan coverage determines whether a specific indexed plan can actually partition work. |
| Slicing, padding, concatenation, reverse, triangular masks, and `embed_diagonal` | These are dedicated sequential CPU loops today because their per-output indexing patterns do not yet have a strided-kernel/backend-native parallel primitive. They still run on the engine, and source comments mark the intentional sequential path. |

CPU affine-strided copy, permutation, broadcast, map, zip-map, and axis
reduction delegate to `strided-rs`, while tenferro supplies operation semantics,
validation, dtype and placement checks, error translation, and execution
resources. Einsum/dot-general is the benchmark-backed tenferro exception:
tenferro owns its planning, optimized preparation, and provider integration.

Even a host-to-host materialization goes through the engine. The engine owns a
persistent buffer pool, can allocate an uninitialized output when the copy fully
overwrites it, and runs the kernel on the caller's pool. A pool-free copy, a
temporary buffer pool, or Rayon's ambient global pool would create a second
memory and threading policy. Memory reuse and thread policy are execution
resources, not tensor metadata.

Use the engine's canonicalization operation when a metadata-only view must
become compact:

<!-- snippet-source: docs/tutorial-code/src/bin/execution_snippets.rs#parallelism_and_caching_2 -->
```rust
use tenferro_cpu::{thread_pool_builder, CpuEngine};
use tenferro_tensor::{TensorStructural, TypedTensor};

let tensor = TypedTensor::<f64>::from_vec_col_major(
    vec![2, 3],
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let transposed = tensor.as_view().transpose_view([1, 0])?.into_tensor_read()?;
let mut engine = CpuEngine::new(thread_pool_builder().num_threads(4).build()?);
let compact = engine.to_contiguous_read(transposed)?;

assert_eq!(compact.shape(), &[3, 2]);
assert_eq!(compact.as_slice::<f64>()?, &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
```
<!-- end-snippet-source -->

Canonicalization preserves placement; it does not silently upload host data or
download device data.

## BLAS And LAPACK Threads

For `blas`, the engine's pool width controls tenferro-native work, but
the linked BLAS/LAPACK provider has its own thread controls. Set provider thread
variables before process start when appropriate:

```bash
RAYON_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
VECLIB_MAXIMUM_THREADS=4 \
./your-tenferro-app
```

Use the variables that match your actual provider. For example, OpenBLAS mainly
uses `OPENBLAS_NUM_THREADS`; Intel MKL uses `MKL_NUM_THREADS`; Accelerate uses
`VECLIB_MAXIMUM_THREADS`; OpenMP-backed providers usually also obey
`OMP_NUM_THREADS`. For provider discovery at build time, non-standard OpenBLAS
installs commonly need `OPENBLAS_LIB_DIR`; non-standard MKL installs commonly
need `MKLROOT` or `MKL_LIB_DIR`.

These variables limit vendor thread counts; they do not let tenferro verify or
enforce vendor worker affinity. tenferro guarantees only the calling thread's
mask, so a caller already pinned to one CPU also confines a newly created vendor
team. Affinity is a property of the thread and pool you build, in both compiled
configurations.

There is one CPU resource contract for both compiled providers: tenferro passes
`Sequential` or the selected pool with a thread budget to the lower numerical
library (cpueinsum, tcontract, tlinalg, tlinalg-blas), and the lower library
decides its own lanes within that budget. Vendor-internal parallelism is
outside tenferro's resource contract.

An adaptation that needs to clamp a vendor's mode to one thread
for every finite domain budget. It must never select provider-controlled auto
mode inside such a call; inability to guarantee that requires the conservative
`GlobalOrUncontrolled` declaration.

## Avoid Oversubscription

Do not accidentally multiply outer application parallelism by inner kernel
parallelism. If an outer loop already runs many independent tenferro calls in
parallel, use a smaller inner backend:

<!-- snippet-source: docs/tutorial-code/src/bin/execution_snippets.rs#parallelism_and_caching_3 -->
```rust
use tenferro_cpu::{thread_pool_builder, CpuEngine};

let engine = CpuEngine::new(thread_pool_builder().num_threads(1).build()?);
assert_eq!(engine.num_threads(), 1);
```
<!-- end-snippet-source -->

For BLAS/LAPACK providers, apply the same rule to provider thread variables.
For benchmarks, pin all relevant thread counts and report them with the result.

When the host already partitions coarse work across caller-owned Rayon pools,
give each partition its own `CpuEngine` over its own pool. One computation uses
one engine at a time (every operation takes `&mut`), so a partition must admit
at most one coarse tenferro operation per pool and account for any physical CPU
oversubscription itself. See
[CPU Execution and Thread Pools](cpu-execution.md).

## Reuse Runtime State

Reuse execution objects when you repeat related work:

- `tenferro_runtime::Runtime` retains immutable engine/extension registration
  snapshots, prepared-plan cache entries, and registered runtime cache owners
  for the `Runtime::prepare` / `Runtime::run` pipeline.
- `EagerRuntime` retains eager extension plans and compiled inner extension
  programs across immediate operations. It stores no engine: every eager call
  takes the one the caller holds and registers it with the runtime on first
  use.
- `GraphCompiler` retains graph lowering and static extension planning caches.
  Its compiled artifact keeps semantic program plus compiler options, not
  backend staging.
- `Runtime` owns prepared-plan caching and registered runtime cache owners.
  CPU buffer pools and extension caches remain owned by the caller-held engine
  that the registration describes.

<!-- snippet-source: docs/tutorial-code/src/bin/execution_snippets.rs#parallelism_and_caching_4 -->
```rust
use tenferro_cpu::{thread_pool_builder, CpuEngine};
use tenferro_runtime::{GraphCompiler, Runtime};

let mut compiler = GraphCompiler::new();
let engine = CpuEngine::new(thread_pool_builder().num_threads(4).build()?);
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
let runtime = builder.build()?;
```
<!-- end-snippet-source -->

Short-lived scripts can usually ignore cache tuning. Services, notebooks, and
benchmark harnesses should treat caches as part of runtime resource management.

## Cache Limits

Runtime, compiler, eager, and CPU backend caches are bounded by default and can be
configured independently.

<!-- snippet-source: docs/tutorial-code/src/bin/execution_snippets.rs#parallelism_and_caching_5 -->
```rust
use std::num::NonZeroUsize;
use tenferro_cpu::CpuEngine;
use tenferro_runtime::extension::ExtensionCacheLimits;
use tenferro_runtime::ExecutionEngine;

let mut engine = CpuEngine::serial();
let mut caches = engine.take_extension_caches();
caches.set_limits(
    ExtensionCacheLimits::new(
        NonZeroUsize::new(128).ok_or("positive cache-entry limit")?,
    )
    .with_max_retained_bytes(
        NonZeroUsize::new(64 * 1024 * 1024).ok_or("positive cache-byte limit")?,
    ),
);
engine.restore_extension_caches(caches);
```
<!-- end-snippet-source -->

For CPU engines, the buffer pool has its own retention limit:

<!-- snippet-source: docs/tutorial-code/src/bin/execution_snippets.rs#parallelism_and_caching_6 -->
```rust
use tenferro_cpu::CpuEngine;

let mut engine = CpuEngine::serial();
engine.set_buffer_pool_limit_bytes(32 * 1024 * 1024);
```
<!-- end-snippet-source -->

## Clearing Cached Memory

Clear caches when a long-running process changes workload phase, when a
notebook has finished a large experiment, or when memory pressure matters more
than reusing old plans and buffers.

<!-- snippet-source: docs/tutorial-code/src/bin/execution_snippets.rs#parallelism_and_caching_7 -->
```rust
use tenferro_ad::EagerRuntime;
use tenferro_runtime::{GraphCompiler, Runtime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
let eager = EagerRuntime::new()?;
let runtime = Runtime::builder().build()?;
let mut compiler = GraphCompiler::new();

runtime.clear_prepared_cache()?;
runtime.clear_caches()?;
compiler.clear_caches();
eager.clear_caches()?;
Ok(())
}
main()?;
```
<!-- end-snippet-source -->

For CPU engines, clear the buffer pool through the engine:

<!-- snippet-source: docs/tutorial-code/src/bin/execution_snippets.rs#parallelism_and_caching_8 -->
```rust
use tenferro_cpu::CpuEngine;

let mut engine = CpuEngine::serial();
engine.reset_buffer_pool();
```
<!-- end-snippet-source -->

`retained_bytes` in cache stats is tenferro's logical retained-data
estimate. It is not operating-system RSS and does not include allocator arena
slack, thread stacks, or provider-owned memory.

`CacheStats` also reports cache events when the owner can observe them:
`hits`, `misses`, `evictions`, and explicit `clears`. Extension cache
selectors scope event counters the same way they scope `entries` and
`retained_bytes`, so `ExtensionCacheSelector::Family` and
`ExtensionCacheSelector::Cache` report only the selected family or cache.

## CUDA Transfer Boundaries

CUDA transfers are explicit. Upload once near the boundary of a CUDA workload,
run supported operations on CUDA tensors, then download only when host
inspection or CPU execution is needed. Repeated upload/download inside tight
loops usually dominates runtime. See [Devices and GPU](devices-and-gpu.md) for
the current CUDA coverage and setup commands.
