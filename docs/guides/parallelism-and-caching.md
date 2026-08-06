# Parallelism and Caching

tenferro keeps CPU parallelism and execution caches explicit. Use this page to
control CPU thread counts, avoid provider oversubscription, and release cached
memory in long-running processes.

Thread count is not CPU affinity. For NUMA-node definitions, pinned engines,
external BLAS safety, and the location of elementwise Rayon work, see
[CPU Execution and NUMA Placement](cpu-execution.md).

For tensor memory layout and column-major buffers, see
[Memory Order](memory-order.md). That is part of the tensor data model, not the
parallelism contract.

## CPU Backend Provider

Provider choice, feature combinations, thread ownership, and the external
TBLIS example are maintained in [Choosing A Backend](choosing-a-backend.md).
Use that guide for the decision and capability matrix; this page keeps the
runtime mechanics below for users who have already selected a provider.

## CPU Thread Count

Use `CpuBackend::with_threads(n)` when one backend should carry a fixed CPU
parallelism policy:

<!-- snippet-source: docs/tutorial-code/src/bin/execution_snippets.rs#parallelism_and_caching_1 -->
```rust
use tenferro_cpu::CpuBackend;

let backend = CpuBackend::with_threads(4)?;
assert_eq!(backend.num_threads(), 4);
```
<!-- end-snippet-source -->

`CpuBackend::new()` reads `RAYON_NUM_THREADS` and falls back to the
process-visible CPU count when the variable is unset.

```bash
RAYON_NUM_THREADS=4 cargo run --release
```

For `cpu-faer`, tenferro passes the `CpuContext` thread count to faer-backed
kernels. A one-thread context uses sequential faer execution; a multi-thread
context uses faer's Rayon parallelism with the requested thread count.
The external TBLIS provider example clamps TBLIS calls to one thread while it
owns the call, then restores the previous TBLIS setting. `CpuBackend` thread
counts still apply to tenferro-owned CPU kernels and fallback paths.

## CPU Operation Parallelism

`CpuContext` owns the Rayon pool used by tenferro-owned CPU tensor kernels.
Standalone backend calls and compiled `BackendSession` operations cross the
selected domain executor exactly once before dispatch. Provider-facing
execution contexts are already entered and cannot install or submit nested
executor work.

Inside that entry, tenferro-native strided kernels use the policy selected by
`CpuExecutionContext`: `Inner` work whose selected executor advertises Rayon
uses only that executor up to its validated budget, while `Sequential`,
engine-outer children, and external-worker `Inner` contexts stay sequential.
An ambient Rayon pool is never an implicit fallback. External BLAS/LAPACK
workers remain provider-owned and may fan out independently.

| Operation family | Threading behavior |
| --- | --- |
| Elementwise and analytic ops | `strided-kernel` map/zip kernels and erased fused replay use the already-entered context's native policy. Rayon-capable `Inner` may use the selected executor; all other modes above are sequential. |
| Reductions | sum/product erased replay and typed max/min reductions use the same selected native policy and never ambient Rayon. Upstream strided plan coverage determines whether a specific multi-axis reduction shape can actually partition work. |
| View materialization, transpose/permute, broadcast, convert, and diagonal extraction | `strided-kernel` copy/map kernels use the same selected native policy; layout fallback and linalg input materialization are included. |
| `dot_general` through `cpu-faer` | faer receives `Par::rayon(n)` only for `Inner` execution whose selected executor advertises Rayon and whose validated budget is greater than one; otherwise it receives `Par::Seq`. |
| GEMM and linalg through `cpu-blas` | Threading is owned by the linked BLAS/LAPACK provider, not Rayon. Configure the provider variables below. |
| Supported `dot_general` contractions through an external TBLIS provider | The example provider clamps TBLIS to one thread per call; unsupported TBLIS shapes fall back to the compiled faer/BLAS provider in preferred mode. |
| Indexed gather/scatter and dynamic slice/update | These delegate to strided erased plans with an explicit `ExecContext` derived from `CpuExecutionContext`; upstream strided plan coverage determines whether a specific indexed plan can actually partition work. |
| Slicing, padding, concatenation, reverse, triangular masks, and `embed_diagonal` | These are dedicated sequential CPU loops today because their per-output indexing patterns do not yet have a strided-kernel/backend-native parallel primitive. They still run inside the selected executor entry, and source comments mark the intentional sequential path. |

CPU affine-strided copy, permutation, broadcast, map, zip-map, and axis
reduction delegate to `strided-rs`, while tenferro supplies operation semantics,
validation, dtype and placement checks, error translation, and execution
resources. Einsum/dot-general is the benchmark-backed tenferro exception:
tenferro owns its planning, optimized preparation, and provider integration.

Even a host-to-host materialization must enter through `CpuBackend`. The
backend owns a persistent buffer pool, can allocate an uninitialized output
when the copy fully overwrites it, and runs the kernel in the configured
`CpuContext` Rayon pool. That scope also preserves nested-execution safety and
the kernel's serial/parallel threshold. A context-free copy, a temporary buffer
pool, or Rayon's ambient global pool would create a second memory and threading
policy. Memory reuse and thread policy are execution resources, not tensor
metadata.

Use the backend-owned canonicalization operation when a metadata-only view must
become compact:

<!-- snippet-source: docs/tutorial-code/src/bin/execution_snippets.rs#parallelism_and_caching_2 -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{TypedTensor};
use tenferro_tensor::backend::TensorViewCanonicalization;

let tensor = TypedTensor::<f64>::from_vec_col_major(
    vec![2, 3],
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let transposed = tensor.as_view().transpose_view([1, 0])?;
let mut backend = CpuBackend::with_threads(4)?;
let compact = backend.to_contiguous(&transposed)?;

assert_eq!(compact.shape(), &[3, 2]);
assert_eq!(compact.as_slice()?, &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
```
<!-- end-snippet-source -->

Canonicalization preserves placement; it does not silently upload host data or
download device data.

## BLAS And LAPACK Threads

For `cpu-blas`, `CpuBackend::with_threads(n)` controls tenferro-native work, but
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

These variables limit thread counts; they do not let tenferro verify or enforce
provider worker affinity. External BLAS therefore supports only
`CpuPlacement::Auto` and executes under an exclusive coordinator permit. Use
the faer backend when tenferro-managed NUMA placement is required.

Custom CPU provider bundles declare count and placement separately through
their provider traits. Bundle installation validates those declarations
against every registered domain, including lazily constructible managed NUMA
domains. `CpuBackend::from_external_managed_domains_with_provider_bundle`
performs external-domain registry construction and this validation atomically.
The provider bundle currently covers the `dot_general` family; linalg provider
selection remains separate. The current built-in BLAS adapter does not apply
and restore a genuinely local setter per call, so the ordinary
external-managed constructor rejects that strict standard BLAS bundle.
OpenBLAS's `openblas_set_num_threads_local` does not change this conclusion:
despite its name, it applies a process-global count and returns the old value
for restoration, so concurrent threads can observe the temporary setting.
Applications can use the custom-bundle constructor with an adapter that
declares and enforces suitable controls. Parallel OpenBLAS remains available
only through provider-owned, process-exclusive compatibility execution; it is
not a strict per-call thread-budget guarantee.

A provider declaring `BinaryClampToOne` must select its single-threaded mode
for every finite domain budget. It must never select provider-controlled auto
mode inside such a call; inability to guarantee that requires the conservative
`GlobalOrUncontrolled` declaration.

## Avoid Oversubscription

Do not accidentally multiply outer application parallelism by inner kernel
parallelism. If an outer loop already runs many independent tenferro calls in
parallel, use a smaller inner backend:

<!-- snippet-source: docs/tutorial-code/src/bin/execution_snippets.rs#parallelism_and_caching_3 -->
```rust
use tenferro_cpu::CpuBackend;

let backend = CpuBackend::with_threads(1)?;
```
<!-- end-snippet-source -->

For BLAS/LAPACK providers, apply the same rule to provider thread variables.
For benchmarks, pin all relevant thread counts and report them with the result.

## Reuse Runtime State

Reuse execution objects when you repeat related work:

- `tenferro_runtime::Runtime` retains immutable engine/extension registration
  snapshots, prepared-plan cache entries, and registered runtime cache owners
  for the `Runtime::prepare_for` and `Runtime::run_compiled*` pipelines.
- `EagerRuntime` retains eager extension plans and compiled inner extension
  programs across immediate operations. CPU-backed eager runtimes also keep a
  private `Runtime` snapshot so placement-bound CPU views can refresh runtime
  registration metadata by epoch comparison without holding idle resources.
  The CPU registration is built by `tenferro-cpu` and includes the runtime
  execution bridge.
- `GraphCompiler` retains graph lowering and static extension planning caches.
  Its compiled artifact keeps semantic program plus compiler options, not
  backend staging.
- `Runtime` owns prepared-plan caching and registered runtime cache owners.
  CPU backend buffer pools remain owned by the registered CPU backend.

<!-- snippet-source: docs/tutorial-code/src/bin/execution_snippets.rs#parallelism_and_caching_4 -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, Runtime};

let mut compiler = GraphCompiler::new();
let backend = CpuBackend::with_threads(4)?;
let mut builder = Runtime::builder();
builder.register_engine(tenferro_cpu::runtime_engine_registration(&backend)?)?;
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
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::extension::ExtensionCacheLimits;

let eager = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
eager.set_extension_cache_limits(ExtensionCacheLimits::new(
    NonZeroUsize::new(128).ok_or("positive cache-entry limit")?,
).with_max_retained_bytes(
    NonZeroUsize::new(64 * 1024 * 1024).ok_or("positive cache-byte limit")?,
))?;
```
<!-- end-snippet-source -->

For CPU backends, the CPU buffer pool has its own retention limit:

<!-- snippet-source: docs/tutorial-code/src/bin/execution_snippets.rs#parallelism_and_caching_6 -->
```rust
use tenferro_cpu::CpuBackend;

let mut backend = CpuBackend::new();
backend.set_buffer_pool_limit_bytes(32 * 1024 * 1024)?;
```
<!-- end-snippet-source -->

## Clearing Cached Memory

Clear caches when a long-running process changes workload phase, when a
notebook has finished a large experiment, or when memory pressure matters more
than reusing old plans and buffers.

<!-- snippet-source: docs/tutorial-code/src/bin/execution_snippets.rs#parallelism_and_caching_7 -->
```rust
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, Runtime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
let eager = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
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

For CPU backends, clear the buffer pool through the backend:

<!-- snippet-source: docs/tutorial-code/src/bin/execution_snippets.rs#parallelism_and_caching_8 -->
```rust
use tenferro_cpu::CpuBackend;

let mut backend = CpuBackend::new();
backend.reset_buffer_pool()?;
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
