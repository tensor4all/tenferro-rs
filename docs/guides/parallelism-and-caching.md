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

At least one CPU provider feature must be compiled. `cpu-faer` is the default.
`cpu-blas` can be compiled by itself or together with `cpu-faer`. `cpu-tblis`
is an optional `dot_general` contraction provider; it is additive and still
requires at least one of `cpu-faer` or `cpu-blas` for fallback and linalg
coverage.
`blas-openblas`, `blas-accelerate`, and `blas-mkl` are explicit BLAS/LAPACK
source-provider features that also enable `cpu-blas`; enable at most one of
them in a single resolved Cargo feature graph.

`CpuBackend::new()` chooses a provider from the features compiled into the
current binary:

| Compiled CPU provider features | `CpuBackend::new()` provider |
| --- | --- |
| `cpu-faer` only | faer |
| `cpu-blas` only | BLAS/LAPACK |
| `cpu-faer` and `cpu-blas` | BLAS/LAPACK |

This is the default provider for that backend instance. If multiple complete
CPU providers are compiled, select `CpuBackendKind::Faer` or
`CpuBackendKind::Blas` explicitly when a specific call path should use one of
them. TBLIS is not a complete backend kind; opt into TBLIS `dot_general`
attempts with `DotGeneralProvider::TblisIfAvailable` or require TBLIS with
`DotGeneralProvider::TblisRequired`. Explicit base-provider selection returns a
configuration error if the requested provider was not compiled into the binary:

```rust
use tenferro_cpu::CpuBackend;
use tenferro_cpu::CpuBackendKind;

let backend = CpuBackend::with_threads_and_kind(4, CpuBackendKind::Faer).unwrap();
assert_eq!(backend.num_threads(), 4);
assert_eq!(backend.kind(), CpuBackendKind::Faer);
```

## CPU Thread Count

Use `CpuBackend::with_threads(n)` when one backend should carry a fixed CPU
parallelism policy:

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::GraphExecutor;

let executor = GraphExecutor::new(CpuBackend::with_threads(4).unwrap());
assert_eq!(executor.backend().num_threads(), 4);
```

`CpuBackend::new()` reads `RAYON_NUM_THREADS` and falls back to the
process-visible CPU count when the variable is unset.

```bash
RAYON_NUM_THREADS=4 cargo run --release
```

For `cpu-faer`, tenferro passes the `CpuContext` thread count to faer-backed
kernels. A one-thread context uses sequential faer execution; a multi-thread
context uses faer's Rayon parallelism with the requested thread count.
For `cpu-tblis`, TBLIS owns its internal threading policy; `CpuBackend` thread
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
| Elementwise and analytic ops | `strided-kernel` map/zip kernels use the already-entered context's native policy. Rayon-capable `Inner` may use the selected executor; all other modes above are sequential. |
| Reductions | `strided-kernel::reduce_axis` uses the same selected native policy and never ambient Rayon. |
| View materialization, transpose/permute, broadcast, convert, and diagonal extraction | `strided-kernel` copy/map kernels use the same selected native policy; layout fallback and linalg input materialization are included. |
| `dot_general` through `cpu-faer` | faer receives `Par::rayon(n)` only for `Inner` execution whose selected executor advertises Rayon and whose validated budget is greater than one; otherwise it receives `Par::Seq`. |
| GEMM and linalg through `cpu-blas` | Threading is owned by the linked BLAS/LAPACK provider, not Rayon. Configure the provider variables below. |
| Supported `dot_general` contractions through `cpu-tblis` | TBLIS owns provider threading; unsupported TBLIS shapes fall back to the compiled faer/BLAS provider. |
| Indexing, scatter/gather, slicing, padding, concatenation, reverse, triangular masks, and `embed_diagonal` | These are dedicated sequential CPU loops today because their per-output indexing patterns do not yet have a strided-kernel/backend-native parallel primitive. They still run inside the selected executor entry, and source comments mark the intentional sequential path. |

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

```rust
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{TensorViewCanonicalization, TypedTensor};

let tensor = TypedTensor::<f64>::from_vec_col_major(
    vec![2, 3],
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
).unwrap();
let transposed = tensor.as_view().transpose_view([1, 0]).unwrap();
let mut backend = CpuBackend::with_threads(4).unwrap();
let compact = backend.to_contiguous(&transposed).unwrap();

assert_eq!(compact.shape(), &[3, 2]);
assert_eq!(compact.as_slice().unwrap(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
```

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

```rust
use tenferro_cpu::CpuBackend;

let backend = CpuBackend::with_threads(1);
```

For BLAS/LAPACK and TBLIS providers, apply the same rule to provider thread
variables. For benchmarks, pin all relevant thread counts and report them with
the result.

## Reuse Runtime State

Reuse execution objects when you repeat related work:

- `tenferro_runtime::Runtime` retains immutable engine/extension registration
  snapshots, prepared-plan cache entries, and registered runtime cache owners
  for the `Runtime::prepare_for` pipeline.
- `EagerRuntime` retains eager extension plans and compiled inner extension
  programs across immediate operations. CPU-backed eager runtimes also keep a
  private `Runtime` snapshot so placement-bound CPU views can refresh runtime
  registration metadata by epoch comparison without holding idle resources.
- `GraphCompiler` retains graph lowering and static extension planning caches.
- `GraphExecutor<B>` retains runtime extension plans, compiled inner extension
  programs, backend analysis, and reusable backend buffers.

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, GraphExecutor};

let mut compiler = GraphCompiler::new();
let mut executor = GraphExecutor::new(CpuBackend::with_threads(4));
```

Short-lived scripts can usually ignore cache tuning. Services, notebooks, and
benchmark harnesses should treat caches as part of runtime resource management.

## Cache Limits

Runtime, compiler, executor, and eager caches are bounded by default and can be
configured independently.

```rust
use std::num::NonZeroUsize;
use tenferro_ad::EagerRuntime;
use tenferro_runtime::extension::ExtensionCacheLimits;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, GraphExecutor, PreparedPlanCacheLimits, Runtime};

let eager = EagerRuntime::with_cpu_backend(CpuBackend::new());
eager.set_extension_cache_limits(ExtensionCacheLimits::new(
    NonZeroUsize::new(128).unwrap(),
)).unwrap();

let runtime = Runtime::builder().build().unwrap();
runtime.set_prepared_cache_limits(PreparedPlanCacheLimits::new(
    NonZeroUsize::new(128).unwrap(),
)).unwrap();

let mut compiler = GraphCompiler::new();
compiler.set_compile_cache_capacity(NonZeroUsize::new(128).unwrap());
compiler
    .extension_caches_mut()
    .set_limits(ExtensionCacheLimits::new(NonZeroUsize::new(128).unwrap()));

let mut executor = GraphExecutor::new(CpuBackend::new());
executor
    .extension_executor_mut()
    .set_cache_limits(ExtensionCacheLimits::new(NonZeroUsize::new(128).unwrap()));
executor.set_gemm_analysis_cache_capacity(512);
```

For CPU executors, the CPU buffer pool has its own retention limit:

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::GraphExecutor;

let mut executor = GraphExecutor::new(CpuBackend::new());
executor.set_buffer_pool_limit_bytes(32 * 1024 * 1024);
```

## Clearing Cached Memory

Clear caches when a long-running process changes workload phase, when a
notebook has finished a large experiment, or when memory pressure matters more
than reusing old plans and buffers.

```rust
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, GraphExecutor, Runtime};

let eager = EagerRuntime::with_cpu_backend(CpuBackend::new());
let runtime = Runtime::builder().build().unwrap();
let mut compiler = GraphCompiler::new();
let mut executor = GraphExecutor::new(CpuBackend::new());

runtime.clear_prepared_cache().unwrap();
compiler.clear_caches();
executor.clear_caches();
eager.clear_caches().unwrap();
```

For CPU executors, `clear_all_caches()` also clears the CPU buffer pool:

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::GraphExecutor;

let mut executor = GraphExecutor::new(CpuBackend::new());
executor.clear_all_caches();
```

`retained_bytes` in cache stats is tenferro's logical retained-data
estimate. It is not operating-system RSS and does not include allocator arena
slack, thread stacks, or provider-owned memory.

## CUDA Transfer Boundaries

CUDA transfers are explicit. Upload once near the boundary of a CUDA workload,
run supported operations on CUDA tensors, then download only when host
inspection or CPU execution is needed. Repeated upload/download inside tight
loops usually dominates runtime. See [Devices and GPU](devices-and-gpu.md) for
the current CUDA coverage and setup commands.
