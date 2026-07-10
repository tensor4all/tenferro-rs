# Parallelism and Caching

tenferro keeps CPU parallelism and execution caches explicit. Use this page to
control CPU thread counts, avoid provider oversubscription, and release cached
memory in long-running processes.

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

This is the default provider for that backend instance. If multiple providers
are compiled, select a provider explicitly when a specific call path should use
faer, BLAS, or TBLIS. `CpuBackendKind::Tblis` first attempts supported TBLIS
`dot_general` contractions and then falls back to the compiled faer/BLAS
provider for shapes outside the TBLIS path. Explicit selection returns a
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
Standalone backend calls and compiled `BackendSession` execution enter that
pool before dispatching tensor-sized kernels.

| Operation family | Threading behavior |
| --- | --- |
| Elementwise and analytic ops | `strided-kernel` map/zip kernels run under `CpuContext` and can use Rayon when the context has more than one thread. |
| Reductions | `strided-kernel::reduce_axis` runs under `CpuContext` and can use Rayon when the context has more than one thread. |
| Materialized transpose/permute, broadcast, convert, and diagonal extraction | `strided-kernel` copy/map kernels run under `CpuContext` and can use Rayon for tensor-sized copies. |
| `dot_general` through `cpu-faer` | faer receives `Par::Seq` for one-thread contexts and `Par::rayon(0)` inside the owned `CpuContext` pool for multi-thread contexts. |
| GEMM and linalg through `cpu-blas` | Threading is owned by the linked BLAS/LAPACK provider, not Rayon. Configure the provider variables below. |
| Supported `dot_general` contractions through `cpu-tblis` | TBLIS owns provider threading; unsupported TBLIS shapes fall back to the compiled faer/BLAS provider. |
| Indexing, scatter/gather, slicing, padding, concatenation, reverse, triangular masks, and `embed_diagonal` | These are dedicated sequential CPU loops today because their per-output indexing patterns do not yet have a strided-kernel/backend-native parallel primitive. They still run inside `CpuContext::install`, and source comments mark the intentional sequential path. |

## BLAS And LAPACK Threads

For `cpu-blas`, `CpuBackend::with_threads(n)` controls tenferro's CPU context,
but the linked BLAS/LAPACK provider has its own thread controls. Set provider
thread variables before process start:

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

- `EagerRuntime` retains eager extension plans and compiled inner extension
  programs across immediate operations.
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

Compiler, executor, and eager caches are bounded by default and can be
configured independently.

```rust
use std::num::NonZeroUsize;
use tenferro_ad::EagerRuntime;
use tenferro_runtime::extension::ExtensionCacheLimits;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, GraphExecutor};

let eager = EagerRuntime::with_cpu_backend(CpuBackend::new());
eager.set_extension_cache_limits(ExtensionCacheLimits::new(
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
use tenferro_runtime::{GraphCompiler, GraphExecutor};

let eager = EagerRuntime::with_cpu_backend(CpuBackend::new());
let mut compiler = GraphCompiler::new();
let mut executor = GraphExecutor::new(CpuBackend::new());

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
