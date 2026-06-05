# Parallelism and Caching

tenferro keeps CPU parallelism and execution caches explicit. Use this page to
control CPU thread counts, avoid provider oversubscription, and release cached
memory in long-running processes.

For tensor memory layout and column-major buffers, see
[Memory Order](memory-order.md). That is part of the tensor data model, not the
parallelism contract.

## CPU Backend Provider

At least one CPU provider feature must be compiled. `cpu-faer` is the default.
`cpu-blas` can be compiled by itself or together with `cpu-faer`.

`CpuBackend::new()` chooses a provider from the features compiled into the
current binary:

| Compiled CPU provider features | `CpuBackend::new()` provider |
| --- | --- |
| `cpu-faer` only | faer |
| `cpu-blas` only | BLAS/LAPACK |
| `cpu-faer` and `cpu-blas` | BLAS/LAPACK |

This is the default provider for that backend instance, not a dynamic fallback
chain. If both providers are compiled, select a provider explicitly when a
specific call path should use faer or BLAS. Explicit selection returns a
configuration error if the requested provider was not compiled into the binary:

```rust
use tenferro_cpu::CpuBackend;
use tenferro_cpu::CpuBackendKind;

let backend = CpuBackend::try_with_threads_and_kind(4, CpuBackendKind::Faer).unwrap();
assert_eq!(backend.num_threads(), 4);
assert_eq!(backend.kind(), CpuBackendKind::Faer);
```

## CPU Thread Count

Use `CpuBackend::with_threads(n)` when one backend should carry a fixed CPU
parallelism policy:

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::GraphExecutor;

let executor = GraphExecutor::new(CpuBackend::with_threads(4));
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
`OMP_NUM_THREADS`.

## Avoid Oversubscription

Do not accidentally multiply outer application parallelism by inner kernel
parallelism. If an outer loop already runs many independent tenferro calls in
parallel, use a smaller inner backend:

```rust
use tenferro_cpu::CpuBackend;

let backend = CpuBackend::with_threads(1);
```

For BLAS/LAPACK providers, apply the same rule to provider thread variables.
For benchmarks, pin all relevant thread counts and report them with the result.

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
));

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
eager.clear_caches();
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
