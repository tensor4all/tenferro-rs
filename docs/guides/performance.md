# Performance

tenferro's fast path is the normal path: use explicit memory order, reuse graph
compilation state, and reuse backend execution state.

## Column-major storage

tenferro stores dense tensors in column-major order.

```rust
use tenferro_runtime::TracedTensor;

let a = TracedTensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);
```

This logical matrix is:

```text
[[1, 3, 5],
 [2, 4, 6]]
```

Use `from_vec_row_major` for flat data copied from row-major systems such as
PyTorch, NumPy, or JAX. Owned export through `try_into_vec_col_major::<T>()`
returns the physical column-major host buffer.

## Control CPU thread count

Use `CpuBackend::with_threads(n)` when you want an execution-specific CPU
parallelism hint:

```rust
use tenferro_runtime::{CpuBackend, GraphExecutor};

let executor = GraphExecutor::new(CpuBackend::with_threads(4));
assert_eq!(executor.backend().num_threads(), 4);
```

`CpuBackend::new()` reads `RAYON_NUM_THREADS` and falls back to the
process-visible CPU count when the variable is unset.

```bash
RAYON_NUM_THREADS=4 cargo run --release
```

For `cpu-blas` builds, also configure the BLAS/OpenMP provider used by your
system:

```bash
RAYON_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
VECLIB_MAXIMUM_THREADS=4 \
./your-tenferro-app
```

## Reuse compiler and executor state

Use one `EagerRuntime` per eager backend context when you want to retain eager
einsum plans across immediate operations. Use one `GraphCompiler` per traced
workload when you want to retain graph lowering and static einsum planning
caches. Use one `GraphExecutor<B>` per backend execution context when you want
to retain runtime einsum plans, backend analysis, and reusable CPU buffers.

## Cache management

Compiler caches and executor caches are bounded by default and can be inspected
or cleared independently.

```rust
use std::num::NonZeroUsize;
use tenferro_runtime::extension::ExtensionCacheLimits;
use tenferro_ad::EagerRuntime;
use tenferro_runtime::{CpuBackend, GraphCompiler, GraphExecutor};

let eager = EagerRuntime::with_cpu_backend(CpuBackend::new());
eager.set_extension_cache_limits(ExtensionCacheLimits::new(NonZeroUsize::new(128).unwrap()));
assert_eq!(eager.cache_stats().extensions.entries, 0);

let mut compiler = GraphCompiler::new();
compiler.set_compile_cache_capacity(NonZeroUsize::new(128).unwrap());
compiler
    .extension_caches_mut()
    .set_limits(ExtensionCacheLimits::new(NonZeroUsize::new(128).unwrap()));
assert_eq!(compiler.cache_stats().compile.entries, 0);

let mut executor = GraphExecutor::new(CpuBackend::new());
executor
    .extension_executor_mut()
    .set_cache_limits(ExtensionCacheLimits::new(NonZeroUsize::new(128).unwrap()));
executor.set_gemm_analysis_cache_capacity(512);
assert_eq!(executor.cache_stats().extensions.entries, 0);
assert_eq!(executor.cache_stats().backend.entries, 0);

compiler.clear_caches();
executor.clear_caches();
eager.clear_caches();
```

For CPU executors, `cpu_cache_stats()` also reports the CPU buffer pool.
`clear_all_caches()` clears executor-owned caches and the CPU buffer pool.

```rust
use tenferro_runtime::{CpuBackend, GraphExecutor};

let mut executor = GraphExecutor::new(CpuBackend::new());
executor.set_buffer_pool_limit_bytes(32 * 1024 * 1024);

let stats = executor.cpu_cache_stats();
assert_eq!(stats.buffer_pool.entries, 0);

executor.clear_all_caches();
assert_eq!(executor.cpu_cache_stats().buffer_pool.entries, 0);
```

`retained_bytes` in cache stats is tenferro's logical retained-payload
estimate. It is not operating-system RSS and does not include allocator arena
slack.

## CUDA transfer boundaries

CUDA transfers are explicit. Upload once near the boundary of a CUDA workload,
run supported operations on CUDA tensors, then download only when host
inspection or CPU execution is needed. Repeated upload/download inside tight
loops usually dominates runtime. See [Devices and GPU](devices-and-gpu.md) for
the current CUDA coverage and setup commands.

## Einsum path optimization

For multi-input traced contractions, `GraphCompiler` plans concrete-shape
einsums and `GraphExecutor` caches runtime plans for symbolic-shape einsums.
Reuse both objects for repeated shapes and subscripts.

Direct and typed einsum routes execute immediately without retaining a hidden
plan cache. `EagerTensor` einsum uses the owning `EagerRuntime` extension cache.
For traced workloads, reuse `GraphCompiler` and `GraphExecutor<B>` to retain
compile-time and runtime extension plans.
