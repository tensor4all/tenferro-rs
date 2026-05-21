# Performance

tenferro is designed so the common fast path is also the normal user path: keep tensors lazy, reuse one engine, and let the backend handle execution details.

## Column-major storage

tenferro stores dense tensors in column-major order. That is the biggest difference PyTorch and JAX users usually need to internalize first.

```rust
use tenferro::TracedTensor;

let a = TracedTensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
```

This means the logical matrix is:

```text
[[1, 3, 5],
 [2, 4, 6]]
```

not:

```text
[[1, 2, 3],
 [4, 5, 6]]
```

If you are porting examples from PyTorch or JAX, check the flat-data order first.
Convert row-major flat data to column-major before calling `Tensor::from_vec`,
or write literals directly in column-major order. Owned export through
`try_into_vec::<T>()` returns the column-major host buffer.

## Control CPU thread count

Use `CpuBackend::with_threads(n)` when you want an engine-specific CPU
parallelism hint:

```rust
use tenferro::{CpuBackend, Engine};

let engine = Engine::new(CpuBackend::with_threads(4));
assert_eq!(engine.backend().num_threads(), 4);
```

`CpuBackend::new()` reads `RAYON_NUM_THREADS` and falls back to the
process-visible CPU count when the variable is unset. This is convenient for
applications that configure parallelism at process startup:

```bash
RAYON_NUM_THREADS=4 cargo run --release
```

For the default `cpu-faer` backend, tenferro passes this value to faer as a
kernel parallelism hint: one thread uses sequential execution, and larger values
use faer's Rayon-backed parallel path. tenferro does not create a private Rayon
thread pool and does not pin faer work to a socket. `RAYON_NUM_THREADS` still
matters for applications that rely on the process-global Rayon pool, so set it
consistently with the backend thread budget.

If you need socket-level placement or independent worker pools, split the
workload at the process level for example with MPI or `taskset`/`numactl`, and
give each process its own CPU thread budget.

For `cpu-blas` builds, also configure the BLAS/OpenMP provider. A common
single-process setup is:

```bash
RAYON_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
VECLIB_MAXIMUM_THREADS=4 \
./your-tenferro-app
```

Set only the variables used by your BLAS provider. When running many processes
or outer task parallelism, lower both `RAYON_NUM_THREADS` and BLAS/OpenMP thread
counts per process to avoid oversubscription. For small tensor-network
contractions, start with one CPU thread per process and increase only after
benchmarking the target workload.

## Reuse the same engine

`Engine` is the right place to keep around between evaluations. In practice that means:

- Create one engine per workload or benchmark run.
- Reuse it across repeated evaluations.
- Avoid rebuilding the engine in tight loops unless you need to reset backend state.

## Cache management

`Engine` owns the long-lived runtime caches used by traced execution: compiled
execution programs, parsed einsum notation, optimized N-ary einsum plans, and
backend-specific analysis such as CPU GEMM shape analysis. These caches are
bounded by default and can be inspected or cleared explicitly.

```rust
use std::num::NonZeroUsize;
use tenferro::{CpuBackend, Engine};

let mut engine = Engine::new(CpuBackend::new());

engine.set_compile_cache_capacity(NonZeroUsize::new(128).unwrap());
engine.set_einsum_cache_capacity(NonZeroUsize::new(128).unwrap());
engine.set_gemm_analysis_cache_capacity(512);

let stats = engine.cache_stats();
assert_eq!(stats.compile.entries, 0);
assert_eq!(stats.einsum_plans.entries, 0);
assert_eq!(stats.backend.entries, 0);

engine.clear_caches();
```

For CPU engines, `cpu_cache_stats()` also reports the CPU buffer pool.
`clear_all_caches()` clears engine-owned caches and the CPU buffer pool.
CPU thread count is a kernel-level parallelism hint; tenferro does not retain
or expose a process-wide CPU thread-pool cache.

```rust
use tenferro::{CpuBackend, Engine};

let mut engine = Engine::new(CpuBackend::new());
engine.set_buffer_pool_limit_bytes(32 * 1024 * 1024);

let stats = engine.cpu_cache_stats();
assert_eq!(stats.buffer_pool.entries, 0);

engine.clear_all_caches();
assert_eq!(engine.cpu_cache_stats().buffer_pool.entries, 0);
```

`retained_bytes` in cache stats is tenferro's logical retained-payload estimate.
It is not operating-system RSS and does not include allocator arena slack.

## Buffer reuse is automatic

You do not need to manage scratch buffers manually. Keep your code simple, reuse the same `Engine`, and let tenferro reuse temporary storage behind the scenes.

## CUDA transfer boundaries

CUDA transfers are explicit. Upload once near the boundary of a CUDA workload,
run supported operations on CUDA tensors, then download only when host
inspection or CPU execution is needed. Repeated upload/download inside tight
loops usually dominates runtime. See [Devices and GPU](devices-and-gpu.md) for
the current CUDA coverage and setup commands.

## Einsum path optimization

For multi-input contractions, tenferro chooses a contraction order automatically and caches it on the engine. The normal advice is:

- Start with plain `einsum(&mut engine, ...)`.
- Reuse the same engine for repeated shapes and subscripts.
- Benchmark before trying to outsmart the optimizer.

Standalone eager einsum in the `tenferro-einsum` crate has its own per-thread
bounded plan cache. Use `eager_einsum_cache_stats()`,
`set_eager_einsum_cache_capacity(...)`, and `clear_eager_einsum_cache()` when
calling that crate directly outside the `tenferro::Engine` path.
