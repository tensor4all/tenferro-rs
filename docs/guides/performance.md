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

## Control CPU thread count

Use `CpuBackend::with_threads(n)` when you want explicit CPU parallelism control.

```rust
use tenferro::{CpuBackend, Engine};

let mut engine = Engine::new(CpuBackend::with_threads(4));
```

## Reuse the same engine

`Engine` is the right place to keep around between evaluations. In practice that means:

- Create one engine per workload or benchmark run.
- Reuse it across repeated evaluations.
- Avoid rebuilding the engine in tight loops unless you need to reset backend state.

## Buffer reuse is automatic

You do not need to manage scratch buffers manually. Keep your code simple, reuse the same `Engine`, and let tenferro reuse temporary storage behind the scenes.

## Einsum path optimization

For multi-input contractions, tenferro chooses a contraction order automatically and caches it on the engine. The normal advice is:

- Start with plain `einsum(&mut engine, ...)`.
- Reuse the same engine for repeated shapes and subscripts.
- Benchmark before trying to outsmart the optimizer.
