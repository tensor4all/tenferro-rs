# Core Concepts

## Three tensor layers

tenferro provides three tensor types for different use cases:

| Layer | Type | Use case | Requires Engine? |
|-------|------|----------|-----------------|
| `TypedTensor<T>` | Statically typed | Direct computation, compile-time dtype safety | No (needs `CpuBackend`) |
| `Tensor` | Dynamic dtype enum | Mixed-dtype workflows, FFI | No (needs `CpuBackend`) |
| `TracedTensor` | Lazy graph handle | Automatic differentiation, graph optimization | Yes |

Choose the simplest layer that meets your needs:

- **No AD needed** -> `Tensor` + `CpuBackend`
- **AD (grad/vjp/jvp) needed** -> `TracedTensor` + `Engine`
- **Compile-time dtype safety** -> `TypedTensor<T>` + `CpuBackend`

## TypedTensor<T>: statically typed storage

`TypedTensor<T>` stores concrete tensor data with a compile-time scalar type.
Use it when you want dtype safety in Rust code and you do not want runtime
dtype dispatch.

```rust
use tenferro_tensor::{Tensor, TypedTensor};

// Column-major buffer: columns are [1, 2], [3, 4], [5, 6].
let typed = TypedTensor::<f64>::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
assert_eq!(typed.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

// Wrap a typed tensor in the matching dynamic enum when you need `Tensor`.
let dynamic = Tensor::F64(typed.clone());
assert_eq!(dynamic.shape(), &[2, 3]);
```

## Tensor: eager computation

`Tensor` holds concrete data and supports immediate computation. Every
operation takes a backend context (`&mut impl TensorBackend`) and returns
results immediately.

```rust
use tenferro_tensor::{Tensor, TensorBackend, cpu::CpuBackend};

let mut ctx = CpuBackend::new();

// Column-major buffer: columns are [1, 2], [3, 4], [5, 6].
let a = Tensor::new(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
let b = Tensor::new(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

let c = a.matmul(&b, &mut ctx).unwrap();
assert_eq!(c.shape(), &[2, 2]);

let (_u, s, _vt) = a.svd(&mut ctx).unwrap();
assert_eq!(s.shape(), &[2]); // min(2, 3) singular values
```

If you have used NumPy or PyTorch in eager mode, this is the familiar pattern.

## TracedTensor: lazy computation with AD

`TracedTensor` is the value you pass around in user code. You create it from a shape and flat data, then combine it with operations such as `+`, `*`, `reshape`, `transpose`, `einsum`, or `svd`.

In PyTorch terms, a `TracedTensor` feels like a tensor object you can keep composing. In JAX terms, it is closer to building up a staged computation, except tenferro keeps the staging model explicit instead of hiding it behind `jit`.

## Engine: the executor

`Engine` owns the backend that actually runs your computation. It also keeps reusable execution state, so the normal pattern is to build one engine and reuse it across many evaluations.

If you think in PyTorch, `Engine` is the closest thing to the runtime that turns your tensor program into real values. If you think in JAX, it plays the role of the execution context you pass to traced work, but as an explicit Rust value.

## eval(): get results

Operations on `TracedTensor` are lazy. Nothing is materialized until you call `.eval(&mut engine)`, which returns a concrete `Tensor`.

```text
Input data -> TracedTensor -> operations -> .eval(&mut engine) -> Tensor result
```

## Execution model comparison

| Library | Mental model | tenferro equivalent |
|---|---|---|
| NumPy | Eager, no AD | `Tensor` + `CpuBackend` |
| PyTorch (eager) | Eager with autograd | `TracedTensor` + `Engine` |
| JAX (`jit`) | Staged/lazy computation | `TracedTensor` + `Engine` |

## Minimal examples

### Eager (no AD)

```rust
use tenferro_tensor::{Tensor, TensorBackend, cpu::CpuBackend};

let mut ctx = CpuBackend::new();
let a = Tensor::new(vec![2], vec![1.0_f64, 2.0]);
let b = Tensor::new(vec![2], vec![3.0_f64, 4.0]);
let sum = a.add(&b, &mut ctx).unwrap();

assert_eq!(sum.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
```

### Lazy with AD

```rust
use tenferro::{CpuBackend, Engine, TracedTensor};

let a = TracedTensor::new(vec![2], vec![1.0_f64, 2.0]);
let b = TracedTensor::new(vec![2], vec![3.0_f64, 4.0]);
let mut sum = &a + &b;

let mut engine = Engine::new(CpuBackend::new());
let result = sum.eval(&mut engine).unwrap();

assert_eq!(result.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
```
