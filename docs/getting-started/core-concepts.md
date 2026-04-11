# Core Concepts

## TracedTensor: your tensor handle

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

| Library | Typical mental model |
|---|---|
| PyTorch | Eager: each operation produces data immediately |
| JAX | Usually eager arrays, with `jit` used to stage larger computations |
| tenferro | Lazy by default: compose first, then call `eval` |

## Minimal example

```rust
use tenferro::{CpuBackend, Engine, TracedTensor};

let a = TracedTensor::new(vec![2], vec![1.0_f64, 2.0]);
let b = TracedTensor::new(vec![2], vec![3.0_f64, 4.0]);
let mut sum = &a + &b;

let mut engine = Engine::new(CpuBackend::new());
let result = sum.eval(&mut engine).unwrap();

assert_eq!(result.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
```
