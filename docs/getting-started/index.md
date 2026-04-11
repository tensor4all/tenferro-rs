# Getting Started

tenferro gives you a lazy tensor handle, an execution engine, and a small set of familiar tensor-building blocks. If you already know PyTorch or JAX, the main difference is that you build a computation first and explicitly evaluate it later.

## Installation

Use a local checkout while the crate is still evolving:

```toml
[dependencies]
tenferro = { path = "/path/to/tenferro-rs/tenferro" }
```

Or switch to crates.io once published:

```toml
[dependencies]
tenferro = "..."
```

## Hello einsum

This is the tenferro equivalent of `torch.einsum("ij,jk->ik", a, b)` or `jnp.einsum("ij,jk->ik", a, b)`.

```rust
use tenferro::{einsum::einsum, CpuBackend, Engine, TracedTensor};

let a = TracedTensor::new(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
let b = TracedTensor::new(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

let mut engine = Engine::new(CpuBackend::new());
let mut c = einsum(&mut engine, &[&a, &b], "ij,jk->ik").unwrap();
let result = c.eval(&mut engine).unwrap();

assert_eq!(result.shape(), &[2, 2]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
```

## Hello grad

This is the tenferro equivalent of differentiating `sum(x * x)` in PyTorch or JAX.

```rust
use tenferro::{CpuBackend, Engine, TracedTensor};

let x = TracedTensor::new(vec![3], vec![1.0_f64, 2.0, 3.0]);
let loss = (&x * &x).reduce_sum(&[0]);
let mut grad = loss.grad(&x).unwrap();

let mut engine = Engine::new(CpuBackend::new());
let result = grad.eval(&mut engine).unwrap();

assert_eq!(result.shape(), &[3]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);
```

## Next Steps

- [Core concepts](./core-concepts.md)
- [PyTorch and JAX mapping](./pytorch-jax-mapping.md)
- [Tensor operations guide](../guides/tensor-operations.md)
- [Einsum guide](../guides/einsum.md)
- [Autodiff guide](../guides/autodiff.md)
- [Linear algebra guide](../guides/linear-algebra.md)
