# Getting Started

tenferro supports both eager tensor execution for direct computation and lazy
traced tensors for automatic differentiation. If you already know NumPy,
PyTorch, or JAX, choose the layer that matches the workflow you need.

## Installation

The `tenferro` crate re-exports everything you need. Use a local checkout
while the crates are still evolving:

```toml
[dependencies]
tenferro = { path = "/path/to/tenferro-rs/tenferro" }
```

Or switch to crates.io once published:

```toml
[dependencies]
tenferro = "..."
```

> If you only need eager tensor operations without tracing or AD, you can
> depend on `tenferro-tensor` alone for a smaller build.

## Hello eager

This is the simplest way to use tenferro: direct computation without tracing or
AD, similar to NumPy.

```rust
use tenferro_tensor::{Tensor, cpu::CpuBackend};

let mut ctx = CpuBackend::new();

// tenferro uses column-major (Fortran) storage, not row-major like NumPy.
// For shape [2, 3] with data [1,2,3,4,5,6]:
//   column 0 = [1, 2], column 1 = [3, 4], column 2 = [5, 6]
let a = Tensor::new(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

// SVD
let (_u, s, _vt) = a.svd(&mut ctx).unwrap();
assert_eq!(s.shape(), &[2]);

// Matrix multiply
let b = Tensor::new(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
let c = a.matmul(&b, &mut ctx).unwrap();
assert_eq!(c.shape(), &[2, 2]);
```

## Hello einsum (lazy)

This is the tenferro equivalent of `torch.einsum("ij,jk->ik", a, b)` or `jnp.einsum("ij,jk->ik", a, b)`.

```rust,ignore
use tenferro::{einsum::einsum, CpuBackend, Engine, TracedTensor};

// Column-major buffers: `a` has columns [1, 2], [3, 4], [5, 6].
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

```rust,ignore
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
- [Eager operations guide](../guides/eager-operations.md)
- [Tensor operations guide](../guides/tensor-operations.md)
- [Einsum guide](../guides/einsum.md)
- [Autodiff guide](../guides/autodiff.md)
- [Linear algebra guide](../guides/linear-algebra.md)
