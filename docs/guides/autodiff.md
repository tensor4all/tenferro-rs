# Autodiff

tenferro exposes transform-oriented autodiff directly on `TracedTensor`.
Eager tensors also support scalar-loss reverse-mode via `backward()`, but
this page is about the traced APIs. If you know PyTorch or JAX, the mental
model is:

- `grad` for scalar-loss reverse mode
- `vjp` for vector-Jacobian products
- `jvp` for Jacobian-vector products
- Higher-order AD via composition (e.g., `jvp(grad(f))` for Hessian-vector products)

## Reverse-mode gradient with `grad`

PyTorch equivalent: `torch.autograd.grad(loss, x)`  
JAX equivalent: `jax.grad(f)(x)`

If you want eager scalar-loss accumulation instead, see the eager operations
guide and use `EagerTensor::backward()`.

```rust
use tenferro::{CpuBackend, Engine, TracedTensor};

let x = TracedTensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]);
let loss = (&x * &x).reduce_sum(&[0]);
let mut grad = loss.grad(&x).unwrap();

let mut engine = Engine::new(CpuBackend::new());
let result = grad.eval(&mut engine).unwrap();

assert_eq!(result.shape(), &[3]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);
```

## Gradient through einsum

Like PyTorch and JAX, tenferro differentiates through tensor contractions.

```rust
use tenferro::{einsum::einsum, CpuBackend, Engine, TracedTensor};

let a = TracedTensor::from_vec(vec![2, 3], vec![1.0_f64, -2.0, 0.5, 3.0, 1.25, -0.75]);
let b = TracedTensor::from_vec(vec![3, 2], vec![2.0_f64, 0.25, -1.5, 4.0, 0.75, -0.5]);

let mut engine = Engine::new(CpuBackend::new());
let y = einsum(&mut engine, &[&a, &b], "ij,jk->ik").unwrap();
let loss = y.reduce_sum(&[0, 1]);
let mut grad_a = loss.grad(&a).unwrap();

let result = grad_a.eval(&mut engine).unwrap();
assert_eq!(result.shape(), &[2, 3]);
```

## Vector-Jacobian product with `vjp`

PyTorch equivalent: `torch.autograd.grad(y, a, grad_outputs=cotangent)`  
JAX equivalent: `jax.vjp`

```rust
use tenferro::{einsum::einsum, CpuBackend, Engine, TracedTensor};

let a = TracedTensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
let b = TracedTensor::from_vec(vec![3, 2], vec![0.5_f64, -1.0, 2.0, 1.5, -0.25, 3.0]);
let cotangent = TracedTensor::from_vec(vec![2, 2], vec![1.0_f64, -0.5, 0.25, 2.0]);

let mut engine = Engine::new(CpuBackend::new());
let y = einsum(&mut engine, &[&a, &b], "ij,jk->ik").unwrap();
let mut ct_a = y.vjp(&a, &cotangent);

let result = ct_a.eval(&mut engine).unwrap();
assert_eq!(result.shape(), &[2, 3]);
```

## Jacobian-vector product with `jvp`

PyTorch equivalent: `torch.func.jvp`  
JAX equivalent: `jax.jvp`

```rust
use tenferro::{einsum::einsum, CpuBackend, Engine, TracedTensor};

let a = TracedTensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
let b = TracedTensor::from_vec(vec![3, 2], vec![0.5_f64, -1.0, 2.0, 1.5, -0.25, 3.0]);
let tangent = TracedTensor::from_vec(vec![2, 3], vec![1.0_f64, -0.5, 0.25, 0.0, 2.0, -1.0]);

let mut engine = Engine::new(CpuBackend::new());
let y = einsum(&mut engine, &[&a, &b], "ij,jk->ik").unwrap();
let mut dy = y.jvp(&a, &tangent);

let result = dy.eval(&mut engine).unwrap();
assert_eq!(result.shape(), &[2, 2]);
```
