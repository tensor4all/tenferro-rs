# Eager Operations

This guide covers using tenferro for direct computation with eager
reverse-mode autodiff on scalar losses: the path you would choose for
NumPy-like workflows where you control the computation explicitly and call
`backward()` when you need gradients.

## Setup

```rust
use tenferro::{CpuBackend, Tensor, TypedTensor};

let mut ctx = CpuBackend::new();
```

Every eager operation requires a backend context. `CpuBackend` is the standard
CPU backend using the faer linear algebra library. With the `cuda` feature, the
same concrete and eager APIs can execute supported operations on the CubeCL/CUDA
backend when tensors are explicitly placed on the GPU.

`EagerContext` is the gradient-owning wrapper for eager AD state. If you share
one context across multiple tracked tensors, their gradients accumulate into
the same state and you can reset them together with `clear_grads()`.

Most eager operations are methods on `Tensor`. `TypedTensor<T>` is useful when
you want compile-time dtype safety for construction or direct host-side data
access.

## Creating tensors

```rust
use tenferro::{Tensor, TypedTensor};

// Dynamic dtype (`Tensor`)
let a = Tensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

// Static dtype (`TypedTensor`)
let b = TypedTensor::<f64>::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

// Convert between layers for a specific dtype.
let c = Tensor::F64(b.clone());
assert_eq!(c.shape(), &[2, 3]);
```

The flat buffers above are in column-major order, so a `[2, 3]` tensor stores
its columns as `[1, 2]`, `[3, 4]`, and `[5, 6]`.

## Arithmetic

```rust
use tenferro::{CpuBackend, Tensor};

let mut ctx = CpuBackend::new();
let a = Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]);
let b = Tensor::from_vec(vec![3], vec![4.0_f64, 5.0, 6.0]);

let sum = a.add(&b, &mut ctx).unwrap();
let product = a.mul(&b, &mut ctx).unwrap();
let negated = a.neg(&mut ctx).unwrap();

assert_eq!(sum.as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);
assert_eq!(product.as_slice::<f64>().unwrap(), &[4.0, 10.0, 18.0]);
assert_eq!(negated.as_slice::<f64>().unwrap(), &[-1.0, -2.0, -3.0]);
```

## Linear algebra

```rust
use tenferro::{CpuBackend, Tensor};

let mut ctx = CpuBackend::new();
let a = Tensor::from_vec(vec![3, 3], vec![
    2.0_f64, 1.0, 0.0,
    1.0, 3.0, 1.0,
    0.0, 1.0, 2.0,
]);

// SVD
let (_u, s, _vt) = a.svd(&mut ctx).unwrap();

// QR
let (_q, _r) = a.qr(&mut ctx).unwrap();

// Cholesky (for positive definite matrices)
let chol = a.cholesky(&mut ctx).unwrap();

// Eigendecomposition (symmetric)
let (eigenvalues, eigenvectors) = a.eigh(&mut ctx).unwrap();

// Solve Ax = b
let b = Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]);
let x = a.solve(&b, &mut ctx).unwrap();

assert_eq!(s.shape(), &[3]);
assert_eq!(chol.shape(), &[3, 3]);
assert_eq!(eigenvalues.shape(), &[3]);
assert_eq!(eigenvectors.shape(), &[3, 3]);
assert_eq!(x.shape(), &[3]);
```

## Shape operations

```rust
use tenferro::{CpuBackend, Tensor};

let mut ctx = CpuBackend::new();
let a = Tensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

// Transpose
let at = a.transpose(&[1, 0], &mut ctx).unwrap();
assert_eq!(at.shape(), &[3, 2]);

// Reshape
let flat = a.reshape(&[6], &mut ctx).unwrap();
assert_eq!(flat.shape(), &[6]);

// Reduce
let col_sum = a.reduce_sum(&[0], &mut ctx).unwrap();
assert_eq!(col_sum.shape(), &[3]);
```

## Einsum

```rust
use tenferro::tensor::einsum;
use tenferro::{CpuBackend, Tensor};

let mut ctx = CpuBackend::new();

// Column-major buffers: `a` has columns [1, 2], [3, 4], [5, 6].
let a = Tensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
let b = Tensor::from_vec(vec![3, 4], vec![
    1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0,
    7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
]);

let c = einsum(&mut ctx, &[&a, &b], "ij,jk->ik").unwrap();
assert_eq!(c.shape(), &[2, 4]);
```

## Extracting data

```rust
use tenferro::Tensor;

let t = Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]);
let data: &[f64] = t.as_slice::<f64>().unwrap();
assert_eq!(data, &[1.0, 2.0, 3.0]);
```

## Column-major storage

tenferro stores tensors in column-major (Fortran) order. For a `[2, 3]` tensor
with data `[1, 2, 3, 4, 5, 6]`, the layout is:

```text
Column 0: [1, 2]
Column 1: [3, 4]
Column 2: [5, 6]
```

This matches Fortran, Julia, and MATLAB conventions but differs from C/NumPy
row-major order.

## Eager reverse-mode gradients

Eager tensors support scalar-loss reverse-mode autodiff with accumulation.
Repeated `backward()` calls add to the existing gradients, and you clear them
explicitly when you want a fresh pass.

```rust
use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};

let ctx = EagerContext::with_backend(CpuBackend::new());
let x = EagerTensor::requires_grad_in(Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]), ctx.clone());
let y = EagerTensor::requires_grad_in(Tensor::from_vec(vec![2], vec![3.0_f64, 4.0]), ctx.clone());

let loss = (&x * &y).reduce_sum(&[0]).unwrap();
loss.backward().unwrap();
assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[3.0, 4.0]);

let loss = (&x * &y).reduce_sum(&[0]).unwrap();
loss.backward().unwrap();
assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[6.0, 8.0]);

x.clear_grad();
assert!(x.grad().is_none());

let loss = (&x * &y).reduce_sum(&[0]).unwrap();
loss.backward().unwrap();
assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[3.0, 4.0]);

ctx.clear_grads();
assert!(x.grad().is_none());
assert!(y.grad().is_none());
```

## When to use eager vs lazy

| Scenario | Recommended |
|----------|-------------|
| Data preprocessing | Eager (`Tensor` + a backend) |
| TCI inner loops | Eager |
| Exploratory computation | Eager |
| Need scalar-loss reverse-mode gradients | Eager (`EagerTensor::backward()`) |
| Need transform AD (`grad` / `vjp` / `jvp` / HVP) | Lazy traced (`TracedTensor` + `Engine<B>`) |
| CUDA execution for supported operations | Eager (`Tensor` / `EagerTensor<B>`) or lazy traced (`TracedTensor` + `Engine<B>`) with explicit upload/download |
