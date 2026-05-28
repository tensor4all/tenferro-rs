# Eager Operations

This guide covers immediate execution: direct no-AD tensor computation and
`EagerTensor` forward execution with optional PyTorch-like reverse-mode
autodiff on scalar losses. Start with `TypedTensor<T, R>` or `Tensor` for
no-AD work. Use `EagerTensor` when you want operations to run immediately
inside an `EagerRuntime`, and create tracked variables when the workflow needs
gradient accumulation and `backward()`.

## Setup

```rust
use tenferro_runtime::{CpuBackend, Tensor, TypedTensor};

let mut ctx = CpuBackend::new();
```

Every direct tensor operation requires a backend context. `CpuBackend` is the
standard CPU backend using the faer linear algebra library. With the `cuda`
feature, the same concrete and eager surfaces can execute supported operations
on the CUDA backend when tensors are explicitly placed on the GPU.

`EagerRuntime` owns the eager backend and the optional gradient slots for
tracked eager tensors. Untracked eager tensors are forward-only. If you share
one context across multiple tracked tensors, their gradients accumulate into
the same state and you can reset them together with `clear_grads()`.

Most broad concrete operations are available as `tenferro_runtime::tensor` free
functions, with method wrappers kept for compatibility. `TypedTensor<T, R>` is
the first layer to consider when you want compile-time dtype safety, optional
rank typing, or typed data that may be host-backed or backend-backed. Einsum is
provided by the separate `tenferro-einsum` standard extension.

For CUDA, eager means the operation is submitted immediately. It does not mean
the host waits after every GPU kernel. Host synchronization happens at
download/read boundaries or inside operations that must inspect device-side
status. See [Execution Models](execution-models.md) and
[Devices and GPU](devices-and-gpu.md).

## Creating tensors

```rust
use tenferro_runtime::{Tensor, TypedTensor};
use tenferro_tensor::Rank;

// Dynamic dtype (`Tensor`)
let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

// Static dtype (`TypedTensor`)
let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let ranked: TypedTensor<f64, Rank<2>> = b.clone().try_into_rank::<2>().unwrap();
assert_eq!(ranked.shape(), &[2, 3]);

// Convert between layers for a specific dtype.
let c = Tensor::F64(b.clone());
assert_eq!(c.shape(), &[2, 3]);
```

The flat buffers above are in column-major order, so a `[2, 3]` tensor stores
its columns as `[1, 2]`, `[3, 4]`, and `[5, 6]`.
Owned tensors stay compact column-major. Metadata-only strided views live on
`TypedTensorView` and `TypedTensorViewMut`; compact-only operation boundaries
may canonicalize such views within the same placement, but they do not silently
upload CPU tensors or download CUDA tensors.

## Arithmetic

```rust
use tenferro_runtime::{tensor, CpuBackend, Tensor};

let mut ctx = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
let b = Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]);

let sum = tensor::add(&a, &b, &mut ctx).unwrap();
let product = tensor::mul(&a, &b, &mut ctx).unwrap();
let negated = tensor::neg(&a, &mut ctx).unwrap();

assert_eq!(sum.as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);
assert_eq!(product.as_slice::<f64>().unwrap(), &[4.0, 10.0, 18.0]);
assert_eq!(negated.as_slice::<f64>().unwrap(), &[-1.0, -2.0, -3.0]);
```

## Linear algebra

```rust
use tenferro_linalg::LinalgBackend;
use tenferro_runtime::{tensor, CpuBackend, Tensor};

let mut ctx = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![3, 3], vec![
    2.0_f64, 1.0, 0.0,
    1.0, 3.0, 1.0,
    0.0, 1.0, 2.0,
]);

// SVD
let svd = LinalgBackend::svd(&mut ctx, &a).unwrap();

// QR
let qr = LinalgBackend::qr(&mut ctx, &a).unwrap();

// Cholesky (for positive definite matrices)
let chol = LinalgBackend::cholesky(&mut ctx, &a).unwrap();

// Eigendecomposition (symmetric)
let eigh = LinalgBackend::eigh(&mut ctx, &a).unwrap();

// Solve Ax = b
let b = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
let x = LinalgBackend::solve(&mut ctx, &a, &b).unwrap();

let s = &svd[1];
assert_eq!(s.shape(), &[3]);
assert_eq!(qr[0].shape(), &[3, 3]);
assert_eq!(chol.shape(), &[3, 3]);
let eigenvalues = &eigh[0];
let eigenvectors = &eigh[1];
assert_eq!(eigenvalues.shape(), &[3]);
assert_eq!(eigenvectors.shape(), &[3, 3]);
assert_eq!(x.shape(), &[3]);
```

## Shape operations

```rust
use tenferro_runtime::{CpuBackend, Tensor};

let mut ctx = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

// Transpose
let at = tensor::transpose(&a, &[1, 0], &mut ctx).unwrap();
assert_eq!(at.shape(), &[3, 2]);

// Reshape
let flat = tensor::reshape(&a, &[6], &mut ctx).unwrap();
assert_eq!(flat.shape(), &[6]);

// Reduce
let col_sum = tensor::reduce_sum(&a, &[0], &mut ctx).unwrap();
assert_eq!(col_sum.shape(), &[3]);
```

## Einsum

Use `tenferro_einsum::eager_tensor::einsum` when working with `EagerTensor`.
For traced graph execution, use `tenferro_einsum::traced_tensor::einsum` and
register `tenferro_einsum::register_runtime` on the `GraphExecutor`.

## Extracting data

```rust
use tenferro_runtime::Tensor;

let t = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
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

## Eager Forward And Reverse-Mode Gradients

Eager tensors always compute the forward value immediately. Tracked eager
tensors also support scalar-loss reverse-mode autodiff with accumulation.
Repeated `backward()` calls add to the existing gradients, and you clear them
explicitly when you want a fresh pass.

```rust
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_runtime::CpuBackend;

let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]), ctx.clone());
let y = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]), ctx.clone());

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

## When To Use Each Immediate Layer

| Scenario | Recommended |
|----------|-------------|
| Fixed scalar type and no AD | `TypedTensor<T, R>` |
| Dynamic dtype and no AD | `Tensor` + a backend |
| Data preprocessing | `Tensor` + a backend |
| Tight inner loops | Direct/eager execution |
| Exploratory computation | Direct/eager execution |
| Immediate forward execution through one runtime | `EagerTensor` |
| Need scalar-loss reverse-mode gradients | tracked `EagerTensor` variables + `backward()` |
| Need transform AD (`grad` / `vjp` / `jvp` / HVP via composition) | Lazy traced (`TracedTensor` + `GraphCompiler` + `GraphExecutor<B>`) |
| CUDA execution for supported operations | Eager (`Tensor` / `EagerTensor`) or lazy traced (`TracedTensor` + `GraphExecutor<B>`) with explicit upload/download |
