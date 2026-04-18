# Tensor Operations

This guide covers the everyday tensor operations you would usually reach for in PyTorch or JAX: creating tensors, applying elementwise math, changing shapes, and reducing over axes.

## Create tensors from shape and data

This is the tenferro equivalent of `torch.tensor(...)` or `jnp.array(...)`.

```rust
use tenferro::{CpuBackend, Engine, TracedTensor};

let mut a = TracedTensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

let mut engine = Engine::new(CpuBackend::new());
let result = a.eval(&mut engine).unwrap();

assert_eq!(result.shape(), &[2, 3]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
```

## Elementwise arithmetic

Like PyTorch and JAX, tenferro supports familiar elementwise operators on tensors with compatible shapes.

```rust
use tenferro::{CpuBackend, Engine, TracedTensor};

let a = TracedTensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]);
let b = TracedTensor::from_vec(vec![3], vec![4.0_f64, 5.0, 6.0]);
let mut sum = &a + &b;
let mut product = &a * &b;

let mut engine = Engine::new(CpuBackend::new());
let sum_result = sum.eval(&mut engine).unwrap();
let product_result = product.eval(&mut engine).unwrap();

assert_eq!(sum_result.as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);
assert_eq!(product_result.as_slice::<f64>().unwrap(), &[4.0, 10.0, 18.0]);
```

## Elementwise math functions

The common unary math ops are methods on `TracedTensor`, similar to `torch.exp(x)` or `jnp.exp(x)`.

```rust
use tenferro::{CpuBackend, Engine, TracedTensor};

let x = TracedTensor::from_vec(vec![3], vec![0.0_f64, 1.0, 2.0]);
let mut y = x.exp();

let mut engine = Engine::new(CpuBackend::new());
let result = y.eval(&mut engine).unwrap();
let data = result.as_slice::<f64>().unwrap();

assert!((data[0] - 1.0).abs() < 1e-12);
assert!((data[1] - std::f64::consts::E).abs() < 1e-12);
assert!((data[2] - 7.38905609893065).abs() < 1e-12);
```

## Reshape and transpose

These match the ideas behind `reshape` and `transpose` in PyTorch and JAX.

```rust
use tenferro::{CpuBackend, Engine, TracedTensor};

let a = TracedTensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
let mut reshaped = a.reshape(&[6]);
let mut transposed = a.transpose(&[1, 0]);

let mut engine = Engine::new(CpuBackend::new());
let reshaped_result = reshaped.eval(&mut engine).unwrap();
let transposed_result = transposed.eval(&mut engine).unwrap();

assert_eq!(reshaped_result.shape(), &[6]);
assert_eq!(reshaped_result.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
assert_eq!(transposed_result.shape(), &[3, 2]);
assert_eq!(
    transposed_result.as_slice::<f64>().unwrap(),
    &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
);
```

## Explicit broadcast

PyTorch and JAX often hide broadcasting behind arithmetic. tenferro also supports broadcasted arithmetic, but it exposes explicit broadcasting as a standalone operation when you want it.

```rust
use tenferro::{CpuBackend, Engine, TracedTensor};

let v = TracedTensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]);
let mut repeated = v.broadcast(&[3, 2], &[0]);

let mut engine = Engine::new(CpuBackend::new());
let result = repeated.eval(&mut engine).unwrap();

assert_eq!(result.shape(), &[3, 2]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
```

## Reduce over axes

Reduction looks closest to `torch.sum(x, dim=...)` or `jnp.sum(x, axis=...)`.

```rust
use tenferro::{CpuBackend, Engine, TracedTensor};

let a = TracedTensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
let mut row_sums = a.reduce_sum(&[1]);
let mut total = a.reduce_sum(&[0, 1]);

let mut engine = Engine::new(CpuBackend::new());
let rows = row_sums.eval(&mut engine).unwrap();
let total_value = total.eval(&mut engine).unwrap();

assert_eq!(rows.shape(), &[2]);
assert_eq!(rows.as_slice::<f64>().unwrap(), &[9.0, 12.0]);
assert_eq!(total_value.shape(), &[] as &[usize]);
assert_eq!(total_value.as_slice::<f64>().unwrap(), &[21.0]);
```
