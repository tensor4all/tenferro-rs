# Tensor Operations

This guide covers everyday tensor operations: elementwise math, shape changes,
broadcasting, reductions, and concrete backend execution. These operations are
available through different tensor layers depending on whether you need no-AD
computation, eager scalar-loss AD, or traced graph execution.

## Layer Coverage

| Layer | Operation style |
| --- | --- |
| `TypedTensor<T>` | Typed storage and selected typed operation wrappers; convert to `Tensor` for the broad dynamic operation surface |
| `Tensor` | Concrete no-AD operations through an explicit backend |
| `EagerTensor` | Immediate operations that also record state for `backward()` |
| `TracedTensor` | Lazy operations that build a graph for compile/run reuse |
| CUDA | Supported operation/dtype combinations run on CUDA tensors with explicit upload/download |

## Concrete Tensor Example

Use `Tensor` with a backend when you want direct no-AD computation.

```rust
use tenferro::{CpuBackend, Tensor};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
let b = Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]);

let sum = a.add(&b, &mut backend).unwrap();
let product = a.mul(&b, &mut backend).unwrap();

assert_eq!(sum.as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);
assert_eq!(product.as_slice::<f64>().unwrap(), &[4.0, 10.0, 18.0]);
```

## Eager AD Example

Use `EagerTensor` when the same immediate computation should accumulate
gradients for a scalar loss.

```rust
use tenferro::{EagerRuntime, Tensor};

let ctx = EagerRuntime::new();
let x = ctx.variable_from(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]));
let y = (&x * &x).reduce_sum(&[0]).unwrap();

y.backward().unwrap();
assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);
```

## Traced Tensor Example

Use `TracedTensor` when operations should build a graph first and execute later.

```rust
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
let b = TracedTensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]);
let sum = &a + &b;
let product = &a * &b;

let mut compiler = GraphCompiler::new();
let program = compiler.compile_many(&[&sum, &product]).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
let outputs = executor.run_many(&program).unwrap();

assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);
assert_eq!(outputs[1].as_slice::<f64>().unwrap(), &[4.0, 10.0, 18.0]);
```

## Elementwise Math Functions

```rust
use tenferro::{EagerRuntime, Tensor};

let ctx = EagerRuntime::new();
let x = ctx.variable_from(Tensor::from_vec_col_major(vec![3], vec![0.0_f64, 1.0, 2.0]));
let y = x.exp().unwrap();

let data = y.data().as_slice::<f64>().unwrap();

assert!((data[0] - 1.0).abs() < 1e-12);
assert!((data[1] - std::f64::consts::E).abs() < 1e-12);
assert!((data[2] - 7.38905609893065).abs() < 1e-12);
```

## Reshape And Transpose

```rust
use tenferro::{CpuBackend, Tensor};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);
let reshaped = a.reshape(&[6], &mut backend).unwrap();
let transposed = a.transpose(&[1, 0], &mut backend).unwrap();

assert_eq!(reshaped.shape(), &[6]);
assert_eq!(reshaped.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
assert_eq!(transposed.shape(), &[3, 2]);
assert_eq!(transposed.as_slice::<f64>().unwrap(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
```

## Explicit Broadcast

```rust
use tenferro::{EagerRuntime, Tensor};

let ctx = EagerRuntime::new();
let v = ctx.variable_from(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]));
let repeated = v.broadcast_in_dim(&[3, 2], &[0]).unwrap();

assert_eq!(repeated.data().shape(), &[3, 2]);
assert_eq!(repeated.data().as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
```

## Reduce Over Axes

```rust
use tenferro::{CpuBackend, Tensor};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);
let row_sums = a.reduce_sum(&[1], &mut backend).unwrap();
let total = a.reduce_sum(&[0, 1], &mut backend).unwrap();

assert_eq!(row_sums.shape(), &[2]);
assert_eq!(row_sums.as_slice::<f64>().unwrap(), &[9.0, 12.0]);
assert_eq!(total.shape(), &[] as &[usize]);
assert_eq!(total.as_slice::<f64>().unwrap(), &[21.0]);
```
