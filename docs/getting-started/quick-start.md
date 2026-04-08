# Quick Start

## Installation

Add tenferro to your `Cargo.toml`. For local development within the workspace:

```toml
[dependencies]
tenferro = { path = "../tenferro" }
tenferro-tensor = { path = "../tenferro-tensor" }
```

The `cpu-faer` feature is enabled by default. Exactly one CPU backend
(`cpu-faer` or `cpu-blas`) must be active at build time.

## Creating tensors

tenferro stores data in **column-major** (Fortran) order: the leftmost
dimension has the smallest stride and varies fastest in memory.

Create a concrete `Tensor` by wrapping a `TypedTensor`:

```rust,ignore
use tenferro::{Tensor, TypedTensor};

// A 2x3 matrix in column-major layout:
//   [[1, 3, 5],
//    [2, 4, 6]]
// Flat data: column 0 = [1, 2], column 1 = [3, 4], column 2 = [5, 6]
let a = Tensor::F64(TypedTensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));

// A scalar (rank-0 tensor):
let s = Tensor::F64(TypedTensor::from_vec(vec![], vec![42.0]));

// A vector:
let v = Tensor::F64(TypedTensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]));
```

## Creating traced tensors

`TracedTensor` is the lazy, graph-aware wrapper. All operations on traced
tensors build a computation graph; nothing executes until you call `.eval()`.

```rust,ignore
use tenferro::{Tensor, TypedTensor, TracedTensor};

let tensor = Tensor::F64(TypedTensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
let traced = TracedTensor::from_tensor(tensor);
```

## Basic operations

Arithmetic operators are overloaded for `&TracedTensor`:

```rust,ignore
use tenferro::{Tensor, TypedTensor, TracedTensor};

let a = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]))
);
let b = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]))
);

// Elementwise add, mul, div, sub, neg
let sum = &a + &b;
let product = &a * &b;
let quotient = &a / &b;
let negated = -&a;

// Method-style equivalents
let sum2 = a.add(&b);
let product2 = a.mul(&b);

// Unary math: exp, log, sin, cos, pow
let e = a.exp();
let s = a.sin();
let p = a.pow(&b);

// Reduction: sum over specified axes
let row_sums = a.reduce_sum(&[1]);   // sum over columns -> shape [2]
let total = a.reduce_sum(&[0, 1]);   // sum all -> scalar
```

## Engine and evaluation

`Engine` holds the backend and caches. Create one with `CpuBackend::new()`,
then call `.eval(&mut engine)` on any traced tensor to execute the graph:

```rust,ignore
use tenferro::{CpuBackend, Engine, Tensor, TypedTensor, TracedTensor};

let a = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]))
);
let b = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]))
);
let mut result = &a + &b;

let mut engine = Engine::new(CpuBackend::new());
let output = result.eval(&mut engine).unwrap();

// output is &Tensor -- pattern match to read data
match output {
    Tensor::F64(inner) => {
        let data: &[f64] = inner.host_data();
        assert_eq!(data, &[6.0, 8.0, 10.0, 12.0]);
    }
    _ => panic!("unexpected dtype"),
}
```

## Evaluating multiple outputs

Use `eval_all` when you need several outputs from the same graph.
This compiles the graph once and evaluates all outputs in a single pass:

```rust,ignore
use tenferro::engine::Engine;
use tenferro::traced::{eval_all, TracedTensor};
use tenferro::{CpuBackend, Tensor, TypedTensor, svd};

let a = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 2.0]))
);
let (mut u, mut s, mut vt) = svd(&a);

let mut engine = Engine::new(CpuBackend::new());
let results = eval_all(&mut engine, &mut [&mut u, &mut s, &mut vt]).unwrap();
// results[0] = U, results[1] = singular values, results[2] = V^T
```

## Matrix multiplication

For rank-2 tensors, use the `matmul` convenience function:

```rust,ignore
use tenferro::{matmul, CpuBackend, Engine, Tensor, TypedTensor, TracedTensor};

let a = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
);
let b = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
);
let mut c = matmul(&a, &b);

let mut engine = Engine::new(CpuBackend::new());
let result = c.eval(&mut engine).unwrap();
// result shape: [2, 2]
// C = A @ B = [[22, 49], [28, 64]] col-major: [22, 28, 49, 64]
```
