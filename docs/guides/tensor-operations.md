# Tensor Operations

This guide covers everyday traced tensor operations: creating graph inputs,
applying elementwise math, changing shapes, broadcasting, and reducing over
axes. Traced operations build a graph; `GraphCompiler` lowers the graph and
`GraphExecutor` runs it on a backend.

## Create tensors from shape and data

```rust
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&a).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
let result = executor.run(&program).unwrap();

assert_eq!(result.shape(), &[2, 3]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
```

## Elementwise arithmetic

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

## Elementwise math functions

```rust
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let x = TracedTensor::from_vec_col_major(vec![3], vec![0.0_f64, 1.0, 2.0]);
let y = x.exp();

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&y).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
let result = executor.run(&program).unwrap();
let data = result.as_slice::<f64>().unwrap();

assert!((data[0] - 1.0).abs() < 1e-12);
assert!((data[1] - std::f64::consts::E).abs() < 1e-12);
assert!((data[2] - 7.38905609893065).abs() < 1e-12);
```

## Reshape and transpose

```rust
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);
let reshaped = a.reshape(&[6]);
let transposed = a.transpose(&[1, 0]);

let mut compiler = GraphCompiler::new();
let program = compiler.compile_many(&[&reshaped, &transposed]).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
let outputs = executor.run_many(&program).unwrap();

assert_eq!(outputs[0].shape(), &[6]);
assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
assert_eq!(outputs[1].shape(), &[3, 2]);
assert_eq!(outputs[1].as_slice::<f64>().unwrap(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
```

## Explicit broadcast

```rust
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let v = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
let repeated = v.broadcast(&[3, 2], &[0]);

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&repeated).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
let result = executor.run(&program).unwrap();

assert_eq!(result.shape(), &[3, 2]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
```

## Reduce over axes

```rust
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);
let row_sums = a.reduce_sum(&[1]);
let total = a.reduce_sum(&[0, 1]);

let mut compiler = GraphCompiler::new();
let program = compiler.compile_many(&[&row_sums, &total]).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
let outputs = executor.run_many(&program).unwrap();

assert_eq!(outputs[0].shape(), &[2]);
assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[9.0, 12.0]);
assert_eq!(outputs[1].shape(), &[] as &[usize]);
assert_eq!(outputs[1].as_slice::<f64>().unwrap(), &[21.0]);
```
