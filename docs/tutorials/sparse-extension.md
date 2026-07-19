# Sparse Tensor Extension

Use the sparse extension tutorial when an operation needs a domain-specific
tensor representation but should still compose with traced execution and AD.
The nested `ext/sparse` crate implements COO sparse matrix multiplication as an
extension crate.

The tutorial intentionally keeps the sparse structure fixed. Coordinates and
logical shapes are metadata; the nonzero values are dense tenferro tensors and
are the differentiable graph inputs.

## COO Wrapper

The public wrapper stores sparse data using ordinary tenferro tensors:

```rust
use tenferro_ext_sparse::SparseCooTensor;
use tenferro_tensor::Tensor;

let coords = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![0_i64, 0, 0, 1, 1, 0],
)?;
let values = Tensor::from_vec_col_major(vec![3], vec![2.0_f64, 1.0, 3.0])?;
let sparse = SparseCooTensor::from_parts(vec![2, 2], coords, values)?;
```

The coordinate tensor has shape `[2, nnz]` in column-major order. Each column is
`[row, col]`. Values have shape `[nnz]`.

## Eager Contract

`sparse_matmul_eager` contracts two COO matrices and returns a COO output with
a deterministic output coordinate order:

```rust
use tenferro_ext_sparse::{sparse_matmul_eager, SparseCooTensor};
use tenferro_tensor::Tensor;

let left = SparseCooTensor::from_parts(
    vec![2, 2],
    Tensor::from_vec_col_major(vec![2, 3], vec![0_i64, 0, 0, 1, 1, 0])?,
    Tensor::from_vec_col_major(vec![3], vec![2.0_f64, 1.0, 3.0])?,
)?;
let right = SparseCooTensor::from_parts(
    vec![2, 2],
    Tensor::from_vec_col_major(vec![2, 3], vec![0_i64, 0, 1, 0, 0, 1])?,
    Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 70.0, 20.0])?,
)?;
let product = sparse_matmul_eager(&left, &right)?;
```

The implementation builds a sparse contraction plan from the two coordinate
sets. That plan is the extension payload for traced execution; the values stay
as graph inputs so AD can see them.

## Traced Contract

For traced execution, use `SparseCooTracedTensor` and register the sparse
runtime before executing the compiled graph:

```rust
use tenferro_cpu::CpuBackend;
use tenferro_ext_sparse::{register_runtime, sparse_matmul, SparseCooTracedTensor};
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
use tenferro_tensor::Tensor;

let coords = Tensor::from_vec_col_major(vec![2, 1], vec![0_i64, 0])?;
let left = SparseCooTracedTensor::from_parts(
    vec![1, 1],
    coords.clone(),
    TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64])?,
)?;
let right = SparseCooTracedTensor::from_parts(
    vec![1, 1],
    coords,
    TracedTensor::from_vec_col_major(vec![1], vec![3.0_f64])?,
)?;
let out = sparse_matmul(&left, &right)?;

let mut compiler = GraphCompiler::new();
let program = compiler.compile(out.values())?;
let mut executor = GraphExecutor::new(CpuBackend::new());
executor.register_extension(register_runtime)?;
let values = executor.run(&program)?;
```

## AD Rules

With the `autodiff` feature enabled, `sparse_ad_rules()` registers JVP and VJP
rules for sparse-sparse matmul. The derivatives are with respect to the dense
nonzero value tensors, not the fixed coordinates:

```rust
use tenferro_ad::AdContext;
use tenferro_ext_sparse::sparse_ad_rules;

let ad = AdContext::builder()
    .with_extension_rules(sparse_ad_rules()?)
    .build()?;
let loss = out.values().reduce_sum(Some(&[0]))?;
let grad_left_values = ad.grad(&loss, left.values())?;
```

The tests compare eager sparse results, traced runtime execution, and gradients
against dense reference calculations.

## Run It

From the repository root:

```bash
cargo test --manifest-path ext/sparse/Cargo.toml --release --features autodiff
```

CI runs that command because this tutorial is short enough to execute on every
pull request.
