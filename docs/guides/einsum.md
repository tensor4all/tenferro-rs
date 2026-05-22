# Einsum

If you already use `torch.einsum(...)` or `jnp.einsum(...)`, tenferro keeps
the same subscript language. The traced API builds an einsum graph with a
`GraphCompiler`; a `GraphExecutor` runs the compiled program.

- PyTorch: `torch.einsum("ij,jk->ik", a, b)`
- JAX: `jnp.einsum("ij,jk->ik", a, b)`
- tenferro: `tenferro::traced_tensor::einsum(&mut compiler, &[&a, &b], "ij,jk->ik")`

## Matrix multiply

```rust
use tenferro::traced_tensor::einsum;
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);
let b = TracedTensor::from_vec_col_major(
    vec![3, 2],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);

let mut compiler = GraphCompiler::new();
let c = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();
let program = compiler.compile(&c).unwrap();

let mut executor = GraphExecutor::new(CpuBackend::new());
let result = executor.run(&program).unwrap();

assert_eq!(result.shape(), &[2, 2]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
```

## Trace and diagonal

Repeated labels select or reduce diagonals, matching the standard NumPy,
PyTorch, and JAX idioms.

```rust
use tenferro::traced_tensor::einsum;
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let matrix = TracedTensor::from_vec_col_major(
    vec![3, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
);

let mut compiler = GraphCompiler::new();
let trace = einsum(&mut compiler, &[&matrix], "ii->").unwrap();
let diagonal = einsum(&mut compiler, &[&matrix], "ii->i").unwrap();
let program = compiler.compile_many(&[&trace, &diagonal]).unwrap();

let mut executor = GraphExecutor::new(CpuBackend::new());
let outputs = executor.run_many(&program).unwrap();

assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[15.0]);
assert_eq!(outputs[1].as_slice::<f64>().unwrap(), &[1.0, 5.0, 9.0]);
```

## Outer product and diagonal embedding

```rust
use tenferro::traced_tensor::einsum;
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let u = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
let v = TracedTensor::from_vec_col_major(vec![3], vec![3.0_f64, 4.0, 5.0]);

let mut compiler = GraphCompiler::new();
let outer = einsum(&mut compiler, &[&u, &v], "i,j->ij").unwrap();
let diag = einsum(&mut compiler, &[&v], "i->ii").unwrap();
let program = compiler.compile_many(&[&outer, &diag]).unwrap();

let mut executor = GraphExecutor::new(CpuBackend::new());
let outputs = executor.run_many(&program).unwrap();

assert_eq!(outputs[0].shape(), &[2, 3]);
assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[3.0, 6.0, 4.0, 8.0, 5.0, 10.0]);
assert_eq!(outputs[1].shape(), &[3, 3]);
```

## N-ary contraction

tenferro accepts more than two inputs and chooses a contraction order
automatically. Concrete-shape einsums are planned by `GraphCompiler`; symbolic
einsums are planned at runtime and cached by `GraphExecutor` for the observed
input shapes.

```rust
use tenferro::traced_tensor::einsum;
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
let b = TracedTensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]);
let c = TracedTensor::from_vec_col_major(vec![2, 2], vec![9.0_f64, 10.0, 11.0, 12.0]);

let mut compiler = GraphCompiler::new();
let out = einsum(&mut compiler, &[&a, &b, &c], "ij,jk,kl->il").unwrap();
let program = compiler.compile(&out).unwrap();

let mut executor = GraphExecutor::new(CpuBackend::new());
let result = executor.run(&program).unwrap();

assert_eq!(result.shape(), &[2, 2]);
```

## Concrete vs symbolic shapes

| Mode | Constructor | When to use |
| --- | --- | --- |
| Concrete | `from_vec_col_major`, `from_vec_row_major`, `from_tensor_concrete_shape`, `input_concrete_shape(dtype, shape)` | Shape fixed at graph-build time |
| Symbolic | `from_tensor_symbolic_shape`, `input_symbolic_shape(dtype, rank)` | Shape only known when a program runs |

```rust
use tenferro::traced_tensor::einsum;
use tenferro::{CpuBackend, DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};

let a = TracedTensor::input_symbolic_shape(DType::F64, 2);
let b = TracedTensor::from_vec_col_major(
    vec![3, 2],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);

let mut compiler = GraphCompiler::new();
let c = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();
let program = compiler
    .compile_with_input_specs(&c, &[(&a, DType::F64, &[2, 3])])
    .unwrap();
let a_concrete = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);
let mut executor = GraphExecutor::new(CpuBackend::new());
let result = executor.run_with_inputs(&program, &[(&a, &a_concrete)]).unwrap();

assert_eq!(result.shape(), &[2, 2]);
```

## Batched matrix multiply

PyTorch and JAX users often put the batch axis first. In tenferro, trailing
batch axes line up naturally with column-major storage, so this example keeps
the batch dimension on the right.

```rust
use tenferro::traced_tensor::einsum;
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(
    vec![2, 2, 2],
    vec![1.0_f64, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0],
);
let b = TracedTensor::from_vec_col_major(
    vec![2, 2, 2],
    vec![5.0_f64, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0],
);

let mut compiler = GraphCompiler::new();
let c = einsum(&mut compiler, &[&a, &b], "ijk,jlk->ilk").unwrap();
let program = compiler.compile(&c).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
let result = executor.run(&program).unwrap();

assert_eq!(result.shape(), &[2, 2, 2]);
```
