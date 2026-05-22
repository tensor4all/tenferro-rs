# Einsum

If you already use `torch.einsum(...)` or `jnp.einsum(...)`, tenferro keeps
the same subscript language. Einsum is not a traced-only feature: tenferro has
typed, concrete, eager, and traced routes.

- PyTorch: `torch.einsum("ij,jk->ik", a, b)`
- JAX: `jnp.einsum("ij,jk->ik", a, b)`
- tenferro typed: `tenferro::typed_tensor::einsum(&mut backend, &[&a, &b], "ij,jk->ik")`
- tenferro concrete: `tenferro::tensor::einsum(&mut backend, &[&a, &b], "ij,jk->ik")`
- tenferro eager: `tenferro::eager_tensor::einsum(&[&a, &b], "ij,jk->ik")`
- tenferro traced: `tenferro::traced_tensor::einsum(&mut compiler, &[&a, &b], "ij,jk->ik")`

## Layer Coverage

| Layer | Einsum entry point |
| --- | --- |
| `TypedTensor<T>` | `tenferro::typed_tensor::einsum` |
| `Tensor` | `tenferro::tensor::einsum` |
| `EagerTensor` | `tenferro::eager_tensor::einsum` |
| `TracedTensor` | `tenferro::traced_tensor::einsum` |

CUDA is a backend/device choice for supported `Tensor`, `EagerTensor`, and
`TracedTensor` execution paths. It is not a separate einsum entry point.

## Concrete Matrix Multiply

Use the concrete route for no-AD backend execution.

```rust
use tenferro::tensor::einsum;
use tenferro::{CpuBackend, Tensor};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);
let b = Tensor::from_vec_col_major(
    vec![3, 2],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);

let c = einsum(&mut backend, &[&a, &b], "ij,jk->ik").unwrap();

assert_eq!(c.shape(), &[2, 2]);
assert_eq!(c.as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
```

## Typed Matrix Multiply

Use the typed route when all inputs share one compile-time scalar type.

```rust
use tenferro::typed_tensor::einsum;
use tenferro::{CpuBackend, TypedTensor};

let mut backend = CpuBackend::new();
let a = TypedTensor::<f64>::from_vec_col_major(
    vec![2, 2],
    vec![1.0, 2.0, 3.0, 4.0],
);
let b = TypedTensor::<f64>::from_vec_col_major(
    vec![2, 2],
    vec![5.0, 6.0, 7.0, 8.0],
);

let c = einsum(&mut backend, &[&a, &b], "ij,jk->ik").unwrap();

assert_eq!(c.shape.as_slice(), &[2, 2]);
assert_eq!(c.host_data(), &[23.0, 34.0, 31.0, 46.0]);
```

## Traced Matrix Multiply

Use the traced route when einsum should be part of a graph compiled by
`GraphCompiler` and executed by `GraphExecutor`.

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
use tenferro::tensor::einsum;
use tenferro::{CpuBackend, Tensor};

let mut backend = CpuBackend::new();
let matrix = Tensor::from_vec_col_major(
    vec![3, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
);

let trace = einsum(&mut backend, &[&matrix], "ii->").unwrap();
let diagonal = einsum(&mut backend, &[&matrix], "ii->i").unwrap();

assert_eq!(trace.as_slice::<f64>().unwrap(), &[15.0]);
assert_eq!(diagonal.as_slice::<f64>().unwrap(), &[1.0, 5.0, 9.0]);
```

## Outer product and diagonal embedding

```rust
use tenferro::eager_tensor::einsum;
use tenferro::{EagerRuntime, Tensor};

let ctx = EagerRuntime::new();
let u = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]));
let v = ctx.variable_from(Tensor::from_vec_col_major(vec![3], vec![3.0_f64, 4.0, 5.0]));

let outer = einsum(&[&u, &v], "i,j->ij").unwrap();
let diag = einsum(&[&v], "i->ii").unwrap();

assert_eq!(outer.data().shape(), &[2, 3]);
assert_eq!(outer.data().as_slice::<f64>().unwrap(), &[3.0, 6.0, 4.0, 8.0, 5.0, 10.0]);
assert_eq!(diag.data().shape(), &[3, 3]);
```

## N-ary contraction

tenferro accepts more than two inputs and chooses a contraction order
automatically. Direct, typed, and eager routes execute immediately through
their backend or runtime. In traced mode, `GraphCompiler` plans
concrete-shape einsums and `GraphExecutor` caches runtime plans for symbolic
input shapes.

```rust
use tenferro::typed_tensor::einsum;
use tenferro::{CpuBackend, TypedTensor};

let mut backend = CpuBackend::new();
let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
let c = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![9.0, 10.0, 11.0, 12.0]);

let out = einsum(&mut backend, &[&a, &b, &c], "ij,jk,kl->il").unwrap();

assert_eq!(out.shape.as_slice(), &[2, 2]);
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
use tenferro::tensor::einsum;
use tenferro::{CpuBackend, Tensor};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(
    vec![2, 2, 2],
    vec![1.0_f64, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0],
);
let b = Tensor::from_vec_col_major(
    vec![2, 2, 2],
    vec![5.0_f64, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0],
);

let result = einsum(&mut backend, &[&a, &b], "ijk,jlk->ilk").unwrap();

assert_eq!(result.shape(), &[2, 2, 2]);
```
