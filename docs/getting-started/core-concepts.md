# Core Concepts

## Data And Execution

tenferro keeps tensor values, eager AD state, graph compilation, and backend
execution as separate pieces. That separation is intentional: it makes memory
order, compilation reuse, and backend placement explicit.

| Piece | Use |
| --- | --- |
| `TypedTensor<T>` | Concrete tensor storage with a compile-time scalar type |
| `Tensor` | Concrete tensor storage with a runtime dtype enum |
| `EagerTensor` + `EagerRuntime` | PyTorch-style scalar-loss `backward()` |
| `TracedTensor` | Lazy graph-building handle for transform AD and reuse |
| `GraphCompiler` | Lowers traced outputs into reusable `GraphProgram`s |
| `GraphExecutor<B>` | Runs compiled programs on a backend such as `CpuBackend` |

Choose the smallest surface that matches the workflow:

- Direct numeric work -> `Tensor` or `TypedTensor<T>` with a backend.
- Scalar-loss eager AD -> `EagerTensor` inside an `EagerRuntime`.
- Transform AD, graph reuse, or symbolic inputs -> `TracedTensor` compiled by
  `GraphCompiler` and run by `GraphExecutor`.

## Memory Order

tenferro stores dense tensors in column-major order. Constructors name the input
order explicitly:

```rust
use tenferro::{Tensor, TypedTensor};

let typed = TypedTensor::<f64>::from_vec_col_major(
    vec![2, 3],
    vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
);
assert_eq!(typed.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

let dynamic = Tensor::from_vec_row_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);
assert_eq!(dynamic.as_slice::<f64>().unwrap(), typed.as_slice());
```

Use `from_vec_row_major` for data written in the conventional row-by-row style.
Use `from_vec_col_major` when the buffer is already in tenferro's physical
order.

## Direct Tensor Execution

`Tensor` and `TypedTensor<T>` operations run immediately through a backend.

```rust
use tenferro::{CpuBackend, Tensor, TensorBackend};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
let b = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

let c = a.matmul(&b, &mut backend).unwrap();
assert_eq!(c.shape(), &[2, 2]);
```

## Eager AD

`EagerTensor` wraps concrete values in an `EagerRuntime` that owns gradient
state. It is the right fit for scalar-loss reverse-mode workflows.

```rust
use tenferro::{EagerRuntime, Tensor};

let ctx = EagerRuntime::new();
let x = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]));
let loss = (&x * &x).reduce_sum(&[0]).unwrap();
loss.backward().unwrap();

assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[2.0, 4.0]);
```

## Traced Graph Execution

`TracedTensor` operations are lazy. Build a graph, compile it once, then run the
compiled program through a backend executor.

```rust
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
let b = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);
let sum = &a + &b;

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&sum).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
let result = executor.run(&program).unwrap();

assert_eq!(result.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
```

For symbolic placeholders, provide input specs at compile time and concrete
bindings at run time:

```rust
use tenferro::{CpuBackend, DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};

let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
let y = &x + &x;

let mut compiler = GraphCompiler::new();
let program = compiler
    .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
    .unwrap();
let bound = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
let mut executor = GraphExecutor::new(CpuBackend::new());
let result = executor.run_with_inputs(&program, &[(&x, &bound)]).unwrap();

assert_eq!(result.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
```
