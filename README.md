# tenferro-rs

Dense tensor computation in Rust.

Start with `Tensor` and `CpuBackend` for ndarray-like CPU work. Move to
`EagerTensor` when you want immediate execution with optional `backward()`, and
move to `TracedTensor` when you need graph reuse, transform AD, or repeated
compile/run execution. CUDA is available as an explicit backend/device choice
for supported operations.

## Direct Tensor Execution

This is the closest entry point for users coming from ndarray: create concrete
tensors, pass a backend context to operations, and get concrete tensors back.

```rust
use tenferro::{CpuBackend, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CpuBackend::new();

    let a = Tensor::from_vec_row_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let b = Tensor::from_vec_row_major(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]);

    let c = a.matmul(&b, &mut backend)?;
    let (_shape, row_major) = c.try_into_vec_row_major::<f64>()?;
    assert_eq!(row_major, vec![19.0, 22.0, 43.0, 50.0]);

    let diag = Tensor::from_vec_row_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]);
    let (_u, singular_values, _vt) = diag.svd(&mut backend)?;
    let mut values = singular_values.try_into_vec_row_major::<f64>()?.1;
    values.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
    assert_eq!(values, vec![1.0, 2.0]);

    Ok(())
}
```

## If You Know ndarray

tenferro stores dense tensors internally in column-major order, following
Fortran, Julia, and MATLAB. Many Rust arrays and Vec-backed matrix examples are
row-major. Use the row-major constructors and exports when that is the data you
have:

```rust
use tenferro::Tensor;

let t = Tensor::from_vec_row_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);

assert_eq!(
    t.clone().try_into_vec_row_major::<f64>().unwrap().1,
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
);
assert_eq!(
    t.as_slice::<f64>().unwrap(),
    &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
);
```

Use `from_vec_col_major` and `try_into_vec_col_major` when your buffer is
already in tenferro's native layout.

## Standard Operation Crates

Linear algebra, einsum, and FFT are standard tenferro operation crates. They are
separate dependencies for modularity, but they are maintained as part of the
normal tenferro stack.

| Crate | What it provides |
| --- | --- |
| `tenferro-linalg` | SVD, QR, Cholesky, LU, solve, eig/eigh, triangular solve, and related linalg APIs |
| `tenferro-einsum` | NumPy/JAX-style einsum with contraction planning |
| `tenferro-fft` | FFT extension operations |

## EagerTensor Execution

`EagerTensor` also runs operations immediately, but through an `EagerRuntime`.
That runtime owns execution state, extension caches, and optional gradient
slots. Use untracked eager tensors for immediate forward execution, and tracked
variables when you need scalar-loss reverse-mode `backward()`.
Forward-mode AD (`jvp`) is part of traced execution, not `EagerTensor`.

```rust
use tenferro::{EagerRuntime, EagerTensor, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = EagerRuntime::new();

    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_row_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]),
        ctx.clone(),
    );
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_row_major(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]),
        ctx.clone(),
    );

    let product = a.matmul(&b)?;
    assert_eq!(
        product.data().clone().try_into_vec_row_major::<f64>()?.1,
        vec![19.0, 22.0, 43.0, 50.0]
    );

    let diag = EagerTensor::from_tensor_in(
        Tensor::from_vec_row_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]),
        ctx.clone(),
    );
    let (_u, singular_values, _vt) = tenferro_linalg::eager_tensor::svd(&diag)?;
    let mut values = singular_values
        .data()
        .clone()
        .try_into_vec_row_major::<f64>()?
        .1;
    values.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
    assert_eq!(values, vec![1.0, 2.0]);

    let einsum_product = tenferro_einsum::eager_tensor::einsum(&[&a, &b], "ij,jk->ik")?;
    assert_eq!(
        einsum_product
            .data()
            .clone()
            .try_into_vec_row_major::<f64>()?
            .1,
        vec![19.0, 22.0, 43.0, 50.0]
    );

    let x = ctx.variable_from(Tensor::from_vec_row_major(vec![2], vec![1.0_f64, 2.0]));
    let loss = (&x * &x).reduce_sum(&[0])?;
    loss.backward()?;
    assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[2.0, 4.0]);

    Ok(())
}
```

## Traced Execution

Use traced execution when you want to build a graph, compile it once, and run it
through a `GraphExecutor`. This is the entry point for graph reuse and
transform-style AD (`grad`, `vjp`, `jvp`, and HVP by composition).

```rust
use tenferro::{traced_tensor, CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = TracedTensor::from_vec_row_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let b = TracedTensor::from_vec_row_major(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]);
    let c = traced_tensor::matmul(&a, &b);

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&c)?;

    let mut executor = GraphExecutor::new(CpuBackend::new());
    let out = executor.run(&program)?;

    assert_eq!(
        out.try_into_vec_row_major::<f64>()?.1,
        vec![19.0, 22.0, 43.0, 50.0]
    );

    Ok(())
}
```

For traced einsum, linalg, and custom operations, add the corresponding
standard operation crate and register its runtime on the `GraphExecutor`.

## Choosing An API

| Goal | Start with |
| --- | --- |
| ndarray-like CPU tensor computation | `Tensor` or `TypedTensor<T>` with `CpuBackend` |
| Immediate execution plus scalar-loss reverse-mode `backward()` | `EagerTensor` with `EagerRuntime` |
| Graph reuse, forward-mode / transform AD (`jvp`, `vjp`, `grad`), symbolic inputs, or repeated compile/run | `TracedTensor`, `GraphCompiler`, and `GraphExecutor` |
| CUDA execution | The same tensor layer plus explicit upload/download on the CUDA backend |
| Standard linalg, einsum, or FFT | `tenferro-linalg`, `tenferro-einsum`, or `tenferro-fft` |

## Documentation

**<https://tensor4all.org/tenferro-rs/>**

- [Getting Started](https://tensor4all.org/tenferro-rs/getting-started/) - run the first checked CPU example
- [Guides](https://tensor4all.org/tenferro-rs/guides/choosing-an-api.html) - tensor layers, execution models, tensor ops, einsum, linalg, autodiff, memory order, and CUDA
- [API Reference](https://tensor4all.org/tenferro-rs/api/) - rustdoc links for every crate
- [Internals](https://tensor4all.org/tenferro-rs/internals/) - architecture, specification, contributor pointers

## Crate Layout

Use the `tenferro` crate for application code. It is the public facade for
concrete tensor values, eager execution, traced execution, autodiff, backend
selection, and selected runtime reexports.

Operation families such as einsum, linear algebra, and FFT are standard
operation crates rather than modules inside `tenferro`. Applications import the
operation crates they use explicitly.

Core runtime infrastructure is split below the facade:

- `tenferro-tensor` owns dense tensors and backend kernels.
- `tenferro-ops` owns the graph operation vocabulary and core AD rules.
- `tenferro-runtime` owns extension runtime registration and extension cache
  storage.
- `tenferro` owns the public tensor facade, graph construction, eager AD, and
  selected runtime reexports.

Internal workspace crates are documented through the API reference for
contributors who need implementation details.

## Development Model

tenferro-rs uses agentic AI development: a small human-maintainer team develops
the project with AI agents for implementation, documentation, review, issue
triage, and verification. Human maintainers own design decisions, review
outcomes, and merge decisions.

Bug reports should include a minimal reproducer, expected behavior, actual
behavior, and the backend/device involved. Feature requests can stay informal:
describe what you are trying to do, what is hard today, and any examples or
related APIs that help explain the desired workflow.
