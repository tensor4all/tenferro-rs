# tenferro-rs

tenferro-rs is a modular Rust tensor stack for scientific computing and
general-purpose tensor workflows.

It is inspired by JAX and PyTorch: eager and traced execution, automatic
differentiation, GPU backends, and extensible operation families. At the same
time, it is designed to remain usable in lightweight settings. If you only need
an ndarray-like host tensor data model, `tenferro-tensor-core` provides tensor
and view types without requiring AD, GPU runtimes, linalg backends, or provider
linking.

Higher-level crates opt into execution, AD, CUDA, BLAS/LAPACK, and operation
families explicitly. Standard operation families such as linalg, einsum, and
FFT live in their own crates.

External crates can add custom tensor operations and AD rules through the
extension mechanism. This extensibility model is strongly influenced by the
Julia ecosystem, where operation semantics and AD rules can be supplied outside
a single monolithic tensor type.

`tenferro` means tensor computation with an iron/Rust flavor: `tensor` +
`ferro`.

Optional capabilities are selected on the crate that owns the operation family.
For example, CUDA linalg with extension AD uses concrete backend features rather
than a public `gpu` feature:

```toml
[dependencies]
tenferro-ad = { path = "tenferro-ad", features = ["cuda"] }
tenferro-gpu = { path = "tenferro-gpu", features = ["cuda"] }
tenferro-linalg = { path = "tenferro-linalg", features = ["autodiff", "cuda"] }
```

Start with `Tensor` and `CpuBackend` for ndarray-like CPU work. Move to
`EagerTensor` when you want immediate execution with optional `backward()`, and
move to `TracedTensor` when you need graph reuse, transform AD, or repeated
compile/run execution. CUDA is available as an explicit backend/device choice
for supported operations.

## Direct Tensor Execution

This is the closest entry point for users coming from ndarray: create concrete
tensors, pass a backend context to operations, and get concrete tensors back.

```rust
use tenferro_runtime::{tensor, CpuBackend, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CpuBackend::new();

    let a = Tensor::from_vec_row_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let b = Tensor::from_vec_row_major(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]);

    let c = tensor::matmul(&a, &b, &mut backend)?;
    let (_shape, row_major) = c.try_into_vec_row_major::<f64>()?;
    assert_eq!(row_major, vec![19.0, 22.0, 43.0, 50.0]);

    Ok(())
}
```

## If You Know ndarray

tenferro stores dense tensors internally in column-major order, following
Fortran, Julia, and MATLAB. Many Rust arrays and Vec-backed matrix examples are
row-major. Use the row-major constructors and exports when that is the data you
have:

```rust
use tenferro_runtime::Tensor;

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

In application code, import tensor/runtime types from `tenferro-runtime`,
AD/eager types from `tenferro-ad`, and operation families from their standard
crates.

| Crate | What it provides |
| --- | --- |
| `tenferro-linalg` | Traced/eager SVD, QR, Cholesky, LU, solve, eig/eigh, triangular solve, and linalg AD rules behind the `autodiff` feature |
| `tenferro-einsum` | NumPy/JAX-style einsum with contraction planning and tensordot contraction sugar |
| `tenferro-fft` | FFT extension operations |

## EagerTensor Execution

`EagerTensor` also runs operations immediately, but through an `EagerRuntime`.
That runtime owns execution state, extension caches, and optional gradient
slots. Use untracked eager tensors for immediate forward execution, and tracked
variables when you need scalar-loss reverse-mode `backward()`.
Forward-mode AD (`jvp`) is part of traced execution, not `EagerTensor`.
The linalg eager helper below requires `tenferro-linalg` with the `autodiff`
feature enabled.

```rust
use tenferro_ad::{eager_tensor, EagerRuntime, EagerTensor, Tensor};

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

    let product = eager_tensor::matmul(&a, &b)?;
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
use tenferro_runtime::{traced_tensor, CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

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

## Documentation

**<https://tensor4all.org/tenferro-rs/>**

- [Getting Started](https://tensor4all.org/tenferro-rs/getting-started/) - run the first checked CPU example
- [Guides](https://tensor4all.org/tenferro-rs/guides/choosing-an-api.html) - tensor layers, execution models, tensor ops, einsum, linalg, autodiff, memory order, and CUDA
- [API Reference](https://tensor4all.org/tenferro-rs/api/) - rustdoc links for every crate
- [Internals](https://tensor4all.org/tenferro-rs/internals/) - architecture, specification, contributor pointers

## Development Model

tenferro-rs is pre-1.0 and intentionally evolves quickly. Public APIs, crate
boundaries, backend contracts, and feature flags may change while the design
stabilizes.

Agentic AI workflows are a first-class development path for this repository.
The project uses AI agents for implementation, documentation, review, issue
triage, migration, and verification. Human maintainers own design decisions,
review outcomes, and merge decisions.

If you build against `main`, pin commits and expect breaking changes. For
non-trivial upgrades, AI-assisted migration is recommended.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the external contribution policy.
Bug-fix pull requests are welcome. New features must start as feature request
issues before implementation PRs are opened.

Repository-local AI workflows are available for issue intake and scoped
bug-fix PR preparation across Codex CLI, Claude Code, and OpenCode; see
`CONTRIBUTING.md` for the supported entry points.
