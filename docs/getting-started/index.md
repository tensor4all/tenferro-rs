# Getting Started

tenferro supports tensor computation without autodiff, immediate execution with
optional `backward()` on scalar losses, traced graph execution, `grad`, `vjp`,
and `jvp` on traced graphs, einsum, linear algebra, and CUDA execution through
the feature-gated CUDA backend.

## Mental Model

tenferro has three independent choices. Pick the smallest tensor layer that
matches the program, decide whether work should run immediately or through a
compiled traced program, then choose a CPU or CUDA backend explicitly.

| Choice | Question | Common starting point |
| --- | --- | --- |
| Tensor layer | What kind of value do I pass around? | `TypedTensor<T>` for typed CPU values, `Tensor` for runtime dtype |
| Execution model | When does computation run? | Direct/eager for ordinary code, traced for `grad`, `vjp`, `jvp`, or reuse |
| Backend/device | Where does computation run? | `CpuBackend`; upload/download explicitly for CUDA |

CUDA, eager execution, and traced graphs are not competing APIs. They compose
when a workflow needs them, but the first CPU program below only needs the
runtime crate and CPU backend.

## Setup

Start with the runtime crate and CPU backend crate. Use a local checkout while
the crates are still evolving:

```toml
[dependencies]
tenferro-runtime = { path = "/path/to/tenferro-rs/crates/tenferro-runtime" }
tenferro-cpu = { path = "/path/to/tenferro-rs/crates/tenferro-cpu" }
```

If you create a scratch binary crate inside the `tenferro-rs` checkout, add an
empty `[workspace]` table to that scratch crate's `Cargo.toml`. Otherwise Cargo
will treat it as an unlisted member of the parent workspace and report a
workspace-membership error.

For a scratch crate directly under the checkout root, relative paths are less
error-prone:

```toml
[workspace]

[dependencies]
tenferro-runtime = { path = "../crates/tenferro-runtime" }
tenferro-cpu = { path = "../crates/tenferro-cpu" }
```

The first build still needs network access unless dependencies are already
vendored or cached. The workspace pins git dependencies and uses crates.io
packages even when the tenferro crates themselves are local path dependencies.
The default `cpu-faer` provider may take several minutes to compile the first
time on a fresh machine; later incremental builds are much faster.

With default features, this compiles the `cpu-faer` provider, so
`CpuBackend::new()` uses faer. To use the LAPACK/BLAS CPU provider, enable
`cpu-blas` and link a BLAS/LAPACK provider from the build environment:

```toml
[dependencies]
tenferro-runtime = { path = "/path/to/tenferro-rs/crates/tenferro-runtime", default-features = false, features = ["cpu-blas"] }
tenferro-cpu = { path = "/path/to/tenferro-rs/crates/tenferro-cpu", default-features = false, features = ["cpu-blas"] }
```

CPU backend features are additive. At least one of `cpu-faer` or `cpu-blas`
must be enabled, and builds may enable both. `CpuBackend::new()` selects the
compiled default provider: BLAS when `cpu-blas` is compiled, otherwise faer.
Use `CpuBackend::with_kind` when a program needs explicit provider selection
within a build that has multiple providers. `cpu-tblis` may be enabled as an
optional `dot_general` contraction provider, but it does not replace `cpu-faer`
or `cpu-blas`; one of those providers is still required for fallback and linalg
coverage. The `cpu-blas` backend needs a BLAS/LAPACK provider. Link one from
the system toolchain with `cpu-blas`, or enable exactly one explicit provider
feature to select a source provider for BLAS, LAPACK, and strided einsum:

```toml
[dependencies]
tenferro-runtime = { path = "/path/to/tenferro-rs/crates/tenferro-runtime", default-features = false, features = ["blas-openblas"] }
tenferro-cpu = { path = "/path/to/tenferro-rs/crates/tenferro-cpu", default-features = false, features = ["blas-openblas"] }
```

The explicit provider features are `blas-openblas`, `blas-accelerate`, and
`blas-mkl`. Cargo features are additive, so tenferro rejects builds that enable
more than one explicit BLAS provider.

Provider build scripts may need environment variables when using system
installations. OpenBLAS setups commonly use `OPENBLAS_LIB_DIR`; MKL setups
commonly use `MKLROOT` or `MKL_LIB_DIR`; Accelerate is the macOS system
framework provider.

Add `tenferro-ad`, `tenferro-einsum`, `tenferro-linalg`, `tenferro-fft`, or
`tenferro-gpu` when a workflow needs those layers. Enable `autodiff` on
operation crates such as `tenferro-linalg` when extension AD rules are needed.
For CPU-only eager or traced AD, add:

```toml
[dependencies]
tenferro-ad = { path = "/path/to/tenferro-rs/crates/tenferro-ad" }
```

Enable concrete backend features such as `cuda` on each crate that needs GPU
support:

```toml
[dependencies]
tenferro-ad = { path = "/path/to/tenferro-rs/crates/tenferro-ad", features = ["cuda"] }
tenferro-gpu = { path = "/path/to/tenferro-rs/crates/tenferro-gpu", features = ["cuda"] }
tenferro-linalg = { path = "/path/to/tenferro-rs/crates/tenferro-linalg", features = ["autodiff", "cuda"] }
```

Switch to crates.io once published:

```toml
[dependencies]
tenferro-runtime = "..."
```

## First CPU Program

<!-- snippet-source: crates/tenferro-runtime/examples/cpu_quickstart.rs -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorOpsExt};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CpuBackend::new();

    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?;
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0])?;

    let c = a.matmul(&b, &mut backend)?;

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.as_slice::<f64>().unwrap(), &[19.0, 43.0, 22.0, 50.0]);

    Ok(())
}
```
<!-- end-snippet-source -->

Expected output: the program exits silently because the shape and value
assertions pass. The `from_vec_col_major` buffer is column-major: the leftmost
axis varies fastest in memory.

## Next Steps

After the first CPU program, read [Core Concepts](./core-concepts.md) for the
main mental model or [Choosing a Tensor API](../guides/choosing-an-api.md) to
pick between `TypedTensor`, `Tensor`, `EagerTensor`, and `TracedTensor`. The
sidebar links to the full guides for memory order, CUDA, tensor operations,
autodiff, einsum, and linear algebra.
