# Getting Started

tenferro supports tensor computation without autodiff, immediate execution with
optional `backward()` on scalar losses, traced graph execution, `grad`, `vjp`,
and `jvp` on traced graphs, einsum, linear algebra, and CUDA execution through
the feature-gated CUDA backend.

## Setup

Start with the runtime crate. Use a local checkout while the crates are still
evolving:

```toml
[dependencies]
tenferro-runtime = { path = "/path/to/tenferro-rs/crates/tenferro-runtime" }
```

With default features, this compiles the `cpu-faer` provider, so
`CpuBackend::new()` uses faer. To use the LAPACK/BLAS CPU provider, enable
`cpu-blas` and link a BLAS/LAPACK provider:

```toml
[dependencies]
tenferro-runtime = { path = "/path/to/tenferro-rs/crates/tenferro-runtime", default-features = false, features = ["cpu-blas"] }
```

CPU backend features are additive. At least one of `cpu-faer` or `cpu-blas`
must be enabled, and builds may enable both. `CpuBackend::new()` selects the
compiled default provider: BLAS when `cpu-blas` is compiled, otherwise faer.
Use `CpuBackend::with_kind` when a program needs explicit provider selection
within a build that has multiple providers. The `cpu-blas` backend needs a
BLAS/LAPACK provider. Link one from the system toolchain, or enable the provider
feature on `tenferro-tensor` to build against OpenBLAS:

```toml
[dependencies]
tenferro-runtime = { path = "/path/to/tenferro-rs/crates/tenferro-runtime", default-features = false, features = ["cpu-blas"] }
tenferro-tensor = { path = "/path/to/tenferro-rs/crates/tenferro-tensor", default-features = false, features = ["src-openblas"] }
```

Add `tenferro-ad`, `tenferro-einsum`, `tenferro-linalg`, `tenferro-fft`, or
`tenferro-gpu` when a workflow needs those layers. Enable `autodiff` on
operation crates such as `tenferro-linalg` when extension AD rules are needed.
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
use tenferro_runtime::{tensor, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CpuBackend::new();

    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]);
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0]);

    let c = tensor::matmul(&a, &b, &mut backend)?;

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.as_slice::<f64>().unwrap(), &[19.0, 43.0, 22.0, 50.0]);

    Ok(())
}
```
<!-- end-snippet-source -->

## Next Steps

After the first CPU program, read [Core Concepts](./core-concepts.md) for the
main mental model or [Choosing a Tensor API](../guides/choosing-an-api.md) to
pick between `TypedTensor`, `Tensor`, `EagerTensor`, and `TracedTensor`. The
sidebar links to the full guides for memory order, CUDA, tensor operations,
autodiff, einsum, and linear algebra.
