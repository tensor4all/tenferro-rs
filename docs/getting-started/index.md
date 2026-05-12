# Getting Started

tenferro supports direct CPU tensor computation, eager scalar-loss AD, traced
execution, transform AD, einsum, linear algebra, and experimental CUDA
execution for selected operations.

## Installation

The `tenferro` crate re-exports everything you need. Use a local checkout
while the crates are still evolving:

```toml
[dependencies]
tenferro = { path = "/path/to/tenferro-rs/tenferro" }
```

This uses the default `cpu-faer` backend. To use the LAPACK/BLAS CPU backend,
disable default features and enable `cpu-blas`:

```toml
[dependencies]
tenferro = { path = "/path/to/tenferro-rs/tenferro", default-features = false, features = ["cpu-blas"] }
```

Exactly one CPU backend must be enabled: `cpu-faer` or `cpu-blas`. The
`cpu-blas` backend needs a BLAS/LAPACK provider. Link one from the system
toolchain, or enable `src-openblas` to build against OpenBLAS:

```toml
[dependencies]
tenferro = { path = "/path/to/tenferro-rs/tenferro", default-features = false, features = ["src-openblas"] }
```

Or switch to crates.io once published:

```toml
[dependencies]
tenferro = "..."
```

## First CPU Program

<!-- snippet-source: tenferro/examples/cpu_quickstart.rs -->
```rust
use tenferro::{CpuBackend, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CpuBackend::new();

    let a = Tensor::from_vec(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]);
    let b = Tensor::from_vec(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0]);

    let c = a.matmul(&b, &mut backend)?;

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.as_slice::<f64>().unwrap(), &[19.0, 43.0, 22.0, 50.0]);

    Ok(())
}
```
<!-- end-snippet-source -->

## Next Steps

- [Choosing an API](../guides/choosing-an-api.md)
- [Devices and GPU](../guides/devices-and-gpu.md)
- [Core concepts](./core-concepts.md)
- [PyTorch and JAX mapping](./pytorch-jax-mapping.md)
- [Eager operations guide](../guides/eager-operations.md)
- [Tensor operations guide](../guides/tensor-operations.md)
- [Einsum guide](../guides/einsum.md)
- [Autodiff guide](../guides/autodiff.md)
- [Linear algebra guide](../guides/linear-algebra.md)
