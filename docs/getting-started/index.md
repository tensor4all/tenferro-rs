# Getting Started

tenferro supports no-AD tensor computation, eager forward execution with
optional scalar-loss `backward()`, traced execution, transform AD, einsum,
linear algebra, and CUDA execution through the feature-gated CUDA backend.

## Setup

Start with the runtime crate. Use a local checkout while the crates are still
evolving:

```toml
[dependencies]
tenferro-runtime = { path = "/path/to/tenferro-rs/tenferro-runtime" }
```

This uses the default `cpu-faer` backend. To use the LAPACK/BLAS CPU backend,
disable default features and enable `cpu-blas`:

```toml
[dependencies]
tenferro-runtime = { path = "/path/to/tenferro-rs/tenferro-runtime", default-features = false, features = ["cpu-blas"] }
```

Exactly one CPU backend must be enabled: `cpu-faer` or `cpu-blas`. The
`cpu-blas` backend needs a BLAS/LAPACK provider. Link one from the system
toolchain, or enable the provider feature on `tenferro-tensor` to build against
OpenBLAS:

```toml
[dependencies]
tenferro-runtime = { path = "/path/to/tenferro-rs/tenferro-runtime", default-features = false, features = ["cpu-blas"] }
tenferro-tensor = { path = "/path/to/tenferro-rs/tenferro-tensor", default-features = false, features = ["src-openblas"] }
```

Add `tenferro-ad`, `tenferro-einsum`, `tenferro-linalg`, `tenferro-fft`, or
`tenferro-gpu` when a workflow needs those layers. Enable `autodiff` on
operation crates such as `tenferro-linalg` when extension AD rules are needed.
Switch to crates.io once published:

```toml
[dependencies]
tenferro-runtime = "..."
```

## First CPU Program

<!-- snippet-source: tenferro-runtime/examples/cpu_quickstart.rs -->
```rust
use tenferro_runtime::{CpuBackend, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CpuBackend::new();

    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]);
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0]);

    let c = a.matmul(&b, &mut backend)?;

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.as_slice::<f64>().unwrap(), &[19.0, 43.0, 22.0, 50.0]);

    Ok(())
}
```
<!-- end-snippet-source -->

## Next Steps

- [Core concepts](./core-concepts.md)
- [Choosing a tensor layer](../guides/choosing-an-api.md)
- [Execution models](../guides/execution-models.md)
- [Memory order](../guides/memory-order.md)
- [Devices and GPU](../guides/devices-and-gpu.md)
- [PyTorch and JAX mapping](./pytorch-jax-mapping.md)
- [Eager operations guide](../guides/eager-operations.md)
- [Tensor operations guide](../guides/tensor-operations.md)
- [Einsum guide](../guides/einsum.md)
- [Autodiff guide](../guides/autodiff.md)
- [Linear algebra guide](../guides/linear-algebra.md)
