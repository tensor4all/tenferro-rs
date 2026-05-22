# tenferro

tenferro is a dense tensor computation library for Rust users who want
PyTorch- and JAX-style tensor workflows with explicit Rust ownership and
backend control.

It supports:

- eager tensor operations with `Tensor`, `TypedTensor`, and a backend,
- eager scalar-loss reverse-mode AD with `EagerTensor`,
- lazy traced execution with `TracedTensor`, `GraphCompiler`, and
  `GraphExecutor`,
- transform AD with `grad`, `vjp`, `jvp`, and HVP composition,
- einsum and linear algebra,
- CUDA execution through the feature-gated CubeCL backend.

## Start Here

- [Getting Started](getting-started/index.md)
- [Choosing an API](guides/choosing-an-api.md)
- [Custom Tensor Operations](guides/custom-operations.md)
- [Devices and GPU](guides/devices-and-gpu.md)
- [API Reference](api/index.md)

## First CPU Example

<!-- snippet-source: tenferro/examples/cpu_quickstart.rs -->
```rust
use tenferro::{CpuBackend, Tensor};

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

## Mental Model

| Workflow | Use |
| --- | --- |
| Direct CPU computation | `Tensor` or `TypedTensor` with `CpuBackend` |
| Scalar-loss eager AD | `EagerTensor` with `EagerRuntime` |
| Transform AD and graph optimization | `TracedTensor` with `GraphCompiler` and `GraphExecutor` |
| CUDA execution | `tenferro::cuda::CudaBackend` with the `cuda` feature and explicit upload/download |
