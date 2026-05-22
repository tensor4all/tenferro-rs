# tenferro

tenferro is a dense tensor computation stack for Rust users who want direct
typed tensor computation, PyTorch-like eager autodiff, JAX-like traced graph
transforms, einsum, linear algebra, and explicit CPU/CUDA backend control.

The project covers both no-AD and AD workflows. Start with the lowest layer
that solves your problem, then add autodiff, graph compilation, or CUDA only
when the workflow needs them.

## Start Here

- [Core Concepts](getting-started/core-concepts.md)
- [Choosing a Tensor Layer](guides/choosing-an-api.md)
- [Execution Models](guides/execution-models.md)
- [Memory Order](guides/memory-order.md)
- [Devices and GPU](guides/devices-and-gpu.md)
- [API Reference](api/index.md)

Installation details live in the README and the short
[Installation](guides/installation.md) guide. The online guides focus on the
tensor stack, memory model, execution modes, operation coverage, and extension
model.

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

tenferro has three independent axes. Do not read CUDA, eager AD, and traced
graphs as competing APIs; they answer different questions.

| Axis | Question | Choices |
| --- | --- | --- |
| Data layer | What kind of tensor value do I hold? | `TypedTensor<T>`, `Tensor`, `EagerTensor`, `TracedTensor` |
| Execution model | When does computation run? | Direct backend call, eager AD, traced compile/run |
| Device/backend | Where does computation run? | CPU backend or CUDA backend, with explicit transfer |

Most no-AD code starts with `TypedTensor<T>` when the scalar type is known at
compile time, or `Tensor` when dtype must be dynamic. `EagerTensor` adds
PyTorch-style scalar-loss `backward()`. `TracedTensor` adds graph compilation,
transform AD, symbolic inputs, and reuse.

## Guides By Workflow

| Workflow | Start with |
| --- | --- |
| No-AD computation with a fixed scalar type | [`TypedTensor<T>` and direct tensor workflows](guides/choosing-an-api.md) |
| No-AD dynamic dtype computation | [`Tensor` with a backend](guides/eager-operations.md) |
| PyTorch-like scalar-loss autodiff | [`EagerTensor` and `EagerRuntime`](guides/eager-operations.md) |
| JAX-like graph transforms and repeated execution | [`TracedTensor`, `GraphCompiler`, and `GraphExecutor`](guides/execution-models.md) |
| CUDA execution | [Explicit upload/download and CUDA backend coverage](guides/devices-and-gpu.md) |
| External operations and AD rules | [Custom operations](guides/custom-operations.md) and [`tenferro-fft`](guides/tenferro-fft.md) |
