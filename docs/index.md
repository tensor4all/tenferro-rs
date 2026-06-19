# tenferro

tenferro is a dense tensor computation stack for Rust users who want typed
tensor computation, PyTorch-like immediate execution with `backward()`,
JAX-like traced graphs, einsum, linear algebra, and explicit CPU, CUDA, and
experimental WebGPU backend control.

The project covers both ordinary tensor computation and autodiff workflows.
Start with the smallest API that solves your problem, then add autodiff, graph
compilation, CUDA, or experimental WebGPU only when the workflow needs them.

![tenferro-rs architecture overview](assets/tenferro-architecture.svg)

## Where To Start

| Workflow | Start with |
| --- | --- |
| First setup and a checked CPU program | [Getting Started](getting-started/index.md) |
| Core terms and the three main choices | [Core Concepts](getting-started/core-concepts.md) |
| Step-by-step runnable examples | [Tutorials](tutorials/index.md) |
| Choosing between `TypedTensor`, `Tensor`, `EagerTensor`, and `TracedTensor` | [Choosing a Tensor API](guides/choosing-an-api.md) |
| Understanding direct, eager, and traced execution | [Execution Models](guides/execution-models.md) |
| Column-major storage and row-major import/export | [Memory Order](guides/memory-order.md) |
| CPU, CUDA, and experimental WebGPU backend behavior | [Devices and GPU](guides/devices-and-gpu.md) |
| Static-shaped StableHLO and PJRT plugin loading | [XLA and PJRT](guides/xla.md) |
| Runtime-dependent dimensions in traced graphs | [Dynamic and symbolic shapes](design/dynamic-symbolic-shapes.md) |
| API documentation for every crate | [API Reference](api/index.md) |

## First CPU Example

<!-- snippet-source: crates/tenferro-runtime/examples/cpu_quickstart.rs -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{tensor, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CpuBackend::new();

    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?;
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0])?;

    let c = tensor::matmul(&a, &b, &mut backend)?;

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.as_slice::<f64>().unwrap(), &[19.0, 43.0, 22.0, 50.0]);

    Ok(())
}
```
<!-- end-snippet-source -->

## Mental Model

tenferro has three independent choices. GPU providers, eager execution, and
traced graphs are not competing APIs; they answer different questions.

| Choice | Question | Options |
| --- | --- | --- |
| Tensor API | What kind of value do I pass around? | `TypedTensor<T>`, `Tensor`, `EagerTensor`, `TracedTensor` |
| Execution timing | When does computation run? | Direct backend call, immediate eager execution, traced compile/run |
| Backend/device | Where does computation run? | CPU, CUDA, or experimental WebGPU backend, with explicit transfer |

Most code without autodiff starts with `TypedTensor<T>` when the scalar type is
known at compile time, or `Tensor` when dtype must be selected at runtime.
`EagerTensor` adds immediate execution through an `EagerRuntime` and optional
`backward()` on scalar losses. `TracedTensor` adds graph compilation, `grad`,
`vjp`, and `jvp` on traced graphs, symbolic inputs, and reuse.

## Get In Touch

Questions, design discussions, and contributor coordination for tenferro happen
in the tenferro Matrix room:

- [#tenferro-tensor4all:matrix.org](https://matrix.to/#/#tenferro-tensor4all:matrix.org)

Matrix is an open, federated chat protocol. You can join from a Matrix client
such as Element, or through the browser flow opened by the link above.

Use GitHub issues for bug reports, feature requests, and decisions that need
tracking; use Matrix for lightweight discussion before filing or implementing
changes.
