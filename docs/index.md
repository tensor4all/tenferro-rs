# tenferro

tenferro is a Rust-native tensor computation stack with opt-in autodiff for
scientific workloads: typed tensors, dynamic tensors, PyTorch-style immediate
execution with `backward()` and `EagerRuntime` functional `grad`, `vjp`, and
`jvp`, JAX-style traced graphs, einsum, linear algebra, FFT, and explicit CPU,
CUDA, and experimental WebGPU backend control.

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
| CPU affinity, NUMA placement, faer, and external BLAS behavior | [CPU Execution and NUMA Placement](guides/cpu-execution.md) |
| Static-shaped StableHLO and PJRT plugin loading | [XLA and PJRT](guides/xla.md) |
| Runtime-dependent dimensions in traced graphs | [Dynamic and symbolic shapes](design/dynamic-symbolic-shapes.md) |
| API documentation for every crate | [API Reference](api/index.md) |

## First Direct Tensor Example

<!-- snippet-source: docs/tutorial-code/src/bin/direct_linalg_quickstart.rs -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_linalg::LinalgBackend;
use tenferro_runtime::{TensorRead, TensorView, TypedTensor, TypedTensorOpsExt};

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error < 1.0e-12,
            "value {index}: actual={actual}, expected={expected}, error={error}"
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CpuBackend::new();

    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![3.0, 0.0, 0.0, 1.0])?;
    let identity = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])?;

    let product = a.matmul(&identity, &mut backend)?;
    assert_eq!(product.shape(), &[2, 2]);
    assert_close(product.host_data()?, &[3.0, 0.0, 0.0, 1.0]);

    let svd = backend.svd_read(TensorRead::from_view(TensorView::F64(product.as_view())))?;
    assert_eq!(svd.len(), 3);
    assert_eq!(svd[0].shape(), &[2, 2]);
    assert_eq!(svd[1].shape(), &[2]);
    assert_eq!(svd[2].shape(), &[2, 2]);
    assert_close(svd[1].as_slice::<f64>().unwrap(), &[3.0, 1.0]);

    Ok(())
}
```
<!-- end-snippet-source -->

## Dependency Footprint

![tenferro-rs dependency footprint](assets/dependency-footprint.svg)

The checked-in diagram is rendered by Graphviz from the transitive-reduced
workspace graph produced by `scripts/gen_dep_graph.py`; node positions and
arrow attachment points are not maintained by hand. Regenerate it with:

```bash
python3 scripts/gen_dep_graph.py --format svg --output docs/assets/dependency-footprint.svg
```

Graphviz is needed only for regeneration. The semantic node and edge inventory
can be checked without rerendering:

```bash
python3 scripts/gen_dep_graph.py --check-svg docs/assets/dependency-footprint.svg
```

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
`EagerTensor` adds immediate execution through an `EagerRuntime`, optional
stateful `backward()` on scalar losses, and functional eager `grad`, `vjp`, and
`jvp` transforms. `TracedTensor` adds graph compilation, `grad`, `vjp`, and
`jvp` on traced graphs, symbolic inputs, and reuse.

## Get In Touch

Questions, design discussions, and contributor coordination for tenferro happen
in the tenferro Matrix room:

- [#tenferro-tensor4all:matrix.org](https://matrix.to/#/#tenferro-tensor4all:matrix.org)

Matrix is an open, federated chat protocol. You can join from a Matrix client
such as Element, or through the browser flow opened by the link above.

Use GitHub issues for bug reports, feature requests, and decisions that need
tracking; use Matrix for lightweight discussion before filing or implementing
changes.
