# tenferro-rs Design Documents

## Core Design

| Document | Description |
|----------|-------------|
| [architecture.md](./architecture.md) | Workspace layers, crate dependency graph, device layer, ecosystem relationships |
| [tensor-prims.md](./tensor-prims.md) | `TensorPrims<A>` trait, `PrimDescriptor`, plan-based execution, CPU/GPU backends |
| [einsum.md](./einsum.md) | Einsum public API (9 functions), N-ary contraction tree, algebra dispatch |
| [contraction-pipeline.md](./contraction-pipeline.md) | Binary contraction pipeline, copy elision, `MakeContiguous`, HPTT experiments |
| [tensor.md](./tensor.md) | `Tensor<T>`, `TensorView`, ownership model, async `CompletionEvent` |
| [algebra.md](./algebra.md) | `HasAlgebra`, `Semiring`, tropical and user-defined algebra extensibility |
| [autodiff.md](./autodiff.md) | `chainrules-core`/`chainrules` AD architecture, linalg AD rules, SVD rrule |

## Reference

| Document | Description |
|----------|-------------|
| [reference/libtorch.md](./reference/libtorch.md) | PyTorch/libtorch C++ tensor infrastructure survey |
| [reference/itensor-ecosystem.md](./reference/itensor-ecosystem.md) | ITensor Julia ecosystem analysis and Rust mapping |
| [reference/einsum-algorithm-comparison.md](./reference/einsum-algorithm-comparison.md) | strided-rs vs omeinsum-rs optimization comparison (historical decision record) |

## Integrations

| Document | Description |
|----------|-------------|
| [integrations/burn.md](./integrations/burn.md) | Burn framework interop for hybrid NN + Tensor Network models |

## Implementation Plans

See [`../plans/`](../plans/) for dated implementation plans and design decisions.
