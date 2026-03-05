# tenferro-rs Design Documents

## Core Design

| Document | Description |
|----------|-------------|
| [architecture.md](./architecture.md) | Workspace layers, crate dependency graph, device layer, ecosystem relationships |
| [device.md](./device.md) | `tenferro-device`: memory spaces, compute devices, error types, device selection |
| [tensor-prims.md](./tensor-prims.md) | `TensorPrims<A>` trait, `PrimDescriptor`, plan-based execution, CPU/GPU backends |
| [einsum.md](./einsum.md) | Einsum public API (9 functions), N-ary contraction tree, algebra dispatch |
| [contraction-pipeline.md](./contraction-pipeline.md) | Binary contraction pipeline, copy elision, `MakeContiguous`, HPTT experiments |
| [tensor.md](./tensor.md) | `Tensor<T>`, `TensorView`, ownership model, async `CompletionEvent` |
| [inplace-indexing.md](./inplace-indexing.md) | Design for partial in-place updates (`set_item_`, `IndexPut` extension, AD safety policy) |
| [algebra.md](./algebra.md) | `HasAlgebra`, `Semiring`, tropical and user-defined algebra extensibility |
| [autodiff.md](./autodiff.md) | AD architecture (`chainrules-core` contracts, `Tape<V>` + `Variable`/`DynTape` coexistence), including `retain_graph`/`create_graph` usage examples |
| [linalg.md](./linalg.md) | `tenferro-linalg` decompositions, solvers, utilities, stateless AD rules |
| [linalg-backend-api.md](./linalg-backend-api.md) | Proposed tensor-level backend layer for linalg decompositions and solves |
| [linalg-gemm-prims.md](./linalg-gemm-prims.md) | Planned migration of `tenferro-linalg` GEMM paths onto `tenferro-prims::BatchedGemm` |
| [capi.md](./capi.md) | C-API (FFI): opaque handles, DLPack interop, einsum + SVD + AD rules |
| [capi-error-handling.md](./capi-error-handling.md) | C-API error handling policy: status mapping, shared helpers, last-error API |
| [testing.md](./testing.md) | Testing strategy, handwritten linalg test coverage, gradient check method |

## AD Formula Notes

| Document | Description |
|----------|-------------|
| [AD Formula Notes](../AD/index.md) | Mathematical derivations for SVD, QR, LU, and other rrule/frule formulas |

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

See [Implementation Plans](https://github.com/tensor4all/tenferro-rs/tree/main/docs/plans) for dated implementation plans and design decisions.
