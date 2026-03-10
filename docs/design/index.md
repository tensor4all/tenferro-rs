# tenferro-rs Design Documents

## Core Design

| Document | Description |
|----------|-------------|
| [architecture.md](./architecture.md) | Workspace layers, dependency direction, and protocol boundaries after the prims/linalg split |
| [device.md](./device.md) | `tenferro-device`: memory spaces, compute devices, error types, device selection |
| [tensor-prims.md](./tensor-prims.md) | `tenferro-prims` protocol families: semiring core, semiring fast paths, scalar prims, analytic prims |
| [einsum.md](./einsum.md) | Einsum public API (9 functions), N-ary contraction tree, algebra dispatch |
| [contraction-pipeline.md](./contraction-pipeline.md) | Binary contraction pipeline, copy elision, `MakeContiguous`, HPTT experiments |
| [tensor.md](./tensor.md) | `Tensor<T>`, `TensorView`, ownership model, async `CompletionEvent` |
| [inplace-indexing.md](./inplace-indexing.md) | Design for partial in-place updates (`set_item_`, `IndexPut` extension, AD safety policy) |
| [algebra.md](./algebra.md) | `HasAlgebra`, `Semiring`, tropical and user-defined algebra extensibility |
| [autodiff.md](./autodiff.md) | AD architecture (`chainrules-core` contracts, homogeneous `Tape<V>` graphs, `Variable<V>` query/mutation APIs), including `retain_graph`/`create_graph` usage examples |
| [einsum-dyadtensor.md](./einsum-dyadtensor.md) | AD integration design for `tenferro-einsum` + dyadtensor-style wrappers on top of homogeneous `Tape<V>` and rank-0 tensor scalar semantics |
| [linalg-prims.md](./linalg-prims.md) | `tenferro-linalg-prims`: backend-facing factorization and solve contracts |
| [linalg.md](./linalg.md) | `tenferro-linalg` public/composite layer and its relationship to prims/linalg-prims |
| [capi.md](./capi.md) | C-API (FFI): opaque handles, DLPack interop, einsum + SVD + AD rules |
| [capi-error-handling.md](./capi-error-handling.md) | C-API error handling policy: status mapping, shared helpers, last-error API |
| [testing.md](./testing.md) | Testing and performance verification strategy, including the external einsum benchmark gate |

## Historical Proposals

These documents remain useful as background, but they are not the primary
source of truth for the current split architecture:

| Document | Description |
|----------|-------------|
| [linalg-backend-api.md](./linalg-backend-api.md) | Earlier proposal for a tensor-level linalg backend layer |
| [linalg-gemm-prims.md](./linalg-gemm-prims.md) | Earlier migration notes for GEMM-backed linalg helpers |

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
