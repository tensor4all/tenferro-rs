# Implementation Design Notes

Implementation-focused design notes for specific subsystems. For high-level
architecture see [Architecture](../architecture/). For normative specs see
[Specification](../spec/).

## Core Design

| Document | Description |
|----------|-------------|
| [supported-ops.md](./supported-ops.md) | Crate-by-crate inventory of supported primal and AD operations |
| [device.md](./device.md) | tenferro-device: memory spaces, compute devices, error types |
| [tensor.md](./tensor.md) | Tensor representation, ownership model |
| [tensor-prims.md](./tensor-prims.md) | Tensor primitive protocol families |
| [algebra.md](./algebra.md) | HasAlgebra, Semiring, tropical and user-defined algebra |
| [einsum.md](./einsum.md) | Einsum public API, N-ary contraction tree, algebra dispatch |
| [contraction-pipeline.md](./contraction-pipeline.md) | Binary contraction pipeline, copy elision |
| [einsum-dyadtensor.md](./einsum-dyadtensor.md) | AD integration for einsum + frontend |
| [gpu-backend-design.md](./gpu-backend-design.md) | GPU backend architecture |
| [inplace-indexing.md](./inplace-indexing.md) | Partial in-place updates design |
| [linalg.md](./linalg.md) | Linalg public/composite layer |
| [linalg-prims.md](./linalg-prims.md) | Backend-facing factorization and solve contracts |
| [linalg-backend-api.md](./linalg-backend-api.md) | Earlier linalg backend layer proposal |
| [linalg-gemm-prims.md](./linalg-gemm-prims.md) | GEMM-backed linalg migration notes |
| [einsum-cpu-porting-notes.md](./einsum-cpu-porting-notes.md) | CPU einsum porting notes |
| [capi.md](./capi.md) | C-API: opaque handles, DLPack, einsum + SVD + AD |
| [capi-error-handling.md](./capi-error-handling.md) | C-API error handling policy |
| [testing.md](./testing.md) | Testing and performance verification strategy |

## Proposal Sets

| Document | Description |
|----------|-------------|
| [design_v3/README.md](./design_v3/README.md) | Non-canonical proposal set for traced graph, AD, shape metadata, and tropical externalization |

## Reference

See [Reference](../reference/) for external system surveys and comparison notes.
