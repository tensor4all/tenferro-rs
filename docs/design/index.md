# Implementation Design Notes

Implementation-focused design notes for specific subsystems. For high-level
architecture see [Architecture](../architecture/). For normative specs see
[Specification](../spec/).

## Core Design

| Document | Description |
|----------|-------------|
| [api-and-convention-freeze.md](./api-and-convention-freeze.md) | Clean-break release posture for public API, API consistency detection, crate ownership, naming conventions, docs ownership, and audit/remediation workflow |
| [supported-ops.md](./supported-ops.md) | Crate-by-crate inventory of supported primal and AD operations |
| [tensor.md](./tensor.md) | Tensor representation, ownership model |
| [tensor-prims.md](./tensor-prims.md) | Tensor backend protocol and execution surface |
| [algebra.md](./algebra.md) | Algebra boundary, external numeric extensions, tropical paths |
| [einsum.md](./einsum.md) | Einsum public API, N-ary contraction tree, algebra dispatch |
| [contraction-pipeline.md](./contraction-pipeline.md) | Binary contraction pipeline, copy elision |
| [dot-general-overhead.md](./dot-general-overhead.md) | Fixed-cost analysis for many small `dot_general` contractions |
| [dynamic-symbolic-shapes.md](./dynamic-symbolic-shapes.md) | Dynamic and symbolic shape metadata contract |
| [extension-runtime-restructure.md](./extension-runtime-restructure.md) | Historical migration note for extension crates, backend-aware execution, multi-output, linalg migration boundaries, and autodiff feature gates |
| [einsum-extension-migration-audit.md](./einsum-extension-migration-audit.md) | Source-to-target migration checklist for moving einsum into tenferro-einsum without regenerating the implementation |
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
| [review-decision-records.md](./review-decision-records.md) | Relationship between historical plans, reviewer-facing work logs, and durable design records |

## Reference

See [Reference](../reference/) for external system surveys and comparison notes.
