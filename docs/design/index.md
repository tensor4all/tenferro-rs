# Implementation Design Notes

Implementation-focused design notes for specific subsystems. For high-level
architecture see [Architecture](../architecture/). For normative specs see
[Specification](../spec/). Historical implementation plans and superseded
migration notes live under [Plans](../plans/), not this active design index.

## Core Design

| Document | Description |
|----------|-------------|
| [api-and-convention-freeze.md](./api-and-convention-freeze.md) | Clean-break release posture for public API, API consistency detection, crate ownership, naming conventions, docs ownership, and audit/remediation workflow |
| [supported-ops.md](./supported-ops.md) | Crate-by-crate inventory of supported primal and AD operations |
| [tensor.md](./tensor.md) | Tensor representation, ownership model |
| [tensor-prims.md](./tensor-prims.md) | Tensor backend protocol and execution surface |
| [backend-capability.md](./backend-capability.md) | Runtime backend capability descriptor for typed/erased tensor support and generated CUDA coverage |
| [output-modes.md](./output-modes.md) | Output-update vocabulary for `_read`, `_into`, `_add_to`, and dot `_into_accum` surfaces |
| [algebra.md](./algebra.md) | Algebra boundary, external numeric extensions, tropical paths |
| [einsum.md](./einsum.md) | Einsum public API, N-ary contraction tree, algebra dispatch |
| [contraction-pipeline.md](./contraction-pipeline.md) | Binary contraction pipeline, copy elision |
| [dot-general-overhead.md](./dot-general-overhead.md) | Fixed-cost analysis for many small `dot_general` contractions |
| [dynamic-symbolic-shapes.md](./dynamic-symbolic-shapes.md) | Dynamic and symbolic shape metadata contract. Covers `DynamicTruncate`, `transpose_scatter`, symbolic `slice_sizes`, and the `Exact`/`UpperBound`/`Unknown` extent model. This is the key design doc for understanding how tenferro differs from XLA-style shape-specialized compilation. |
| [gpu-backend-design.md](./gpu-backend-design.md) | GPU backend architecture |
| [xla-backend.md](./xla-backend.md) | Experimental StableHLO lowering and PJRT plugin-loading peer executor over static `GraphProgram` values |
| [inplace-indexing.md](./inplace-indexing.md) | Partial in-place updates design |
| [linalg.md](./linalg.md) | Linalg public/composite layer |
| [linalg-prims.md](./linalg-prims.md) | Backend-facing factorization and solve contracts |
| [capi.md](./capi.md) | C-API: opaque handles, DLPack, einsum + SVD + AD |
| [capi-error-handling.md](./capi-error-handling.md) | C-API error handling policy |
| [testing.md](./testing.md) | Testing and performance verification strategy |
| [review-decision-records.md](./review-decision-records.md) | Relationship between historical plans, reviewer-facing work logs, and durable design records |

## Reference

See [Reference](../reference/) for external system surveys and comparison notes.
