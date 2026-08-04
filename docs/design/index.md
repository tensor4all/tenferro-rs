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
| [storage-ownership-contracts.md](./storage-ownership-contracts.md) | Normative Phase 1 contracts for the storage ownership redesign (#1555): span access and retirement, allocation groups, submission, method distribution, raw handles and reclamation, documentation ownership, and AD value retention |
| [tensor-prims.md](./tensor-prims.md) | Tensor backend protocol and execution surface |
| [backend-capability.md](./backend-capability.md) | Runtime backend capability descriptor for typed/erased tensor support and generated CUDA coverage |
| [integer-semantics.md](./integer-semantics.md) | CPU/CUDA integer arithmetic semantics, wrapping behavior, and structured domain errors |
| [output-modes.md](./output-modes.md) | Output-update vocabulary for `_read`, `_into`, `_add_to`, and dot `_into_accum` surfaces |
| [algebra.md](./algebra.md) | Algebra boundary, external numeric extensions, tropical paths |
| [einsum.md](./einsum.md) | Einsum public API, N-ary contraction tree, algebra dispatch |
| [fft-backend-execution.md](./fft-backend-execution.md) | FFT backend capability, validated plan descriptors, no-fallback placement semantics, and backend-neutral cache ownership |
| [contraction-pipeline.md](./contraction-pipeline.md) | Binary contraction pipeline, copy elision |
| [dot-general-overhead.md](./dot-general-overhead.md) | Fixed-cost analysis for many small `dot_general` contractions |
| [dynamic-symbolic-shapes.md](./dynamic-symbolic-shapes.md) | Dynamic and symbolic shape architecture. Covers `DynamicTruncate`, `transpose_scatter`, symbolic `slice_sizes`, the `Exact`/`UpperBound`/`Unknown` extent model, and extension equalities from graph-owned scopes through compiled guards. This is the key design doc for understanding how tenferro differs from XLA-style shape-specialized compilation. |
| [gpu-backend-design.md](./gpu-backend-design.md) | GPU backend architecture |
| [xla-backend.md](./xla-backend.md) | Experimental StableHLO lowering and PJRT plugin-loading peer executor over static `CompiledGraph` values |
| [inplace-indexing.md](./inplace-indexing.md) | Partial in-place updates design |
| [linalg.md](./linalg.md) | Linalg public/composite layer |
| [linalg-prims.md](./linalg-prims.md) | Backend-facing factorization and solve contracts |
| [capi.md](./capi.md) | C-API: opaque handles, DLPack, einsum + SVD + AD |
| [capi-error-handling.md](./capi-error-handling.md) | C-API error handling policy |
| [testing.md](./testing.md) | Testing and performance verification strategy |
| [execution-engine-provider-architecture.md](./execution-engine-provider-architecture.md) | Authoritative runtime-owned execution-engine, provider registration, preparation, and scheduled execution architecture |
| [review-decision-records.md](./review-decision-records.md) | Relationship between historical plans, reviewer-facing work logs, and durable design records |
| [change-aware-ci.md](./change-aware-ci.md) | Conservative pull-request classification, shared command profiles, stable required checks, and trusted RunPod recovery |
| [ci-cache-trust.md](./ci-cache-trust.md) | CI cache namespace trust model, the single trusted default-branch cache writer, material-input cache keys, and immutable GPU archive artifact reuse across allocation retries |
| [runpod-gpu-provisioning.md](./runpod-gpu-provisioning.md) | Live price-ordered RunPod GPU candidate selection, the pre-registration CUDA smoke proof (NVRTC -> PTX -> launch -> sync), the bounded delete-and-retry provision loop, and paid-cost observability |

## Reference

See [Reference](../reference/) for external system surveys and comparison notes.
