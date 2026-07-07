# Backend Capability Descriptor

This note records the first backend capability descriptor introduced for
tenferro's concrete tensor layer.

## Problem

`TypedTensor<T>` and `Tensor` are two faces of the same concrete tensor value:
the typed face carries dtype knowledge in Rust trait bounds, while the erased
face discovers dtype support at runtime. Before this descriptor, backend
support was encoded as scattered dispatch arms and string-based
`BackendFailure` errors. That made CUDA documentation drift likely and made
CPU/CUDA parity tests hard to derive from one source of truth.

## Schema

The descriptor lives in `tenferro-tensor` and each backend exposes a static
table through `TensorBackendCapability`.

Each entry is keyed by:

- backend id,
- `tenferro-core-ops::PrimitiveOpKind`,
- representative input dtype.

Each entry records:

- output dtype,
- owned-result support,
- borrowed-input support,
- caller-provided output support,
- strided-output support,
- accumulation/read-modify-write support.

Support is three-valued:

| Level | Meaning |
| --- | --- |
| `Unsupported` | The backend must reject this op/dtype/axis explicitly. |
| `FallbackCopy` | The operation is accepted through a generic copy/materialization path. |
| `Native` | The backend has a direct implementation for that axis. |

The core-op catalog remains the semantic source for dtype policy and output
dtype mapping. Capability entries add backend implementation status; they must
not duplicate or redefine the catalog's op vocabulary.

## Current Scope

The first scope covers core elementwise, analytic, reduction, and
`dot_general` descriptors for CPU and CUDA. Linalg, FFT, indexing, structural
ops, and provider installation details remain outside this first table unless a
future change explicitly extends the schema for those families.

The CUDA guide's first-scope core primitive table is rendered from
`tenferro_gpu::cuda_capabilities()` and checked by a test. Manual rows remain
only for coverage outside this first descriptor scope.

## Consumers

Typed code continues to use compile-time trait bounds. Erased dispatch code can
query `TensorBackendCapability::capability` or
`TensorBackendCapability::require_capability` before selecting a runtime path.
The latter returns the structured `UnsupportedOpDType` error when an entry is
missing or a requested axis is unsupported.

Runtime-dtype dispatch should keep the descriptor check at the dtype-stripping
seam. Backend code performs the descriptor check there, and the shared
`with_scalar!`-style macros centralize the O(1) dtype match and structured
rejection before backend kernels run; they do not add tensor materialization. For
eager execution this adds one small descriptor/table lookup to the existing
per-operation dispatch cost, so it is acceptable for ordinary eager paths but
should not be repeated inside element loops or kernel launch inner loops.

Parity tests should derive cases from descriptor entries. Supported CPU and CUDA
entries run smoke cases and compare results; unsupported descriptor entries
must remain explicit so future support additions update the descriptor and the
generated CUDA table in the same change.
