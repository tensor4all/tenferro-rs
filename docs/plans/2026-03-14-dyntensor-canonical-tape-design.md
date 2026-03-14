# DynTensor Canonical Tape Design

## Goal

Make `tenferro-dyadtensor` use a single homogeneous AD graph value while keeping
structured tensor support, rank-0 tensor scalar semantics, and existing `Diag`
representation. The target end state is `Tape<DynTensor>`, not
`Tape<StructuredTensor<T>>`.

## Why This Change

The current design still bakes scalar type into the tape payload:

- `AdTensor<T>` reverse mode uses `TrackedValue<StructuredTensor<T>>`
- `Tape` is homogeneous in `StructuredTensor<T>`
- `DynAdTensor` is only a dynamic facade over typed graph values

That is clean for homogeneous typed graphs, but it is a poor foundation for:

- PyTorch-like runtime-typed AD values
- explicit dtype casts at the AD layer
- implicit mixed-dtype promotion within tensor operations
- a canonical dynamic payload that can cross runtime/storage/FFI boundaries

PyTorch keeps runtime-typed tensors on the autograd graph. tenferro should do
the same at the dyadtensor layer.

## Design Principles

- Keep one graph value type per tape: `Tape<DynTensor>`
- Keep scalar semantics as rank-0 tensors; do not reintroduce scalar graph types
- Keep `Diag` as a supported structured special case for now
- Keep linalg AD dense-only for now
- Keep structured AD limited to operations with clear dense reference semantics
- Avoid compatibility shims and temporary adapter layers
- Keep public API examples and docs aligned with the final model

## Target Model

### Canonical Primal Dynamic Type

`DynTensor` becomes the canonical dynamic primal tensor type and wraps
`StructuredTensor<T>` for each supported dtype:

- `F32(StructuredTensor<f32>)`
- `F64(StructuredTensor<f64>)`
- `C32(StructuredTensor<Complex32>)`
- `C64(StructuredTensor<Complex64>)`

Dense tensors are represented as dense `StructuredTensor<T>`. `Diag` stays as an
optimized structured special case. General `axis_classes` remain supported.

This means `DynTensor` replaces the current split between:

- dynamic dense tensor payloads
- structured primal snapshots

### Canonical AD Dynamic Type

`DynAdTensor` remains the public dynamic AD tensor type, but its internal model
changes. It becomes a thin dynamic wrapper over graph values whose payload is
`DynTensor`.

The public mental model becomes:

- primal dynamic tensor: `DynTensor`
- AD dynamic tensor: `DynAdTensor`
- typed building blocks: `Tensor<T>`, `StructuredTensor<T>`

### Homogeneous Tape

The tape payload becomes `DynTensor`:

- `Tape<DynTensor>`
- `TrackedValue<DynTensor>`
- `DualValue<DynTensor>`

This preserves a homogeneous graph value model while allowing runtime dtype
variation inside that single value type.

## Typed Convenience Layer

`AdTensor<T>` should stop being the core graph representation. It may remain as
typed convenience if useful, but it must become a facade over the canonical
dynamic graph model rather than owning a separate typed tape model.

The core engine should not require:

- `TrackedValue<StructuredTensor<T>>`
- `Tape<StructuredTensor<T>>`

in dyadtensor public or internal operation wiring.

## Scalar Semantics

Scalars stay represented as rank-0 tensors.

- No `DynAdScalar`
- No scalar-only reverse graph path
- No scalar-only tape payload

This keeps scalar/tensor operations uniform and keeps dyadtensor aligned with a
tensor-only graph model.

## Promotion and Cast Policy

Implicit promotion and explicit cast remain distinct concepts.

### Implicit Promotion

Operation-local dtype joins choose a result dtype, similar to PyTorch
`result_type`. This is part of op execution.

### Explicit Cast

User-visible dtype conversion should be a separate API, similar to
`tensor.to(dtype=...)`.

Under the new model, this becomes possible at the AD layer because the tape
payload is `DynTensor`, not `StructuredTensor<T>`.

Current `promote_to(...)` can be reduced to an internal helper or deprecated in
favor of the explicit cast API and operation-local promotion.

## Structured AD Support Boundary

### Structured AD Allowed

Structured AD stays supported for operations with a clean dense reference:

- einsum
- reduction
- layout-preserving linear ops such as scale-like tensor operations
- reshape/diagonal-style layout operations where tangent/cotangent structure is clear

### Dense-Only AD

Linalg AD remains dense-only:

- `svd`
- `qr`
- `lu`
- `eig/eigen`
- `solve`
- `solve_triangular`
- `lstsq`
- `slogdet`
- related matrix-function families

If a structured tensor is not dense, linalg AD should reject it explicitly.

### `Diag`

`Diag` remains supported as a structured representation and participates in the
same dense-reference structured AD policy. It is not removed in this redesign.

## Correctness Policy

### Dense Linalg AD

Dense linalg AD is checked against `tensor-ad-oracles`:

- all oracle DB records parse
- all records classify
- all supported records replay

### Structured AD

Structured AD is checked against dense reference behavior:

- dense lift consistency for `frule`
- adjoint identity for `rrule`
- dense reference or finite-difference-of-grad for `hvp`

### Unsupported Cases

Structured linalg AD must fail explicitly and be tested as unsupported behavior.

## File and Module Consequences

The main crate-wide changes are:

- `core/dynamic/dyn_tensor.rs`
  - change payload from `Tensor<T>` to `StructuredTensor<T>`
- `core/value/tensor.rs`
  - remove typed tape ownership from `AdTensor<T>`
- `tape/registry.rs`
  - change reverse rule registration from `StructuredTensor<T>` to `DynTensor`
- `ops/**`
  - use `DynTensor`-centric graph values
- docs
  - update autodiff, API index, and crate docs

## Non-Goals

- removing `Diag`
- implementing structured linalg AD
- adding custom GPU scalar kernels
- preserving old typed-tape internals for compatibility

## Success Criteria

- dyadtensor AD graph payload is `DynTensor`
- dyadtensor tape is `Tape<DynTensor>`
- `DynAdTensor` is the canonical dynamic AD tensor API
- scalar AD remains rank-0 tensor based
- `Diag` remains supported
- linalg AD is explicitly dense-only
- docs and tests reflect the new model
