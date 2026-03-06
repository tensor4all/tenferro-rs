# Issue 280 Structured DynAdTensor Design

## Goal

Make `tenferro_dyadtensor::AdTensor` and `tenferro_dyadtensor::DynAdTensor`
the canonical structured AD tensor carriers by replacing the current dense-only
root tensor payload with a structured payload that supports `Dense` and `Diag`
without eager dense materialization.

## Problem

Today the crate exposes two parallel tensor families:

- root `AdTensor<T>` / `DynAdTensor` are dense-only AD carriers
- `partial_diag::*` exposes structured primal tensors outside the root AD path

That split is the wrong abstraction boundary. Downstream integrations that want
both native AD metadata and structure awareness have to choose between:

- preserving AD metadata only for dense tensors
- falling back to primal-only snapshots for diagonal tensors

Issue #280 requires removing that split instead of preserving it behind another
wrapper.

## Chosen Direction

Introduce a root `StructuredTensor<T>` and rebase root `AdTensor<T>` on top of
it:

```rust
pub struct StructuredTensor<T> {
    logical_dims: Vec<usize>,
    axis_classes: Vec<usize>,
    payload: Tensor<T>,
}

pub struct AdTensor<T>(pub AdValue<StructuredTensor<T>>);
```

`Dense` and `Diag` are representation cases of the same root carrier:

- dense: `axis_classes = [0, 1, 2, ...]`
- diag: `axis_classes = [0, 0, ...]`

The existing `partial_diag` public family will be removed. Any reusable logic
from `partial_diag` will be moved into internal structured modules rather than
kept as a second public abstraction.

## Alternatives Considered

### Re-export `partial_diag::AdTensor` as the root type

Rejected because it conflates the structured primal payload with the AD carrier
boundary and drags the current `partial_diag` naming and responsibilities into
the canonical root API.

### Keep root dense-only and add another root enum beside `partial_diag`

Rejected because it preserves the parallel abstraction problem that issue #280
exists to remove.

## Public API Shape

The root public API will be:

- `tenferro_dyadtensor::StructuredTensor<T>`
- `tenferro_dyadtensor::AdTensor<T>`
- `tenferro_dyadtensor::DynAdTensor`

Root constructors will be consolidated around the structured payload:

- `StructuredTensor::new(logical_dims, axis_classes, payload)`
- `StructuredTensor::from_dense(payload)`
- `StructuredTensor::from_diagonal_vector(payload, logical_rank)`
- `AdTensor::new_primal(structured)`
- `AdTensor::new_forward(structured, tangent)`
- `AdTensor::new_reverse(structured, node, tape, tangent)`

Required root inspection methods:

- `logical_dims()`
- `axis_classes()`
- `payload()`
- `to_dense()`
- layout predicates such as `is_dense()` and `is_diag()`

`dims()` on the AD carrier should continue to report logical dimensions so
downstream code does not have to special-case compressed payload rank.

`partial_diag` public typed and dynamic wrappers will be removed. Metadata
helpers worth keeping, such as axis-class planning, will either move to root
exports or internal `structured::*` modules.

## Internal Module Shape

The structured implementation will be split into focused modules instead of a
single replacement file:

- `structured/layout.rs`
  - `StructuredTensor<T>`
  - canonicalization and layout validation
  - dense/diag constructors
- `structured/meta.rs`
  - axis-class planning helpers migrated from `partial_diag::meta`
- `structured/einsum.rs`
  - structured contraction planning and execution
- additional small helper files as needed for dense materialization or shared
  representation utilities

The current `partial_diag` implementation code is not thrown away. It is
decomposed and moved under `structured::*`, then the old public module is
removed.

## AD and Reverse Semantics

The cotangent space must be the same structured payload space as the primal:

- `StructuredTensor<T>` primals produce `StructuredTensor<T>` tangents
- reverse pullbacks return `StructuredTensor<T>` cotangents
- dense and diag both remain inside the root AD carrier throughout eager ops

This means graph-changing tensor operations cannot keep using dense-only helper
paths that assume `Tensor<T>` cotangents. Structured-aware helpers must allocate
fresh reverse nodes and register reverse rules in the structured cotangent
space.

For the first migration slice, structured-aware native support is required for:

- `scale`
- `axpby`
- `real_part`
- `imag_part`
- `compose_complex`
- `conj`
- contraction / einsum
- full reduction / `sum`

## Dense Fallback Boundary

Dense fallback is allowed only as an explicit implementation detail for linalg
operations that are not yet structure-aware:

- `svd`
- `qr`
- `eig`
- `solve`
- related factorization-based eager ops

The public input and output types still remain root `AdTensor<T>`. Unsupported
structured linalg should materialize internally with `to_dense()`, run the
existing dense path, and then wrap outputs back into dense-form
`StructuredTensor::from_dense(...)`.

This keeps the public root type unified while allowing staged migration.

## Migration Plan

The intended implementation sequence is:

1. Introduce `StructuredTensor<T>` and move structured layout helpers out of
   `partial_diag`
2. Rebase root `AdTensor<T>` and `DynAdTensor` onto `StructuredTensor<T>`
3. Port root eager tensor operations to the structured carrier for `Dense` and
   `Diag`
4. Update reverse tape plumbing and mixed scalar/tensor bridges to use
   structured cotangents
5. Route dense-only linalg wrappers through explicit internal dense fallback
6. Remove the public `partial_diag` module and reconcile docs/tests to the root
   API

## Testing Requirements

The migration must add coverage for:

- root `AdTensor` carrying dense structured payloads
- root `AdTensor` carrying diagonal structured payloads
- forward and reverse mode preservation for dense and diag through:
  - `scale`
  - `axpby`
  - `real_part`
  - `imag_part`
  - `compose_complex`
  - `conj`
  - `einsum`
  - `sum`
- explicit dense fallback behavior for at least one linalg path
- removal of `partial_diag` public APIs without losing root documentation

The migration should not weaken coverage thresholds or test tolerances.

## Risks

- The diff will be wide because `api/mod.rs`, `api/ad.rs`, reverse tape storage,
  root tensor wrappers, and documentation all assume dense-only tensor payloads
- `partial_diag` removal changes many doc examples and tests at once
- Multi-output linalg wrappers will need careful auditing once root tensors stop
  being plain `Tensor<T>` carriers

## Follow-up Boundaries

This issue intentionally stops at a stable `Dense + Diag` root carrier.
Possible future extensions after the root contract stabilizes:

- general `PartialDiag`
- block-sparse payloads
- external buffers
- symmetry-aware payloads

Those should extend `StructuredTensor<T>` rather than reintroducing a second
root tensor family.
