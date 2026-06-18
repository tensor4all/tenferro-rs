# Remove Row-Major Tensor Surface Design

## Goal

Remove row-major storage from the concrete tensor surface for now. tenferro
will expose one dense owned tensor layout: contiguous column-major.

## Decision

`Tensor` and `TypedTensor` should no longer carry a public runtime
`MemoryOrder` choice. The row-major APIs added for boundary convenience are
removed instead of kept as compatibility shims.

This means:

- `Tensor::from_vec` and `TypedTensor::from_vec` accept column-major buffers.
- owned host export returns column-major buffers.
- CPU, GPU, einsum, and linalg paths can assume contiguous column-major tensor
  storage.
- NumPy, PyTorch, JAX, ndarray, and oracle inputs that start as row-major flat
  buffers must be converted before constructing a tensor, using local helper
  code outside the public tensor API.

## Rationale

The current runtime `MemoryOrder` tag is not represented in the type system. If
row-major and column-major tensors share the same `Tensor` type, output order
policy becomes implicit and hard to reason about. Sticky output order would make
operation results depend on input provenance, while canonicalizing outputs to
column-major would reduce row-major to an import/export convenience.

Because tenferro's internal performance model, linalg layout, einsum layout,
and CubeCL GPU storage are already column-major, a single public column-major
tensor surface is simpler and less error-prone.

## Scope

Remove:

- `MemoryOrder::RowMajor` and the runtime tensor `order` field if it becomes
  unnecessary.
- `from_vec_row_major`, `from_vec_col_major`, and `from_vec_with_order` when
  they only exist to expose layout choice.
- `to_row_major`, `to_col_major`, `to_order`, `into_vec_row_major`,
  `into_vec_col_major`, and `try_into_vec_with_order`.
- row-major behavior claims from user-facing docs.
- row-major-specific production tests.

Keep:

- column-major shape, indexing, and buffer semantics.
- private or test-local row-major-to-column-major conversion helpers when an
  external fixture format requires them.
- explicit documentation that tenferro flat buffers are column-major.

## Compatibility

This is a breaking API cleanup. Do not add deprecated aliases or hidden
compatibility shims. Call sites should be updated to construct column-major
tensors directly or perform explicit conversion before construction.

## Testing

The implementation should verify:

- tensor construction and owned export still round-trip column-major buffers;
- element access still follows column-major offsets;
- CPU operation tests remain correct without row-major variants;
- CubeCL upload/download tests no longer mention row-major canonicalization;
- user docs and doctests no longer reference removed row-major APIs.
