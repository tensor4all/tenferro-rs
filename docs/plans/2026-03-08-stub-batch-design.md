# Stub Batch Design

## Scope

Fix the remaining non-GPU-runtime public stub issues in one batch:

- `#375` `tenferro-mdarray` conversion functions
- `#378` `tenferro-burn` public einsum and conversion entrypoints
- `#384` `tenferro-capi` DLPack import/export
- `#385` `tenferro-linalg` `linalg-lapack` tensor dispatch
- `#388` `tenferro-tropical-capi` tropical einsum C entrypoints

Also clarify in `README.md` and `AGENTS.md` that GPU support is currently stubbed and planned for future implementation.

## Design

### 1. CPU-facing conversion and dispatch gaps

`tenferro-mdarray` will implement dense copy/move conversions using the storage orders each library already defines: mdarray is row-major and tenferro can represent row-major tensors explicitly. `tenferro-linalg` already has a complete slice-level LAPACK backend, so the tensor-level `cpu_lapack` module should stop returning placeholder device errors and instead reuse the same tensor-shape validation and batch loops as the faer path.

### 2. C API interop

`tenferro-capi` DLPack support should be real zero-copy for CPU `f64` tensors. Export will consume the tensor handle, keep the `Tensor<f64>` alive inside a DLPack manager object, and surface exact shape/stride metadata. Import will validate DLPack dtype/device, attach the foreign buffer through `DataBuffer::from_external`, and call the incoming DLPack deleter when the resulting tensor is released.

`tenferro-tropical-capi` should stop being nine unconditional argument rejections. The primal path will promote `Tensor<f64>` handles into tropical scalars, call the existing tropical/tenferro einsum machinery, then unwrap back to `f64`. Reverse-mode will call the existing `tropical_einsum_rrule`. Forward-mode will be added in the tropical crate using the same winner-tracking logic as reverse-mode so the C API can expose a real JVP surface instead of a stub.

### 3. Burn bridge

The Burn bridge should become honest and usable for the implemented subset. Primitive conversions will use Burn `TensorData` and support backends whose float element type is `f64`. Forward einsum for `NdArray<f64>` will convert primitives into tenferro tensors, call standard einsum, then convert back. The autodiff backend will register unary/binary custom backward ops using Burn's `Backward` API and tenferro `einsum_rrule`; larger arities remain out of scope and should fail clearly instead of deferring to `todo!()`.

## Testing

- Add module-local tests for `tenferro-mdarray` round-trips.
- Add `linalg-lapack + provider-inject` tensor-level tests that register mock BLAS/LAPACK symbols and verify tensor dispatch no longer returns the placeholder error.
- Extend `tenferro-capi` tests with DLPack export/import round-trips and ownership checks.
- Add `tenferro-tropical-capi` tests for primal, rrule, and frule on valid matmul-style inputs.
- Add `tenferro-burn` tests for conversion round-trips, forward einsum, and autodiff unary/binary gradient propagation.

## Risks

- DLPack ownership bugs can double-free or leak memory. The manager object and deleter path must be exercised in tests.
- Burn autodiff integration is the most coupled part of this batch. Keep it to unary/binary first and fail clearly outside that subset.
- New public tensor constructors for external buffers must preserve existing invariants around shape, stride, and offset validation.
