# Borrowed Extension Reads

## Status

Accepted design for issue #1709. Implementation requires an independent design-review verdict before code changes.

## Goal

Read-only extension execution borrows compact same-placement tensor storage across the runtime boundary without duplicating input tensors. Materialization and transfer remain explicit choices owned by the extension executor.

## Existing boundary

`define_extension_runtime!` already generates `PreparedOperationExecutor::execute` over `&[TensorRead<'_>]`; the runtime passes those reads unchanged to `execute_reads`. No second execution framework or adapter is needed.

The missing author-facing primitive is direct typed host access on a dtype-erased `TensorRead`. `TensorView::as_slice<T>` already implements the required dtype, compact-layout, bounds, and host-access checks.

## Public contract

Add one method to `TensorRead`:

```rust
pub fn as_slice<T: TensorScalar>(&self) -> Result<&'a [T]>;
```

It delegates to the existing `TensorView::as_slice<T>` contract without allocation. The returned slice keeps the original `TensorRead<'a>` storage lifetime and may outlive the method's `&self` borrow. This is sound because cloning `TensorRead` only copies a borrowed tensor reference or shallow view metadata/ownership handles; it never creates temporary backing storage. Pointer-identity tests lock down that invariant.

- compact host `TensorRead::Tensor` and compact host `TensorRead::View` return a slice borrowing the original storage;
- a noncompact view returns the existing typed layout `InvalidArgument` validation error;
- backend-owned storage returns the existing typed host-access/runtime-state error and is never downloaded;
- dtype mismatch remains typed.

An executor that requires canonical compact storage calls `BackendSession::to_contiguous_read` explicitly. It calls it once per selected noncompact input and reuses the returned owner. No runtime layer silently materializes, transfers, or retries on CPU.

## Extension fixture and instrumentation

Extend the workspace-external authoring fixture from #1708 with a read-only four-input operation.

- The compact path calls `TensorRead::as_slice::<f64>()` on each input, records the borrowed pointers, and verifies they match the four caller-owned input buffers. It does not call `to_contiguous_read`.
- A materializing operation explicitly calls `to_contiguous_read` once for one noncompact view before reading it; an atomic callback counter distinguishes this path from the borrowed path.
- A host-only operation given backend-owned storage fails through `as_slice` with the existing typed placement/host-access error before producing output.
- A small counting allocator around the fixture's measured calls records the steady-state allocation delta after warm-up. The borrowed compact path may allocate its declared output but must show no additional full-input-sized allocation; the explicit materialization path must show exactly one full-input-sized allocation. Pointer identity and callback counters remain the primary deterministic assertions; allocator byte counts are supporting evidence because runtime bookkeeping may allocate small objects.

Existing runtime integration tests also assert that prepared extension execution forwards `TensorRead` values directly and does not invoke `to_contiguous_read` before the generated executor callback.

## Documentation

The custom-operation guide makes `execute_reads` the default recipe, demonstrates `TensorRead::as_slice`, and states:

- compact host reads borrow;
- noncompact inputs are consumed directly only by kernels that support their strides, otherwise explicitly materialized once;
- backend storage is rejected by a host executor and never transferred implicitly.

## Non-goals

- No implicit CPU/GPU transfer or CPU fallback.
- No shallow owner clone API.
- No guarantee that arbitrary strided views are accepted by every extension.
- No automatic materialization policy in the macro or runtime.
- No second execution framework or new allocation subsystem.

## Verification

- TensorRead unit tests cover compact owner, compact view, noncompact view, dtype mismatch, and backend-owned rejection.
- The external fixture covers four borrowed inputs, one explicit materialization, pointer identity, typed rejection, callback counts, and allocation-size evidence.
- Runtime tests cover unchanged eager/compiled numerical behavior and failure transactionality.
- Run tensor/runtime tests and doctests, the external fixture, public API documentation checks, targeted clippy, modified-file coverage review, and the combined PR gates.
