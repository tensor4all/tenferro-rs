# Tensor Async/Ownership Integration Design (Validated)

Date: 2026-02-14  
Scope: `tenferro` unified tensor design (`Tensor`, `TensorView`, `CompletionEvent`)

## Goal

Align asynchronous execution with Rust ownership semantics while keeping the public API easy to use.

## Decisions

1. Use one async contract for both CPU and GPU execution.
2. Keep tensor state as `event: Option<CompletionEvent>`.
3. Keep `wait` as `wait(&self)` (internal mutability handles state transition).
4. Keep a single `TensorView<'a, T>` type with `event: Option<&'a CompletionEvent>`.
5. Public `view()` is a synchronous readable view (it waits when pending).
6. Internal `pub(crate) as_operand_view()` is non-blocking and propagates pending events.
7. CPU-readable methods use `wait_if_pending()` at entry (no `debug_assert!(event.is_none())` contract check).

## API Contract

- Public read path:
  - `view()`, `view_mut()`, `to_tensor()`, `contiguous()`, `conj()` must return only after data is ready.
- Execution path:
  - `as_operand_view()` may carry pending event references into `einsum` internals.
- Readiness:
  - `is_ready()` reflects whether the tensor is pending (`event.is_none()` => ready).

## Data Flow

1. `einsum` receives operand views from internal non-blocking conversion.
2. Backend submit consumes dependency events and schedules execution.
3. Returned output tensor stores new `CompletionEvent` in `event`.
4. Any public CPU-read operation first calls `wait_if_pending()`, then returns data/view.

This keeps accelerator and threaded CPU pipelines intact while preserving intuitive public semantics.

## Error Handling and Safety

- If event wait fails, return backend/device error through existing `Result` channel.
- `view_mut()` always waits before exposing mutable access.
- No public API exposes a pending-readable state.

## Verification Plan

1. Unit test: pending tensor blocks on `view()` and becomes ready after wait.
2. Unit test: internal operand path chains two `einsum` calls without host sync.
3. Unit test: `view_mut()` forces completion before mutable borrow.
4. Unit test: CPU-threaded backend also uses the same event contract.
