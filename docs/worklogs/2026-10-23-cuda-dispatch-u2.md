# U2, part 2: per-family dispatch

Date: 2026-10-23. Issue: [tenferro-rs#2208](https://github.com/tensor4all/tenferro-rs/issues/2208)
unit U2, continuing the launcher slice
([2026-10-23-cuda-launch-u2.md](2026-10-23-cuda-launch-u2.md)).

## What landed

- `cuda_next::Dispatcher`: plan through the kernel crate and marshal the ABI arguments
  for the elementwise, comparison, conversion, gather and scatter families —
  `unary`, `binary` (with its checked-integer status word), `compare`, `select`,
  `clamp`, `convert`, `gather`, `scatter_add`. Every plan comes from the crate's
  `plan_*` function; the module adds no validation of its own.
- `KernelModules::for_entries` prepares only the modules the requested entries export,
  so a dispatch path pays for what it launches instead of the whole catalogue.
- `CudaRuntime::download_bytes` reads a runtime allocation back, which is what a
  dispatch test (or the future read path) needs when the operand has no tensor wrapper.

## Evidence

`tests/seam_dispatch.rs` on the local A100 (driver 580.126.09, CUDA 12.6), each case
against a host reference:

- `unary`: negation over 1024 f64 values, and `Sqrt` with IEEE propagation (a negative
  input yields NaN, never an error).
- `binary`: elementwise `Max`, with the checked-integer status word asserted clear.
- `compare` + `select`: one Bool byte per comparison, then the predicate chooses between
  two operands element by element.
- `clamp` + `convert`: the clamped sequence, then the f64 → f32 conversion route.
- `gather`: a windowed gather over a `[3, 3]` operand at start 1, compared with the
  operand's rows.
- `scatter_add`: two additive updates at distinct starts, checked against the expected
  destination.

## Observations

- **`clamp` takes per-element bounds**, not scalars: the entry indexes `lower`/`upper`
  with the same element index as the input (the ABI documents the formula, not a
  broadcast). A caller that wants a scalar clamp passes a full-length constant array,
  and the dispatch test now does exactly that.
- A rank-0 index layout is `Layout::compact(&[])`; `&[0]` is a rank-1 layout with extent
  zero and the gather/scatter planner rejects it with `IndexVectorMismatch`. Recorded
  because the error message does not name the cause.
- Still open for U2: the windowing family, the reductions (U4) and the tensor-surface
  wiring (`TensorStructural`/`TensorElementwise` for the new engine).
