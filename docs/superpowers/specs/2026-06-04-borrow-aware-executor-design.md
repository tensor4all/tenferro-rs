# Borrow-Aware Executor Design

Date: 2026-06-04

## Context

`GraphExecutor::run_many_with_inputs` currently accepts borrowed input bindings as
`&Tensor`, but it resolves them into owned `Tensor` values before execution. For
host-backed tensors this clones the underlying `Vec`, so a large benchmark such
as `bin_matmul_1024` copies both inputs before the timed `DotGeneral` execution.

The runtime already has a read-only tensor abstraction:

- `TensorRead<'a>` represents either `&'a Tensor` or `TensorView<'a>`.
- CPU backend APIs already provide `_read` variants for elementwise, reduction,
  and `dot_general`.

The executor should use this abstraction directly instead of forcing owned input
slots.

## Goals

- Add a clean borrowed-input execution path for the whole executor, not only
  `DotGeneral`.
- Avoid cloning caller-owned inputs when the caller passes read-only bindings.
- Preserve the current owned execution path and public behavior.
- Keep reclaim semantics correct: owned temporaries may be returned to backend
  pools, borrowed inputs must never be reclaimed.
- Let benchmarks compare PyTorch and tenferro with no input clone and no output
  clone/copy in the timed loop.

## Non-Goals

- Do not remove the existing `run_many_with_inputs` API.
- Do not require every backend to optimize every `TensorRead` layout immediately.
- Do not introduce a special-case benchmark-only execution path.
- Do not change tensor ownership semantics outside runtime execution.

## Public API

Add a borrowed-input API on `GraphExecutor`:

```rust
pub fn run_many_with_input_reads<'a>(
    &mut self,
    program: &GraphProgram,
    bindings: &[(&TracedTensor, TensorRead<'a>)],
) -> Result<Vec<Tensor>>
```

This API validates placeholder keys, dtypes, and shapes using the same rules as
`run_many_with_inputs`, but it does not clone bound inputs. Existing APIs remain
unchanged:

- `run_many_with_inputs` keeps accepting `&Tensor` and remains non-consuming.
- `eval_exec_ir` keeps accepting owned `Vec<Tensor>`.
- `eval_exec_ir_non_consuming` keeps cloning and is explicitly the legacy
  non-consuming owned path.

## Slot Model

Introduce an internal borrowed-aware slot type:

```rust
enum ExecSlot<'a> {
    Owned(Tensor),
    Read(TensorRead<'a>),
}
```

Rules:

- Program inputs from `run_many_with_input_reads` are initialized as
  `ExecSlot::Read`.
- Program inputs from owned APIs are initialized as `ExecSlot::Owned`.
- Instruction outputs are always stored as `ExecSlot::Owned`.
- Reads from a slot use `TensorRead<'_>` where possible.
- Reads requiring an owned `&Tensor` materialize a borrowed slot into an owned
  tensor at the boundary.
- Output collection returns owned tensors. If a graph output aliases a borrowed
  input, that output is materialized at collection time.

This keeps borrow lifetimes local to one executor call while allowing owned
intermediates to behave exactly as they do today.

## Dispatch

Add read-aware accessors beside the current `get` helper:

- `get_owned` returns `&Tensor` and materializes only at explicit fallback
  boundaries.
- `get_read` returns `TensorRead<'_>` for either owned or borrowed slots.
- shape resolution reads shape metadata from either slot variant.

Read-capable ops should call backend `_read` methods:

- Elementwise unary, binary, and ternary ops use `*_read`.
- Reductions use `reduce_*_read`.
- `DotGeneral` uses `dot_general_read`.
- `DotGeneralWithConj` should gain a read-capable backend method or use a
  localized materialize fallback until backend support is added.

Ops without a suitable `_read` backend API use a single explicit materialization
fallback. The fallback should be visible in the code path, not hidden as an
accidental clone inside input resolution.

## Reclaim Semantics

Last-use reclaim must only reclaim owned slots:

- `ExecSlot::Owned(tensor)` may be taken and passed to `reclaim_buffer`.
- `ExecSlot::Read(_)` is cleared from the slot but is never reclaimed.
- Segment-level reclaim follows the same rule.
- Final outputs are taken only if owned. Borrowed final outputs are materialized
  and returned as owned tensors.

This preserves backend buffer-pool behavior for intermediates while keeping
caller-owned input memory untouched.

## Runtime Paths

The borrowed path should cover the same execution modes as the owned path:

- Single-session execution.
- Unsegmented execution.
- Segmented execution with fused elementwise segments.

Fused elementwise execution currently collects `&Tensor` inputs. It should either
accept `TensorRead` inputs directly or materialize only the specific borrowed
inputs needed for a backend fusion path that cannot consume reads. The preferred
steady state is read-aware fusion input collection.

## Benchmark Policy

For fair framework comparison:

- Tenferro traced benchmarks should use `run_many_with_input_reads`.
- The timed loop should not clone input tensors.
- The timed loop should not clone the output tensor. Use `black_box` on the
  returned tensor or inspect shape/dtype without copying data.
- PyTorch benchmarks should not clone inputs or outputs in the comparable
  baseline.
- Separate benchmark rows may explicitly measure `out=` or preallocated-output
  variants, but those rows must be labeled separately.

This makes `bin_matmul_1024` measure the actual backend operation and executor
overhead, not input/output copying.

## Error Handling

- Borrowed input validation should report the same placeholder mismatch,
  duplicate binding, dtype mismatch, and shape mismatch errors as the current
  owned path.
- Materialization fallback errors should name the op that forced
  materialization.
- Attempting to collect a missing output slot should keep the existing
  `MissingValue` behavior.

## Tests

Add focused tests for:

- `run_many_with_input_reads` executes a simple graph without cloning inputs.
- Borrowed input slots are not reclaimed on last use.
- Owned intermediates are still reclaimed on last use.
- Elementwise, reduction, and `DotGeneral` execute from borrowed inputs.
- Unsupported read path materializes exactly at the fallback boundary.
- A graph whose final output is a borrowed input returns an owned materialized
  tensor.
- Existing `run_many_with_inputs` behavior remains unchanged.

For clone avoidance, use instrumentation that can distinguish input cloning from
normal output allocation, such as backend reclaim stats or a test-only tensor
layout/path that would make unwanted materialization observable.

## Implementation Notes

The implementation should be incremental but keep the architecture coherent:

1. Add `ExecSlot<'a>` and read/owned slot helpers.
2. Add borrowed input resolution for `GraphExecutor`.
3. Thread borrowed slots through single-session, unsegmented, and segmented
   execution.
4. Convert backend dispatch for read-capable ops to `_read`.
5. Add materialization fallbacks for remaining ops.
6. Update benchmarks to use the borrowed-input API and remove timed output
   clones.

The first complete implementation should include elementwise, reduction, and
dot-general coverage rather than adding a narrow `DotGeneral` fast path.
