# Eager Tidu Recorder Design

**Date:** 2026-04-24

**Status:** Proposed and approved

## Goal

Simplify `tenferro::EagerTensor` reverse-mode recording after the `tidu-rs`
eager recorder update.

The current eager path records concrete operations by manually constructing
`GradNode` values, stable input aliases, output keys, and saved forward replay
data in the `tenferro` facade crate. `tidu-rs` main now owns that generic
recording logic through `record_eager_op`, `EagerValue`, `EagerOutput`, and
`EagerKeySource`.

## Decision

Update the workspace `tidu` dependency to revision `97e4ae1` and route
`EagerTensor` operation recording through `tidu::record_eager_op`.

`tenferro` will continue to own concrete tensor execution, tensor metadata
registration, eager gradient slots, and backend-aware `BackwardCallbacks`.
`tidu` will own the generic eager AD bookkeeping:

- fresh stable input aliases
- fresh user-visible output keys
- saved forward input and output replay data
- input `GradEdge` construction
- shared `GradNode` construction for multi-output operations

## Scope

This change covers only eager AD recording in the `tenferro` facade crate.
It does not add new AD rules, change `PrimitiveOp::linearize`, change
`PrimitiveOp::transpose_rule`, or modify the eager gradient accumulation
contract.

## Architecture

Add a small `EagerTensor`-local key source that implements
`tidu::EagerKeySource<StdTensorOp>` using the existing `next_input_key()`
allocator.

Introduce a shared helper that:

1. receives the already executed eager output tensors,
2. converts input `EagerTensor`s to `tidu::EagerValue`,
3. calls `tidu::record_eager_op`,
4. registers metadata for the returned output keys and saved replay values,
5. builds `EagerTensor::new_result` values from the returned `EagerOutput`
   metadata.

Single-output n-ary ops and multi-output unary ops should both use this helper.
That leaves `tidu` as the single source of truth for eager AD graph recording.

## Testing

Use existing eager reverse-mode tests as behavioral coverage, then add focused
regression coverage for the simplification:

- a multi-output eager linalg backward path still shares one recorded node and
  seeds the selected output correctly,
- eager einsum AD still records through the n-ary path,
- repeated backward accumulation remains unchanged.

Run focused tests before broader checks:

```bash
cargo test -p tenferro --test eager_tensor
cargo test -p tenferro --test eager_linalg
cargo test -p tenferro --test eager_einsum_ad
cargo fmt --all --check
```
