# Einsum Views, Into Outputs, And Layout Helpers

## Summary

Implemented the concrete tensor API extensions for issues #1241-#1244:
borrowed typed strided view inputs for typed einsum, preallocated output
execution, reusable concrete plan `execute_into` entry points, and tensor
layout inspection helpers.

## Issue Ledger

- #1241: `TypedTensorEinsumExt` now accepts borrowed `TypedTensorView` inputs,
  including strided complex views.
- #1242: concrete einsum exposes `einsum_into`/`einsum_read_into` entry points
  that write into caller-provided output storage.
- #1243: `ConcreteEinsumPlan` exposes `execute_into`,
  `execute_typed_into`, and `execute_read_into` for repeated fixed-expression
  workloads.
- #1244: tensor read/write/view surfaces expose layout inspection and
  column-major assertion helpers.

## Context Read

- `AGENTS.md` and `REPOSITORY_RULES.md`
- shared tensor4all common, Rust, performance, numerical, and docs/test rules
- `docs/spec/api-conventions.md`
- `docs/design/tensor.md`
- `docs/design/einsum.md`
- `docs/worklogs/2026-06-27-concrete-einsum-public-api.md`
- existing tensor, CPU dot-general, and concrete einsum tests

## Design Review

An external Claude plan review was run before implementation. The usable
review feedback was design-level because Claude's own file tools could not
read the workspace in the first attempts. The incorporated points were:

- validate output dtype and shape before allocation or writes;
- make aliasing and output-layout behavior explicit;
- require evidence that the GEMM-compatible path reaches a direct backend
  `*_into` call before owned-result fallback;
- justify the dtype-erased mutable view boundary rather than adding only a
  typed-only output API.

## Decisions

- Added `TensorViewMut<'a>` and `TensorWrite<'a>` as the dtype-erased mutable
  output surface. This mirrors the existing `TensorView<'a>` and
  `TensorRead<'a>` read-side design while keeping typed callers on
  `&mut TypedTensor<T>` where static dtype information is already available.
- Kept `einsum_into` as a validating write API, not a resizing API. If the
  destination dtype or shape differs from the expression result, the call
  returns an error before writing.
- Added layout helpers to owned tensors, read views, mutable views, and
  read/write erased wrappers. The helpers report shape, strides, offset, and
  column-major compactness without materializing data.
- Routed GEMM-compatible binary contractions such as `ij,jk->ik` through
  `TensorDot::dot_general_read_into`. On the CPU faer provider this constructs
  a mutable strided output view over caller storage and writes the final result
  directly.
- Kept the general einsum fallback simple: it may allocate intermediates and,
  for non-direct cases, may allocate an owned final result before copying into
  the caller's output. The public guarantee is preallocated destination support,
  with allocation-free final output targeted at direct GEMM-compatible paths.
- Left the CPU BLAS direct-into provider as a fallback path for this branch.
  The default CPU faer provider has the direct output path; BLAS-specific
  direct output can be added behind the same backend hook later.

## Alternatives Rejected

- A typed-only `einsum_into(&mut TypedTensor<T>)` surface for all callers:
  rejected because it would leave dtype-erased `Tensor` users without a
  symmetric preallocated-output API.
- Resizing the destination tensor on mismatch: rejected because it hides
  allocation in an API whose purpose is caller-owned output storage and makes
  view destinations impossible to reason about.
- A separate public "compiled typed einsum" type: rejected in favor of adding
  `execute_into` methods to the existing `ConcreteEinsumPlan` contract.

## Verification

Incremental TDD checks were run while implementing the branch:

- `cargo test -p tenferro-tensor types_tests -- --nocapture`
- `cargo test -p tenferro-cpu dot_general_read_into -- --nocapture`
- `cargo test -p tenferro-cpu strided_dot_into_does_not_acquire_output_buffer -- --nocapture`
- `cargo test -p tenferro-einsum einsum_into -- --nocapture`
- `cargo test -p tenferro-einsum concrete_einsum_plan_execute -- --nocapture`
- `cargo test -p tenferro-einsum concrete`
- `cargo test -p tenferro-einsum borrowed_strided_complex_views -- --nocapture`
- `cargo test -p tenferro-einsum gemm_fast_path_dispatches -- --nocapture`

Final branch verification is recorded in the PR once the full checklist has
passed.

## Remaining Risks

- The no-final-allocation evidence for GEMM-compatible `einsum_into` is based
  on the faer direct path and source contracts. It is not a global allocator
  count for every contraction expression.
- CPU BLAS currently falls back to owned-result execution before copying into
  the output destination.
- Safe Rust borrowing prevents ordinary owned tensor input/output aliasing at
  the public API boundary, but backend-shared storage aliasing is not
  exhaustively detected at runtime.
