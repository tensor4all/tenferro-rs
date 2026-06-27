# Concrete Einsum Public API

## Summary

Implemented the accepted concrete einsum API stack for issues #1235-#1238:
dtype-erased `Tensor` execution, typed `TypedTensor<T>` execution including
complex tensors, `TensorRead` view execution, prepared concrete plans, and the
matching guide/spec/design documentation.

## Context Read

- `AGENTS.md` and `REPOSITORY_RULES.md`
- shared tensor4all common repository/docs rules
- `docs/spec/api-conventions.md`
- `docs/spec/operation-categories.md`
- `docs/worklogs/2026-06-19-operation-surface.md`
- `docs/design/einsum.md`
- existing `tenferro-einsum` eager, typed eager, traced, and extension tests

## Decisions

- Exposed concrete einsum through crate-root extension traits:
  `TensorEinsumExt`, `TypedTensorEinsumExt`, and `TensorReadEinsumExt`.
- Kept public module free functions out of the release API. The implementation
  uses a private `concrete` module and re-exports only the supported root
  traits/type.
- Used unsuffixed `einsum` for compact borrowed `Tensor` and `TypedTensor`
  inputs. Used `einsum_read` only for the `TensorRead` surface.
- Made `[&lhs, &rhs].einsum(...)` and `[lhs_read, rhs_read].einsum_read(...)`
  the ergonomic receiver forms by implementing traits for both slices and
  fixed-size arrays.
- Added `ConcreteEinsumPlan` as the prepared-plan public contract. Preparation
  captures input count, dtype, shape, and the optimized contraction tree;
  execution validates later inputs against that metadata before running the
  stored tree.
- Reused the existing eager executor and read-capable execution path instead of
  duplicating contraction logic or adding a new backend hook.

## Alternatives Rejected

- Public `tenferro_einsum::einsum(...)` and `einsum_read(...)` free functions:
  rejected because the current operation-surface contract uses crate-root
  extension traits for extension-family crates.
- Public consuming `Vec<Tensor>` APIs: deferred because the accepted user-facing
  contract centered on borrowed compact tensors, typed tensors, read views, and
  prepared execution. The internal consuming path remains test-covered.
- A separate cached execution function family: rejected in favor of
  `ConcreteEinsumPlan`, which gives repeated execution a single owner and
  validation boundary.

## Verification

- `cargo test -p tenferro-einsum concrete -- --nocapture` first failed on the
  missing public API names, then passed after implementation.
- `cargo test -p tenferro-einsum`
- `cargo test -p tenferro-einsum --features autodiff`
- `cargo test -p tenferro-einsum -p tenferro-fft`
- `cargo test -p tenferro-einsum -p tenferro-fft --features tenferro-einsum/autodiff,tenferro-fft/autodiff`
- `cargo fmt --all --check`
- `python3 scripts/check-doc-snippets.py --check`
- `python3 scripts/check-guide-dependency-snippets.py`
- `python3 scripts/check-api-consistency.py --fail-on-findings`
- `python3 scripts/check-operation-categories.py --fail-on-findings`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
- `git diff --check`

## Remaining Risk

GPU-specific concrete execution was not run locally. The API reuses the same
backend-explicit tensor execution path as existing concrete/eager einsum
helpers, so CUDA coverage remains tied to the existing GPU test lanes.
