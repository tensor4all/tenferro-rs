# Issue 1370 shape-guard executor enforcement

## Scope

This stage enforces compiled `ExecProgram::shape_guards` immediately before
execution. It does not add graph-owned constraint scope, AD propagation, or new
extension declarations.

## Context read

- Shared tensor4all repository, Rust, performance, documentation, and testing
  rules.
- `REPOSITORY_RULES.md` public-boundary and test-organization contracts.
- `docs/design/dynamic-symbolic-shapes.md` and
  `docs/spec/backend-contract.md`.
- CodeGraph call paths for every `GraphExecutor` execution entry and the
  extension-owned nonsegmented execution path.

## Design

- One crate-private `validate_shape_guards` walks guards in stored order over a
  small ordered vector of borrowed input-shape slices.
- `GraphExecutor` validates input count first, then guards, before workspace
  allocation, segmentation, backend sessions, or extension dispatch.
- Owned tensor, value-output, borrowed/non-consuming, segmented, nonsegmented
  extension-owned, and zero-instruction roots share the same validator.
- Compile-cache semantic identity remains unchanged. Cache hits restore the
  current program's guard vector, and an executor regression test verifies the
  resulting error uses current provenance.

## TDD evidence

The initial focused test used a counted extension runtime. Before enforcement,
the failing `[7] == 2 * [3]` guard returned `Ok`, incremented the runtime
counter, and failed at `unwrap_err`. After enforcement, the same input returns
the typed `ShapeConstraintViolation` with `7` and `6`, leaves the counter at
zero, and the valid `[6]`, `[3]` case dispatches exactly once.

## Verification

- Focused executor, exec, segment, and extension release tests passed.
- Full `tenferro-runtime` release tests and doctests passed.
- `cargo test --workspace --all-targets --release` passed.
- Workspace and standalone tropical CI-parity clippy passed with warnings
  denied.
- Workspace rustdoc and docs-site checks passed.
- Workspace coverage passed the repository checker for all 153 included files;
  `graph/executor.rs` reached 90.34%, `shape_constraint/solver.rs` 97.19%, and
  `exec.rs` 79.17% against its configured 75% threshold.

## Residual risk

Guard evaluation allocates only a rank-bounded vector of borrowed shape slices.
No tensor data is cloned or materialized. High-level binding resolution may
still perform its pre-existing default-input upload before reaching the
execution-program boundary; guard validation itself introduces no backend or
extension side effect.
