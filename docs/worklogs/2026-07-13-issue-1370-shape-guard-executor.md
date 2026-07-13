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
- High-level graph runs first select bound, default, or deferred-zero input
  metadata in program order, validate input specs and guards, and only then
  clone, upload, or allocate the selected values. Owned and borrowed wrappers
  share this source-selection step so their precedence cannot drift.
- Compile-cache semantic identity remains unchanged. Cache hits restore the
  current program's guard vector, and an executor regression test verifies the
  resulting error uses current provenance.

## TDD evidence

The initial focused test used a counted extension runtime. Before enforcement,
the failing `[7] == 2 * [3]` guard returned `Ok`, incremented the runtime
counter, and failed at `unwrap_err`. After enforcement, the same input returns
the typed `ShapeConstraintViolation` with `7` and `6`, leaves the counter at
zero, and the valid `[6]`, `[3]` case dispatches exactly once.

A follow-up counted-backend regression first failed because a scalar default
was uploaded before the guard rejected the run. After metadata preflight, bad
default-backed owned/value and borrowed/value wrappers report zero uploads and
zero dispatches, while good runs upload and dispatch once. An explicit binding
test confirms guards evaluate the selected runtime shape, and direct tests of
the extension execution context and backend-cache lowering helper cover both
owner-scoped execution paths.

The private owned and borrowed materializers accept an internal deferred-zero
factory. Module-local regressions inject counted and failing factories into
that exact production seam: failing guards make zero factory calls, passing
guards make one call with the selected dtype and shape, and factory errors are
observable only after metadata and guard validation. The counted backend also
confirms rejected public wrappers never enter a backend session, while accepted
wrappers enter exactly once.

## Verification

- Focused executor, exec, segment, and extension release tests passed.
- Full `tenferro-runtime` release tests and doctests passed.
- `cargo test --workspace --all-targets --release` passed.
- Workspace and standalone tropical CI-parity clippy passed with warnings
  denied.
- Workspace rustdoc and docs-site checks passed.
- Workspace coverage passed the repository checker for all 153 included files;
  `graph/executor.rs` reached 91.47% and `exec.rs` 79.51% against its configured
  75% threshold. Focused coverage also recorded two calls to the counted-backend
  instantiation of `eval_exec_ir_with_backend_cache`.

## Residual risk

Guard evaluation allocates only a program-input-count-bounded vector of borrowed
shape slices.
No tensor data is cloned, uploaded, zero-filled, or otherwise materialized
until input metadata and all retained guards pass. High-level graph wrappers
therefore introduce no backend or extension side effect on guard failure.
