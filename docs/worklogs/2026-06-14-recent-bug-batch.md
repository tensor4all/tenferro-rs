# Recent Bug Batch

## Summary

Fixed the recent panic/invalid-behavior bug batch covering issues #1027 through
#1033 on one branch. The changes keep the public behavior fallible at API
boundaries, add focused regression tests, and avoid broad rewrites outside the
affected layers.

## Context Read

- `AGENTS.md`, `REPOSITORY_RULES.md`, `CONTRIBUTING.md`
- `ai/contribution-workflows/bugfix-pr.md`
- Recent GitHub issue bodies for #1027, #1028, #1029, #1030, #1031, #1032, and
  #1033
- Affected modules in `tenferro-ad`, `tenferro-runtime`,
  `tenferro-internal-ops`, `tenferro-cpu`, `tenferro-gpu`, `tenferro-einsum`,
  `tenferro-linalg`, and `tenferro-fft`

## Decisions

- Removed eager `Add`/`Mul`/`Neg` operator overloads instead of preserving
  panic-based convenience operators. Eager callers now use existing fallible
  methods such as `add`, `mul`, and `neg`.
- Made traced `jvp`, `vjp`, and optional variants return `Result`, preserving
  optional semantics for inactive outputs without panicking.
- Added compact hand-written `Debug` implementations for `EagerTensor` and
  `TracedTensor`, and recorded the public-type `Debug` rule in
  `REPOSITORY_RULES.md`.
- Propagated `InvalidCompiledGraph` through graph compilation and shape
  inference for missing slot metadata and incompatible broadcast dimensions.
- Validated `DotGeneral` AD transpose dimensions at the rule boundary and
  returned `ADRuleError` rather than panicking.
- Added explicit overflow rejection for CubeCL/WebGPU launch cube counts instead
  of truncating to `u32`.
- Replaced raw LAPACK provider pointer `transmute` with a checked helper using
  `transmute_copy` after a size check. This keeps the unavoidable FFI conversion
  localized and error-producing.

## Deferred

- A full repository-wide audit of public types missing `Debug` is intentionally
  deferred to a follow-up issue. This PR only adds `Debug` where needed by the
  current fallible API work.
- Existing shape-inference config assertions for gather/slice/pad are outside
  this batch's issue scope and remain unchanged.

## Verification

- `cargo fmt --all --check`
- `cargo check -p tenferro-runtime --tests`
- `cargo check -p tenferro-ad --tests`
- `cargo check -p tenferro-internal-ops -p tenferro-cpu -p tenferro-gpu -p tenferro-einsum -p tenferro-linalg -p tenferro-fft --tests`
- `cargo check -p tenferro-ad -p tenferro-fft --features tenferro-fft/autodiff --tests`
- `cargo check -p tenferro-tutorial-code --bins`
- `cargo check --workspace --all-targets`
- `cargo test --workspace --release`

Coverage, clippy parity, generated docs, and docs-site checks were not run in
this time-focused batch pass.
