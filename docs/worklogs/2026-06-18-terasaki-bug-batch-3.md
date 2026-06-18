# Terasaki Bug Batch 3

## Summary

Fixed the #1111-#1119 `terasakisatoshi` bug batch as one non-squash PR
candidate, then used the same audit rules to remove legacy compatibility APIs
that would keep the underlying design problems alive.

The final direction is intentionally compatibility-breaking: public panic shims,
process-global AD registration helpers, ambiguous feature aliases, stale root
operation re-exports, and compatibility shape/accessor aliases were removed
instead of preserving them behind `try_*` or alias wrappers.

## Classification Ledger

- Fixed in this PR: #1111, #1112, #1113, #1114, #1116, #1117, #1118, and #1119.
- Docs/enhancement outside this bug-fix batch: #1115 and #1054.
- Related design cleanup included in scope: `tidu` AD rule error contract,
  tensor public panic shims, explicit extension AD rule sets, required
  `execute_reads`, exact-vs-bound tensor metadata, and concrete backend feature
  names.

## Same-Root-Cause Sweep

- Replaced public panic/defaulting APIs with typed `Result`-returning canonical
  operations instead of adding `try_*` compatibility escapes.
- Removed legacy public aliases: root operation re-exports, `Cubecl*` public
  backend names, `Extension*Trait` names, `TropicalEinsumKind`,
  `MetadataScope`, owned-tensor `as_physical_slice*`, and
  `is_contiguous_col_major`.
- Removed process-global extension AD rule registration and made callers pass
  explicit rule sets or contexts.
- Split scratch-buffer acquisition into full-overwrite uninitialized buffers
  and read-before-write zeroed buffers, with one-line invariants at
  uninitialized acquisitions.
- Made `TensorMeta` store only extents; callers now choose `exact_shape()` or
  `bound_shape()` explicitly.
- Made extension runtimes implement `execute_reads` explicitly; the macro no
  longer permits an omitted read executor.
- Removed the `tenferro-internal-ops/gpu` umbrella feature alias; downstream
  code uses `cuda`, `webgpu`, or `rocm` explicitly.

## Rule Updates

- `REPOSITORY_RULES.md`, `bugfix-pr.md`, and `repository-remediation.md` now
  prefer root-cause API redesign over compatibility shims unless compatibility
  is explicitly required.
- The remediation workflow requires same-root-cause searches, source comments
  for intentional false positives, and rule inventory before adding new audit
  rules.
- Standard extension rules now disallow new process-global registration shims
  unless a maintainer explicitly approves a legacy bridge.

## Verification

Local verification before push:

- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `cargo check --workspace --all-targets`
- `cargo check -p tenferro-cpu --features provider-inject --all-targets`
- `cargo check -p tenferro-linalg --features provider-inject --all-targets`
- `cargo check -p tenferro-gpu --features cuda`
- `cargo check -p tenferro-ad --features cuda`
- `cargo check -p tenferro-linalg --features cuda,autodiff`
- `cargo check -p tenferro-einsum --features autodiff --all-targets`
- `cargo test -p tenferro-internal-extension-macros`
- `cargo test -p tenferro-internal-ops metadata`
- `cargo test -p tenferro-tensor accessors`
- `cargo test -p tenferro-tensor-core view_validation_reports_rank_permutation_and_slice_errors`
- `cargo test -p tenferro-runtime --test public_surface_contract`
- `cargo test -p tenferro-gpu --test public_surface_contract`
- `cargo test -p tenferro-cpu --test provider_feature_contract`

No cloud GPU runtime checks were used; GPU coverage here is local source
contract tests plus CUDA feature compilation.
