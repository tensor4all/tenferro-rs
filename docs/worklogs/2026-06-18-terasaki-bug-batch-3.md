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
- Added `ShapeGuardContext::rank_of()` and audited AD graph-emission rules so
  rank-only or runtime-shape-source paths do not call exact-shape metadata.
  Reduction AD, reshape/broadcast transpose, embed-diag transpose, and scatter
  inverse-gather construction now accept upper-bound metadata where the emitted
  graph can read the actual runtime shape. Linalg solve AD now uses rank
  metadata for rank-only validation, and broadcast transpose detects
  upper-bound singleton axes before restoring the runtime input shape.
- Made extension runtimes implement `execute_reads` explicitly; the macro no
  longer permits an omitted read executor.
- Removed the `tenferro-internal-ops/gpu` umbrella feature alias; downstream
  code uses `cuda`, `webgpu`, or `rocm` explicitly.
- Removed the no-op `AdContextBuilder::with_core_rules()` compatibility-style
  method. Core primitive AD rules are always available; `AdContextBuilder`
  now represents only owned extension rule sets.
- Fresh-agent repository-rule audit found one remaining stale
  `with_core_rules()` call in `ext/tropical`; the external extension tests now
  use the canonical builder contract.
- Updated the README architecture SVG so `tenferro-xla` appears as the L3
  StableHLO/PJRT peer executor, matching the crate table and architecture docs.
- Reworked `CI_gpu.yml` so same-repository PRs wait for the aggregate
  `repository rules review` gate and all cheap non-GPU checks before building
  the CUDA test archive or starting the `ubuntu-gpu` runner. This avoids
  spending GPU runner time on PR revisions already rejected by repository
  policy, lint, docs, coverage, or CPU tests, while still allowing the
  maintainer-controlled no-LLM review path to satisfy the policy gate.
- Hardened the repository-rules review bot so malformed or schema-drifting LLM
  responses produce a structured review finding instead of a Python traceback.
  The prompt now caps findings per chunk and explicitly suppresses known
  false-positive classes around compatibility shims, Rust `?` propagation,
  doctest hidden result tails, and private-helper dead-code guesses.
- Fixed stale `TracedTensor::concrete_shape` rustdoc that still said symbolic
  dimensions panic even though the API now returns a typed error.
- Continued the public panic API cleanup after the interrupted Codex process:
  removed `EagerTensor::data()`, made eager tensor import/materialization paths
  return `Result`, kept `EagerPrimitiveBuilder` crate-private and fallible, and
  removed hidden eager graph/broadcast helper exports that were only sibling
  reach-through.
- Replaced the ad hoc untracked broadcast helper with one narrow documented
  extension contract, `tenferro_ad::extension::adopt_untracked_eager_value`,
  so operation-family crates can adopt lazy backend `TensorValue`s without
  exposing operation-specific eager internals.
- Closed the prototype `einsum_whole_program_untracked` public eager API; it is
  now test-only, while user-facing eager einsum remains through `einsum`,
  `einsum_subscripts`, and `tensordot`.
- Updated CUDA-feature-only tests for the fallible traced/tensor APIs so the
  GPU test archive compile catches API drift instead of relying only on
  non-CUDA workspace builds.

## Rule Updates

- `REPOSITORY_RULES.md`, `bugfix-pr.md`, and `repository-remediation.md` now
  prefer root-cause API redesign over compatibility shims unless compatibility
  is explicitly required.
- The remediation workflow requires same-root-cause searches, source comments
  for intentional false positives, and rule inventory before adding new audit
  rules.
- Standard extension rules now disallow new process-global registration shims
  unless a maintainer explicitly approves a legacy bridge.
- `REPOSITORY_RULES.md` now requires AD graph-emission rules to distinguish
  rank, exact extents, conservative extents, and runtime shape sources instead
  of using exact-shape queries as a default.
- `REPOSITORY_RULES.md` now records CI cost discipline: expensive GPU or
  larger-runner lanes must sit behind cheaper repository-policy and non-GPU
  checks.
- The review-bot prompt now encodes the corresponding audit heuristics so future
  repository-rule runs do not rediscover compatibility-shim and doctest
  false positives as blockers.
- `REPOSITORY_RULES.md` now states that `#[doc(hidden)] pub` is not privacy and
  that public infallible accessors/constructors must not remain when
  materialization, metadata registration, validation, backend transfer, or lock
  state can fail.
- Source-contract tests now forbid reintroducing the removed eager public
  panic/accessor helpers and the closed eager einsum prototype helper.

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
- `cargo test -p tenferro-internal-ops upper_bound_input_metadata -- --nocapture`
- `cargo test -p tenferro-internal-ops transpose_broadcast_reduces_upper_bound_singleton_input_axes -- --nocapture`
- `cargo test -p tenferro-linalg --lib --features autodiff triangular_solve -- --nocapture`
- `cargo test -p tenferro-linalg --lib --features autodiff cholesky_jvp_uses_rank_when_input_metadata_is_upper_bound -- --nocapture`
- `cargo test -p tenferro-ad --test checkpoint_truncate_integration -- --nocapture`
- `cargo test -p tenferro-ad --test extension_op -- --nocapture`
- `cargo test -p tenferro-gpu --test cubecl_launch_contract -- --nocapture`
- `cargo test --manifest-path ext/tropical/Cargo.toml --features autodiff --test tropical_ad -- --nocapture`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `/opt/homebrew/bin/python3.12 scripts/test-doc-consistency.py`
- `python3 scripts/test-repository-rules-review.py`
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD --output-json /tmp/review-debug-after4.json --no-dotenv`
- `cargo check -p tenferro-runtime --all-targets`
- `cargo check -p tenferro-einsum --features autodiff --all-targets`
- `cargo check -p tenferro-fft --features autodiff --all-targets`

Additional local verification after the public API cleanup:

- `cargo fmt --all --check`
- `cargo test -p tenferro-ad --test fallible_api eager_public_tensor_accessors_are_fallible_source_contract -- --nocapture`
- `cargo test -p tenferro-ad eager_backward_transpose_runs_without_extension_executor -- --nocapture`
- `cargo test -p tenferro-ad eager_backward_zero_from_exact_metadata_covers_dtypes_and_dynamic_none -- --nocapture`
- `cargo test -p tenferro-einsum --test public_surface_contract -- --nocapture`
- `cargo test -p tenferro-einsum --features autodiff whole_program_untracked_matches_per_op_nary_result -- --nocapture`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
- `git diff --check`

Additional local verification while babysitting PR #1125 CI:

- `python3 scripts/test-doc-consistency.py`
- `cargo test -p tenferro-ad --features cuda --test gpu_f32_fusion --no-run`
- `CUDARC_CUDA_VERSION=12080 cargo test --no-run --package tenferro-gpu --package tenferro-ad --package tenferro-linalg --features cuda --release`
- `cargo fmt --all --check`
- `git diff --check`

No successful cloud GPU runtime check had completed at the time of the first
local verification; GPU coverage there was local source-contract tests plus
CUDA feature compilation. During PR CI babysitting, the no-LLM repository review
path was used because the base `pull_request_target` review bot could not use
this PR's malformed-LLM-JSON parser fix. The PR now gates CUDA archive building
and GPU runner use behind the aggregate repository review gate plus cheap
non-GPU checks.
