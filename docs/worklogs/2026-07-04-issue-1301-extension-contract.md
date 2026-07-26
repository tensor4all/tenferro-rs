# Extension contract split work log

Date: 2026-07-04

## Session summary

Issue #1301 changes the extension contract from a mandatory context-free
execution hook and one combined AD rule trait to explicit capabilities:

- `ExtensionOp::eager_execute` was removed from the required trait surface.
- `HostReference` and `ExtensionOp::host_reference()` now model optional
  host/reference execution.
- `HostReferenceRuntime<B>` adapts that optional capability into a registered
  runtime when an owner deliberately chooses reference execution.
- `ExtensionAdRule` was split into `SemanticLinearizeRule`,
  `SemanticLinearTransposeRule`, and `SemanticPrimalVjpRule`.
- `SemanticExtensionRuleSet` now stores and checks duplicates per AD role.
- Extension transpose dispatch uses linear-transpose rules for linearized
  helper ops and primal-VJP rules only for primary primal fallback.

## Context read

| Source | Why it was read | Decision impact |
| --- | --- | --- |
| `REPOSITORY_RULES.md` | Confirm extension, docs, and test expectations. | Kept the change scoped to extension contracts and updated normative docs. |
| Shared tensor4all rules | Confirm common Rust, performance, docs, and numerical rules. | Avoided new global state and kept host execution as an explicit runtime opt-in. |
| Issue #1301 | Source of the requested breaking contract change. | Drove the required API names, role split, and dispatch behavior. |
| `docs/spec/extension-op.md` and `docs/spec/backend-contract.md` | Locate normative extension dispatch wording. | Updated the contract to remove mandatory context-free execution and document role-specific AD dispatch. |
| In-tree extensions under `crates/` and `ext/` | Find implementers of the old trait surface. | Migrated einsum, linalg, FFT, tropical, and sparse to host-reference/rule-role APIs. |

## Decisions made

- **Host/reference execution is an optional capability.** Backend-only
  extension families can implement `ExtensionOp` without any host execution
  hook. `HostReferenceRuntime` reports `NoHostReference` if it is registered
  for such a family.
- **Runtime registration remains mandatory for execution.** Removing
  `eager_execute` does not reintroduce silent fallback. Owners must register
  a concrete runtime or the host-reference adapter.
- **AD rule roles are independent.** A family may register linearize,
  linear-transpose, and primal-VJP rules separately. Duplicate checks are per
  `(family_id, role)`.
- **Primary fallback uses primal VJP for extensions.** Standard core ops still
  receive the linearized role in the usual linear-transpose path; extension
  primal fallback dispatches `OperationRole::Primary` so it reaches the
  primal-VJP registry.
- **Standalone examples should not require `tenferro-ad` only for rule types.**
  The sparse extension now imports role traits from the internal ops surface,
  matching tropical, so its `autodiff` feature does not accidentally compile
  `tenferro-ad` without a CPU backend feature.

## Verification performed

- `cargo check -p tenferro-internal-ops --features autodiff`
- `cargo check -p tenferro-runtime`
- `cargo check -p tenferro-ad`
- `cargo check -p tenferro-einsum --features autodiff`
- `cargo check -p tenferro-linalg --features autodiff`
- `cargo check -p tenferro-fft`
- `cargo check --manifest-path ext/tropical/Cargo.toml --features autodiff`
- `cargo check --manifest-path ext/sparse/Cargo.toml --features autodiff`
- `cargo test -p tenferro-internal-ops --features autodiff ext_op -- --nocapture`
- `cargo test -p tenferro-runtime --test extension_runtime -- --nocapture`
- `cargo test -p tenferro-ad --test extension_op -- --nocapture`
- `cargo test -p tenferro-einsum --features autodiff extension -- --nocapture`
- `cargo test -p tenferro-linalg --features autodiff extension -- --nocapture`
- `cargo test -p tenferro-linalg --features autodiff transpose --lib -- --nocapture`
- `cargo test -p tenferro-fft -- --nocapture`
- `cargo test -p tenferro-fft --features autodiff --lib fft_transpose_rule_respects_inactive_linearized_input -- --nocapture`
- `cargo test --manifest-path ext/sparse/Cargo.toml --features autodiff -- --nocapture`
- `cargo test --manifest-path ext/tropical/Cargo.toml --features autodiff -- --nocapture`
- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `git diff --check`
- `rg -n "ExtensionAdRule|register_rule\\(|with_rule\\(|lookup_rule\\(|is_rule_registered\\(|eager_execute|ExtensionOp::eager_execute|op\\.eager_execute" docs/guides docs/spec crates ext -g '*.md' -g '*.rs'`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`

## Remaining risks

- No known local implementation risks remain after the checks above. External
  PR CI may still expose environment-specific failures.
