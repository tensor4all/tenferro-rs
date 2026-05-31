# tidu AD rule contract migration work log

Issue: <https://github.com/tensor4all/tidu-rs/issues/26>

Date: 2026-05-31

## Session summary

This change migrates tenferro's active AD rule contract imports from the
former `chainrules` crates to the rule contract now exported by `tidu`.

The companion `tidu-rs` PR first absorbed `ADKey`, `ADRuleError`,
`ADRuleResult`, `DiffPassId`, and `PrimitiveOp` into `tidu` and exposed them
from the crate root. This tenferro PR then:

- removes workspace `chainrules` and `chainrules-core` dependencies
- wires optional `autodiff` features in operation crates through `tidu`
- updates source and test imports from `chainrules_core` to `tidu`
- pins `tidu` to the merged `tidu-rs` commit that contains the rule contract

No AD rule semantics were changed in tenferro.

## Context read

| Source | Why it was read | Decision impact |
| --- | --- | --- |
| `AGENTS.md` | Confirm repository workflow and PR requirements. | Kept workspace dependencies centralized and added this work log before PR creation. |
| `REPOSITORY_RULES.md` | Review AD source-of-truth, standard extension, and work-log rules. | Preserved graph-level `linearize` and `transpose_rule` terminology and avoided reintroducing ChainRules-style APIs. |
| `Cargo.toml` and crate manifests | Locate active dependency declarations and feature boundaries. | Replaced only active `chainrules` dependencies with `tidu` wiring. |
| `tenferro-internal-ops/src/ad/` | Check the owning AD rule contract. | Kept implementations and rule registry behavior unchanged while changing the imported trait/error types. |
| `ext/tropical/` | Check the out-of-workspace extension manifest and AD import path. | Added its own optional `tidu` git dependency because it has a separate manifest and lockfile. |
| `tidu-rs` PR #27 | Confirm the merged commit that exports the absorbed rule contract. | Pinned tenferro manifests to `fe7fb27cf34338d3214fa9f1cc56ce4d0691626e`. |

## Decisions made

- **`tidu` is now the shared AD rule contract dependency.** Active tenferro
  crates import `ADKey`, `ADRuleError`, `ADRuleResult`, `DiffPassId`, and
  `PrimitiveOp` from `tidu`.
- **Feature ownership stays with operation crates.** `tenferro-einsum`,
  `tenferro-linalg`, `tenferro-fft`, and `ext/tropical` keep their
  `autodiff` features and depend on `tidu` only when that feature is enabled.
- **No semantic AD rewrite in this PR.** The tenferro rule implementations,
  registry behavior, and tests continue to exercise the same graph-level
  rule paths.

## Rejected or deferred alternatives

- **No compatibility aliases for `chainrules_core`.** The correct owning
  abstraction is the `tidu` rule contract, so downstream crates should import
  it directly instead of carrying aliases.
- **No historical docs cleanup.** `docs/plans/` and old coverage artifacts
  contain stale `chainrules` mentions by design; repository rules say those
  historical records may contradict the current API.
- **No broader AD API rename.** This migration follows the existing
  `linearize` / `transpose_rule` model and does not design a new pullback API.

## Verification performed

- `cargo fmt --all --check`
- `cargo test --release -p tenferro-internal-ops --features autodiff`
- `cargo test --release -p tenferro-ad`
- `cargo check -p tenferro-einsum --features autodiff`
- `cargo check -p tenferro-linalg --features autodiff`
- `cargo check -p tenferro-cpu --features cpu-faer -p tenferro-fft --features autodiff`
- `(cd ext/tropical && cargo check --features autodiff)`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `python3 scripts/check-ad-boundaries.py`
- `python3 scripts/check-linalg-ad-boundaries.py`
- `scripts/check-tensor-core-deps.sh`
- `rg -n "chainrules|1694275361d2f16abfb3f25ad7941407c88b7c09" Cargo.toml Cargo.lock ext/tropical/Cargo.toml ext/tropical/Cargo.lock tenferro-*/Cargo.toml ext/tropical/src tenferro-*/src tenferro-*/tests`
- `git diff --check`

## Remaining risk

- Full workspace coverage, docs, and docs-site checks were not run locally for
  this narrow dependency migration. The PR should rely on CI for the complete
  repository gate.
