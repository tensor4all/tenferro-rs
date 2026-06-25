# Layout Overflow And AD One-Like Fixes

## Summary

Fixes #1206 and #1207 in one related bug-fix batch. The changes reject
non-empty logical tensor layouts whose shape product overflows, while preserving
empty-view behavior, and replace the AD analytic one-like shortcut that built
constant one through `Exp(0)`.

## Context Read

- GitHub issues #1206 and #1207
- `REPOSITORY_RULES.md`
- `crates/tenferro-tensor-core/src/lib.rs`
- `crates/tenferro-tensor-core/src/layout.rs`
- `crates/tenferro-tensor/src/types.rs`
- `crates/tenferro-internal-ops/src/ad/analytic.rs`
- `crates/tenferro-internal-ops/src/ad/zeros.rs`
- `crates/tenferro-internal-ops/src/ad/contraction.rs`

## Root Cause

#1206 came from validating only reachable physical bounds in
`TensorLayout::from_parts`. A non-empty broadcast layout with zero strides can
touch only one buffer element while still having an overflowing logical element
count, which made later `n_elements()` overflow paths reachable.

#1207 came from `emit_one_like_fixed` constructing one as
`Exp(anchor + -anchor)`. That made a constant seed depend on an analytic
operation and on the anchor's arithmetic semantics, instead of using the
existing graph constant and broadcast machinery.

## Decision

Add a shared tensor-core logical element-count helper for layout and host-view
validation. The helper is zero-aware: any zero dimension means the logical count
is zero, so empty views with large earlier dimensions remain valid.

Move the existing contraction one-like constant builder into the shared AD
zero-like helper module. Analytic AD rules now build dtype-aware scalar one
constants and broadcast them to the anchor rank, matching the existing
zero-like pattern and avoiding `Exp` as a constant shortcut.

## Verification

- `cargo fmt --all --check`
- `cargo test -p tenferro-tensor-core -p tenferro-tensor -p tenferro-internal-ops`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `/Users/hiroshi/.local/bin/python3.11 scripts/check-docs-site.py`

## Residual Risk

The default system `python3` on this machine is 3.9, while
`scripts/check-docs-site.py` requires Python 3.11 or newer for guide dependency
snippet parsing. The docs-site check passed with the local Python 3.11
interpreter.
