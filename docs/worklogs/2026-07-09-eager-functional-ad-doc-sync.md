# Eager Functional AD Documentation Sync

## Summary

Synchronized public documentation and the architecture SVG with the current
eager AD implementation. `EagerRuntime` already exposes functional `grad`,
`vjp`, and `jvp`; the stale docs made eager AD look like `backward()` only and
made traced graphs look like the only transform surface.

## Context Read

- `AGENTS.md`, shared tensor4all rules, and `REPOSITORY_RULES.md`.
- `EagerRuntime::{grad,vjp,jvp}` and the eager functional transform path in
  `crates/tenferro-ad/src/eager.rs` and `crates/tenferro-ad/src/eager/functional.rs`.
- Existing eager JVP and HVP tests in `crates/tenferro-ad/src/eager/tests.rs`.
- Active user docs under `README.md`, `docs/getting-started/`, `docs/guides/`,
  `docs/spec/`, `docs/tutorials/`, and `docs/assets/tenferro-architecture.svg`.

## Decisions

- Keep the API unchanged because the implementation already supports eager
  functional JVP.
- Update docs to distinguish stateful eager `backward()` from functional eager
  transforms, and to describe traced transforms as the compiled graph reuse
  surface rather than the only `grad`/`vjp`/`jvp` surface.
- Add a source-contract check to `scripts/check-docs-site.py` so README,
  getting-started docs, key guides/specs, and the architecture SVG cannot drift
  back to the older wording unnoticed.
- Keep historical `docs/plans/` and `docs/reference/` untouched, but update
  active design/spec notes that asserted stale eager limitations.
- Change the SVG font CSS from shorthand `system-ui` to explicit `DejaVu Sans`
  properties so local SVG-to-PNG conversion renders the diagram reliably.

## Verification

- `git diff --check`
- `cargo fmt --all --check`
- `python3 scripts/check-docs-site.py --quiet`
- `python3 scripts/check-docs-site.py`
- `cargo test -p tenferro-ad jvp`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `cargo doc --workspace --no-deps`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD --worktree --output-json /tmp/repository-rules-review-worktree.json`
- `convert -background white -alpha remove -alpha off docs/assets/tenferro-architecture.svg /tmp/tenferro-architecture-white.png`

## Residual Risks

- The docs still keep the traced tutorial focused on compiled graph AD. That is
  intentional; eager functional examples live in the eager operations guide.
