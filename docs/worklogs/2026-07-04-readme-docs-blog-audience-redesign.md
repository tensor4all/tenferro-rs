# 2026-07-04 README / entry docs / blog audience redesign

## Summary

Retargeted the README, docs landing (`docs/index.md`), and getting-started
entry pages at the primary audience settled in the design spec: Rust
developers and researchers doing numerical / scientific computing, arriving
cold from crates.io, search, or GitHub. Unified the project identity line
across surfaces and compressed README narrative sections in favor of the
introduction blog post, which becomes the canonical home of the project
story. A companion PR in `tensor4all/tensor4all.github.io` aligns the blog
post opening (en/ja/zh) with the same one-liner.

## Context read

- `docs/superpowers/specs/2026-07-04-readme-docs-blog-audience-design.md`
  (design spec; decisions and rejected alternatives recorded there)
- `README.md`, `docs/index.md`, `docs/getting-started/*.md`, `docs/_quarto.yml`
- Blog post sources (en/ja/zh) in `tensor4all/tensor4all.github.io`
- `scripts/check-doc-snippets.py`, `.github/workflows/ci.yml`
- crates.io: verified `tenferro-runtime` / `tenferro-cpu` max version 0.2.0

## Chosen design

- One-liner everywhere: "A Rust-native tensor & autodiff stack for
  scientific computing." (README, docs landing, blog opening, GitHub repo
  description). PyTorch/JAX demoted to explanatory anchors.
- README restructured to the reader's decision sequence: identity → quick
  example → "Is tenferro for You?" (including respectful positioning vs
  ndarray, Burn/candle, faer) → API/crate maps → docs links → compressed
  Project section (why / stability / engineering discipline / AI-assisted
  development, each one paragraph with links).
- Getting Started setup switched to crates.io-first: the crates were
  published on 2026-06-23, so the previous local-checkout-first text and the
  "Switch to crates.io once published" block were factual drift.
- `docs/getting-started/core-concepts.md` intentionally unchanged (already
  task-oriented; no stale positioning).

## Residual risks

- `cargo llvm-cov` was not rerun for this diff: it changes Markdown only, no
  Rust source or tests, so coverage is identical to `origin/main`.
- The docs site rebuilds from `main`; until the companion website PR merges,
  the blog opening briefly lags the README one-liner. Both PRs are tracked
  to merge in the same session.
- README line target (≈170) is approximate; the delivered README is 245
  lines (down from 329). The acceptance criterion is the first-screen
  decision flow — the quick example starts at line 21 — not the exact count.
