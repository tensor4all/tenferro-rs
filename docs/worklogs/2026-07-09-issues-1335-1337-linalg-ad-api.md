# Issues 1335-1337 Linalg AD API

## Summary

This work log records the single-PR implementation for GitHub issues #1335,
#1336, and #1337. The changes clarify linalg AD support routes, make traced
VJP prefer registered custom primal VJP rules, add rank-preserving axis slicing
builders, and replace decomposition epsilon helper methods with options-based
SVD/Eigh APIs.

## Context Read

- `AGENTS.md`
- `REPOSITORY_RULES.md`
- Shared tensor4all rules: common docs/tests, Rust performance, Rust numerical
- GitHub issues #1335, #1336, #1337
- Existing linalg AD manifest, traced/eager linalg APIs, runtime/eager shape
  packing helpers, and linalg extension op pruning

## Classification Ledger

| Issue/finding | Classification | Resolution |
| --- | --- | --- |
| #1335 AD support manifest routes | API clarification / behavior fix | Added user-visible JVP/VJP mode support with explicit implementation routes. Linalg SVD/Eigh remain supported via `linearize -> linear_transpose`; solve-like paths are marked as `LinearizeThenCustomLinearTranspose`. |
| #1335 custom VJP precedence | AD dispatch fix | Traced VJP now attempts registered custom primal VJP rules before canonical linearize-transpose. It falls back only on `Unsupported`; malformed/internal rule errors are surfaced directly. |
| #1336 axis slicing builder | Public API addition | Added traced and eager `slice_axis`, `slice_builder`, and builder range/step/take operations. Added traced `take_axis` as a `usize` wrapper over `index_select`. |
| #1336 SVD truncation examples | Documentation | Documented fixed-rank truncation as `svd()` plus `slice_axis`/`take_axis`; dynamic-shape tutorial remains based on `dynamic_truncate`. |
| #1337 decomposition options | Public API replacement | Added `SvdOptions`, `SvdGauge`, `EighOptions`, and `DEFAULT_DECOMPOSITION_DERIVATIVE_EPS`. Removed `svd_with_eps`/`eigh_with_eps` from the traced API in favor of `*_with_options`. |
| #1337 SVD canonical gauge | Numerical API option | `SvdGauge::CanonicalPivot` makes each U column's max-absolute pivot positive-real and applies the inverse sign/phase to the matching VT row, preserving reconstruction. |

## Design Notes

- `derivative_eps` is API-level AD regularization for repeated or nearly
  repeated spectral values. It is validated as positive and finite, but it is
  not a backend solver tolerance.
- Full SVD pruning to `SvdVals` deliberately drops SVD vector gauge options,
  because values-only execution should not retain vector policy.
- Concrete `LinalgBackend` gained default `svd_with_options` and
  `eigh_with_options` methods so options are available across concrete, eager,
  and traced surfaces.
- Canonical SVD gauge mutates only compact host outputs from the backend. It
  does not introduce implicit GPU-to-CPU transfers.

## Verification

- `cargo fmt --all`
- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `RUSTFLAGS='-C link-arg=-fuse-ld=bfd' cargo test --workspace --release`
- `cargo test -p tenferro-linalg --features autodiff --lib --tests`
- `cargo test -p tenferro-linalg --features autodiff ad::support::tests`
- `cargo test -p tenferro-ad --test extension_op traced_vjp_`
- `cargo test -p tenferro-runtime --test runtime_public_api traced_tensor_methods_cover_structural_surface`
- `cargo test -p tenferro-ad --test eager_tensor eager_slice_axis_and_builder_preserve_column_major_values`
- `RUSTFLAGS='-C link-arg=-fuse-ld=bfd' cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD --output-json /tmp/repository-rules-review.json`
- `RUSTFLAGS='-C link-arg=-fuse-ld=bfd' cargo test -p tenferro-linalg --features autodiff --doc --jobs 1`

The first coverage check showed `crates/tenferro-linalg/src/ad/support.rs`
below its per-file threshold because private const helper functions used to
build the manifest were not executed at runtime. Module-local tests now call
those helpers directly; the rerun passed all 149 checked files.

## Residual Risks

- Plain `cargo test -p tenferro-linalg --features autodiff --doc --jobs 1`
  failed locally in `rust-lld` with Bus error before the bfd retry. The bfd
  retry passed all linalg doctests, so this is treated as a local linker
  environment issue rather than a doctest/API failure.
- `SvdGauge::CanonicalPivot` is a post-processing policy on full SVD outputs.
  Values-only SVD and pruned graphs intentionally do not apply vector gauge
  behavior.
