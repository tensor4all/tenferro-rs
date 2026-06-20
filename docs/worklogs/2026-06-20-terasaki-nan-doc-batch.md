# Terasaki NaN And Docs Batch

## Summary

Fixed a bounded follow-up batch of open `terasakisatoshi` reports:

- #1129: empty eager concatenate now reports a typed tensor validation error
  instead of `Error::Internal`.
- #1130: same-dtype unsupported CPU `Bool` arithmetic reports unsupported dtype
  instead of a self-contradictory dtype mismatch.
- #1131: einsum ellipsis is rejected at parse validation with a specific
  unsupported-ellipsis error instead of being treated as literal labels.
- #1132: traced FFT rejects a concrete zero-length implicit transform axis at
  API validation time.
- #1133 and #1134: CPU maximum/minimum and reduce_max/reduce_min propagate NaN
  in the same order-independent style expected by JAX/PyTorch users; zero-length
  reduced axes return typed validation errors instead of sentinel values.
- #1054 and #1115: first-user docs now cover local checkout setup, scratch
  workspace isolation, dependency snippets, column-major and reduction-axis
  notes, extension runtime setup context, README discoverability, and the
  missing WebGPU quickstart details.
- The WebGPU backend now compiles under its `webgpu` feature after the
  fallible `TypedTensor::from_*` constructor migration, which keeps the new
  WebGPU guide example attached to a checked public surface.

## Context Read

- `AGENTS.md`, shared tensor4all rules, and `REPOSITORY_RULES.md`.
- Open GitHub issues #1054, #1115, #1129, #1130, #1131, #1132, #1133, #1134.
- Affected modules in `tenferro-ad`, `tenferro-cpu`, `tenferro-einsum`,
  `tenferro-fft`, README, getting-started docs, eager/autodiff guides,
  extension guides, and GPU docs.

## Decisions

- Kept #1131 small: this PR does not implement ellipsis semantics. It turns the
  previous misleading parser behavior into a clear unsupported-feature error,
  which matches the issue's short-term acceptance path.
- Treated same-dtype unsupported CPU arithmetic as backend unsupported dtype,
  while preserving `DTypeMismatch` for genuinely different dtypes.
- Used explicit NaN propagation helpers for CPU max/min reductions rather than
  relying on `Float::max`/`Float::min`, whose NaN behavior is not the numerical
  API contract users expect from JAX/PyTorch-style tensor operations.
- Rejected zero-length reduced axes for min/max because the old identity
  sentinels (`+/-inf`) were observable user values, not valid reductions.
- Documented local path examples as layout-dependent and added `[workspace]`
  only to scratch-crate examples inside the repository checkout.
- Added WebGPU documentation as experimental direct-backend usage, not as a
  promise of broad GPU operation coverage.
- Kept the WebGPU code fix limited to feature-compilation follow-through:
  propagate fallible tensor constructors and fix dense-stride `Result`
  handling without changing kernel semantics.

## Verification

- `cargo fmt --all --check`
- `cargo test -p tenferro-ad --test eager_tensor`
- `cargo test -p tenferro-cpu --lib`
- `cargo test -p tenferro-einsum --lib`
- `cargo test -p tenferro-fft --test fft_ops`
- `cargo check -p tenferro-gpu --no-default-features --features webgpu`
- `cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer --test webgpu_backend_contract`
- `cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer --test webgpu_matmul_runtime`
- `python3 scripts/check-guide-dependency-snippets.py`
- `python3 scripts/check-doc-snippets.py`
- `python3 scripts/check-docs-site.py`
- `cargo doc --workspace --no-deps`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `git diff --check`
