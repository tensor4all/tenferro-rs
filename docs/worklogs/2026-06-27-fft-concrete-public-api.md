# FFT Concrete Public API

## Summary

Implemented #1239 by adding concrete non-AD FFT wrappers for `Tensor` and
`TensorRead` while keeping the existing traced FFT API unchanged.

## Context Read

- `CONTRIBUTING.md`
- `REPOSITORY_RULES.md`
- `docs/spec/api-conventions.md`
- `docs/spec/operation-categories.md`
- `docs/guides/tenferro-fft.md`
- `crates/tenferro-fft/src/lib.rs`

## Decisions

- Added `TensorFftExt` for compact concrete `Tensor` values, with unsuffixed
  `fft`, `ifft`, `rfft`, and `irfft` methods.
- Added `TensorReadFftExt` for borrowed/read-oriented inputs, with
  `fft_read`, `ifft_read`, `rfft_read`, and `irfft_read` methods.
- Kept `TracedTensorFftExt` unchanged.
- Did not add public module free functions. The public surface remains
  crate-root extension traits.
- Reused the existing host `rustfft` execution helper and validation path.
  Backend-backed tensors or views continue to return explicit host/download
  errors instead of transferring data implicitly.

## Alternatives Rejected

- `tenferro_fft::fft(...)` free functions: rejected because extension-family
  operation surfaces are crate-root extension traits under the current API
  convention.
- `TypedTensor<T>` wrappers in the same change: deferred because FFT can change
  dtype (`rfft` real to complex, `irfft` complex to real), so typed return
  contracts need a separate design.

## Verification

- RED: `cargo test -p tenferro-fft concrete -- --nocapture` failed on missing
  `TensorFftExt` and `TensorReadFftExt`.
- GREEN: `cargo test -p tenferro-fft concrete -- --nocapture`
- `cargo test -p tenferro-fft`
- `cargo test -p tenferro-fft --features autodiff`
- `cargo test -p tenferro-einsum -p tenferro-fft`
- `cargo test -p tenferro-einsum -p tenferro-fft --features tenferro-einsum/autodiff,tenferro-fft/autodiff`
- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `python3 scripts/check-doc-snippets.py --check`
- `python3 scripts/check-guide-dependency-snippets.py`
- `python3 scripts/check-api-consistency.py --fail-on-findings`
- `python3 scripts/check-operation-categories.py --fail-on-findings`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
- `git diff --check`

## Remaining Risk

CUDA/cuFFT is still future work. These wrappers expose the current CPU-host FFT
implementation and preserve the existing no-implicit-transfer behavior.
