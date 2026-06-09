# Complex VJP, Eigh DType, And Oracle CI Policy

## Session Summary

Fixed two correctness bugs: complex analytic VJP transpose rules now conjugate
holomorphic derivative coefficients under the tenferro/tidu real-inner-product
convention, and public `eigh` returns real eigenvalue tensors for complex
Hermitian inputs. The PR also records the complex VJP convention in the AD spec
and documents the intended oracle CI tiers.

## Context Read

- `REPOSITORY_RULES.md`
- `docs/spec/ad-contract.md`
- `docs/oracle/index.md`
- `docs/oracle/tensor-ad-oracles-support.md`
- Prior worklog `docs/worklogs/2026-06-02-linalg-values-batch-optimizations.md`
- Local tenferro AD transpose rules and linalg CPU/GPU/extension metadata
- `tidu-rs` complex AD guide and regression tests
- Vendored `tensor-ad-oracles` records for representative complex VJP cases

## Decisions Made

- Complex VJP uses the tidu/JAX-style real inner product
  `<a, b> = Re(conj(a) * b)`. For holomorphic elementwise maps, the transpose
  multiplies cotangents by `conj(f'(z))`.
- JVP rules keep the ordinary holomorphic coefficient `f'(z)` and do not
  conjugate it.
- Analytic transpose rules for `exp`, `log`, `sin`, `cos`, `tanh`, `sqrt`,
  `rsqrt`, `expm1`, `log1p`, `pow`, and `div` now conjugate fixed complex
  derivative coefficients.
- Public `eigh(C32)` returns `(F32 eigenvalues, C32 eigenvectors)` and
  public `eigh(C64)` returns `(F64 eigenvalues, C64 eigenvectors)` on CPU,
  matching the existing GPU path and traced extension contract.
- Oracle CI policy is split into a PR sentinel tier on standard Linux runners
  and a post-merge full supported-oracle tier sharded across standard Linux
  runners.

## Rejected Or Deferred Alternatives

- Did not switch to PyTorch's different complex pullback presentation. The
  tenferro primitive-rule surface is graph-level `linearize` and
  `transpose_rule`, and tidu already defines the real-inner-product convention.
- Did not add a large Linux runner or GPU runner requirement for oracle CI. The
  chosen policy keeps PR and post-merge oracle replay on standard Linux runners.
- Did not claim full oracle replay coverage in CI. The historical root-facade
  replay harness has been removed, so scalar analytic oracle replay remains a
  follow-up to restore in the owning crate.

## Verification Performed

- `cargo test -p tenferro-ad complex_unary_vjps_conjugate_holomorphic_derivatives --test ad -- --nocapture`
- `cargo test -p tenferro-ad complex_div_and_pow_vjps_conjugate_holomorphic_coefficients --test ad -- --nocapture`
- `cargo test -p tenferro-linalg test_batched_complex_eigh --lib -- --nocapture`
- `cargo test -p tenferro-linalg cpu_linalg_accepts_c32_happy_paths --lib -- --nocapture`
- `cargo test -p tenferro-linalg traced_metadata_matches_linalg_extension_shapes_and_dtypes --test traced_extension -- --nocapture`
- `cargo test -p tenferro-internal-ops`
- `cargo test -p tenferro-ad --test ad`
- `cargo test -p tenferro-linalg`
- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`

## Remaining Risks

- The scalar analytic tensor-ad-oracles families are present in the vendored
  database but are still unsupported by the current tenferro replay adapter.
  The focused regression tests cover the fixed bug until crate-local oracle
  replay is restored.
