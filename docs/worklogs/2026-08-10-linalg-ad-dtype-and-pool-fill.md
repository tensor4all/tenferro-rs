# Work log: linalg AD dtype boundary (#1649) and pooled full-overwrite fill (#1640)

Session summary for the combined bug-fix batch. Both findings were filed by
`tensor4all-ai-bot`; both were verified against current `main`
(`1458b509efa87f6810dc15b73cc3ec9afcd12b19`) before implementation.

## Context read

- `crates/tenferro-linalg/src/ad/rules/support.rs` — `broadcast_scalar_constant_with_dtype`,
  `fixed_scale_with_dtype`, `linear_scale_with_dtype`, `self_adjoint_from_lower_linear`.
- `crates/tenferro-linalg/src/ad/rules/mod.rs` — `linearize_svd`, `linearize_svd_values`,
  `linearize_eigh`, `linearize_eigh_values`, `linearize_cholesky`, `linearize_qr`, and the
  existing `invalid_dim_expr` / `LINALG_AD_OP_PREFIX` error conventions.
- `crates/tenferro-linalg/src/validation.rs` / `src/traced.rs` / `src/eager_composites.rs` —
  duplicate `ensure_float_or_complex` helpers; traced decomposition constructors and their
  rustdoc; `eigvalsh -> eigh_values` and `norm -> svd_values` reachability.
- `crates/tenferro-cpu/src/structural.rs` — `filled_tensor_from_pool`, `zeroed_tensor_from_pool`,
  `embed_diagonal` Bool path, `typed_embed_diagonal_with_pool`.
- `crates/tenferro-internal-cpu-kernels/src/pooled_uninit_output.rs` and `buffer_pool.rs` —
  `PooledUninitOutput` full-overwrite guard contract and drop safety.
- `crates/tenferro-cpu/src/provider.rs`, `dot_runtime.rs`, `exec_session.rs` — provider
  requests expose initialized `TensorWrite` only (no uninit destination variant).
- Issues #1649, #1640, and the accepted repair boundary in the #1649 design-audit comment.
- REPOSITORY_RULES.md (dtype-aware AD seeds, full-overwrite buffer contract, AD Rule
  Coverage, invariant markers, module-local test placement).

## Decisions

### #1649 — linalg AD constant dtype

- The reported F64 constant for I32/I64/Bool in `broadcast_scalar_constant_with_dtype` is
  real but the conversion-based suggestion (round factor) was rejected: rounding `0.5` or
  `eps^2` would invent integer/bool derivative semantics. Integer/bool SVD/Eigh/QR/Cholesky
  are unsupported by the primal backends; the correct repair is to reject them.
- **Boundary validation**: add `ensure_float_or_complex(op, dtype)` to the traced
  constructors `svd_with_options` (covers `svd`), `svd_full`, `eigh_with_options` (covers
  `eigh`), `qr_with_options` (covers `qr`), `cholesky`, and `eigvalsh` (its `eigh_values`
  route bypasses `eigh_with_options`). `svd_values` needs no guard (only reachable via the
  already-guarded `norm`). `eig`/`lu`/`solve`/`triangular_solve` are intentionally not
  touched: out of the accepted boundary, `eig` already guards AD, and existing
  metadata-promotion tests rely on integer `solve`/`triangular_solve` construction.
- **Defensive AD invariant**: the scale/constant helper chain becomes fallible and rejects
  I32/I64/Bool with `ADRuleError::invalid_input("tenferro-linalg.{op}", Jvp, ...)` before
  any constant op is emitted, so an F64 derivative constant is never published. `op` is
  threaded as `&'static str` using the rule function names (`linearize_svd`, ...) to match
  the existing `invalid_dim_expr` convention.
- **DRY**: consolidate the three identical `ensure_float_or_complex` copies
  (`validation.rs`, `traced.rs`, `eager_composites.rs`) into the shared
  `crate::validation::ensure_float_or_complex` (made `pub(crate)`).
- **Coverage policy**: the new integer/bool rejection arm is an intentional defensive arm on
  a linalg AD file that already carries below-default thresholds (per the AD Rule Coverage
  section). No threshold changes and no line-padding.

### #1640 — pooled full-overwrite fill

- `filled_tensor_from_pool` is the root finding: it zero-initialized the pooled buffer and
  then fully overwrote it. Replaced with a single pass through
  `PooledUninitOutput::<T>::new` + `as_uninit_slice_mut().fill(MaybeUninit::new(fill))` +
  `assume_init()`, preserving the operation-specific shape-product error attribution. The
  armed uninit token is safely discarded on error/unwind by `PooledUninitOutput::Drop`.
  `PoolScalar` already implies `Copy + TensorScalar`, so the bounds simplify.
- **Deferred (follow-up issue, not this PR)**: `dot_runtime.rs` canonical-operand
  allocation and `exec_session.rs` dot-output allocation also zero-fill before full
  overwrite, but converting them requires an uninit destination variant across the
  `CpuLayoutTransformRequest` / `CpuDotGeneralRequest` provider contracts (all provider
  implementations plus test providers), and zero-K GEMM with `beta = 0` has distinct
  overwrite semantics. This is a provider-contract change, deliberately out of bug-fix
  scope.

## Verification

- Design reviewed by independent gpt-review before implementation; findings incorporated
  (`eigvalsh` boundary, no-constant-published assertion, source-contract regression for the
  full-overwrite contract, svd_full rustdoc, module-local test placement, eager helper
  dedup).
- Post-implementation: gpt-review of the full diff; `cargo fmt`, focused tests for
  tenferro-linalg and tenferro-cpu, clippy, and the repository-required
  `check-pr-fast.sh` gate.

## Residual risks / follow-ups

- Integer/bool dtype rejection at trace time is not yet applied to `eig`/`lu`/
  `full_piv_lu`/`solve`/`triangular_solve`; those still fail at backend execution. A
  follow-up could extend the same boundary for consistency (accepted boundary deliberately
  limited the scope to SVD/Eigh/QR/Cholesky).
- Provider-contract uninit GEMM/layout-transform outputs remain as a separate issue.
