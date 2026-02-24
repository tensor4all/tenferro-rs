# tenferro-linalg CPU Completion Design

**Date**: 2026-02-25
**EPIC**: [#216](https://github.com/tensor4all/tenferro-rs/issues/216)
**Child issues**: #217, #218, #219, #220, #221, #222

## Summary

Complete the CPU-only path of `tenferro-linalg`: implement remaining stubs
(`eig`, `matrix_exp`), add systematic finite-difference AD validation, raise
coverage to >=95%, and align docs.

## Current State

The crate is more mature than the issue description assumes:

- **No `todo!()`**: All 27+ forward functions are implemented via faer backend
- **All 28 AD rules** (14 rrule + 14 frule) are implemented
- **No GPU code**: Crate is already purely CPU
- **50 tests** exist; coverage thresholds are low (lib.rs: 30%, faer_backend: 70%)
- **Stubs**: `eig()` and `matrix_exp()` intentionally return errors

## Design Decisions

1. **`eig()` returns complex output always** — general eigendecomposition of
   real matrices can produce complex eigenvalues. `eig()` always returns
   `Complex64`/`Complex32` eigenvalues+vectors regardless of input scalar
   type. Matches numpy/scipy convention.

2. **Reference PyTorch for all AD math** — all rrule/frule formulas must
   match PyTorch's `torch.linalg` backward implementations. No inventing
   math from scratch. Existing AD rules should be audited against PyTorch.

3. **Minimal JSON test harness** — `linalg_cases.json` stores input matrices
   and expected forward outputs. FD checks are Rust-native using these
   matrices. No gradient values in JSON.

## Phases

### Phase 1: Module Cleanup (#217)

Near no-op since no GPU code exists. Actions:

- Add module-level doc comments clarifying CPU-only status
- Add `#[cfg(not(feature = "gpu"))]` compile-time assertion on faer backend
- Document in `backend/mod.rs` where future GPU backends would slot in
- Verify `cargo test -p tenferro-linalg` still passes

### Phase 2: Implement eig() + matrix_exp() (#218)

#### `eig()` — General Eigendecomposition

- Add `eig()` to `LinalgBackend<T>` trait
- Implement via faer's eigendecomposition for general (non-symmetric) matrices
- Return type: always complex (`Complex64` for f64 input, etc.)
- Add `eig_rrule()` and `eig_frule()` following PyTorch's
  `torch.linalg.eig` backward
- Replace current error-returning stubs

#### `matrix_exp()` — Matrix Exponential

- Implement scaling-and-squaring with Padé approximation, following
  PyTorch's `matrix_exp` implementation
- Add `matrix_exp_rrule()` and `matrix_exp_frule()` following PyTorch's
  backward (uses auxiliary matrix approach)
- Replace current error-returning stub

### Phase 3: Testing Infrastructure (#219 + #222)

#### JSON Test Case Database

- Expand `tests/data/linalg_cases.json` with structured test matrices:
  - Small (2x2, 3x3), medium (5x5, 10x10)
  - Real (f64) and complex (Complex64)
  - Edge cases: near-singular, ill-conditioned, symmetric, triangular
  - Batched variants
- JSON schema: `{operation, dtype, input_matrices, expected_outputs}`

#### Finite-Difference Verification

- Shared FD utilities in test helper module:
  - `fd_jacobian(f, x, eps)` — numerical Jacobian for rrule validation
  - `fd_jvp(f, x, dx, eps)` — directional derivative for frule validation
- Per-operation FD tests for all 14 operations (both rrule and frule)
- Tolerance policy: `eps=1e-6` for FD step, `atol=1e-4` for comparison
  (tighter for well-conditioned cases)
- Deterministic: fixed seeds for any random test data

### Phase 4: Coverage + Docs (#220 + #221)

#### Coverage Gate

- Raise `lib.rs` threshold: 30% → 95%
- Raise `faer_backend.rs` threshold: 70% → 95%
- Add targeted tests to close gaps (error paths, edge cases, batch dims)
- Verify locally with `cargo llvm-cov` before pushing

#### Documentation

- Update `docs/design/linalg.md` to reflect completed CPU status
- Remove "API skeleton only" language where it applies to linalg
- Document CPU/GPU boundary and current faer-only backend
- Update `docs/design/testing.md` with FD validation strategy
- Cross-link to PyTorch reference implementations

## Correspondence with PyTorch

| tenferro-linalg | PyTorch | Notes |
|-----------------|---------|-------|
| `svd_rrule` | `torch.linalg.svd` backward | F-matrix formulation |
| `qr_rrule` | `torch.linalg.qr` backward | |
| `lu_rrule` | `torch.linalg.lu` backward | |
| `eigen_rrule` | `torch.linalg.eigh` backward | Symmetric only |
| `eig_rrule` | `torch.linalg.eig` backward | General, complex output |
| `cholesky_rrule` | `torch.linalg.cholesky` backward | |
| `solve_rrule` | `torch.linalg.solve` backward | |
| `inv_rrule` | `torch.linalg.inv` backward | |
| `det_rrule` | `torch.linalg.det` backward | |
| `slogdet_rrule` | `torch.linalg.slogdet` backward | |
| `pinv_rrule` | `torch.linalg.pinv` backward | |
| `matrix_exp_rrule` | `torch.matrix_exp` backward | Scaling+squaring |
| `norm_rrule` | `torch.linalg.norm` backward | Fro/Nuclear/Spectral |
| `lstsq_rrule` | `torch.linalg.lstsq` backward | |
