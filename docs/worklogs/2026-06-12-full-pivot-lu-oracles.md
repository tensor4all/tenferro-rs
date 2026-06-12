# Full-Pivot LU Oracle Coverage

Date: 2026-06-12

## Summary

This work resolves the oracle gap tracked by
<https://github.com/tensor4all/tenferro-rs/issues/987>. The vendored
`tensor-ad-oracles` tree now includes a local `full_pivot_lu/identity` family
with fixed-pivot finite-difference, JVP, VJP, and HVP references. The linalg AD
support manifest now treats full-pivot LU and full-pivot LU solve as
oracle-backed instead of pending.

## Code And Documents Read

- `AGENTS.md`
- `REPOSITORY_RULES.md`
- `crates/tenferro-linalg/src/ad.rs`
- `crates/tenferro-linalg/src/ad/support.rs`
- `crates/tenferro-linalg/src/ad/rules/mod.rs`
- `crates/tenferro-linalg/src/ad/rules/solve.rs`
- `crates/tenferro-linalg/tests/ad_support_manifest.rs`
- `crates/tenferro-linalg/tests/traced_ad_explicit.rs`
- `third_party/tensor-ad-oracles/**`
- `/home/shinaoka/tensor4all/tensor-ad-oracles/**`

## Decisions

- Vendored only the full-pivot LU oracle pieces from the upstream
  `tensor-ad-oracles` checkout instead of refreshing unrelated JAX or
  structural-family work.
- Kept the full-pivot oracle source as `source_repo = "tensor-ad-oracles"` so
  generated provenance does not claim an upstream PyTorch full-pivot LU OpInfo.
- Marked `full_piv_lu` factor outputs as `SupportedViaLinearize`; row pivots,
  column pivots, and parity remain nondifferentiable metadata.
- Marked `full_piv_lu_solve` linearize support as `SupportedViaLinearize` and
  transpose support as `Supported`, matching the existing rule implementation.
- Updated the oracle support snapshot to include the new 12 full-pivot LU
  records and the pre-existing one-record `tropical_einsum_maxplus` family that
  was missing from the unsupported table.
- Fixed a stale absolute path in the vendored oracle `test_case_loader` tests
  so full unittest discovery works in worktrees and other checkout locations.

## Verification

- `uv run python -m py_compile generators/full_pivot_lu.py generators/pytorch_v1.py validators/replay.py`
- `uv run python -m unittest discover -s tests -v`
- `uv run python -m unittest tests.test_pytorch_v1.PytorchV1RegistryTests.test_build_case_spec_index_includes_local_full_pivot_lu tests.test_pytorch_v1.PytorchV1RegistryTests.test_main_list_prints_case_registry tests.test_solve_generation.SolveGenerationTests.test_generate_full_pivot_lu_records_cover_rectangular_cases tests.test_math_registry.MathRegistryTests.test_repo_lu_note_exposes_full_pivot_lu_family -v`
- `uv run python scripts/validate_schema.py`
- `uv run python scripts/verify_cases.py`
- `uv run python scripts/check_replay.py`
- `uv run python scripts/check_regeneration.py`
- `uv run python scripts/check_math_registry.py`
- `uv run python scripts/check_tolerances.py`
- `uv run python -m unittest tests.test_upstream_ad_tolerance_script -v`
- `uv run python scripts/check_upstream_ad_tolerances.py`
- `cargo test -p tenferro-linalg --features autodiff --test ad_support_manifest --release`
- `cargo test -p tenferro-linalg --features autodiff --test traced_ad_explicit full_piv --release`
- `cargo test -p tenferro-linalg --test full_piv_lu --release`

## Remaining Risks

- Full workspace, coverage, and rustdoc checks were not run in this local pass.
- CUDA full-pivot LU remains unsupported, as documented by the GPU backend
  status; this change only clears CPU/traced AD oracle coverage.
