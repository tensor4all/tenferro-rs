# Terasaki Bug Batch

## Summary

Fixed a bounded batch of recent `terasakisatoshi` bug reports in one PR. The
batch focuses on narrow public-boundary defects: unchecked shape arithmetic,
validation order, tutorial drift, dtype promotion parity, provider registration
status propagation, and LAPACK singular-status handling.

The PR also records a general repository audit rule for this class of bugs:
public API boundaries must validate user-derived shape, axis, dtype, padding,
slice, gather/scatter, linalg, allocation, launch, and FFI inputs before fast
paths, allocation, backend launch, or unchecked arithmetic.

## Context Read

- `REPOSITORY_RULES.md` and existing work-log guidance.
- Open `terasakisatoshi` GitHub issues #1054 through #1084.
- Affected modules in `tenferro-ad`, `tenferro-runtime`, `tenferro-cpu`,
  `tenferro-gpu`, `tenferro-linalg`, and `docs/tutorial-code`.
- Existing source-contract tests for GPU allocation and CPU/linalg behavior.

## Decisions

- Fixed unchecked CPU and GPU output allocation shape products with fallible
  checked products at allocation helpers instead of relying on individual call
  sites to remember overflow checks.
- Kept CPU indexing validation local to each public operation: pad rejects
  negative edge padding and shape overflow, and gather rejects collapsed
  dimensions whose slice size is not one.
- Made CPU `reduce_max` and `reduce_min` validate axes before the empty-axis
  fast path. The issue's literal "empty axes plus invalid axis value" cannot be
  represented by `&[usize]`, so the regression locks the source pattern.
- Aligned eager owned `Clamp` dtype promotion with the existing read path by
  promoting the input and both bounds together.
- Made typed BLAS/LAPACK provider registration wrappers return
  `ProviderRegistrationError` instead of discarding status codes. The BLAS
  typed wrapper uses the status-returning LP64 C registration entry points.
- Treated #1065 as a readability and regression-coverage fix: Rust's
  `let _ = expr?` still propagates the error, but the source now uses direct
  `?` propagation to make the intended validation obvious.
- Updated tutorial source and prose together so snippet and tutorial tests
  exercise the documented examples.

## Deferred

- #1054 and #1056 are not part of this PR.
- #1073 was not closed; the existing three-way repeated-label trace test for
  `iii` passes, so it needs a sharper reproducer before changing einsum logic.
- #1071, #1072, #1074, #1075, #1076, and #1080 through #1084 are broader
  safety, panic, concurrency, or overflow audits and should be split into
  follow-up PRs.

## Verification

- `cargo fmt --all --check`
- `cargo test -p tenferro-ad --test eager_tensor context_and_promotion::clamp_promotes_all_three_operands_to_common_dtype -- --exact`
- `cargo test -p tenferro-runtime graph::executor::tests::compile_with_input_specs_rejects_computed_placeholder_specs -- --exact`
- `cargo test -p tenferro-cpu --lib tests::indexing_coverage::cpu_indexing_validation_covers_error_branches -- --exact`
- `cargo test -p tenferro-cpu --lib tests::elementwise_reduction_helpers::test_reduction_helpers_cover_complex_and_error_paths -- --exact`
- `cargo test -p tenferro-cpu --test runtime_error_tests cpu_pooled_output_allocation_uses_checked_shape_product -- --exact`
- `cargo test -p tenferro-cpu --test runtime_error_tests cpu_reduce_max_min_validate_axes_before_empty_fast_path -- --exact`
- `cargo test -p tenferro-cpu --features provider-inject --lib inject::tests::typed_registration_status_helpers_report_errors -- --exact`
- `cargo test -p tenferro-cpu --features provider-inject --test inject_tests provider_inject_dot_general_uses_registered_blas -- --exact`
- `cargo test -p tenferro-gpu --test public_surface_contract cubecl_output_allocations_use_checked_shape_products -- --exact`
- `cargo test -p tenferro-linalg --test cpu_linalg_source_contract lapack_full_piv_lu_rejects_positive_getc2_info -- --exact`
- `cargo test -p tenferro-linalg --features cpu-blas,blas-accelerate --test full_piv_lu full_piv_lu_blas_rejects_singular_matrix -- --exact`
- `cargo test -p tenferro-linalg --features provider-inject --test inject_tests provider_inject_full_piv_lu_solve_uses_registered_lapack -- --exact`
- `cargo test -p tenferro-linalg --features provider-inject --test inject_dual_abi_tests ilp64_gemm_provider_reaches_lp64_consumer -- --exact`
- `cargo test -p tenferro-tutorial-code --release tutorial_binaries_run_successfully -- --exact`
- `python3 scripts/check-doc-snippets.py`
- `python3.11 scripts/check-guide-dependency-snippets.py`
- `cargo check -p tenferro-gpu --features cuda`
- `cargo check -p tenferro-linalg --features cuda`
