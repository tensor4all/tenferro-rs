# Terasaki Bug Batch

## Summary

Fixed a bounded batch of recent `terasakisatoshi` bug reports in one PR. The
batch focuses on public-boundary defects: unchecked shape arithmetic,
validation order, tutorial drift, dtype promotion parity, provider registration
status propagation, LAPACK singular-status handling, panic-safe CPU buffer-pool
ownership, poisoned-lock handling, traced symbolic/rank validation, and
selected GPU/CPU batched offset overflow checks.

The PR also records a general repository audit rule for this class of bugs:
public API boundaries must validate user-derived shape, axis, dtype, padding,
slice, gather/scatter, linalg, allocation, launch, and FFI inputs before fast
paths, allocation, backend launch, or unchecked arithmetic.

The follow-up audit added two more durable rules: batch pointer-offset loops
must check both stride products and `batch * stride`, and public cache/runtime
locks must not fabricate default state after poison.

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
- Made CPU buffer-pool acquisition return initialized zero values instead of
  stale or uninitialized elements, keeping the existing unsafe signature only
  for compatibility.
- Replaced panic-unsafe CPU buffer-pool lending with a Drop-backed loan guard so
  the pool is restored when a backend session or linalg pool closure panics.
- Held eager gradient slot locks through accumulation writes to remove the
  read-compute-write race reported in #1075.
- Allowed traced binary operations over same-rank symbolic-shape tensors without
  forcing `concrete_shape()`, and stopped reduction helpers from underflowing
  output rank when too many axes are supplied.
- Added early traced linalg rank/axis validation and removed assertion panics
  from linalg extension metadata inference on bad input counts.
- Added fallible poison-reporting paths for EagerRuntime locks, global
  extension AD rule registry lookup/registration, and CUDA extension cache
  inspection/clear methods.
- Added checked arithmetic for CPU triangular mask indexing and GPU triangular
  batched linalg pointer offsets.
- Added the eager SVD singular-value-sum backward MWE as a regression test; it
  already passes on this branch, so #1056 is covered without code changes.

## Deferred

- #1054 remains outside this PR until its current reproducer and intended
  behavior are rechecked.
- #1073 was not closed; the existing three-way repeated-label trace test for
  `iii` passes, so it needs a sharper reproducer before changing einsum logic.
- #1082 requires a larger design decision because `DimExpr` and shape inference
  currently expose infallible arithmetic in several public-facing paths.
- #1084 is a broad public-panic audit. This PR fixes representative traced,
  linalg, poison, and buffer-pool cases and records the rule; remaining cases
  should be handled as a follow-up sweep.
- #1081 is partially fixed for CPU triangular masks and GPU triangular batched
  linalg offsets. Tensor stride/accessor API changes need a separate API design.

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

Additional targeted checks after the second audit pass:

- `cargo test -p tenferro-cpu acquired_buffers_are_initialized_on_fresh_and_reused_paths`
- `cargo test -p tenferro-cpu restores_buffers_after_panic`
- `cargo test -p tenferro-ad eager_runtime_grad_accumulation_keeps_slot_locked_through_update`
- `cargo test -p tenferro-runtime traced_broadcast_binary_accepts_symbolic_same_rank_input`
- `cargo test -p tenferro-runtime traced_reduction_with_too_many_axes_does_not_underflow_rank`
- `cargo test -p tenferro-linalg without_panicking`
- `cargo test -p tenferro-linalg infer_output_meta_returns_empty_on_input_count_mismatch`
- `cargo test -p tenferro-ad eager_runtime_synchronize_reports_poisoned_backend_lock`
- `cargo test -p tenferro-ad eager_runtime_register_extension_reports_poisoned_executor_lock`
- `cargo test -p tenferro-gpu --features cuda cuda_extension_cache_try_methods_report_poisoned_lock`
- `cargo test -p tenferro-internal-ops global_extension_rule_registry_does_not_expect_on_poison_contract --features autodiff`
- `cargo test -p tenferro-linalg --features autodiff,cpu-faer svd_singular_value_sum_backward_does_not_panic -- --nocapture`
- `cargo test -p tenferro-linalg gpu_triangular_solve_batched_offsets_use_checked_arithmetic`
- `cargo test -p tenferro-cpu triu`
- `cargo test -p tenferro-cpu test_triangular_masks_use_checked_index_arithmetic_contract`
