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

After #1086 merged, a final audit follow-up added targeted coverage/fixes for
the remaining low-risk `terasakisatoshi` reports: repeated-label einsum
coverage for #1073, tensor/tensor-core stride and accessor checked arithmetic
plus CPU/GPU batched offset arithmetic for #1081, symbolic-shape traced linalg
and AD seed error handling as representative #1084 cases, and tropical traced
extension fallibility. It also corrected the #1080 buffer-pool fix to split
full-overwrite raw acquisition from explicit zeroed acquisition instead of
zero-filling every pooled buffer.
The repository rule update was generalized: performance-sensitive operations
that look potentially dangerous must carry a nearby invariant comment so later
agents do not "fix" false positives with hidden initialization, copies, or
checks.

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
- Corrected the CPU buffer-pool fix after review: `pool_acquire` remains the
  unsafe full-overwrite path for hot kernels, while
  `pool_acquire_zeroed` / `BufferPool::acquire_zeroed` are explicit safe paths
  for callers that may read before every element is overwritten. Zero-dimension
  GEMM returns use the zeroed path; normal BLAS/faer `beta = 0` full-overwrite
  paths keep raw acquisition.
- Added one-line invariant comments at raw pooled-buffer acquisition sites and
  uninitialized pooled-output helper call sites, including the faer `beta != 0`
  accumulation branch.
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
- Added eager and traced regression coverage for #1073 (`iii`, `iiii`, and
  mixed repeated-label cases). These pass with the existing recursive planner
  and eager implementation, so the issue is covered without changing einsum
  logic.
- Completed #1081's listed overflow class: tensor and tensor-core column-major
  stride construction, tensor offset helpers, CPU triangular/reshape/
  concatenate/scatter boundary arithmetic, and GPU batched linalg stride and
  `batch * stride` pointer offsets now use checked arithmetic.
- Made traced linalg helpers return typed symbolic-shape errors instead of
  panicking when `inv`, `pinv`, `pinv_with_rtol`, or `norm(..., keepdim=true)`
  receive a symbolic-shape tensor.
- Made traced `jvp`/`vjp` return typed errors for symbolic seed tensors instead
  of panicking while registering seed metadata.
- Switched tropical traced einsum through the fallible extension builder and
  added symbolic-shape coverage for the standard `ij,jk->ik` path.

## Deferred

- #1054 remains outside this PR until its current reproducer and intended
  behavior are rechecked.
- #1082 requires a larger design decision because `DimExpr` and shape inference
  currently expose infallible arithmetic in several public-facing paths.
- #1084 is a broad public-panic audit. This PR fixes representative traced,
  linalg, poison, and buffer-pool cases and records the rule; remaining cases
  should be handled as a follow-up sweep.
- #1084 still has broader public-panic surface in AD structural helpers, FFT,
  and several traced shape-manipulation helpers. This follow-up fixes symbolic
  AD seed and traced linalg representative cases, but does not attempt the
  whole API redesign.

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

Additional targeted checks after the final audit follow-up:

- `cargo fmt --all --check`
- `cargo test -p tenferro-einsum eager_einsum_handles_three_or_more_repeated_labels -- --nocapture`
- `cargo test -p tenferro-einsum einsum_three_or_more_repeated_labels_keep_and_mix -- --nocapture`
- `cargo test -p tenferro-tensor col_major_helpers_cover_scalar_and_higher_rank_shapes -- --nocapture`
- `cargo test -p tenferro-tensor linear_offset_helpers_check_overflow -- --nocapture`
- `cargo test -p tenferro-tensor-core compact_ -- --nocapture`
- `cargo test -p tenferro-cpu cpu_reshape_concatenate_scatter_use_checked_boundary_arithmetic_contract -- --nocapture`
- `cargo test -p tenferro-cpu buffer_pool -- --nocapture`
- `cargo test -p tenferro-linalg gpu_ --test gpu_linalg_source_contract -- --nocapture`
- `cargo check -p tenferro-linalg --features cuda`
- `cargo test -p tenferro-linalg traced_linalg_helpers_reject_symbolic_shapes_without_panicking -- --nocapture`
- `cargo test -p tenferro-linalg without_panicking -- --nocapture`
- `cargo test --manifest-path ext/tropical/Cargo.toml traced_einsum_accepts_symbolic_shapes_without_panicking -- --nocapture`
- `cargo test -p tenferro-ad traced_jvp_vjp_return_errors_for_symbolic_seed_tensors -- --nocapture`
