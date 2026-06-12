# Issues 1000 And 1001 Remediation Batch

## Session Summary

This batch resolves GitHub issue #1001 and fixes several concrete, testable
findings from the #1000 umbrella audit. It wires tenferro-owned CPU tensor
kernels to the existing `CpuContext` parallelism contract, fixes the cached faer
GEMM helper so it enters the owned Rayon pool, removes a CPU concatenate
per-output segment scan, reuses LAPACK batched input scratch, tightens CPU and
CUDA/CubeCL validation, placement, and device-native fast-path contracts,
keeps LU GPU kernels from specializing on matrix-size extents, aligns the
linalg values-only AD support manifest with finite-difference evidence, adds
local finite-difference coverage for einsum and FFT extension AD rules, removes
per-component gather/scatter index-vector allocation, and fixes the
publish-layout, API-consistency scope/false-positive gaps, and active
crate-ownership documentation drift detected by the audit. Superseded
historical design/reference notes are archived under `docs/plans/historical/`
so the active design and reference indexes only point at current material.
The final #1000 cleanup slice documents runtime dtype-conversion semantics,
routes traced `eigvals` through a general eigenvalues-only extension/backend
hook instead of materializing eigenvectors and discarding them, and keeps
zero-sized GPU LU parity initialization device-native.

The PR should use `Closes #1001` and `Closes #1000`. Remaining public tracing
builder panics are classified as design-gated because their current public
signatures do not return `Result`; compatible extension runtime output-count
loss paths are already fixed on the included `main`.

## Context Read

- `AGENTS.md`
- `CONTRIBUTING.md`
- `REPOSITORY_RULES.md`
- `ai/contribution-workflows/repository-remediation.md`
- GitHub issues #1000 and #1001
- Online tensor4all shared rules:
  - `rules/index.md`
  - `rules/common/repository.md`
  - `rules/common/performance.md`
  - `rules/common/docs-and-tests.md`
  - `rules/rust/index.md`
  - `rules/rust/performance.md`
- `crates/tenferro-cpu/src/context.rs`
- `crates/tenferro-cpu/src/backend.rs`
- `crates/tenferro-cpu/src/elementwise.rs`
- `crates/tenferro-cpu/src/analytic.rs`
- `crates/tenferro-cpu/src/reduction.rs`
- `crates/tenferro-cpu/src/structural.rs`
- `crates/tenferro-cpu/src/indexing.rs`
- `crates/tenferro-gpu/src/cubecl/dispatch.rs`
- `crates/tenferro-gpu/src/cubecl/gemm.rs`
- `crates/tenferro-gpu/src/cubecl/interop.rs`
- `crates/tenferro-gpu/tests/cubecl_launch_contract.rs`
- `crates/tenferro-linalg/src/gpu/linalg.rs`
- `crates/tenferro-linalg/src/gpu/kernels.rs`
- `crates/tenferro-linalg/src/cpu/backend.rs`
- `crates/tenferro-linalg/src/cpu/linalg/lapack_linalg/helpers.rs`
- `crates/tenferro-linalg/src/ad/support.rs`
- `crates/tenferro-linalg/tests/backend_errors.rs`
- `crates/tenferro-linalg/tests/cpu_linalg_source_contract.rs`
- `crates/tenferro-linalg/tests/ad_support_manifest.rs`
- `crates/tenferro-linalg/tests/gpu_linalg_source_contract.rs`
- `docs/design/linalg-prims.md`
- `docs/reference/jax-stablehlo-needed.md`
- `docs/reference/libtorch.md`
- `docs/reference/pytorch-dense-cpu-parity.md`
- `docs/spec/primitive-catalog.md`
- `docs/guides/parallelism-and-caching.md`
- `docs/design/tensor-prims.md`
- `docs/design/exec-session.md`
- `docs/design/dot-general-overhead.md`
- `docs/design/contraction-pipeline.md`
- `scripts/check-publish-layout.py`

## Reference Code

- strided-rs `main` at `71bdd913158a87437e51f4f9b69cba4cac6f5082`.
- `strided-kernel` at that revision has a `parallel` feature, Rayon-backed
  threading helpers, `MaybeSendSync` / `MaybeSync` bounds for parallel builds,
  and large-array tests that exercise parallel `map_into` paths.
- `strided-einsum2/parallel` propagates `strided-kernel/parallel` and
  `strided-perm/parallel`.

## Classification Ledger

| Finding | Classification | Current action |
| --- | --- | --- |
| `strided-kernel` lacked the `parallel` feature in the tenferro workspace dependency graph | Auto Fix / Fixed | Updated the strided-rs pin to current `main` and enabled `strided-kernel/parallel` in the workspace dependency. Added a source-contract test that checks the manifest wiring. |
| `cpu-faer` did not propagate `strided-einsum2/parallel` | Auto Fix / Fixed | Added `strided-einsum2/parallel` to `cpu-faer` and to provider feature aliases that enable `strided-einsum2`. Verified with `cargo tree -p tenferro-cpu -e features -i strided-kernel`. |
| Cached faer GEMM helper did not enter the owned `CpuContext` Rayon pool | Auto Fix / Fixed | Changed `install_with_pool_and_gemm_cache` to run through `ctx.install(...)`. Added a regression test that observes the configured non-ambient Rayon thread count. |
| Elementwise, analytic, reduction, and structural materialization kernels delegated to strided-kernel but could not compile with `parallel` bounds | Auto Fix / Fixed | Added the necessary `Send` / `Sync` bounds to sealed CPU scalar and private helper surfaces so the parallel strided-kernel API is actually compiled. |
| `embed_diagonal`, triangular masks, and indexing-family kernels remain dedicated sequential loops | Intentional Sequential / Documented | Added nearby source comments explaining that these loops do not yet map to a strided-kernel or backend-native parallel primitive. They still run inside `CpuContext::install`, so a future parallel implementation can use the same policy. |
| BLAS/LAPACK threading | Provider-owned | Left unchanged. Docs now distinguish provider-owned BLAS/LAPACK threading from Rayon-backed tenferro/faer work. |
| Active docs said `CpuContext` does not own a Rayon pool | Auto Fix / Fixed | Updated active guide/design docs to match `CpuContext` ownership and session entry behavior. |
| README/publish metadata drift for implementation crates | Auto Fix / Fixed | Updated `scripts/check-publish-layout.py` and the README implementation-crate table so published implementation crates inherit publish metadata while `tenferro-internal-ops` remains explicitly unpublished. |
| CPU concatenate scanned input segment ends for every output element | Auto Fix / Fixed | Replaced the hot-loop linear `.position(...)` scan with `slice::partition_point` over precomputed ordered segment ends. Added a source-contract test for the complexity pattern and kept existing concatenate behavior tests green. |
| Active AD rules exceeded oracle/replay evidence | Policy-doc gap / Fixed | Kept the inactive oracle replay snapshot honest, added crate-local finite-difference coverage for values-only linalg (`SvdVals`, `EighVals`, `EigVals`), einsum, and FFT C2C JVP rules, documented those local coverage rows in `docs/oracle/tensor-ad-oracles-support.md`, and aligned the linalg AD support manifest to `SupportedViaLinearize`. Full-pivot LU oracle coverage landed separately in #1016 and is included through `main`. |
| CUDA linalg `solve` and prepared LU solve could validate dtype or zero fast paths before residency | Source-risk / Fixed | Reordered residency checks ahead of dtype-pair and zero-dimension handling and added source-contract coverage. |
| CubeCL interop downloads could return empty host tensors before checking residency | Source-risk / Fixed | `download_typed_tensor` now validates CubeCL buffer and runtime/device residency before the empty fast path. Added source-contract coverage. |
| CubeCL GEMM zero-contracting fast path built a host zero `Vec` and uploaded it | Source-risk / Fixed | Replaced host zero materialization with device allocation plus the existing device `fill_zero_kernel`. Added source-contract coverage. |
| CUDA LU zero-sized factor parity built a host one `Vec` and uploaded it | Source-risk / Fixed | Added a linalg `fill_one_kernel` and initialized zero-sized LU parity on the device. Added source-contract coverage. |
| CPU `full_piv_lu_solve` could return a zero-sized output before validating dtype-pair support | Source-risk / Fixed | Moved dtype-pair validation ahead of the zero-dimension fast path and added backend-error coverage for mixed and unsupported dtype pairs. |
| CPU gather/scatter index-component lookup allocated an index vector per component | Source-risk / Fixed | Reused caller-owned `index_scratch` across gather/scatter component lookups and added source-contract coverage. |
| LAPACK batched helpers allocated a fresh input `Vec` for every batch slice | Source-risk / Fixed | Reused pooled input scratch tensors and refilled them from each batch slice. Added a source-contract test that rejects per-batch input `to_vec()` copies. |
| CUDA LU helper kernels specialized on matrix-size extent `k` and unrolled loops over it | Source-risk / Fixed | Changed `k` to a runtime kernel argument and replaced unrolled `0..k` loops with runtime `while` loops. Rank and axis-count `#[comptime]` parameters remain intentional because they define indexing structure. |
| Active reference docs still blurred primitive metadata, graph vocabulary, and execution IR ownership | Policy-doc gap / Fixed | Updated active docs to distinguish `tenferro-core-ops` primitive metadata, `tenferro-internal-ops::StdTensorOp`, and `tenferro-runtime::ExecOp`; refreshed the computegraph trait excerpt. |
| Historical design/reference notes were still linked from active indexes | Policy-doc gap / Fixed | Moved superseded migration, linalg, einsum, and external-survey notes to `docs/plans/historical/`; removed them from `docs/design/index.md` and `docs/reference/index.md`. |
| API consistency checker missed some rendered user-facing docs and flagged README implementation-crate inventory as jargon | Tooling gap / Fixed | Expanded the user-doc jargon check to `docs/index.md`, tutorials, and performance docs while keeping internals/spec/architecture out of scope; exempted README's implementation-crate inventory table. |
| Traced `norm`, `pinv`, and `pinv_with_rtol` could encode floating scalar constants as integer/bool tensors | Source-risk / Fixed | Added traced-helper dtype validation so integer and boolean inputs return unsupported-dtype errors before `f64` scalar constants can be rounded or converted to booleans. |
| CPU structural `convert` semantics were tested but not documented for public users | Low/Medium docs gap / Fixed | Documented runtime-dtype conversion semantics in the tensor guide and public convert rustdocs: Rust primitive numeric casts, real-part extraction for complex-to-real/integer, nonzero bool conversion, and zero-imaginary real-to-complex conversion. |
| Traced `eigvals` materialized full general eigendecomposition outputs and discarded eigenvectors | Source-risk / Fixed | Added internal `EigVals` extension op, hidden backend hook, CPU Faer/LAPACK values-only implementations, traced `eigvals` routing, source-contract coverage, and direct `eigvals` JVP regression coverage. |
| #1000 public API and extension-boundary panic risks | Fixed where compatible / Design-gated otherwise | Included the `main` fixes from #1015 for extension runtime output-count validation and materialization error boundaries. Existing panic-capture tests still document `TracedTensor::dot_general` and `extension::apply` panic behavior for invalid builder inputs; changing those to typed errors requires a public API design because the signatures return `TracedTensor` / `Vec<TracedTensor>`, not `Result`. |
| #1000 broader performance/materialization risks | Fixed for accepted current-tree source risks | This PR fixes CPU parallelism wiring, CPU concatenate segment lookup, CPU gather/scatter index scratch allocation, LAPACK batched input scratch reuse, CubeCL GEMM zero host materialization, CUDA LU zero-sized parity host materialization, and LU `k` specialization. |
| #1000 broader docs/tooling drift | Fixed for accepted current-tree docs/tooling findings | This PR fixes stale `CpuContext` docs, publish-layout drift, active crate-ownership docs, historical-doc indexing, and API-consistency scope/false-positive gaps. Snippet-source tooling remains intentionally scoped to `snippet-source` blocks. |

## Decisions Made

- Used the current strided-rs `main` revision because it contains the parallel
  feature wiring and Rayon-backed strided-kernel implementation required by
  #1001.
- Kept `CpuContext` as the only tenferro-owned CPU thread policy source.
- Kept faer using `Par::rayon(0)` only after entering `CpuContext::install`,
  so faer joins the backend-owned Rayon pool rather than selecting a separate
  pool or thread count.
- Left BLAS/LAPACK provider threading provider-owned.
- Did not parallelize the indexing-family and triangular/embedding loops in
  this PR. Their indexing patterns need separate design or backend-native
  helpers before parallelization would be reviewable.
- Treated source-contract tests as appropriate for #1000 cases where the
  accepted finding is a source-risk without a hardware-independent behavior
  reproducer.
- Kept GPU kernel `rank` and axis-count values as `#[comptime]` because they
  define indexing structure, while treating matrix-size extent `k` as runtime
  data to avoid per-shape kernel specialization.
- Did not disable active AD implementations solely because the old root-facade
  oracle replay harness is inactive. Active extension AD rules now have
  explicit crate-local finite-difference coverage rows while the replay
  snapshot still truthfully records replay-adapter support.
- Treated historical design and reference material as archive content rather
  than current design after active-index notes proved too weak to prevent docs
  drift.
- Kept forward `eigvals` values-only but allowed the `EigVals` AD rule to emit
  an internal full `Eig` op during linearization, because the eigenvalue
  derivative needs eigenvectors while the ordinary forward public path does not.

## Verification Performed

- RED: `cargo test -p tenferro-cpu cpu_tensor_kernel_parallel_features_are_wired`
  failed before the implementation because `strided-kernel/parallel` was not
  enabled.
- RED: `cargo test -p tenferro-cpu cached_faer_gemm_pool_helper_enters_owned_rayon_pool`
  failed before the implementation because the cached faer helper observed the
  ambient Rayon pool instead of the configured `CpuContext` pool.
- RED: `python3 scripts/check-publish-layout.py` failed before the publish
  metadata fix because README/publish metadata omitted implementation crates.
- RED: `cargo test --workspace --release` initially exposed that the first
  publish-layout fix incorrectly made `tenferro-internal-ops` inherit the
  workspace publish setting. The final script distinguishes published crates
  from unpublished internal workspace crates.
- RED:
  `cargo test -p tenferro-linalg --test gpu_linalg_source_contract gpu_solve_paths_validate_residency_before_dtype_and_zero_fast_paths`
  failed before the linalg residency-ordering fix.
- RED:
  `cargo test -p tenferro-linalg --features autodiff --test ad_support_manifest linalg_ad_support_manifest_marks_values_only_rules_finite_diff_backed`
  failed before the values-only manifest entries were aligned with finite-diff
  coverage.
- RED:
  `cargo test -p tenferro-cpu --test backend_capability_contracts concatenate_hot_loop_does_not_linearly_scan_input_segments`
  failed before the concatenate lookup change.
- RED:
  `cargo test -p tenferro-gpu --test cubecl_launch_contract cubecl_interop_download_validates_buffer_before_empty_fast_path`
  failed before CubeCL interop downloads validated residency ahead of the empty
  fast path.
- RED:
  `cargo test -p tenferro-gpu --test cubecl_launch_contract cubecl_gemm_zero_contracting_path_stays_device_native`
  failed before the GEMM zero-contracting path stopped materializing host zeros.
- RED:
  `cargo test -p tenferro-linalg --test backend_errors full_piv_lu_solve_rejects_invalid_dtype_pairs_before_zero_dim_fast_path`
  failed before CPU `full_piv_lu_solve` validated dtype pairs before the
  zero-dimension fast path.
- RED:
  `cargo test -p tenferro-linalg --test cpu_linalg_source_contract`
  failed before LAPACK batched helpers stopped allocating fresh input vectors
  per batch.
- RED:
  `cargo test -p tenferro-linalg --test gpu_linalg_source_contract gpu_lu_shape_extent_k_is_runtime_not_compile_time_specialized`
  failed while LU helper kernels still used `#[comptime] k` and unrolled loops.
- RED:
  `cargo test -p tenferro-linalg --test gpu_linalg_source_contract gpu_zero_sized_lu_factor_parity_is_filled_on_device`
  failed before zero-sized LU parity stopped materializing a host one-vector.
- RED:
  `cargo test -p tenferro-cpu --test backend_capability_contracts gather_scatter_index_component_reuses_index_scratch`
  failed before gather/scatter index-component lookup reused caller-owned
  scratch.
- RED:
  `cargo test -p tenferro-linalg --features autodiff --test ad_support_manifest linalg_values_only_finite_diff_support_is_documented_next_to_oracle_snapshot`
  failed before the oracle snapshot documented crate-local finite-difference
  coverage for values-only linalg rules.
- RED:
  `cargo test -p tenferro-linalg --test traced_extension traced_`
  failed before traced `norm` and `pinv` rejected integer/bool inputs.
- RED:
  `cargo test -p tenferro-linalg --test linalg_internal_path_contract`
  failed before traced `eigvals` emitted `EigVals` and the backend surface/CPU
  backend exposed an `eig_values` hook.
- GREEN: `cargo test -p tenferro-cpu`
- GREEN: `cargo test -p tenferro-cpu concatenate`
- GREEN: `cargo test -p tenferro-gpu --test cubecl_launch_contract`
- GREEN: `cargo test -p tenferro-linalg --test gpu_linalg_source_contract`
- GREEN: `cargo test -p tenferro-linalg --test cpu_linalg_source_contract`
- GREEN: `cargo test -p tenferro-linalg --test backend_errors`
- GREEN: `cargo test -p tenferro-linalg --test traced_extension traced_`
- GREEN: `cargo test -p tenferro-linalg --test linalg_internal_path_contract`
- GREEN: `cargo test -p tenferro-linalg --test traced_correctness`
- GREEN: `cargo test -p tenferro-linalg --features autodiff --test traced_ad_explicit`
- GREEN:
  `cargo test -p tenferro-einsum --features autodiff --test traced_ad_migration grad_einsum_matmul_real_matches_finite_diff_for_both_inputs`
- GREEN:
  `cargo test -p tenferro-fft --features autodiff --test fft_ops fft_c64_jvp_matches_finite_diff`
- GREEN: `cargo check -p tenferro-linalg --features cpu-blas`
- GREEN:
  `cargo check -p tenferro-linalg --no-default-features --features cpu-blas`
- GREEN:
  `cargo check -p tenferro-linalg --no-default-features --features cpu-faer`
- GREEN: `cargo check -p tenferro-linalg --features cuda`
- BLOCKED:
  `cargo test -p tenferro-linalg --no-default-features --features cpu-blas --test backend_errors cpu_values_only_decompositions_cover_real_complex_and_batched_inputs`
  could not link in this environment because LAPACK symbols such as `sgeev_`,
  `dgeev_`, and `sgetc2_` were unavailable to the local linker.
- GREEN:
  `cargo test -p tenferro-linalg --features autodiff --test ad_support_manifest`
- GREEN: `cargo test -p tenferro-internal-ops --test publication_contract`
- GREEN: `python3 scripts/check-publish-layout.py`
- Feature graph check:
  `cargo tree -p tenferro-cpu -e features -i strided-kernel`
- GREEN: `cargo fmt --all --check`
- GREEN: `cargo clippy --workspace --all-targets -- -D warnings`
- GREEN:
  `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- GREEN: `cargo test --workspace --release`
- GREEN: `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- GREEN: `python3 scripts/check-coverage.py coverage.json`
- GREEN: `cargo doc --workspace --no-deps`
- GREEN: `python3 scripts/check-doc-snippets.py --check`
- GREEN: `python3 scripts/check-docs-site.py`
- GREEN: `python3 scripts/check-api-consistency.py --fail-on-findings`
- GREEN:
  `cargo test -p tenferro-cpu --test inject_tests --release --no-default-features --features "cpu-blas,provider-inject"`
- GREEN: `git diff --check`

## Remaining Risks

- No strict speedup assertion was added. The regression coverage verifies
  feature wiring and pool entry; performance benchmarking should be done with
  pinned thread counts when reviewing actual speedups.
- Dedicated indexing and triangular/embedding loops remain sequential by
  design for this PR.
- Changing non-`Result` public tracing builders such as `dot_general` and
  `extension::apply` to typed errors remains a public API design item rather
  than a compatible bug-fix edit.
