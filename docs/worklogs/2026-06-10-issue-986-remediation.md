# Issue 986 Remediation Pass

## Summary

This work log tracks the batched remediation pass for issue #986 against
`origin/main` commit `043259ab5cc46dfc665159c02a480bdfb2fac8a9`.

The first implemented batch fixed docs and rustdoc issues that were safe for
automatic remediation. Later batches fixed runtime, performance, device,
structural AD, complex AD, public-surface, runtime/cache-boundary, and
JAX-compatible boundary AD items. Remaining unresolved items are broad
verify-first coverage/performance investigations or repository policy gates,
not silently deferred auto-fix items.

## Context Read

- `AGENTS.md`
- `REPOSITORY_RULES.md`
- `ai/contribution-workflows/repository-remediation.md`
- GitHub issue #986 body and historical comments
- GitHub issue #986 verification comment
  `2026-06-10T00:03:13Z`, which re-checked queue items against
  `origin/main` `043259ab5cc46dfc665159c02a480bdfb2fac8a9`
- Current source and docs referenced by the classification table below

## Subagents

Read-only explorer subagents classified independent domains:

- public API, rustdoc, and active docs
- AD semantics and oracle coverage
- runtime, performance, layout, and cache
- GPU and device placement

Worker subagents implemented method-level rustdoc examples in disjoint files:

- `crates/tenferro-runtime/src/graph/executor.rs`
- `crates/tenferro-ad/src/traced.rs`

Worker subagents also implemented disjoint runtime/performance fixes:

- `crates/tenferro-runtime/src/segment.rs`
- `crates/tenferro-fft/src/lib.rs`
- `crates/tenferro-einsum/src/planning/tree.rs`

The coordinating agent reviewed and integrated the results.

Later worker subagents implemented disjoint AD fixes:

- C2C FFT transpose convention and regression tests
- structural `Reshape`/`BroadcastInDim` transpose arity and singleton broadcast
  VJP tests
- diagonal extraction/embedding VJP edge cases and eager finite-difference tests

## Classification Ledger

Status values:

- `fixed`: fixed in this remediation branch
- `stale`: current worktree no longer has the historical issue
- `verify-first`: needs a narrower current failing path before implementation
- `design-gated`: requires maintainer/design decision before code changes
- `design-gate accepted`: maintainer discussion resolved the design gate and
  the item was implemented in this remediation branch
- `remaining-auto`: classified as automatically fixable, not yet implemented

| Finding | Classification | Current status |
| --- | --- | --- |
| #2 public `tenferro_gpu::cubecl`/doc-hidden modules | Design Gate Accepted | fixed; CubeCL implementation modules and raw buffer internals are no longer exposed as public API |
| #3 stale `tenferro_ops::ExtensionFamilyId` docs | Auto Fix | fixed |
| #4 coverage policy vs thresholds | Design Gate | design-gated |
| #6 FFT C2C transpose/oracle coverage | Auto Fix | fixed; existing C2C rule used the wrong adjoint convention |
| #10 `TensorDeviceTransfer` clone defaults | Design Gate Accepted | fixed; backend-buffer transfers now require explicit backend behavior and CPU rejects backend buffers clearly |
| #12 compile-cache string fingerprint | Auto Fix | fixed |
| #13 segment later-instruction scans | Auto Fix | fixed |
| #16 publishable internal `tenferro_ops` crate | Design Gate Accepted | fixed; internal ops crate is no longer publishable |
| #19 public tensor representation hooks | Design Gate Accepted | fixed; low-level tensor representation fields moved behind narrower accessors |
| #20 `ReduceProd` zero-input AD | Auto Fix | fixed |
| #21 broad indexing AD coverage | Verify First | verify-first |
| #29 README ROCm support overclaim | Auto Fix | fixed |
| #32 public `TracedTensor` fields | Design Gate Accepted | fixed; traced tensor internals moved behind accessors |
| #37 traced graph builder clones | Stale / Upstream Fixed | stale; computegraph `origin/main` already contains shared `Arc<OperationKey>` keys and cached fingerprints, and tenferro pins that revision |
| #40 FFT output zero-initialization | Auto Fix | fixed |
| #42 complex `Abs` AD convention | Auto Fix After Contract Decision | fixed; complex `Abs` now follows JAX real-output convention (`C32 -> F32`, `C64 -> F64`) with matching JVP/VJP |
| #43 complex `Sign` AD convention | Contract/Test Fix | fixed; AD contract documents JAX zero-AD convention and regression coverage confirms complex `Sign` JVP/VJP are zero |
| #44/#52/#101 shape-source arity AD | Auto Fix | fixed; narrowed to current `transpose_reshape` arity gap plus verification of existing broadcast arity behavior |
| #49 public fusion IR backend API | Design Gate Accepted | fixed; fusion IR backend internals are no longer top-level public API |
| #56/#73 executor workspace allocation/clones | Verify First | verify-first |
| #65 explicit singleton `BroadcastInDim` VJP | Auto Fix | fixed |
| #78/#79 diagonal VJP edge cases | Auto Fix | fixed |
| #80 repeated-label einsum VJP | Auto Fix | fixed; verified with helper-level repeated-label projection regression |
| #87 FFT device/backend-buffer input boundary | Stale / Out of Scope | stale; current behavior rejects device/backend-buffer input with a clear host-only diagnostic |
| #106 integer/bool/lossy `Convert` AD policy | Design Gate Accepted | fixed; follows JAX-like `float0` convention for integer/bool boundaries while floating/complex casts remain differentiable |
| #109 lazy value/view API rustdoc examples | Auto Fix | fixed |
| #110 `LuSolvePrepared` mixed complex adjoint flags | Auto Fix | fixed |
| #111 `TracedTensorAdExt` rustdoc examples | Auto Fix | fixed |
| #112 CUDA zero-sized-output residency validation | Auto Fix | fixed |
| #113 gather/scatter boundary VJP | Contract Fix | fixed by contract; indexing AD follows JAX/StableHLO-style `promise_in_bounds`, so gradients are guaranteed for in-bounds indices only |
| #114 `Pow` zero-base singularity | Auto Fix | fixed |
| #115 terminal lazy view base clone | Auto Fix | fixed |
| #116 scalar helper public contract | Design Gate Accepted | fixed; low-level scalar conjugation/conversion hooks are no longer exposed as public tensor API |
| #116 mixed real/complex binary VJPs | Auto Fix | fixed for binary arithmetic `Add`/`Mul`/`Div`/`Pow`; verified with direct mixed primitive JVP/VJP regressions for both real-input positions |
| #117 einsum fallback greedy `HashSet` rebuild | Auto Fix | fixed |
| #117 eager helper host tensors in CUDA contexts | Auto Fix | fixed |
| #118 active crate/backend ownership docs | Auto Fix | fixed |
| #119 primitive catalog `Constant` docs | Auto Fix | fixed |
| #120 active in-place indexing ChainRules framing | Auto Fix | fixed |
| #121 einsum retained-byte cache accounting | Design Gate Accepted | fixed; extension cache retained bytes can be recomputed from mutated cached values |
| #122 CPU provider-injection rustdoc placeholders | Auto Fix | fixed |
| #123 eager extension materializes lazy/view tensors | Design Gate Accepted | fixed; eager extension dispatch now uses `TensorRead` and read-capable runtimes can consume views directly |
| #124 oracle docs active CI/replay claims | Auto Fix | fixed |
| #125 max/min/clamp boundary AD convention | Auto Fix After Contract Decision | fixed; `Maximum`/`Minimum` ties split cotangents like JAX and `Clamp` uses strict JAX boundary masks |
| #126 device AD seeds and missing tangents | Auto Fix | fixed for eager seed/missing-tangent helpers and traced rank-0 default inputs; non-scalar default input auto-upload remains intentionally out of scope |

## First Batch: Docs And Rustdoc

Implemented:

- Updated active docs for extension family imports, ROCm status, tensor/CPU/GPU
  crate ownership, `Constant` IR status, in-place indexing AD vocabulary,
  linalg GPU ownership, testing AD vocabulary, and oracle replay CI status.
- Replaced undefined CPU provider-injection doctest placeholders with
  compiling empty-set registration examples.
- Added method-level examples for `TracedTensorAdExt`.
- Added method-level examples for lazy output and borrowed-input executor APIs.
- Added missing API index links for internal implementation crates required by
  `scripts/check-docs-site.py`.

## Second Batch: Runtime And Operation Performance

Implemented:

- Replaced segment construction's repeated later-instruction scans with a
  single `SegmentUseSummary` pass that records program outputs and last input
  use by slot.
- Replaced fully overwritten FFT CPU output zero-fill with `MaybeUninit`
  output buffers while keeping zero-filled scratch lanes for padded FFT input.
- Reworked the einsum self-greedy fallback to maintain live needed-label
  counts instead of rebuilding a `HashSet` for every candidate pair.
- Removed the obsolete private `contraction_cost` helper after replacing its
  hot-path use, matching the remediation workflow's rule against artificial
  dead-code references.

## Third Batch: Compile Cache Key

Implemented:

- Replaced graph compile-cache `String`/`Debug` fingerprints with private
  structural keys for `ExecProgram`, `ExecInstruction`, and `ExecOp`.
- Preserved extension payload collision safety by keeping extension payloads in
  `CacheKey` and using `payload_eq` for final equality after family and payload
  hash checks.
- Updated compile-cache retained-byte accounting to include structural key
  payloads instead of string capacity.

## Fourth Batch: Terminal Lazy View Sharing

Implemented:

- Changed non-consuming terminal lazy-view input conversion from deep-cloning a
  live `Owned(Tensor)` to promoting the slot into a shared `Arc<Tensor>` held
  by both the slot and returned `TensorValue`.
- Preserved existing borrowed `TensorRead` behavior, which must still
  materialize when an owned lazy output is required.

## Fifth Batch: Pow Zero-Base AD

Implemented:

- Replaced `Pow` base-side AD coefficients of the form `y * pow(x, y) / x`
  with `y * pow(x, y - 1)` in both `linearize_pow` and `transpose_pow`.
- Added a regression for `x=0, y=2` in the traced gradient path, including
  finite-difference verification for the base cotangent.

## Sixth Batch: ReduceProd Zero AD

Implemented:

- Replaced `ReduceProd` AD coefficients of the form `product / input` with a
  zero-safe coefficient builder used by both JVP and VJP.
- Added single-zero and multiple-zero regression coverage for reduced axes.

## Seventh Batch: CUDA Zero-Output Validation

Implemented:

- Moved CubeCL input/output residency and buffer argument validation before
  zero-sized-output early returns in raw unary, tensor unary, nullary-into,
  tensor-into, binary, comparison, logical tensor binary, and ternary launch
  helpers.
- Applied the same ordering to elementwise fusion launch inputs/outputs.
- Added a non-hardware source-contract test that checks validation happens
  before the zero-output shortcut.

## Eighth Batch: FFT C2C Transpose Convention

Implemented:

- Reclassified the historical FFT C2C item from "missing transpose rule" to
  "existing transpose rule uses the wrong adjoint convention."
- Changed C2C FFT transpose to emit the opposite transform direction with the
  adjoint normalization (`Backward` <-> `Forward`, `Ortho` unchanged).
- Added C64 VJP regressions for both `fft(..., FftNorm::Backward)` and
  `ifft(..., FftNorm::Backward)`.

## Ninth Batch: Structural And Diagonal AD Edges

Implemented:

- Updated `Reshape` transpose to return one cotangent slot per primal input,
  including `None` for dynamic shape-source inputs.
- Kept existing shape-source handling for `BroadcastInDim` and added explicit
  singleton-axis VJP reduction followed by a reshape back to the input shape.
- Fixed rectangular `ExtractDiag` VJP by padding the embedded cotangent back to
  both diagonal axes of the primal input shape.
- Fixed shifted-axis `EmbedDiag` VJP by extracting the shifted source diagonal
  and transposing the result back when insertion occurs before the source axis.
- Added focused structural helper tests and eager finite-difference diagonal
  regressions.

## Tenth Batch: Eager Device-Placement Helpers

Implemented:

- Changed eager `index_select` to upload hidden generated index tensors through
  the eager backend before importing them as constants and dispatching gather.
- Changed eager `Constant` and `ShapeOf` execution helpers to upload generated
  host tensors through the active backend before returning them to eager
  execution.
- Changed eager AD scalar seeds and missing-tangent zeroes to allocate the host
  zero tensor and then upload it through the active backend before use.
- Added non-hardware source-contract tests covering hidden eager indices,
  eager generated constants/shape scalars, and eager AD zero seed upload.
- Left traced `grad()` scalar seed placement as a residual path for the later
  graph-executor input placement fix.

## Eleventh Batch: Remaining AD Edge Fixes

Implemented:

- Fixed `LuSolvePrepared` adjoint flag handling so transpose-only and
  conjugate-only prepared solves preserve the missing flag in the adjoint
  solve instead of collapsing to `(false, false)`.
- Fixed repeated-label einsum VJP projection so labels repeated three or more
  times project each extra occurrence back to the first axis for that label.
- Fixed mixed real/complex binary arithmetic AD so direct primitive
  `Add`/`Mul`/`Div`/`Pow` graphs convert linearized tangents and fixed
  coefficients to the promoted output dtype, project complex cotangents back to
  real tangent spaces, and avoid evaluating raw mixed primal `Div`/`Pow`
  outputs when building coefficients.
- Fixed direct mixed real/complex `DotGeneral` AD so JVP tangents and fixed
  contraction operands are promoted before backend execution and VJP cotangents
  are projected back to real inputs after any transpose needed to restore input
  layout.

Residual:

- The repeated-label einsum regression is helper-level because an attempted
  public end-to-end VJP regression exposed a separate symbolic shape issue.
- Mixed real/complex AD risks remain outside the binary arithmetic slice:
  `Maximum`/`Minimum`/`Select`/`Clamp` need boundary-convention decisions
  tracked under #125 before changing tie/boundary semantics.
- The AD-local dtype-promotion helper intentionally mirrors the runtime helper
  for this patch because `tenferro-internal-ops` cannot depend on
  `tenferro-runtime`; a lower-crate shared promotion helper remains a cleanup
  follow-up if duplication becomes broader.

## Twelfth Batch: Traced Scalar Default Placement

Implemented:

- Changed `GraphExecutor` default-input resolution to route rank-0 host default
  tensors through `backend.upload_host_tensor()` before execution, covering
  traced `grad()` scalar cotangent seeds stored as default inputs.
- Applied the same rule to owned and borrowed-input execution paths.
- Preserved the no-hidden-transfer boundary by leaving explicit bindings,
  non-scalar default inputs, and already backend-resident scalar defaults
  untouched.
- Added source-contract tests with an upload-rejecting backend for owned and
  borrowed scalar defaults plus guard tests for explicit scalar bindings,
  non-scalar defaults, and backend-resident scalar defaults.

## Thirteenth Batch: JAX-Compatible Elementwise AD Contracts

Implemented:

- Audited JAX's current `abs`, `sign`, `max`, `min`, and `clamp` AD rules and
  used them as the tie/boundary convention where tenferro's contract had been
  ambiguous.
- Changed complex `Abs` primal dtype inference and traced execution metadata
  to return real magnitudes (`C32 -> F32`, `C64 -> F64`) instead of
  real-embedded complex tensors.
- Updated CPU complex `Abs` execution to return real tensors and added traced
  dtype/execution regressions.
- Updated complex `Abs` JVP/VJP to use JAX's real-output convention:
  `d abs(z) = Re(conj(sign(z)) * dz)` and `z_bar = abs_bar * sign(z)`.
- Documented and tested complex `Sign` as a zero-AD operation.
- Updated `Maximum`/`Minimum` JVP/VJP to split contributions equally at ties,
  matching JAX's balanced equality rule.
- Updated `Clamp` JVP/VJP to use strict JAX boundary masks, so exact lower and
  upper boundaries receive zero contribution.
- Documented the indexing AD `promise_in_bounds` contract in
  `docs/spec/ad-contract.md` and the indexing AD implementation comments.
- Reclassified computegraph clone issue #37 as stale/upstream-fixed after
  verifying the pinned computegraph revision already shares derived operation
  keys via `Arc`.

## Fourteenth Batch: Public Surface And Transfer Boundaries

Implemented:

- Hid CUDA/CubeCL implementation modules and raw buffer internals that were
  previously exposed as public API, while preserving the intended high-level
  CUDA backend entry points.
- Marked the internal ops crate as non-publishable instead of presenting
  `tenferro_ops` as an external SDK surface.
- Moved tensor representation fields and traced tensor internals behind
  narrower accessors.
- Removed top-level public exposure of fusion IR backend implementation
  details.
- Kept low-level scalar conjugation/conversion helpers internal to their
  owning backend/tensor layers.
- Replaced implicit transfer defaults with explicit backend behavior and clear
  CPU errors for backend-buffer transfers.

## Fifteenth Batch: Convert, Cache, And Eager Extension Read Boundaries

Implemented:

- Defined `Convert` AD to follow JAX's `float0`-style inactive tangent
  convention at integer/bool boundaries while keeping floating/complex casts
  differentiable.
- Changed extension cache retained-byte accounting so entries can report
  dynamic retained bytes after nested backend caches grow.
- Added a read-capable `ExtensionRuntime` dispatch boundary. Existing
  tensor-only runtimes keep an explicit materializing fallback, while
  read-aware runtimes can consume `TensorRead` views directly.
- Routed eager extension dispatch through `TensorRead` instead of
  `EagerTensor::materialized_arc()`.
- Added an einsum runtime read override for eager extension dispatch. Compact
  tensor inputs still use the existing cached runtime program path; view inputs
  use the read-capable eager einsum executor.
- Added a README link to the official
  `tensor4all/tenferro-benchmark` benchmark suite.
- Added broad graph compile-cache key payload coverage so the repository
  coverage gate covers structural key variants and retained-byte accounting.
- Removed a stale private intra-doc link in runtime shape-inference docs.

## Verification

- `cargo fmt --all --check`
- `cargo test -p tenferro-ad --doc`
- `cargo test -p tenferro-runtime --doc`
- `cargo test -p tenferro-cpu --features provider-inject --doc`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
- `git diff --check`
- `cargo test -p tenferro-einsum self_greedy`
- `cargo test -p tenferro-fft --test fft_ops`
- `cargo test -p tenferro-runtime segment`
- `cargo test -p tenferro-runtime graph::cache`
- `cargo test -p tenferro-runtime compile_cache`
- `cargo test -p tenferro-einsum --test traced_graph_cache`
- `cargo test -p tenferro-runtime lazy_view_input_conversion_shares_live_owned_tensor`
- `cargo test -p tenferro-runtime lazy`
- `cargo test -p tenferro-runtime value`
- `cargo test -p tenferro-ad grad_pow`
- `cargo test -p tenferro-ad reduce_prod_ --test ad`
- `cargo test -p tenferro-gpu --test cubecl_launch_contract cubecl_zero_length_launches_validate_buffers_before_returning`
- `cargo test -p tenferro-gpu`
- `cargo test -p tenferro-gpu --features cuda --no-run`
- `cargo fmt --all --check`
- `cargo test -p tenferro-internal-ops --features autodiff structural_tests -- --nocapture`
- `cargo test -p tenferro-ad --test ad_structural_primitives diag`
- `cargo test -p tenferro-ad --test ad diag`
- `cargo test -p tenferro-fft --features autodiff --test fft_ops`
- `git diff --check -- crates/tenferro-internal-ops/src/ad/structural.rs crates/tenferro-internal-ops/src/ad/registry.rs crates/tenferro-internal-ops/src/ad/diagonal.rs crates/tenferro-internal-ops/src/ad/tests/mod.rs crates/tenferro-internal-ops/src/ad/tests/structural_tests.rs crates/tenferro-ad/tests/ad_structural_primitives.rs crates/tenferro-fft/src/lib.rs crates/tenferro-fft/tests/fft_ops.rs`
- `cargo test -p tenferro-ad --test eager_device_placement_contract`
- `cargo test -p tenferro-ad zero_like_tensor_covers_non_f64_dtypes`
- `cargo test -p tenferro-ad constant_from_creates_untracked_leaf`
- `cargo test -p tenferro-ad eager_forward_helpers_synthesize_tangent_values_from_primal_data`
- `cargo test -p tenferro-ad --test eager_tensor index_select`
- `cargo test -p tenferro-ad grad --test ad`
- `cargo test -p tenferro-ad eager_exec::tests`
- `cargo test -p tenferro-ad --features cuda --no-run`
- `git diff --check -- crates/tenferro-ad/src/eager.rs crates/tenferro-ad/src/eager_builder.rs crates/tenferro-ad/src/eager_exec.rs crates/tenferro-ad/src/shape_packing.rs crates/tenferro-ad/tests/eager_device_placement_contract.rs`
- `cargo test -p tenferro-linalg --features autodiff adjoint_lu_solve_flags_preserve_mixed_complex_adjoint_cases`
- `cargo test -p tenferro-einsum --features autodiff repeated_label_projection_projects_each_extra_occurrence`
- `git diff --check -- crates/tenferro-linalg/src/ad/rules/solve.rs crates/tenferro-linalg/src/ad/rules/solve/tests.rs crates/tenferro-einsum/src/extension.rs crates/tenferro-einsum/src/extension/tests.rs crates/tenferro-ad/tests/ad.rs`
- `cargo test -p tenferro-ad mixed_ --test ad`
- `cargo test -p tenferro-ad complex_div_and_pow_vjps_conjugate_holomorphic_coefficients --test ad`
- `cargo test -p tenferro-ad convert_eval_jvp_and_vjp_follow_real_complex_adjoint_rules --test ad`
- `cargo test -p tenferro-ad grad --test ad`
- `cargo test -p tenferro-ad --test ad`
- `cargo test -p tenferro-internal-ops --features autodiff`
- `cargo test -p tenferro-runtime`
- `cargo test -p tenferro-runtime --test graph_default_input_placement`
- `cargo fmt --all --check`
- `git diff --check`
- `cargo test -p tenferro-ad test_abs_complex_outputs_real_dtype --test dtype_propagation`
- `cargo test -p tenferro-ad traced_abs_of_complex_tensor_returns_real_tensor --test primitive_ops`
- `cargo test -p tenferro-cpu test_tier2_elementwise_ops_complex --lib`
- `cargo test -p tenferro-ad complex_abs_ad_matches_jax_real_output_convention --test ad`
- `cargo test -p tenferro-ad complex_sign_ad_is_zero_like_jax --test ad`
- `cargo test -p tenferro-ad elementwise_extrema_ties_split_cotangents_like_jax --test ad`
- `cargo test -p tenferro-ad clamp_ad_uses_strict_jax_boundary_masks --test ad`
- `cargo test -p tenferro-internal-ops --features autodiff elementwise_tests`
- `cargo test -p tenferro-cpu --test runtime_error_tests cpu_device_transfer_rejects_backend_buffers_at_boundary`
- `cargo test -p tenferro-ad --test dynamic_truncate dynamic_truncate_rejects_backend_size_binding_on_cpu`
- `cargo test -p tenferro-ad --test ad convert_ad_treats_integer_and_bool_boundaries_as_inactive_like_jax_float0`
- `cargo test -p tenferro-ad --test ad convert_eval_jvp_and_vjp_follow_real_complex_adjoint_rules`
- `cargo test -p tenferro-runtime extension_cache::tests --lib`
- `cargo test -p tenferro-einsum --features autodiff --test traced_graph_cache runtime_planned_einsum_reuses_extension_runtime_caches`
- `cargo test -p tenferro-einsum --features autodiff --test traced_graph_cache extension_runtime_cache_limits_bound_runtime_planned_einsum_entries`
- `cargo test -p tenferro-ad eager_extension_dispatch_does_not_initialize_lazy_view_materialization_cache --lib`
- `cargo test -p tenferro-runtime --test extension_runtime`
- `cargo test -p tenferro-internal-extension-macros extension_runtime_macro_generates_optional_read_executor`
- `cargo test -p tenferro-einsum execute_einsum_extension_reads_consumes_strided_view_inputs --lib`
- `cargo test -p tenferro-einsum --features autodiff generic_outer_product_uses_broadcast_views_without_materialized_broadcast_ops --lib`
- `cargo test -p tenferro-einsum --features autodiff tensor_value_view_paths_materialize_and_read --lib`
- `cargo test -p tenferro-cpu --release tensor_read_elementwise_dispatch_covers_view_and_complex_scalar_branches --lib`
- `cargo test -p tenferro-runtime graph::cache::tests::cache_key_and_stats_cover_exec_op_payload_variants --lib`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`

## Residual Risks

- Remaining unresolved items are broad verify-first coverage/performance
  investigations or repository-policy items that need a separate policy
  decision, not known narrow auto-fix items.
- CUDA zero-output validation was verified with source-contract tests and
  CUDA-feature compile-only checks, not with CUDA hardware execution.
- The eager and traced scalar-default device-placement fixes were verified with
  source-contract tests and CPU behavior tests, not CUDA hardware execution.
  Non-scalar graph default inputs remain non-uploaded to avoid silently moving
  user-sized tensors across device boundaries.
- Mixed real/complex binary arithmetic AD is fixed for `Add`/`Mul`/`Div`/`Pow`.
  The same direct mixed-dtype projection issue is also fixed for `DotGeneral`.
  Boundary-sensitive `Maximum`/`Minimum`/`Clamp` AD now follows the documented
  JAX-compatible convention.
- Extension runtimes that do not override the new read execution entry point
  still materialize view inputs at the explicit runtime ABI fallback. The
  standard einsum runtime overrides the read path; linalg and FFT remain
  tensor-only because their current backend contracts require compact tensors
  or host FFT inputs.

## PR-Pre Residual Audit

Two read-only subagents audited the unresolved ledger before PR creation. They
found no additional `remaining-auto` item. The stale FFT device/backend-buffer
finding was removed from the active fix queue because current FFT code rejects
device and backend-buffer inputs with a host-only diagnostic.

Remaining `Verify First` items:

- #21 indexing AD coverage: broad coverage debt, with no narrow failing path
  identified in this pass.
- #56/#73 executor workspace allocation/clones: owned execution paths already
  reuse workspace; borrowed paths still need performance/lifetime investigation.

Remaining design-gated groups:

- Repository policy decisions: #4 coverage policy.

Resolved during the final contract and boundary passes:

- #2, #16, #19, #32, #49, and scalar-helper #116: public surface narrowed
  after the compatibility decision that backward compatibility is not required.
- #10 transfer defaults: explicit backend behavior now replaces implicit clone
  defaults.
- #37 traced graph builder clones: upstream-fixed in computegraph and already
  pinned by tenferro.
- #42 complex `Abs`: JAX-compatible real-output convention implemented.
- #43 complex `Sign`: JAX zero-AD convention documented and tested.
- #106 `Convert`: JAX `float0`-like inactive integer/bool boundary contract
  documented and implemented.
- #113 gather/scatter boundary VJP: resolved as an in-bounds AD contract via
  `promise_in_bounds`.
- #121 retained-byte accounting: extension cache entries can dynamically
  report nested cache growth.
- #123 eager extension materialization: eager extension dispatch now uses
  `TensorRead`, with an einsum read-capable runtime override.
- #125 max/min/clamp boundaries: JAX-compatible tie split and strict clamp
  masks implemented.
