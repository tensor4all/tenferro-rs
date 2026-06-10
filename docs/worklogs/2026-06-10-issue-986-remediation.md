# Issue 986 Remediation Pass

## Summary

This work log tracks the batched remediation pass for issue #986 against
`origin/main` commit `043259ab5cc46dfc665159c02a480bdfb2fac8a9`.

The first implemented batch fixed docs and rustdoc issues that were safe for
automatic remediation. Broader public API, AD policy, and public cache/runtime
boundary changes remain design-gated.

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
- `remaining-auto`: classified as automatically fixable, not yet implemented

| Finding | Classification | Current status |
| --- | --- | --- |
| #2 public `tenferro_gpu::cubecl`/doc-hidden modules | Design Gate | design-gated |
| #3 stale `tenferro_ops::ExtensionFamilyId` docs | Auto Fix | fixed |
| #4 coverage policy vs thresholds | Design Gate | design-gated |
| #6 FFT C2C transpose/oracle coverage | Auto Fix | fixed; existing C2C rule used the wrong adjoint convention |
| #10 `TensorDeviceTransfer` clone defaults | Design Gate | design-gated |
| #12 compile-cache string fingerprint | Auto Fix | fixed |
| #13 segment later-instruction scans | Auto Fix | fixed |
| #16 publishable internal `tenferro_ops` crate | Design Gate | design-gated |
| #19 public tensor representation hooks | Design Gate | design-gated |
| #20 `ReduceProd` zero-input AD | Auto Fix | fixed |
| #21 broad indexing AD coverage | Verify First | verify-first |
| #29 README ROCm support overclaim | Auto Fix | fixed |
| #32 public `TracedTensor` fields | Design Gate | design-gated |
| #37 traced graph builder clones | Verify First | verify-first |
| #40 FFT output zero-initialization | Auto Fix | fixed |
| #42/#43 complex `Abs`/`Sign` AD | Verify First | verify-first; split `Abs` from `Sign` policy |
| #44/#52/#101 shape-source arity AD | Auto Fix | fixed; narrowed to current `transpose_reshape` arity gap plus verification of existing broadcast arity behavior |
| #49 public fusion IR backend API | Design Gate | design-gated |
| #56/#73 executor workspace allocation/clones | Verify First | verify-first |
| #65 explicit singleton `BroadcastInDim` VJP | Auto Fix | fixed |
| #78/#79 diagonal VJP edge cases | Auto Fix | fixed |
| #80 repeated-label einsum VJP | Auto Fix | fixed; verified with helper-level repeated-label projection regression |
| #87 FFT under CUDA-backed execution | Verify First | stale/docs-only; current behavior rejects device/backend-buffer input with a clear host-only diagnostic |
| #106 integer/bool/lossy `Convert` AD policy | Design Gate | design-gated |
| #109 lazy value/view API rustdoc examples | Auto Fix | fixed |
| #110 `LuSolvePrepared` mixed complex adjoint flags | Auto Fix | fixed |
| #111 `TracedTensorAdExt` rustdoc examples | Auto Fix | fixed |
| #112 CUDA zero-sized-output residency validation | Auto Fix | fixed |
| #113 gather/scatter boundary VJP | Verify First | verify-first; narrow to current boundary cases |
| #114 `Pow` zero-base singularity | Auto Fix | fixed |
| #115 terminal lazy view base clone | Auto Fix | fixed |
| #116 scalar helper public contract | Design Gate | design-gated |
| #116 mixed real/complex binary VJPs | Auto Fix | fixed for binary arithmetic `Add`/`Mul`/`Div`/`Pow`; verified with direct mixed primitive JVP/VJP regressions for both real-input positions |
| #117 einsum fallback greedy `HashSet` rebuild | Auto Fix | fixed |
| #117 eager helper host tensors in CUDA contexts | Auto Fix | fixed |
| #118 active crate/backend ownership docs | Auto Fix | fixed |
| #119 primitive catalog `Constant` docs | Auto Fix | fixed |
| #120 active in-place indexing ChainRules framing | Auto Fix | fixed |
| #121 einsum retained-byte cache accounting | Design Gate | design-gated |
| #122 CPU provider-injection rustdoc placeholders | Auto Fix | fixed |
| #123 eager extension materializes lazy/view tensors | Design Gate | design-gated |
| #124 oracle docs active CI/replay claims | Auto Fix | fixed |
| #125 max/min/clamp boundary AD convention | Design Gate | design-gated |
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

`cargo doc --workspace --no-deps` emitted an existing private intra-doc link
warning in `crates/tenferro-runtime/src/shape_infer.rs`.

## Residual Risks

- The remaining unresolved items are verify-first or design-gated rather than
  straightforward auto-fix items.
- Design-gated items should become focused design or child issues rather than
  being implemented silently in this remediation branch.
- CUDA zero-output validation was verified with source-contract tests and
  CUDA-feature compile-only checks, not with CUDA hardware execution.
- The eager and traced scalar-default device-placement fixes were verified with
  source-contract tests and CPU behavior tests, not CUDA hardware execution.
  Non-scalar graph default inputs remain non-uploaded to avoid silently moving
  user-sized tensors across device boundaries.
- Mixed real/complex binary arithmetic AD is fixed for `Add`/`Mul`/`Div`/`Pow`.
  The same direct mixed-dtype projection issue is also fixed for `DotGeneral`.
  Similar risks remain in boundary-sensitive elementwise ops, but they require
  #125 semantics before implementation.
