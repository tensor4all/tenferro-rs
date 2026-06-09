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
| #6 FFT C2C transpose/oracle coverage | Auto Fix | remaining-auto |
| #10 `TensorDeviceTransfer` clone defaults | Design Gate | design-gated |
| #12 compile-cache string fingerprint | Auto Fix | fixed |
| #13 segment later-instruction scans | Auto Fix | fixed |
| #16 publishable internal `tenferro_ops` crate | Design Gate | design-gated |
| #19 public tensor representation hooks | Design Gate | design-gated |
| #20 `ReduceProd` zero-input AD | Auto Fix | remaining-auto |
| #21 broad indexing AD coverage | Verify First | verify-first |
| #29 README ROCm support overclaim | Auto Fix | fixed |
| #32 public `TracedTensor` fields | Design Gate | design-gated |
| #37 traced graph builder clones | Verify First | verify-first |
| #40 FFT output zero-initialization | Auto Fix | fixed |
| #42/#43 complex `Abs`/`Sign` AD | Verify First | verify-first; split `Abs` from `Sign` policy |
| #44/#52/#101 shape-source arity AD | Auto Fix | remaining-auto |
| #49 public fusion IR backend API | Design Gate | design-gated |
| #56/#73 executor workspace allocation/clones | Verify First | verify-first |
| #65 explicit singleton `BroadcastInDim` VJP | Auto Fix | remaining-auto |
| #78/#79 diagonal VJP edge cases | Auto Fix | remaining-auto |
| #80 repeated-label einsum VJP | Auto Fix | remaining-auto |
| #87 FFT under CUDA-backed execution | Verify First | verify-first; current behavior rejects GPU input |
| #106 integer/bool/lossy `Convert` AD policy | Design Gate | design-gated |
| #109 lazy value/view API rustdoc examples | Auto Fix | fixed |
| #110 `LuSolvePrepared` mixed complex adjoint flags | Auto Fix | remaining-auto |
| #111 `TracedTensorAdExt` rustdoc examples | Auto Fix | fixed |
| #112 CUDA zero-sized-output residency validation | Auto Fix | remaining-auto |
| #113 gather/scatter boundary VJP | Verify First | verify-first; narrow to current boundary cases |
| #114 `Pow` zero-base singularity | Auto Fix | remaining-auto |
| #115 terminal lazy view base clone | Auto Fix | remaining-auto |
| #116 scalar helper public contract | Design Gate | design-gated |
| #116 mixed real/complex binary VJPs | Auto Fix | remaining-auto |
| #117 einsum fallback greedy `HashSet` rebuild | Auto Fix | fixed |
| #117 eager helper host tensors in CUDA contexts | Auto Fix | remaining-auto |
| #118 active crate/backend ownership docs | Auto Fix | fixed |
| #119 primitive catalog `Constant` docs | Auto Fix | fixed |
| #120 active in-place indexing ChainRules framing | Auto Fix | fixed |
| #121 einsum retained-byte cache accounting | Design Gate | design-gated |
| #122 CPU provider-injection rustdoc placeholders | Auto Fix | fixed |
| #123 eager extension materializes lazy/view tensors | Design Gate | design-gated |
| #124 oracle docs active CI/replay claims | Auto Fix | fixed |
| #125 max/min/clamp boundary AD convention | Design Gate | design-gated |
| #126 device AD seeds and missing tangents | Auto Fix | remaining-auto |

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

`cargo doc --workspace --no-deps` emitted an existing private intra-doc link
warning in `crates/tenferro-runtime/src/shape_infer.rs`.

## Residual Risks

- The remaining auto-fix items touch AD, runtime/performance, and GPU placement
  behavior and should be handled in later coherent commits.
- Design-gated items should become focused design or child issues rather than
  being implemented silently in this remediation branch.
