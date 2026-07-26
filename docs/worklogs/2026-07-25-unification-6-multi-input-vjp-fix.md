# Worklog: Unification 6 — Multi-Input Semantic Eager VJP Fix

Date: 2026-07-25

## Summary

Fixed the gated semantic eager VJP path to handle multi-input core graphs (e.g., `x * y`).
The RED regression test `eager_runtime_vjp_uses_semantic_trace_for_multi_input_graph_when_gate_enabled`
now passes.

## S0/S1 Evidence

S0 (bound-program transform cache) and S1 (symbolic-shape eager recording) were
verified on this checkpoint branch:

```bash
# S0: bound-program transform cache
cargo test -p tenferro-ad semantic_program_transform_cache_reuses_bound_programs_without_stale_bindings --test integration -- --nocapture
# result: pass

# S1: symbolic-shape eager recording
cargo test -p tenferro-ad eager_recording_retains_symbolic_semantic_trace_for_shape_churn --lib -- --nocapture
# result: pass
```

## Multi-Input VJP Fix

Three root causes were identified and fixed:

### 1. Unresolved InputDim in imported metadata (tenferro-runtime)

`ImportTransaction::prepare` copied source-program metadata verbatim, preserving
symbolic `DimExpr::InputDim { input_idx, axis }` expressions. When the VJP
transform's `normalize_ad_value` used this metadata to create `Reshape` ops,
`InputDim` references beyond index 0 required additional tensor inputs that
weren't provided → `ProgramBuildError::Arity { expected: 2, actual: 1 }`.

Fix: Added `resolve_dim_expr_from_input_shapes` that resolves `InputDim` to
`Const` using concrete tensor binding shapes. Applied during import for both
input values and operation output values.

### 2. Cotangent seed used unresolved source metadata (tenferro-ad)

`semantic_vjp` created cotangent seed inputs using `input.program.value_metadata(source)`
(the original source program's unresolved metadata). After fix #1, the imported
values had resolved metadata, but the seed input still used the unresolved source.

Fix: Changed to `builder.value_metadata(imported_source)` to use the imported
value's resolved metadata.

### 3. Wrong index semantics for derivative maps (tenferro-ad)

`semantic_eager_vjp_optional` confused `derivative_input_indices` (indexed by
source OUTPUT) with `derivative_output_indices` (indexed by source INPUT):
- `derivative_input_indices().get(wrt_input_index)` — used input index where output index was needed
- `derivative_output_indices().first()` — always looked at first input

Fix: `.first()` for derivative_input_indices (always 1 output), `.get(wrt_input_index)` for derivative_output_indices.

## Verified Commands

```bash
# Multi-input VJP (was RED, now green)
cargo test -p tenferro-ad eager_runtime_vjp_uses_semantic_trace_for_multi_input_graph_when_gate_enabled --lib -- --nocapture
# result: pass

# Single-input VJP (no regression)
cargo test -p tenferro-ad eager_runtime_vjp_can_use_semantic_trace_when_gate_enabled --lib -- --nocapture
# result: pass

# S0 cache test (no regression)
cargo test -p tenferro-ad semantic_program_transform_cache_reuses_bound_programs_without_stale_bindings --test integration -- --nocapture
# result: pass

# S1 shape churn test (no regression)
cargo test -p tenferro-ad eager_recording_retains_symbolic_semantic_trace_for_shape_churn --lib -- --nocapture
# result: pass

# Full VJP suite
cargo test -p tenferro-ad eager_runtime_vjp --lib -- --nocapture
# result: 3 passed

# Full semantic transform integration suite
cargo test -p tenferro-ad semantic_transform --test integration -- --nocapture
# result: 30 passed

# Full tenferro-runtime lib tests
cargo test -p tenferro-runtime --lib -- --nocapture
# result: 322 passed
```

## Residual Risks

- The `resolve_dim_expr_from_input_shapes` function resolves InputDim to concrete
  values during import. This preserves correctness but may reduce shape-polymorphic
  cache reuse if the same program structure is reused with different concrete shapes.
  The S1 test still passes because it tests the recording layer (not import).
- The `derivative_input_indices` / `derivative_output_indices` swap assumes
  single-output programs (guarded by `source.output_count() != 1` check above).

## Unification 7 Review (2026-07-25)

Code review of AD/eager/runtime critical paths:

- **Old execution path references**: Clean. `GraphExecutor`, `ExtensionRuntime`/`ExtensionExecutor`,
  `HostReference`/`host_reference()` all removed. Only `define_extension_runtime!` macro remains
  (current infrastructure, not legacy).
- **Extension-family registration**: Well-structured through `define_extension_runtime!` macro
  in extension crates (einsum, linalg, fft). `ExtensionModule` trait and `ExtensionModuleRegistrar`
  provide clean registration.
- **Cache ownership**: `AdTransformCache` shared between `EagerRuntime` and `AdContext` via
  `Arc<AdTransformCache>`. Extension caches owned by `EagerRuntime.extension_caches: Mutex<ExtensionCacheStore>`.
- **Lock/mutex pattern**: `EagerRuntime` uses separate `Mutex` for `backend`, `extension_caches`,
  `grad_slots`, `value_records`, `value_ptr_records`. No lock ordering issues identified.
- **Hidden materialization**: `materialized_arc()` is the established eager tensor materialization
  pattern. Pre-existing concern, not a regression.

## S2 Decision (2026-07-25)

S2 (incremental structure digest at eager record time) is **deferred**.

Rationale: The `SemanticAdTransformCacheKey` uses `semantic_fingerprint()` which matches
programs with identical symbolic structure across different concrete shapes (S1). The VJP
transform is already cached per structure. The `compile(output_trace)` step runs per
invocation but is bounded. Full struct digest + prepared derivative plan caching can be
added later when profile data justifies it.

## Unification 8 State (2026-07-25)

Performance gate harness: `scripts/run-unification-performance-gate.sh` exists.
Baseline pinned at `c6418eecfe2d38ca09d6e6386760fcb23982691e` (origin/main).
Gate is intentionally terminal — intermediate benchmarks are diagnostic only.

Fallback policy for #1460:
- Primary path: semantic eager VJP (gated, becoming default)
- Fallback is NOT tidu (tidu is oracle/comparison only during migration)
- Accepted fallback: semantic-rule-fragment direct executor if exact-dim specialization
  churn cannot be made non-regressive

Benchmarks not run at this stage per policy: "Do not spend time repeating benchmarks
while the code is still structurally dirty."
