# Binary Strict Lowering Design

## Status

Approved on 2026-03-26.

## Context

`tensor4all-tensorbackend` still carries a bridge-local binary fast path in
[`binary_dense_primal_gemm_with_ids`](../../../../tensor4all-rs/crates/tensor4all-tensorbackend/src/tenferro_bridge.rs).
That helper is fast because it bypasses `tenferro-einsum`'s generic planning
and execution setup, but it leaves the system in an inconsistent state:

- binary dense primal einsum goes through a bridge-only lowering
- generic binary einsum goes through `tenferro-einsum`
- n-ary einsum goes through `tenferro-einsum`

This split makes performance comparisons hard to interpret, duplicates einsum
semantics across repositories, and prevents the backend stack from converging on
one execution path.

Recent measurements show:

- binary no-plan generic overhead has already been reduced substantially inside
  `tenferro-einsum`
- remaining n-ary generic overhead is no longer dominated by `optimize`
- downstream pairwise benchmarks are still faster mainly because the bridge-only
  binary lowering exists

## Goal

Move the bridge-local binary dense primal GEMM lowering into `tenferro-einsum`,
verify the CPU performance, and delete the bridge-local fast path so binary
einsum flows through one tenferro-owned implementation.

## Non-Goals

- This design does not require a GPU benchmark pass.
- This design does not introduce a new public primitive family yet.
- This design does not immediately optimize non-dense, conjugated, repeated-label,
  or generic semiring cases.
- This design does not yet perform the repository-wide `ctx` plumbing audit for
  all prims; that remains a separate follow-up.

## Chosen Direction

Implement a backend-generic internal fast lowering inside `tenferro-einsum`,
but optimize and validate only the CPU path first.

This means:

- the *surface* lives in `tenferro-einsum`
- the *implementation substrate* uses existing backend-generic tensor ops and
  `TensorSemiringCore::BatchedGemm`
- the *performance target* for this pass is CPU

Later, after CPU results are acceptable, the internal API can be cleaned up and
lifted into a more explicit CPU/GPU-generic arrangement without keeping the
bridge helper alive.

## Why Not CPU-Only in `tenferro-prims`

A CPU-only lowering inside `tenferro-prims` would be fast to write, but it would
push einsum label analysis into the primitive layer. That is the wrong boundary:

- `tenferro-prims` should execute tensor primitives, not parse einsum semantics
- the bridge helper currently reasons about labels, free/contracted partitions,
  and output permutations; that logic belongs with einsum planning
- if we later need the same path on CUDA or ROCm, a CPU-local implementation
  would have to be redesigned anyway

So the lowering belongs above the primitive layer, even if the first optimized
backend is CPU.

## Internal API Shape

Add an internal `StrictBinaryLoweringPlan` in `tenferro-einsum`.

It should contain:

- operand/output label partitions
- axis permutations for lhs and rhs
- fused `m`, `k`, `n`
- reshaped matrix dims
- canonical output dims
- final output permutation

The plan is intentionally narrower than the generic `StepPlan`: it targets the
bridge helper’s exact “GEMM without generic machinery” semantics.

## Eligibility Rules

The strict lowering applies only when all of the following hold:

- exactly two operands
- dense primal tensors
- unconjugated inputs
- no repeated labels inside either operand
- contraction labels do not appear in the output
- output labels are exactly a permutation of `lhs_free + rhs_free`
- backend can execute `TensorSemiringCore::BatchedGemm`

If any check fails, execution falls back to the existing generic binary einsum
path with no semantic change.

## Execution Model

The strict executor performs:

1. compute lhs/rhs free and contracted labels
2. permute lhs to `[lhs_free..., contract...]`
3. permute rhs to `[contract..., rhs_free...]`
4. make operands contiguous if needed
5. reshape to 2D matrices `[m, k]` and `[k, n]`
6. call `BatchedGemm`
7. reshape to canonical output `[lhs_free..., rhs_free...]`
8. apply final output permute if needed

This is intentionally the same lowering that made the bridge helper fast.

## Ownership and Layering

`tensor4all-tensorbackend` should stop owning binary einsum lowering logic.

After migration:

- bridge builds `Subscripts`
- bridge forwards to `tenferro-einsum`
- `tenferro-einsum` decides whether strict lowering applies
- if not, `tenferro-einsum` falls back internally

This makes tenferro the single owner of einsum lowering semantics.

## Migration Strategy

### Phase 1: Tenferro internalization

- add `StrictBinaryLoweringPlan`
- add strict executor
- route binary APIs through “strict first, generic fallback”
- verify CPU micro-benchmarks

### Phase 2: Bridge cleanup

- remove `binary_dense_primal_gemm_with_ids`
- remove the bridge-only binary branch
- always call tenferro binary APIs from `einsum_native_tensors`
- verify downstream CPU benchmarks

## Testing Strategy

### Unit tests

- eligibility succeeds on dense primal matmul-like cases
- repeated-label and conjugated cases reject strict lowering
- final output permutation matches generic einsum
- strict path and generic path produce equal tensors

### Micro-benchmarks

- existing binary path breakdown benchmark
- compare strict lowering against `einsum_with_plan` and generic binary API

### Downstream benchmarks

- `bench_native_einsum_fit_patterns`
- compare pairwise and generic ratios before/after bridge removal

No GPU benchmark is required in this phase.

## Success Criteria

This design is successful if:

- bridge-local binary lowering is deleted
- binary dense primal CPU performance does not regress versus the bridge helper
- downstream pairwise/generic comparisons become comparisons between tenferro
  paths, not bridge vs tenferro paths
- the resulting design is a clean stepping stone to a later CPU/GPU-generic
  internal fast-lowering surface
