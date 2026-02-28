# Unified Binary Einsum Dispatcher Design

## Problem

The einsum dispatch layer has two separate dispatchers with duplicated priority logic:

1. **Tree path** (`execute_pairwise_with_plan`): dispatches on `StepStrategy` enum (ElementwiseMul / OuterProduct / Gemm / Contract)
2. **Non-tree fallback** (`execute_pairwise_contraction`): its own 4-way priority cascade

The `StepStrategy::Gemm` path never tries the Contract extension, even though Contract now handles strided GEMM with zero copy (FaerGemm). This causes 34% of gm_queen time to be wasted on unnecessary contiguous copies in `prepare_one_operand`.

## Design

**Key insight**: Contract and permute+GEMM are both implementations of the same operation — binary einsum: `(subs_a, subs_b, subs_c, A, B) → C`. They should share a common interface, with Contract as the priority path and permute+GEMM as the fallback.

### Unified Priority Chain

```
binary_einsum(subs_a, subs_b, subs_c, A, B, alpha, beta, C):
  1. ElementwiseMul extension  (subs_a == subs_b == subs_c)
  2. OuterProduct via ElementwiseMul  (disjoint labels)
  3. Contract extension  (pre-reduce if needed, then Contract prim)
  4. Fallback: pre-reduce + permute + BatchedGemm + permute
```

Steps 3 and 4 share the same pre-reduce logic. The only difference is whether the core contraction uses Contract prim (strided, zero-copy) or the permute+BatchedGemm decomposition.

### Changes

- **Merge** `StepStrategy::Gemm` and `StepStrategy::Contract` into a single `StepStrategy::Contraction` that holds the pre-computed `GemmPlan` (used by the fallback path)
- **Remove** `execute_pairwise_contraction` — fold its logic into `execute_pairwise_with_plan`
- **Remove** `execute_gemm_with_plan` as a separate entry point — it becomes the fallback inside the unified contraction path
- **Keep** `GemmPlan` pre-computation for the fallback path
- **Add** Contract-first attempt inside the unified contraction handler

### Dispatch Flow

```
execute_pairwise_with_plan:
  match strategy:
    ElementwiseMul →
      1. pre-computed ElementwiseMul prim plan
      2. on-the-fly ElementwiseMul plan
      3. naive fallback (if no extension)

    OuterProduct →
      1. broadcast + ElementwiseMul (requires extension)
      2. falls through to Contraction if no extension

    Contraction →
      1. pre-reduce unique-only axes (shared logic, if needed)
      2. if Contract ext available → Contract prim (strided, zero-copy)
      3. else → prepare_one_operand + BatchedGemm + permute (fallback)
```

### What Stays the Same

- `compile_step_plans` classification logic (ElementwiseMul / OuterProduct detection)
- `GemmPlan` pre-computation (batch/lo/ro/sum modes, m/n/k, target orders)
- `OuterProductPlan` pre-computation
- `prepare_one_operand` (still needed for fallback path)
- `execute_outer_with_plan` (outer product execution)
- All prim-level code (`execute_contract`, `try_execute_contract_gemm`, `execute_batched_gemm_strided`)

## Post-Implementation Finding: Remove Contract Extension from Einsum

After implementing the unified dispatcher, benchmarks revealed that Contract-first routing causes hangs on large instances (gm_queen5_5_3.wcsp). Root cause analysis:

**Contract prim vs Fallback GEMM — algorithm comparison:**

| Aspect | Contract prim | Fallback GEMM (prepare_one_operand + BatchedGemm) |
|--------|---------------|---------------------------------------------------|
| try_fuse_group succeeds | faer strided GEMM (zero-copy) | faer strided GEMM (zero-copy) |
| try_fuse_group fails | **O(n^rank) naive loop** | copy to contiguous → GEMM |

Both paths use the same `faer::strided_gemm` when dimension groups are fusable. The only difference is the failure case: Contract falls to a naive element-by-element loop, while Fallback GEMM copies to contiguous and runs GEMM.

einsum2 (strided-rs) uses the same algorithm as Fallback GEMM — copy when needed, always GEMM.

**Decision:** Remove Contract extension from `execute_contraction_unified` (delete L123-131 in dispatch.rs). Always use the Fallback GEMM path, which is a strict superset of Contract's capabilities. Contract prim itself remains available for other use cases but einsum should not route through it.
