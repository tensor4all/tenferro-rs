# Unified Binary Einsum Dispatcher — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Merge the two separate dispatch paths (Gemm vs Contract) into a single unified contraction handler that always tries Contract extension first, falling back to permute+BatchedGemm.

**Architecture:** Replace `StepStrategy::Gemm` + `StepStrategy::Contract` with `StepStrategy::Contraction(Option<GemmPlan>)`. Create `execute_contraction_unified` that implements: pre-reduce → try Contract ext → fallback GEMM. Remove the duplicate non-tree dispatcher `execute_pairwise_contraction`.

**Tech Stack:** Rust, tenferro-einsum crate only (no prims changes needed)

**File:** `tenferro-einsum/src/lib.rs` (all changes in this single file)

---

### Task 1: Merge StepStrategy variants

**Files:**
- Modify: `tenferro-einsum/src/lib.rs:1670-1679` (StepStrategy enum)
- Modify: `tenferro-einsum/src/lib.rs:1854` (compile_step_plans Gemm arm)
- Modify: `tenferro-einsum/src/lib.rs:1878` (compile_step_plans Contract arm)

**Step 1: Change the enum**

Replace:
```rust
enum StepStrategy {
    ElementwiseMul,
    OuterProduct(OuterProductPlan),
    Gemm(GemmPlan),
    Contract,
}
```

With:
```rust
enum StepStrategy {
    ElementwiseMul,
    OuterProduct(OuterProductPlan),
    /// Contraction: try Contract extension first, fall back to permute+GEMM.
    /// Some(plan) = GEMM-compatible (has pre-computed GemmPlan for fallback).
    /// None = not GEMM-compatible (trace-like, Contract extension only).
    Contraction(Option<GemmPlan>),
}
```

**Step 2: Update compile_step_plans**

Change `StepStrategy::Gemm(GemmPlan { ... })` → `StepStrategy::Contraction(Some(GemmPlan { ... }))`
Change `StepStrategy::Contract` → `StepStrategy::Contraction(None)`

**Step 3: Run tests**

```bash
cargo test -p tenferro-einsum
```

Expected: compile error (match arms in execute_pairwise_with_plan and execute_tree reference old variants). This is OK — we'll fix them in the next tasks.

---

### Task 2: Create execute_contraction_unified

**Files:**
- Modify: `tenferro-einsum/src/lib.rs` (add new function near execute_gemm_with_plan)

**Step 1: Write the unified function**

```rust
/// Unified contraction handler: pre-reduce → Contract ext → fallback GEMM.
fn execute_contraction_unified<Alg, Backend>(
    ctx: &mut Backend::Context,
    gemm_plan: Option<&GemmPlan>,
    prim_plan: Option<&Backend::Plan>,  // pre-computed Contract plan (for non-GEMM patterns)
    subs_a: &[u32],
    subs_b: &[u32],
    subs_c: &[u32],
    a: &Tensor<Alg::Scalar>,
    b: &Tensor<Alg::Scalar>,
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
    pool: &mut BufferPool<Alg::Scalar>,
) -> Result<()>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    match gemm_plan {
        Some(plan) => {
            // GEMM-compatible pattern: pre-reduce, then try Contract, then fallback GEMM

            // 1. Pre-reduce unique-only axes
            let (a_reduced, b_reduced) = /* pre-reduce logic from execute_gemm_with_plan */;

            let a_ref = a_reduced.as_ref().unwrap_or(a);
            let b_ref = b_reduced.as_ref().unwrap_or(b);

            // 2. Try Contract extension (strided GEMM, zero-copy)
            if Backend::has_extension_for(Extension::Contract) {
                let desc = PrimDescriptor::Contract {
                    modes_a: plan.subs_a.clone(),
                    modes_b: plan.subs_b.clone(),
                    modes_c: subs_c.to_vec(),
                };
                let shapes = [a_ref.dims(), b_ref.dims(), output.dims()];
                let pp = Backend::plan(ctx, &desc, &shapes)?;
                return Backend::execute(ctx, &pp, alpha, &[a_ref, b_ref], beta, output);
            }

            // 3. Fallback: prepare + BatchedGemm + permute
            execute_gemm_after_reduce::<Alg, Backend>(
                ctx, plan, subs_c, a_ref, b_ref, alpha, beta, output, pool,
            )
        }
        None => {
            // Non-GEMM pattern (trace-like): Contract extension only
            if let Some(pp) = prim_plan {
                Backend::execute(ctx, pp, alpha, &[a, b], beta, output)
            } else {
                let desc = PrimDescriptor::Contract {
                    modes_a: subs_a.to_vec(),
                    modes_b: subs_b.to_vec(),
                    modes_c: subs_c.to_vec(),
                };
                let shapes = [a.dims(), b.dims(), output.dims()];
                let pp = Backend::plan(ctx, &desc, &shapes)?;
                Backend::execute(ctx, &pp, alpha, &[a, b], beta, output)
            }
        }
    }
}
```

**Step 2: Extract GEMM-after-reduce logic from execute_gemm_with_plan**

Split `execute_gemm_with_plan` into:
- Pre-reduce part (moved into `execute_contraction_unified`)
- Post-reduce GEMM part (renamed to `execute_gemm_after_reduce` — keeps prepare_one_operand + BatchedGemm + permute logic)

---

### Task 3: Wire up execute_pairwise_with_plan

**Files:**
- Modify: `tenferro-einsum/src/lib.rs:1908-1957` (execute_pairwise_with_plan match)

**Step 1: Update match arms**

```rust
match &plan.strategy {
    StepStrategy::ElementwiseMul => {
        if let Some(pp) = prim_plan {
            Backend::execute(ctx, pp, alpha, &[a, b], beta, output)
        } else if Backend::has_extension_for(Extension::ElementwiseMul) {
            let desc = PrimDescriptor::ElementwiseMul;
            let shapes = [a.dims(), b.dims(), output.dims()];
            let pp = Backend::plan(ctx, &desc, &shapes)?;
            Backend::execute(ctx, &pp, alpha, &[a, b], beta, output)
        } else {
            // Fall back to unified contraction (Contract handles element-wise fine)
            execute_contraction_unified::<Alg, Backend>(
                ctx, None, None, subs_a, subs_b, subs_c, a, b, alpha, beta, output, pool,
            )
        }
    }
    StepStrategy::OuterProduct(op_plan) => {
        if Backend::has_extension_for(Extension::ElementwiseMul) {
            execute_outer_with_plan::<Alg, Backend>(ctx, op_plan, subs_c, a, b, alpha, beta, output)
        } else {
            // Fall back to unified contraction
            execute_contraction_unified::<Alg, Backend>(
                ctx, None, None, subs_a, subs_b, subs_c, a, b, alpha, beta, output, pool,
            )
        }
    }
    StepStrategy::Contraction(gemm_plan_opt) => {
        execute_contraction_unified::<Alg, Backend>(
            ctx, gemm_plan_opt.as_ref(), prim_plan, subs_a, subs_b, subs_c,
            a, b, alpha, beta, output, pool,
        )
    }
}
```

**Step 2: Run tests**

```bash
cargo test -p tenferro-einsum
```

Expected: PASS (all existing tests should pass — behavior unchanged)

---

### Task 4: Update execute_tree plan pre-computation

**Files:**
- Modify: `tenferro-einsum/src/lib.rs:2213-2264` (prim_plans pre-computation in execute_tree)

**Step 1: Update the pre-computation loop**

Change:
```rust
let needs_contract = use_contract && matches!(sp.strategy, StepStrategy::Contract);
```

To:
```rust
// Pre-compute Contract plans for Contraction(None) steps only.
// Contraction(Some(_)) steps pre-reduce at runtime, so Contract plan
// must be computed after reduction with the reduced subscripts/shapes.
let needs_contract = use_contract
    && matches!(sp.strategy, StepStrategy::Contraction(None));
```

**Step 2: Run tests**

```bash
cargo test -p tenferro-einsum
```

Expected: PASS

---

### Task 5: Remove dead code

**Files:**
- Modify: `tenferro-einsum/src/lib.rs`

**Step 1: Remove `execute_pairwise_contraction` (L1537-1601)**

No longer called from anywhere.

**Step 2: Remove `fallback_pairwise_contraction` (L1100-1249)**

The GEMM fallback logic is now in `execute_gemm_after_reduce`.

**Step 3: Remove `try_outer_elementwise_contraction` (L995-1089)**

Only called by `execute_pairwise_contraction` which is removed.

**Step 4: Remove `is_gemm_fallback_compatible` (L969-989)**

Only called by `execute_pairwise_contraction` which is removed.

**Step 5: Remove old `execute_gemm_with_plan` if fully replaced**

The pre-reduce + post-reduce logic is now split between `execute_contraction_unified` and `execute_gemm_after_reduce`.

**Step 6: Run full test suite**

```bash
cargo test -p tenferro-einsum
cargo fmt --all --check
```

Expected: PASS, clean formatting

**Step 7: Commit**

```bash
git add tenferro-einsum/src/lib.rs
git commit -m "refactor: unify einsum dispatcher — Contract-first, GEMM fallback"
```

---

### Task 6: Benchmark validation

**Step 1: Run gm_queen benchmark**

```bash
cd /path/to/tenferro-einsum-benchmark
cargo bench --bench gm_queen -- --sample-size 10
```

Expected: opt_flops should be significantly faster than previous 6.5s (closer to strided-rs 2.2s), because Contract now uses strided GEMM (FaerGemm) instead of prepare+copy+BatchedGemm.

**Step 2: Verify no regression on other benchmarks**

```bash
cargo bench -- --sample-size 10
```
