use std::collections::{HashMap, HashSet};

use num_traits::{One, Zero};
use tenferro_algebra::{Algebra, HasAlgebra, Scalar};
use tenferro_device::Result;
use tenferro_prims::{Extension, PrimDescriptor, ReduceOp, TensorPrims};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::classify::is_gemm_fallback_compatible;
use crate::plan::{GemmPlan, OuterProductPlan, ReducePlan, StepPlan, StepStrategy};
use crate::pool::BufferPool;
use crate::prepare::{prepare_gemm_operands, prepare_one_operand, reduce_unique_only_axes};
use crate::classify::classify_modes;
use crate::util::alloc_tensor_from_pool;

/// Execute a pairwise contraction using a pre-computed step plan.
///
/// This avoids per-step HashMap/HashSet allocations by using the pre-computed
/// strategy, mode classification, sizes, and target subscripts.
pub(crate) fn execute_pairwise_with_plan<Alg, Backend>(
    ctx: &mut Backend::Context,
    plan: &StepPlan,
    prim_plan: Option<&Backend::Plan>,
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
                // Fall back to non-plan path
                execute_pairwise_contraction::<Alg, Backend>(
                    ctx, subs_a, subs_b, subs_c, a, b, alpha, beta, output, pool,
                )
            }
        }
        StepStrategy::OuterProduct(op_plan) => {
            if !Backend::has_extension_for(Extension::ElementwiseMul) {
                // Fall back to non-plan path
                return execute_pairwise_contraction::<Alg, Backend>(
                    ctx, subs_a, subs_b, subs_c, a, b, alpha, beta, output, pool,
                );
            }
            execute_outer_with_plan::<Alg, Backend>(ctx, op_plan, subs_c, a, b, alpha, beta, output)
        }
        StepStrategy::Gemm(gemm_plan) => {
            // Gemm decomposition path: pre-computed plan with strided faer access.
            // This is faster than Contract for all sizes because it avoids
            // dense-matrix copy overhead in execute_contract.
            execute_gemm_with_plan::<Alg, Backend>(
                ctx, gemm_plan, subs_c, a, b, alpha, beta, output, pool,
            )
        }
        StepStrategy::Contract => {
            if let Some(pp) = prim_plan {
                Backend::execute(ctx, pp, alpha, &[a, b], beta, output)
            } else {
                // Fallback: compute plan at runtime
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

/// Execute a pairwise contraction of two tensors via TensorPrims.
fn execute_pairwise_contraction<Alg, Backend>(
    ctx: &mut Backend::Context,
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
    // Check for element-wise multiplication (same labels, same order)
    if subs_a == subs_b && subs_a == subs_c && Backend::has_extension_for(Extension::ElementwiseMul)
    {
        let desc = PrimDescriptor::ElementwiseMul;
        let shapes = [a.dims(), b.dims(), output.dims()];
        let plan = Backend::plan(ctx, &desc, &shapes)?;
        return Backend::execute(ctx, &plan, alpha, &[a, b], beta, output);
    }

    // Specialized outer-product path (disjoint binary einsum).
    if try_outer_elementwise_contraction::<Alg, Backend>(
        ctx, subs_a, subs_b, subs_c, a, b, alpha, beta, output,
    )? {
        return Ok(());
    }

    // Prefer GEMM decomposition when labels fit the matrix contraction model.
    // This avoids the generic Contract kernel for common high-throughput cases.
    if is_gemm_fallback_compatible(subs_a, subs_b, subs_c) {
        return fallback_pairwise_contraction::<Alg, Backend>(
            ctx, subs_a, subs_b, subs_c, a, b, alpha, beta, output, pool,
        );
    }

    // General contraction via Contract extension
    if Backend::has_extension_for(Extension::Contract) {
        let desc = PrimDescriptor::Contract {
            modes_a: subs_a.to_vec(),
            modes_b: subs_b.to_vec(),
            modes_c: subs_c.to_vec(),
        };
        let shapes = [a.dims(), b.dims(), output.dims()];
        let plan = Backend::plan(ctx, &desc, &shapes)?;
        Backend::execute(ctx, &plan, alpha, &[a, b], beta, output)
    } else {
        // Contract is disabled in extension dispatch. For patterns that cannot
        // be lowered by fallback_pairwise_contraction, directly try Contract.
        // This keeps compatibility while making GEMM decomposition the default.
        let desc = PrimDescriptor::Contract {
            modes_a: subs_a.to_vec(),
            modes_b: subs_b.to_vec(),
            modes_c: subs_c.to_vec(),
        };
        let shapes = [a.dims(), b.dims(), output.dims()];
        let plan = Backend::plan(ctx, &desc, &shapes)?;
        Backend::execute(ctx, &plan, alpha, &[a, b], beta, output)
    }
}

/// Try a specialized outer-product lowering via broadcast + ElementwiseMul.
///
/// This path targets disjoint binary einsum (no summed/shared modes), e.g.:
/// `i,j->ij`. It avoids GEMM(k=1) overhead and can be substantially faster.
fn try_outer_elementwise_contraction<A, B>(
    ctx: &mut B::Context,
    subs_a: &[u32],
    subs_b: &[u32],
    subs_c: &[u32],
    a: &Tensor<A::Scalar>,
    b: &Tensor<A::Scalar>,
    alpha: A::Scalar,
    beta: A::Scalar,
    output: &mut Tensor<A::Scalar>,
) -> Result<bool>
where
    A: Algebra,
    A::Scalar: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    if !B::has_extension_for(Extension::ElementwiseMul) {
        return Ok(false);
    }

    let unique = |subs: &[u32]| -> bool {
        let mut seen = HashSet::with_capacity(subs.len());
        subs.iter().all(|&m| seen.insert(m))
    };
    if !unique(subs_a) || !unique(subs_b) || !unique(subs_c) {
        return Ok(false);
    }

    let set_a: HashSet<u32> = subs_a.iter().copied().collect();
    let set_b: HashSet<u32> = subs_b.iter().copied().collect();
    // Must be pure outer: disjoint labels between inputs.
    if set_a.iter().any(|m| set_b.contains(m)) {
        return Ok(false);
    }

    let canonical_modes: Vec<u32> = subs_a.iter().chain(subs_b.iter()).copied().collect();
    let set_c: HashSet<u32> = subs_c.iter().copied().collect();
    if set_c != canonical_modes.iter().copied().collect::<HashSet<_>>() {
        return Ok(false);
    }

    let mut size_dict = HashMap::new();
    for (&m, &d) in subs_a.iter().zip(a.dims()) {
        size_dict.insert(m, d);
    }
    for (&m, &d) in subs_b.iter().zip(b.dims()) {
        size_dict.insert(m, d);
    }

    let canonical_shape: Vec<usize> = canonical_modes.iter().map(|m| size_dict[m]).collect();
    let a_ext_shape: Vec<usize> = canonical_modes
        .iter()
        .map(|m| if set_a.contains(m) { size_dict[m] } else { 1 })
        .collect();
    let b_ext_shape: Vec<usize> = canonical_modes
        .iter()
        .map(|m| if set_b.contains(m) { size_dict[m] } else { 1 })
        .collect();

    let a_reshaped = a.reshape(&a_ext_shape)?;
    let b_reshaped = b.reshape(&b_ext_shape)?;
    let a_bcast = a_reshaped.broadcast(&canonical_shape)?;
    let b_bcast = b_reshaped.broadcast(&canonical_shape)?;

    if canonical_modes == subs_c {
        let desc = PrimDescriptor::ElementwiseMul;
        let shapes = [a_bcast.dims(), b_bcast.dims(), output.dims()];
        let plan = B::plan(ctx, &desc, &shapes)?;
        B::execute(ctx, &plan, alpha, &[&a_bcast, &b_bcast], beta, output)?;
        return Ok(true);
    }

    let memory_space = output.logical_memory_space();
    let mut temp =
        Tensor::<A::Scalar>::zeros(&canonical_shape, memory_space, MemoryOrder::ColumnMajor);
    let desc = PrimDescriptor::ElementwiseMul;
    let shapes = [a_bcast.dims(), b_bcast.dims(), temp.dims()];
    let plan = B::plan(ctx, &desc, &shapes)?;
    B::execute(
        ctx,
        &plan,
        A::Scalar::one(),
        &[&a_bcast, &b_bcast],
        A::Scalar::zero(),
        &mut temp,
    )?;
    let desc = PrimDescriptor::Permute {
        modes_a: canonical_modes,
        modes_b: subs_c.to_vec(),
    };
    let shapes = [temp.dims(), output.dims()];
    let plan = B::plan(ctx, &desc, &shapes)?;
    B::execute(ctx, &plan, alpha, &[&temp], beta, output)?;
    Ok(true)
}

/// Fallback pairwise contraction using core primitives only.
///
/// Decomposes a binary contraction into:
/// 1. Permute A → [batch, lo, sum], B → [batch, sum, ro]
/// 2. MakeContiguous (conditional copy)
/// 3. Reshape to [batch..., m, k] and [batch..., k, n]
/// 4. BatchedGemm → temp [batch..., m, n]
/// 5. Reshape temp → [batch..., lo..., ro...]
/// 6. Permute to output with alpha/beta
fn fallback_pairwise_contraction<A, B>(
    ctx: &mut B::Context,
    subs_a: &[u32],
    subs_b: &[u32],
    subs_c: &[u32],
    a: &Tensor<A::Scalar>,
    b: &Tensor<A::Scalar>,
    alpha: A::Scalar,
    beta: A::Scalar,
    output: &mut Tensor<A::Scalar>,
    pool: &mut BufferPool<A::Scalar>,
) -> Result<()>
where
    A: Algebra,
    A::Scalar: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    // Pre-reduce axes that are present only in one input and not in output.
    // This expands GEMM decomposition coverage to general binary contractions.
    let (a_reduced, subs_a_reduced) =
        reduce_unique_only_axes::<A, B>(ctx, a, subs_a, subs_b, subs_c, pool)?;
    let (b_reduced, subs_b_reduced) =
        reduce_unique_only_axes::<A, B>(ctx, b, subs_b, subs_a, subs_c, pool)?;

    let (batch_modes, lo_modes, ro_modes, sum_modes) =
        classify_modes(&subs_a_reduced, &subs_b_reduced, subs_c);

    // Build size_dict from input subscripts and shapes
    let mut size_dict: HashMap<u32, usize> = HashMap::new();
    for (&label, &dim) in subs_a_reduced.iter().zip(a_reduced.dims()) {
        size_dict.insert(label, dim);
    }
    for (&label, &dim) in subs_b_reduced.iter().zip(b_reduced.dims()) {
        size_dict.insert(label, dim);
    }

    let batch_sizes: Vec<usize> = batch_modes.iter().map(|m| size_dict[m]).collect();
    let lo_sizes: Vec<usize> = lo_modes.iter().map(|m| size_dict[m]).collect();
    let ro_sizes: Vec<usize> = ro_modes.iter().map(|m| size_dict[m]).collect();
    let sum_sizes: Vec<usize> = sum_modes.iter().map(|m| size_dict[m]).collect();

    let m: usize = lo_sizes.iter().product::<usize>().max(1);
    let n: usize = ro_sizes.iter().product::<usize>().max(1);
    let k: usize = sum_sizes.iter().product::<usize>().max(1);

    // --- Steps 1-4: Permute, fuse, GEMM ---
    let target_a: Vec<u32> = batch_modes
        .iter()
        .chain(lo_modes.iter())
        .chain(sum_modes.iter())
        .copied()
        .collect();
    let target_b: Vec<u32> = batch_modes
        .iter()
        .chain(sum_modes.iter())
        .chain(ro_modes.iter())
        .copied()
        .collect();
    let c_gemm_shape: Vec<usize> = batch_sizes
        .iter()
        .copied()
        .chain(std::iter::once(m))
        .chain(std::iter::once(n))
        .collect();

    let memory_space = a.logical_memory_space();

    // Try fusability-aware path: permute (metadata-only) + fuse groups → zero-copy GEMM
    let (a_reshaped, b_reshaped) = prepare_gemm_operands::<A, B>(
        ctx,
        &a_reduced,
        &subs_a_reduced,
        &target_a,
        &batch_sizes,
        lo_modes.len(),
        sum_modes.len(),
        &b_reduced,
        &subs_b_reduced,
        &target_b,
        sum_modes.len(),
        ro_modes.len(),
        m,
        n,
        k,
        pool,
    )?;

    let mut temp = alloc_tensor_from_pool::<A::Scalar>(&c_gemm_shape, memory_space, pool);

    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: batch_sizes.clone(),
        m,
        n,
        k,
    };
    let shapes = [a_reshaped.dims(), b_reshaped.dims(), temp.dims()];
    let plan = B::plan(ctx, &desc, &shapes)?;
    B::execute(
        ctx,
        &plan,
        A::Scalar::one(),
        &[&a_reshaped, &b_reshaped],
        A::Scalar::zero(),
        &mut temp,
    )?;

    // --- Step 5: Reshape temp → [batch..., lo..., ro...] and permute to output ---
    let expanded_shape: Vec<usize> = batch_sizes
        .iter()
        .chain(lo_sizes.iter())
        .chain(ro_sizes.iter())
        .copied()
        .collect();
    let temp_expanded = temp.reshape(&expanded_shape)?;

    // Canonical mode labels for the expanded temp
    let canonical_modes: Vec<u32> = batch_modes
        .iter()
        .chain(lo_modes.iter())
        .chain(ro_modes.iter())
        .copied()
        .collect();

    // If canonical == subs_c, we can skip the final permute
    if canonical_modes == subs_c {
        // Direct copy with alpha/beta
        if alpha == A::Scalar::one() && beta == A::Scalar::zero() {
            *output = temp_expanded;
        } else {
            let desc = PrimDescriptor::Permute {
                modes_a: canonical_modes.clone(),
                modes_b: subs_c.to_vec(),
            };
            let shapes = [temp_expanded.dims(), output.dims()];
            let plan = B::plan(ctx, &desc, &shapes)?;
            B::execute(ctx, &plan, alpha, &[&temp_expanded], beta, output)?;
        }
    } else {
        // Permute from canonical order to output order with alpha/beta
        let desc = PrimDescriptor::Permute {
            modes_a: canonical_modes,
            modes_b: subs_c.to_vec(),
        };
        let shapes = [temp_expanded.dims(), output.dims()];
        let plan = B::plan(ctx, &desc, &shapes)?;
        B::execute(ctx, &plan, alpha, &[&temp_expanded], beta, output)?;
    }

    Ok(())
}

/// Execute outer product contraction using a pre-computed plan.
fn execute_outer_with_plan<Alg, Backend>(
    ctx: &mut Backend::Context,
    plan: &OuterProductPlan,
    subs_c: &[u32],
    a: &Tensor<Alg::Scalar>,
    b: &Tensor<Alg::Scalar>,
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
) -> Result<()>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let a_reshaped = a.reshape(&plan.a_ext_shape)?;
    let b_reshaped = b.reshape(&plan.b_ext_shape)?;
    let a_bcast = a_reshaped.broadcast(&plan.canonical_shape)?;
    let b_bcast = b_reshaped.broadcast(&plan.canonical_shape)?;

    if !plan.needs_final_permute {
        let desc = PrimDescriptor::ElementwiseMul;
        let shapes = [a_bcast.dims(), b_bcast.dims(), output.dims()];
        let prim_plan = Backend::plan(ctx, &desc, &shapes)?;
        Backend::execute(ctx, &prim_plan, alpha, &[&a_bcast, &b_bcast], beta, output)
    } else {
        let memory_space = output.logical_memory_space();
        let mut temp = Tensor::<Alg::Scalar>::zeros(
            &plan.canonical_shape,
            memory_space,
            MemoryOrder::ColumnMajor,
        );
        let desc = PrimDescriptor::ElementwiseMul;
        let shapes = [a_bcast.dims(), b_bcast.dims(), temp.dims()];
        let prim_plan = Backend::plan(ctx, &desc, &shapes)?;
        Backend::execute(
            ctx,
            &prim_plan,
            Alg::Scalar::one(),
            &[&a_bcast, &b_bcast],
            Alg::Scalar::zero(),
            &mut temp,
        )?;
        let desc = PrimDescriptor::Permute {
            modes_a: plan.canonical_modes.clone(),
            modes_b: subs_c.to_vec(),
        };
        let shapes = [temp.dims(), output.dims()];
        let prim_plan = Backend::plan(ctx, &desc, &shapes)?;
        Backend::execute(ctx, &prim_plan, alpha, &[&temp], beta, output)
    }
}

/// Execute pre-reduction of unique-only axes using a pre-computed plan.
fn execute_reduce_with_plan<Alg, Backend>(
    ctx: &mut Backend::Context,
    reduce: &ReducePlan,
    tensor: &Tensor<Alg::Scalar>,
    pool: &mut BufferPool<Alg::Scalar>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let memory_space = tensor.logical_memory_space();
    let mut reduced = alloc_tensor_from_pool::<Alg::Scalar>(&reduce.out_shape, memory_space, pool);
    let desc = PrimDescriptor::Reduce {
        modes_a: reduce.original_subs.clone(),
        modes_c: reduce.kept_subs.clone(),
        op: ReduceOp::Sum,
    };
    let shapes = [tensor.dims(), reduced.dims()];
    let prim_plan = Backend::plan(ctx, &desc, &shapes)?;
    Backend::execute(
        ctx,
        &prim_plan,
        Alg::Scalar::one(),
        &[tensor],
        Alg::Scalar::zero(),
        &mut reduced,
    )?;
    Ok(reduced)
}

/// Execute GEMM contraction using a pre-computed plan.
fn execute_gemm_with_plan<Alg, Backend>(
    ctx: &mut Backend::Context,
    plan: &GemmPlan,
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
    // Pre-reduce unique-only axes if needed
    let a_reduced;
    let a_ref = if let Some(ref reduce) = plan.reduce_a {
        a_reduced = execute_reduce_with_plan::<Alg, Backend>(ctx, reduce, a, pool)?;
        &a_reduced
    } else {
        a
    };
    let b_reduced;
    let b_ref = if let Some(ref reduce) = plan.reduce_b {
        b_reduced = execute_reduce_with_plan::<Alg, Backend>(ctx, reduce, b, pool)?;
        &b_reduced
    } else {
        b
    };

    let memory_space = a.logical_memory_space();
    let nb = plan.batch_sizes.len();

    // Prepare GEMM operands with fusability check
    let a_gemm_shape: Vec<usize> = plan
        .batch_sizes
        .iter()
        .copied()
        .chain(std::iter::once(plan.m))
        .chain(std::iter::once(plan.k))
        .collect();
    let b_gemm_shape: Vec<usize> = plan
        .batch_sizes
        .iter()
        .copied()
        .chain(std::iter::once(plan.k))
        .chain(std::iter::once(plan.n))
        .collect();

    let a_prepared = prepare_one_operand::<Alg, Backend>(
        ctx,
        a_ref,
        &plan.subs_a,
        &plan.target_a,
        nb,
        plan.lo_modes.len(),
        plan.sum_modes.len(),
        &a_gemm_shape,
        pool,
    )?;
    let b_prepared = prepare_one_operand::<Alg, Backend>(
        ctx,
        b_ref,
        &plan.subs_b,
        &plan.target_b,
        nb,
        plan.sum_modes.len(),
        plan.ro_modes.len(),
        &b_gemm_shape,
        pool,
    )?;

    // Execute GEMM
    let mut temp = alloc_tensor_from_pool::<Alg::Scalar>(&plan.c_gemm_shape, memory_space, pool);
    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: plan.batch_sizes.clone(),
        m: plan.m,
        n: plan.n,
        k: plan.k,
    };
    let shapes = [a_prepared.dims(), b_prepared.dims(), temp.dims()];
    let prim_plan = Backend::plan(ctx, &desc, &shapes)?;
    Backend::execute(
        ctx,
        &prim_plan,
        Alg::Scalar::one(),
        &[&a_prepared, &b_prepared],
        Alg::Scalar::zero(),
        &mut temp,
    )?;

    // Reshape and permute to output
    let temp_expanded = temp.reshape(&plan.expanded_shape)?;

    if !plan.needs_final_permute {
        if alpha == Alg::Scalar::one() && beta == Alg::Scalar::zero() {
            *output = temp_expanded;
        } else {
            let desc = PrimDescriptor::Permute {
                modes_a: plan.canonical_modes.clone(),
                modes_b: subs_c.to_vec(),
            };
            let shapes = [temp_expanded.dims(), output.dims()];
            let prim_plan = Backend::plan(ctx, &desc, &shapes)?;
            Backend::execute(ctx, &prim_plan, alpha, &[&temp_expanded], beta, output)?;
        }
    } else {
        let desc = PrimDescriptor::Permute {
            modes_a: plan.canonical_modes.clone(),
            modes_b: subs_c.to_vec(),
        };
        let shapes = [temp_expanded.dims(), output.dims()];
        let prim_plan = Backend::plan(ctx, &desc, &shapes)?;
        Backend::execute(ctx, &prim_plan, alpha, &[&temp_expanded], beta, output)?;
    }

    Ok(())
}
