// One and Zero are needed to call Alg::Scalar::one() / ::zero() via the
// Scalar supertrait. Rust incorrectly flags these as unused when the call
// site is an associated type (Alg::Scalar) rather than a concrete type.
#[allow(unused_imports)]
use num_traits::{One, Zero};
use tenferro_algebra::{Algebra, HasAlgebra, Scalar};
use tenferro_device::Result;
use tenferro_prims::{Extension, PrimDescriptor, ReduceOp, TensorPrims};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::plan::{GemmPlan, OuterProductPlan, ReducePlan, StepPlan, StepStrategy};
use crate::pool::BufferPool;
use crate::prepare::prepare_one_operand;
use crate::util::alloc_tensor_from_pool;

/// Execute a pairwise contraction using a pre-computed step plan.
///
/// Dispatches to the appropriate handler based on the pre-computed strategy:
/// ElementwiseMul, OuterProduct, or Contraction (unified Contract/GEMM path).
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
                execute_contraction_unified::<Alg, Backend>(
                    ctx, None, None, subs_a, subs_b, subs_c, a, b, alpha, beta, output, pool,
                )
            }
        }
        StepStrategy::OuterProduct(op_plan) => {
            if Backend::has_extension_for(Extension::ElementwiseMul) {
                execute_outer_with_plan::<Alg, Backend>(ctx, op_plan, subs_c, a, b, alpha, beta, output)
            } else {
                execute_contraction_unified::<Alg, Backend>(
                    ctx, None, None, subs_a, subs_b, subs_c, a, b, alpha, beta, output, pool,
                )
            }
        }
        StepStrategy::Contraction(gemm_plan_opt) => {
            execute_contraction_unified::<Alg, Backend>(
                ctx,
                gemm_plan_opt.as_ref(),
                prim_plan,
                subs_a,
                subs_b,
                subs_c,
                a,
                b,
                alpha,
                beta,
                output,
                pool,
            )
        }
    }
}

/// Unified contraction handler: pre-reduce → Contract ext → fallback GEMM.
fn execute_contraction_unified<Alg, Backend>(
    ctx: &mut Backend::Context,
    gemm_plan: Option<&GemmPlan>,
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
    match gemm_plan {
        Some(plan) => {
            // GEMM-compatible: pre-reduce, then try Contract ext, then fallback GEMM

            // 1. Pre-reduce unique-only axes
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
            // Non-GEMM pattern (trace-like): Contract prim
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

/// Execute GEMM fallback after pre-reduction has already been applied.
///
/// Takes already-reduced tensors (a_ref, b_ref) and performs:
/// prepare_one_operand + BatchedGemm + reshape + permute to output.
fn execute_gemm_after_reduce<Alg, Backend>(
    ctx: &mut Backend::Context,
    plan: &GemmPlan,
    subs_c: &[u32],
    a_ref: &Tensor<Alg::Scalar>,
    b_ref: &Tensor<Alg::Scalar>,
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
    let memory_space = a_ref.logical_memory_space();
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
