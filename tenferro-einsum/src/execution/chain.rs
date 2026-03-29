use tenferro_algebra::{Conjugate, HasAlgebra, Scalar, Semiring};
use tenferro_device::{Error, Result};
use tenferro_prims::{
    SemiringBinaryOp, SemiringCoreDescriptor, SemiringFastPathDescriptor, TensorSemiringCore,
    TensorSemiringFastPath, TensorTempPoolContext,
};
use tenferro_tensor::Tensor;

use crate::execution::backend::{BackendContext, BackendPlan, EinsumBackend};
use crate::execution::dispatch::execute_pairwise_with_plan;
use crate::execution::pool::TensorBufferPool;
use crate::execution::strict_binary::try_execute_strict_binary_plan_into;
use crate::execution::util::{alloc_tensor_from_pool, infer_memory_space};
use crate::planning::plan::StepPlan;
use crate::planning::tree::{ContractionTree, LinearChainPlan};

pub(crate) fn maybe_plan_ewmul_for_step<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    step_plan: &StepPlan,
    shape_a: &[usize],
    shape_b: &[usize],
    shape_c: &[usize],
) -> Option<BackendPlan<Alg, Backend>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    if step_plan.strict_binary.is_some() {
        return None;
    }

    if !<Backend as TensorSemiringFastPath<Alg>>::has_fast_path(
        SemiringFastPathDescriptor::ElementwiseBinary {
            op: SemiringBinaryOp::Mul,
        },
    ) {
        return None;
    }

    if step_plan.gemm.m != 1 || step_plan.gemm.n != 1 || step_plan.gemm.k != 1 {
        return None;
    }
    if step_plan.diag_a.is_some() || step_plan.diag_b.is_some() {
        return None;
    }

    let desc = SemiringFastPathDescriptor::ElementwiseBinary {
        op: SemiringBinaryOp::Mul,
    };
    let shapes = [shape_a, shape_b, shape_c];
    <Backend as TensorSemiringFastPath<Alg>>::plan(ctx, &desc, &shapes).ok()
}

pub(crate) fn maybe_plan_gemm_for_step<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    step_plan: &StepPlan,
    skip_when_strict_binary: bool,
) -> Option<BackendPlan<Alg, Backend>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    if skip_when_strict_binary && step_plan.strict_binary.is_some() {
        return None;
    }

    if <Backend as TensorSemiringFastPath<Alg>>::has_fast_path(
        SemiringFastPathDescriptor::Contract {
            modes_a: Vec::new(),
            modes_b: Vec::new(),
            modes_c: Vec::new(),
        },
    ) {
        return None;
    }
    let desc = SemiringCoreDescriptor::BatchedGemm {
        batch_dims: step_plan.gemm.batch_sizes.clone(),
        m: step_plan.gemm.m,
        n: step_plan.gemm.n,
        k: step_plan.gemm.k,
    };
    let shapes = [
        step_plan.gemm.a_gemm_shape.as_slice(),
        step_plan.gemm.b_gemm_shape.as_slice(),
        step_plan.gemm.c_gemm_shape.as_slice(),
    ];
    <Backend as TensorSemiringCore<Alg>>::plan(ctx, &desc, &shapes).ok()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_binary_step_with_plans<Alg, Backend, P>(
    ctx: &mut BackendContext<Alg, Backend>,
    step_plan: &StepPlan,
    ewmul_plan: Option<&BackendPlan<Alg, Backend>>,
    gemm_plan: Option<&BackendPlan<Alg, Backend>>,
    subs_a: &[u32],
    subs_b: &[u32],
    subs_c: &[u32],
    left: &Tensor<Alg::Scalar>,
    right: &Tensor<Alg::Scalar>,
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
    pool: &mut P,
    lazy_final: bool,
) -> Result<()>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
    P: TensorBufferPool<Alg::Scalar> + ?Sized,
{
    if gemm_plan.is_some() {
        return execute_pairwise_with_plan::<Alg, Backend, P>(
            ctx, step_plan, ewmul_plan, gemm_plan, subs_a, subs_b, subs_c, left, right, alpha,
            beta, output, pool, lazy_final,
        );
    }

    let contract_desc = SemiringFastPathDescriptor::Contract {
        modes_a: step_plan.gemm.subs_a.clone(),
        modes_b: step_plan.gemm.subs_b.clone(),
        modes_c: subs_c.to_vec(),
    };
    if <Backend as TensorSemiringFastPath<Alg>>::has_fast_path(contract_desc) {
        return execute_pairwise_with_plan::<Alg, Backend, P>(
            ctx, step_plan, ewmul_plan, gemm_plan, subs_a, subs_b, subs_c, left, right, alpha,
            beta, output, pool, lazy_final,
        );
    }

    if let Some(strict) = &step_plan.strict_binary {
        let strict_done = !left.is_conjugated()
            && !right.is_conjugated()
            && try_execute_strict_binary_plan_into::<Alg, Backend>(
                ctx, strict, left, right, alpha, beta, output,
            )?
            .is_some();
        if strict_done {
            return Ok(());
        }
    }
    execute_pairwise_with_plan::<Alg, Backend, P>(
        ctx, step_plan, ewmul_plan, gemm_plan, subs_a, subs_b, subs_c, left, right, alpha, beta,
        output, pool, lazy_final,
    )
}

pub(crate) fn execute_binary_step<Alg, Backend, P>(
    ctx: &mut BackendContext<Alg, Backend>,
    step_plan: &StepPlan,
    subs_a: &[u32],
    subs_b: &[u32],
    subs_c: &[u32],
    left: &Tensor<Alg::Scalar>,
    right: &Tensor<Alg::Scalar>,
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
    pool: &mut P,
    lazy_final: bool,
) -> Result<()>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
    P: TensorBufferPool<Alg::Scalar> + ?Sized,
{
    let ewmul_plan = maybe_plan_ewmul_for_step::<Alg, Backend>(
        ctx,
        step_plan,
        left.dims(),
        right.dims(),
        output.dims(),
    );
    let gemm_plan = maybe_plan_gemm_for_step::<Alg, Backend>(ctx, step_plan, false);
    execute_binary_step_with_plans::<Alg, Backend, P>(
        ctx,
        step_plan,
        ewmul_plan.as_ref(),
        gemm_plan.as_ref(),
        subs_a,
        subs_b,
        subs_c,
        left,
        right,
        alpha,
        beta,
        output,
        pool,
        lazy_final,
    )
}

pub(crate) fn execute_linear_chain_tree<Alg, Backend, P>(
    ctx: &mut BackendContext<Alg, Backend>,
    tree: &ContractionTree,
    chain: &LinearChainPlan,
    operands: &[&Tensor<Alg::Scalar>],
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
    pool: &mut P,
    lazy_final: bool,
) -> Result<()>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
    P: TensorBufferPool<Alg::Scalar> + ?Sized,
{
    let n_inputs = tree.subscripts.inputs.len();
    let first_step = &tree.steps[0];
    let first_result_shape = &tree.step_output_shapes[0];
    let memory_space = infer_memory_space(operands)?;

    let mut current =
        alloc_tensor_from_pool::<Alg::Scalar, _, P>(ctx, first_result_shape, memory_space, pool)?;
    execute_binary_step::<Alg, Backend, P>(
        ctx,
        &tree.step_plans[0],
        &tree.operand_subs[first_step.left],
        &tree.operand_subs[first_step.right],
        &tree.operand_subs[n_inputs],
        operands[first_step.left],
        operands[first_step.right],
        Alg::one(),
        Alg::zero(),
        &mut current,
        pool,
        true,
    )?;

    for (attach_idx, attachment) in chain.attachments.iter().enumerate() {
        let step_idx = attach_idx + 1;
        let current_subs = &tree.operand_subs[n_inputs + step_idx - 1];
        let operand = operands[attachment.operand];
        let is_last = step_idx == tree.steps.len() - 1;

        if is_last {
            let (subs_a, subs_b, tensor_a, tensor_b) = if attachment.prev_on_left {
                (
                    current_subs,
                    &tree.operand_subs[attachment.operand],
                    &current,
                    operand,
                )
            } else {
                (
                    &tree.operand_subs[attachment.operand],
                    current_subs,
                    operand,
                    &current,
                )
            };
            execute_binary_step::<Alg, Backend, P>(
                ctx,
                &tree.step_plans[step_idx],
                subs_a,
                subs_b,
                &tree.subscripts.output,
                tensor_a,
                tensor_b,
                alpha,
                beta,
                output,
                pool,
                lazy_final,
            )?;
            if let Some(data) = current.try_into_data_vec() {
                pool.return_buf(data);
            }
            return Ok(());
        }

        let mut next = alloc_tensor_from_pool::<Alg::Scalar, _, P>(
            ctx,
            &tree.step_output_shapes[step_idx],
            memory_space,
            pool,
        )?;
        let (subs_a, subs_b, tensor_a, tensor_b) = if attachment.prev_on_left {
            (
                current_subs,
                &tree.operand_subs[attachment.operand],
                &current,
                operand,
            )
        } else {
            (
                &tree.operand_subs[attachment.operand],
                current_subs,
                operand,
                &current,
            )
        };
        let result_subs = &tree.operand_subs[n_inputs + step_idx];
        execute_binary_step::<Alg, Backend, P>(
            ctx,
            &tree.step_plans[step_idx],
            subs_a,
            subs_b,
            result_subs,
            tensor_a,
            tensor_b,
            Alg::one(),
            Alg::zero(),
            &mut next,
            pool,
            true,
        )?;
        if let Some(data) = current.try_into_data_vec() {
            pool.return_buf(data);
        }
        current = next;
    }

    Err(Error::InvalidArgument(
        "linear chain execution requires at least one attachment or a binary fast path".into(),
    ))
}

#[cfg(test)]
#[path = "chain_tests.rs"]
mod tests;
