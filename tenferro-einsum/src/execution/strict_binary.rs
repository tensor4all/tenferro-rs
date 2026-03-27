use tenferro_algebra::{Conjugate, HasAlgebra, Scalar, Semiring};
use tenferro_device::Result;
use tenferro_prims::{SemiringCoreDescriptor, TensorSemiringCore, TensorTempPoolContext};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::execution::backend::{BackendContext, EinsumBackend};
use crate::execution::util::infer_memory_space;
use crate::layout::try_fuse_group_in_target_order;
use crate::planning::plan::StrictBinaryLoweringPlan;

fn is_identity_perm(perm: &[usize]) -> bool {
    perm.iter().enumerate().all(|(i, &axis)| i == axis)
}

fn permute_if_needed<T>(tensor: &Tensor<T>, perm: &[usize]) -> Result<Tensor<T>> {
    if is_identity_perm(perm) {
        Ok(tensor.clone())
    } else {
        tensor.permute(perm)
    }
}

fn inverse_perm(perm: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; perm.len()];
    for (output_axis, &canonical_axis) in perm.iter().enumerate() {
        inverse[canonical_axis] = output_axis;
    }
    inverse
}

fn bind_output_as_strict_matrix<T>(
    output: &Tensor<T>,
    plan: &StrictBinaryLoweringPlan,
) -> Result<Option<Tensor<T>>> {
    let canonical_rank = plan.canonical_output_dims.len();
    if output.dims().len() != canonical_rank || canonical_rank == 0 {
        return Ok(None);
    }

    let output_dims = output.dims();
    let output_strides = output.strides();
    let canonical_axes = inverse_perm(&plan.output_perm);
    let canonical_dims: Vec<usize> = canonical_axes
        .iter()
        .map(|&axis| output_dims[axis])
        .collect();
    let canonical_strides: Vec<isize> = canonical_axes
        .iter()
        .map(|&axis| output_strides[axis])
        .collect();

    if canonical_dims != plan.canonical_output_dims {
        return Ok(None);
    }

    let lhs_rank = plan.lhs_free_labels.len();
    let rhs_rank = plan.rhs_free_labels.len();
    let fused_left =
        try_fuse_group_in_target_order(&canonical_dims[..lhs_rank], &canonical_strides[..lhs_rank]);
    let fused_right = try_fuse_group_in_target_order(
        &canonical_dims[lhs_rank..lhs_rank + rhs_rank],
        &canonical_strides[lhs_rank..lhs_rank + rhs_rank],
    );

    match (fused_left, fused_right) {
        (Some((_, m_stride)), Some((_, n_stride))) => output
            .view_as_strided(vec![plan.m, plan.n], vec![m_stride, n_stride])
            .map(Some),
        _ => Ok(None),
    }
}

struct PreparedStrictBinaryInputs<T> {
    left_matrix: Tensor<T>,
    right_matrix: Tensor<T>,
}

fn prepare_strict_binary_inputs<T: Scalar>(
    plan: &StrictBinaryLoweringPlan,
    left: &Tensor<T>,
    right: &Tensor<T>,
) -> Result<PreparedStrictBinaryInputs<T>> {
    let left_permuted = permute_if_needed(left, &plan.lhs_perm)?;
    let right_permuted = permute_if_needed(right, &plan.rhs_perm)?;
    let left_matrix = left_permuted
        .contiguous(MemoryOrder::ColumnMajor)
        .reshape(&plan.lhs_matrix_dims)?;
    let right_matrix = right_permuted
        .contiguous(MemoryOrder::ColumnMajor)
        .reshape(&plan.rhs_matrix_dims)?;
    Ok(PreparedStrictBinaryInputs {
        left_matrix,
        right_matrix,
    })
}

fn try_execute_strict_binary_gemm<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    plan: &StrictBinaryLoweringPlan,
    inputs: &PreparedStrictBinaryInputs<Alg::Scalar>,
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
) -> Result<bool>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    let desc = SemiringCoreDescriptor::BatchedGemm {
        batch_dims: vec![],
        m: plan.m,
        n: plan.n,
        k: plan.k,
    };
    let prim_plan = match <Backend as TensorSemiringCore<Alg>>::plan(
        ctx,
        &desc,
        &[
            inputs.left_matrix.dims(),
            inputs.right_matrix.dims(),
            output.dims(),
        ],
    ) {
        Ok(plan) => plan,
        Err(_) => return Ok(false),
    };
    <Backend as TensorSemiringCore<Alg>>::execute(
        ctx,
        &prim_plan,
        alpha,
        &[&inputs.left_matrix, &inputs.right_matrix],
        beta,
        output,
    )?;
    Ok(true)
}

pub(crate) fn try_execute_strict_binary_plan<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    plan: &StrictBinaryLoweringPlan,
    left: &Tensor<Alg::Scalar>,
    right: &Tensor<Alg::Scalar>,
) -> Result<Option<Tensor<Alg::Scalar>>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    let inputs = prepare_strict_binary_inputs(plan, left, right)?;
    let memory_space = infer_memory_space(&[&inputs.left_matrix, &inputs.right_matrix])?;
    let mut out_matrix =
        Tensor::<Alg::Scalar>::zeros(&[plan.m, plan.n], memory_space, MemoryOrder::ColumnMajor)?;
    if !try_execute_strict_binary_gemm::<Alg, Backend>(
        ctx,
        plan,
        &inputs,
        Alg::one(),
        Alg::zero(),
        &mut out_matrix,
    )? {
        return Ok(None);
    }

    let reshaped = out_matrix.reshape(&plan.canonical_output_dims)?;
    if is_identity_perm(&plan.output_perm) {
        Ok(Some(reshaped))
    } else {
        reshaped.permute(&plan.output_perm).map(Some)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_execute_strict_binary_plan_into<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    plan: &StrictBinaryLoweringPlan,
    left: &Tensor<Alg::Scalar>,
    right: &Tensor<Alg::Scalar>,
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
) -> Result<Option<()>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    let inputs = prepare_strict_binary_inputs(plan, left, right)?;
    let placeholder =
        Tensor::<Alg::Scalar>::zeros(&[], output.logical_memory_space(), MemoryOrder::ColumnMajor)?;
    let output_tensor = std::mem::replace(output, placeholder);
    if let Some(mut fused_output) = bind_output_as_strict_matrix(&output_tensor, plan)? {
        let output_dims = output_tensor.dims().to_vec();
        let output_strides = output_tensor.strides().to_vec();
        drop(output_tensor);

        let executed = try_execute_strict_binary_gemm::<Alg, Backend>(
            ctx,
            plan,
            &inputs,
            alpha,
            beta,
            &mut fused_output,
        )?;
        let restored = fused_output.view_as_strided(output_dims, output_strides)?;
        drop(fused_output);
        *output = restored;
        return Ok(executed.then_some(()));
    }

    *output = output_tensor;
    let Some(strict_output) =
        try_execute_strict_binary_plan::<Alg, Backend>(ctx, plan, left, right)?
    else {
        return Ok(None);
    };
    let desc = SemiringCoreDescriptor::MakeContiguous;
    let prim_plan = <Backend as TensorSemiringCore<Alg>>::plan(
        ctx,
        &desc,
        &[strict_output.dims(), output.dims()],
    )?;
    <Backend as TensorSemiringCore<Alg>>::execute(
        ctx,
        &prim_plan,
        alpha,
        &[&strict_output],
        beta,
        output,
    )?;
    Ok(Some(()))
}
