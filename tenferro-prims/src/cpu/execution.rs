use strided_view::{StridedView, StridedViewMut};
use tenferro_algebra::Scalar;
use tenferro_device::Result;
use tenferro_tensor::Tensor;

use crate::validate_execute_inputs;

use super::batched_gemm::execute_batched_gemm;
use super::context::CpuContext;
use super::contract::{execute_contract, execute_elementwise_binary};
use super::plan::CpuPlan;
use super::reduction::{execute_anti_diag, execute_anti_trace, execute_reduce_sum, execute_trace};
use super::{tensor_to_view, tensor_to_view_mut};

pub(super) fn execute_make_contiguous<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
) -> Result<()> {
    if alpha == T::one() && beta == T::zero() {
        strided_perm::copy_into_par(output, input)
            .map_err(|e| tenferro_device::Error::StrideError(e.to_string()))?;
    } else {
        let dims = output.dims().to_vec();
        crate::for_each_index(&dims, |idx| {
            let val = alpha * input.get(idx);
            if beta == T::zero() {
                output.set(idx, val);
            } else {
                output.set(idx, val + beta * output.get(idx));
            }
        });
    }
    Ok(())
}

pub(super) fn execute_semiring_plan<S: Scalar>(
    ctx: &mut CpuContext,
    plan: &CpuPlan<S>,
    alpha: S,
    inputs: &[&Tensor<S>],
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    let views: Vec<StridedView<S>> = inputs
        .iter()
        .map(|tensor| tensor_to_view(tensor))
        .collect::<Result<Vec<_>>>()?;
    let view_refs: Vec<&StridedView<S>> = views.iter().collect();
    let mut out_view = tensor_to_view_mut(output)?;

    match plan {
        CpuPlan::MakeContiguous { .. } => {
            validate_execute_inputs(inputs, 1, "MakeContiguous")?;
            execute_make_contiguous(alpha, view_refs[0], beta, &mut out_view)
        }
        CpuPlan::BatchedGemm {
            batch_dims,
            m,
            n,
            k,
            ..
        } => {
            validate_execute_inputs(inputs, 2, "BatchedGemm")?;
            execute_batched_gemm(
                ctx,
                alpha,
                &view_refs,
                beta,
                &mut out_view,
                batch_dims,
                *m,
                *n,
                *k,
            )
        }
        CpuPlan::ReduceAdd { reduced_axes, .. } => {
            validate_execute_inputs(inputs, 1, "ReduceAdd")?;
            execute_reduce_sum(alpha, view_refs[0], beta, &mut out_view, reduced_axes)
        }
        CpuPlan::Trace {
            components,
            comp_dims,
            free_axes,
            ..
        } => {
            validate_execute_inputs(inputs, 1, "Trace")?;
            execute_trace(
                alpha,
                view_refs[0],
                beta,
                &mut out_view,
                components,
                comp_dims,
                free_axes,
            )
        }
        CpuPlan::AntiTrace {
            free_axes,
            components,
            comp_dims,
            ..
        } => {
            validate_execute_inputs(inputs, 1, "AntiTrace")?;
            execute_anti_trace(
                alpha,
                view_refs[0],
                beta,
                &mut out_view,
                components,
                comp_dims,
                free_axes,
            )
        }
        CpuPlan::AntiDiag {
            free_axes,
            components,
            comp_dims,
            generative_comps,
            ..
        } => {
            validate_execute_inputs(inputs, 1, "AntiDiag")?;
            execute_anti_diag(
                alpha,
                view_refs[0],
                beta,
                &mut out_view,
                components,
                comp_dims,
                free_axes,
                generative_comps,
            )
        }
        CpuPlan::ElementwiseBinary { op, .. } => {
            validate_execute_inputs(inputs, 2, "ElementwiseBinary")?;
            execute_elementwise_binary(alpha, &view_refs, beta, &mut out_view, *op)
        }
        CpuPlan::Contract {
            modes_a,
            modes_b,
            modes_c,
            gemm_spec,
            ..
        } => {
            validate_execute_inputs(inputs, 2, "Contract")?;
            execute_contract(
                alpha,
                &view_refs,
                beta,
                &mut out_view,
                modes_a,
                modes_b,
                modes_c,
                gemm_spec.as_ref(),
            )
        }
    }
}
