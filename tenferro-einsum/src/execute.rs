use std::collections::HashMap;

use tenferro_algebra::{HasAlgebra, Scalar, Semiring};
use tenferro_device::{Error, Result};
use tenferro_prims::{
    SemiringBinaryOp, SemiringCoreDescriptor, SemiringFastPathDescriptor, TensorSemiringCore,
    TensorSemiringFastPath,
};
use tenferro_tensor::Tensor;

use crate::backend::{BackendContext, BackendPlan, EinsumBackend};
use crate::dispatch::execute_pairwise_with_plan;
use crate::nested::NestedEinsum;
use crate::pool::BufferPool;
use crate::tree::ContractionTree;
use crate::unary::execute_single_tensor_einsum;
use crate::util::{alloc_tensor_from_pool, infer_memory_space};

// Import einsum_with_subscripts for recursive NestedEinsum execution.
use crate::api::einsum_with_subscripts;

/// Execute a ContractionTree against concrete input tensors.
///
/// When `lazy_final` is true, the final step may produce a non-contiguous
/// view (lazy permute) instead of a physical copy. This is safe when the
/// output tensor is internally allocated (e.g. by `einsum_with_plan`) but
/// NOT when it is a user-provided buffer (e.g. `einsum_with_plan_into`).
pub(crate) fn execute_tree<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    tree: &ContractionTree,
    operands: &[&Tensor<Alg::Scalar>],
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
    pool: &mut BufferPool<Alg::Scalar>,
    lazy_final: bool,
) -> Result<()>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
{
    let n_inputs = tree.subscripts.inputs.len();

    if tree.steps.is_empty() {
        // Single-tensor case
        if n_inputs != 1 {
            return Err(Error::InvalidArgument(
                "ContractionTree with no steps requires exactly 1 input".into(),
            ));
        }
        return execute_single_tensor_einsum::<Alg, Backend>(
            ctx,
            &tree.subscripts.inputs[0],
            &tree.subscripts.output,
            operands[0],
            alpha,
            beta,
            output,
            pool,
        );
    }

    // Multi-tensor case: follow the contraction tree.
    // Step plans are pre-compiled and cached in the ContractionTree.
    let step_plans = &tree.step_plans;

    // Pre-compute Backend prim plans for ElementwiseMul extension.
    // Only pre-compute for steps where all dimensions are batch (m==n==k==1)
    // and no diagonal extraction is needed (shapes are known at plan time).
    let use_ewmul = <Backend as TensorSemiringFastPath<Alg>>::has_fast_path(
        SemiringFastPathDescriptor::ElementwiseBinary {
            op: SemiringBinaryOp::Mul,
        },
    );
    let prim_plans: Vec<Option<BackendPlan<Alg, Backend>>> = if use_ewmul {
        step_plans
            .iter()
            .enumerate()
            .map(|(step_idx, sp)| {
                // Only pre-compute EwMul for pure batch steps without diagonal extraction.
                // Diagonal extraction changes shapes at runtime, so we can't plan ahead.
                if sp.gemm.m != 1 || sp.gemm.n != 1 || sp.gemm.k != 1 {
                    return None;
                }
                if sp.diag_a.is_some() || sp.diag_b.is_some() {
                    return None;
                }
                let step = &tree.steps[step_idx];
                let shape_a: &[usize] = if step.left < n_inputs {
                    operands[step.left].dims()
                } else {
                    &tree.step_output_shapes[step.left - n_inputs]
                };
                let shape_b: &[usize] = if step.right < n_inputs {
                    operands[step.right].dims()
                } else {
                    &tree.step_output_shapes[step.right - n_inputs]
                };
                let is_last = step_idx == tree.steps.len() - 1;
                let shape_c: &[usize] = if is_last {
                    output.dims()
                } else {
                    &tree.step_output_shapes[step_idx]
                };
                let desc = SemiringFastPathDescriptor::ElementwiseBinary {
                    op: SemiringBinaryOp::Mul,
                };
                let shapes = [shape_a, shape_b, shape_c];
                <Backend as TensorSemiringFastPath<Alg>>::plan(ctx, &desc, &shapes).ok()
            })
            .collect()
    } else {
        (0..step_plans.len()).map(|_| None).collect()
    };

    // Pre-compute BatchedGemm plans for all steps.
    // Shapes are fully determined by StepPlan (m, k, n, batch_sizes) —
    // independent of actual tensor data or strides.
    let use_contract = <Backend as TensorSemiringFastPath<Alg>>::has_fast_path(
        SemiringFastPathDescriptor::Contract {
            modes_a: Vec::new(),
            modes_b: Vec::new(),
            modes_c: Vec::new(),
        },
    );
    let gemm_plans: Vec<Option<BackendPlan<Alg, Backend>>> = if !use_contract {
        step_plans
            .iter()
            .map(|sp| {
                let desc = SemiringCoreDescriptor::BatchedGemm {
                    batch_dims: sp.gemm.batch_sizes.clone(),
                    m: sp.gemm.m,
                    n: sp.gemm.n,
                    k: sp.gemm.k,
                };
                let shapes = [
                    sp.gemm.a_gemm_shape.as_slice(),
                    sp.gemm.b_gemm_shape.as_slice(),
                    sp.gemm.c_gemm_shape.as_slice(),
                ];
                <Backend as TensorSemiringCore<Alg>>::plan(ctx, &desc, &shapes).ok()
            })
            .collect()
    } else {
        (0..step_plans.len()).map(|_| None).collect()
    };

    // Use Vec-indexed storage instead of HashMap for O(1) access.
    let memory_space = infer_memory_space(operands)?;
    let total_slots = n_inputs + tree.steps.len();
    let mut intermediates: Vec<Option<Tensor<Alg::Scalar>>> = Vec::with_capacity(total_slots);
    intermediates.resize_with(total_slots, || None);

    // Count remaining uses for each operand/index in the contraction schedule.
    let mut use_counts = vec![0usize; total_slots];
    for step in &tree.steps {
        use_counts[step.left] += 1;
        use_counts[step.right] += 1;
    }

    for (step_idx, step) in tree.steps.iter().enumerate() {
        let left: &Tensor<Alg::Scalar> = if step.left < n_inputs {
            operands[step.left]
        } else {
            intermediates[step.left].as_ref().ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "missing intermediate tensor at index {}",
                    step.left
                ))
            })?
        };
        let right: &Tensor<Alg::Scalar> = if step.right < n_inputs {
            operands[step.right]
        } else {
            intermediates[step.right].as_ref().ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "missing intermediate tensor at index {}",
                    step.right
                ))
            })?
        };

        let subs_left = &tree.operand_subs[step.left];
        let subs_right = &tree.operand_subs[step.right];
        let is_last = step_idx == tree.steps.len() - 1;

        if is_last {
            // Last step: write directly to output with alpha/beta.
            execute_pairwise_with_plan::<Alg, Backend>(
                ctx,
                &step_plans[step_idx],
                prim_plans[step_idx].as_ref(),
                gemm_plans[step_idx].as_ref(),
                subs_left,
                subs_right,
                &tree.subscripts.output,
                left,
                right,
                alpha,
                beta,
                output,
                pool,
                lazy_final,
            )?;
        } else {
            // Intermediate step: create new tensor with alpha=1, beta=0.
            // lazy_output=true: may return a non-contiguous view to avoid
            // a physical permute; the next step handles arbitrary strides.
            let result_idx = n_inputs + step_idx;
            let subs_result = &tree.operand_subs[result_idx];
            let result_shape = &tree.step_output_shapes[step_idx];
            let mut result =
                alloc_tensor_from_pool::<Alg::Scalar>(result_shape, memory_space, pool);
            execute_pairwise_with_plan::<Alg, Backend>(
                ctx,
                &step_plans[step_idx],
                prim_plans[step_idx].as_ref(),
                gemm_plans[step_idx].as_ref(),
                subs_left,
                subs_right,
                subs_result,
                left,
                right,
                Alg::one(),
                Alg::zero(),
                &mut result,
                pool,
                true,
            )?;
            intermediates[result_idx] = Some(result);
        }

        // Release consumed intermediates when their last use is complete.
        let mut consumed = [step.left, step.right];
        consumed.sort_unstable();
        for (i, idx) in consumed.iter().enumerate() {
            if i == 1 && consumed[0] == consumed[1] {
                continue;
            }
            use_counts[*idx] = use_counts[*idx].saturating_sub(1);
            if *idx >= n_inputs && use_counts[*idx] == 0 {
                if let Some(t) = intermediates[*idx].take() {
                    if let Some(data) = t.try_into_data_vec() {
                        pool.return_buf(data);
                    }
                }
            }
        }
    }

    Ok(())
}

/// Execute a [`NestedEinsum`] tree recursively (bottom-up).
///
/// Each leaf returns a clone of the corresponding input tensor. Each internal
/// node recursively evaluates its children, then calls
/// [`einsum_with_subscripts`] on the intermediate results.
pub(crate) fn execute_nested<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    nested: &NestedEinsum,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
{
    // Validate operand count at the top level
    let n_leaves = nested.count_leaves();
    if operands.len() != n_leaves {
        return Err(Error::InvalidArgument(format!(
            "NestedEinsum expects {n_leaves} operands, got {}",
            operands.len()
        )));
    }
    execute_nested_inner::<Alg, Backend>(ctx, nested, operands, size_dict)
}

/// Recursive inner implementation (no operand count check — done by caller).
fn execute_nested_inner<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    nested: &NestedEinsum,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
{
    match nested {
        NestedEinsum::Leaf(idx) => Ok(operands[*idx].clone()),
        NestedEinsum::Node {
            subscripts,
            children,
        } => {
            let intermediates: Vec<Tensor<Alg::Scalar>> = children
                .iter()
                .map(|child| execute_nested_inner::<Alg, Backend>(ctx, child, operands, size_dict))
                .collect::<Result<_>>()?;

            let refs: Vec<&Tensor<Alg::Scalar>> = intermediates.iter().collect();
            einsum_with_subscripts::<Alg, Backend>(ctx, subscripts, &refs, size_dict)
        }
    }
}
