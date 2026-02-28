use std::collections::HashMap;

use num_traits::{One, Zero};
use tenferro_algebra::{Algebra, HasAlgebra, Scalar};
use tenferro_device::{Error, Result};
use tenferro_prims::{Extension, PrimDescriptor, TensorPrims};
use tenferro_tensor::Tensor;

use crate::dispatch::execute_pairwise_with_plan;
use crate::nested::NestedEinsum;
use crate::plan::{compile_step_plans, StepStrategy};
use crate::pool::BufferPool;
use crate::tree::ContractionTree;
use crate::unary::execute_single_tensor_einsum;
use crate::util::{alloc_tensor_from_pool, infer_memory_space};

// Import einsum_with_subscripts for recursive NestedEinsum execution.
use crate::api::einsum_with_subscripts;

/// Execute a ContractionTree against concrete input tensors.
pub(crate) fn execute_tree<Alg, Backend>(
    ctx: &mut Backend::Context,
    tree: &ContractionTree,
    operands: &[&Tensor<Alg::Scalar>],
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
    // Pre-compile step plans to avoid per-step HashMap/HashSet allocations.
    let step_plans = compile_step_plans(tree);

    // Pre-compute Backend prim plans for Contract/ElementwiseMul extensions.
    // This moves plan() calls out of the per-step hot loop.
    let use_contract = Backend::has_extension_for(Extension::Contract);
    let use_ewmul = Backend::has_extension_for(Extension::ElementwiseMul);
    let prim_plans: Vec<Option<Backend::Plan>> = if use_contract || use_ewmul {
        step_plans
            .iter()
            .enumerate()
            .map(|(step_idx, sp)| {
                // Pre-compute Contract plans for Contraction(None) steps only.
                // Contraction(Some(_)) steps pre-reduce at runtime, so Contract plan
                // must be computed after reduction with the reduced subscripts/shapes.
                let needs_contract = use_contract
                    && matches!(sp.strategy, StepStrategy::Contraction(None));
                let needs_ewmul = use_ewmul && matches!(sp.strategy, StepStrategy::ElementwiseMul);
                if !needs_contract && !needs_ewmul {
                    return None;
                }
                let step = &tree.steps[step_idx];
                let subs_a = &tree.operand_subs[step.left];
                let subs_b = &tree.operand_subs[step.right];
                let is_last = step_idx == tree.steps.len() - 1;
                let subs_c = if is_last {
                    &tree.subscripts.output
                } else {
                    &tree.operand_subs[n_inputs + step_idx]
                };
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
                let shape_c: &[usize] = if is_last {
                    output.dims()
                } else {
                    &tree.step_output_shapes[step_idx]
                };
                let desc = if needs_contract {
                    PrimDescriptor::Contract {
                        modes_a: subs_a.to_vec(),
                        modes_b: subs_b.to_vec(),
                        modes_c: subs_c.to_vec(),
                    }
                } else {
                    PrimDescriptor::ElementwiseMul
                };
                let shapes = [shape_a, shape_b, shape_c];
                Backend::plan(ctx, &desc, &shapes).ok()
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
            // Last step: write directly to output with alpha/beta
            execute_pairwise_with_plan::<Alg, Backend>(
                ctx,
                &step_plans[step_idx],
                prim_plans[step_idx].as_ref(),
                subs_left,
                subs_right,
                &tree.subscripts.output,
                left,
                right,
                alpha,
                beta,
                output,
                pool,
            )?;
        } else {
            // Intermediate step: create new tensor with alpha=1, beta=0
            let result_idx = n_inputs + step_idx;
            let subs_result = &tree.operand_subs[result_idx];
            let result_shape = &tree.step_output_shapes[step_idx];
            let mut result =
                alloc_tensor_from_pool::<Alg::Scalar>(result_shape, memory_space, pool);
            execute_pairwise_with_plan::<Alg, Backend>(
                ctx,
                &step_plans[step_idx],
                prim_plans[step_idx].as_ref(),
                subs_left,
                subs_right,
                subs_result,
                left,
                right,
                Alg::Scalar::one(),
                Alg::Scalar::zero(),
                &mut result,
                pool,
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
    ctx: &mut Backend::Context,
    nested: &NestedEinsum,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
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
    ctx: &mut Backend::Context,
    nested: &NestedEinsum,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
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
