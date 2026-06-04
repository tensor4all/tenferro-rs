// TODO: Remaining einsum optimizations
//
// The current v2 einsum lowering is correct and already removes some
// intermediate permutations by keeping N-ary intermediates in canonical dot
// order. The following optimizations from v1 / the spec are still partial or
// not yet implemented:
//
// Compiler passes (spec: optimizer-passes.md):
//   - TransposeFolding: partially absorb Transpose into DotGeneral
//     dimension_numbers when the free/contract/batch axis order remains
//     compatible with the lowering.
//     v1 equivalent: lazy permutation (dispatch.rs:446-454).
//     Impact: eliminates physical copies for supported permuted GEMM inputs.
//   - DotDimensionSorter: sort contracting dims to avoid transposes.
//     v1 equivalent: implicit (modes already ordered).
//   - DotDecomposer: canonicalize DotGeneral to [batch, M, K] × [batch, K, N].
//     v1 equivalent: fusability check + partial materialization.
//     Impact: maps arbitrary DotGeneral to BatchedGemm without extra copies.
//   - ReductionSimplification: hoist independent ReduceSum before DotGeneral.
//     v1 equivalent: pre-reduction (dispatch.rs:121-139).
//
// Einsum-level optimizations:
//   - Diagonal embedding ("i->ii"): requires Scatter op (not yet implemented).
//   - Hyper-edge einsum ("ik,k,kj->ij"): 3+ tensors sharing an index.
//     Currently decomposed into binary steps; v1 handled this with a
//     specialized dispatch path.
//   - Binary diagonal ("ii,jk->ijk"): v1 diagonal plan in dispatch.rs.
//     Currently works via ExtractDiag + standard contraction, but v1 had
//     fused paths for better performance.
//
// Execution-level optimizations:
//   - Stride-aware engine: v1 inspects strides at dispatch time and uses
//     BLAS trans flags for transposed inputs. v2 engine does physical copies.
//   - Buffer pooling: v1 reuses buffers via Arc refcount + pool.
//     v2 has last_use liveness analysis but no pool.

use std::collections::{HashMap, HashSet};

use computegraph::graph::GraphBuilder;
use computegraph::types::{OperationRole, ValueRef};

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::DotGeneralConfig;

use crate::planning::tree::ContractionTree;
use crate::{Error, Result};

#[derive(Clone, Debug)]
struct LabeledVal {
    val: ValueRef<StdTensorOp>,
    labels: Vec<u32>,
    shape: Vec<usize>,
}

fn label_size_map(labels: &[u32], shape: &[usize]) -> Vec<(u32, usize)> {
    labels.iter().copied().zip(shape.iter().copied()).collect()
}

fn builder_invalid_argument(message: impl Into<String>) -> Error {
    Error::InvalidArgument(format!("einsum builder: {}", message.into()))
}

fn find_label_axis(labels: &[u32], label: u32) -> Result<usize> {
    labels
        .iter()
        .position(|candidate| *candidate == label)
        .ok_or_else(|| builder_invalid_argument(format!("missing label {label} in {labels:?}")))
}

fn find_label_size(label: u32, label_sizes: &[&[(u32, usize)]]) -> Result<usize> {
    for sizes in label_sizes {
        for &(candidate, size) in *sizes {
            if candidate == label {
                return Ok(size);
            }
        }
    }
    Err(builder_invalid_argument(format!(
        "missing size for label {label}"
    )))
}

fn select_outer_product_label_order(
    canonical_labels: &[u32],
    target_labels: Option<&[u32]>,
) -> Vec<u32> {
    let Some(target_labels) = target_labels else {
        return canonical_labels.to_vec();
    };
    if target_labels.len() != canonical_labels.len() {
        return canonical_labels.to_vec();
    }

    let mut used = vec![false; canonical_labels.len()];
    for &label in target_labels {
        let Some(axis) = canonical_labels
            .iter()
            .enumerate()
            .find_map(|(axis, candidate)| (*candidate == label && !used[axis]).then_some(axis))
        else {
            return canonical_labels.to_vec();
        };
        used[axis] = true;
    }

    target_labels.to_vec()
}

fn labeled_operand<'a>(
    operands: &'a [LabeledVal],
    index: usize,
    role: &'static str,
) -> Result<&'a LabeledVal> {
    operands
        .get(index)
        .ok_or_else(|| builder_invalid_argument(format!("missing {role} operand at index {index}")))
}

fn reduce_val(
    builder: &mut GraphBuilder<StdTensorOp>,
    lv: &LabeledVal,
    reduce_labels: &HashSet<u32>,
) -> LabeledVal {
    if reduce_labels.is_empty() {
        return lv.clone();
    }
    let reduce_axes: Vec<usize> = lv
        .labels
        .iter()
        .enumerate()
        .filter(|(_, l)| reduce_labels.contains(l))
        .map(|(i, _)| i)
        .collect();
    if reduce_axes.is_empty() {
        return lv.clone();
    }
    let reduce_set: HashSet<usize> = reduce_axes.iter().copied().collect();
    let new_labels: Vec<u32> = lv
        .labels
        .iter()
        .enumerate()
        .filter(|(i, _)| !reduce_set.contains(i))
        .map(|(_, &l)| l)
        .collect();
    let new_shape: Vec<usize> = lv
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| !reduce_set.contains(i))
        .map(|(_, &s)| s)
        .collect();
    let outputs = builder.add_operation(
        StdTensorOp::ReduceSum { axes: reduce_axes },
        vec![lv.val.clone()],
        OperationRole::Primary,
    );
    LabeledVal {
        val: ValueRef::Local(outputs[0]),
        labels: new_labels,
        shape: new_shape,
    }
}

/// Embed diagonal axes when the output requires higher multiplicity of a label
/// than the current tensor has. For example, "i->ii" needs to embed axis 0
/// into a new axis 1 of the same size.
fn embed_repeated(
    builder: &mut GraphBuilder<StdTensorOp>,
    lv: &LabeledVal,
    output_labels: &[u32],
) -> Result<LabeledVal> {
    // Count how many times each label appears in output vs current labels.
    let mut result = lv.clone();
    for &label in output_labels {
        let current_count = result.labels.iter().filter(|&&l| l == label).count();
        let output_count = output_labels.iter().filter(|&&l| l == label).count();
        if output_count > current_count {
            // Need to embed: find the existing axis with this label and
            // insert a duplicate axis after it.
            let axis_a = find_label_axis(&result.labels, label)?;
            // Insert the new axis right after axis_a.
            let axis_b = axis_a + 1;
            let n = result.shape[axis_a];
            let outputs = builder.add_operation(
                StdTensorOp::EmbedDiag { axis_a, axis_b },
                vec![result.val.clone()],
                OperationRole::Primary,
            );
            let mut new_labels = result.labels.clone();
            new_labels.insert(axis_b, label);
            let mut new_shape = result.shape.clone();
            new_shape.insert(axis_b, n);
            result = LabeledVal {
                val: ValueRef::Local(outputs[0]),
                labels: new_labels,
                shape: new_shape,
            };
            // Recurse to handle cases like "i->iii" (multiple embeddings).
            return embed_repeated(builder, &result, output_labels);
        }
    }
    Ok(result)
}

fn diagonalize_repeated(builder: &mut GraphBuilder<StdTensorOp>, lv: &LabeledVal) -> LabeledVal {
    let mut seen: HashMap<u32, usize> = HashMap::new();
    for (i, &label) in lv.labels.iter().enumerate() {
        if let Some(&first) = seen.get(&label) {
            // Found repeated label at axes `first` and `i`
            let outputs = builder.add_operation(
                StdTensorOp::ExtractDiag {
                    axis_a: first,
                    axis_b: i,
                },
                vec![lv.val.clone()],
                OperationRole::Primary,
            );
            let mut new_labels = lv.labels.clone();
            new_labels.remove(i);
            let mut new_shape = lv.shape.clone();
            new_shape.remove(i);
            let result = LabeledVal {
                val: ValueRef::Local(outputs[0]),
                labels: new_labels,
                shape: new_shape,
            };
            // Recurse in case there are more repeated labels
            return diagonalize_repeated(builder, &result);
        }
        seen.insert(label, i);
    }
    lv.clone()
}

fn binary_contract(
    builder: &mut GraphBuilder<StdTensorOp>,
    lhs: &LabeledVal,
    rhs: &LabeledVal,
    survive_labels: &[u32],
    reorder_result: bool,
) -> Result<LabeledVal> {
    let survive_set: HashSet<u32> = survive_labels.iter().copied().collect();
    let rhs_label_set: HashSet<u32> = rhs.labels.iter().copied().collect();
    let lhs_label_set: HashSet<u32> = lhs.labels.iter().copied().collect();

    // Pre-reduce: labels in lhs only, not in rhs and not in output
    let lhs_reduce: HashSet<u32> = lhs
        .labels
        .iter()
        .filter(|l| !rhs_label_set.contains(l) && !survive_set.contains(l))
        .copied()
        .collect();
    let rhs_reduce: HashSet<u32> = rhs
        .labels
        .iter()
        .filter(|l| !lhs_label_set.contains(l) && !survive_set.contains(l))
        .copied()
        .collect();

    let lhs = reduce_val(builder, lhs, &lhs_reduce);
    let rhs = reduce_val(builder, rhs, &rhs_reduce);

    let lhs_label_set: HashSet<u32> = lhs.labels.iter().copied().collect();
    let rhs_label_set: HashSet<u32> = rhs.labels.iter().copied().collect();

    // Classify labels
    let mut batch_labels = Vec::new();
    let mut contracting_labels = Vec::new();
    let mut lhs_free_labels = Vec::new();
    let mut rhs_free_labels = Vec::new();

    // Preserve order from lhs for batch and contracting
    for &l in &lhs.labels {
        if rhs_label_set.contains(&l) {
            if survive_set.contains(&l) {
                if !batch_labels.contains(&l) {
                    batch_labels.push(l);
                }
            } else if !contracting_labels.contains(&l) {
                contracting_labels.push(l);
            }
        } else if !lhs_free_labels.contains(&l) {
            lhs_free_labels.push(l);
        }
    }

    for &l in &rhs.labels {
        if !lhs_label_set.contains(&l) && !rhs_free_labels.contains(&l) {
            rhs_free_labels.push(l);
        }
    }

    // Build label->size map
    let lhs_sizes: Vec<(u32, usize)> = label_size_map(&lhs.labels, &lhs.shape);
    let rhs_sizes: Vec<(u32, usize)> = label_size_map(&rhs.labels, &rhs.shape);

    let label_to_size = |label: u32| -> Result<usize> {
        find_label_size(label, &[lhs_sizes.as_slice(), rhs_sizes.as_slice()])
    };

    let result = if !contracting_labels.is_empty() {
        // Use DotGeneral
        let lhs_contracting_dims: Vec<usize> = contracting_labels
            .iter()
            .map(|l| find_label_axis(&lhs.labels, *l))
            .collect::<Result<_>>()?;
        let rhs_contracting_dims: Vec<usize> = contracting_labels
            .iter()
            .map(|l| find_label_axis(&rhs.labels, *l))
            .collect::<Result<_>>()?;
        let lhs_batch_dims: Vec<usize> = batch_labels
            .iter()
            .map(|l| find_label_axis(&lhs.labels, *l))
            .collect::<Result<_>>()?;
        let rhs_batch_dims: Vec<usize> = batch_labels
            .iter()
            .map(|l| find_label_axis(&rhs.labels, *l))
            .collect::<Result<_>>()?;

        let config = DotGeneralConfig {
            lhs_contracting_dims,
            rhs_contracting_dims,
            lhs_batch_dims,
            rhs_batch_dims,
        };

        // DotGeneral output order: lhs_free + rhs_free + batch (col-major batch trailing)
        let result_labels: Vec<u32> = lhs_free_labels
            .iter()
            .chain(rhs_free_labels.iter())
            .chain(batch_labels.iter())
            .copied()
            .collect();
        let result_shape: Vec<usize> = result_labels
            .iter()
            .map(|&l| label_to_size(l))
            .collect::<Result<_>>()?;

        let outputs = builder.add_operation(
            StdTensorOp::DotGeneral { config },
            vec![lhs.val.clone(), rhs.val.clone()],
            OperationRole::Primary,
        );

        LabeledVal {
            val: ValueRef::Local(outputs[0]),
            labels: result_labels,
            shape: result_shape,
        }
    } else {
        // No contracting dims -> element-wise multiply with broadcasting
        outer_product(
            builder,
            &lhs,
            &rhs,
            &batch_labels,
            &lhs_free_labels,
            &rhs_free_labels,
            &label_to_size,
            reorder_result.then_some(survive_labels),
        )?
    };

    if !reorder_result {
        return Ok(result);
    }

    // Reorder to match the caller-visible order if needed.
    let current_labels = &result.labels;
    if current_labels.is_empty() {
        return Ok(result);
    }

    // Filter survivor labels to those present in result (to handle final reduction later)
    let result_label_set: HashSet<u32> = current_labels.iter().copied().collect();
    let target_labels: Vec<u32> = survive_labels
        .iter()
        .filter(|l| result_label_set.contains(l))
        .copied()
        .collect();

    if current_labels.len() == target_labels.len() && *current_labels == target_labels {
        return Ok(result);
    }

    // Build permutation
    let perm: Vec<usize> = target_labels
        .iter()
        .map(|l| find_label_axis(current_labels, *l))
        .collect::<Result<_>>()?;

    if perm.iter().enumerate().all(|(i, &p)| i == p) {
        return Ok(result);
    }

    let new_shape: Vec<usize> = perm.iter().map(|&p| result.shape[p]).collect();
    let outputs = builder.add_operation(
        StdTensorOp::Transpose { perm },
        vec![result.val.clone()],
        OperationRole::Primary,
    );

    Ok(LabeledVal {
        val: ValueRef::Local(outputs[0]),
        labels: target_labels,
        shape: new_shape,
    })
}

fn outer_product(
    builder: &mut GraphBuilder<StdTensorOp>,
    lhs: &LabeledVal,
    rhs: &LabeledVal,
    batch_labels: &[u32],
    lhs_free_labels: &[u32],
    rhs_free_labels: &[u32],
    label_to_size: &dyn Fn(u32) -> Result<usize>,
    target_labels: Option<&[u32]>,
) -> Result<LabeledVal> {
    let canonical_labels: Vec<u32> = lhs_free_labels
        .iter()
        .chain(rhs_free_labels.iter())
        .chain(batch_labels.iter())
        .copied()
        .collect();
    let combined_labels = select_outer_product_label_order(&canonical_labels, target_labels);
    let combined_shape: Vec<usize> = combined_labels
        .iter()
        .map(|&l| label_to_size(l))
        .collect::<Result<_>>()?;

    if lhs.labels == rhs.labels {
        // Same labels: just Mul
        let outputs = builder.add_operation(
            StdTensorOp::Mul,
            vec![lhs.val.clone(), rhs.val.clone()],
            OperationRole::Primary,
        );
        return Ok(LabeledVal {
            val: ValueRef::Local(outputs[0]),
            labels: lhs.labels.clone(),
            shape: lhs.shape.clone(),
        });
    }

    // Broadcast both to combined shape, then Mul
    let lhs_dims: Vec<usize> = lhs
        .labels
        .iter()
        .map(|l| find_label_axis(&combined_labels, *l))
        .collect::<Result<_>>()?;
    let rhs_dims: Vec<usize> = rhs
        .labels
        .iter()
        .map(|l| find_label_axis(&combined_labels, *l))
        .collect::<Result<_>>()?;

    let lhs_bc = builder.add_operation(
        StdTensorOp::BroadcastInDim {
            shape: DimExpr::from_concrete(&combined_shape),
            dims: lhs_dims,
        },
        vec![lhs.val.clone()],
        OperationRole::Primary,
    );
    let rhs_bc = builder.add_operation(
        StdTensorOp::BroadcastInDim {
            shape: DimExpr::from_concrete(&combined_shape),
            dims: rhs_dims,
        },
        vec![rhs.val.clone()],
        OperationRole::Primary,
    );
    let outputs = builder.add_operation(
        StdTensorOp::Mul,
        vec![ValueRef::Local(lhs_bc[0]), ValueRef::Local(rhs_bc[0])],
        OperationRole::Primary,
    );
    Ok(LabeledVal {
        val: ValueRef::Local(outputs[0]),
        labels: combined_labels,
        shape: combined_shape,
    })
}

/// Lower a planned einsum contraction tree into a compute graph graph.
///
/// # Errors
///
/// Returns an error if the supplied tree, input values, or input shapes are
/// internally inconsistent.
pub(crate) fn build_einsum_graph(
    builder: &mut GraphBuilder<StdTensorOp>,
    tree: &ContractionTree,
    input_vals: &[ValueRef<StdTensorOp>],
    input_shapes: &[Vec<usize>],
) -> Result<ValueRef<StdTensorOp>> {
    let subscripts = &tree.subscripts;
    let input_count = subscripts.inputs.len();
    if input_count != input_vals.len() {
        return Err(builder_invalid_argument(format!(
            "number of subscripts inputs ({input_count}) must match number of input values ({})",
            input_vals.len()
        )));
    }
    if input_vals.len() != input_shapes.len() {
        return Err(builder_invalid_argument(format!(
            "number of input values ({}) must match number of input shapes ({})",
            input_vals.len(),
            input_shapes.len()
        )));
    }

    let output_labels = &subscripts.output;

    let mut labeled: Vec<LabeledVal> = input_vals
        .iter()
        .zip(subscripts.inputs.iter())
        .zip(input_shapes.iter())
        .map(|((val, labels), shape)| {
            if labels.len() != shape.len() {
                return Err(builder_invalid_argument(format!(
                    "labels length ({}) must match shape rank ({})",
                    labels.len(),
                    shape.len()
                )));
            }
            Ok(LabeledVal {
                val: val.clone(),
                labels: labels.clone(),
                shape: shape.clone(),
            })
        })
        .collect::<Result<_>>()?;

    // Diagonalize repeated indices in each input
    for lv in &mut labeled {
        *lv = diagonalize_repeated(builder, lv);
    }

    if input_count == 1 || tree.step_count() == 0 {
        // Unary: reduce, embed, and reorder
        let lv = &labeled[0];
        let output_set: HashSet<u32> = output_labels.iter().copied().collect();
        let reduce_labels: HashSet<u32> = lv
            .labels
            .iter()
            .filter(|l| !output_set.contains(l))
            .copied()
            .collect();
        let result = reduce_val(builder, lv, &reduce_labels);

        // Embed diagonal axes if output needs higher multiplicity
        let result = embed_repeated(builder, &result, output_labels)?;

        // Reorder if needed
        if result.labels == *output_labels {
            return Ok(result.val);
        }
        let perm: Vec<usize> = output_labels
            .iter()
            .map(|l| find_label_axis(&result.labels, *l))
            .collect::<Result<_>>()?;
        if perm.iter().enumerate().all(|(i, &p)| i == p) {
            return Ok(result.val);
        }
        let outputs = builder.add_operation(
            StdTensorOp::Transpose { perm },
            vec![result.val],
            OperationRole::Primary,
        );
        return Ok(ValueRef::Local(outputs[0]));
    }

    // N >= 2: use contraction tree from v1
    // Operand indices: 0..input_count are originals, input_count+step_idx are intermediates
    for step_idx in 0..tree.step_count() {
        let (left, right) = tree.step_pair(step_idx).ok_or_else(|| {
            builder_invalid_argument(format!("missing contraction pair for step {step_idx}"))
        })?;
        // Use the step's intermediate output subscripts so that labels needed
        // by later contractions are preserved (not pre-reduced away).
        let (_, _, step_out_labels) = tree.step_subscripts(step_idx).ok_or_else(|| {
            builder_invalid_argument(format!(
                "missing contraction subscripts for step {step_idx}"
            ))
        })?;
        let is_last = step_idx + 1 == tree.step_count();
        let result = binary_contract(
            builder,
            labeled_operand(&labeled, left, "left")?,
            labeled_operand(&labeled, right, "right")?,
            step_out_labels,
            is_last,
        )?;
        // Push intermediate as new entry in labeled
        labeled.push(result);
    }

    // The final result is the last intermediate: labeled[input_count + step_count - 1]
    let final_idx = input_count + tree.step_count() - 1;
    let result = labeled_operand(&labeled, final_idx, "final result")?;

    // Final reduction if result has labels not in output
    let output_set: HashSet<u32> = output_labels.iter().copied().collect();
    let extra_labels: HashSet<u32> = result
        .labels
        .iter()
        .filter(|l| !output_set.contains(l))
        .copied()
        .collect();
    let result = reduce_val(builder, result, &extra_labels);

    // Final reorder if needed
    if result.labels == *output_labels {
        return Ok(result.val);
    }

    if result.labels.is_empty() && output_labels.is_empty() {
        return Ok(result.val);
    }

    let perm: Vec<usize> = output_labels
        .iter()
        .map(|l| find_label_axis(&result.labels, *l))
        .collect::<Result<_>>()?;
    if perm.iter().enumerate().all(|(i, &p)| i == p) {
        return Ok(result.val);
    }
    let outputs = builder.add_operation(
        StdTensorOp::Transpose { perm },
        vec![result.val.clone()],
        OperationRole::Primary,
    );
    Ok(ValueRef::Local(outputs[0]))
}
