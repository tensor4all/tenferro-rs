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
//   - SemiringFastPath: optional fused patterns (contract, elementwise_mul/add).
//     Trait exists but no implementation.

use std::collections::{HashMap, HashSet};

use computegraph::fragment::FragmentBuilder;
use computegraph::types::{OpMode, ValRef};
use computegraph::GraphOp;

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::semiring_ops::SemiringOps;
use tenferro_tensor::DotGeneralConfig;

use crate::planning::tree::ContractionTree;

#[derive(Clone, Debug)]
struct LabeledVal<Op: GraphOp> {
    val: ValRef<Op>,
    labels: Vec<u32>,
    shape: Vec<usize>,
}

fn label_size_map(labels: &[u32], shape: &[usize]) -> Vec<(u32, usize)> {
    labels.iter().copied().zip(shape.iter().copied()).collect()
}

fn reduce_val<Op: GraphOp + SemiringOps>(
    builder: &mut FragmentBuilder<Op>,
    lv: &LabeledVal<Op>,
    reduce_labels: &HashSet<u32>,
) -> LabeledVal<Op> {
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
    let outputs = builder.add_op(
        Op::reduce_sum(reduce_axes, DimExpr::from_concrete(&lv.shape)),
        vec![lv.val.clone()],
        OpMode::Primal,
    );
    LabeledVal {
        val: ValRef::Local(outputs[0]),
        labels: new_labels,
        shape: new_shape,
    }
}

/// Embed diagonal axes when the output requires higher multiplicity of a label
/// than the current tensor has. For example, "i->ii" needs to embed axis 0
/// into a new axis 1 of the same size.
fn embed_repeated<Op: GraphOp + SemiringOps>(
    builder: &mut FragmentBuilder<Op>,
    lv: &LabeledVal<Op>,
    output_labels: &[u32],
) -> LabeledVal<Op> {
    // Count how many times each label appears in output vs current labels.
    let mut result = lv.clone();
    for &label in output_labels {
        let current_count = result.labels.iter().filter(|&&l| l == label).count();
        let output_count = output_labels.iter().filter(|&&l| l == label).count();
        if output_count > current_count {
            // Need to embed: find the existing axis with this label and
            // insert a duplicate axis after it.
            let axis_a = result
                .labels
                .iter()
                .position(|&l| l == label)
                .expect("label must exist in current tensor for embedding");
            // Insert the new axis right after axis_a.
            let axis_b = axis_a + 1;
            let n = result.shape[axis_a];
            let outputs = builder.add_op(
                Op::embed_diag(axis_a, axis_b),
                vec![result.val.clone()],
                OpMode::Primal,
            );
            let mut new_labels = result.labels.clone();
            new_labels.insert(axis_b, label);
            let mut new_shape = result.shape.clone();
            new_shape.insert(axis_b, n);
            result = LabeledVal {
                val: ValRef::Local(outputs[0]),
                labels: new_labels,
                shape: new_shape,
            };
            // Recurse to handle cases like "i->iii" (multiple embeddings).
            return embed_repeated(builder, &result, output_labels);
        }
    }
    result
}

fn diagonalize_repeated<Op: GraphOp + SemiringOps>(
    builder: &mut FragmentBuilder<Op>,
    lv: &LabeledVal<Op>,
) -> LabeledVal<Op> {
    let mut seen: HashMap<u32, usize> = HashMap::new();
    for (i, &label) in lv.labels.iter().enumerate() {
        if let Some(&first) = seen.get(&label) {
            // Found repeated label at axes `first` and `i`
            let outputs = builder.add_op(
                Op::extract_diag(first, i),
                vec![lv.val.clone()],
                OpMode::Primal,
            );
            let mut new_labels = lv.labels.clone();
            new_labels.remove(i);
            let mut new_shape = lv.shape.clone();
            new_shape.remove(i);
            let result = LabeledVal {
                val: ValRef::Local(outputs[0]),
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

fn binary_contract<Op: GraphOp + SemiringOps>(
    builder: &mut FragmentBuilder<Op>,
    lhs: &LabeledVal<Op>,
    rhs: &LabeledVal<Op>,
    survive_labels: &[u32],
    reorder_result: bool,
) -> LabeledVal<Op> {
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

    let label_to_size = |l: u32| -> usize {
        for &(label, size) in &lhs_sizes {
            if label == l {
                return size;
            }
        }
        for &(label, size) in &rhs_sizes {
            if label == l {
                return size;
            }
        }
        panic!("label {} not found in any operand", l);
    };

    let result = if !contracting_labels.is_empty() {
        // Use DotGeneral
        let lhs_contracting_dims: Vec<usize> = contracting_labels
            .iter()
            .map(|l| lhs.labels.iter().position(|x| x == l).unwrap())
            .collect();
        let rhs_contracting_dims: Vec<usize> = contracting_labels
            .iter()
            .map(|l| rhs.labels.iter().position(|x| x == l).unwrap())
            .collect();
        let lhs_batch_dims: Vec<usize> = batch_labels
            .iter()
            .map(|l| lhs.labels.iter().position(|x| x == l).unwrap())
            .collect();
        let rhs_batch_dims: Vec<usize> = batch_labels
            .iter()
            .map(|l| rhs.labels.iter().position(|x| x == l).unwrap())
            .collect();

        let config = DotGeneralConfig {
            lhs_contracting_dims,
            rhs_contracting_dims,
            lhs_batch_dims,
            rhs_batch_dims,
            lhs_rank: lhs.shape.len(),
            rhs_rank: rhs.shape.len(),
        };

        // DotGeneral output order: lhs_free + rhs_free + batch (col-major batch trailing)
        let result_labels: Vec<u32> = lhs_free_labels
            .iter()
            .chain(rhs_free_labels.iter())
            .chain(batch_labels.iter())
            .copied()
            .collect();
        let result_shape: Vec<usize> = result_labels.iter().map(|&l| label_to_size(l)).collect();

        let outputs = builder.add_op(
            Op::dot_general(config),
            vec![lhs.val.clone(), rhs.val.clone()],
            OpMode::Primal,
        );

        LabeledVal {
            val: ValRef::Local(outputs[0]),
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
        )
    };

    if !reorder_result {
        return result;
    }

    // Reorder to match the caller-visible order if needed.
    let current_labels = &result.labels;
    if current_labels.is_empty() {
        return result;
    }

    // Filter survivor labels to those present in result (to handle final reduction later)
    let result_label_set: HashSet<u32> = current_labels.iter().copied().collect();
    let target_labels: Vec<u32> = survive_labels
        .iter()
        .filter(|l| result_label_set.contains(l))
        .copied()
        .collect();

    if current_labels.len() == target_labels.len() && *current_labels == target_labels {
        return result;
    }

    // Build permutation
    let perm: Vec<usize> = target_labels
        .iter()
        .map(|l| current_labels.iter().position(|x| x == l).unwrap())
        .collect();

    if perm.iter().enumerate().all(|(i, &p)| i == p) {
        return result;
    }

    let new_shape: Vec<usize> = perm.iter().map(|&p| result.shape[p]).collect();
    let outputs = builder.add_op(
        Op::transpose_op(perm),
        vec![result.val.clone()],
        OpMode::Primal,
    );

    LabeledVal {
        val: ValRef::Local(outputs[0]),
        labels: target_labels,
        shape: new_shape,
    }
}

fn outer_product<Op: GraphOp + SemiringOps>(
    builder: &mut FragmentBuilder<Op>,
    lhs: &LabeledVal<Op>,
    rhs: &LabeledVal<Op>,
    batch_labels: &[u32],
    lhs_free_labels: &[u32],
    rhs_free_labels: &[u32],
    label_to_size: &dyn Fn(u32) -> usize,
) -> LabeledVal<Op> {
    let combined_labels: Vec<u32> = lhs_free_labels
        .iter()
        .chain(rhs_free_labels.iter())
        .chain(batch_labels.iter())
        .copied()
        .collect();
    let combined_shape: Vec<usize> = combined_labels.iter().map(|&l| label_to_size(l)).collect();

    if lhs.labels == rhs.labels {
        // Same labels: just Mul
        let outputs = builder.add_op(
            Op::mul_op(),
            vec![lhs.val.clone(), rhs.val.clone()],
            OpMode::Primal,
        );
        return LabeledVal {
            val: ValRef::Local(outputs[0]),
            labels: lhs.labels.clone(),
            shape: lhs.shape.clone(),
        };
    }

    // Broadcast both to combined shape, then Mul
    let lhs_dims: Vec<usize> = lhs
        .labels
        .iter()
        .map(|l| combined_labels.iter().position(|x| x == l).unwrap())
        .collect();
    let rhs_dims: Vec<usize> = rhs
        .labels
        .iter()
        .map(|l| combined_labels.iter().position(|x| x == l).unwrap())
        .collect();

    let lhs_bc = builder.add_op(
        Op::broadcast_in_dim(DimExpr::from_concrete(&combined_shape), lhs_dims),
        vec![lhs.val.clone()],
        OpMode::Primal,
    );
    let rhs_bc = builder.add_op(
        Op::broadcast_in_dim(DimExpr::from_concrete(&combined_shape), rhs_dims),
        vec![rhs.val.clone()],
        OpMode::Primal,
    );
    let outputs = builder.add_op(
        Op::mul_op(),
        vec![ValRef::Local(lhs_bc[0]), ValRef::Local(rhs_bc[0])],
        OpMode::Primal,
    );
    LabeledVal {
        val: ValRef::Local(outputs[0]),
        labels: combined_labels,
        shape: combined_shape,
    }
}

pub fn build_einsum_fragment<Op: GraphOp + SemiringOps>(
    builder: &mut FragmentBuilder<Op>,
    tree: &ContractionTree,
    input_vals: &[ValRef<Op>],
    input_shapes: &[Vec<usize>],
) -> ValRef<Op> {
    let subscripts = &tree.subscripts;
    let n_inputs = subscripts.inputs.len();
    assert_eq!(
        n_inputs,
        input_vals.len(),
        "number of subscripts inputs must match number of input values"
    );
    assert_eq!(
        input_vals.len(),
        input_shapes.len(),
        "number of input values must match number of input shapes"
    );

    let output_labels = &subscripts.output;

    let mut labeled: Vec<LabeledVal<Op>> = input_vals
        .iter()
        .zip(subscripts.inputs.iter())
        .zip(input_shapes.iter())
        .map(|((val, labels), shape)| {
            assert_eq!(
                labels.len(),
                shape.len(),
                "labels length must match shape rank"
            );
            LabeledVal {
                val: val.clone(),
                labels: labels.clone(),
                shape: shape.clone(),
            }
        })
        .collect();

    // Diagonalize repeated indices in each input
    for lv in &mut labeled {
        *lv = diagonalize_repeated(builder, lv);
    }

    if n_inputs == 1 || tree.step_count() == 0 {
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
        let result = embed_repeated(builder, &result, output_labels);

        // Reorder if needed
        if result.labels == *output_labels {
            return result.val;
        }
        let perm: Vec<usize> = output_labels
            .iter()
            .map(|l| result.labels.iter().position(|x| x == l).unwrap())
            .collect();
        if perm.iter().enumerate().all(|(i, &p)| i == p) {
            return result.val;
        }
        let outputs = builder.add_op(Op::transpose_op(perm), vec![result.val], OpMode::Primal);
        return ValRef::Local(outputs[0]);
    }

    // N >= 2: use contraction tree from v1
    // Operand indices: 0..n_inputs are originals, n_inputs+step_idx are intermediates
    for step_idx in 0..tree.step_count() {
        let (left, right) = tree.step_pair(step_idx).unwrap();
        // Use the step's intermediate output subscripts so that labels needed
        // by later contractions are preserved (not pre-reduced away).
        let (_, _, step_out_labels) = tree.step_subscripts(step_idx).unwrap();
        let is_last = step_idx + 1 == tree.step_count();
        let result = binary_contract(
            builder,
            &labeled[left],
            &labeled[right],
            step_out_labels,
            is_last,
        );
        // Push intermediate as new entry in labeled
        labeled.push(result);
    }

    // The final result is the last intermediate: labeled[n_inputs + step_count - 1]
    let final_idx = n_inputs + tree.step_count() - 1;
    let result = &labeled[final_idx];

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
        return result.val;
    }

    if result.labels.is_empty() && output_labels.is_empty() {
        return result.val;
    }

    let perm: Vec<usize> = output_labels
        .iter()
        .map(|l| result.labels.iter().position(|x| x == l).unwrap())
        .collect();
    if perm.iter().enumerate().all(|(i, &p)| i == p) {
        return result.val;
    }
    let outputs = builder.add_op(
        Op::transpose_op(perm),
        vec![result.val.clone()],
        OpMode::Primal,
    );
    ValRef::Local(outputs[0])
}
