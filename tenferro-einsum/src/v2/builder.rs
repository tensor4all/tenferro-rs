use std::collections::HashSet;

use computegraph::fragment::FragmentBuilder;
use computegraph::types::{OpMode, ValRef};
use computegraph::GraphOp;

use tenferro_ops::config::DotGeneralConfig;
use tenferro_ops::semiring_ops::SemiringOps;

use super::types::Subscripts;

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
        Op::reduce_sum(reduce_axes),
        vec![lv.val.clone()],
        OpMode::Primal,
    );
    LabeledVal {
        val: ValRef::Local(outputs[0]),
        labels: new_labels,
        shape: new_shape,
    }
}

fn binary_contract<Op: GraphOp + SemiringOps>(
    builder: &mut FragmentBuilder<Op>,
    lhs: &LabeledVal<Op>,
    rhs: &LabeledVal<Op>,
    output_labels: &[u32],
) -> LabeledVal<Op> {
    let output_set: HashSet<u32> = output_labels.iter().copied().collect();
    let rhs_label_set: HashSet<u32> = rhs.labels.iter().copied().collect();
    let lhs_label_set: HashSet<u32> = lhs.labels.iter().copied().collect();

    // Pre-reduce: labels in lhs only, not in rhs and not in output
    let lhs_reduce: HashSet<u32> = lhs
        .labels
        .iter()
        .filter(|l| !rhs_label_set.contains(l) && !output_set.contains(l))
        .copied()
        .collect();
    let rhs_reduce: HashSet<u32> = rhs
        .labels
        .iter()
        .filter(|l| !lhs_label_set.contains(l) && !output_set.contains(l))
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
            if output_set.contains(&l) {
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
        };

        // DotGeneral output order: batch + lhs_free + rhs_free
        let result_labels: Vec<u32> = batch_labels
            .iter()
            .chain(lhs_free_labels.iter())
            .chain(rhs_free_labels.iter())
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

    // Reorder to match output_labels if needed
    let current_labels = &result.labels;
    if current_labels.is_empty() {
        return result;
    }

    // Filter output_labels to those present in result (to handle final reduction later)
    let result_label_set: HashSet<u32> = current_labels.iter().copied().collect();
    let target_labels: Vec<u32> = output_labels
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
    // Combined label order: batch + lhs_free + rhs_free
    let combined_labels: Vec<u32> = batch_labels
        .iter()
        .chain(lhs_free_labels.iter())
        .chain(rhs_free_labels.iter())
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
        Op::broadcast_in_dim(combined_shape.clone(), lhs_dims),
        vec![lhs.val.clone()],
        OpMode::Primal,
    );
    let rhs_bc = builder.add_op(
        Op::broadcast_in_dim(combined_shape.clone(), rhs_dims),
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
    subscripts: &Subscripts,
    input_vals: &[ValRef<Op>],
    input_shapes: &[Vec<usize>],
) -> ValRef<Op> {
    assert_eq!(
        subscripts.n_inputs(),
        input_vals.len(),
        "number of subscripts inputs must match number of input values"
    );
    assert_eq!(
        input_vals.len(),
        input_shapes.len(),
        "number of input values must match number of input shapes"
    );

    let mut labeled: Vec<LabeledVal<Op>> = input_vals
        .iter()
        .zip(subscripts.input_labels.iter())
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

    let output_labels = &subscripts.output_labels;

    if labeled.len() == 1 {
        // Unary: just reduce and reorder
        let output_set: HashSet<u32> = output_labels.iter().copied().collect();
        let reduce_labels: HashSet<u32> = labeled[0]
            .labels
            .iter()
            .filter(|l| !output_set.contains(l))
            .copied()
            .collect();
        let result = reduce_val(builder, &labeled[0], &reduce_labels);

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

    // N >= 2: use greedy contraction path
    let path = optimize_contraction_path(subscripts, input_shapes);

    for (left, right) in &path {
        let l = *left;
        let r = *right;
        // Contract l and r
        let result = binary_contract(builder, &labeled[l], &labeled[r], output_labels);
        // Replace left with result, mark right as consumed (set to empty dummy)
        labeled[l] = result;
        labeled[r] = LabeledVal {
            val: ValRef::Local(usize::MAX),
            labels: vec![],
            shape: vec![],
        };
    }

    // Find the surviving labeled val (the one that's not consumed)
    let result = labeled
        .iter()
        .find(|lv| !lv.labels.is_empty() || output_labels.is_empty())
        .unwrap();

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

pub fn optimize_contraction_path(
    subscripts: &Subscripts,
    input_shapes: &[Vec<usize>],
) -> Vec<(usize, usize)> {
    let n = subscripts.n_inputs();
    if n <= 1 {
        return vec![];
    }
    if n == 2 {
        return vec![(0, 1)];
    }

    // Greedy: pick the pair with smallest intermediate tensor size
    let mut alive: Vec<bool> = vec![true; n];
    let mut labels: Vec<Vec<u32>> = subscripts.input_labels.clone();
    let mut shapes: Vec<Vec<usize>> = input_shapes.to_vec();
    let mut path = Vec::new();
    let output_set: HashSet<u32> = subscripts.output_labels.iter().copied().collect();

    for _ in 0..(n - 1) {
        let mut best_pair = (0, 0);
        let mut best_cost = usize::MAX;
        let alive_indices: Vec<usize> = (0..n).filter(|&i| alive[i]).collect();

        for i in 0..alive_indices.len() {
            for j in (i + 1)..alive_indices.len() {
                let li = alive_indices[i];
                let lj = alive_indices[j];
                let cost = intermediate_size(
                    &labels[li],
                    &shapes[li],
                    &labels[lj],
                    &shapes[lj],
                    &output_set,
                );
                if cost < best_cost {
                    best_cost = cost;
                    best_pair = (li, lj);
                }
            }
        }

        let (l, r) = best_pair;
        path.push((l, r));

        // Update labels/shapes for l, mark r as consumed
        let mut new_labels = Vec::new();
        let combined: HashSet<u32> = labels[l].iter().chain(labels[r].iter()).copied().collect();
        // Result labels: labels that appear in output or in other alive operands
        let other_labels: HashSet<u32> = alive_indices
            .iter()
            .filter(|&&i| i != l && i != r)
            .flat_map(|&i| labels[i].iter().copied())
            .collect();

        for &label in &labels[l] {
            if output_set.contains(&label) || other_labels.contains(&label) {
                if !new_labels.contains(&label) {
                    new_labels.push(label);
                }
            }
        }
        for &label in &labels[r] {
            if output_set.contains(&label) || other_labels.contains(&label) {
                if !new_labels.contains(&label) {
                    new_labels.push(label);
                }
            }
        }

        let label_to_size = |l: u32| -> usize {
            for (&lab, &sz) in labels[best_pair.0].iter().zip(shapes[best_pair.0].iter()) {
                if lab == l {
                    return sz;
                }
            }
            for (&lab, &sz) in labels[best_pair.1].iter().zip(shapes[best_pair.1].iter()) {
                if lab == l {
                    return sz;
                }
            }
            // Check all alive
            for &i in &alive_indices {
                for (&lab, &sz) in labels[i].iter().zip(shapes[i].iter()) {
                    if lab == l {
                        return sz;
                    }
                }
            }
            panic!("label {} not found", l);
        };

        let new_shape: Vec<usize> = new_labels.iter().map(|&l| label_to_size(l)).collect();

        let _ = combined;
        labels[l] = new_labels;
        shapes[l] = new_shape;
        alive[r] = false;
    }

    path
}

fn intermediate_size(
    lhs_labels: &[u32],
    lhs_shape: &[usize],
    rhs_labels: &[u32],
    rhs_shape: &[usize],
    output_set: &HashSet<u32>,
) -> usize {
    // The intermediate result has all labels that are in output or in either operand
    // minus the contracting dims (in both but not output).
    let mut result_labels = Vec::new();
    for (i, &l) in lhs_labels.iter().enumerate() {
        if output_set.contains(&l) || !rhs_labels.contains(&l) {
            // Free or batch or output-only
            if !result_labels.contains(&(l, lhs_shape[i])) {
                result_labels.push((l, lhs_shape[i]));
            }
        }
        // contracting: in both, not in output -> excluded
    }
    for (i, &l) in rhs_labels.iter().enumerate() {
        if !lhs_labels.contains(&l) {
            if !result_labels.contains(&(l, rhs_shape[i])) {
                result_labels.push((l, rhs_shape[i]));
            }
        }
    }
    result_labels
        .iter()
        .map(|(_, s)| s)
        .product::<usize>()
        .max(1)
}
