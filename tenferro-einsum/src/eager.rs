use std::collections::{HashMap, HashSet};

use tenferro_tensor::{DotGeneralConfig, Error, Result, Tensor, TensorBackend, TensorExec};

use crate::{ContractionTree, Subscripts};

const EAGER_EINSUM_OP: &str = "eager_einsum";

enum TensorValue<'a> {
    Borrowed(&'a Tensor),
    Owned(Tensor),
}

impl TensorValue<'_> {
    fn as_tensor(&self) -> &Tensor {
        match self {
            Self::Borrowed(tensor) => tensor,
            Self::Owned(tensor) => tensor,
        }
    }

    fn into_tensor(self) -> Tensor {
        match self {
            Self::Borrowed(tensor) => tensor.clone(),
            Self::Owned(tensor) => tensor,
        }
    }
}

struct LabeledTensor<'a> {
    tensor: TensorValue<'a>,
    labels: Vec<u32>,
}

impl LabeledTensor<'_> {
    fn tensor(&self) -> &Tensor {
        self.tensor.as_tensor()
    }

    fn shape(&self) -> &[usize] {
        self.tensor().shape()
    }
}

fn eager_invalid_config(message: impl Into<String>) -> Error {
    Error::InvalidConfig {
        op: EAGER_EINSUM_OP,
        message: message.into(),
    }
}

fn take_labeled<'a>(
    labeled: &mut [Option<LabeledTensor<'a>>],
    index: usize,
    role: &'static str,
) -> Result<LabeledTensor<'a>> {
    labeled
        .get_mut(index)
        .ok_or_else(|| eager_invalid_config(format!("missing {role} operand at index {index}")))?
        .take()
        .ok_or_else(|| eager_invalid_config(format!("missing {role} operand at index {index}")))
}

fn find_label_axis(labels: &[u32], label: u32) -> Result<usize> {
    labels
        .iter()
        .position(|candidate| *candidate == label)
        .ok_or_else(|| eager_invalid_config(format!("label {label} missing from tensor labels")))
}

fn label_size(label: u32, operands: &[&LabeledTensor<'_>]) -> Result<usize> {
    for operand in operands {
        if let Some(axis) = operand
            .labels
            .iter()
            .position(|candidate| *candidate == label)
        {
            return Ok(operand.shape()[axis]);
        }
    }
    Err(eager_invalid_config(format!(
        "label {label} missing from eager einsum operands"
    )))
}

fn reduce_tensor<'a>(
    exec: &mut dyn TensorExec,
    operand: LabeledTensor<'a>,
    reduce_labels: &HashSet<u32>,
) -> Result<LabeledTensor<'a>> {
    if reduce_labels.is_empty() {
        return Ok(operand);
    }

    let reduce_axes: Vec<usize> = operand
        .labels
        .iter()
        .enumerate()
        .filter(|(_, label)| reduce_labels.contains(label))
        .map(|(axis, _)| axis)
        .collect();
    if reduce_axes.is_empty() {
        return Ok(operand);
    }

    let reduce_set: HashSet<usize> = reduce_axes.iter().copied().collect();
    let labels: Vec<u32> = operand
        .labels
        .iter()
        .enumerate()
        .filter(|(axis, _)| !reduce_set.contains(axis))
        .map(|(_, label)| *label)
        .collect();
    let tensor = exec.reduce_sum(operand.tensor(), &reduce_axes)?;
    Ok(LabeledTensor {
        tensor: TensorValue::Owned(tensor),
        labels,
    })
}

fn diagonalize_repeated<'a>(
    exec: &mut dyn TensorExec,
    mut operand: LabeledTensor<'a>,
) -> Result<LabeledTensor<'a>> {
    loop {
        let mut seen = HashMap::new();
        let mut repeated_pair = None;
        for (axis, label) in operand.labels.iter().copied().enumerate() {
            if let Some(first_axis) = seen.insert(label, axis) {
                repeated_pair = Some((first_axis, axis));
                break;
            }
        }

        let Some((axis_a, axis_b)) = repeated_pair else {
            return Ok(operand);
        };

        let tensor = exec.extract_diagonal(operand.tensor(), axis_a, axis_b)?;
        let mut labels = operand.labels;
        labels.remove(axis_b);
        operand = LabeledTensor {
            tensor: TensorValue::Owned(tensor),
            labels,
        };
    }
}

fn embed_repeated<'a>(
    exec: &mut dyn TensorExec,
    mut operand: LabeledTensor<'a>,
    output_labels: &[u32],
) -> Result<LabeledTensor<'a>> {
    loop {
        let mut embedded = false;
        for &label in output_labels {
            let current_count = operand
                .labels
                .iter()
                .filter(|candidate| **candidate == label)
                .count();
            let output_count = output_labels
                .iter()
                .filter(|candidate| **candidate == label)
                .count();
            if output_count > current_count {
                let axis_a = find_label_axis(&operand.labels, label)?;
                let axis_b = axis_a + 1;
                let tensor = exec.embed_diagonal(operand.tensor(), axis_a, axis_b)?;
                let mut labels = operand.labels;
                labels.insert(axis_b, label);
                operand = LabeledTensor {
                    tensor: TensorValue::Owned(tensor),
                    labels,
                };
                embedded = true;
                break;
            }
        }

        if !embedded {
            return Ok(operand);
        }
    }
}

fn transpose_to_labels<'a>(
    exec: &mut dyn TensorExec,
    operand: LabeledTensor<'a>,
    target_labels: &[u32],
) -> Result<LabeledTensor<'a>> {
    if operand.labels == target_labels {
        return Ok(operand);
    }

    let perm: Vec<usize> = target_labels
        .iter()
        .map(|label| find_label_axis(&operand.labels, *label))
        .collect::<Result<_>>()?;
    if perm
        .iter()
        .enumerate()
        .all(|(axis, target)| axis == *target)
    {
        return Ok(operand);
    }

    let tensor = exec.transpose(operand.tensor(), &perm)?;
    Ok(LabeledTensor {
        tensor: TensorValue::Owned(tensor),
        labels: target_labels.to_vec(),
    })
}

fn outer_product<'a>(
    exec: &mut dyn TensorExec,
    lhs: LabeledTensor<'a>,
    rhs: LabeledTensor<'a>,
    batch_labels: &[u32],
    lhs_free_labels: &[u32],
    rhs_free_labels: &[u32],
) -> Result<LabeledTensor<'a>> {
    if lhs.labels == rhs.labels {
        let tensor = exec.mul(lhs.tensor(), rhs.tensor())?;
        return Ok(LabeledTensor {
            tensor: TensorValue::Owned(tensor),
            labels: lhs.labels,
        });
    }

    let combined_labels: Vec<u32> = lhs_free_labels
        .iter()
        .chain(rhs_free_labels.iter())
        .chain(batch_labels.iter())
        .copied()
        .collect();
    let combined_shape: Vec<usize> = combined_labels
        .iter()
        .map(|label| label_size(*label, &[&lhs, &rhs]))
        .collect::<Result<_>>()?;
    let lhs_dims: Vec<usize> = lhs
        .labels
        .iter()
        .map(|label| find_label_axis(&combined_labels, *label))
        .collect::<Result<_>>()?;
    let rhs_dims: Vec<usize> = rhs
        .labels
        .iter()
        .map(|label| find_label_axis(&combined_labels, *label))
        .collect::<Result<_>>()?;

    let lhs_tensor = exec.broadcast_in_dim(lhs.tensor(), &combined_shape, &lhs_dims)?;
    let rhs_tensor = exec.broadcast_in_dim(rhs.tensor(), &combined_shape, &rhs_dims)?;
    let tensor = exec.mul(&lhs_tensor, &rhs_tensor)?;
    Ok(LabeledTensor {
        tensor: TensorValue::Owned(tensor),
        labels: combined_labels,
    })
}

fn binary_contract<'a>(
    exec: &mut dyn TensorExec,
    lhs: LabeledTensor<'a>,
    rhs: LabeledTensor<'a>,
    survive_labels: &[u32],
    reorder_result: bool,
) -> Result<LabeledTensor<'a>> {
    let survive_set: HashSet<u32> = survive_labels.iter().copied().collect();
    let rhs_label_set: HashSet<u32> = rhs.labels.iter().copied().collect();
    let lhs_label_set: HashSet<u32> = lhs.labels.iter().copied().collect();

    let lhs_reduce: HashSet<u32> = lhs
        .labels
        .iter()
        .filter(|label| !rhs_label_set.contains(label) && !survive_set.contains(label))
        .copied()
        .collect();
    let rhs_reduce: HashSet<u32> = rhs
        .labels
        .iter()
        .filter(|label| !lhs_label_set.contains(label) && !survive_set.contains(label))
        .copied()
        .collect();

    let lhs = reduce_tensor(exec, lhs, &lhs_reduce)?;
    let rhs = reduce_tensor(exec, rhs, &rhs_reduce)?;

    let lhs_label_set: HashSet<u32> = lhs.labels.iter().copied().collect();
    let rhs_label_set: HashSet<u32> = rhs.labels.iter().copied().collect();

    let mut batch_labels = Vec::new();
    let mut contracting_labels = Vec::new();
    let mut lhs_free_labels = Vec::new();
    let mut rhs_free_labels = Vec::new();

    for &label in &lhs.labels {
        if rhs_label_set.contains(&label) {
            if survive_set.contains(&label) {
                if !batch_labels.contains(&label) {
                    batch_labels.push(label);
                }
            } else if !contracting_labels.contains(&label) {
                contracting_labels.push(label);
            }
        } else if !lhs_free_labels.contains(&label) {
            lhs_free_labels.push(label);
        }
    }

    for &label in &rhs.labels {
        if !lhs_label_set.contains(&label) && !rhs_free_labels.contains(&label) {
            rhs_free_labels.push(label);
        }
    }

    let result = if contracting_labels.is_empty() {
        outer_product(
            exec,
            lhs,
            rhs,
            &batch_labels,
            &lhs_free_labels,
            &rhs_free_labels,
        )?
    } else {
        let lhs_contracting_dims: Vec<usize> = contracting_labels
            .iter()
            .map(|label| find_label_axis(&lhs.labels, *label))
            .collect::<Result<_>>()?;
        let rhs_contracting_dims: Vec<usize> = contracting_labels
            .iter()
            .map(|label| find_label_axis(&rhs.labels, *label))
            .collect::<Result<_>>()?;
        let lhs_batch_dims: Vec<usize> = batch_labels
            .iter()
            .map(|label| find_label_axis(&lhs.labels, *label))
            .collect::<Result<_>>()?;
        let rhs_batch_dims: Vec<usize> = batch_labels
            .iter()
            .map(|label| find_label_axis(&rhs.labels, *label))
            .collect::<Result<_>>()?;
        let labels: Vec<u32> = lhs_free_labels
            .iter()
            .chain(rhs_free_labels.iter())
            .chain(batch_labels.iter())
            .copied()
            .collect();
        let config = DotGeneralConfig {
            lhs_contracting_dims,
            rhs_contracting_dims,
            lhs_batch_dims,
            rhs_batch_dims,
            lhs_rank: lhs.labels.len(),
            rhs_rank: rhs.labels.len(),
        };
        let tensor = exec.dot_general(lhs.tensor(), rhs.tensor(), &config)?;
        LabeledTensor {
            tensor: TensorValue::Owned(tensor),
            labels,
        }
    };

    if !reorder_result {
        return Ok(result);
    }

    let result_label_set: HashSet<u32> = result.labels.iter().copied().collect();
    let target_labels: Vec<u32> = survive_labels
        .iter()
        .filter(|label| result_label_set.contains(label))
        .copied()
        .collect();
    transpose_to_labels(exec, result, &target_labels)
}

fn eager_einsum_exec(
    exec: &mut dyn TensorExec,
    inputs: &[&Tensor],
    tree: &ContractionTree,
) -> Result<Tensor> {
    let subscripts = &tree.subscripts;
    let n_inputs = subscripts.inputs.len();
    let output_labels = &subscripts.output;

    let mut labeled: Vec<Option<LabeledTensor<'_>>> = inputs
        .iter()
        .zip(subscripts.inputs.iter())
        .map(|(tensor, labels)| {
            Some(LabeledTensor {
                tensor: TensorValue::Borrowed(tensor),
                labels: labels.clone(),
            })
        })
        .collect();

    for index in 0..labeled.len() {
        let operand = take_labeled(&mut labeled, index, "input")?;
        labeled[index] = Some(diagonalize_repeated(exec, operand)?);
    }

    if n_inputs == 1 || tree.step_count() == 0 {
        let operand = take_labeled(&mut labeled, 0, "input")?;
        let output_set: HashSet<u32> = output_labels.iter().copied().collect();
        let reduce_labels: HashSet<u32> = operand
            .labels
            .iter()
            .filter(|label| !output_set.contains(label))
            .copied()
            .collect();
        let reduced = reduce_tensor(exec, operand, &reduce_labels)?;
        let embedded = embed_repeated(exec, reduced, output_labels)?;
        let reordered = transpose_to_labels(exec, embedded, output_labels)?;
        return Ok(reordered.tensor.into_tensor());
    }

    for step_idx in 0..tree.step_count() {
        let (left, right) = tree.step_pair(step_idx).ok_or_else(|| {
            eager_invalid_config(format!("missing contraction pair for step {step_idx}"))
        })?;
        let (_, _, step_output_labels) = tree.step_subscripts(step_idx).ok_or_else(|| {
            eager_invalid_config(format!(
                "missing contraction subscripts for step {step_idx}"
            ))
        })?;
        let lhs = take_labeled(&mut labeled, left, "lhs")?;
        let rhs = take_labeled(&mut labeled, right, "rhs")?;
        let result = binary_contract(
            exec,
            lhs,
            rhs,
            step_output_labels,
            step_idx + 1 == tree.step_count(),
        )?;
        labeled.push(Some(result));
    }

    let final_index = n_inputs + tree.step_count() - 1;
    let result = take_labeled(&mut labeled, final_index, "result")?;
    let output_set: HashSet<u32> = output_labels.iter().copied().collect();
    let extra_labels: HashSet<u32> = result
        .labels
        .iter()
        .filter(|label| !output_set.contains(label))
        .copied()
        .collect();
    let reduced = reduce_tensor(exec, result, &extra_labels)?;
    let reordered = transpose_to_labels(exec, reduced, output_labels)?;
    Ok(reordered.tensor.into_tensor())
}

/// Eager N-ary einsum on concrete [`Tensor`] values.
///
/// This applies the same contraction-tree optimization strategy used by the
/// traced einsum path, but executes each contraction immediately against the
/// provided backend context.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::eager_einsum;
/// use tenferro_tensor::{Tensor, TensorBackend, cpu::CpuBackend};
///
/// let mut ctx = CpuBackend::new();
/// let a = Tensor::new(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let b = Tensor::new(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let c = eager_einsum(&mut ctx, &[&a, &b], "ij,jk->ik").unwrap();
///
/// assert_eq!(c.shape(), &[2, 2]);
/// assert_eq!(c.as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
/// ```
pub fn eager_einsum(
    ctx: &mut impl TensorBackend,
    inputs: &[&Tensor],
    subscripts: &str,
) -> Result<Tensor> {
    if inputs.is_empty() {
        return Err(eager_invalid_config(
            "eager einsum requires at least one input tensor",
        ));
    }

    let subs = Subscripts::parse(subscripts)
        .map_err(|err| eager_invalid_config(format!("invalid subscripts: {err}")))?;
    if subs.inputs.len() != inputs.len() {
        return Err(eager_invalid_config(format!(
            "eager einsum subscripts expect {} inputs, got {}",
            subs.inputs.len(),
            inputs.len()
        )));
    }

    let shapes: Vec<&[usize]> = inputs.iter().map(|tensor| tensor.shape()).collect();
    let tree = ContractionTree::optimize(&subs, &shapes).map_err(|err| {
        eager_invalid_config(format!("failed to optimize contraction tree: {err}"))
    })?;
    ctx.with_exec_session(|exec| eager_einsum_exec(exec, inputs, &tree))
}
