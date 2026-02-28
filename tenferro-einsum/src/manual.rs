use std::collections::{HashMap, HashSet};

use num_traits::{One, Zero};
use tenferro_algebra::Scalar;
use tenferro_device::Result;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::subscripts::Subscripts;
use crate::util::{compute_output_shape, tensor_get, unflatten_index};

/// Execute a manual einsum without TensorPrims (for AD pullback).
/// Only supports 1-tensor and 2-tensor contractions.
pub(crate) fn manual_einsum<T: Scalar>(
    subs: &Subscripts,
    operands: &[Tensor<T>],
    size_dict: &HashMap<u32, usize>,
) -> Result<Tensor<T>> {
    let output_shape = compute_output_shape(&subs.output, size_dict)?;
    let n_output: usize = output_shape.iter().product();

    // Collect all unique labels
    let mut all_labels: Vec<u32> = Vec::new();
    let mut all_label_set = HashSet::new();
    for input_subs in &subs.inputs {
        for &l in input_subs {
            if all_label_set.insert(l) {
                all_labels.push(l);
            }
        }
    }
    for &l in &subs.output {
        if all_label_set.insert(l) {
            all_labels.push(l);
        }
    }

    // Build label → size mapping
    let all_dims: Vec<usize> = all_labels
        .iter()
        .map(|l| size_dict.get(l).copied().unwrap_or(1))
        .collect();
    let n_total: usize = all_dims.iter().product();

    // Build label → position in all_labels
    let label_to_pos: HashMap<u32, usize> = all_labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i))
        .collect();

    // Allocate output
    let strides = strided_view::col_major_strides(&output_shape);
    let mut out_data = vec![T::zero(); n_output];

    // Iterate over all index combinations
    for flat in 0..n_total.max(1) {
        let idx = unflatten_index(flat, &all_dims);

        // Compute output position
        let out_idx: Vec<usize> = subs.output.iter().map(|l| idx[label_to_pos[l]]).collect();

        // Compute product of all input elements
        let mut product = T::one();
        for (op_idx, input_subs) in subs.inputs.iter().enumerate() {
            let in_idx: Vec<usize> = input_subs.iter().map(|l| idx[label_to_pos[l]]).collect();
            product = product * tensor_get(&operands[op_idx], &in_idx);
        }

        // Accumulate into output
        let out_pos: isize = out_idx
            .iter()
            .zip(strides.iter())
            .map(|(&i, &s)| i as isize * s)
            .sum();
        if !out_idx.is_empty() {
            out_data[out_pos as usize] = out_data[out_pos as usize] + product;
        } else {
            out_data[0] = out_data[0] + product;
        }
    }

    Tensor::from_slice(&out_data, &output_shape, MemoryOrder::ColumnMajor)
}
