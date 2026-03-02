// These imports are used by manual_einsum, which is intentionally dead code
// (reserved for future AD rules). Allow the lint to avoid noise.
#[allow(unused_imports)]
use std::collections::{HashMap, HashSet};

#[allow(unused_imports)]
use num_traits::{One, Zero};
use tenferro_algebra::Scalar;
use tenferro_device::Result;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::subscripts::Subscripts;
#[allow(unused_imports)]
use crate::util::{compute_output_shape, tensor_get, unflatten_index};

/// Execute a manual einsum without TensorPrims (for AD pullback).
/// Only supports 1-tensor and 2-tensor contractions.
#[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use tenferro_algebra::Standard;
    use tenferro_prims::{CpuBackend, CpuContext};

    use super::*;
    use crate::api::einsum_with_subscripts;

    fn tensor(data: &[f64], dims: &[usize]) -> Tensor<f64> {
        Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
    }

    #[test]
    fn manual_einsum_matches_matmul() {
        let mut ctx = CpuContext::new(1);
        let subs = Subscripts::parse("ij,jk->ik").unwrap();
        let a = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let size_dict = HashMap::from([
            ('i' as u32, 2usize),
            ('j' as u32, 2usize),
            ('k' as u32, 2usize),
        ]);

        let manual = manual_einsum(&subs, &[a.clone(), b.clone()], &size_dict).unwrap();
        let backend =
            einsum_with_subscripts::<Standard<f64>, CpuBackend>(&mut ctx, &subs, &[&a, &b], None)
                .unwrap();

        assert_eq!(manual.dims(), &[2, 2]);
        assert_eq!(manual.buffer().as_slice(), backend.buffer().as_slice());
    }

    #[test]
    fn manual_einsum_trace_returns_scalar() {
        let subs = Subscripts::parse("ii->").unwrap();
        let a = tensor(&[1.0, 0.0, 0.0, 3.0], &[2, 2]);
        let size_dict = HashMap::from([('i' as u32, 2usize)]);

        let result = manual_einsum(&subs, &[a], &size_dict).unwrap();
        let data = result.buffer().as_slice().unwrap();

        assert!(result.dims().is_empty());
        assert!((data[result.offset() as usize] - 4.0).abs() < 1e-10);
    }
}
