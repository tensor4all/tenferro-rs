use std::collections::{HashMap, HashSet};

use tenferro_algebra::Scalar;
use tenferro_device::{LogicalMemorySpace, Result};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::syntax::subscripts::Subscripts;

pub(super) fn make_delta<T: Scalar>(n: usize, _space: LogicalMemorySpace) -> Result<Tensor<T>> {
    let mut data = vec![T::zero(); n * n];
    for i in 0..n {
        data[i * n + i] = T::one();
    }
    Tensor::from_slice(&data, &[n, n], MemoryOrder::ColumnMajor)
}

pub(super) struct DeltaContext<T> {
    pub delta_tensors: Vec<Tensor<T>>,
    pub base_subs: Subscripts,
    pub embed_subs: Option<Subscripts>,
}

pub(super) fn build_delta_context<T: Scalar>(
    subs: &Subscripts,
    rev_inputs_subs: &[Vec<u32>],
    rev_output: &[u32],
    size_dict: &HashMap<u32, usize>,
    space: LogicalMemorySpace,
) -> Result<DeltaContext<T>> {
    let all_input_labels: HashSet<u32> = rev_inputs_subs
        .iter()
        .flat_map(|labels| labels.iter().copied())
        .collect();

    let unique_output: Vec<u32> = {
        let mut seen = HashSet::new();
        rev_output
            .iter()
            .filter(|l| seen.insert(**l))
            .copied()
            .collect()
    };

    let mut delta_labels: Vec<u32> = Vec::new();
    let mut delta_tensors: Vec<Tensor<T>> = Vec::new();
    for &label in &unique_output {
        if !all_input_labels.contains(&label) {
            let dim = *size_dict.get(&label).ok_or_else(|| {
                tenferro_device::Error::InvalidArgument(format!(
                    "einsum: missing dimension for label {}",
                    label
                ))
            })?;
            delta_tensors.push(make_delta::<T>(dim, space)?);
            delta_labels.push(label);
        }
    }

    let mut base_inputs: Vec<Vec<u32>> = rev_inputs_subs.to_vec();
    for &label in &delta_labels {
        let max_label = subs
            .inputs
            .iter()
            .flat_map(|v| v.iter())
            .chain(subs.output.iter())
            .copied()
            .max()
            .unwrap_or(0);
        let fresh = max_label + 1 + (base_inputs.len() as u32);
        base_inputs.push(vec![label, fresh]);
    }

    let base_subs = Subscripts {
        inputs: base_inputs,
        output: unique_output.clone(),
    };

    let needs_embedding = unique_output != rev_output;
    let embed_subs = if needs_embedding {
        Some(Subscripts {
            inputs: vec![unique_output],
            output: rev_output.to_vec(),
        })
    } else {
        None
    };

    Ok(DeltaContext {
        delta_tensors,
        base_subs,
        embed_subs,
    })
}
