use std::collections::{HashMap, HashSet};

use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_einsum::Subscripts;
use tenferro_tensor::Tensor;

pub(crate) fn col_major_flat_index(shape: &[usize], idx: &[usize]) -> usize {
    let mut flat = 0;
    let mut stride = 1;
    for (d, &i) in idx.iter().enumerate() {
        flat += i * stride;
        stride *= shape[d];
    }
    flat
}

pub(crate) fn contracted_modes(subs: &Subscripts) -> Vec<u32> {
    let all_input_labels: HashSet<u32> = subs
        .inputs
        .iter()
        .flat_map(|inp| inp.iter())
        .copied()
        .collect();
    all_input_labels
        .into_iter()
        .filter(|m| !subs.output.contains(m))
        .collect()
}

pub(crate) fn dims_for_modes<T: Scalar>(
    operands: &[&Tensor<T>],
    subs: &Subscripts,
    modes: &[u32],
    missing_label: &str,
) -> Result<Vec<usize>> {
    modes
        .iter()
        .map(|m| {
            for (op_idx, input_modes) in subs.inputs.iter().enumerate() {
                if let Some(pos) = input_modes.iter().position(|x| x == m) {
                    return Ok(operands[op_idx].dims()[pos]);
                }
            }
            Err(Error::InvalidArgument(format!(
                "{missing_label} mode {m} not found in inputs"
            )))
        })
        .collect()
}

pub(crate) fn validate_mode_dimensions<T: Scalar>(
    operands: &[&Tensor<T>],
    subs: &Subscripts,
    output_shape: &[usize],
    contracted: &[u32],
    contracted_dims: &[usize],
) -> Result<()> {
    let mut dim_map: HashMap<u32, usize> = HashMap::new();
    for (pos, &m) in subs.output.iter().enumerate() {
        dim_map.insert(m, output_shape[pos]);
    }
    for (pos, &m) in contracted.iter().enumerate() {
        dim_map.insert(m, contracted_dims[pos]);
    }
    for (op_idx, input_modes) in subs.inputs.iter().enumerate() {
        let op_dims = operands[op_idx].dims();
        for (pos, &m) in input_modes.iter().enumerate() {
            if let Some(&expected) = dim_map.get(&m) {
                if op_dims[pos] != expected {
                    return Err(Error::InvalidArgument(format!(
                        "dimension mismatch for mode {m}: operand {op_idx} has size {} but expected {expected}",
                        op_dims[pos]
                    )));
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_modes_in_scope(
    input_modes: &[u32],
    output_modes: &[u32],
    contracted: &[u32],
    context: &str,
) -> Result<()> {
    for &mode in input_modes {
        if !output_modes.contains(&mode) && !contracted.contains(&mode) {
            return Err(Error::InvalidArgument(format!(
                "mode {mode} in {context} not found in output or contracted modes"
            )));
        }
    }
    Ok(())
}

pub(crate) fn build_mode_values(
    output_modes: &[u32],
    out_idx: &[usize],
    contracted: &[u32],
    contracted_idx: &[usize],
) -> HashMap<u32, usize> {
    let mut mode_values = HashMap::new();
    for (pos, &m) in output_modes.iter().enumerate() {
        mode_values.insert(m, out_idx[pos]);
    }
    for (pos, &m) in contracted.iter().enumerate() {
        mode_values.insert(m, contracted_idx[pos]);
    }
    mode_values
}

pub(crate) fn operand_index_from_mode_values(
    input_modes: &[u32],
    mode_values: &HashMap<u32, usize>,
    context: &str,
) -> Result<Vec<usize>> {
    input_modes
        .iter()
        .map(|m| {
            mode_values.get(m).copied().ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "mode {m} missing from mode map while building {context}"
                ))
            })
        })
        .collect()
}
