use std::collections::HashMap;

use tenferro_device::{Error, Result as DeviceResult};

use crate::syntax::subscripts::Subscripts;
use crate::util::{build_size_dict, compute_output_shape};

/// Narrow lowering plan for the bridge-equivalent binary GEMM path.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct StrictBinaryLoweringPlan {
    pub(crate) size_dict: HashMap<u32, usize>,
    pub(crate) output_shape: Vec<usize>,
    pub(crate) lhs_free_labels: Vec<u32>,
    pub(crate) rhs_free_labels: Vec<u32>,
    pub(crate) contract_labels: Vec<u32>,
    pub(crate) lhs_perm: Vec<usize>,
    pub(crate) rhs_perm: Vec<usize>,
    pub(crate) lhs_matrix_dims: Vec<usize>,
    pub(crate) rhs_matrix_dims: Vec<usize>,
    pub(crate) canonical_output_dims: Vec<usize>,
    pub(crate) output_perm: Vec<usize>,
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
}

fn product(dims: &[usize]) -> usize {
    dims.iter().product::<usize>().max(1)
}

fn positions_if_unique(labels: &[u32]) -> Option<HashMap<u32, usize>> {
    let mut positions = HashMap::with_capacity(labels.len());
    for (axis, &label) in labels.iter().enumerate() {
        if positions.insert(label, axis).is_some() {
            return None;
        }
    }
    Some(positions)
}

fn compile_strict_binary_lowering_plan_from_parts(
    lhs_labels: &[u32],
    rhs_labels: &[u32],
    out_labels: &[u32],
    lhs_dims: &[usize],
    rhs_dims: &[usize],
    size_dict: &HashMap<u32, usize>,
) -> DeviceResult<Option<StrictBinaryLoweringPlan>> {
    let output_shape = compute_output_shape(out_labels, size_dict)?;

    let Some(lhs_positions) = positions_if_unique(lhs_labels) else {
        return Ok(None);
    };
    let Some(rhs_positions) = positions_if_unique(rhs_labels) else {
        return Ok(None);
    };
    if positions_if_unique(out_labels).is_none() {
        return Ok(None);
    }

    let lhs_free_labels: Vec<u32> = lhs_labels
        .iter()
        .copied()
        .filter(|label| !rhs_positions.contains_key(label))
        .collect();
    let rhs_free_labels: Vec<u32> = rhs_labels
        .iter()
        .copied()
        .filter(|label| !lhs_positions.contains_key(label))
        .collect();
    let contract_labels: Vec<u32> = lhs_labels
        .iter()
        .copied()
        .filter(|label| rhs_positions.contains_key(label))
        .collect();

    if contract_labels
        .iter()
        .any(|label| out_labels.contains(label))
    {
        return Ok(None);
    }

    let canonical_output_labels: Vec<u32> = lhs_free_labels
        .iter()
        .chain(rhs_free_labels.iter())
        .copied()
        .collect();
    if out_labels.len() != canonical_output_labels.len() {
        return Ok(None);
    }
    if !out_labels
        .iter()
        .all(|label| canonical_output_labels.contains(label))
    {
        return Ok(None);
    }

    let lhs_free_axes: Vec<usize> = lhs_free_labels
        .iter()
        .map(|label| lhs_positions[label])
        .collect();
    let rhs_free_axes: Vec<usize> = rhs_free_labels
        .iter()
        .map(|label| rhs_positions[label])
        .collect();
    let contract_axes_lhs: Vec<usize> = contract_labels
        .iter()
        .map(|label| lhs_positions[label])
        .collect();
    let contract_axes_rhs: Vec<usize> = contract_labels
        .iter()
        .map(|label| rhs_positions[label])
        .collect();

    let lhs_perm: Vec<usize> = lhs_free_axes
        .iter()
        .chain(contract_axes_lhs.iter())
        .copied()
        .collect();
    let rhs_perm: Vec<usize> = contract_axes_rhs
        .iter()
        .chain(rhs_free_axes.iter())
        .copied()
        .collect();

    let lhs_free_dims: Vec<usize> = lhs_free_axes.iter().map(|&axis| lhs_dims[axis]).collect();
    let rhs_free_dims: Vec<usize> = rhs_free_axes.iter().map(|&axis| rhs_dims[axis]).collect();
    let contract_dims: Vec<usize> = contract_axes_lhs
        .iter()
        .map(|&axis| lhs_dims[axis])
        .collect();
    if lhs_free_dims.contains(&0) || rhs_free_dims.contains(&0) || contract_dims.contains(&0) {
        return Ok(None);
    }

    let m = product(&lhs_free_dims);
    let k = product(&contract_dims);
    let n = product(&rhs_free_dims);

    let lhs_matrix_dims = vec![m, k];
    let rhs_matrix_dims = vec![k, n];
    let canonical_output_dims: Vec<usize> = lhs_free_dims
        .iter()
        .chain(rhs_free_dims.iter())
        .copied()
        .collect();
    let output_perm: Vec<usize> = out_labels
        .iter()
        .map(|label| {
            canonical_output_labels
                .iter()
                .position(|candidate| candidate == label)
                .ok_or_else(|| {
                    Error::InvalidArgument(format!("strict lowering: missing output label {label}"))
                })
        })
        .collect::<DeviceResult<_>>()?;

    Ok(Some(StrictBinaryLoweringPlan {
        size_dict: size_dict.clone(),
        output_shape,
        lhs_free_labels,
        rhs_free_labels,
        contract_labels,
        lhs_perm,
        rhs_perm,
        lhs_matrix_dims,
        rhs_matrix_dims,
        canonical_output_dims,
        output_perm,
        m,
        n,
        k,
    }))
}

pub(crate) fn compile_strict_binary_lowering_plan(
    subscripts: &Subscripts,
    shapes: &[&[usize]],
    extra: Option<&HashMap<u32, usize>>,
) -> DeviceResult<Option<StrictBinaryLoweringPlan>> {
    if subscripts.inputs.len() != 2 {
        return Err(Error::InvalidArgument(format!(
            "binary einsum requires exactly 2 inputs, got {}",
            subscripts.inputs.len()
        )));
    }

    let size_dict = build_size_dict(subscripts, shapes, extra)?;
    compile_strict_binary_lowering_plan_from_parts(
        &subscripts.inputs[0],
        &subscripts.inputs[1],
        &subscripts.output,
        shapes[0],
        shapes[1],
        &size_dict,
    )
}

pub(crate) fn compile_strict_binary_lowering_step_plan(
    subs_a: &[u32],
    subs_b: &[u32],
    subs_c: &[u32],
    size_dict: &HashMap<u32, usize>,
) -> DeviceResult<Option<StrictBinaryLoweringPlan>> {
    let shape_from_subs = |subs: &[u32]| -> DeviceResult<Vec<usize>> {
        subs.iter()
            .map(|label| {
                size_dict.get(label).copied().ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "strict lowering: missing dimension for label {label}"
                    ))
                })
            })
            .collect()
    };

    let lhs_dims = shape_from_subs(subs_a)?;
    let rhs_dims = shape_from_subs(subs_b)?;
    compile_strict_binary_lowering_plan_from_parts(
        subs_a, subs_b, subs_c, &lhs_dims, &rhs_dims, size_dict,
    )
}
