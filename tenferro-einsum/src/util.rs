use std::collections::{HashMap, HashSet};

use tenferro_device::{Error, Result};

use crate::syntax::subscripts::Subscripts;

/// Build a label -> size mapping from subscripts and input shapes.
pub(crate) fn build_size_dict(
    subscripts: &Subscripts,
    shapes: &[&[usize]],
    extra: Option<&HashMap<u32, usize>>,
) -> Result<HashMap<u32, usize>> {
    if subscripts.inputs.len() != shapes.len() {
        return Err(Error::InvalidArgument(format!(
            "expected {} input shapes, got {}",
            subscripts.inputs.len(),
            shapes.len()
        )));
    }
    let mut size_dict: HashMap<u32, usize> = HashMap::new();
    for (i, input_subs) in subscripts.inputs.iter().enumerate() {
        if input_subs.len() != shapes[i].len() {
            return Err(Error::InvalidArgument(format!(
                "input {} has {} subscript labels but shape has {} dimensions",
                i,
                input_subs.len(),
                shapes[i].len()
            )));
        }
        for (j, &label) in input_subs.iter().enumerate() {
            let size = shapes[i][j];
            if let Some(&existing) = size_dict.get(&label) {
                if existing != size {
                    return Err(Error::ShapeMismatch {
                        expected: vec![existing],
                        got: vec![size],
                    });
                }
            } else {
                size_dict.insert(label, size);
            }
        }
    }
    if let Some(sd) = extra {
        for (&label, &size) in sd {
            size_dict.entry(label).or_insert(size);
        }
    }
    Ok(size_dict)
}

/// Compute output shape from output subscripts and size dictionary.
pub(crate) fn compute_output_shape(
    output_subs: &[u32],
    size_dict: &HashMap<u32, usize>,
) -> Result<Vec<usize>> {
    output_subs
        .iter()
        .map(|&label| {
            size_dict
                .get(&label)
                .copied()
                .ok_or_else(|| Error::InvalidArgument(format!("unknown size for label {label}")))
        })
        .collect()
}

/// Compute intermediate subscripts when contracting two operands.
/// Keeps labels from left/right that appear in the `needed` set.
pub(crate) fn intermediate_subs(
    subs_left: &[u32],
    subs_right: &[u32],
    needed: &HashSet<u32>,
) -> Vec<u32> {
    let mut seen = HashSet::new();
    let mut output = Vec::new();
    for &l in subs_left.iter().chain(subs_right.iter()) {
        if needed.contains(&l) && seen.insert(l) {
            output.push(l);
        }
    }
    output
}

/// Compute the cost (output size) of contracting two operands.
pub(crate) fn contraction_cost(
    subs_a: &[u32],
    subs_b: &[u32],
    needed: &HashSet<u32>,
    size_dict: &HashMap<u32, usize>,
) -> Result<usize> {
    let out_subs = intermediate_subs(subs_a, subs_b, needed);
    let mut cost = 1usize;
    for label in out_subs {
        let size = size_dict.get(&label).copied().ok_or_else(|| {
            Error::InvalidArgument(format!(
                "unknown size for label {label} in contraction cost"
            ))
        })?;
        cost = cost.saturating_mul(size);
    }
    Ok(cost.max(1))
}
