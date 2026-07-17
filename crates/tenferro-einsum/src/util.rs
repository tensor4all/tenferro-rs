use std::collections::{HashMap, HashSet};

use crate::syntax::subscripts::Subscripts;
use crate::{Error, Result};

/// Build a label -> size mapping from subscripts and input shapes.
pub(crate) fn build_size_dict(
    subscripts: &Subscripts,
    shapes: &[&[usize]],
    extra: Option<&HashMap<u32, usize>>,
) -> Result<HashMap<u32, usize>> {
    if subscripts.inputs.len() != shapes.len() {
        return Err(Error::planning(format!(
            "expected {} input shapes, got {}",
            subscripts.inputs.len(),
            shapes.len()
        )));
    }
    let mut size_dict: HashMap<u32, usize> = HashMap::new();
    for (i, input_subs) in subscripts.inputs.iter().enumerate() {
        if input_subs.len() != shapes[i].len() {
            return Err(Error::planning(format!(
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
                    return Err(Error::shape_mismatch("einsum", [existing], [size]));
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
                .ok_or_else(|| Error::planning(format!("unknown size for label {label}")))
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

/// Map each source label occurrence to a distinct matching target axis.
///
/// Repeated labels are occurrence-sensitive: mapping `["i", "j", "i"]` into
/// `["i", "i", "j"]` must use axes `[0, 2, 1]`, not `[0, 2, 0]`.
pub(crate) fn map_label_occurrences(
    source_labels: &[u32],
    target_labels: &[u32],
) -> Option<Vec<usize>> {
    let mut used = vec![false; target_labels.len()];
    source_labels
        .iter()
        .map(|label| {
            let axis = target_labels
                .iter()
                .enumerate()
                .find_map(|(axis, target)| (!used[axis] && target == label).then_some(axis))?;
            used[axis] = true;
            Some(axis)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::map_label_occurrences;

    #[test]
    fn label_occurrence_mapping_uses_each_target_axis_once() {
        assert_eq!(
            map_label_occurrences(
                &[b'i' as u32, b'j' as u32, b'i' as u32],
                &[b'i' as u32, b'i' as u32, b'j' as u32,]
            ),
            Some(vec![0, 2, 1])
        );
        assert_eq!(
            map_label_occurrences(
                &[b'i' as u32, b'i' as u32, b'i' as u32],
                &[b'i' as u32, b'i' as u32,]
            ),
            None
        );
    }
}
