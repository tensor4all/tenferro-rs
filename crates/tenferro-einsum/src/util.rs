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
        let mut local_sizes = HashMap::new();
        for (j, &label) in input_subs.iter().enumerate() {
            let size = shapes[i][j];
            if let Some(previous) = local_sizes.insert(label, size) {
                if previous != size {
                    return Err(Error::shape_mismatch("einsum", [previous], [size]));
                }
            }
            if let Some(existing) = size_dict.get_mut(&label) {
                if *existing == size || *existing == 1 {
                    *existing = size;
                } else if size != 1 {
                    return Err(Error::shape_mismatch("einsum", [*existing], [size]));
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

/// One extent as the broadcast decision sees it: known, or still symbolic.
pub(crate) trait Extent {
    fn known(self) -> Option<usize>;
}

impl Extent for usize {
    fn known(self) -> Option<usize> {
        Some(self)
    }
}

impl Extent for Option<usize> {
    fn known(self) -> Option<usize> {
        self
    }
}

/// Whether a label shared between operands repeats with extents that differ
/// only by a size-1 axis, i.e. whether the operands need singleton-dimension
/// broadcasting.
///
/// This is the single broadcast decision both einsum surfaces call: eager
/// passes fully concrete extents, tracing passes the ones it already knows and
/// skips the rest, so a symbolic extent can only narrow the decision. A label
/// repeated inside one operand is a diagonal, whose extents must match exactly,
/// so each operand joins the size map only after it is read.
pub(crate) fn requires_broadcast<E: Extent + Copy>(shapes: &[&[E]], labels: &[Vec<u32>]) -> bool {
    let mut sizes = HashMap::<u32, usize>::new();
    for (labels, shape) in labels.iter().zip(shapes) {
        let mut operand_sizes = HashMap::<u32, usize>::new();
        for (&label, &extent) in labels.iter().zip(shape.iter()) {
            let Some(size) = extent.known() else { continue };
            if let Some(&previous) = sizes.get(&label) {
                if previous != size && (previous == 1 || size == 1) {
                    return true;
                }
            }
            operand_sizes.insert(label, size);
        }
        sizes.extend(operand_sizes);
    }
    false
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
    use super::{map_label_occurrences, requires_broadcast};

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

    #[test]
    fn broadcast_decision_only_counts_labels_shared_between_operands() {
        assert!(requires_broadcast(
            &[&[2, 1], &[4, 2, 3]],
            &[vec![0, 1], vec![2, 3, 1]]
        ));
        assert!(!requires_broadcast(
            &[&[2, 3], &[4, 2, 3]],
            &[vec![0, 1], vec![2, 3, 1]]
        ));
        // A label repeated inside one operand is a diagonal, whose extents must
        // match exactly; it is never a singleton broadcast.
        assert!(!requires_broadcast(&[&[1, 3]], &[vec![0, 0]]));
        // A symbolic extent is skipped, not guessed.
        assert!(requires_broadcast(
            &[&[None, Some(1)], &[Some(4), Some(2), Some(3)]],
            &[vec![0, 1], vec![2, 3, 1]]
        ));
        assert!(!requires_broadcast(
            &[&[None, Some(2)], &[Some(4), Some(2), Some(3)]],
            &[vec![0, 1], vec![2, 3, 1]]
        ));
    }
}
