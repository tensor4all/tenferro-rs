use std::collections::{HashMap, HashSet};

use crate::{EinsumAxis, EinsumNotation, Error, Result, Subscripts};

pub(crate) fn notation_without_ellipsis(notation: &EinsumNotation) -> Result<Subscripts> {
    let inputs = notation
        .inputs
        .iter()
        .map(|term| {
            term.iter()
                .map(|axis| match axis {
                    EinsumAxis::Label(label) => Ok(*label),
                    EinsumAxis::Ellipsis => Err(Error::invalid_subscripts(
                        "ellipsis requires rank-aware resolution",
                    )),
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let output = notation
        .output
        .iter()
        .map(|axis| match axis {
            EinsumAxis::Label(label) => Ok(*label),
            EinsumAxis::Ellipsis => Err(Error::invalid_subscripts(
                "ellipsis requires rank-aware resolution",
            )),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Subscripts { inputs, output })
}

/// Resolve rank-polymorphic notation into the canonical planner representation.
///
/// Ellipsis labels are allocated outside the caller label set. This function
/// also validates per-operand diagonal dimensions and cross-operand
/// equal-or-one broadcasting before any backend or planner code runs.
pub(crate) fn resolve_einsum_notation(
    notation: &EinsumNotation,
    shapes: &[&[usize]],
) -> Result<Subscripts> {
    let notation = canonicalize_ellipsis_labels(notation);
    if notation.inputs.is_empty() {
        return Err(Error::invalid_subscripts(
            "einsum requires at least one input term",
        ));
    }
    if notation.inputs.len() != shapes.len() {
        return Err(Error::invalid_argument(
            "einsum",
            "inputs",
            format!(
                "einsum notation expects {} inputs, got {}",
                notation.inputs.len(),
                shapes.len()
            ),
        ));
    }

    let mut used_labels = HashSet::new();
    let mut ellipsis_ranks = Vec::with_capacity(notation.inputs.len());
    let mut has_input_ellipsis_with_axes = false;
    for (term, shape) in notation.inputs.iter().zip(shapes) {
        let explicit_count = term
            .iter()
            .filter(|axis| matches!(axis, EinsumAxis::Label(_)))
            .count();
        let has_ellipsis = term.contains(&EinsumAxis::Ellipsis);
        if !has_ellipsis && explicit_count != shape.len() {
            return Err(Error::rank_mismatch("einsum", explicit_count, shape.len()));
        }
        if has_ellipsis && explicit_count > shape.len() {
            return Err(Error::rank_mismatch("einsum", explicit_count, shape.len()));
        }
        let ellipsis_rank = shape.len().saturating_sub(explicit_count);
        has_input_ellipsis_with_axes |= has_ellipsis && ellipsis_rank > 0;
        ellipsis_ranks.push(ellipsis_rank);
        for axis in term {
            if let EinsumAxis::Label(label) = axis {
                used_labels.insert(*label);
            }
        }
    }

    let ellipsis_rank = ellipsis_ranks.iter().copied().max().unwrap_or(0);
    let ellipsis_labels = fresh_labels(&used_labels, ellipsis_rank)?;
    let input_has_ellipsis = notation
        .inputs
        .iter()
        .any(|term| term.contains(&EinsumAxis::Ellipsis));
    let output_has_ellipsis = notation.output.contains(&EinsumAxis::Ellipsis);
    if output_has_ellipsis && !input_has_ellipsis {
        return Err(Error::invalid_subscripts(
            "einsum output ellipsis requires an input ellipsis",
        ));
    }
    if has_input_ellipsis_with_axes && !output_has_ellipsis {
        return Err(Error::invalid_subscripts(
            "einsum output must contain '...' when an input ellipsis covers axes",
        ));
    }

    let mut inputs = Vec::with_capacity(notation.inputs.len());
    for (term_index, (term, shape)) in notation.inputs.iter().zip(shapes).enumerate() {
        let labels = expand_term(term, ellipsis_ranks[term_index], &ellipsis_labels);
        if labels.len() != shape.len() {
            return Err(Error::rank_mismatch("einsum", labels.len(), shape.len()));
        }
        inputs.push(labels);
    }

    let output = expand_output(&notation.output, &ellipsis_labels);
    let mut label_sizes = HashMap::new();
    for (labels, shape) in inputs.iter().zip(shapes) {
        let mut local_sizes = HashMap::new();
        for (&label, &size) in labels.iter().zip(shape.iter()) {
            if let Some(previous) = local_sizes.insert(label, size) {
                if previous != size {
                    return Err(Error::shape_mismatch("einsum", [previous], [size]));
                }
            }
            merge_broadcast_size(&mut label_sizes, label, size)?;
        }
    }
    for &label in &output {
        if !label_sizes.contains_key(&label) {
            return Err(Error::invalid_subscripts(format!(
                "output label {label} is missing from all input terms"
            )));
        }
    }

    Ok(Subscripts { inputs, output })
}

fn canonicalize_ellipsis_labels(notation: &EinsumNotation) -> EinsumNotation {
    let has_ellipsis = notation
        .inputs
        .iter()
        .chain(std::iter::once(&notation.output))
        .any(|term| term.contains(&EinsumAxis::Ellipsis));
    if !has_ellipsis {
        return notation.clone();
    }

    let mut labels = HashMap::new();
    let mut next = 0_u32;
    let mut remap = |axis: EinsumAxis| match axis {
        EinsumAxis::Ellipsis => EinsumAxis::Ellipsis,
        EinsumAxis::Label(label) => {
            let canonical = *labels.entry(label).or_insert_with(|| {
                let value = next;
                next = next.saturating_add(1);
                value
            });
            EinsumAxis::Label(canonical)
        }
    };
    EinsumNotation {
        inputs: notation
            .inputs
            .iter()
            .map(|term| term.iter().copied().map(&mut remap).collect())
            .collect(),
        output: notation.output.iter().copied().map(remap).collect(),
    }
}

fn fresh_labels(used: &HashSet<u32>, count: usize) -> Result<Vec<u32>> {
    let mut labels = Vec::with_capacity(count);
    let mut candidate = u32::MAX;
    while labels.len() < count {
        if !used.contains(&candidate) {
            labels.push(candidate);
            if labels.len() == count {
                break;
            }
        }
        candidate = candidate.checked_sub(1).ok_or_else(|| {
            Error::invalid_subscripts("not enough integer labels available for ellipsis axes")
        })?;
    }
    labels.reverse();
    Ok(labels)
}

fn expand_term(term: &[EinsumAxis], rank: usize, ellipsis_labels: &[u32]) -> Vec<u32> {
    let offset = ellipsis_labels.len().saturating_sub(rank);
    let mut labels = Vec::new();
    for axis in term {
        match axis {
            EinsumAxis::Label(label) => labels.push(*label),
            EinsumAxis::Ellipsis => labels.extend_from_slice(&ellipsis_labels[offset..]),
        }
    }
    labels
}

fn expand_output(term: &[EinsumAxis], ellipsis_labels: &[u32]) -> Vec<u32> {
    let mut output = Vec::new();
    for axis in term {
        match axis {
            EinsumAxis::Label(label) => output.push(*label),
            EinsumAxis::Ellipsis => output.extend_from_slice(ellipsis_labels),
        }
    }
    output
}

fn merge_broadcast_size(sizes: &mut HashMap<u32, usize>, label: u32, size: usize) -> Result<()> {
    let Some(existing) = sizes.get_mut(&label) else {
        sizes.insert(label, size);
        return Ok(());
    };
    if *existing == size || *existing == 1 {
        *existing = size;
    } else if size != 1 {
        return Err(Error::shape_mismatch("einsum", [*existing], [size]));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
