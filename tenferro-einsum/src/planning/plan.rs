use std::collections::{HashMap, HashSet};

use crate::planning::classify::classify_modes;
pub(crate) use crate::planning::strict_binary::{
    compile_strict_binary_lowering_step_plan, StrictBinaryLoweringPlan,
};
use crate::planning::tree::ContractionTree;
use tenferro_device::Result as DeviceResult;

/// Pre-computed information for reducing axes unique to one operand.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ReducePlan {
    /// Subscripts of the operand before reduction.
    pub(crate) original_subs: Vec<u32>,
    /// Subscripts after reduction (labels kept).
    pub(crate) kept_subs: Vec<u32>,
    /// Shape of the tensor after reduction.
    pub(crate) out_shape: Vec<usize>,
}

/// Pre-computed diagonal extraction plan for one operand.
///
/// When an operand has repeated labels (e.g. `A[i,i,j]`), this plan
/// describes how to extract the diagonal via one or more
/// `Tensor::diagonal` stages before the main contraction.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DiagStage {
    /// Axis pairs to pass to `Tensor::diagonal` for this stage.
    pub(crate) axis_pairs: Vec<(usize, usize)>,
    /// Subscripts after this stage: `[non_used..., diag_labels...]`.
    pub(crate) result_subs: Vec<u32>,
}

/// Pre-computed multi-stage diagonal extraction plan for one operand.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DiagPlan {
    /// Sequential diagonal stages, each using only disjoint axis pairs.
    pub(crate) stages: Vec<DiagStage>,
    /// Final subscripts after all diagonal extraction stages.
    pub(crate) result_subs: Vec<u32>,
}

/// Pre-computed GEMM decomposition plan for a pairwise contraction step.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct GemmPlan {
    /// Pre-reduction plan for left operand (None if no reduction needed).
    pub(crate) reduce_a: Option<ReducePlan>,
    /// Pre-reduction plan for right operand (None if no reduction needed).
    pub(crate) reduce_b: Option<ReducePlan>,
    /// Subscripts of A after any pre-reduction.
    pub(crate) subs_a: Vec<u32>,
    /// Subscripts of B after any pre-reduction.
    pub(crate) subs_b: Vec<u32>,
    /// Left-only dimension modes.
    pub(crate) lo_modes: Vec<u32>,
    /// Right-only dimension modes.
    pub(crate) ro_modes: Vec<u32>,
    /// Summed (contracted) dimension modes.
    pub(crate) sum_modes: Vec<u32>,
    /// Pre-computed left-only dimension sizes.
    pub(crate) lo_sizes: Vec<usize>,
    /// Pre-computed right-only dimension sizes.
    pub(crate) ro_sizes: Vec<usize>,
    /// Pre-computed summed dimension sizes.
    pub(crate) sum_sizes: Vec<usize>,
    /// Pre-computed batch dimension sizes.
    pub(crate) batch_sizes: Vec<usize>,
    /// Fused left-only size (product of lo dimensions).
    pub(crate) m: usize,
    /// Fused right-only size (product of ro dimensions).
    pub(crate) n: usize,
    /// Fused summed size (product of sum dimensions).
    pub(crate) k: usize,
    /// Target subscript order for A: [lo..., sum..., batch...].
    pub(crate) target_a: Vec<u32>,
    /// Target subscript order for B: [sum..., ro..., batch...].
    pub(crate) target_b: Vec<u32>,
    /// Shape of GEMM output: [m, n, batch...].
    pub(crate) c_gemm_shape: Vec<usize>,
    /// Expanded shape: [lo..., ro..., batch...].
    pub(crate) expanded_shape: Vec<usize>,
    /// Canonical mode order of expanded output: [lo, ro, batch].
    pub(crate) canonical_modes: Vec<u32>,
    /// Whether a final permute is needed (canonical_modes != subs_c).
    pub(crate) needs_final_permute: bool,
    /// Prepared A shape for GEMM: [m, k, batch...].
    pub(crate) a_gemm_shape: Vec<usize>,
    /// Prepared B shape for GEMM: [k, n, batch...].
    pub(crate) b_gemm_shape: Vec<usize>,
}

/// Pre-computed plan for a single contraction tree step.
///
/// Every binary pattern is decomposed as:
///   diagonal extraction → pre-reduction → GEMM
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct StepPlan {
    /// Diagonal extraction for left operand (None if no repeated labels).
    pub(crate) diag_a: Option<DiagPlan>,
    /// Diagonal extraction for right operand (None if no repeated labels).
    pub(crate) diag_b: Option<DiagPlan>,
    /// Optional strict binary lowering recipe for this step.
    pub(crate) strict_binary: Option<StrictBinaryLoweringPlan>,
    /// GEMM decomposition (always present after diagonal extraction).
    pub(crate) gemm: GemmPlan,
}

/// Pre-compute the reduction plan for axes unique to one operand.
pub(crate) fn compute_reduce_plan(
    subs_self: &[u32],
    subs_other: &[u32],
    subs_out: &[u32],
    size_dict: &HashMap<u32, usize>,
) -> Option<ReducePlan> {
    let other_set: HashSet<u32> = subs_other.iter().copied().collect();
    let out_set: HashSet<u32> = subs_out.iter().copied().collect();

    let mut has_reduction = false;
    let mut kept_subs = Vec::with_capacity(subs_self.len());
    for &label in subs_self {
        if !other_set.contains(&label) && !out_set.contains(&label) {
            has_reduction = true;
        } else {
            kept_subs.push(label);
        }
    }

    if !has_reduction {
        return None;
    }

    let out_shape: Vec<usize> = kept_subs.iter().map(|m| size_dict[m]).collect();
    Some(ReducePlan {
        original_subs: subs_self.to_vec(),
        kept_subs,
        out_shape,
    })
}

/// Pre-compute a diagonal extraction plan for an operand with repeated labels.
///
/// Returns `None` if all labels are unique (no diagonal extraction needed).
pub(crate) fn compute_diag_plan_for_labels(
    subs: &[u32],
    labels_to_extract: &HashSet<u32>,
) -> Option<DiagPlan> {
    fn label_positions(subs: &[u32]) -> HashMap<u32, Vec<usize>> {
        let mut positions = HashMap::new();
        for (i, &label) in subs.iter().enumerate() {
            positions.entry(label).or_insert_with(Vec::new).push(i);
        }
        positions
    }

    fn build_diag_stage(subs: &[u32], labels_to_extract: &HashSet<u32>) -> Option<DiagStage> {
        let positions_by_label = label_positions(subs);
        let mut seen = HashSet::new();
        let repeated_labels: Vec<u32> = subs
            .iter()
            .copied()
            .filter(|label| {
                labels_to_extract.contains(label)
                    && positions_by_label
                        .get(label)
                        .is_some_and(|positions| positions.len() > 1)
                    && seen.insert(*label)
            })
            .collect();

        if repeated_labels.is_empty() {
            return None;
        }

        let mut axis_pairs = Vec::new();
        let mut used = vec![false; subs.len()];
        let mut diag_labels = Vec::new();

        for label in repeated_labels {
            let positions = &positions_by_label[&label];
            for chunk in positions.chunks(2) {
                if let [left, right] = chunk {
                    axis_pairs.push((*left, *right));
                    used[*left] = true;
                    used[*right] = true;
                    diag_labels.push(label);
                }
            }
        }

        if axis_pairs.is_empty() {
            return None;
        }

        let mut result_subs = Vec::with_capacity(subs.len() - axis_pairs.len());
        for (i, &label) in subs.iter().enumerate() {
            if !used[i] {
                result_subs.push(label);
            }
        }
        result_subs.extend(diag_labels);

        Some(DiagStage {
            axis_pairs,
            result_subs,
        })
    }

    let mut current_subs = subs.to_vec();
    let mut stages = Vec::new();
    while let Some(stage) = build_diag_stage(&current_subs, labels_to_extract) {
        current_subs = stage.result_subs.clone();
        stages.push(stage);
    }

    if stages.is_empty() {
        None
    } else {
        Some(DiagPlan {
            stages,
            result_subs: current_subs,
        })
    }
}

pub(crate) fn compute_diag_plan(subs: &[u32]) -> Option<DiagPlan> {
    let mut repeated_labels = HashSet::new();
    let mut label_positions: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, &label) in subs.iter().enumerate() {
        label_positions.entry(label).or_default().push(i);
    }
    for &label in subs {
        if label_positions
            .get(&label)
            .is_some_and(|positions| positions.len() > 1)
        {
            repeated_labels.insert(label);
        }
    }
    compute_diag_plan_for_labels(subs, &repeated_labels)
}

pub(crate) fn compile_pairwise_step_plan(
    subs_a: &[u32],
    subs_b: &[u32],
    subs_c: &[u32],
    size_dict: &HashMap<u32, usize>,
) -> DeviceResult<StepPlan> {
    // 1. Diagonal extraction for repeated labels
    let diag_a = compute_diag_plan(subs_a);
    let diag_b = compute_diag_plan(subs_b);

    let eff_subs_a = diag_a
        .as_ref()
        .map(|d| d.result_subs.as_slice())
        .unwrap_or(subs_a);
    let eff_subs_b = diag_b
        .as_ref()
        .map(|d| d.result_subs.as_slice())
        .unwrap_or(subs_b);

    // 2. Pre-reduction for unique-only axes
    let reduce_a = compute_reduce_plan(eff_subs_a, eff_subs_b, subs_c, size_dict);
    let reduce_b = compute_reduce_plan(eff_subs_b, eff_subs_a, subs_c, size_dict);

    let effective_a = reduce_a
        .as_ref()
        .map(|r| r.kept_subs.clone())
        .unwrap_or_else(|| eff_subs_a.to_vec());
    let effective_b = reduce_b
        .as_ref()
        .map(|r| r.kept_subs.clone())
        .unwrap_or_else(|| eff_subs_b.to_vec());

    // 3. Classify modes and build GemmPlan
    let (batch_modes, lo_modes, ro_modes, sum_modes) =
        classify_modes(&effective_a, &effective_b, subs_c);

    let batch_sizes: Vec<usize> = batch_modes.iter().map(|m| size_dict[m]).collect();
    let lo_sizes: Vec<usize> = lo_modes.iter().map(|m| size_dict[m]).collect();
    let ro_sizes: Vec<usize> = ro_modes.iter().map(|m| size_dict[m]).collect();
    let sum_sizes: Vec<usize> = sum_modes.iter().map(|m| size_dict[m]).collect();

    let m = product_or_one_for_empty(&lo_sizes);
    let n = product_or_one_for_empty(&ro_sizes);
    let k = product_or_one_for_empty(&sum_sizes);

    let target_a: Vec<u32> = lo_modes
        .iter()
        .chain(sum_modes.iter())
        .chain(batch_modes.iter())
        .copied()
        .collect();
    let target_b: Vec<u32> = sum_modes
        .iter()
        .chain(ro_modes.iter())
        .chain(batch_modes.iter())
        .copied()
        .collect();

    let a_gemm_shape: Vec<usize> = std::iter::once(m)
        .chain(std::iter::once(k))
        .chain(batch_sizes.iter().copied())
        .collect();
    let b_gemm_shape: Vec<usize> = std::iter::once(k)
        .chain(std::iter::once(n))
        .chain(batch_sizes.iter().copied())
        .collect();
    let c_gemm_shape: Vec<usize> = std::iter::once(m)
        .chain(std::iter::once(n))
        .chain(batch_sizes.iter().copied())
        .collect();

    let expanded_shape: Vec<usize> = lo_sizes
        .iter()
        .chain(ro_sizes.iter())
        .chain(batch_sizes.iter())
        .copied()
        .collect();

    let canonical_modes: Vec<u32> = lo_modes
        .iter()
        .chain(ro_modes.iter())
        .chain(batch_modes.iter())
        .copied()
        .collect();

    let needs_final_permute = canonical_modes.as_slice() != subs_c;
    let strict_binary =
        compile_strict_binary_lowering_step_plan(subs_a, subs_b, subs_c, size_dict)?;

    Ok(StepPlan {
        diag_a,
        diag_b,
        strict_binary,
        gemm: GemmPlan {
            reduce_a,
            reduce_b,
            subs_a: effective_a,
            subs_b: effective_b,
            lo_modes,
            ro_modes,
            sum_modes,
            lo_sizes,
            ro_sizes,
            sum_sizes,
            batch_sizes,
            m,
            n,
            k,
            target_a,
            target_b,
            c_gemm_shape,
            expanded_shape,
            canonical_modes,
            needs_final_permute,
            a_gemm_shape,
            b_gemm_shape,
        },
    })
}

fn product_or_one_for_empty(sizes: &[usize]) -> usize {
    if sizes.is_empty() {
        1
    } else {
        sizes.iter().product()
    }
}

/// Compile step plans for all steps in a contraction tree.
///
/// Every binary pattern is decomposed as:
///   1. Diagonal extraction (if repeated labels in operand)
///   2. Pre-reduction (if unique-only axes)
///   3. GEMM (always, after steps 1-2 make all labels unique)
pub(crate) fn compile_step_plans(tree: &ContractionTree) -> DeviceResult<Vec<StepPlan>> {
    let n_inputs = tree.subscripts.inputs.len();
    let size_dict = &tree.size_dict;

    tree.steps
        .iter()
        .enumerate()
        .map(|(step_idx, step)| {
            let subs_a = &tree.operand_subs[step.left];
            let subs_b = &tree.operand_subs[step.right];
            let is_last = step_idx == tree.steps.len() - 1;
            let subs_c = if is_last {
                &tree.subscripts.output
            } else {
                &tree.operand_subs[n_inputs + step_idx]
            };
            compile_pairwise_step_plan(subs_a, subs_b, subs_c, size_dict)
        })
        .collect()
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
