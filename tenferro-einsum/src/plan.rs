use std::collections::{HashMap, HashSet};

use crate::classify::classify_modes;
use crate::tree::ContractionTree;

/// Pre-computed information for reducing axes unique to one operand.
pub(crate) struct ReducePlan {
    /// Subscripts of the operand before reduction.
    pub(crate) original_subs: Vec<u32>,
    /// Subscripts after reduction (labels kept).
    pub(crate) kept_subs: Vec<u32>,
    /// Shape of the tensor after reduction.
    pub(crate) out_shape: Vec<usize>,
}

/// Pre-computed GEMM decomposition plan for a pairwise contraction step.
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
    /// Pre-computed batch dimension sizes.
    pub(crate) batch_sizes: Vec<usize>,
    /// Fused left-only size (product of lo dimensions).
    pub(crate) m: usize,
    /// Fused right-only size (product of ro dimensions).
    pub(crate) n: usize,
    /// Fused summed size (product of sum dimensions).
    pub(crate) k: usize,
    /// Target subscript order for A: [batch..., lo..., sum...].
    pub(crate) target_a: Vec<u32>,
    /// Target subscript order for B: [batch..., sum..., ro...].
    pub(crate) target_b: Vec<u32>,
    /// Shape of GEMM output: [batch..., m, n].
    pub(crate) c_gemm_shape: Vec<usize>,
    /// Expanded shape: [batch..., lo..., ro...].
    pub(crate) expanded_shape: Vec<usize>,
    /// Canonical mode order of expanded output: [batch, lo, ro].
    pub(crate) canonical_modes: Vec<u32>,
    /// Whether a final permute is needed (canonical_modes != subs_c).
    pub(crate) needs_final_permute: bool,
}

/// Pre-computed outer-product plan for a pairwise contraction step.
pub(crate) struct OuterProductPlan {
    /// Canonical mode order: [a_modes..., b_modes...].
    pub(crate) canonical_modes: Vec<u32>,
    /// Shape of A after reshape for broadcast.
    pub(crate) a_ext_shape: Vec<usize>,
    /// Shape of B after reshape for broadcast.
    pub(crate) b_ext_shape: Vec<usize>,
    /// Full canonical shape.
    pub(crate) canonical_shape: Vec<usize>,
    /// Whether a final permute is needed (canonical_modes != subs_c).
    pub(crate) needs_final_permute: bool,
}

/// Pre-computed contraction strategy for one tree step.
pub(crate) enum StepStrategy {
    /// subs_a == subs_b == subs_c: direct ElementwiseMul.
    ElementwiseMul,
    /// Disjoint binary einsum: broadcast + ElementwiseMul.
    OuterProduct(OuterProductPlan),
    /// Contraction: try Contract extension first, fall back to permute+GEMM.
    /// Some(plan) = GEMM-compatible (has pre-computed GemmPlan for fallback).
    /// None = not GEMM-compatible (trace-like, Contract extension only).
    Contraction(Option<GemmPlan>),
}

/// Pre-computed plan for a single contraction tree step.
pub(crate) struct StepPlan {
    pub(crate) strategy: StepStrategy,
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

/// Compile step plans for all steps in a contraction tree.
///
/// Pre-computes strategy, mode classification, sizes, and target subscripts
/// for each step, eliminating per-step HashMap/HashSet allocations at execution time.
pub(crate) fn compile_step_plans(tree: &ContractionTree) -> Vec<StepPlan> {
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

            // Check ElementwiseMul: same labels, same order
            if subs_a == subs_b && subs_a.as_slice() == subs_c {
                return StepPlan {
                    strategy: StepStrategy::ElementwiseMul,
                };
            }

            // Check outer product: disjoint labels
            let set_a: HashSet<u32> = subs_a.iter().copied().collect();
            let set_b: HashSet<u32> = subs_b.iter().copied().collect();
            if !set_a.iter().any(|m| set_b.contains(m)) {
                let set_c: HashSet<u32> = subs_c.iter().copied().collect();
                let canonical_modes: Vec<u32> =
                    subs_a.iter().chain(subs_b.iter()).copied().collect();
                let canonical_set: HashSet<u32> = canonical_modes.iter().copied().collect();
                // Unique labels in each operand and output, and output = a ∪ b
                let unique_a = subs_a.len() == set_a.len();
                let unique_b = subs_b.len() == set_b.len();
                let unique_c = subs_c.len() == set_c.len();
                if unique_a && unique_b && unique_c && set_c == canonical_set {
                    let canonical_shape: Vec<usize> =
                        canonical_modes.iter().map(|m| size_dict[m]).collect();
                    let a_ext_shape: Vec<usize> = canonical_modes
                        .iter()
                        .map(|m| if set_a.contains(m) { size_dict[m] } else { 1 })
                        .collect();
                    let b_ext_shape: Vec<usize> = canonical_modes
                        .iter()
                        .map(|m| if set_b.contains(m) { size_dict[m] } else { 1 })
                        .collect();
                    let needs_final_permute = canonical_modes.as_slice() != subs_c;
                    return StepPlan {
                        strategy: StepStrategy::OuterProduct(OuterProductPlan {
                            canonical_modes,
                            a_ext_shape,
                            b_ext_shape,
                            canonical_shape,
                            needs_final_permute,
                        }),
                    };
                }
            }

            // Check GEMM compatibility
            let unique = |subs: &[u32]| -> bool {
                let mut seen = HashSet::with_capacity(subs.len());
                subs.iter().all(|&m| seen.insert(m))
            };
            if unique(subs_a) && unique(subs_b) && unique(subs_c) {
                let set_c: HashSet<u32> = subs_c.iter().copied().collect();
                let set_ab: HashSet<u32> = set_a.union(&set_b).copied().collect();
                if set_c.is_subset(&set_ab) {
                    // GEMM-compatible: pre-compute the full plan
                    let reduce_a = compute_reduce_plan(subs_a, subs_b, subs_c, size_dict);
                    let reduce_b = compute_reduce_plan(subs_b, subs_a, subs_c, size_dict);

                    let effective_a = reduce_a
                        .as_ref()
                        .map(|r| r.kept_subs.clone())
                        .unwrap_or_else(|| subs_a.to_vec());
                    let effective_b = reduce_b
                        .as_ref()
                        .map(|r| r.kept_subs.clone())
                        .unwrap_or_else(|| subs_b.to_vec());

                    let (batch_modes, lo_modes, ro_modes, sum_modes) =
                        classify_modes(&effective_a, &effective_b, subs_c);

                    let batch_sizes: Vec<usize> =
                        batch_modes.iter().map(|m| size_dict[m]).collect();
                    let lo_sizes: Vec<usize> = lo_modes.iter().map(|m| size_dict[m]).collect();
                    let ro_sizes: Vec<usize> = ro_modes.iter().map(|m| size_dict[m]).collect();
                    let sum_sizes: Vec<usize> = sum_modes.iter().map(|m| size_dict[m]).collect();

                    let m = lo_sizes.iter().product::<usize>().max(1);
                    let n = ro_sizes.iter().product::<usize>().max(1);
                    let k = sum_sizes.iter().product::<usize>().max(1);

                    let target_a: Vec<u32> = batch_modes
                        .iter()
                        .chain(lo_modes.iter())
                        .chain(sum_modes.iter())
                        .copied()
                        .collect();
                    let target_b: Vec<u32> = batch_modes
                        .iter()
                        .chain(sum_modes.iter())
                        .chain(ro_modes.iter())
                        .copied()
                        .collect();

                    let c_gemm_shape: Vec<usize> = batch_sizes
                        .iter()
                        .copied()
                        .chain(std::iter::once(m))
                        .chain(std::iter::once(n))
                        .collect();

                    let expanded_shape: Vec<usize> = batch_sizes
                        .iter()
                        .chain(lo_sizes.iter())
                        .chain(ro_sizes.iter())
                        .copied()
                        .collect();

                    let canonical_modes: Vec<u32> = batch_modes
                        .iter()
                        .chain(lo_modes.iter())
                        .chain(ro_modes.iter())
                        .copied()
                        .collect();

                    let needs_final_permute = canonical_modes.as_slice() != subs_c;

                    return StepPlan {
                        strategy: StepStrategy::Contraction(Some(GemmPlan {
                            reduce_a,
                            reduce_b,
                            subs_a: effective_a,
                            subs_b: effective_b,
                            lo_modes,
                            ro_modes,
                            sum_modes,
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
                        })),
                    };
                }
            }

            // Fallback: not GEMM-compatible, Contract extension only
            StepPlan {
                strategy: StepStrategy::Contraction(None),
            }
        })
        .collect()
}
