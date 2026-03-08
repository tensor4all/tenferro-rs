use std::collections::{HashMap, HashSet};

use tenferro_device::{Error, Result};

use crate::plan::{compile_step_plans, StepPlan};
use crate::subscripts::Subscripts;
use crate::util::{build_size_dict, compute_output_shape, contraction_cost, intermediate_subs};

/// A single step in the contraction sequence.
pub(crate) struct ContractionStep {
    pub(crate) left: usize,
    pub(crate) right: usize,
}

/// Contraction tree determining pairwise contraction order for N-ary einsum.
///
/// When contracting more than two tensors, the order in which pairwise
/// contractions are performed significantly affects performance.
/// `ContractionTree` encodes this order as a binary tree.
///
/// # Optimization
///
/// Use [`ContractionTree::optimize`] for automatic cost-based optimization
/// (e.g., greedy algorithm based on tensor sizes), or
/// [`ContractionTree::from_pairs`] for manual specification.
pub struct ContractionTree {
    /// Original subscripts.
    pub(crate) subscripts: Subscripts,
    /// Steps in the contraction (empty for single-tensor case).
    pub(crate) steps: Vec<ContractionStep>,
    /// Label → dimension size mapping.
    pub(crate) size_dict: HashMap<u32, usize>,
    /// Subscripts for each operand (0..n_inputs from input, then intermediates).
    pub(crate) operand_subs: Vec<Vec<u32>>,
    /// Pre-computed output shapes for each intermediate step (indexed by step_idx).
    pub(crate) step_output_shapes: Vec<Vec<usize>>,
    /// Pre-compiled step plans (cached to avoid recomputation per execute call).
    pub(crate) step_plans: Vec<StepPlan>,
}

impl ContractionTree {
    /// Automatically compute an optimized contraction order.
    ///
    /// Uses a cost-based heuristic (greedy algorithm) to determine
    /// the pairwise contraction sequence that minimizes total operation count.
    ///
    /// # Arguments
    ///
    /// * `subscripts` — Einsum subscripts for all tensors
    /// * `shapes` — Shape of each input tensor
    ///
    /// # Errors
    ///
    /// Returns an error if subscripts and shapes are inconsistent.
    pub fn optimize(subscripts: &Subscripts, shapes: &[&[usize]]) -> Result<Self> {
        let n_inputs = subscripts.inputs.len();
        if n_inputs <= 1 {
            return Self::from_pairs(subscripts, shapes, &[]);
        }

        let size_dict = build_size_dict(subscripts, shapes, None)?;
        let mut available: Vec<usize> = (0..n_inputs).collect();
        let mut operand_subs: Vec<Vec<u32>> = subscripts.inputs.clone();
        let mut pairs: Vec<(usize, usize)> = Vec::new();

        while available.len() > 1 {
            // Compute labels needed by remaining operands and final output
            let mut best_i = 0;
            let mut best_j = 1;
            let mut best_cost = usize::MAX;

            for i in 0..available.len() {
                for j in (i + 1)..available.len() {
                    let li = available[i];
                    let lj = available[j];
                    // Labels needed by remaining operands (excluding this pair) + final output
                    let mut needed = HashSet::new();
                    needed.extend(subscripts.output.iter().copied());
                    for &idx in &available {
                        if idx != li && idx != lj {
                            needed.extend(operand_subs[idx].iter().copied());
                        }
                    }
                    let cost =
                        contraction_cost(&operand_subs[li], &operand_subs[lj], &needed, &size_dict);
                    if cost < best_cost {
                        best_cost = cost;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            let left = available[best_i];
            let right = available[best_j];
            pairs.push((left, right));

            // Compute intermediate subscripts
            let mut needed = HashSet::new();
            needed.extend(subscripts.output.iter().copied());
            for &idx in &available {
                if idx != left && idx != right {
                    needed.extend(operand_subs[idx].iter().copied());
                }
            }
            let new_subs = intermediate_subs(&operand_subs[left], &operand_subs[right], &needed);
            let new_idx = operand_subs.len();
            operand_subs.push(new_subs);

            // Remove consumed (higher index first), add intermediate
            available.remove(best_j);
            available.remove(best_i);
            available.push(new_idx);
        }

        Self::from_pairs(subscripts, shapes, &pairs)
    }

    /// Manually build a contraction tree from a pairwise contraction sequence.
    ///
    /// Each pair `(i, j)` specifies which two tensors (or intermediate results)
    /// to contract next. Intermediate results are assigned indices starting
    /// from the number of input tensors.
    ///
    /// # Arguments
    ///
    /// * `subscripts` — Einsum subscripts for all tensors
    /// * `shapes` — Shape of each input tensor
    /// * `pairs` — Ordered list of pairwise contractions
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Three tensors: A[ij] B[jk] C[kl] -> D[il]
    /// // Contract B and C first, then A with the result:
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    /// let shapes = [&[3, 4][..], &[4, 5], &[5, 6]];
    /// let tree = ContractionTree::from_pairs(
    ///     &subs,
    ///     &shapes,
    ///     &[(1, 2), (0, 3)],  // B*C -> T(index=3), then A*T -> D
    /// ).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the pairs do not form a valid contraction sequence.
    pub fn from_pairs(
        subscripts: &Subscripts,
        shapes: &[&[usize]],
        pairs: &[(usize, usize)],
    ) -> Result<Self> {
        let n_inputs = subscripts.inputs.len();
        let required_steps = n_inputs.saturating_sub(1);
        if pairs.len() != required_steps {
            return Err(Error::InvalidArgument(format!(
                "explicit contraction path for {n_inputs} operands must have {required_steps} steps, got {}",
                pairs.len()
            )));
        }
        let size_dict = build_size_dict(subscripts, shapes, None)?;

        let mut operand_subs: Vec<Vec<u32>> = subscripts.inputs.clone();
        let mut live = vec![false; n_inputs + pairs.len()];
        for slot in live.iter_mut().take(n_inputs) {
            *slot = true;
        }
        let mut steps = Vec::new();

        for (step_idx, &(left, right)) in pairs.iter().enumerate() {
            let next_idx = n_inputs + step_idx;
            if left == right {
                return Err(Error::InvalidArgument(format!(
                    "pair ({left}, {right}) must reference two distinct live operands"
                )));
            }
            if left >= next_idx || right >= next_idx {
                return Err(Error::InvalidArgument(format!(
                    "pair ({left}, {right}) references non-existent operand"
                )));
            }
            if !live[left] || !live[right] {
                return Err(Error::InvalidArgument(format!(
                    "pair ({left}, {right}) references an operand or intermediate that is no longer live"
                )));
            }

            // Labels needed by other live operands + final output
            let mut needed: HashSet<u32> = subscripts.output.iter().copied().collect();
            for (idx, subs) in operand_subs.iter().enumerate() {
                if idx != left && idx != right && live[idx] {
                    needed.extend(subs.iter().copied());
                }
            }

            let new_subs = intermediate_subs(&operand_subs[left], &operand_subs[right], &needed);
            operand_subs.push(new_subs);
            live[left] = false;
            live[right] = false;
            live[next_idx] = true;
            steps.push(ContractionStep { left, right });
        }

        let live_count = live.iter().filter(|&&is_live| is_live).count();
        if live_count != 1 {
            return Err(Error::InvalidArgument(format!(
                "explicit contraction path must leave exactly one live result, got {live_count}"
            )));
        }

        // Pre-compute output shapes for each intermediate step.
        let step_output_shapes: Vec<Vec<usize>> = (0..steps.len())
            .map(|step_idx| {
                let result_idx = n_inputs + step_idx;
                compute_output_shape(&operand_subs[result_idx], &size_dict)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut tree = Self {
            subscripts: subscripts.clone(),
            steps,
            size_dict,
            operand_subs,
            step_output_shapes,
            step_plans: Vec::new(),
        };
        tree.step_plans = compile_step_plans(&tree).map_err(|e| Error::InvalidArgument(e))?;
        Ok(tree)
    }
}
