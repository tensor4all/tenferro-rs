use std::collections::{HashMap, HashSet};

use omeco::{
    CodeOptimizer, EinCode as OmecoEinCode, Initializer, NestedEinsum, ScoreFunction, TreeSA,
};
use tenferro_device::{Error, Result};

use crate::execution::util::{
    build_size_dict, compute_output_shape, contraction_cost, intermediate_subs,
};
use crate::planning::plan::{compile_step_plans, StepPlan};
use crate::syntax::subscripts::Subscripts;

/// A single step in the contraction sequence.
pub(crate) struct ContractionStep {
    pub(crate) left: usize,
    pub(crate) right: usize,
}

/// Public options for automatic contraction-path optimization.
///
/// The default planner uses TreeSA with a greedy initializer and zero annealing
/// iterations. This keeps the public API on a single optimizer family while
/// making the default behavior effectively "greedy-only".
#[derive(Debug, Clone)]
pub struct ContractionOptimizerOptions {
    /// Inverse-temperature schedule for TreeSA.
    pub betas: Vec<f64>,
    /// Number of independent TreeSA trials.
    pub ntrials: usize,
    /// Annealing iterations per temperature level.
    pub niters: usize,
    /// Score function used by TreeSA.
    pub score: ScoreFunction,
}

impl Default for ContractionOptimizerOptions {
    fn default() -> Self {
        Self {
            betas: Vec::new(),
            ntrials: 1,
            niters: 0,
            score: ScoreFunction::default(),
        }
    }
}

impl ContractionOptimizerOptions {
    fn to_treesa(&self) -> TreeSA {
        TreeSA::new(
            self.betas.clone(),
            self.ntrials,
            self.niters,
            Initializer::Greedy,
            self.score.clone(),
        )
    }

    fn validate(&self) -> Result<()> {
        if self.ntrials == 0 {
            return Err(Error::InvalidArgument(
                "contraction optimizer ntrials must be at least 1".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChainAttachment {
    pub(crate) prev_on_left: bool,
    pub(crate) operand: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LinearChainPlan {
    pub(crate) first_pair: (usize, usize),
    pub(crate) attachments: Vec<ChainAttachment>,
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
        Self::optimize_with_options(subscripts, shapes, &ContractionOptimizerOptions::default())
    }

    /// Automatically compute an optimized contraction order with explicit
    /// planner options.
    ///
    /// This routes automatic planning through TreeSA using the provided
    /// configuration. The default options correspond to a greedy-initialized
    /// TreeSA with zero annealing iterations.
    ///
    /// # Errors
    ///
    /// Returns an error if subscripts, shapes, or planner options are invalid.
    pub fn optimize_with_options(
        subscripts: &Subscripts,
        shapes: &[&[usize]],
        options: &ContractionOptimizerOptions,
    ) -> Result<Self> {
        let n_inputs = subscripts.inputs.len();
        if n_inputs <= 1 {
            return Self::from_pairs(subscripts, shapes, &[]);
        }

        options.validate()?;
        let size_dict = build_size_dict(subscripts, shapes, None)?;
        let pairs =
            if let Some(omeco_pairs) = optimize_omeco_pairs(subscripts, &size_dict, options)? {
                omeco_pairs
            } else {
                optimize_self_greedy_pairs(subscripts, &size_dict)
            };
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

    /// Return the number of pairwise contraction steps in this tree.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    /// let tree = ContractionTree::from_pairs(
    ///     &subs,
    ///     &[&[2, 2], &[2, 2], &[2, 2]],
    ///     &[(1, 2), (0, 3)],
    /// )
    /// .unwrap();
    /// assert_eq!(tree.step_count(), 2);
    /// ```
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Return the operand indices for a pairwise contraction step.
    ///
    /// The returned indices refer to the original inputs (`0..n_inputs`) and
    /// then to intermediates (`n_inputs..`) produced by earlier steps.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    /// let tree = ContractionTree::from_pairs(
    ///     &subs,
    ///     &[&[2, 2], &[2, 2], &[2, 2]],
    ///     &[(1, 2), (0, 3)],
    /// )
    /// .unwrap();
    /// assert_eq!(tree.step_pair(0), Some((1, 2)));
    /// ```
    #[must_use]
    pub fn step_pair(&self, step_idx: usize) -> Option<(usize, usize)> {
        self.steps.get(step_idx).map(|step| (step.left, step.right))
    }

    /// Return the `(lhs, rhs, output)` subscripts for a pairwise step.
    ///
    /// The output subscripts are the intermediate labels preserved after the
    /// contraction, or the final output labels on the last step.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    /// let tree = ContractionTree::from_pairs(
    ///     &subs,
    ///     &[&[2, 2], &[2, 2], &[2, 2]],
    ///     &[(1, 2), (0, 3)],
    /// )
    /// .unwrap();
    /// let (lhs, rhs, out) = tree.step_subscripts(0).unwrap();
    /// assert_eq!(lhs, &[1, 2]);
    /// assert_eq!(rhs, &[2, 3]);
    /// assert_eq!(out, &[1, 3]);
    /// ```
    #[must_use]
    pub fn step_subscripts(&self, step_idx: usize) -> Option<(&[u32], &[u32], &[u32])> {
        let n_inputs = self.subscripts.inputs.len();
        let step = self.steps.get(step_idx)?;
        let result_idx = n_inputs + step_idx;
        Some((
            &self.operand_subs[step.left],
            &self.operand_subs[step.right],
            &self.operand_subs[result_idx],
        ))
    }

    pub(crate) fn linear_chain_plan(&self) -> Option<LinearChainPlan> {
        if self.steps.is_empty() {
            return Some(LinearChainPlan {
                first_pair: (0, 0),
                attachments: Vec::new(),
            });
        }

        let n_inputs = self.subscripts.inputs.len();
        let first = self.steps.first()?;
        if first.left >= n_inputs || first.right >= n_inputs {
            return None;
        }

        let mut seen_inputs = vec![false; n_inputs];
        seen_inputs[first.left] = true;
        seen_inputs[first.right] = true;
        let mut attachments = Vec::with_capacity(self.steps.len().saturating_sub(1));
        let mut prev_result_idx = n_inputs;

        for (step_idx, step) in self.steps.iter().enumerate().skip(1) {
            let (prev_on_left, operand) = if step.left == prev_result_idx && step.right < n_inputs {
                (true, step.right)
            } else if step.right == prev_result_idx && step.left < n_inputs {
                (false, step.left)
            } else {
                return None;
            };

            if seen_inputs[operand] {
                return None;
            }
            seen_inputs[operand] = true;
            attachments.push(ChainAttachment {
                prev_on_left,
                operand,
            });
            prev_result_idx = n_inputs + step_idx;
        }

        Some(LinearChainPlan {
            first_pair: (first.left, first.right),
            attachments,
        })
    }
}

fn optimize_omeco_pairs(
    subscripts: &Subscripts,
    size_dict: &HashMap<u32, usize>,
    options: &ContractionOptimizerOptions,
) -> Result<Option<Vec<(usize, usize)>>> {
    let code = OmecoEinCode::new(subscripts.inputs.clone(), subscripts.output.clone());
    let optimizer = options.to_treesa();
    let Some(nested) = optimizer.optimize(&code, size_dict) else {
        return Ok(None);
    };

    let mut next_operand = subscripts.inputs.len();
    let mut pairs = Vec::with_capacity(subscripts.inputs.len().saturating_sub(1));
    nested_to_pairs(&nested, &mut next_operand, &mut pairs)?;
    Ok(Some(pairs))
}

fn nested_to_pairs(
    nested: &NestedEinsum<u32>,
    next_operand: &mut usize,
    pairs: &mut Vec<(usize, usize)>,
) -> Result<usize> {
    match nested {
        NestedEinsum::Leaf { tensor_index } => Ok(*tensor_index),
        NestedEinsum::Node { args, .. } => {
            if args.len() != 2 {
                return Err(Error::InvalidArgument(format!(
                    "omeco returned non-binary contraction node with {} children",
                    args.len()
                )));
            }
            let left = nested_to_pairs(&args[0], next_operand, pairs)?;
            let right = nested_to_pairs(&args[1], next_operand, pairs)?;
            pairs.push((left, right));
            let result_idx = *next_operand;
            *next_operand += 1;
            Ok(result_idx)
        }
    }
}

fn optimize_self_greedy_pairs(
    subscripts: &Subscripts,
    size_dict: &HashMap<u32, usize>,
) -> Vec<(usize, usize)> {
    let n_inputs = subscripts.inputs.len();
    let mut available: Vec<usize> = (0..n_inputs).collect();
    let mut operand_subs: Vec<Vec<u32>> = subscripts.inputs.clone();
    let mut pairs: Vec<(usize, usize)> = Vec::new();

    while available.len() > 1 {
        let mut best_i = 0;
        let mut best_j = 1;
        let mut best_cost = usize::MAX;

        for i in 0..available.len() {
            for j in (i + 1)..available.len() {
                let li = available[i];
                let lj = available[j];
                let mut needed = HashSet::new();
                needed.extend(subscripts.output.iter().copied());
                for &idx in &available {
                    if idx != li && idx != lj {
                        needed.extend(operand_subs[idx].iter().copied());
                    }
                }
                let cost =
                    contraction_cost(&operand_subs[li], &operand_subs[lj], &needed, size_dict);
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
        available.remove(best_j);
        available.remove(best_i);
        available.push(new_idx);
    }

    pairs
}

#[cfg(test)]
mod tests;
