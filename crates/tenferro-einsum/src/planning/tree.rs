use std::collections::{HashMap, HashSet};
use std::fmt;
use std::mem::{size_of, size_of_val};

use omeco::{
    CodeOptimizer, EinCode as OmecoEinCode, Initializer, NestedEinsum, ScoreFunction, TreeSA,
};

use crate::cache::{saturating_sum, vec_of_vec_retained_bytes, vec_retained_bytes};
use crate::planning::plan::{compile_step_plans, DiagPlan, GemmPlan, ReducePlan, StepPlan};
use crate::syntax::subscripts::Subscripts;
use crate::util::{build_size_dict, intermediate_subs};
use crate::{Error, Result};

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

    pub(crate) fn validate(&self) -> Result<()> {
        if self.ntrials == 0 {
            return Err(Error::InvalidArgument(
                "contraction optimizer ntrials must be at least 1".into(),
            ));
        }
        if self.betas.iter().any(|value| value.is_nan()) {
            return Err(Error::InvalidArgument(
                "contraction optimizer betas must not contain NaN".into(),
            ));
        }
        if self.score.tc_weight.is_nan()
            || self.score.sc_weight.is_nan()
            || self.score.rw_weight.is_nan()
            || self.score.sc_target.is_nan()
        {
            return Err(Error::InvalidArgument(
                "contraction optimizer score fields must not contain NaN".into(),
            ));
        }
        Ok(())
    }
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
    /// Subscripts for each operand (0..input_count from input, then intermediates).
    pub(crate) operand_subs: Vec<Vec<u32>>,
    /// Pre-compiled step plans (cached to avoid recomputation per execute call).
    pub(crate) step_plans: Vec<StepPlan>,
}

impl fmt::Debug for ContractionTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContractionTree")
            .field("input_count", &self.subscripts.inputs.len())
            .field("output_rank", &self.subscripts.output.len())
            .field("steps_len", &self.steps.len())
            .field("size_dict_len", &self.size_dict.len())
            .field("operand_subs_len", &self.operand_subs.len())
            .field("step_plans_len", &self.step_plans.len())
            .finish_non_exhaustive()
    }
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
        options.validate()?;
        let input_count = subscripts.inputs.len();
        if input_count <= 1 {
            return Self::from_pairs(subscripts, shapes, &[]);
        }

        let size_dict = build_size_dict(subscripts, shapes, None)?;
        let pairs =
            if let Some(omeco_pairs) = optimize_omeco_pairs(subscripts, &size_dict, options)? {
                omeco_pairs
            } else {
                optimize_self_greedy_pairs(subscripts, &size_dict)?
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
    /// ```rust
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
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
        let input_count = subscripts.inputs.len();
        let required_steps = input_count.saturating_sub(1);
        if pairs.len() != required_steps {
            return Err(Error::InvalidArgument(format!(
                "explicit contraction path for {input_count} operands must have {required_steps} steps, got {}",
                pairs.len()
            )));
        }
        let size_dict = build_size_dict(subscripts, shapes, None)?;

        let mut operand_subs: Vec<Vec<u32>> = subscripts.inputs.clone();
        let mut live = vec![false; input_count + pairs.len()];
        for slot in live.iter_mut().take(input_count) {
            *slot = true;
        }
        let mut steps = Vec::new();

        for (step_idx, &(left, right)) in pairs.iter().enumerate() {
            let next_idx = input_count + step_idx;
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

        let mut tree = Self {
            subscripts: subscripts.clone(),
            steps,
            size_dict,
            operand_subs,
            step_plans: Vec::new(),
        };
        tree.step_plans = compile_step_plans(&tree)?;
        Ok(tree)
    }

    /// Return the number of pairwise contraction steps in this tree.
    ///
    /// # Examples
    ///
    /// ```rust
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
    /// The returned indices refer to the original inputs (`0..input_count`) and
    /// then to intermediates (`input_count..`) produced by earlier steps.
    ///
    /// # Examples
    ///
    /// ```rust
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
    /// ```rust
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
        let input_count = self.subscripts.inputs.len();
        let step = self.steps.get(step_idx)?;
        let result_idx = input_count + step_idx;
        let output_subs = if step_idx + 1 == self.steps.len() {
            &self.subscripts.output
        } else {
            &self.operand_subs[result_idx]
        };
        Some((
            &self.operand_subs[step.left],
            &self.operand_subs[step.right],
            output_subs,
        ))
    }

    /// Return the precomputed lowering plan for one pairwise contraction step.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    ///
    /// assert_eq!(tree.step_plan(0).unwrap().gemm().m(), 2);
    /// ```
    #[must_use]
    pub fn step_plan(&self, step_idx: usize) -> Option<crate::lowering::PairwiseStepPlan<'_>> {
        self.step_plans
            .get(step_idx)
            .map(crate::lowering::PairwiseStepPlan::new)
    }

    #[doc(hidden)]
    #[must_use]
    pub(crate) fn retained_bytes_for_cache_stats(&self) -> usize {
        saturating_sum([
            size_of::<Self>(),
            subscripts_retained_bytes(&self.subscripts),
            self.steps
                .capacity()
                .saturating_mul(size_of::<ContractionStep>()),
            self.size_dict
                .capacity()
                .saturating_mul(size_of::<u32>().saturating_add(size_of::<usize>())),
            vec_of_vec_retained_bytes(&self.operand_subs),
            self.step_plans
                .capacity()
                .saturating_mul(size_of::<StepPlan>()),
            saturating_sum(self.step_plans.iter().map(step_plan_retained_bytes)),
        ])
    }
}

fn subscripts_retained_bytes(subscripts: &Subscripts) -> usize {
    saturating_sum([
        vec_of_vec_retained_bytes(&subscripts.inputs),
        vec_retained_bytes(&subscripts.output),
    ])
}

fn reduce_plan_retained_bytes(plan: &ReducePlan) -> usize {
    saturating_sum([
        vec_retained_bytes(&plan.original_subs),
        vec_retained_bytes(&plan.kept_subs),
        vec_retained_bytes(&plan.out_shape),
    ])
}

fn diag_plan_retained_bytes(plan: &DiagPlan) -> usize {
    saturating_sum([
        vec_retained_bytes(&plan.stages),
        saturating_sum(plan.stages.iter().map(|stage| {
            saturating_sum([
                vec_retained_bytes(&stage.axis_pairs),
                vec_retained_bytes(&stage.result_subs),
            ])
        })),
        vec_retained_bytes(&plan.result_subs),
    ])
}

fn gemm_plan_retained_bytes(plan: &GemmPlan) -> usize {
    saturating_sum([
        plan.reduce_a.as_ref().map_or(0, reduce_plan_retained_bytes),
        plan.reduce_b.as_ref().map_or(0, reduce_plan_retained_bytes),
        vec_retained_bytes(&plan.subs_a),
        vec_retained_bytes(&plan.subs_b),
        vec_retained_bytes(&plan.lo_modes),
        vec_retained_bytes(&plan.ro_modes),
        vec_retained_bytes(&plan.sum_modes),
        vec_retained_bytes(&plan.lo_sizes),
        vec_retained_bytes(&plan.ro_sizes),
        vec_retained_bytes(&plan.sum_sizes),
        vec_retained_bytes(&plan.batch_sizes),
        vec_retained_bytes(&plan.target_a),
        vec_retained_bytes(&plan.target_b),
        vec_retained_bytes(&plan.c_gemm_shape),
        vec_retained_bytes(&plan.expanded_shape),
        vec_retained_bytes(&plan.canonical_modes),
        vec_retained_bytes(&plan.a_gemm_shape),
        vec_retained_bytes(&plan.b_gemm_shape),
    ])
}

fn step_plan_retained_bytes(plan: &StepPlan) -> usize {
    saturating_sum([
        plan.diag_a.as_ref().map_or(0, diag_plan_retained_bytes),
        plan.diag_b.as_ref().map_or(0, diag_plan_retained_bytes),
        plan.strict_binary.as_ref().map_or(0, size_of_val),
        gemm_plan_retained_bytes(&plan.gemm),
    ])
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

fn build_operand_label_sets(operand_subs: &[Vec<u32>]) -> Vec<HashSet<u32>> {
    operand_subs
        .iter()
        .map(|subs| subs.iter().copied().collect())
        .collect()
}

fn build_needed_label_counts(
    output_subs: &[u32],
    available: &[usize],
    operand_label_sets: &[HashSet<u32>],
) -> HashMap<u32, usize> {
    let mut counts = HashMap::new();
    for &label in output_subs {
        counts.entry(label).or_insert(1);
    }
    for &idx in available {
        add_labels_to_counts(&mut counts, &operand_label_sets[idx]);
    }
    counts
}

fn add_labels_to_counts(counts: &mut HashMap<u32, usize>, labels: &HashSet<u32>) {
    for &label in labels {
        *counts.entry(label).or_insert(0) += 1;
    }
}

fn remove_labels_from_counts(counts: &mut HashMap<u32, usize>, labels: &HashSet<u32>) {
    for &label in labels {
        match counts.get(&label).copied() {
            Some(1) => {
                counts.remove(&label);
            }
            Some(count) => {
                counts.insert(label, count - 1);
            }
            None => {}
        }
    }
}

fn candidate_label_is_needed(
    label: u32,
    left: usize,
    right: usize,
    operand_label_sets: &[HashSet<u32>],
    needed_label_counts: &HashMap<u32, usize>,
) -> bool {
    let mut selected_count = 0;
    if operand_label_sets[left].contains(&label) {
        selected_count += 1;
    }
    if operand_label_sets[right].contains(&label) {
        selected_count += 1;
    }
    needed_label_counts.get(&label).copied().unwrap_or(0) > selected_count
}

fn collect_candidate_intermediate_subs(
    subs_left: &[u32],
    subs_right: &[u32],
    left: usize,
    right: usize,
    operand_label_sets: &[HashSet<u32>],
    needed_label_counts: &HashMap<u32, usize>,
    output: &mut Vec<u32>,
) {
    output.clear();
    for &label in subs_left.iter().chain(subs_right.iter()) {
        if candidate_label_is_needed(label, left, right, operand_label_sets, needed_label_counts)
            && !output.contains(&label)
        {
            output.push(label);
        }
    }
}

#[derive(Clone, Copy)]
struct CandidateCostContext<'a> {
    operand_label_sets: &'a [HashSet<u32>],
    needed_label_counts: &'a HashMap<u32, usize>,
    size_dict: &'a HashMap<u32, usize>,
}

fn candidate_contraction_cost(
    subs_left: &[u32],
    subs_right: &[u32],
    left: usize,
    right: usize,
    context: CandidateCostContext<'_>,
    candidate_subs: &mut Vec<u32>,
) -> Result<usize> {
    collect_candidate_intermediate_subs(
        subs_left,
        subs_right,
        left,
        right,
        context.operand_label_sets,
        context.needed_label_counts,
        candidate_subs,
    );
    let mut cost = 1usize;
    for &label in candidate_subs.iter() {
        let size = context.size_dict.get(&label).copied().ok_or_else(|| {
            Error::InvalidArgument(format!(
                "unknown size for label {label} in contraction cost"
            ))
        })?;
        cost = cost.saturating_mul(size);
    }
    Ok(cost.max(1))
}

fn optimize_self_greedy_pairs(
    subscripts: &Subscripts,
    size_dict: &HashMap<u32, usize>,
) -> Result<Vec<(usize, usize)>> {
    let input_count = subscripts.inputs.len();
    let mut available: Vec<usize> = (0..input_count).collect();
    let mut operand_subs: Vec<Vec<u32>> = subscripts.inputs.clone();
    let mut operand_label_sets = build_operand_label_sets(&operand_subs);
    let mut needed_label_counts =
        build_needed_label_counts(&subscripts.output, &available, &operand_label_sets);
    let mut candidate_subs = Vec::new();
    let mut pairs: Vec<(usize, usize)> = Vec::new();

    while available.len() > 1 {
        let mut best_i = 0;
        let mut best_j = 1;
        let mut best_cost = usize::MAX;

        for i in 0..available.len() {
            for j in (i + 1)..available.len() {
                let li = available[i];
                let lj = available[j];
                let cost = candidate_contraction_cost(
                    &operand_subs[li],
                    &operand_subs[lj],
                    li,
                    lj,
                    CandidateCostContext {
                        operand_label_sets: &operand_label_sets,
                        needed_label_counts: &needed_label_counts,
                        size_dict,
                    },
                    &mut candidate_subs,
                )?;
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

        let mut new_subs = Vec::new();
        collect_candidate_intermediate_subs(
            &operand_subs[left],
            &operand_subs[right],
            left,
            right,
            &operand_label_sets,
            &needed_label_counts,
            &mut new_subs,
        );
        let new_idx = operand_subs.len();
        let new_label_set: HashSet<u32> = new_subs.iter().copied().collect();
        remove_labels_from_counts(&mut needed_label_counts, &operand_label_sets[left]);
        remove_labels_from_counts(&mut needed_label_counts, &operand_label_sets[right]);
        add_labels_to_counts(&mut needed_label_counts, &new_label_set);
        operand_subs.push(new_subs);
        operand_label_sets.push(new_label_set);
        available.remove(best_j);
        available.remove(best_i);
        available.push(new_idx);
    }

    Ok(pairs)
}

#[cfg(test)]
mod tests;
