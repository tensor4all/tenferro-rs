use std::hash::Hasher;

use omeco::ScoreFunction;

use crate::{
    ContractionOptimizerOptions, ContractionTree, Error, NestedEinsum, Result, Subscripts,
};

/// Controls how the contraction path is determined for N-ary einsum.
///
/// The traced API resolves this enum into a shape-independent plan
/// specification and stores that specification in the einsum extension
/// payload. Concrete traced inputs can resolve a [`ContractionTree`] while the
/// op is built; symbolic traced inputs carry the plan specification until the
/// extension runtime sees concrete execution shapes.
///
/// Planner options and explicit paths are part of the extension payload
/// identity. Two otherwise identical traced einsum ops with different
/// optimizer options, explicit paths, or fixed plan identities are not treated
/// as the same extension op, and their compile/runtime plan cache entries are
/// kept separate.
pub enum EinsumOptimize {
    /// Automatic optimization via omeco TreeSA.
    Auto(ContractionOptimizerOptions),
    /// No optimization: contract operands left-to-right.
    False,
    /// Parenthesized notation specifying contraction order.
    Nested(NestedEinsum),
    /// JAX-compatible position-based contraction path.
    ///
    /// Each pair references positions in a shrinking operand list. After each
    /// contraction, the two operands are removed and the result is appended.
    /// Because this representation is independent of concrete dimension
    /// values, it can be used with symbolic traced inputs.
    Path(Vec<(usize, usize)>),
    /// Pre-computed contraction tree.
    ///
    /// A tree contains concrete shape-dependent planning results. It is
    /// accepted when shapes are concrete, then converted into fixed contraction
    /// pairs for the extension payload. Use [`EinsumOptimize::Path`] instead
    /// when building a traced graph from symbolic inputs.
    Tree(ContractionTree),
}

impl Default for EinsumOptimize {
    /// Default: time-optimized automatic planning.
    fn default() -> Self {
        Self::Auto(default_auto_options())
    }
}

#[derive(Clone, Debug)]
pub(crate) enum EinsumPlanSpec {
    Auto(ContractionOptimizerOptions),
    LeftToRight,
    Path(Vec<(usize, usize)>),
    FixedPairs(Vec<(usize, usize)>),
}

/// Return the default automatic optimizer options.
#[must_use]
pub(crate) fn default_auto_options() -> ContractionOptimizerOptions {
    ContractionOptimizerOptions {
        score: ScoreFunction::time_optimized(),
        ..Default::default()
    }
}

pub(crate) fn plan_spec_from_optimize(
    optimize: EinsumOptimize,
    subscripts: &Subscripts,
) -> Result<EinsumPlanSpec> {
    match optimize {
        EinsumOptimize::Auto(options) => {
            options.validate()?;
            Ok(EinsumPlanSpec::Auto(options))
        }
        EinsumOptimize::False => Ok(EinsumPlanSpec::LeftToRight),
        EinsumOptimize::Nested(nested) => {
            let pairs = nested_to_v1_pairs(&nested, subscripts.inputs.len())?;
            validate_fixed_pairs(&pairs, subscripts.inputs.len())?;
            Ok(EinsumPlanSpec::FixedPairs(pairs))
        }
        EinsumOptimize::Path(path) => {
            let _ = jax_path_to_v1_pairs(&path, subscripts.inputs.len())?;
            Ok(EinsumPlanSpec::Path(path))
        }
        EinsumOptimize::Tree(_) => Err(Error::InvalidArgument(
            "precomputed contraction tree requires concrete input shapes; use Path or parenthesized notation for symbolic traced einsum"
                .into(),
        )),
    }
}

pub(crate) fn resolve_einsum_strategy_with_spec(
    optimize: EinsumOptimize,
    subscripts: &Subscripts,
    shapes: &[&[usize]],
) -> Result<(EinsumPlanSpec, ContractionTree)> {
    match optimize {
        EinsumOptimize::Tree(tree) => {
            let pairs = tree_pairs(&tree);
            let spec = EinsumPlanSpec::FixedPairs(pairs);
            let tree = resolve_plan_spec(&spec, subscripts, shapes)?;
            Ok((spec, tree))
        }
        optimize => {
            let spec = plan_spec_from_optimize(optimize, subscripts)?;
            let tree = resolve_plan_spec(&spec, subscripts, shapes)?;
            Ok((spec, tree))
        }
    }
}

pub(crate) fn resolve_plan_spec(
    spec: &EinsumPlanSpec,
    subscripts: &Subscripts,
    shapes: &[&[usize]],
) -> Result<ContractionTree> {
    match spec {
        EinsumPlanSpec::Auto(options) => {
            ContractionTree::optimize_with_options(subscripts, shapes, options)
        }
        EinsumPlanSpec::LeftToRight => {
            let n = subscripts.inputs.len();
            if n <= 1 {
                ContractionTree::from_pairs(subscripts, shapes, &[])
            } else {
                let path: Vec<(usize, usize)> = (0..n - 1).map(|_| (0, 1)).collect();
                let pairs = jax_path_to_v1_pairs(&path, n)?;
                ContractionTree::from_pairs(subscripts, shapes, &pairs)
            }
        }
        EinsumPlanSpec::Path(path) => {
            let pairs = jax_path_to_v1_pairs(path, subscripts.inputs.len())?;
            ContractionTree::from_pairs(subscripts, shapes, &pairs)
        }
        EinsumPlanSpec::FixedPairs(pairs) => ContractionTree::from_pairs(subscripts, shapes, pairs),
    }
}

pub(crate) fn hash_einsum_plan_spec(spec: &EinsumPlanSpec, state: &mut dyn Hasher) {
    match spec {
        EinsumPlanSpec::Auto(options) => {
            state.write_u8(0);
            hash_optimizer_options(options, state);
        }
        EinsumPlanSpec::LeftToRight => state.write_u8(1),
        EinsumPlanSpec::Path(path) => {
            state.write_u8(2);
            hash_pairs(path, state);
        }
        EinsumPlanSpec::FixedPairs(pairs) => {
            state.write_u8(3);
            hash_pairs(pairs, state);
        }
    }
}

pub(crate) fn plan_specs_equal(lhs: &EinsumPlanSpec, rhs: &EinsumPlanSpec) -> bool {
    match (lhs, rhs) {
        (EinsumPlanSpec::Auto(lhs), EinsumPlanSpec::Auto(rhs)) => {
            optimizer_options_equal_by_bits(lhs, rhs)
        }
        (EinsumPlanSpec::LeftToRight, EinsumPlanSpec::LeftToRight) => true,
        (EinsumPlanSpec::Path(lhs), EinsumPlanSpec::Path(rhs)) => lhs == rhs,
        (EinsumPlanSpec::FixedPairs(lhs), EinsumPlanSpec::FixedPairs(rhs)) => lhs == rhs,
        _ => false,
    }
}

fn tree_pairs(tree: &ContractionTree) -> Vec<(usize, usize)> {
    (0..tree.step_count())
        .filter_map(|step| tree.step_pair(step))
        .collect()
}

fn validate_fixed_pairs(pairs: &[(usize, usize)], n_inputs: usize) -> Result<()> {
    let required_steps = n_inputs.saturating_sub(1);
    if pairs.len() != required_steps {
        return Err(Error::InvalidArgument(format!(
            "explicit contraction path for {n_inputs} operands must have {required_steps} steps, got {}",
            pairs.len()
        )));
    }

    let mut live = vec![false; n_inputs + pairs.len()];
    for slot in live.iter_mut().take(n_inputs) {
        *slot = true;
    }

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

        live[left] = false;
        live[right] = false;
        live[next_idx] = true;
    }

    let live_count = live.iter().filter(|&&is_live| is_live).count();
    if live_count != 1 {
        return Err(Error::InvalidArgument(format!(
            "explicit contraction path must leave exactly one live result, got {live_count}"
        )));
    }

    Ok(())
}

fn hash_pairs(pairs: &[(usize, usize)], state: &mut dyn Hasher) {
    state.write_usize(pairs.len());
    for &(left, right) in pairs {
        state.write_usize(left);
        state.write_usize(right);
    }
}

fn hash_optimizer_options(options: &ContractionOptimizerOptions, state: &mut dyn Hasher) {
    state.write_usize(options.ntrials);
    state.write_usize(options.niters);
    state.write_usize(options.betas.len());
    for value in &options.betas {
        state.write_u64(value.to_bits());
    }
    state.write_u64(options.score.tc_weight.to_bits());
    state.write_u64(options.score.sc_weight.to_bits());
    state.write_u64(options.score.rw_weight.to_bits());
    state.write_u64(options.score.sc_target.to_bits());
}

fn optimizer_options_equal_by_bits(
    lhs: &ContractionOptimizerOptions,
    rhs: &ContractionOptimizerOptions,
) -> bool {
    lhs.ntrials == rhs.ntrials
        && lhs.niters == rhs.niters
        && f64_slices_equal_by_bits(&lhs.betas, &rhs.betas)
        && score_functions_equal_by_bits(&lhs.score, &rhs.score)
}

/// Convert JAX-style position-based path to fixed-ID pairs.
///
/// JAX format: each pair `(i, j)` refers to positions in a shrinking list.
/// After contraction, the two operands are removed and the result is appended.
/// Fixed-ID format keeps original operands at `0..n_inputs` and gives
/// intermediate at step `k` the ID `n_inputs + k`.
///
/// # Errors
///
/// Returns an error if a path step references the same position twice or a
/// position outside the current shrinking list.
pub(crate) fn jax_path_to_v1_pairs(
    jax_path: &[(usize, usize)],
    n_inputs: usize,
) -> Result<Vec<(usize, usize)>> {
    let required_steps = n_inputs.saturating_sub(1);
    if jax_path.len() != required_steps {
        return Err(Error::InvalidArgument(format!(
            "explicit contraction path for {n_inputs} operands must have {required_steps} steps, got {}",
            jax_path.len()
        )));
    }

    let mut positions: Vec<usize> = (0..n_inputs).collect();
    let mut v1_pairs = Vec::with_capacity(jax_path.len());

    for (step, &(pos_a, pos_b)) in jax_path.iter().enumerate() {
        if pos_a == pos_b {
            return Err(Error::InvalidArgument(format!(
                "path step {step} references the same operand position twice: {pos_a}"
            )));
        }
        let current_len = positions.len();
        if pos_a >= current_len || pos_b >= current_len {
            return Err(Error::InvalidArgument(format!(
                "path step {step} references operand positions ({pos_a}, {pos_b}) with only {current_len} live operands"
            )));
        }

        let (lo, hi) = if pos_a < pos_b {
            (pos_a, pos_b)
        } else {
            (pos_b, pos_a)
        };
        let id_a = positions[lo];
        let id_b = positions[hi];
        v1_pairs.push((id_a, id_b));

        positions.remove(hi);
        positions.remove(lo);
        positions.push(n_inputs + step);
    }

    Ok(v1_pairs)
}

/// Convert a [`NestedEinsum`] tree into fixed-ID pairs.
///
/// # Errors
///
/// Returns an error if a leaf references an input outside `0..n_inputs` or if
/// a node has no children.
pub(crate) fn nested_to_v1_pairs(
    nested: &NestedEinsum,
    n_inputs: usize,
) -> Result<Vec<(usize, usize)>> {
    let mut pairs = Vec::with_capacity(n_inputs.saturating_sub(1));
    let mut next_id = n_inputs;
    let root_id = walk_nested(nested, n_inputs, &mut pairs, &mut next_id)?;
    if n_inputs == 0 || root_id >= next_id {
        return Err(Error::InvalidArgument(
            "nested einsum did not produce a valid root operand".into(),
        ));
    }
    Ok(pairs)
}

fn walk_nested(
    nested: &NestedEinsum,
    n_inputs: usize,
    pairs: &mut Vec<(usize, usize)>,
    next_id: &mut usize,
) -> Result<usize> {
    match nested {
        NestedEinsum::Leaf(idx) => {
            if *idx >= n_inputs {
                return Err(Error::InvalidArgument(format!(
                    "nested einsum leaf {idx} is outside 0..{n_inputs}"
                )));
            }
            Ok(*idx)
        }
        NestedEinsum::Node { children, .. } => {
            let Some(first) = children.first() else {
                return Err(Error::InvalidArgument(
                    "nested einsum node must have at least one child".into(),
                ));
            };
            let mut result_id = walk_nested(first, n_inputs, pairs, next_id)?;
            for child in &children[1..] {
                let child_id = walk_nested(child, n_inputs, pairs, next_id)?;
                pairs.push((result_id, child_id));
                result_id = *next_id;
                *next_id += 1;
            }
            Ok(result_id)
        }
    }
}

fn f64_slices_equal_by_bits(lhs: &[f64], rhs: &[f64]) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
}

fn score_functions_equal_by_bits(lhs: &ScoreFunction, rhs: &ScoreFunction) -> bool {
    lhs.tc_weight.to_bits() == rhs.tc_weight.to_bits()
        && lhs.sc_weight.to_bits() == rhs.sc_weight.to_bits()
        && lhs.rw_weight.to_bits() == rhs.rw_weight.to_bits()
        && lhs.sc_target.to_bits() == rhs.sc_target.to_bits()
}

#[cfg(test)]
mod tests;
