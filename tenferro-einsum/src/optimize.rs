use omeco::ScoreFunction;
use tenferro_device::{Error, Result};

use crate::{ContractionOptimizerOptions, ContractionTree, NestedEinsum, Subscripts};

/// Controls how the contraction path is determined for N-ary einsum.
///
/// This is the runtime-independent strategy enum from the current traced
/// implementation. Frontends can resolve it to a [`ContractionTree`] once
/// concrete input shapes are available.
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
    Path(Vec<(usize, usize)>),
    /// Pre-computed contraction tree.
    Tree(ContractionTree),
}

impl Default for EinsumOptimize {
    /// Default: FLOPS-first automatic optimization.
    fn default() -> Self {
        Self::Auto(ContractionOptimizerOptions {
            score: ScoreFunction::time_optimized(),
            ..Default::default()
        })
    }
}

/// Return the default automatic optimizer options.
#[must_use]
pub(crate) fn default_auto_options() -> ContractionOptimizerOptions {
    match EinsumOptimize::default() {
        EinsumOptimize::Auto(options) => options,
        _ => unreachable!("EinsumOptimize::default must be automatic optimization"),
    }
}

/// Compare optimizer options with the default policy using bitwise float
/// equality.
#[must_use]
pub(crate) fn is_default_auto_options(options: &ContractionOptimizerOptions) -> bool {
    let default = default_auto_options();
    options.ntrials == default.ntrials
        && options.niters == default.niters
        && f64_slices_equal_by_bits(&options.betas, &default.betas)
        && score_functions_equal_by_bits(&options.score, &default.score)
}

/// Resolve an [`EinsumOptimize`] strategy to a concrete contraction tree.
///
/// # Errors
///
/// Returns an error if subscripts and shapes are inconsistent or if an
/// explicit path references invalid operands.
pub(crate) fn resolve_einsum_strategy(
    optimize: EinsumOptimize,
    subscripts: &Subscripts,
    shapes: &[&[usize]],
) -> Result<ContractionTree> {
    match optimize {
        EinsumOptimize::Auto(opts) => {
            ContractionTree::optimize_with_options(subscripts, shapes, &opts)
        }
        EinsumOptimize::False => {
            let n = subscripts.inputs.len();
            if n <= 1 {
                ContractionTree::from_pairs(subscripts, shapes, &[])
            } else {
                let jax_path: Vec<(usize, usize)> = (0..n - 1).map(|_| (0, 1)).collect();
                let v1_pairs = jax_path_to_v1_pairs(&jax_path, n)?;
                ContractionTree::from_pairs(subscripts, shapes, &v1_pairs)
            }
        }
        EinsumOptimize::Nested(nested) => {
            let n = subscripts.inputs.len();
            let v1_pairs = nested_to_v1_pairs(&nested, n)?;
            ContractionTree::from_pairs(subscripts, shapes, &v1_pairs)
        }
        EinsumOptimize::Path(jax_path) => {
            let n = subscripts.inputs.len();
            let v1_pairs = jax_path_to_v1_pairs(&jax_path, n)?;
            ContractionTree::from_pairs(subscripts, shapes, &v1_pairs)
        }
        EinsumOptimize::Tree(tree) => Ok(tree),
    }
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
