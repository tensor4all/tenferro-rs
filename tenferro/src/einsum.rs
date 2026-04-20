//! N-ary einsum with configurable contraction strategy.
//!
//! This module provides free functions [`einsum`] and [`einsum_with`]. They
//! build a lazy computation graph; call `.eval(&mut engine)` on the result to
//! trigger execution.
//!
//! # Quick start
//!
//! ```ignore
//! use tenferro::einsum::einsum;
//! use tenferro::engine::Engine;
//! use tenferro::traced::TracedTensor;
//!
//! let mut engine = Engine::new(CpuBackend::new());
//! let a = TracedTensor::from_tensor_concrete_shape(tensor_a);
//! let b = TracedTensor::from_tensor_concrete_shape(tensor_b);
//!
//! // Matrix multiply
//! let c = einsum(&mut engine, &[&a, &b], "ij,jk->ik");
//! let result = c.eval(&mut engine);
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use computegraph::fragment::FragmentBuilder;
use computegraph::types::ValRef;
use omeco::ScoreFunction;
use tenferro_einsum::builder::build_einsum_fragment;
use tenferro_einsum::{ContractionOptimizerOptions, ContractionTree, NestedEinsum, Subscripts};
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::TensorBackend;

use super::checkpoint::CheckpointNode;
use super::engine::Engine;
use super::error::{Error, Result};
use super::metadata::register_fragment_metadata;
use super::sym_dim::SymDim;
use super::traced::{concrete_shape, next_traced_id, try_concrete_shape, TracedTensor};

/// Controls how the contraction path is determined for N-ary einsum.
///
/// # Variants
///
/// ## `Auto` -- Automatic optimization (default: FLOPS-first)
///
/// Uses omeco's TreeSA optimizer. The default scoring prioritizes
/// time complexity (FLOPS). Customize via `ContractionOptimizerOptions`.
///
/// ```ignore
/// use omeco::ScoreFunction;
/// use tenferro_einsum::ContractionOptimizerOptions;
/// use tenferro::einsum::{einsum_with, EinsumOptimize};
///
/// // Default: FLOPS-first (minimize computation time)
/// einsum_with(&mut engine, &[&a, &b, &c], "ij,jk,kl->il",
///     EinsumOptimize::default());
///
/// // Space-optimized (minimize peak intermediate tensor size)
/// einsum_with(&mut engine, &[&a, &b, &c], "ij,jk,kl->il",
///     EinsumOptimize::Auto(ContractionOptimizerOptions {
///         score: ScoreFunction::space_optimized(20.0),
///         ..Default::default()
///     }));
///
/// // Balanced (FLOPS + space, omeco default)
/// einsum_with(&mut engine, &[&a, &b, &c], "ij,jk,kl->il",
///     EinsumOptimize::Auto(ContractionOptimizerOptions {
///         score: ScoreFunction::default(),
///         ..Default::default()
///     }));
///
/// // Custom: space-heavy with FLOPS tiebreaker
/// einsum_with(&mut engine, &[&a, &b, &c], "ij,jk,kl->il",
///     EinsumOptimize::Auto(ContractionOptimizerOptions {
///         score: ScoreFunction::new(
///             0.1,   // tc_weight (FLOPS, low priority)
///             1.0,   // sc_weight (space, high priority)
///             0.0,   // rw_weight (read-write, ignored)
///             15.0,  // sc_target (no penalty below 2^15 elements)
///         ),
///         ..Default::default()
///     }));
///
/// // Full TreeSA: multiple trials with annealing
/// einsum_with(&mut engine, &[&a, &b, &c], "ij,jk,kl->il",
///     EinsumOptimize::Auto(ContractionOptimizerOptions {
///         score: ScoreFunction::time_optimized(),
///         ntrials: 10,
///         niters: 50,
///         betas: vec![0.01, 0.1, 1.0, 10.0],
///         ..Default::default()
///     }));
/// ```
///
/// ## `False` -- No optimization
///
/// Contracts operands left-to-right in the order given.
/// Useful for debugging or when the input order is already optimal.
///
/// ```ignore
/// einsum_with(&mut engine, &[&a, &b, &c], "ij,jk,kl->il",
///     EinsumOptimize::False);
/// ```
///
/// ## `Nested` -- Parenthesized notation
///
/// Specifies contraction order using a pre-parsed [`NestedEinsum`] tree.
/// Most human-readable way to control order.
///
/// ```ignore
/// use tenferro_einsum::NestedEinsum;
///
/// // "Contract A*B first, then result with C"
/// einsum_with(&mut engine, &[&a, &b, &c], "ij,jk,kl->il",
///     EinsumOptimize::Nested(NestedEinsum::parse("(ij,jk),kl->il").unwrap()));
///
/// // "Contract B*C first, then A with result"
/// einsum_with(&mut engine, &[&a, &b, &c], "ij,jk,kl->il",
///     EinsumOptimize::Nested(NestedEinsum::parse("ij,(jk,kl)->il").unwrap()));
/// ```
///
/// ## `Path` -- JAX-compatible explicit path
///
/// Each pair specifies positions in a shrinking operand list.
/// After each step, the two contracted operands are removed and
/// the result is appended to the end.
///
/// Compatible with `jax.numpy.einsum(optimize=path)` and
/// `opt_einsum.contract_path` output.
///
/// ```ignore
/// // 3 operands: A(0), B(1), C(2)
/// // Step 1: contract positions 1,2 (B,C) -> T. List: [A, T]
/// // Step 2: contract positions 0,1 (A,T) -> result
/// einsum_with(&mut engine, &[&a, &b, &c], "ij,jk,kl->il",
///     EinsumOptimize::Path(vec![(1, 2), (0, 1)]));
///
/// // Step 1: contract positions 0,1 (A,B) -> T. List: [C, T]
/// // Step 2: contract positions 0,1 (C,T) -> result
/// einsum_with(&mut engine, &[&a, &b, &c], "ij,jk,kl->il",
///     EinsumOptimize::Path(vec![(0, 1), (0, 1)]));
/// ```
///
/// ## `Tree` -- Pre-computed ContractionTree
///
/// Pass a tree obtained from `ContractionTree::optimize` or other
/// optimization tools. Skips all path computation.
///
/// ```ignore
/// use tenferro_einsum::{ContractionTree, Subscripts};
///
/// let subs = Subscripts::parse("ij,jk,kl->il").unwrap();
/// let shapes = [&[2, 3][..], &[3, 4], &[4, 5]];
/// let tree = ContractionTree::optimize(&subs, &shapes).unwrap();
/// einsum_with(&mut engine, &[&a, &b, &c], "ij,jk,kl->il",
///     EinsumOptimize::Tree(tree));
/// ```
pub enum EinsumOptimize {
    /// Automatic optimization via omeco TreeSA.
    Auto(ContractionOptimizerOptions),
    /// No optimization -- contract left-to-right.
    False,
    /// Parenthesized notation specifying contraction order.
    Nested(NestedEinsum),
    /// JAX-compatible position-based contraction path.
    Path(Vec<(usize, usize)>),
    /// Pre-computed contraction tree.
    Tree(ContractionTree),
}

impl Default for EinsumOptimize {
    /// Default: FLOPS-first automatic optimization.
    ///
    /// Uses `ScoreFunction::time_optimized()`:
    /// - `tc_weight = 1.0` (minimize FLOPS)
    /// - `sc_weight = 0.0` (ignore space)
    fn default() -> Self {
        EinsumOptimize::Auto(ContractionOptimizerOptions {
            score: ScoreFunction::time_optimized(),
            ..Default::default()
        })
    }
}

/// N-ary einsum with default FLOPS-first optimization.
///
/// Builds a lazy computation graph. Call `.eval(&mut engine)` on the
/// result to trigger execution.
///
/// # Examples
///
/// ```ignore
/// use tenferro::einsum::einsum;
/// use tenferro::engine::Engine;
/// use tenferro::traced::TracedTensor;
///
/// // Matrix multiply
/// let c = einsum(&mut engine, &[&a, &b], "ij,jk->ik");
///
/// // 3-tensor chain multiply
/// let d = einsum(&mut engine, &[&a, &b, &c], "ij,jk,kl->il");
///
/// // Inner product
/// let s = einsum(&mut engine, &[&x, &y], "i,i->");
///
/// // Row sum (unary)
/// let r = einsum(&mut engine, &[&a], "ij->i");
///
/// // Hadamard product
/// let h = einsum(&mut engine, &[&a, &b], "ij,ij->ij");
///
/// // Outer product
/// let o = einsum(&mut engine, &[&x, &y], "i,j->ij");
/// ```
pub fn einsum<B: TensorBackend>(
    engine: &mut Engine<B>,
    inputs: &[&TracedTensor],
    subscripts: &str,
) -> Result<TracedTensor> {
    einsum_with(engine, inputs, subscripts, EinsumOptimize::default())
}

/// N-ary einsum with explicit contraction strategy.
///
/// See [`EinsumOptimize`] for all available strategies and examples.
///
/// # Examples
///
/// ```ignore
/// use tenferro::einsum::{einsum_with, EinsumOptimize};
/// use tenferro::engine::Engine;
/// use tenferro::traced::TracedTensor;
///
/// // Left-to-right, no optimizer
/// let c = einsum_with(&mut engine, &[&a, &b, &c], "ij,jk,kl->il",
///     EinsumOptimize::False);
///
/// // JAX-compatible explicit path
/// let c = einsum_with(&mut engine, &[&a, &b, &c], "ij,jk,kl->il",
///     EinsumOptimize::Path(vec![(1, 2), (0, 1)]));
/// ```
pub fn einsum_with<B: TensorBackend>(
    engine: &mut Engine<B>,
    inputs: &[&TracedTensor],
    subscripts: &str,
    optimize: EinsumOptimize,
) -> Result<TracedTensor> {
    if inputs.is_empty() {
        return Err(Error::ContractionError(
            "einsum requires at least one input tensor".into(),
        ));
    }

    let subs =
        Subscripts::parse(subscripts).map_err(|e| Error::InvalidSubscripts(format!("{e}")))?;
    if subs.inputs.len() != inputs.len() {
        return Err(Error::ContractionError(format!(
            "einsum subscripts expect {} inputs, got {}",
            subs.inputs.len(),
            inputs.len()
        )));
    }
    if inputs
        .iter()
        .any(|tensor| try_concrete_shape(tensor).is_none())
    {
        return Ok(build_symbolic_nary_einsum(inputs, subscripts, &subs));
    }
    let shapes: Vec<Vec<usize>> = inputs.iter().map(|t| concrete_shape(t)).collect();
    let shape_refs: Vec<&[usize]> = shapes.iter().map(|s| s.as_slice()).collect();

    match optimize {
        // Reuse TreeSA results for repeated calls with the same equation and input shapes.
        EinsumOptimize::Auto(opts) => {
            let cache_key = (subscripts.to_string(), shapes.clone());
            let tree = if let Some(cached) = engine.einsum_cache.get(&cache_key) {
                cached.clone()
            } else {
                let tree = Arc::new(resolve_strategy(
                    EinsumOptimize::Auto(opts),
                    &subs,
                    &shape_refs,
                )?);
                engine.einsum_cache.put(cache_key, tree.clone());
                tree
            };
            Ok(build_traced_from_tree(
                inputs,
                &subs,
                tree.as_ref(),
                &shapes,
            ))
        }
        optimize => {
            let tree = resolve_strategy(optimize, &subs, &shape_refs)?;
            Ok(build_traced_from_tree(inputs, &subs, &tree, &shapes))
        }
    }
}

fn build_symbolic_nary_einsum(
    inputs: &[&TracedTensor],
    subscripts: &str,
    parsed: &Subscripts,
) -> TracedTensor {
    let mut builder = FragmentBuilder::new();
    let mut input_vals = Vec::with_capacity(inputs.len());
    let mut merged = HashMap::new();
    let mut extra_roots = Vec::new();

    for input in inputs {
        builder.add_parent(input.fragment.clone());
        input_vals.push(ValRef::External(
            input.fragment.vals()[input.val].key.clone(),
        ));
        merged.extend(
            input
                .inputs_map
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
        extra_roots.extend(input.extra_roots.iter().cloned());
    }

    let outputs = builder.add_op(
        StdTensorOp::NaryEinsum {
            subscripts: subscripts.to_string(),
        },
        input_vals,
        computegraph::types::OpMode::Primal,
    );
    builder.set_outputs(outputs.clone());
    let fragment = Arc::new(builder.build());
    register_fragment_metadata(fragment.as_ref(), std::iter::empty());

    TracedTensor {
        id: next_traced_id(),
        rank: parsed.output.len(),
        dtype: inputs[0].dtype,
        fragment,
        val: outputs[0],
        data: None,
        shape_hint: None,
        inputs_map: Arc::new(merged),
        extra_roots,
        checkpoint_chain: None,
    }
}

/// Resolve an [`EinsumOptimize`] strategy to a [`ContractionTree`].
fn resolve_strategy(
    optimize: EinsumOptimize,
    subs: &Subscripts,
    shapes: &[&[usize]],
) -> Result<ContractionTree> {
    match optimize {
        EinsumOptimize::Auto(opts) => ContractionTree::optimize_with_options(subs, shapes, &opts)
            .map_err(|e| Error::ContractionError(format!("{e}"))),
        EinsumOptimize::False => {
            let n = subs.inputs.len();
            if n <= 1 {
                ContractionTree::from_pairs(subs, shapes, &[])
                    .map_err(|e| Error::ContractionError(format!("{e}")))
            } else {
                let jax_path: Vec<(usize, usize)> = (0..n - 1).map(|_| (0, 1)).collect();
                let v1_pairs = jax_path_to_v1_pairs(&jax_path, n);
                ContractionTree::from_pairs(subs, shapes, &v1_pairs)
                    .map_err(|e| Error::ContractionError(format!("{e}")))
            }
        }
        EinsumOptimize::Nested(nested) => {
            let n = subs.inputs.len();
            let v1_pairs = nested_to_v1_pairs(&nested, n);
            ContractionTree::from_pairs(subs, shapes, &v1_pairs)
                .map_err(|e| Error::ContractionError(format!("{e}")))
        }
        EinsumOptimize::Path(jax_path) => {
            let n = subs.inputs.len();
            let v1_pairs = jax_path_to_v1_pairs(&jax_path, n);
            ContractionTree::from_pairs(subs, shapes, &v1_pairs)
                .map_err(|e| Error::ContractionError(format!("{e}")))
        }
        EinsumOptimize::Tree(tree) => Ok(tree),
    }
}

/// Convert JAX-style position-based path to v1 fixed-ID pairs.
///
/// JAX format: each pair `(i, j)` refers to positions in a shrinking list.
/// After contraction, the two operands are removed (higher index first)
/// and the result is appended at the end.
///
/// v1 format: inputs are `0..n`, intermediate at step `k` has ID `n + k`.
fn jax_path_to_v1_pairs(jax_path: &[(usize, usize)], n_inputs: usize) -> Vec<(usize, usize)> {
    // Track which original/intermediate IDs are at each position
    let mut positions: Vec<usize> = (0..n_inputs).collect();
    let mut v1_pairs = Vec::new();

    for (step, &(pos_a, pos_b)) in jax_path.iter().enumerate() {
        let (lo, hi) = if pos_a < pos_b {
            (pos_a, pos_b)
        } else {
            (pos_b, pos_a)
        };
        let id_a = positions[lo];
        let id_b = positions[hi];
        v1_pairs.push((id_a, id_b));

        // Remove higher index first, then lower
        positions.remove(hi);
        positions.remove(lo);
        // Append new intermediate ID
        positions.push(n_inputs + step);
    }

    v1_pairs
}

/// Convert a [`NestedEinsum`] tree into v1 fixed-ID pairs.
///
/// Walks the tree bottom-up. Each `Leaf(i)` maps to original input `i`.
/// Each binary `Node` emits a pair `(left_id, right_id)` and is assigned
/// the next intermediate ID (`n_inputs + step`).
fn nested_to_v1_pairs(nested: &NestedEinsum, n_inputs: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    let mut next_id = n_inputs;
    walk_nested(nested, &mut pairs, &mut next_id);
    pairs
}

/// Recursive walk of `NestedEinsum` that emits v1-style pairs.
///
/// Returns the operand ID for this sub-expression (either a leaf input index
/// or an intermediate ID).
fn walk_nested(
    nested: &NestedEinsum,
    pairs: &mut Vec<(usize, usize)>,
    next_id: &mut usize,
) -> usize {
    match nested {
        NestedEinsum::Leaf(idx) => *idx,
        NestedEinsum::Node { children, .. } => {
            // For binary nodes (the normal case), contract the two children.
            // For N-ary nodes (N > 2), contract left-to-right.
            assert!(
                !children.is_empty(),
                "NestedEinsum::Node must have at least one child"
            );
            let mut result_id = walk_nested(&children[0], pairs, next_id);
            for child in &children[1..] {
                let child_id = walk_nested(child, pairs, next_id);
                pairs.push((result_id, child_id));
                result_id = *next_id;
                *next_id += 1;
            }
            result_id
        }
    }
}

/// Build a [`TracedTensor`] from a contraction tree and inputs.
fn build_traced_from_tree(
    inputs: &[&TracedTensor],
    subscripts: &Subscripts,
    tree: &ContractionTree,
    shapes: &[Vec<usize>],
) -> TracedTensor {
    let out_shape = compute_einsum_output_shape(subscripts, shapes);

    let mut builder = FragmentBuilder::new();

    // Add parents and create ValRef for each input
    let mut input_vals = Vec::new();
    for input in inputs {
        builder.add_parent(input.fragment.clone());
        let val_ref = ValRef::External(input.fragment.vals()[input.val].key.clone());
        input_vals.push(val_ref);
    }

    let result_ref = build_einsum_fragment(&mut builder, tree, &input_vals, shapes);

    match result_ref {
        ValRef::Local(result_local) => {
            builder.set_outputs(vec![result_local]);
            let fragment = Arc::new(builder.build());
            register_fragment_metadata(fragment.as_ref(), std::iter::empty());

            let mut merged = HashMap::new();
            let mut extra_roots = Vec::new();
            for input in inputs {
                merged.extend(input.inputs_map.iter().map(|(k, v)| (k.clone(), v.clone())));
                extra_roots.extend(input.extra_roots.iter().cloned());
            }

            let merged_chain = inputs.iter().fold(None, |acc, input| {
                CheckpointNode::merge_chains(acc, input.checkpoint_chain.clone())
            });

            TracedTensor {
                id: next_traced_id(),
                rank: out_shape.len(),
                dtype: inputs[0].dtype,
                fragment,
                val: result_local,
                data: None,
                shape_hint: Some(out_shape.into_iter().map(SymDim::from).collect()),
                inputs_map: Arc::new(merged),
                extra_roots,
                checkpoint_chain: merged_chain,
            }
        }
        ValRef::External(_) => {
            // Identity pass-through: the einsum doesn't add any ops.
            // Find which input was returned and clone its TracedTensor.
            for (i, iv) in input_vals.iter().enumerate() {
                if *iv == result_ref {
                    return TracedTensor {
                        id: next_traced_id(),
                        rank: out_shape.len(),
                        dtype: inputs[i].dtype,
                        fragment: inputs[i].fragment.clone(),
                        val: inputs[i].val,
                        data: inputs[i].data.clone(),
                        shape_hint: Some(out_shape.into_iter().map(SymDim::from).collect()),
                        inputs_map: inputs[i].inputs_map.clone(),
                        extra_roots: inputs[i].extra_roots.clone(),
                        checkpoint_chain: inputs[i].checkpoint_chain.clone(),
                    };
                }
            }
            panic!("build_einsum_fragment returned unrecognized external ref");
        }
    }
}

/// Compute the output shape from einsum subscripts and input shapes.
fn compute_einsum_output_shape(subscripts: &Subscripts, shapes: &[Vec<usize>]) -> Vec<usize> {
    let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
    let size_dict = tenferro_einsum::build_size_dict(subscripts, &shape_refs, None)
        .unwrap_or_else(|err| panic!("einsum shape computation failed: {err}"));
    tenferro_einsum::compute_output_shape(&subscripts.output, &size_dict)
        .unwrap_or_else(|err| panic!("einsum output shape computation failed: {err}"))
}
