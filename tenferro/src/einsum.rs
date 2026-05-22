//! N-ary einsum with configurable contraction strategy.
//!
//! This module provides free functions [`einsum`] and [`einsum_with`]. They
//! build a lazy computation graph; compile it with [`GraphCompiler`] and run
//! the compiled program with [`crate::GraphExecutor`].
//!
//! # Quick start
//!
//! ```
//! use tenferro::traced_tensor::einsum;
//! use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
//!
//! let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
//! let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);
//! let mut compiler = GraphCompiler::new();
//!
//! // Matrix multiply
//! let c = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();
//! let program = compiler.compile(&c).unwrap();
//! let result = GraphExecutor::new(CpuBackend::new()).run(&program).unwrap();
//! assert_eq!(result.shape(), &[2, 2]);
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use computegraph::fragment::FragmentBuilder;
use computegraph::types::ValRef;
use omeco::ScoreFunction;
use tenferro_einsum::builder::build_einsum_fragment;
use tenferro_einsum::{ContractionOptimizerOptions, ContractionTree, NestedEinsum, Subscripts};
use tenferro_ops::std_tensor_op::{EinsumSubscripts, StdTensorOp};

use super::checkpoint::CheckpointNode;
use super::einsum_subscripts::to_einsum_subscripts;
use super::error::{Error, Result};
use super::graph::GraphCompiler;
use super::metadata::{metadata_scopes_with_new, register_scoped_fragment_metadata};
use super::sym_dim::SymDim;
use super::traced::{concrete_shape, next_traced_id, TracedTensor};

/// Controls how the contraction path is determined for N-ary einsum.
///
/// # Variants
///
/// ## `Auto` -- Automatic optimization (default: FLOPS-first)
///
/// Uses omeco's TreeSA optimizer. The default scoring prioritizes
/// time complexity (FLOPS). Customize via `ContractionOptimizerOptions`.
///
/// ```rust
/// use omeco::ScoreFunction;
/// use tenferro_einsum::ContractionOptimizerOptions;
/// use tenferro::{GraphCompiler, TracedTensor};
/// use tenferro::traced_tensor::{einsum_with, EinsumOptimize};
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
/// let b = TracedTensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]);
/// let c = TracedTensor::from_vec_col_major(vec![4, 5], vec![1.0_f64; 20]);
/// let mut compiler = GraphCompiler::new();
///
/// // Default: FLOPS-first (minimize computation time)
/// einsum_with(&mut compiler, &[&a, &b, &c], "ij,jk,kl->il",
///     EinsumOptimize::default()).unwrap();
///
/// // Space-optimized (minimize peak intermediate tensor size)
/// einsum_with(&mut compiler, &[&a, &b, &c], "ij,jk,kl->il",
///     EinsumOptimize::Auto(ContractionOptimizerOptions {
///         score: ScoreFunction::space_optimized(20.0),
///         ..Default::default()
///     })).unwrap();
///
/// // Balanced (FLOPS + space, omeco default)
/// einsum_with(&mut compiler, &[&a, &b, &c], "ij,jk,kl->il",
///     EinsumOptimize::Auto(ContractionOptimizerOptions {
///         score: ScoreFunction::default(),
///         ..Default::default()
///     })).unwrap();
///
/// // Custom: space-heavy with FLOPS tiebreaker
/// einsum_with(&mut compiler, &[&a, &b, &c], "ij,jk,kl->il",
///     EinsumOptimize::Auto(ContractionOptimizerOptions {
///         score: ScoreFunction::new(
///             0.1,   // tc_weight (FLOPS, low priority)
///             1.0,   // sc_weight (space, high priority)
///             0.0,   // rw_weight (read-write, ignored)
///             15.0,  // sc_target (no penalty below 2^15 elements)
///         ),
///         ..Default::default()
///     })).unwrap();
///
/// // Full TreeSA: multiple trials with annealing
/// einsum_with(&mut compiler, &[&a, &b, &c], "ij,jk,kl->il",
///     EinsumOptimize::Auto(ContractionOptimizerOptions {
///         score: ScoreFunction::time_optimized(),
///         ntrials: 10,
///         niters: 50,
///         betas: vec![0.01, 0.1, 1.0, 10.0],
///         ..Default::default()
///     })).unwrap();
/// ```
///
/// ## `False` -- No optimization
///
/// Contracts operands left-to-right in the order given.
/// Useful for debugging or when the input order is already optimal.
///
/// ```rust
/// use tenferro::{GraphCompiler, TracedTensor};
/// use tenferro::traced_tensor::{einsum_with, EinsumOptimize};
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
/// let b = TracedTensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]);
/// let c = TracedTensor::from_vec_col_major(vec![4, 5], vec![1.0_f64; 20]);
/// let mut compiler = GraphCompiler::new();
///
/// let out = einsum_with(
///     &mut compiler,
///     &[&a, &b, &c],
///     "ij,jk,kl->il",
///     EinsumOptimize::False,
/// )
/// .unwrap();
/// assert_eq!(out.rank, 2);
/// ```
///
/// ## `Nested` -- Parenthesized notation
///
/// Specifies contraction order using a pre-parsed [`NestedEinsum`] tree.
/// Most human-readable way to control order.
///
/// ```rust
/// use tenferro::{GraphCompiler, TracedTensor};
/// use tenferro::traced_tensor::{einsum_with, EinsumOptimize};
/// use tenferro_einsum::NestedEinsum;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
/// let b = TracedTensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]);
/// let c = TracedTensor::from_vec_col_major(vec![4, 5], vec![1.0_f64; 20]);
/// let mut compiler = GraphCompiler::new();
///
/// // "Contract A*B first, then result with C"
/// let out = einsum_with(
///     &mut compiler,
///     &[&a, &b, &c],
///     "ij,jk,kl->il",
///     EinsumOptimize::Nested(NestedEinsum::parse("(ij,jk),kl->il").unwrap()),
/// )
/// .unwrap();
/// assert_eq!(out.rank, 2);
///
/// // "Contract B*C first, then A with result"
/// let out = einsum_with(
///     &mut compiler,
///     &[&a, &b, &c],
///     "ij,jk,kl->il",
///     EinsumOptimize::Nested(NestedEinsum::parse("ij,(jk,kl)->il").unwrap()),
/// )
/// .unwrap();
/// assert_eq!(out.rank, 2);
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
/// ```rust
/// use tenferro::{GraphCompiler, TracedTensor};
/// use tenferro::traced_tensor::{einsum_with, EinsumOptimize};
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
/// let b = TracedTensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]);
/// let c = TracedTensor::from_vec_col_major(vec![4, 5], vec![1.0_f64; 20]);
/// let mut compiler = GraphCompiler::new();
///
/// // 3 operands: A(0), B(1), C(2)
/// // Step 1: contract positions 1,2 (B,C) -> T. List: [A, T]
/// // Step 2: contract positions 0,1 (A,T) -> result
/// let out = einsum_with(
///     &mut compiler,
///     &[&a, &b, &c],
///     "ij,jk,kl->il",
///     EinsumOptimize::Path(vec![(1, 2), (0, 1)]),
/// )
/// .unwrap();
/// assert_eq!(out.rank, 2);
///
/// // Step 1: contract positions 0,1 (A,B) -> T. List: [C, T]
/// // Step 2: contract positions 0,1 (C,T) -> result
/// let out = einsum_with(
///     &mut compiler,
///     &[&a, &b, &c],
///     "ij,jk,kl->il",
///     EinsumOptimize::Path(vec![(0, 1), (0, 1)]),
/// )
/// .unwrap();
/// assert_eq!(out.rank, 2);
/// ```
///
/// ## `Tree` -- Pre-computed ContractionTree
///
/// Pass a tree obtained from `ContractionTree::optimize` or other
/// optimization tools. Skips all path computation.
///
/// ```rust
/// use tenferro::{GraphCompiler, TracedTensor};
/// use tenferro::traced_tensor::{einsum_with, EinsumOptimize};
/// use tenferro_einsum::{ContractionTree, Subscripts};
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
/// let b = TracedTensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]);
/// let c = TracedTensor::from_vec_col_major(vec![4, 5], vec![1.0_f64; 20]);
/// let mut compiler = GraphCompiler::new();
///
/// let subs = Subscripts::parse("ij,jk,kl->il").unwrap();
/// let shapes = [&[2, 3][..], &[3, 4], &[4, 5]];
/// let tree = ContractionTree::optimize(&subs, &shapes).unwrap();
/// let out = einsum_with(
///     &mut compiler,
///     &[&a, &b, &c],
///     "ij,jk,kl->il",
///     EinsumOptimize::Tree(tree),
/// )
/// .unwrap();
/// assert_eq!(out.rank, 2);
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
/// Builds a lazy computation graph. Compile the returned tensor with
/// [`GraphCompiler`] and execute the program with a graph executor.
///
/// # Examples
///
/// ```
/// use tenferro::traced_tensor::einsum;
/// use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
/// let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);
/// let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]);
/// let y = TracedTensor::from_vec_col_major(vec![3], vec![2.0_f64; 3]);
/// let mut compiler = GraphCompiler::new();
///
/// // Matrix multiply
/// let c = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();
///
/// // Inner product
/// let s = einsum(&mut compiler, &[&x, &y], "i,i->").unwrap();
///
/// // Row sum (unary)
/// let r = einsum(&mut compiler, &[&a], "ij->i").unwrap();
///
/// // Hadamard product
/// let h = einsum(&mut compiler, &[&a, &a], "ij,ij->ij").unwrap();
///
/// // Outer product
/// let o = einsum(&mut compiler, &[&x, &y], "i,j->ij").unwrap();
///
/// let program = compiler.compile(&c).unwrap();
/// let result = GraphExecutor::new(CpuBackend::new()).run(&program).unwrap();
/// assert_eq!(result.shape(), &[2, 2]);
/// ```
pub fn einsum(
    compiler: &mut GraphCompiler,
    inputs: &[&TracedTensor],
    subscripts: &str,
) -> Result<TracedTensor> {
    einsum_with(compiler, inputs, subscripts, EinsumOptimize::default())
}

/// N-ary einsum using integer labels and the default contraction strategy.
///
/// # Examples
///
/// ```
/// use tenferro::traced_tensor::einsum_subscripts;
/// use tenferro::{CpuBackend, EinsumSubscripts, GraphCompiler, GraphExecutor, Tensor, TracedTensor};
///
/// let mut compiler = GraphCompiler::new();
/// let a = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]));
/// let b = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]));
/// let subscripts = EinsumSubscripts::new(&[&[0], &[0]], &[]);
/// let dot = einsum_subscripts(&mut compiler, &[&a, &b], &subscripts).unwrap();
/// let program = compiler.compile(&dot).unwrap();
/// let result = GraphExecutor::new(CpuBackend::new()).run(&program).unwrap();
///
/// assert_eq!(result.as_slice::<f64>().unwrap(), &[32.0]);
/// ```
pub fn einsum_subscripts(
    compiler: &mut GraphCompiler,
    inputs: &[&TracedTensor],
    subscripts: &EinsumSubscripts,
) -> Result<TracedTensor> {
    einsum_subscripts_with(compiler, inputs, subscripts, EinsumOptimize::default())
}

/// N-ary einsum with explicit contraction strategy.
///
/// See [`EinsumOptimize`] for all available strategies and examples.
/// Inputs with symbolic or otherwise non-concrete shapes currently support
/// only the default automatic optimization strategy.
///
/// # Examples
///
/// ```
/// use tenferro::traced_tensor::{einsum_with, EinsumOptimize};
/// use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
/// let b = TracedTensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]);
/// let c = TracedTensor::from_vec_col_major(vec![4, 2], vec![1.0_f64; 8]);
/// let mut compiler = GraphCompiler::new();
///
/// // Left-to-right, no optimizer
/// let out = einsum_with(&mut compiler, &[&a, &b, &c], "ij,jk,kl->il",
///     EinsumOptimize::False).unwrap();
///
/// // JAX-compatible explicit path
/// let out_path = einsum_with(&mut compiler, &[&a, &b, &c], "ij,jk,kl->il",
///     EinsumOptimize::Path(vec![(1, 2), (0, 1)])).unwrap();
///
/// let program = compiler.compile(&out).unwrap();
/// let result = GraphExecutor::new(CpuBackend::new()).run(&program).unwrap();
/// assert_eq!(result.shape(), &[2, 2]);
/// ```
pub fn einsum_with(
    compiler: &mut GraphCompiler,
    inputs: &[&TracedTensor],
    subscripts: &str,
    optimize: EinsumOptimize,
) -> Result<TracedTensor> {
    let parsed = compiler.cached_subscripts(subscripts)?;
    einsum_subscripts_with(compiler, inputs, &parsed.subscripts, optimize)
}

/// N-ary einsum with integer labels and explicit contraction strategy.
///
/// Inputs with symbolic or otherwise non-concrete shapes currently support
/// only the default automatic optimization strategy.
///
/// # Examples
///
/// ```
/// use tenferro::traced_tensor::{einsum_subscripts_with, EinsumOptimize};
/// use tenferro::{CpuBackend, EinsumSubscripts, GraphCompiler, GraphExecutor, Tensor, TracedTensor};
///
/// let mut compiler = GraphCompiler::new();
/// let a = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]));
/// let b = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]));
/// let subscripts = EinsumSubscripts::new(&[&[0], &[0]], &[]);
/// let dot = einsum_subscripts_with(
///     &mut compiler,
///     &[&a, &b],
///     &subscripts,
///     EinsumOptimize::False,
/// )
/// .unwrap();
/// let program = compiler.compile(&dot).unwrap();
/// let result = GraphExecutor::new(CpuBackend::new()).run(&program).unwrap();
///
/// assert_eq!(result.as_slice::<f64>().unwrap(), &[32.0]);
/// ```
pub fn einsum_subscripts_with(
    compiler: &mut GraphCompiler,
    inputs: &[&TracedTensor],
    subscripts: &EinsumSubscripts,
    optimize: EinsumOptimize,
) -> Result<TracedTensor> {
    if inputs.is_empty() {
        return Err(Error::ContractionError(
            "einsum requires at least one input tensor".into(),
        ));
    }

    let subs = to_einsum_subscripts(subscripts);
    if subs.inputs.len() != inputs.len() {
        return Err(Error::ContractionError(format!(
            "einsum subscripts expect {} inputs, got {}",
            subs.inputs.len(),
            inputs.len()
        )));
    }
    if inputs.iter().any(|tensor| !has_concrete_shape(tensor)) {
        validate_symbolic_einsum_optimize(&optimize)?;
        return Ok(build_symbolic_nary_einsum(inputs, subscripts, &subs));
    }
    let shapes: Vec<Vec<usize>> = inputs.iter().map(|t| concrete_shape(t)).collect();
    let shape_refs: Vec<&[usize]> = shapes.iter().map(|s| s.as_slice()).collect();

    match optimize {
        // Reuse TreeSA results for repeated calls with the same equation and input shapes.
        EinsumOptimize::Auto(opts) if is_default_auto_options(&opts) => {
            let cache_key = (subscripts.clone(), shapes.clone());
            let tree = compiler.cached_static_einsum_tree(cache_key, || {
                resolve_strategy(EinsumOptimize::Auto(opts), &subs, &shape_refs)
            })?;
            build_traced_from_tree(inputs, &subs, tree.as_ref(), &shapes)
        }
        optimize => {
            let tree = resolve_strategy(optimize, &subs, &shape_refs)?;
            build_traced_from_tree(inputs, &subs, &tree, &shapes)
        }
    }
}

fn validate_symbolic_einsum_optimize(optimize: &EinsumOptimize) -> Result<()> {
    match optimize {
        EinsumOptimize::Auto(opts) if is_default_auto_options(opts) => Ok(()),
        _ => Err(Error::ContractionError(
            "symbolic einsum supports only default automatic optimization".into(),
        )),
    }
}

fn is_default_auto_options(options: &ContractionOptimizerOptions) -> bool {
    let default = default_auto_options();
    options.ntrials == default.ntrials
        && options.niters == default.niters
        && f64_slices_equal_by_bits(&options.betas, &default.betas)
        && score_functions_equal_by_bits(&options.score, &default.score)
}

pub(crate) fn default_auto_options() -> ContractionOptimizerOptions {
    match EinsumOptimize::default() {
        EinsumOptimize::Auto(options) => options,
        _ => unreachable!("EinsumOptimize::default must be automatic optimization"),
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

fn has_concrete_shape(tensor: &TracedTensor) -> bool {
    tensor
        .sym_shape()
        .is_some_and(|shape| shape.iter().all(|dim| dim.constant_value().is_some()))
}

fn build_symbolic_nary_einsum(
    inputs: &[&TracedTensor],
    subscripts: &EinsumSubscripts,
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
            subscripts: subscripts.clone(),
        },
        input_vals,
        computegraph::types::OpMode::Primal,
    );
    builder.set_outputs(outputs.clone());
    let fragment = Arc::new(builder.build());
    let metadata_scope = register_scoped_fragment_metadata(fragment.as_ref(), std::iter::empty());
    let metadata_scopes = metadata_scopes_with_new(
        metadata_scope,
        inputs.iter().map(|input| input.metadata_scopes.as_slice()),
    );

    TracedTensor {
        id: next_traced_id(),
        rank: parsed.output.len(),
        dtype: crate::shape_infer::promote_dtypes(inputs.iter().map(|t| t.dtype)),
        fragment,
        val: outputs[0],
        data: None,
        shape_hint: None,
        inputs_map: Arc::new(merged),
        extra_roots,
        checkpoint_chain: None,
        metadata_scopes,
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
) -> Result<TracedTensor> {
    let out_shape = compute_einsum_output_shape(subscripts, shapes);

    let mut builder = FragmentBuilder::new();

    // Add parents and create ValRef for each input
    let mut input_vals = Vec::new();
    for input in inputs {
        builder.add_parent(input.fragment.clone());
        let val_ref = ValRef::External(input.fragment.vals()[input.val].key.clone());
        input_vals.push(val_ref);
    }

    let result_ref = build_einsum_fragment(&mut builder, tree, &input_vals, shapes)
        .map_err(|err| Error::ContractionError(err.to_string()))?;

    match result_ref {
        ValRef::Local(result_local) => {
            builder.set_outputs(vec![result_local]);
            let fragment = Arc::new(builder.build());
            let metadata_scope =
                register_scoped_fragment_metadata(fragment.as_ref(), std::iter::empty());

            let mut merged = HashMap::new();
            let mut extra_roots = Vec::new();
            for input in inputs {
                merged.extend(input.inputs_map.iter().map(|(k, v)| (k.clone(), v.clone())));
                extra_roots.extend(input.extra_roots.iter().cloned());
            }
            let metadata_scopes = metadata_scopes_with_new(
                metadata_scope,
                inputs.iter().map(|input| input.metadata_scopes.as_slice()),
            );

            let merged_chain = inputs.iter().fold(None, |acc, input| {
                CheckpointNode::merge_chains(acc, input.checkpoint_chain.clone())
            });

            Ok(TracedTensor {
                id: next_traced_id(),
                rank: out_shape.len(),
                dtype: crate::shape_infer::promote_dtypes(inputs.iter().map(|t| t.dtype)),
                fragment,
                val: result_local,
                data: None,
                shape_hint: Some(out_shape.into_iter().map(SymDim::from).collect()),
                inputs_map: Arc::new(merged),
                extra_roots,
                checkpoint_chain: merged_chain,
                metadata_scopes,
            })
        }
        ValRef::External(_) => {
            // Identity pass-through: the einsum doesn't add any ops.
            // Find which input was returned and clone its TracedTensor.
            for (i, iv) in input_vals.iter().enumerate() {
                if *iv == result_ref {
                    return Ok(TracedTensor {
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
                        metadata_scopes: inputs[i].metadata_scopes.clone(),
                    });
                }
            }
            Err(Error::Internal(
                "einsum builder returned an unrecognized external value".into(),
            ))
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
