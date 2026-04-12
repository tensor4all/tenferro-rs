//! High-level einsum with N-ary contraction tree optimization.
//!
//! This crate provides:
//!
//! - **String notation**: `"ij,jk->ik"` (NumPy/PyTorch compatible)
//! - **Parenthesized notation**: `"ij,(jk,kl)->il"` respects user-specified
//!   contraction order via [`NestedEinsum`]
//! - **Integer label notation**: using `u32` labels
//! - **N-ary contraction**: Automatic or manual optimization of pairwise
//!   contraction order via [`ContractionTree`]
//! - **v2 builder**: [`build_einsum_fragment`] lowers einsum into a compute
//!   graph fragment using `DotGeneral`, `ReduceSum`, `Transpose`, etc.
//!
//! # Examples
//!
//! ```ignore
//! use computegraph::fragment::FragmentBuilder;
//! use computegraph::types::ValRef;
//! use tenferro_ops::input_key::TensorInputKey;
//! use tenferro_ops::std_tensor_op::StdTensorOp;
//! use tenferro_einsum::{ContractionTree, Subscripts, build_einsum_fragment};
//!
//! let subs = Subscripts::parse("ij,jk->ik").unwrap();
//! let tree = ContractionTree::optimize(&subs, &[&[2, 3], &[3, 4]]).unwrap();
//!
//! let mut builder = FragmentBuilder::<StdTensorOp>::new();
//! let a = builder.add_input(TensorInputKey::User { id: 0 });
//! let b = builder.add_input(TensorInputKey::User { id: 1 });
//!
//! let result = build_einsum_fragment(
//!     &mut builder,
//!     &tree,
//!     &[ValRef::Local(a), ValRef::Local(b)],
//!     &[vec![2, 3], vec![3, 4]],
//! );
//! ```

pub mod builder;
mod eager;
pub mod planning;
pub mod syntax;
pub(crate) mod util;

// Re-exports for convenience
pub use builder::build_einsum_fragment;
pub use eager::eager_einsum;
pub use planning::tree::{ContractionOptimizerOptions, ContractionTree};
pub use syntax::nested::NestedEinsum;
pub use syntax::subscripts::Subscripts;
pub use util::{build_size_dict, compute_output_shape};

#[cfg(test)]
mod tests;
