//! High-level einsum with N-ary contraction tree optimization.
//!
//! This crate provides:
//!
//! - **String notation**: `"ij,jk->ik"` (NumPy/PyTorch compatible)
//! - **Parenthesized notation**: `"ij,(jk,kl)->il"` respects user-specified
//!   contraction order via [`NestedEinsum`]
//! - **Integer label notation**: using `u32` labels
//! - **Repeated labels**: `"ii->i"` extracts diagonals, `"ii->"` traces, and
//!   `"i->ii"` embeds a vector on a diagonal
//! - **N-ary contraction**: Automatic or manual optimization of pairwise
//!   contraction order via [`ContractionTree`]
//! - **v2 builder**: [`build_einsum_fragment`] lowers einsum into a compute
//!   graph fragment using `DotGeneral`, `ReduceSum`, `Transpose`, etc.
//!
//! # Examples
//!
//! ```
//! use tenferro_einsum::{ContractionTree, Subscripts};
//!
//! let subs = Subscripts::parse("ij,jk->ik").unwrap();
//! let tree = ContractionTree::optimize(&subs, &[&[2, 3], &[3, 4]]).unwrap();
//! assert_eq!(tree.step_count(), 1);
//! ```
//!
//! ```
//! use tenferro_einsum::Subscripts;
//!
//! let trace = Subscripts::parse("ii->").unwrap();
//! let diagonal = Subscripts::parse("ii->i").unwrap();
//! let embedded = Subscripts::parse("i->ii").unwrap();
//! let higher_rank = Subscripts::parse("iij->ij").unwrap();
//!
//! assert!(trace.output.is_empty());
//! assert_eq!(diagonal.output, vec![b'i' as u32]);
//! assert_eq!(embedded.output, vec![b'i' as u32, b'i' as u32]);
//! assert_eq!(higher_rank.inputs[0], vec![b'i' as u32, b'i' as u32, b'j' as u32]);
//! ```

pub mod builder;
mod eager;
pub mod planning;
pub mod syntax;
mod typed_eager;
pub(crate) mod util;

// Re-exports for convenience
pub use builder::build_einsum_fragment;
pub use eager::{
    clear_eager_einsum_cache, eager_einsum, eager_einsum_cache_capacity, eager_einsum_cache_stats,
    eager_einsum_owned, eager_einsum_owned_subscripts, eager_einsum_read_subscripts,
    eager_einsum_subscripts, set_eager_einsum_cache_capacity, DEFAULT_EAGER_EINSUM_CACHE_CAPACITY,
};
pub use planning::tree::{ContractionOptimizerOptions, ContractionTree};
pub use syntax::nested::NestedEinsum;
pub use syntax::subscripts::Subscripts;
pub use typed_eager::typed_eager_einsum;
pub use util::{build_size_dict, compute_output_shape};

#[cfg(test)]
mod tests;
