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
//! - **Extension runtime**: traced einsum lowers to a registered tenferro
//!   extension runtime, keeping core op definitions small.
//! - **Traced tensor namespace**: graph-building helpers are available under
//!   `tenferro_einsum::traced_tensor`.
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

mod builder;
mod cache;
mod eager;
#[cfg(feature = "autodiff")]
pub mod eager_tensor;
mod extension;
mod optimize;
mod planning;
mod subscripts;
mod syntax;
mod traced;
pub mod traced_tensor;
#[cfg(test)]
mod typed_eager;
pub(crate) mod util;

pub use cache::EINSUM_EXTENSION_FAMILY_ID;
pub use extension::register_runtime;
pub use optimize::EinsumOptimize;
pub use planning::tree::{ContractionOptimizerOptions, ContractionTree};
pub use subscripts::{parse_einsum_subscripts, EinsumSubscripts};
pub use syntax::nested::NestedEinsum;
pub use syntax::subscripts::Subscripts;
pub use traced::{einsum, einsum_subscripts, einsum_subscripts_with, einsum_with};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod typed_eager_tests;
