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
//! - **Tensordot sugar**: NumPy-style axis-pair contraction extension methods,
//!   implemented as contraction sugar rather than as linear algebra APIs.
//! - **Concrete execution**: backend-explicit [`TensorEinsumExt`],
//!   [`TypedTensorEinsumExt`], [`TensorReadEinsumExt`], and
//!   [`ConcreteEinsumPlan`] APIs for non-AD tensor values.
//! - **Extension runtime**: traced einsum lowers to a registered tenferro
//!   extension runtime, keeping core op definitions small.
//! - **Tensor extension traits**: graph-building and immediate-execution
//!   helpers are available as methods on `GraphCompiler`, concrete input
//!   slices/arrays, eager input slices/arrays, and tensor receivers.
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
//! use tenferro_cpu::CpuBackend;
//! use tenferro_einsum::TensorEinsumExt;
//! use tenferro_tensor::Tensor;
//!
//! let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
//! let b = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
//! let mut backend = CpuBackend::new();
//!
//! let out = [&a, &b].einsum("ij,jk->ik", &mut backend)?;
//! assert_eq!(out.shape(), &[2, 4]);
//! # Ok::<(), tenferro_tensor::Error>(())
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

mod binary_dot;
mod builder;
mod cache;
mod concrete;
mod eager;
#[cfg(feature = "autodiff")]
mod eager_ad;
mod error;
mod extension;
pub mod lowering;
mod optimize;
mod planning;
mod subscripts;
mod syntax;
mod tensordot;
mod traced;
#[cfg(test)]
mod typed_eager;
pub(crate) mod util;

pub use cache::EINSUM_EXTENSION_FAMILY_ID;
pub use concrete::{
    ConcreteEinsumPlan, TensorEinsumExt, TensorReadEinsumExt, TypedTensorEinsumExt,
};
#[cfg(feature = "autodiff")]
pub use eager_ad::{EagerEinsumExt, EagerTensorEinsumExt};
pub use error::{Error, Result};
#[cfg(feature = "autodiff")]
pub use extension::ad_rules;
pub use extension::register_runtime;
pub use optimize::EinsumOptimize;
pub use planning::tree::{ContractionOptimizerOptions, ContractionTree};
pub use subscripts::{parse_einsum_subscripts, EinsumSubscripts};
pub use syntax::nested::NestedEinsum;
pub use syntax::subscripts::Subscripts;
pub use tensordot::TensorDotAxes;
pub use traced::{GraphCompilerEinsumExt, TracedTensorEinsumExt};

#[cfg(test)]
mod concrete_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod typed_eager_tests;
