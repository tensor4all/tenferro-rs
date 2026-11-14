//! High-level einsum with N-ary contraction tree optimization.
//!
//! This crate provides:
//!
//! - **String notation**: `"ij,jk->ik"` (NumPy/PyTorch compatible)
//! - **Ellipsis notation**: `"...ij,...jk->...ik"` with right-aligned
//!   equal-or-one broadcasting, plus programmatic [`EinsumNotation`]
//! - **Parenthesized notation**: `"ij,(jk,kl)->il"` respects user-specified
//!   contraction order via [`NestedEinsum`]
//! - **Integer label notation**: using `u32` labels
//! - **Repeated labels**: `"ii->i"` extracts diagonals, `"ii->"` traces, and
//!   `"i->ii"` embeds a vector on a diagonal
//! - **N-ary contraction**: Automatic or manual optimization of pairwise
//!   contraction order via [`ContractionTree`]
//! - **Tensordot sugar**: NumPy-style axis-pair contraction, implemented as
//!   contraction sugar rather than as a linear algebra API.
//! - **One call grammar (#2552)** on three layers, declared once in the
//!   operation registry (`docs/spec/generated/api-ops.md`):
//!
//!   | Layer | Traits | Spellings |
//!   |---|---|---|
//!   | concrete | [`EinsumOps`] on a slice/array/`Vec` of `AsView` inputs, [`TensordotOps`] | `[&a, &b].einsum(spec, exec)`, `.einsum_with(spec, &opts, exec)`, `.einsum_into(spec, &opts, exec, dest)` |
//!   | traced | [`TracedEinsumOps`] on `TraceContext`, [`TracedTensordotOps`] | `trace.einsum(&[&t1, &t2], spec)`, `t.tensordot(&u, axes)` |
//!   | eager (`autodiff`) | `EagerEinsumOps` on `EagerSession` | `sess.einsum(&[&x, &y], spec)` |
//!
//!   The spec is text, [`EinsumNotation`] or [`EinsumSubscripts`]; options
//!   ([`EinsumOptions`]) select the contraction order and per-operand
//!   conjugation. [`ConcreteEinsumPlan`] plans once and executes repeatedly.
//! - **Extension runtime**: traced einsum lowers to a registered tenferro
//!   extension runtime, keeping core op definitions small.
//!
//! # Cargo features
//!
//! | Feature | Enables |
//! |---|---|
//! | `autodiff` | The eager layer (`EagerEinsumOps`) and AD rules. Adds the `tenferro-ad` dependency. |
//! | `cuda` | CUDA execution through `tenferro-gpu`. |
//!
//! For contractions without AD, use the concrete [`EinsumOps`] or
//! [`ConcreteEinsumPlan`] with an engine; they need no `autodiff`.
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
//! use tenferro_cpu::CpuEngine;
//! use tenferro_einsum::EinsumOps;
//! use tenferro_tensor::Tensor;
//!
//! let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])?;
//! let b = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12])?;
//! let mut backend = CpuEngine::serial();
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
#![cfg_attr(docsrs, feature(doc_cfg))]
mod api;
mod binary_dot;
mod broadcast;
mod builder;
mod cache;
mod concrete;
mod cpu_concrete;
mod eager;
#[cfg(feature = "autodiff")]
mod eager_ad;
mod ellipsis;
mod error;
mod extension;
pub mod lowering;
mod optimize;
mod planning;
pub mod prelude;
mod subscripts;
mod syntax;
mod tensordot;
mod traced;
#[cfg(test)]
mod typed_eager;
pub(crate) mod util;

#[cfg(feature = "autodiff")]
#[cfg_attr(docsrs, doc(cfg(feature = "autodiff")))]
pub use api::EagerEinsumOps;
pub use api::{
    ConjMask, EinsumInputs, EinsumOps, EinsumOptions, EinsumSpec, Spec, TensordotOps,
    TracedEinsumOps, TracedTensordotOps,
};
pub use cache::EINSUM_EXTENSION_FAMILY_ID;
pub use concrete::ConcreteEinsumPlan;
pub use error::{Error, PlanningError, Result};
pub use extension::extension_module;
#[cfg(feature = "autodiff")]
pub use extension::semantic_ad_rules;
pub use optimize::EinsumOptimize;
pub use planning::tree::{ContractionOptimizerOptions, ContractionTree};
pub use subscripts::{
    parse_einsum_notation, parse_einsum_subscripts, EinsumAxis, EinsumNotation, EinsumSubscripts,
};
pub use syntax::nested::NestedEinsum;
pub use syntax::subscripts::Subscripts;
pub use tensordot::TensorDotAxes;

#[cfg(test)]
mod concrete_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod typed_eager_tests;
