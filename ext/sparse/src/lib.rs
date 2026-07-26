#![deny(missing_docs)]
#![forbid(unsafe_code)]

//! Sparse tensor extension tutorial crate for tenferro.

mod extension;
mod sparse;

#[cfg(feature = "autodiff")]
pub use extension::sparse_semantic_ad_rules;
pub use extension::{extension_modules, sparse_matmul};
pub use sparse::{sparse_matmul_eager, SparseCooTensor, SparseCooTracedTensor};
