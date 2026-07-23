#![deny(missing_docs)]
#![forbid(unsafe_code)]

//! Sparse tensor extension tutorial crate for tenferro.

mod extension;
mod sparse;

pub use extension::{register_runtime, sparse_matmul};
#[cfg(feature = "autodiff")]
pub use extension::{sparse_ad_rules, sparse_semantic_ad_rules};
pub use sparse::{sparse_matmul_eager, SparseCooTensor, SparseCooTracedTensor};
