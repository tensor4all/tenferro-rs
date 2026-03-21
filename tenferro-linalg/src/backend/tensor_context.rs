//! Context-to-backend bridge trait for tensor linalg operations.
//!
//! The ownership boundary now lives in `tenferro-linalg-prims`; this module
//! keeps the trait visible from `tenferro-linalg::backend` for downstream
//! ergonomics.

#[doc(inline)]
pub use tenferro_linalg_prims::backend::TensorLinalgContextFor;
