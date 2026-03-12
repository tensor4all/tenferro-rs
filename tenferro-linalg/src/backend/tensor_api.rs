//! Tensor-level backend trait and result types for linalg decompositions.
//!
//! The backend-facing tensor linalg contract now lives in
//! [`tenferro_linalg_prims`]. This module re-exports that protocol from the
//! public `tenferro-linalg` backend namespace so high-level APIs can stay
//! focused on composition and validation.

#[doc(inline)]
pub use tenferro_linalg_prims::{
    EigTensorResult, EigenTensorResult, KernelLinalgScalar, LinalgCapabilityOp, LinalgScalar,
    LuTensorResult, QrTensorResult, SvdTensorResult, TensorLinalgPrims as TensorLinalgBackend,
};

#[cfg(test)]
mod tests;
