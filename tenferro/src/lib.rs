#![allow(clippy::multiple_bound_locations)]

//! `tenferro`: traced tensor computation with StableHLO-style IR.
//!
//! This crate provides a tracing-based tensor computation framework where
//! operations are recorded into a StableHLO-compatible intermediate
//! representation, then compiled and executed on a backend (e.g., CPU).
//!
//! # Examples
//!
//! ```rust,ignore
//! use tenferro::{CpuBackend, Engine, TracedTensor};
//!
//! let mut engine = Engine::new(CpuBackend::default());
//! // ... build and execute traced computations
//! ```

pub use tenferro_tensor::{DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig};

mod checkpoint;
pub mod compiler;
mod eager;
pub mod eager_einsum;
mod eager_emitter;
pub mod eager_exec;
pub(crate) mod eager_ops;
pub(crate) mod eager_ops_elementwise;
pub(crate) mod eager_ops_linalg;
pub mod einsum;
pub mod engine;
pub mod error;
pub mod exec;
pub mod extension;
mod linalg_api;
mod metadata;
pub mod segment;
pub mod shape_infer;
pub mod sym_dim;
pub mod traced;

pub use eager::{EagerContext, EagerTensor};
pub use engine::Engine;
pub use linalg_api::{
    cholesky, convert, det, eig, eigh, eigh_with_eps, eigvals, eigvalsh, inv, lu, norm, pinv,
    pinv_with_rtol, qr, slogdet, solve, svd, svd_with_eps, triangular_solve,
};
pub use sym_dim::SymDim;
pub use tenferro_tensor::cpu::CpuBackend;
pub use tenferro_tensor::{DType, Tensor, TensorBackend, TensorScalar, TypedTensor};
pub use traced::TracedTensor;

/// Matrix multiplication helper for rank-2 traced tensors.
///
/// This contracts the last dimension of `a` with the first dimension of `b`.
///
/// # Examples
///
/// ```rust,ignore
/// let c = tenferro::matmul(&a, &b);
/// ```
pub fn matmul(a: &TracedTensor, b: &TracedTensor) -> TracedTensor {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![a.rank - 1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    a.dot_general(b, config)
}

/// Elementwise power helper with NumPy-style broadcasting.
///
/// # Examples
///
/// ```rust,ignore
/// let y = tenferro::pow(&base, &exp);
/// ```
pub fn pow(base: &TracedTensor, exp: &TracedTensor) -> TracedTensor {
    base.pow(exp)
}
