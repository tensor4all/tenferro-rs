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

use tenferro_tensor::DotGeneralConfig;

pub mod buffer_pool;
pub mod compiler;
pub mod einsum;
pub mod engine;
pub mod error;
pub mod exec;
mod linalg_api;
pub mod stablehlo;
pub mod traced;

pub use engine::Engine;
pub use linalg_api::{
    cholesky, det, eig, eigh, eigh_with_eps, eigvals, eigvalsh, inv, lu, norm, pinv, qr, slogdet,
    solve, svd, svd_with_eps, triangular_solve,
};
pub use tenferro_tensor::cpu::CpuBackend;
pub use tenferro_tensor::{DType, Tensor, TensorBackend, TypedTensor};
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
        lhs_contracting_dims: vec![a.shape.len() - 1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
        lhs_rank: a.shape.len(),
        rhs_rank: b.shape.len(),
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
