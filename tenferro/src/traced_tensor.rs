//! Traced tensor operations.
//!
//! This module is the public namespace for operations that build traced tensor
//! graphs. The operation names stay independent of execution mode; the module
//! name identifies the tensor family.

use crate::DotGeneralConfig;

pub use crate::einsum::{einsum, einsum_with, EinsumOptimize};
pub use crate::linalg_api::{
    cholesky, convert, det, eig, eigh, eigh_with_eps, eigvals, eigvalsh, full_piv_lu,
    full_piv_lu_solve, inv, lu, norm, pinv, pinv_with_rtol, qr, slogdet, solve, svd, svd_with_eps,
    triangular_solve,
};
pub use crate::traced::{eval_all, CompiledTracedTensor, TracedTensor, TracedTensorId};

/// Matrix multiplication helper for rank-2 traced tensors.
///
/// This contracts the last dimension of `a` with the first dimension of `b`.
///
/// # Examples
///
/// ```rust,ignore
/// let c = tenferro::traced_tensor::matmul(&a, &b);
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
/// let y = tenferro::traced_tensor::pow(&base, &exp);
/// ```
pub fn pow(base: &TracedTensor, exp: &TracedTensor) -> TracedTensor {
    base.pow(exp)
}
