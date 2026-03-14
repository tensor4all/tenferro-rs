//! Eager AD entry points without `_ad` suffix.
//!
//! These functions are thin wrappers around the existing builder APIs
//! (`*_ad(...).run()`) and are intended for integration code paths that prefer
//! explicit eager execution.
//!
//! # Examples
//!
//! ```text
//! use tenferro_dyadtensor::{DynAdTensor, set_default_runtime, RuntimeContext};
//! use tenferro_prims::CpuContext;
//! use tenferro_tensor::{MemoryOrder, Tensor};
//!
//! let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
//! let a = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
//!     .unwrap();
//! let ad_a = DynAdTensor::new_primal(a);
//! let out = ad_a.qr().unwrap();
//! assert_eq!(out.q.dims(), &[2, 2]);
//! ```

use std::collections::HashMap;

use tenferro_algebra::{Scalar, Standard};
pub(crate) use tenferro_einsum as tf_einsum;
use tenferro_linalg::SolveGrad;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::tape;
use crate::{AdTensor, Error, NodeId, Result, StructuredTensor};

use super::{einsum_ad, sum_ad, EinsumRuntimeValue, ScalarRuntimeValue};

mod layout;
mod pullback;
mod scalar_eager;

pub use super::linalg::ad::eager::{
    cholesky, det, eig, eigen, inv, lstsq, lu, matrix_exp, norm, pinv, qr, slogdet, solve,
    solve_triangular, svd,
};
pub(crate) use layout::normalize_cotangent_payload;
pub use pullback::{
    einsum_frule, einsum_hvp, einsum_rrule, pullback, pullback_wrt, solve_triangular_rrule,
};
pub use scalar_eager::{
    acos, acosh, add, asin, asinh, atan, atan2, atanh, cos, cosh, exp, expm1, hypot, log, log1p,
    mean, pow, sin, sinh, sqrt, std, tanh, var,
};

/// Eager AD einsum.
///
/// Equivalent to `crate::einsum_ad(...).run()`.
///
/// # Examples
///
/// ```text
/// let out = tenferro_dyadtensor::ad::einsum("ij,jk->ik", &[&a, &b])?;
/// ```
pub fn einsum<'a, T>(subscripts: &'a str, operands: &'a [&'a AdTensor<T>]) -> Result<AdTensor<T>>
where
    T: EinsumRuntimeValue,
{
    einsum_ad(subscripts, operands).run()
}

/// Eager AD full reduction / sum.
///
/// Equivalent to `crate::sum_ad(...).run()`.
///
/// # Examples
///
/// ```text
/// let out = tenferro_dyadtensor::ad::sum(&x)?;
/// ```
pub fn sum<T>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: ScalarRuntimeValue,
{
    sum_ad(tensor).run()
}

#[cfg(test)]
mod tests;
