//! Eager AD entry points without `_ad` suffix.
//!
//! These functions are thin wrappers around the existing builder APIs
//! (`*_ad(...).run()`) and are intended for integration code paths that prefer
//! explicit eager execution.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_dyadtensor::{ad, set_default_runtime, AdTensor, RuntimeContext};
//! use tenferro_prims::CpuContext;
//! use tenferro_tensor::{MemoryOrder, Tensor};
//!
//! let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
//! let a = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
//!     .unwrap();
//! let ad_a = AdTensor::new_primal(a);
//! let out = ad::qr(&ad_a).unwrap();
//! assert_eq!(out.q.dims(), &[2, 2]);
//! ```

use std::collections::HashMap;

use chainrules_scalarops::ScalarAd;
use tenferro_algebra::{Scalar, Standard};
pub(crate) use tenferro_einsum as tf_einsum;
use tenferro_linalg::{NormKind, SolveGrad};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{reverse_tape, AdScalar, AdTensor, AdValue, Error, NodeId, Result, StructuredTensor};

use super::{
    cholesky_ad, det_ad, eig_ad, eigen_ad, einsum_ad, inv_ad, lstsq_ad, lu_ad, matrix_exp_ad,
    norm_ad, pinv_ad, qr_ad, slogdet_ad, solve_ad, solve_triangular_ad, sum_ad, svd_ad,
    with_linalg_runtime,
};
pub(crate) use super::{
    dispatch_einsum_runtime, ComplexLinalgRuntimeValue, EinsumRuntimeValue, LinalgRuntimeValue,
    RealLinalgRuntimeValue, ScalarRuntimeValue,
};
use super::{
    AdEigResult, AdEigenResult, AdLstsqResult, AdLuResult, AdQrResult, AdSlogdetResult, AdSvdResult,
};

mod eager_linalg;
mod layout;
mod pullback;
mod scalar_eager;

pub use eager_linalg::{
    cholesky, det, eig, eigen, einsum, inv, lstsq, lu, matrix_exp, norm, pinv, qr, slogdet, solve,
    solve_triangular, sum, svd,
};
pub(crate) use layout::normalize_cotangent_payload;
pub use pullback::{
    einsum_frule, einsum_hvp, einsum_rrule, pullback, pullback_wrt, pullback_wrt_mixed,
    pullback_wrt_scalars, solve_triangular_rrule,
};
pub use scalar_eager::{
    acos, acosh, add, asin, asinh, atan, atan2, atanh, cos, cosh, exp, expm1, hypot, log, log1p,
    mean, pow, sin, sinh, sqrt, std, tanh, var,
};

#[cfg(test)]
mod tests;
