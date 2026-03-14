#![allow(clippy::multiple_bound_locations)]

//! `tenferro-dyadtensor`: AD-aware tensor interface layer on top of `tenferro-rs`.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_dyadtensor::{DynAdTensor, StructuredTensor};
//! use tenferro_tensor::{MemoryOrder, Tensor};
//!
//! let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
//! let diag = StructuredTensor::from_diagonal_vector(payload, 2).unwrap();
//! let x = DynAdTensor::new_primal(diag);
//! assert_eq!(x.dims(), &[2, 2]);
//! assert!(x.is_diag());
//! ```
//!
//! Builder `.run()` execution is configured through [`set_default_runtime`].
//! Reverse-mode graphs are created through [`DynTape`] and are exposed back to
//! users through dynamic helpers on [`DynAdTensor`].
//!
//! Module map:
//!
//! - [`core`] defines AD values plus dynamic wrappers.
//! - [`runtime`] owns default-runtime selection and runtime dispatch.
//! - `tape` owns reverse-mode pullback storage.
//! - [`ops`] is operation-first: `einsum`, `scalar`, `reduction`, and
//!   `linalg/*` each keep primal and AD wiring together.
//! - [`StructuredTensor`] and `structured` handle structured layouts.

pub mod core;
pub mod error;
pub mod ops;
pub mod policy;
pub mod runtime;
mod structured;
mod tape;
pub mod traits;

#[doc(hidden)]
pub use core::DynTensorTyped;
pub use core::{AdMode, DynAdTensor, DynScalar, DynTape, NodeId, ScalarType};
pub(crate) use core::{AdScalar, AdTensor, AdValue, DynTensor};
pub use error::{Error, Result};
pub use ops::ad;
pub use ops::chainrules_api;
pub use ops::{
    acos_ad, acosh_ad, add_ad, asin_ad, asinh_ad, atan2_ad, atan_ad, atanh_ad, cholesky,
    cholesky_ad, cos_ad, cosh_ad, det, det_ad, eig, eig_ad, eigen, eigen_ad, einsum, einsum_ad,
    exp_ad, expm1_ad, hypot_ad, inv, inv_ad, log1p_ad, log_ad, lstsq, lstsq_ad, lu, lu_ad,
    matrix_exp, matrix_exp_ad, mean_ad, norm, norm_ad, pinv, pinv_ad, pow_ad, qr, qr_ad, sin_ad,
    sinh_ad, slogdet, slogdet_ad, solve, solve_ad, solve_triangular, solve_triangular_ad, sqrt_ad,
    std_ad, sum_ad, svd, svd_ad, tanh_ad, var_ad, AdEigResult, AdEigenResult, AdLstsqResult,
    AdLuResult, AdQrResult, AdSlogdetResult, AdSvdResult,
};
pub use policy::DiffPolicy;
pub use runtime::{set_default_runtime, with_default_runtime, DefaultRuntimeGuard, RuntimeContext};
pub use structured::meta::{
    plan_axis_classes_for_subscripts, AxisClassMergePlan, AxisClassPlanError, OperandAxisClassPlan,
    OperandAxisClasses,
};
pub use structured::StructuredTensor;
pub use traits::{
    AdResult, AllowedPairs, Differentiable, FactorizeOptions, FactorizeResult, IndexLike, OpRule,
    TensorKernel,
};
