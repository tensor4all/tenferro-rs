#![allow(clippy::multiple_bound_locations)]

//! `tenferro-dyadtensor`: AD-aware tensor interface layer on top of `tenferro-rs`.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_dyadtensor::{AdTensor, StructuredTensor};
//! use tenferro_tensor::{MemoryOrder, Tensor};
//!
//! let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
//! let diag = StructuredTensor::from_diagonal_vector(payload, 2).unwrap();
//! let x = AdTensor::new_primal(diag);
//! assert_eq!(x.dims(), &[2, 2]);
//! assert!(x.is_diag());
//! ```

pub mod ad_value;
pub mod api;
pub mod context;
pub mod dyn_types;
pub mod error;
pub mod policy;
mod reverse_tape;
pub mod runtime;
mod structured;
pub mod traits;

pub use ad_value::{AdMode, AdScalar, AdTensor, AdValue, NodeId, TapeId};
pub use api::ad;
pub use api::chainrules_api;
pub use api::{
    cholesky, cholesky_ad, det, det_ad, eig, eig_ad, eigen, eigen_ad, einsum, einsum_ad, inv,
    inv_ad, lstsq, lstsq_ad, lu, lu_ad, matrix_exp, matrix_exp_ad, norm, norm_ad, pinv, pinv_ad,
    qr, qr_ad, slogdet, slogdet_ad, solve, solve_ad, solve_triangular, solve_triangular_ad, sum_ad,
    svd, svd_ad, AdEigResult, AdEigenResult, AdLstsqResult, AdLuResult, AdQrResult,
    AdSlogdetResult, AdSvdResult,
};
pub use context::{
    set_global_context, try_with_global_context, with_global_context, GlobalContextGuard,
};
pub use dyn_types::{DynAdScalar, DynAdTensor, DynScalar, DynTensor, ScalarType};
pub use error::{Error, Result};
pub use policy::DiffPolicy;
pub use runtime::{set_default_runtime, with_default_runtime, RuntimeContext};
pub use structured::meta::{
    plan_axis_classes_for_subscripts, AxisClassMergePlan, AxisClassPlanError, OperandAxisClassPlan,
    OperandAxisClasses,
};
pub use structured::StructuredTensor;
pub use traits::{
    AdResult, AllowedPairs, Differentiable, FactorizeOptions, FactorizeResult, IndexLike, OpRule,
    TensorKernel,
};
