#![allow(clippy::multiple_bound_locations)]

//! `tenferro-dyadtensor`: AD-aware tensor interface layer on top of `tenferro-rs`.

pub mod ad_value;
pub mod api;
pub mod context;
pub mod dyn_types;
pub mod error;
pub mod partial_diag;
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
    qr, qr_ad, slogdet, slogdet_ad, solve, solve_ad, solve_triangular, solve_triangular_ad, svd,
    svd_ad, AdEigResult, AdEigenResult, AdLstsqResult, AdLuResult, AdQrResult, AdSlogdetResult,
    AdSvdResult,
};
pub use context::{
    set_global_context, try_with_global_context, with_global_context, GlobalContextGuard,
};
pub use dyn_types::{DynAdScalar, DynAdTensor, DynScalar, DynTensor, ScalarType};
pub use error::{Error, Result};
pub use partial_diag::{
    plan_axis_classes_for_subscripts, AxisClassMergePlan, AxisClassPlanError, OperandAxisClassPlan,
    OperandAxisClasses,
};
pub use policy::DiffPolicy;
pub use runtime::{set_default_runtime, with_default_runtime, RuntimeContext};
pub use structured::StructuredTensor;
pub use traits::{
    AdResult, AllowedPairs, Differentiable, FactorizeOptions, FactorizeResult, IndexLike, OpRule,
    TensorKernel,
};
