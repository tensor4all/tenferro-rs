//! `tenferro-dyadtensor`: AD-aware tensor interface layer on top of `tenferro-rs`.

pub mod ad_value;
pub mod api;
pub mod context;
pub mod dyn_types;
pub mod error;
pub mod policy;
pub mod runtime;
pub mod traits;

pub use ad_value::{AdMode, AdScalar, AdTensor, AdValue, NodeId, TapeId};
pub use api::ad;
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
pub use dyn_types::{DynAdTensor, DynAdValue, DynScalar, DynTensor, ScalarType};
pub use error::{Error, Result};
pub use policy::DiffPolicy;
pub use runtime::{set_default_runtime, with_default_runtime, RuntimeContext};
pub use traits::{
    AdResult, AllowedPairs, Differentiable, FactorizeOptions, FactorizeResult, IndexLike, OpRule,
    TensorKernel,
};
