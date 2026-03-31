//! Internal implementation crate. Not a stable public API.
//!
//! # Examples
//!
//! ```text
//! // This crate is wired through tenferro surface crates and is not intended
//! // to be consumed directly.
//! ```

mod linearized;

pub use tenferro_internal_error::{Error, Result};
#[doc(hidden)]
pub use tenferro_internal_frontend_core::DynTensorTyped;

pub mod results {
    pub use tenferro_linalg::{
        CholeskyExResult, EigResult, EigenResult, InvExResult, LstsqResult, LuFactorExResult,
        LuFactorResult, LuPivot, LuResult, QrResult, SlogdetResult, SolveExResult, SvdResult,
    };
}

pub use linearized::{
    cholesky_dyn_value, det_dyn_value, eig_dyn_value, eigen_dyn_value, inv_dyn_value,
    lstsq_dyn_values, lu_dyn_value, matrix_exp_dyn_value, norm_dyn_value, pinv_dyn_value,
    qr_dyn_value, slogdet_dyn_value, solve_dyn_values, solve_triangular_dyn_value, svd_dyn_value,
    CholeskyOp, DetOp, DynEigValues, DynEigenValues, DynLstsqValues, DynLuValues, DynQrValues,
    DynSlogdetValues, DynSvdValues, EigOp, EigenOp, InvOp, LstsqOp, LuOp, MatrixExpOp, NormOp,
    PInvOp, QrOp, SlogdetOp, SolveOp, SolveTriangularOp, SvdOp,
};
pub use results::*;
